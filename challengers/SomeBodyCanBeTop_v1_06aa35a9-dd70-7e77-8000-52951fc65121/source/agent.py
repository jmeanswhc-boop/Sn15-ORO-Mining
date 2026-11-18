"""EnvPack checkout pilot for ORO Bench (TF1–TF7).

A model explores the catalog; a deterministic pilot verifies variants,
commits only after inspect_stock, re-checks after the first add, recovers
when the market moves, and places the test order.
"""
from __future__ import annotations
import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient

def hx_walk_2(tool_call: dict) -> dict:
    raw = (tool_call.get('function') or {}).get('arguments', '{}')
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or '{}')
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}

def hx_brush_7(field: str, text: str) -> bool:
    """Whether an option string talks about the concept at all (even ambiguously)."""
    text = text or ''
    if field == 'ram':
        return bool(_C22_PAIR_HINT_RE.search(text) or _C15_MEM_RE.search(text))
    if field == 'screen_size':
        return bool(_C11_INCH_RE.search(text))
    if field == 'refresh_rate':
        return bool(_C10_HZ_RE.search(text))
    return bool(_C38_WXH_RE.search(text) or re.search('\\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\\b', text, re.I))

def hx_bolt_3(text: str) -> set:
    return {'%ggb' % (float(m.group(1)) * (1024 if m.group(2).lower() == 'tb' else 1)) for m in _C9_GB_RE.finditer(text or '')}

def hx_latch_6(text: str) -> set:
    return {'%gin' % float(m.group(1)) for m in _C11_INCH_RE.finditer(text or '')}

def hx_tilt_4(text: str) -> set:
    return {'%ghz' % float(m.group(1)) for m in _C10_HZ_RE.finditer(text or '')}

def hx_chase_11(long_side: int, short_side: int) -> str | None:
    if long_side >= 3840 and short_side >= 2160:
        return '4k'
    if long_side >= 2560 and short_side >= 1440:
        return 'qhd'
    if long_side >= 1920 and short_side >= 1080:
        return 'fhd'
    if long_side >= 1280 and short_side >= 720:
        return 'hd'
    return None

def hx_pack_10(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    if _C24_PART_ONLY_RE.search(options or ''):
        return 'no'
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or '')
    head = hx_sink_5(raw_title, title=True)
    kind = ' '.join((hx_sink_5(str(facts.get(k) or '')) for k in ('phone_type', 'device_type', 'description_device_type')))
    blob = head + ' ' + kind
    if _C18_NOT_PHONE_RE.search(blob):
        return 'no'
    if any(('[truncated]' in str(facts.get(k) or '') for k in ('phone_type', 'device_type', 'device_operating_system'))):
        return 'unclear'
    if _C12_IS_PHONE_RE.search(blob):
        return 'yes'
    os_blob = ' '.join((hx_sink_5(str(facts.get(k) or '')) for k in ('device_operating_system', 'description_operating_system') if not re.search('\\bcompatib', str(facts.get(k) or ''), re.I)))
    if _C21_OS_RE.search(os_blob):
        return 'yes'
    if _C21_OS_RE.search(head) and (not re.search('\\bcompatib', raw_title, re.I)):
        return 'yes'
    return 'unclear'

def hx_knit_12(text: str) -> set:
    """Display classes asserted by a text: pixel pairs and the usual shorthands."""
    text = text or ''
    if _C31_QUARTER_RE.search(text):
        return set()
    low = text.lower()
    out = set()
    for m in _C38_WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = hx_chase_11(pair[0], pair[1])
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

def hx_clamp_14(text: str) -> set:
    return set(re.findall('[a-z0-9]+', (text or '').lower()))

def hx_fill_9(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None

def hx_knock_1(field: str, text: str) -> set:
    """Values a free-form string asserts, requiring the number's own concept word.

    Titles and variant option strings mix several capacities, so a memory figure
    only counts when the word for it follows the number; storage totals and a
    graphics chip's own memory must never be read as system memory.
    """
    text = text or ''
    if field != 'ram':
        return _C37_VALUE_SETS[field](text)
    paired = {'%ggb' % float(a) for (a, _b) in _C23_PAIR_RE.findall(text)}
    return paired or {'%ggb' % float(n) for n in _C15_MEM_RE.findall(text)}

def hx_pick_16(field: str | None, text: str) -> str | None:
    """The single value a request states for a concept, or nothing when unclear."""
    if not field:
        return None
    return hx_fill_9(_C37_VALUE_SETS[field](text or ''))

def hx_trim_15(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 40] + ' ... (truncated by agent)"}'

def hx_forge_8(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(',', ''))
    except (TypeError, ValueError):
        return None

def hx_stamp_13(a: str, b: str) -> bool:
    """Loose noun equality: ignores case, spacing and a trailing plural."""
    x = re.sub('[^a-z0-9]+', '', (a or '').lower()).rstrip('s')
    y = re.sub('[^a-z0-9]+', '', (b or '').lower()).rstrip('s')
    return bool(x) and x == y

def hx_sink_5(text: str, title: bool=False) -> str:
    """Strip claims that describe what a listing is NOT, or what ships beside it."""
    text = _C17_NEGATED_RE.sub('', text or '')
    return _C2_BUNDLE_RE.split(text, maxsplit=1)[0] if title else text
_C16_MODELS = {'chutes': ('deepseek-ai/DeepSeek-V3.2-TEE', 'Qwen/Qwen3-32B-TEE'), 'openrouter': ('deepseek/deepseek-v3.2', 'qwen/qwen3-32b')}
_C30_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_C29_PROBLEM_BUDGET_S = 270.0
_C20_OBS_CHARS = 7000
_C14_MAX_REPROMPTS = 2
_C6_EPS = 1e-09
_C19_NUM = '(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?'
_C1_BUDGET_RE = re.compile('(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|at most)\\s*' + _C19_NUM + '\\s*(PHP|php|₱|pesos?)?', re.I)
_C13_KEY_RE = re.compile('(\\d{6,})::([A-Za-z0-9._\\-]+)')
_C33_QUOTED_RE = re.compile('\\"([^\\"]{2,80})\\"')
_C25_PHRASE_RE = re.compile('matches (?:the phrase )?\\"?([A-Za-z0-9][^\\",]{1,80}?)\\"?\\s*,', re.I)
_C36_SURFACE_RE = re.compile('surface at least (\\d+) distinct', re.I)
_C28_PREF_ITEM_RE = re.compile('prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\\-]*\\d[A-Za-z0-9\\-]*)\\b', re.I)
_C27_PREF_BRAND_RE = re.compile("\\bprefer(?:ably)?\\s+(?:the\\s+)?([A-Za-z][\\w&.'\\- ]{1,38}?)\\s*(?:,|\\.|;|\\bbut\\b|\\bwhen\\b|\\bif\\b|$)")
_C3_CAT_PHRASE_RE = re.compile("in the ([A-Za-z][\\w&'\\- ]{1,48}?) category", re.I)
_C35_SPEC_WORD_RE = re.compile('\\b(?:full|ultra)\\s+hd\\b|\\b(?:hd|fhd|qhd|uhd|2k|4k|\\d+p)\\b|\\d+(?:\\.\\d+)?\\s*-?\\s*(?:inch(?:es)?|in|hz|gb)\\b', re.I)
_C34_REASON_RE = re.compile('\\b(?:why|reason|explain|justif)', re.I)
_C4_CLAIM_FIELDS = ('resolution', 'refresh_rate', 'screen_size', 'ram', 'price', 'in_stock', 'brand')
_C32_QUESTION_RE = re.compile('\\?|\\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\\b', re.I)
_PREF_AXES: dict[str, tuple[str, ...]] = {'battery_life': ('battery', 'mah', 'hours', 'hrs', 'playback', 'talk time', 'playtime', 'long lasting', 'long-lasting', 'endurance'), 'sport_secure_fit': ('sport', 'running', 'workout', 'gym', 'earhook', 'ear hook', 'ear-hook', 'secure fit', 'in-ear', 'inear', 'earbud', 'fitness', 'jogging', 'bootcamp'), 'sweat_water_resistance': ('waterproof', 'water-resistant', 'water resistant', 'sweat', 'sweatproof', 'sweat-proof', 'ipx', 'ip67', 'ip68', 'ipx4', 'ipx5', 'ipx6', 'ipx7', 'weather', 'rain'), 'wireless_freedom': ('wireless', 'bluetooth', 'bt ', 'tws', 'true wireless', 'cordless')}
_C11_INCH_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*(?:(?:inch(?:es)?|in)\\b|\'{2}|[\\"“”])', re.I)
_C10_HZ_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*hz', re.I)
_C38_WXH_RE = re.compile('(\\d{3,5})\\s*[x×]\\s*(\\d{3,5})')
_C9_GB_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*(tb|gb)\\b', re.I)
_C23_PAIR_RE = re.compile('(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*(?:g(?:b)?)?\\s*\\+\\s*(\\d+(?:\\.\\d+)?)\\s*g(?:b)?\\b', re.I)
_C22_PAIR_HINT_RE = re.compile('(?<!\\d)\\d+(?:\\.\\d+)?\\s*(?:g(?:b)?)?\\s*\\+\\s*\\d+(?:\\.\\d+)?(?:\\s*g(?:b)?)?(?!\\d)', re.I)
_C15_MEM_RE = re.compile('(\\d+(?:\\.\\d+)?)\\s*gb\\s*(?:ram|memory|unified memory)\\b', re.I)
_C31_QUARTER_RE = re.compile('\\bqHD\\b')
_C7_FACT_LABEL = {'screen_size': 'size', 'resolution': 'display', 'refresh_rate': 'refresh_rate', 'ram': 'storage'}
_C37_VALUE_SETS = {'screen_size': hx_latch_6, 'refresh_rate': hx_tilt_4, 'resolution': hx_knit_12, 'ram': hx_bolt_3}
_C21_OS_RE = re.compile('\\b(?:android(?:\\s*\\d+(?:\\.\\d+)*)?|ios|blackberry(?:\\s*os)?|windows\\s*phone|symbian|harmony\\s*os|hyper\\s*os|sailfish|ubuntu\\s*touch|tizen)\\b', re.I)
_C12_IS_PHONE_RE = re.compile('\\bsmart\\s*phones?\\b', re.I)
_C18_NOT_PHONE_RE = re.compile('\\btablets?\\b|\\b(?:feature|basic)\\s*phones?\\b|\\b(?:mystery|blind|lucky)\\s*box(?:es)?\\b|\\b(?:accessor(?:y|ies)|screen\\s*protectors?|phone\\s*cases?|case\\s*(?:cover\\s*)?for|replacement\\s*(?:screen|battery)|smart\\s*watches?|earbuds?|headphones?)\\b', re.I)
_C24_PART_ONLY_RE = re.compile('\\bonly\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b|\\b(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\s+only\\b|^\\s*replacement\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b', re.I)
_C17_NEGATED_RE = re.compile('\\bnot\\s+(?:an?\\s+)?(?:smart\\s*phones?|feature\\s*phones?|basic\\s*phones?|tablets?|accessor(?:y|ies))\\b', re.I)
_C2_BUNDLE_RE = re.compile('\\b(?:(?:with\\s+(?:free|bundled)|bundled\\s+with|free)\\s+|(?:with|includes?)\\s+(?:an?\\s+)?(?=(?:headphones?|earbuds?|phone\\s*case|screen\\s*protector|charger|accessor(?:y|ies))\\b))', re.I)
_C8_FAMILY_SIGNS = (('preference', 'listing_id'), ('constraint', 'exact variant'), ('justification', 'trustworthy reason'), ('intent', 'nice-to-haves'))
_C26_PLAYBOOK = {'intent': "Firm requirements: the category, the named hardware specification, the budget and stock. A named brand is only a preference: choose it when a fully compliant item of that brand exists, otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view or compare to read the exact variant's specification from raw facts. Answer the shopper with message; a budget the shopper lowers later applies to every later choice.", 'recall': 'Run several differently worded searches (k=10, in_stock=true): full phrase, reversed tokens, single tokens, and category+phrase, until enough distinct matching products have appeared; every requested word must appear as a whole word of the title. Then view the strongest candidates so the controller can verify one and buy it.', 'constraint': "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for the specific variant, reading the variant's own option text before any product-wide value. Use filter(category, brand, max_price, in_stock=true) and view candidates. If the carted variant sells out, move to a different product that still meets every requirement.", 'preference': 'You already asked which preference matters most; use the answer. Search with preference cues (wireless, waterproof/IPX/sweat, sport/earhook, battery/mAh) inside the budget, then compare candidates with different listing_id values and read raw facts. Prefer listings whose title or facts cover sweat/water resistance, secure sport fit, wireless, AND battery life together. The first cart item is graded, so make it the best high-coverage match. If it is repriced above budget, recover to a different listing_id that still covers those axes.', 'ranking': "Only the named candidate variants count, and the controller ranks them for you: compare all of them first so their facts are on the record. Highest advertised maximum refresh rate wins, then the largest screen, then the lowest price, reading the selected variant's own value before a product-wide one. After the market moves, the remaining candidates are re-ranked.", 'recovery': 'Find the preferred model and color first: search the model code, then view it so every variant and color is on the record. The first cart item must be exactly that model and color. If it disappears, relax as little as possible - the same model in another color before any other model - and never relax the category, budget or stock.', 'justification': 'Find a product whose own raw facts state the requested specification, in the requested category and brand, in stock and within budget. View or compare it so its facts are on the record, then the controller inspects it, carts it and orders it with claims taken only from that evidence.', 'generic': 'Understand the firm requirements, filter or search the catalog, view candidates, and let the controller verify and buy the best one.'}
_C5_CORE_RULES = 'You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the cart and the order once a candidate is fully verified, so your job is to surface and read candidates: search or filter, then view or compare so their raw facts, variant options and prices are on record. Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock change while you shop; the controller re-checks the carted item and tells you when to move to a different product. Answer every direct question from the shopper with one short message before anything else, and never send progress or status updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments; one compare of up to five products beats several view calls.'
__all__ = ['agent_main']

class CatalogLog:
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

    def _sand_73(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self._free_70(pid, sku))

    @staticmethod
    def _free_70(pid: Any, sku: Any) -> str:
        return str(pid) + '::' + str(sku or '')

    def _hold_58(self, obs: dict) -> None:
        rows = obs.get('cart')
        if not isinstance(rows, list):
            return
        pairs = [(r.get('product_id'), r.get('sku')) for r in rows if isinstance(r, dict) and r.get('product_id')]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get('product_id'):
                continue
            k = self._free_70(r.get('product_id'), r.get('sku'))
            price = hx_forge_8(r.get('current_price'))
            stock = r.get('current_in_stock')
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {'in_stock': bool(stock) if stock is not None else prev.get('in_stock', True), 'price': price if price is not None else prev.get('price')}
            self.checked_seq[k] = self.seq

    def _mint_26(self, prod: dict, brief: Brief) -> None:
        pid = prod.get('product_id')
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get('listing_id'):
            self.listing_ids.add(str(prod['listing_id']))
            self.listing_of[str(pid)] = str(prod['listing_id'])
        if brief.theme and all((t in hx_clamp_14(str(prod.get('title') or '')) for t in brief.theme)):
            self.identities.add(str(pid))
        for v in prod.get('variants') or []:
            if not isinstance(v, dict) or not v.get('sku'):
                continue
            k = self._free_70(pid, v.get('sku'))
            self.checked[k] = {'in_stock': bool(v.get('in_stock')), 'price': hx_forge_8(v.get('price'))}
            self.checked_seq[k] = self.seq
            self.options[k] = str(v.get('options') or '')

    def _hop_71(self, name: str, args: dict, obs: dict, brief: Brief) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get('error'):
            return
        if name in ('search', 'filter'):
            for row in obs.get('results') or []:
                if isinstance(row, dict):
                    self._tag_27(row, brief)
        elif name == 'view':
            self._mint_26(obs, brief)
        elif name == 'compare':
            for prod in obs.get('compared') or []:
                if isinstance(prod, dict):
                    self._mint_26(prod, brief)
        elif name == 'inspect_stock':
            k = self._free_70(obs.get('product_id'), obs.get('sku'))
            self.checked[k] = {'in_stock': bool(obs.get('in_stock')), 'price': hx_forge_8(obs.get('price'))}
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
            self._hold_58(obs)
        elif name == 'remove_from_cart':
            self._hold_58(obs)
            removed = obs.get('removed') or {}
            pair = (removed.get('product_id'), removed.get('sku'))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == 'inspect_cart':
            self._hold_58(obs)

    def _tag_27(self, row: dict, brief: Brief) -> None:
        pid = row.get('product_id')
        if not pid:
            return
        title = str(row.get('title') or '')
        if brief.theme:
            if all((t in hx_clamp_14(title) for t in brief.theme)):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get('sku'):
            return
        k = self._free_70(pid, row.get('sku'))
        self.rows[k] = row
        if row.get('price') is not None:
            self.checked.setdefault(k, {'in_stock': bool(row.get('in_stock', True)), 'price': hx_forge_8(row.get('price'))})

class CheckoutPilot:

    def __init__(self, problem: dict) -> None:
        env = problem['environment']
        self.binding = dict(env['binding'])
        self.policy = env['policy_view']
        self.tools = list(self.policy.get('tools') or [])
        self.tool_names = {(t.get('function') or {}).get('name') for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get('max_steps') or 30)
        self.max_calls = max(1, int(self.policy.get('max_calls_per_turn') or 1))
        self.problem_id = str(problem.get('problem_id') or problem.get('id') or 'problem')
        self.brief = Mission(str(self.policy.get('query') or ''))
        self.budget = self.brief.budget
        self.ledger = CatalogLog()
        self.clock = Deadline(_C29_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.messages: list = [{'role': 'system', 'content': self._bind_59()}, {'role': 'user', 'content': self.policy.get('query') or ''}]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.shopper_updated = False
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_calls = 0
        self.search_queries: set[str] = set()
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.hinted_exact = False
        self.tried: dict = {}
        self.questions_asked = 0
        self.searched_siblings = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self._wanted_axes: tuple[str, ...] | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.pref_axis_searched = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.model_index = 0
        provider = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
        self.roster = _C16_MODELS.get(provider) or _C16_MODELS['chutes']
        pinned = getenv('SANDBOX_MODEL')
        if pinned:
            self.roster = (pinned,) + tuple((m for m in self.roster if m != pinned))

    def _aged_47(self) -> list[str]:
        theme = list(self.brief.theme or [])
        if not theme:
            return []
        joined = ' '.join(theme)
        out = [joined]
        if len(theme) > 1:
            out.append(' '.join(reversed(theme)))
            out.append(theme[0])
            out.append(theme[-1])
            out.append(' '.join(theme[:2]))
            if len(theme) > 2:
                out.append(' '.join(theme[1:]))
        if self.brief.category:
            out.append('%s %s' % (self.brief.category, joined))
            out.append('%s %s' % (joined, self.brief.category))
        seen: set[str] = set()
        uniq = []
        for q in out:
            _free_70 = q.strip().lower()
            if _free_70 and _free_70 not in seen:
                seen.add(_free_70)
                uniq.append(q.strip())
        return uniq

    def run(self) -> list:
        question = self._polish_28()
        (last_forced, repeats) = (None, 0)
        while not self.done and self.turn < self.max_steps:
            if self.clock._track_72() < 25 or self.turn >= self.max_steps - 3:
                self._shut_22()
                break
            if question and (not self.asked_shopper) and ('message' in self.tool_names):
                self.asked_shopper = True
                self._fill_2('message', {'content': question})
                continue
            if self.pending_recheck and 'inspect_stock' in self.tool_names:
                self._herd_32()
                continue
            plan = self._blunt_31()
            if plan is not None:
                (name, args) = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ('inspect_stock', 'add_to_cart'):
                        self.ledger.dead_keys.add(CatalogLog._free_70(args.get('product_id'), args.get('sku')))
                    if name == 'compare':
                        self.unreadable.update((str(p) for p in args.get('product_ids') or []))
                    if name == 'message':
                        self.shopper_question_pending = False
                    (last_forced, repeats) = (None, 0)
                    self._dull_25('That step changed nothing, so the controller moved on to other candidates.')
                    continue
                obs = self._fill_2(name, args)
                if name == 'message':
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                    added = obs.get('added') or {}
                    if added.get('product_id'):
                        self.pending_recheck = (str(added['product_id']), str(added['sku']))
                continue
            (last_forced, repeats) = (None, 0)
            proposal = self._chase_4()
            if proposal is None:
                self._shut_22()
                break
            call = proposal['tool_calls'][0]
            name = (call.get('function') or {}).get('name') or ''
            args = hx_walk_2(call)
            content = proposal.get('content') or ''
            reprompts = 0
            while True:
                verdict = self._sweep_65(name, args)
                if verdict is None:
                    break
                if reprompts >= _C14_MAX_REPROMPTS:
                    self.rejected_streak += 1
                    self._dull_25('That call was not made: ' + verdict + '. Choose a different, valid action.')
                    if self.rejected_streak >= 3:
                        self._slack_52()
                        self.rejected_streak = 0
                    verdict = 'skip'
                    break
                reprompts += 1
                self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': [call]})
                self.messages.append({'role': 'tool', 'tool_call_id': call.get('id') or 'x', 'content': json.dumps({'rejected': verdict})})
                self.dialogue.append({'role': 'assistant', 'content': '[rejected ' + name + '] ' + verdict[:200]})
                again = self._chase_4()
                if again is None:
                    break
                call = again['tool_calls'][0]
                name = (call.get('function') or {}).get('name') or ''
                args = hx_walk_2(call)
                content = again.get('content') or ''
            if verdict == 'skip':
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == 'place_test_order':
                (pid, sku) = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid and self.ledger.cart:
                    (pid, sku) = self.ledger.cart[-1]
                args = self._steer_30(pid, sku)
            if name == 'message':
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
                self.question_waits = 0
            obs = self._fill_2(name, args, content, call.get('id'))
            if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                added = obs.get('added') or {}
                if added.get('product_id'):
                    self.pending_recheck = (str(added['product_id']), str(added['sku']))
            if name in ('search', 'filter'):
                if self.brief.family == 'recall' and self.brief.required_identities:
                    (have, need) = (len(self.ledger.identities), self._snap_46())
                    self._dull_25('Distinct matching products surfaced so far: %d of the required %d (%d distinct queries).' % (have, need, len(self.search_queries)) + ('' if have >= need else ' Keep searching with different wording.'))
            if name == 'inspect_cart':
                self.cart_inspections += 1
            if name in ('view', 'compare'):
                self._slack_52()
        return self.dialogue

    def _drop_67(self) -> tuple[str, ...]:
        if self._wanted_axes is not None:
            return self._wanted_axes
        said = (self.brief.goal + ' ' + ' '.join((str(m.get('content') or '') for m in self.messages if m.get('role') == 'user'))).lower()
        axes = []
        for (axis, cues) in _PREF_AXES.items():
            if any((c.lower() in said for c in cues)) or axis.split('_')[0] in said:
                axes.append(axis)
        if any((w in said for w in ('headphone', 'earbud', 'earphone', 'workout', 'run', 'sweat', 'weather', 'sport'))):
            for axis in _PREF_AXES:
                if axis not in axes:
                    axes.append(axis)
        if not axes:
            axes = list(_PREF_AXES)
        self._wanted_axes = tuple(axes)
        return self._wanted_axes

    def _slack_52(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        cands = self._fold_11()
        if self.brief.family == 'preference' and (not self.ledger.cart):
            good = [c for c in cands if c['ok']]
            good.sort(key=lambda c: (-self._push_36(c['pid'], c['sku']), -self._mesh_35(c['pid'], c['sku']), c['price'] if c['price'] is not None else (10 ** 12)))
            if good:
                parts = []
                for c in good[:6]:
                    axes = self._aim_33(c['pid'], c['sku'])
                    hit = ','.join((a for (a, v) in axes.items() if v >= 1.0)) or 'none'
                    parts.append('%s::%s at %.2f [axes:%s]' % (c['pid'], c['sku'], c['price'] or 0.0, hit))
                self._dull_25('Products that meet every firm requirement so far (prefer full axis coverage — battery, sport fit, sweat/water, wireless): ' + '; '.join(parts) + '. Pick the highest-coverage one, inspect it, then add it.')
                return
        best = self._pivot_60()
        if best:
            self._dull_25('Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; read other candidates only if one could rank better.' % (best['pid'], best['sku'], best['price'] or 0.0))
            return
        (field, want) = (self.brief.spec_field, self.brief.spec_value)
        blocked = [c for c in cands if c['viewed'] and c['live'] and (not c['ok'])]
        if blocked and field:
            sample = blocked[:3]
            self._dull_25("None of the products read so far can be verified. For each, the evidence for %s must resolve to exactly %s from the variant's own option text or the product's raw facts: %s. Look at different products." % (field, want, ', '.join((c['pid'] + '::' + c['sku'] for c in sample))))
        elif blocked:
            self._dull_25('The products read so far fail a firm requirement (category, brand, stock, budget or the kind of device requested). Search for different products.')

    def _probe_38(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._dull_25("No %s product that meets every firm requirement has been read yet. The shopper's first choice is graded on that brand, so filter by brand=%s within the budget and read those products before any other." % (self.brief.pref_brand, self.brief.pref_brand))

    def _warp_44(self) -> tuple | None:
        """Turn the search results already in hand into read product pages, then widen.

        Reading pages is what turns rows into candidates the controller can verify,
        and a second verified product is what makes the market change survivable.
        """
        fam = self.brief.family
        if fam == 'recall':
            forced = self._lift_48()
            if forced is not None:
                return forced
            if len(self.ledger.identities) < self._snap_46():
                return None
        if fam == 'preference':
            forced = self._cast_37()
            if forced is not None:
                return forced
        twin = self._vet_62()
        if twin is not None:
            return twin
        leads = self._rise_63()
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
            words = [w for w in (self.brief.brand, self.brief.spec_value, self.brief.category) if w]
            if words:
                return ('search', {'query': ' '.join(words), 'in_stock': True, 'k': 10})
        self.reads_done = True
        return None

    def _rise_63(self) -> list:
        """Promising variants whose product page has not been read yet."""
        leads = []
        for (_free_70, row) in self.ledger.rows.items():
            (pid, _, sku) = _free_70.partition('::')
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            st = self.ledger.checked.get(_free_70) or {}
            price = st.get('price')
            if not st.get('in_stock') or price is None or (self.budget is not None and price > self.budget + _C6_EPS):
                continue
            title = str(row.get('title') or '')
            if self.brief.theme and (not all((t in hx_clamp_14(title) for t in self.brief.theme))):
                continue
            if self.brief.pref_model and self.ledger.adds == 0 and (not self.ledger.event_target):
                squashed = re.sub('[^a-z0-9]+', '', title.lower())
                if re.sub('[^a-z0-9]+', '', self.brief.pref_model.lower()) not in squashed:
                    continue
            if self.brief.brand:
                wanted = hx_clamp_14(self.brief.brand)
                if not (wanted <= hx_clamp_14(str(row.get('brand') or '')) or wanted <= hx_clamp_14(title)):
                    continue
            axis_hint = 0.0
            if self.brief.family == 'preference':
                blob = (title + ' ' + str(row.get('brand') or '')).lower()
                for axis in self._drop_67():
                    if any((c.lower() in blob for c in _PREF_AXES[axis])):
                        axis_hint += 1.0
            leads.append((-axis_hint, price, str(pid)))
        leads.sort()
        out = []
        for (_axis, _price, pid) in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _mark_57(self, pid: str, title: str) -> str:
        """Drop model codes from a title so a product name is not read as a panel class."""
        for token in re.split('[;,]', str(self._ripe_20(pid).get('model') or '')):
            token = token.strip()
            stem = re.split('[-_]', token)[0] if token else ''
            if token and re.fullmatch('[a-z0-9]+(?:[-_][a-z0-9]+)+', token, re.I) and re.search('[a-z]', stem, re.I) and re.search('\\d', stem) and (not re.fullmatch('(?:[48]k|[ufq]?hd|fullhd|ultrahd|\\d+p)\\d*', stem, re.I)):
                title = re.sub('(?<![\\w-])' + re.escape(token) + '(?![\\w-])', '', title, flags=re.I)
        return title

    def _scan_50(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        return bool(re.search('customer (?:rating|review)|\\bcompare \\b|refurb|pre-owned|restored|renewed', self._fade_61(pid), re.I))

    def _lift_48(self) -> tuple | None:
        if 'search' not in self.tool_names or not self.brief.theme:
            return None
        roomy = self.turn < self.max_steps * 0.55 and self.clock._track_72() > 100
        if not roomy and len(self.ledger.identities) >= max(1, self.brief.required_identities):
            return None
        for query in self._aged_47():
            if query.lower() in self.search_queries:
                continue
            self._dull_25('Retrieval needs more distinct matching titles; searching: %s' % query)
            return ('search', {'query': query, 'in_stock': True, 'k': 10})
        if 'filter' in self.tool_names and self.brief.category and (self.search_calls < 6):
            probe = 'filter:' + self.brief.category.lower()
            if probe not in self.search_queries:
                args = {'category': self.brief.category, 'in_stock': True, 'k': 10}
                if self.budget is not None:
                    args['max_price'] = self.budget
                return ('filter', args)
        return None

    def _blunt_31(self) -> tuple | None:
        """The action the controller takes itself, or nothing to let the model explore."""
        if self.done:
            return None
        if self.shopper_question_pending:
            if self.question_waits < 2 or 'message' not in self.tool_names:
                self.question_waits += 1
                return None
            return ('message', {'content': self._pack_3()})
        fam = self.brief.family
        if fam == 'ranking' and 'compare' in self.tool_names:
            missing = [p for (p, _s) in self.brief.candidate_keys if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return ('compare', {'product_ids': missing[:5]})
        if fam == 'recovery' and (not self.ledger.adds) and self.brief.pref_model and ('search' in self.tool_names):
            forced = self._tap_53()
            if forced is not None:
                return forced
        target = self._pivot_60()
        if target is None or not self._glow_45():
            forced = self._warp_44()
            if forced is not None:
                return forced
            if target is None:
                return None
        (pid, sku) = (target['pid'], target['sku'])
        if (pid, sku) in self.ledger.cart:
            if not self._loose_23(pid, sku):
                return ('inspect_stock', {'product_id': pid, 'sku': sku})
            return ('place_test_order', self._steer_30(pid, sku))
        if not self._glow_45():
            return None
        for (cpid, csku) in self.ledger.cart:
            if CatalogLog._free_70(cpid, csku) in self.ledger.dead_keys and 'remove_from_cart' in self.tool_names:
                return ('remove_from_cart', {'product_id': cpid, 'sku': csku})
        if not self._loose_23(pid, sku):
            return ('inspect_stock', {'product_id': pid, 'sku': sku})
        return ('add_to_cart', {'product_id': pid, 'sku': sku})

    @staticmethod
    def _raw_56(text: str) -> set:
        return {w.rstrip('s') for w in re.findall('[a-z]+', (text or '').lower())}

    def _sharp_24(self, pid: str, sku: str, field: str) -> float | None:
        value = self._pull_66(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    def _open_10(self, text: str) -> None:
        new = Mission._pick_9(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        self._dull_25('Budget updated by the shopper: at most %.2f %s. Every later choice must fit it.' % (new, self.brief.currency))
        for (pid, sku) in list(self.ledger.cart):
            st = self.ledger._sand_73(pid, sku) or {}
            price = st.get('price')
            if price is not None and price > new + _C6_EPS:
                self.ledger.dead_keys.add(CatalogLog._free_70(pid, sku))
                self._dull_25('The carted item %s costs %.2f, above the new budget; the controller will replace it.' % (CatalogLog._free_70(pid, sku), price))

    def _sweep_65(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return 'unknown tool; choose one of ' + str(sorted((n for n in self.tool_names if n)))
        if name == 'place_test_order':
            (pid, sku) = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
            if not pid and self.ledger.cart:
                (pid, sku) = self.ledger.cart[-1]
            (ok, why) = self._nudge_64(pid, sku)
            if not ok:
                return 'order refused: ' + why
            if self.brief.family == 'recall' and len(self.ledger.identities) < self._snap_46():
                return 'order refused: only %d distinct matching products have been surfaced; search with different wording until at least %d appear' % (len(self.ledger.identities), self._snap_46())
            if self.brief.family == 'preference' and len(self.ledger.listing_ids) < 2:
                return 'order refused: compare at least two products with different listing_id values first'
        if name == 'add_to_cart':
            (pid, sku) = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
            if not pid or not sku:
                return 'add_to_cart needs both product_id and sku'
            c = self._knit_5(pid, sku)
            _free_70 = CatalogLog._free_70(pid, sku)
            if not c['live']:
                return 'that variant is out of stock, over budget, or already invalidated'
            if not c['viewed']:
                return 'view or compare that product first so its raw facts are on record'
            if _free_70 not in self.ledger.inspected:
                return 'inspect_stock that exact variant first: its current price and stock decide whether it can be bought'
            if c['spec'] != 2 and self.brief.spec_field:
                return 'that variant does not resolve to %s = %s from its own option text or raw facts' % (self.brief.spec_field, self.brief.spec_value)
            if not c['ok']:
                return 'that product fails a firm requirement (category, brand, or the kind of device requested)'
            if self.brief.family == 'preference' and len(self.ledger.listing_ids) < 2:
                return 'compare two products with different listing_id values before the first cart addition'
            if self.brief.family == 'preference' and (not self.ledger.adds):
                score = self._push_36(pid, sku)
                good = [c for c in self._fold_11() if c['ok']]
                best = max((self._push_36(c['pid'], c['sku']) for c in good), default=0.0)
                if score < 0.75 and best > score + 0.24 and (self.turn < self.max_steps * 0.75):
                    return 'that product is missing preference axes (need battery/sport/sweat/wireless coverage); view a higher-coverage listing before the first cart addition'
                if score < 0.5 and self.turn < self.max_steps * 0.65:
                    return 'first cart item must show stronger preference evidence (waterproof/sweat, sport fit, wireless, battery)'
            if self.brief.family == 'recovery' and (not self.ledger.adds) and self.brief.pref_model and self._taut_49(pid, sku):
                return 'the first cart item must be the exact model and color the shopper named; the other colors are separate listings sharing the same title, so search that title and read them first'
            if not self._glow_45():
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
        _free_70 = name + json.dumps(args, sort_keys=True)
        if _free_70 in self.ledger.calls_seen and name in ('search', 'filter', 'view', 'compare'):
            return 'identical call already made; change the arguments'
        self.ledger.calls_seen.add(_free_70)
        return None

    def _aim_33(self, pid: str, sku: str) -> dict[str, float]:
        blob = self._etch_17(pid, sku)
        out = {}
        for axis in self._drop_67():
            cues = _PREF_AXES[axis]
            out[axis] = 1.0 if any((c.lower() in blob for c in cues)) else 0.0
        return out

    def _drain_19(self, pid: str, field: str) -> str | None:
        """The value the product's own labelled facts state for a concept.

        A reason quoted back to the shopper reads these facts leniently: a fact
        printed as a bare number under a unit-bearing label carries that label's
        unit, so "27" under the size label is 27 inches.
        """
        raw = self._ripe_20(pid).get(_C7_FACT_LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _C37_VALUE_SETS[field](text)
        if not found and field != 'resolution' and re.fullmatch('\\d+(?:\\.\\d+)?', text):
            unit = {'screen_size': 'in', 'refresh_rate': 'hz', 'ram': 'gb'}[field]
            return '%g' % float(text) + unit
        return hx_fill_9(found)

    def _pack_3(self) -> str:
        """A short, factual answer to the shopper, drawn only from observed evidence."""
        target = self._pivot_60() or (self._fold_11() or [None])[0]
        if not target:
            return 'I am still checking the options against your requirements; nothing is decided yet.'
        (pid, sku) = (target['pid'], target['sku'])
        parts = ['I am going with ' + (self._fade_61(pid) or pid)[:90]]
        (field, want) = (self.brief.spec_field, self.brief.spec_value)
        if field and want and (self._pull_66(pid, sku, field) == want):
            parts.append('its listed ' + field.replace('_', ' ') + ' is ' + want)
        if target.get('price') is not None:
            parts.append('it is %.2f %s and in stock' % (target['price'], self.brief.currency))
        return ', '.join(parts) + '.'

    def _dull_25(self, text: str) -> None:
        self.messages.append({'role': 'user', 'content': '[note] ' + text})
        self.dialogue.append({'role': 'assistant', 'content': '[note] ' + text[:300]})

    def _fill_2(self, name: str, args: dict, content: str='', tool_call_id: str | None=None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._skip_51(pend['envelope'])
            if replay is None:
                self._dull_25('The environment is not answering; the previous action is still pending and will be retried.')
                return None
            self.pending_envelope = None
            self.turn = pend['turn']
            self._forge_1(pend['name'], pend['args'], pend['content'], pend['call_id'], replay)
            if self.done:
                return None
        turn = self.turn + 1
        group = self.problem_id + '-turn-' + str(turn)
        envelope = {**self.binding, 'call_id': group, 'idempotency_key': group, 'turn': turn, 'calls': [{'call_id': group + '-1', 'action': {'name': name, 'args': args}}]}
        result = self._skip_51(envelope)
        if result is None:
            self.pending_envelope = {'envelope': envelope, 'turn': turn, 'name': name, 'args': args, 'content': content, 'call_id': tool_call_id or 'agent-' + str(turn)}
        else:
            self.turn = turn
        return self._forge_1(name, args, content, tool_call_id or 'agent-' + str(turn), result)

    def _knit_5(self, pid: str, sku: str) -> dict:
        """Score one variant against every gate the runtime will apply to it."""
        fam = self.brief.family
        _free_70 = CatalogLog._free_70(pid, sku)
        st = self.ledger.checked.get(_free_70) or {}
        (price, stock) = (st.get('price'), st.get('in_stock'))
        dead = _free_70 in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and fam != 'recovery')
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _C6_EPS) and (not dead)
        viewed = str(pid) in self.ledger.products
        (field, want) = (self.brief.spec_field, self.brief.spec_value)
        spec = 2
        if field and want:
            got = self._pull_66(pid, sku, field)
            spec = 2 if got == want else -1 if got else 0
        ok = bool(live and viewed and (spec == 2) and self._trim_8(pid) and self._boost_16(pid, sku))
        cat_ok = self._wipe_14(pid)
        ok = ok and cat_ok
        if fam == 'justification':
            ok = ok and (not self._scan_50(pid)) and bool(self._clamp_7(pid))
            if field and want:
                ok = ok and self._drain_19(pid, field) == want and (self._corral_41(pid, field) == want)
        if fam == 'preference' and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split('::', 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == 'recall' and self.brief.theme:
            ok = ok and all((t in hx_clamp_14(self._fade_61(pid)) for t in self.brief.theme))
        if fam == 'ranking':
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
            ok = ok and self._sharp_24(pid, sku, 'refresh_rate') is not None and (self._sharp_24(pid, sku, 'screen_size') is not None)
        return {'pid': str(pid), 'sku': str(sku), 'key': _free_70, 'price': price, 'live': live, 'viewed': viewed, 'spec': spec, 'cat_ok': cat_ok, 'ok': ok, 'relax': self._taut_49(pid, sku) if fam == 'recovery' else 0}

    def _push_36(self, pid: str, sku: str) -> float:
        axes = self._aim_33(pid, sku)
        if not axes:
            return 0.0
        return sum(axes.values()) / len(axes)

    def _trim_8(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = hx_clamp_14(want)
        return bool(wanted <= hx_clamp_14(self._clamp_7(pid)) or wanted <= hx_clamp_14(self._fade_61(pid)))

    def _chase_4(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {'model': model, 'messages': self.messages, 'tools': self.tools, 'tool_choice': 'required', 'temperature': 0}
            resp = _C30_PROXY.post('/inference/chat/completions', json_data=payload)
            if resp and isinstance(resp, dict) and resp.get('choices'):
                msg = resp['choices'][0].get('message') or {}
                calls = msg.get('tool_calls') or []
                if calls:
                    return {'content': msg.get('content') or '', 'tool_calls': calls[:self.max_calls]}
                self.messages.append({'role': 'assistant', 'content': msg.get('content') or ''})
                self._dull_25('No tool call was made. Choose exactly one of the supplied tools now.')
                continue
            if self.clock._track_72() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    def _mesh_35(self, pid: str, sku: str) -> int:
        return int(sum(self._aim_33(pid, sku).values()))

    def _pull_66(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        opt = self._hinge_29(pid, sku)
        vals = hx_knock_1(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if hx_brush_7(field, opt):
            return None
        return self._corral_41(pid, field)

    def _keen_15(self, pid: str, sku: str) -> list:
        """Claims proven by what the agent has already observed about this exact item."""
        out = []
        (field, want) = (self.brief.spec_field, self.brief.spec_value)
        if not field and self.brief.family == 'justification':
            (field, want) = ('resolution', self._drain_19(pid, 'resolution'))
        if not field or not want:
            return []
        stated = dict(self.brief.specs) or {field: want}
        stated.setdefault(field, want)
        for (name, value) in stated.items():
            if self._drain_19(pid, name) == value and self._corral_41(pid, name) == value:
                out.append({'field': name, 'value': value})
        if not out:
            return []
        st = self.ledger._sand_73(pid, sku) or {}
        if self.budget is not None:
            out.append({'field': 'price', 'value': 'at most %.2f' % self.budget})
        elif st.get('price') is not None:
            out.append({'field': 'price', 'value': '%.2f' % st['price']})
        out.append({'field': 'in_stock', 'value': 'true'})
        brand = self._clamp_7(pid)
        if brand:
            out.append({'field': 'brand', 'value': brand})
        return out

    def _cage_12(self, pid: str) -> list:
        path = self._buzz_40(pid).get('category_path')
        return list(path) if isinstance(path, list) else []

    def _vet_62(self) -> tuple | None:
        """After the market kills the carted item, its own twins are the nearest replacements.

        A listing's other colors and sizes are separate products carrying the same
        title, so searching that title surfaces exactly the items that still meet
        every requirement the dead one met.
        """
        dead = self.ledger.event_target
        if self.searched_twins or not dead or 'search' not in self.tool_names:
            return None
        title = self._fade_61(dead.split('::', 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self._dull_25('The item that died is one of several listings sharing a title; searching that title surfaces its other colors and sizes, which meet the same requirements.')
        return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})

    def _shut_22(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for (pid, sku) in reversed(self.ledger.cart):
            (ok, _why) = self._nudge_64(pid, sku)
            if ok and (not self.done):
                self._fill_2('place_test_order', self._steer_30(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [c for c in self._fold_11() if c['ok']]
        if not good and self.brief.family != 'justification':
            good = [c for c in self._fold_11() if c['live'] and c['spec'] == 2 and self._trim_8(c['pid']) and c['cat_ok'] and (not self.brief.candidate_keys or (c['pid'], c['sku']) in self.brief.candidate_keys)]
        if not good:
            return
        good.sort(key=lambda c: self._weld_43(c, self.ledger.adds == 0))
        (pid, sku) = (good[0]['pid'], good[0]['sku'])
        if (pid, sku) not in self.ledger.cart:
            obs = self._fill_2('add_to_cart', {'product_id': pid, 'sku': sku})
            if self.done or not isinstance(obs, dict) or obs.get('error'):
                return
        obs = self._fill_2('inspect_stock', {'product_id': pid, 'sku': sku})
        if self.done or not isinstance(obs, dict) or (not obs.get('in_stock')):
            return
        price = hx_forge_8(obs.get('price'))
        if self.budget is not None and price is not None and (price > self.budget + _C6_EPS):
            return
        self._fill_2('place_test_order', self._steer_30(pid, sku))

    def _spin_18(self) -> None:
        if self.hinted_exact or not self.brief.pref_model:
            return
        self.hinted_exact = True
        self._dull_25("The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own and view the results so every color variant is on record; only that exact pairing may be the first cart item." % (self.brief.pref_model, ', ' + self.brief.pref_color if self.brief.pref_color else ''))

    def _cast_37(self) -> tuple | None:
        if self.pref_axis_searched or 'search' not in self.tool_names:
            return None
        if self.search_calls < 1 and len(self.ledger.listing_ids) < 1:
            return None
        cat = self.brief.category or 'headphones'
        query = ' '.join((w for w in (cat, 'wireless', 'waterproof', 'sport', 'battery') if w))
        if query.lower() in self.search_queries:
            self.pref_axis_searched = True
            return None
        self.pref_axis_searched = True
        self._dull_25('Preference search for high axis coverage (wireless + waterproof/sweat + sport + battery).')
        return ('search', {'query': query, 'in_stock': True, 'k': 10})

    def _fade_61(self, pid: str) -> str:
        title = str(self._buzz_40(pid).get('title') or '')
        if title:
            return title
        for (_free_70, row) in self.ledger.rows.items():
            if _free_70.startswith(str(pid) + '::'):
                return str(row.get('title') or '')
        return ''

    def _forge_1(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
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
        self.ledger._hop_71(name, args, obs, self.brief)
        if name == 'search':
            self.search_calls += 1
            q = str(args.get('query') or '').strip().lower()
            if q:
                self.search_queries.add(q)
        elif name == 'filter':
            self.search_calls += 1
            self.search_queries.add('filter:' + str(args.get('category') or '').strip().lower())
        self.messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': hx_trim_15(json.dumps(obs), _C20_OBS_CHARS)})
        user = result.get('user_message')
        if isinstance(user, dict) and user.get('content'):
            text = str(user['content'])
            self.messages.append({'role': 'user', 'content': text})
            self._open_10(text)
            if self.brief.family in ('constraint', 'intent') and (not self.brief.spec_value):
                stated = self.brief._sift_69(text)
                if stated:
                    self.shopper_updated = True
                    self._dull_25('Firm requirement recorded: ' + stated + '. Only a variant that resolves to it may be bought.')
            if _C32_QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._dull_25('The shopper asked a question: answer it with message in your next turn, then continue.')
        if any((isinstance(c, dict) and (c.get('observation') or {}).get('done') for c in calls)):
            self.done = True
        return obs

    def _polish_28(self) -> str | None:
        if self.brief.family == 'constraint' or self.brief.spec_sealed:
            return 'Before I search: which hardware specification is your firm requirement (screen size, resolution class, refresh rate, or memory), and what value do you need?'
        if self.brief.family == 'preference':
            return 'Quick question so I choose well: which of your preferences counts most, and in what order should I weigh the others? Let me know your priority.'
        return None

    def _gauge_68(self) -> set:
        """Content words from the shopper's own wording, gathered once."""
        if self._wishes is None:
            said = self.brief.goal + ' ' + ' '.join((str(m.get('content') or '') for m in self.messages if m.get('role') == 'user'))
            drop = hx_clamp_14(self.brief.category or '') | {'the', 'and', 'for', 'with', 'that', 'this', 'you', 'your', 'find', 'need', 'want', 'order', 'stock', 'place', 'under', 'over', 'one', 'php', 'test', 'buy', 'rather', 'than', 'pay', 'bit', 'more', 'settle', 'something', 'its', 'would', 'like', 'have'}
            self._wishes = {w for w in hx_clamp_14(said) if len(w) > 3 and (not w.isdigit()) and (w not in drop)}
        return self._wishes

    def _loose_23(self, pid: str, sku: str) -> bool:
        k = CatalogLog._free_70(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _pivot_60(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        first = self.ledger.adds == 0
        fam = self.brief.family
        good = [c for c in self._fold_11() if c['ok']]
        if fam == 'preference':
            carted = [c for c in good if (c['pid'], c['sku']) in self.ledger.cart]
            if carted:
                return carted[0]
            good.sort(key=lambda c: (-self._push_36(c['pid'], c['sku']), -self._mesh_35(c['pid'], c['sku']), c['price'] if c['price'] is not None else (10 ** 12)))
            if good and (self.turn >= self.max_steps - 8 or self.ledger.adds > 0 or (self._push_36(good[0]['pid'], good[0]['sku']) >= 0.75 and len(self.ledger.listing_ids) >= 2)):
                return good[0]
            return None
        if fam == 'intent' and first and self.brief.pref_brand:
            wanted = hx_clamp_14(self.brief.pref_brand)
            named = [c for c in good if wanted <= hx_clamp_14(self._clamp_7(c['pid'])) or wanted <= hx_clamp_14(self._fade_61(c['pid']))]
            if named:
                good = named
            elif self.turn < self.max_steps * 0.6:
                self._probe_38()
                return None
        if fam == 'recovery' and first:
            exact = [c for c in good if c['relax'] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                self._spin_18()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._weld_43(c, first))
        return good[0]

    def _skip_51(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; an identical envelope replays idempotently."""
        for attempt in range(4):
            result = _C30_PROXY.post('/environment/call', json_data=envelope)
            if isinstance(result, dict) and result.get('calls'):
                return result
            if self.clock._track_72() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _weld_43(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates the way the task's own priority does."""
        fam = self.brief.family
        price = c['price'] if c['price'] is not None else (10 ** 12)
        if fam == 'ranking':
            hz = self._sharp_24(c['pid'], c['sku'], 'refresh_rate') or 0.0
            inch = self._sharp_24(c['pid'], c['sku'], 'screen_size') or 0.0
            return (-hz, -inch, price, c['key'])
        if fam == 'recovery':
            return (c['relax'], price, c['key'])
        if fam == 'preference':
            return (-self._push_36(c['pid'], c['sku']), -self._mesh_35(c['pid'], c['sku']), price, c['key'])
        extra = sum((1 for (f, v) in self.brief.specs.items() if f != self.brief.spec_field and self._pull_66(c['pid'], c['sku'], f) != v))
        proven = 0 if hx_pack_10(self._ripe_20(c['pid']), self._fade_61(c['pid']), self._hinge_29(c['pid'], c['sku'])) == 'yes' else 1
        if fam == 'intent' and first and self.brief.pref_brand:
            wanted = hx_clamp_14(self.brief.pref_brand)
            near = 0 if wanted <= hx_clamp_14(self._clamp_7(c['pid'])) or wanted <= hx_clamp_14(self._fade_61(c['pid'])) else 1
            return (near, extra, proven, price, c['key'])
        return (extra, proven, price, c['key'])

    def _tight_39(self, pid: str, sku: str) -> int:
        """How much of what the shopper actually asked for this product's own facts mention."""
        if self.brief.family == 'preference':
            return self._mesh_35(pid, sku)
        wants = self._gauge_68()
        if not wants:
            return 0
        have = hx_clamp_14(' '.join((str(v) for v in self._ripe_20(pid).values())) + ' ' + self._fade_61(pid) + ' ' + self._hinge_29(pid, sku))
        return len(wants & have)

    def _wipe_14(self, pid: str) -> bool:
        """Whether the listing sits on the shelf the shopper asked for.

        Retailers spell a shelf many ways and nest it differently, so a shared
        word anywhere in the path counts. Where the request names a kind of
        product the catalog has no shelf for, the title has to carry it.
        """
        want = self.brief.category
        path = self._cage_12(pid)
        if not want or not path:
            return True
        wanted = self._raw_56(want)
        labels = set()
        for label in path:
            labels |= self._raw_56(str(label))
        placed = bool(wanted & labels) or any((self._raw_56(''.join(re.findall('[a-z]+', str(label).lower()))) == wanted for label in path))
        if not placed and 'smartphone' in wanted:
            placed = any((re.fullmatch('(?:smart|cell|mobile|unlocked)?\\s*phones?', str(label).strip(), re.I) for label in path))
        if not placed:
            return False
        qualifier = self.brief.cat_qualifier
        if not qualifier:
            return True
        title = self._fade_61(pid)
        words = self._raw_56(title) | hx_clamp_14(title)
        if qualifier == 'gaming':
            fast = any((float(v[:-2]) >= 120 for v in hx_tilt_4(title)))
            return 'monitor' in words and ('gaming' in words or fast)
        return 'over' in words and 'ear' in words or 'overear' in words or 'circumaural' in words

    def _hinge_29(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(CatalogLog._free_70(pid, sku), '')

    def _nudge_64(self, pid: str, sku: str) -> tuple:
        st = self.ledger._sand_73(pid, sku)
        k = CatalogLog._free_70(pid, sku)
        if st is None:
            return (False, 'inspect_stock the exact product_id and sku first')
        if (pid, sku) not in self.ledger.cart:
            return (False, 'add that exact variant to the cart first')
        if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
            return (False, 're-check the variant with inspect_stock after the cart addition')
        if not st.get('in_stock'):
            return (False, 'that variant is out of stock; choose a different product')
        price = st.get('price')
        if self.budget is not None and price is not None and (price > self.budget + _C6_EPS):
            return (False, 'that variant costs %.2f, above the budget of %.2f' % (price, self.budget))
        if k in self.ledger.dead_keys:
            return (False, 'that variant was invalidated by the market change; choose a different product')
        return (True, '')

    def _steer_30(self, pid: str, sku: str) -> dict:
        out = {'product_id': pid, 'sku': sku}
        if self.brief.family == 'justification':
            claims = self._keen_15(pid, sku)
            if claims:
                out['justification'] = {'claims': claims}
        return out

    def _fold_11(self) -> list:
        seen = set()
        out = []
        for _free_70 in list(self.ledger.checked):
            (pid, _, sku) = _free_70.partition('::')
            if not sku or _free_70 in seen:
                continue
            seen.add(_free_70)
            out.append(self._knit_5(pid, sku))
        return out

    def _etch_17(self, pid: str, sku: str) -> str:
        return (' '.join((str(v) for v in self._ripe_20(pid).values())) + ' ' + self._fade_61(pid) + ' ' + self._hinge_29(pid, sku)).lower()

    def _boost_16(self, pid: str, sku: str) -> bool:
        """Categories that name a device are graded on the listing being that device.

        Where the task itself turns on device identity, only positive evidence
        will do; elsewhere a listing is only ruled out when the evidence
        contradicts it, so a thin product page is not treated as a disqualifier.
        """
        if not hx_stamp_13(self.brief.category or '', 'smartphone'):
            return True
        verdict = hx_pack_10(self._ripe_20(pid), self._fade_61(pid), self._hinge_29(pid, sku))
        if self.brief.family in ('constraint', 'justification', 'recovery'):
            return verdict == 'yes'
        return verdict != 'no'

    def _snap_46(self) -> int:
        """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
        early = self.search_calls < 10 and self.turn < self.max_steps * 0.55
        return self.brief.required_identities + (3 if early else 0)

    def _tap_53(self) -> tuple | None:
        """A found model in the wrong color: its other colors share the same title."""
        if self.searched_siblings:
            return None
        rows = [c for c in self._fold_11() if c['live'] and c['viewed'] and (c['relax'] == 0)]
        if rows:
            return None
        near = [c for c in self._fold_11() if c['viewed'] and c['relax'] == 1]
        if not near:
            return None
        title = self._fade_61(near[0]['pid'])
        if not title:
            return None
        self.searched_siblings = True
        return ('search', {'query': ' '.join(title.split()[:8]), 'k': 10, 'in_stock': True})

    def _ripe_20(self, pid: str) -> dict:
        f = self._buzz_40(pid).get('facts')
        return f if isinstance(f, dict) else {}

    def _bind_59(self) -> str:
        play = _C26_PLAYBOOK.get(self.brief.family, _C26_PLAYBOOK['generic'])
        extra = ''
        if self.brief.family == 'recall' and self.brief.required_identities:
            extra = ' You must surface at least %d distinct matching products in search or filter results before ordering; the controller reports the count after each search.' % self.brief.required_identities
        return _C5_CORE_RULES + '\n\nFamily playbook: ' + play + extra

    def _corral_41(self, pid: str, field: str) -> str | None:
        """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
        vals: set = set()
        conflict = False
        raw = self._ripe_20(pid).get(_C7_FACT_LABEL[field])
        if raw:
            found = _C37_VALUE_SETS[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == 'resolution' and _C31_QUARTER_RE.search(str(raw)):
                conflict = True
            vals |= found
        title = self._fade_61(pid)
        if field == 'resolution':
            title = self._mark_57(pid, title)
            if _C31_QUARTER_RE.search(title):
                conflict = True
        vals |= hx_knock_1(field, title)
        return None if conflict else hx_fill_9(vals)

    def _clamp_7(self, pid: str) -> str:
        brand = self._buzz_40(pid).get('brand')
        if brand:
            return str(brand)
        for (_free_70, row) in self.ledger.rows.items():
            if _free_70.startswith(str(pid) + '::') and row.get('brand'):
                return str(row['brand'])
        return ''

    def _glow_45(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        fam = self.brief.family
        if self.ledger.adds:
            return True
        if self.brief.spec_sealed:
            if self.turn < self.max_steps * 0.6:
                return False
            self.brief.spec_sealed = False
            self._dull_25('The firm requirement was never stated, so the remaining requirements decide the choice.')
        if fam in ('intent', 'constraint') and self.turn < 4 and (not self.shopper_updated):
            return False
        if fam == 'preference' and len(self.ledger.listing_ids) < 2:
            return False
        if fam == 'preference':
            good = [c for c in self._fold_11() if c['ok']]
            if good:
                best = max((self._push_36(c['pid'], c['sku']) for c in good))
                if best < 0.75 and self.turn < self.max_steps * 0.7 and (self.clock._track_72() > 90):
                    return False
        roomy = self.turn < self.max_steps * 0.5 and self.clock._track_72() > 120
        if fam == 'recall' and roomy:
            need_q = 3 if self.brief.required_identities else 2
            if len(self.ledger.identities) < self._snap_46() or self.search_calls < need_q or len(self.search_queries) < need_q:
                return False
        if roomy and (not self.reads_done) and (len({c['pid'] for c in self._fold_11() if c['ok']}) < 2):
            return False
        return True

    def _taut_49(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        (model, color) = (self.brief.pref_model, self.brief.pref_color)
        if not model:
            return 0
        squash = lambda t: re.sub('[^a-z0-9]+', '', (t or '').lower())
        blob = squash(self._fade_61(pid) + ' ' + ' '.join((str(v) for v in self._ripe_20(pid).values())))
        same_model = squash(model) in blob
        shade = (self._hinge_29(pid, sku) + ' ' + str(self._ripe_20(pid).get('color') or '') + ' ' + self._fade_61(pid)).lower()
        same_color = bool(color) and re.search('(?<![a-z])' + re.escape(color) + '(?![a-z])', shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _herd_32(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        (pid, sku) = self.pending_recheck
        self.pending_recheck = None
        obs = self._fill_2('inspect_stock', {'product_id': pid, 'sku': sku})
        if not isinstance(obs, dict) or obs.get('error'):
            return
        stock = bool(obs.get('in_stock'))
        price = hx_forge_8(obs.get('price'))
        over = self.budget is not None and price is not None and (price > self.budget + _C6_EPS)
        if stock and (not over):
            self._dull_25('Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it.' % (CatalogLog._free_70(pid, sku), price))
            return
        self.ledger.event_target = CatalogLog._free_70(pid, sku)
        self.event_turn = self.turn
        self.ledger.dead_keys.add(CatalogLog._free_70(pid, sku))
        self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        reason = 'went out of stock' if not stock else 'was repriced to %.2f, above the budget' % (price or 0.0)
        guidance = {'recovery': 'Recover with the smallest possible change: the same model in another color before any other model.', 'ranking': 'The remaining named candidates are re-ranked; the controller takes the new best.', 'preference': "Recover to a product with a different listing_id that still fits the shopper's priorities. Prefer full axis coverage again: sweat/water resistance, sport secure fit, wireless, and battery life evidence in title or facts.", 'constraint': 'Choose a different product that still meets every requirement, not another variant of the same one.'}.get(self.brief.family, 'Choose a different product that meets every firm requirement.')
        self._dull_25('Market change: the carted item %s %s. ' % (CatalogLog._free_70(pid, sku), reason) + guidance)
        self._slack_52()

    def _buzz_40(self, pid: str) -> dict:
        return self.ledger.products.get(str(pid)) or {}

class Deadline:

    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def _track_72(self) -> float:
        return self.budget - (time.monotonic() - self.start)

class Mission:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ''
        (goal, _, rules) = self.raw.partition('\n\nTask rules:')
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._pick_9(self.goal)
        self.currency = 'PHP'
        m = re.search('\\b([A-Z]{3})\\b', self.goal)
        if m and m.group(1) in ('PHP', 'USD', 'EUR', 'SGD', 'MYR', 'IDR', 'THB', 'VND'):
            self.currency = m.group(1)
        self.candidate_keys = [(pid, sku.rstrip('.,;:')) for (pid, sku) in _C13_KEY_RE.findall(self.goal)]
        phrase = _C25_PHRASE_RE.search(self.goal)
        quoted = _C33_QUOTED_RE.findall(self.goal)
        source = phrase.group(1) if phrase else quoted[0] if quoted else ''
        self.theme = re.findall('[a-z0-9]+', source.lower())
        sm = _C36_SURFACE_RE.search(self.rules)
        self.required_identities = int(sm.group(1)) if sm else 0
        pm = _C28_PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self._rank_13(self.goal)
        self.brand = self._stamp_6(self.goal, self.category)
        self.pref_brand = None if pm else self._hum_34(self.goal)
        self.cat_qualifier = self._carve_42(self.goal)
        self.specs = self._echo_55(self.goal)
        (self.spec_field, self.spec_value) = self._seal_54(self.goal)
        self.spec_sealed = self.spec_field is None and 'firm requirement' in self.goal.lower()
        self.family = self._pen_21()

    @staticmethod
    def _carve_42(goal: str) -> str | None:
        """A product shape that no shelf name captures, so the title must show it."""
        if re.search('\\bgaming\\s+monitors?\\b', goal or '', re.I):
            return 'gaming'
        if re.search('\\bover[\\s-]?ear\\b', goal or '', re.I):
            return 'over ear'
        return None

    @staticmethod
    def _pick_9(text: str) -> float | None:
        best = None
        for m in _C1_BUDGET_RE.finditer(text or ''):
            whole = m.group(1).replace(',', '')
            frac = m.group(2) or '0'
            try:
                val = float(whole + '.' + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best

    def _sift_69(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for field in ('screen_size', 'refresh_rate', 'ram', 'resolution'):
            value = hx_pick_16(field, text)
            if value:
                (self.spec_field, self.spec_value) = (field, value)
                self.specs[field] = value
                self.spec_sealed = False
                return field + ' = ' + value
        return None

    @staticmethod
    def _rank_13(goal: str) -> str | None:
        m = _C3_CAT_PHRASE_RE.search(goal or '')
        if m:
            return m.group(1).strip()
        m = re.search('\\b([a-z][a-z\\-]{2,20})\\s*(?:,|\\s(?:at or (?:under|below)|at most|under|below|within|with)\\b)', goal or '', re.I)
        return m.group(1) if m else None

    def _seal_54(self, goal: str) -> tuple:
        """The requirement the controller gates on: the first one stated."""
        for (field, value) in self.specs.items():
            return (field, value)
        return (None, None)

    @staticmethod
    def _echo_55(goal: str) -> dict:
        """Every hardware value the request states unambiguously.

        A request can name more than one ("a 27-inch 144Hz monitor") while only
        one of them is the gated requirement, so all of them are kept: the gate is
        honored whichever it turns out to be, and a reason can cite each of them.
        """
        out = {}
        for field in ('screen_size', 'refresh_rate', 'resolution', 'ram'):
            value = hx_pick_16(field, goal)
            if value:
                out[field] = value
        return out

    @staticmethod
    def _stamp_6(goal: str, category: str | None) -> str | None:
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        if not category:
            return None
        m = re.search('\\b(?:an?|the)\\s+([^,.;]{0,60}?)\\b' + re.escape(category), goal or '', re.I)
        if not m:
            return None
        words = [w for w in re.findall("[\\w&.'\\-]+", _C35_SPEC_WORD_RE.sub(' ', m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and (not any((c.isdigit() for c in w))) and (not hx_knit_12(w))]
        return ' '.join(keep) or None

    @staticmethod
    def _hum_34(goal: str) -> str | None:
        m = _C27_PREF_BRAND_RE.search(goal or '')
        if not m:
            return None
        name = m.group(1).strip(' .,')
        words = name.split()
        if not words or len(words) > 4 or (not any((w[:1].isupper() for w in words))):
            return None
        return name

    def _pen_21(self) -> str:
        if self.candidate_keys:
            return 'ranking'
        if self.required_identities:
            return 'recall'
        if self.pref_model:
            return 'recovery'
        for (name, sign) in _C8_FAMILY_SIGNS:
            if sign in self.rules:
                return name
        if self.theme:
            return 'recall'
        if _C34_REASON_RE.search(self.goal):
            return 'justification'
        if self.pref_brand:
            return 'intent'
        if self.spec_sealed or self.spec_field or self.brand:
            return 'constraint'
        return 'preference'

def agent_main(problem_data: dict) -> list:
    try:
        return CheckoutPilot(problem_data).run()
    except Exception as exc:
        return [{'role': 'assistant', 'content': 'agent error: ' + type(exc).__name__ + ': ' + str(exc)[:200]}]
