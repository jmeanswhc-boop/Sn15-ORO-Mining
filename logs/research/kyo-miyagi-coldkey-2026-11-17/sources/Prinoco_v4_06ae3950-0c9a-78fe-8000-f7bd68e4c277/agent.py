import copy
import json
import math
import re
import time
from src.agent.proxy_client import ProxyClient
HAIKU = 'anthropic/claude-haiku-5.5'
SONNET = 'anthropic/claude-sonnet-5.5'
LUNA = 'openai/gpt-6-luna'
ROLE_ACTOR = [HAIKU, LUNA, HAIKU]
ROLE_ACTOR_ROTATED = [LUNA, HAIKU, HAIKU]
ROLE_READER = [SONNET, LUNA]
ROLE_READER_SETUP = [SONNET, LUNA, HAIKU]
ROLE_CHECKER = [HAIKU, SONNET]
ROLE_AUDITOR = [HAIKU, SONNET]
ROLE_LISTENER = [HAIKU, SONNET]
ROLE_PLANNER = [HAIKU, SONNET]
ROLE_PLANNER_ONE = [HAIKU]
ROLE_HOP = [HAIKU]
CAP_ACTOR = 2500
CAP_READER = 3000
CAP_LISTENER = 2500
CAP_ITEM = 3000
CAP_RANK = 2500
CAP_MEANING = 600
CAP_AUDIT = 6000
CAP_PLANNER = 2500
CAP_PLANNER_ONE = 1500
CAP_HOP = 3000
SOFT_SPEND = 0.03
SPEND_CAP = 0.045
RACE_BUDGET = 4.8
RACE_TASKS = 90.0
EPISODE_CAP = RACE_BUDGET / RACE_TASKS
PLAN_SHARE = 0.15
REPLAN_SHARE = 0.5
REVIEW_SHARE = 0.8
DEADLINE = 265.0
CLOSING = 70.0
LAST_CALL = 28.0
CLOCK_STEP = 30
DOWN_AFTER = 45.0
WORLD_FLOOR = 2.0
LOOP_LIMIT = 80
IDLE_LIMIT = 3
STUCK_LIMIT = 4
RESERVE_LOW = 7
RESERVE_HIGH = 10
REFUSAL_RESERVE = 7
DEFAULT_ALLOWANCE = 3
CODE_FINISH_CALLS = 4
SCRIPTED_PAGES = 3
SCRIPTED_READS = 8
SYNONYM_MIN = 4
VIEW_LEN = 16000
SEARCH_FIELD_LEN = 240
RECORD_FIELD_LEN = 700
GAP = 0.15
LOAD_GAP = 1.0
LOAD_WINDOW = 30.0
MODEL_TRIES = 2
WORLD_TRIES = 5
CHUNK = 16
DECLINE_REMARK = 'Sorry, no listing in this shop satisfies every one of your requirements, so nothing will be purchased.'
JOINERS = [' and ', ' & ', '&', '/']
PUT_WORDS = {'add', 'insert', 'increase', 'put', 'place', 'append', 'include', 'push', 'plus', 'raise', 'increment', 'attach'}
TAKE_WORDS = {'remove', 'discard', 'subtract', 'delete', 'drop', 'decrease', 'release', 'minus', 'reduce', 'clear', 'take', 'decrement', 'detach'}
FAULT_STEMS = ['error', 'invalid', 'fail', 'reject', 'refus', 'denied', 'problem']
LINK = {'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'by', 'at', 'from', 'as', 'is', 'are', 'was', 'be', 'it', 'its', 'this', 'that', 'these', 'those', 'your', 'you', 'no', 'not'}
DECLINE_RX = re.compile('declin|give.?up|cancel|abstain|abandon|refus|no.?(?:purchase|order|buy)|withdraw|reject|pass on|nothing.?(?:fits|suit)')
ORDER_RX = re.compile('purchas|buy|check.?out|proceed|confirm|order|place|pay|complete|submit|finali[sz]')
ID_SHAPE = re.compile('^[A-Za-z0-9][A-Za-z0-9_\\-:.]{3,63}$')
CODE_SHAPE = re.compile('^[A-Za-z0-9_\\-]+$')
NEG_BOUND = re.compile('\\b(?:no|not)\\s+(?:more|over|exceeding|above|greater|higher|less|under|below)\\b')
LIMIT_RX = re.compile('\\b(?:at least|at most|up to|between|minimum|maximum|min|max|no more than|no less than|or more|or less|or fewer|under|over|below|above|less than|more than|greater than|within|not exceeding|or larger|or bigger)\\b')
COMPARISON_RX = re.compile('\\b(?:falls short of|falling short of|short of|nothing but|exactly|only|at least|nothing below|no less than|strictly|just|solely)\\b')
SKIP_WORDS = {'the', 'of', 'an', 'a', 'and'}
LITERAL_VALUES = set()
EXCLUSION_RX = re.compile('\\b(?:without|except|other than|not|no(?!-)|non|never|excluding|exclude|avoid\\w*|anything but|free of|rather than|instead of)\\b')
VALUE_LINKS_RX = re.compile('\\b(?:at most|at least|up to|either|plus|or more|or less|no more than|no less than|more than|less than|between|minimum|maximum|around|about|approximately|roughly|exactly|only|just|strictly|preferably|ideally|must be|should be|any|both)\\b')
REF_RX = re.compile('\\b(?:as shown|sold by|same as|shown on|redirect\\w*|refer(?:s|red|ring|ence)?|the listing|the page|points you|as listed)\\b')
DIFFER_RX = re.compile('\\b(?:unlike|distinct from|anything but|different from|differs? from|set apart from|apart from|not the same as|other than the)\\b')
SELLER_STEMS = ['seller', 'vendor', 'merchant', 'sold by', 'shop', 'store', 'brand', 'maker', 'manufactur', 'supplier', 'retailer']
YES_NO = {'yes', 'no', 'none', 'included', 'true', 'false', 'present', 'absent'}
EMPHASIS_RX = re.compile('\\b(?:strictly|exactly|only|solely)\\b')
CATALOGUE_WORDS = {'brand', 'department', 'availability', 'available', 'seller', 'vendor', 'merchant', 'shop', 'store', 'category', 'categories', 'heading', 'title', 'description', 'price', 'cost', 'money', 'currency', 'code', 'sku', 'address', 'url', 'link', 'image', 'rating', 'review', 'reviews', 'shelf', 'aisle'}
BARE_NAMES = {'name', 'notes', 'note', 'on hand', 'stock', 'media', 'maker', 'shelf', 'id', 'photo', 'images', 'in stock', 'item', 'product', 'listing'}
STORY_REF_CUES = re.compile('\\b(?:same as|sold by|as shown|shown on|the page)\\b')

def compact(value, ceiling=VIEW_LEN):
    try:
        text = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > ceiling:
        text = text[:ceiling] + '...[cut]'
    return text

def clip(value, length, depth=0):
    if depth > 8:
        return None
    if isinstance(value, str):
        return value if len(value) <= length else value[:length] + '...'
    if isinstance(value, dict):
        return {k: clip(v, length, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [clip(v, length, depth + 1) for v in value]
    return value

def read_obj(text):
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith('```'):
        t = re.sub('^```[A-Za-z0-9_\\-]*\\s*', '', t)
        t = re.sub('\\s*```\\s*$', '', t)
    tries = [t]
    if '{' in t and '}' in t:
        tries.append(t[t.find('{'):t.rfind('}') + 1])
    for cand in tries:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None

def is_num(v):
    return isinstance(v, (int, float)) and (not isinstance(v, bool))

def long_lines(value, out=None, depth=0):
    if out is None:
        out = []
    if depth > 4:
        return out
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:10]:
            long_lines(item, out, depth + 1)
    return out

def scan(value, visit, ceiling=6, depth=0):
    if depth > ceiling:
        return
    if isinstance(value, dict):
        visit(value)
        for v in value.values():
            scan(v, visit, ceiling, depth + 1)
    elif isinstance(value, list):
        for v in value:
            scan(v, visit, ceiling, depth + 1)

def dicts_in(value, ceiling=6):
    out = []
    scan(value, out.append, ceiling)
    return out

def id_like(v):
    return isinstance(v, str) and bool(ID_SHAPE.match(v))

def own_id(d):
    for k, v in d.items():
        if k.lower() in ('id', '_id') and isinstance(v, (str, int)) and (not isinstance(v, bool)):
            return str(v)
    for k, v in d.items():
        if k.lower().endswith('id') and id_like(v):
            return v
    return None

def row_identity(d):
    for k, v in d.items():
        if k.lower() == 'id' and isinstance(v, (str, int)) and (not isinstance(v, bool)):
            return str(v)
    for k, v in d.items():
        if k.lower().endswith('id') and isinstance(v, (str, int)) and (not isinstance(v, bool)):
            return str(v)
    try:
        return json.dumps(d, sort_keys=True)
    except Exception:
        return str(d)

def words_of(text):
    return re.findall('[a-z0-9]+', (text or '').lower())

def content_words(text):
    return [w for w in re.findall('[a-z]+', (text or '').lower()) if len(w) >= 3 and w not in LINK]

def split_camel(s):
    return re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', s or '')

def non_property(name):
    t = split_camel((name or '').split('(')[0]).lower()
    ws = re.findall('[a-z0-9]+', t)
    if not ws:
        return True
    if any((w in CATALOGUE_WORDS for w in ws)):
        return True
    return ' '.join(ws) in BARE_NAMES

def _depl(w):
    return w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w

def prop_words(name):
    t = split_camel((name or '').split('(')[0]).lower()
    return [_depl(w) for w in re.findall('[a-z]+', t)]

def reduced(name):
    return ''.join(prop_words(name))

def edit_le(a, b, limit):
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit

def loose_same(a, b):
    ra, rb = (reduced(a), reduced(b))
    if not ra or not rb:
        return False
    if ra == rb:
        return True
    s, l = (ra, rb) if len(ra) <= len(rb) else (rb, ra)
    if len(s) >= 4 and s[:3] == l[:3] and edit_le(s, l, 2 if len(s) >= 6 else 1):
        return True
    return len(s) >= 5 and l.startswith(s)

def _soft(w, i):
    return w[i] in 'aeiouy' or (i > 0 and w[i - 1] == w[i]) or (i + 1 < len(w) and w[i + 1] == w[i])

def spelling_variant(s, l):
    if len(s) == len(l):
        diff = [i for i in range(len(s)) if s[i] != l[i]]
        return len(diff) == 2 and diff[1] == diff[0] + 1 and (s[diff[0]] == l[diff[1]]) and (s[diff[1]] == l[diff[0]])
    gap = len(l) - len(s)
    if gap not in (1, 2):
        return False
    for i in range(len(l)):
        if not _soft(l, i):
            continue
        w = l[:i] + l[i + 1:]
        if gap == 1:
            if w == s:
                return True
            continue
        for j in range(len(w)):
            if _soft(w, j) and w[:j] + w[j + 1:] == s:
                return True
    return False

def strict_same(a, b):
    ra, rb = (reduced(a), reduced(b))
    if not ra or not rb:
        return False
    if ra == rb:
        return True
    s, l = (ra, rb) if len(ra) <= len(rb) else (rb, ra)
    if len(s) >= 4 and s[:3] == l[:3] and spelling_variant(s, l):
        return True
    wa, wb = (prop_words(a), prop_words(b))
    ws, wl = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    return len(wl) == len(ws) + 1 and wl[:len(ws)] == ws and (len(wl[-1]) <= 3) and (sum((len(w) for w in ws)) >= 5)

def kind_set(name):
    return {re.sub('s+$', '', w) for w in content_words(name)} - {''}

def kind_list(name):
    t = (name or '').lower().replace('’', "'").replace("'s", '')
    return [w for w in re.findall('[a-z]+', t) if len(w) >= 3 and w not in LINK]

def plural_forms(w):
    out = {w, w + 's', w + 'es'}
    if w.endswith('y'):
        out.add(w[:-1] + 'ies')
    if w.endswith('fe'):
        out.add(w[:-2] + 'ves')
    elif w.endswith('f'):
        out.add(w[:-1] + 'ves')
    if w.endswith('ies'):
        out.add(w[:-3] + 'y')
    if w.endswith('ves'):
        out.add(w[:-3] + 'f')
        out.add(w[:-3] + 'fe')
    if w.endswith('es'):
        out.add(w[:-2])
    if w.endswith('s'):
        out.add(w[:-1])
    return {f for f in out if len(f) >= 3}

def same_kind(a, b):
    return a == b or a in plural_forms(b) or b in plural_forms(a)

def kind_in(w, pool):
    return any((same_kind(w, p) for p in pool))

def all_kinds(ws, pool):
    return all((kind_in(w, pool) for w in ws))

def kind_score(kinds, to_match, shared):
    score = 0.0
    for w in kinds:
        if kind_in(w, to_match):
            score += 0.5 if kind_in(w, shared) else 1.0
    return score

def value_words(text):
    t = re.sub('[-/+]', ' ', (text or '').lower())
    return [w for w in re.findall('[a-z0-9]+(?:\\.[0-9]+)?', t) if w not in SKIP_WORDS]

def word_in(w, pool):
    if w in pool:
        return True
    for p in pool:
        if w.isalpha() and p.isalpha() and same_kind(w, p):
            return True
        if len(w) >= 6 and len(p) >= 6 and (w[:2] == p[:2]) and edit_le(w, p, 1):
            return True
    return False

def strip_marks(value):
    v = re.sub('\\((?:stated|revealed|changed)[^)]*\\)', ' ', value or '', flags=re.I)
    v = re.sub('\\b(?:stated|revealed)\\b', ' ', v, flags=re.I)
    return v.strip()

def alternatives(value):
    v = strip_marks(value).lower().replace('"', ' ')
    v = EMPHASIS_RX.sub(' ', v)
    out = []
    for piece in re.split('\\bor\\b', v):
        ws = value_words(piece)
        if ws:
            out.append(ws)
    return out

def alt_match(alts, pool):
    if not pool:
        return False
    for alt in alts:
        if all((word_in(w, pool) for w in alt)):
            return True
        if len(alt) >= 2 or any((c.isdigit() for c in ''.join(alt))) or any((any((c.isdigit() for c in p)) for p in pool)):
            glued = ''.join(alt)
            for i in range(len(pool)):
                run = ''
                for j in range(i, len(pool)):
                    run += pool[j]
                    if run == glued:
                        return True
                    if len(run) >= len(glued):
                        break
    return False

def spells(value, field_text):
    return alt_match(alternatives(value), value_words(field_text))

def touches(value, field_text):
    fw = value_words(field_text)
    ft = (field_text or '').lower()
    for alt in alternatives(value):
        for w in alt:
            if word_in(w, fw):
                return True
            if w.isdigit() and re.search('(?<![\\d.])' + re.escape(w) + '(?![\\d])', ft):
                return True
            if len(w) >= 3:
                for f in fw:
                    if len(f) >= 3 and (f.startswith(w[:3]) or w.startswith(f[:3])):
                        return True
    return False

def list_elements(text):
    t = (text or '').lower().strip()
    parts = [p.strip() for p in re.split('[,;|]', t) if p.strip()]
    return parts or ([t] if t else [])

def plus_sides(element):
    sides = [s.strip() for s in element.split('+') if s.strip()]
    return sides if len(sides) >= 2 else []

def figure_written(fig, req):
    for m in re.finditer('(?<![\\d.])' + re.escape(fig) + '(?!\\d|\\.\\d)', req):
        if '.' in fig:
            return True
        rest = req[m.end():].lstrip()
        nxt = re.match('[a-z]+', rest)
        if not nxt or len(nxt.group(0)) <= 3:
            return True
    return False

def positive_value(value, prop=''):
    try:
        v = strip_marks(value).lower().strip()
        bare = re.sub('[^a-z0-9 ]', ' ', v).strip()
        if bare in YES_NO:
            return True
        if bare and (bare in LITERAL_VALUES or v in LITERAL_VALUES):
            return True
        t = NEG_BOUND.sub(' ', v)
        t = re.sub('\\b(?:no|not)\\s+\\w+er\\s+than\\b', ' ', t)
        if re.search('\\d', t):
            t = re.sub('\\b(?:no|not)\\s+\\w+er\\b', ' ', t)
        t2 = VALUE_LINKS_RX.sub(' ', t)
        for piece in re.split('[,;/&()\\[\\]]|\\bor\\b|\\band\\b', t2):
            if len(content_words(piece)) >= 4:
                return False
        m = EXCLUSION_RX.search(t)
        if not m:
            return True
        if content_words(t[:m.start()]):
            return True
        after = content_words(t[m.end():])
        if not after:
            return False
        pw = set(prop_words(prop)) | set(words_of(prop))
        return all((w in pw or _depl(w) in pw for w in after))
    except Exception:
        return True

def value_seen(value, samples):
    if not samples:
        return False
    s = ' '.join(samples).lower() if isinstance(samples, (list, tuple, set)) else str(samples).lower()
    v = VALUE_LINKS_RX.sub(' ', strip_marks(value).lower())
    for w in content_words(v):
        stem = w[:-1] if len(w) > 4 else w
        if re.search('\\b' + re.escape(stem), s):
            return True
    for run in re.findall('\\d+(?:\\.\\d+)?(?:\\s*[:x/]\\s*\\d+(?:\\.\\d+)?)+', v):
        figs = re.findall('\\d+(?:\\.\\d+)?', run)
        pat = '(?<![\\d.])' + '\\D{0,4}'.join((re.escape(f) for f in figs)) + '(?![\\d.])'
        if re.search(pat, s):
            return True
    return False

def single_prop_hint(hint):
    try:
        if ':' not in hint:
            return hint
        head, detail = hint.split(':', 1)
        tag = ''
        m = re.search('\\(([^)]*)\\)\\s*$', head)
        if m:
            tag = m.group(0)
            head = head[:m.start()]
        names = [n.strip() for n in head.split('/') if n.strip()]
        if len(names) < 2:
            return hint
        return '%s%s:%s (or %s)' % (names[0], ' ' + tag if tag else '', detail, ', '.join(names[1:]))
    except Exception:
        return hint

def lead_value(detail, request, product):
    m = re.match('\\s*([A-Za-z]+)(?=\\s*(?:[,;.()\\[\\]]|$|\\s+(\\w+)))', detail or '')
    if not m:
        return ''
    word = m.group(1).lower()
    follow = (m.group(2) or '').lower()
    rest = (detail or '')[m.end():].lstrip()
    if rest and (not re.match('[,;.()\\[\\]]', rest)) and (follow not in LINK):
        return ''
    if len(word) < 3 or word in LINK:
        return ''
    kinds = kind_list(product)
    if not kinds or any((same_kind(word, k) for k in kinds)):
        return ''
    forms = sorted(plural_forms(kinds[0]), key=len, reverse=True)
    req = (request or '').lower()
    pat = '\\b' + re.escape(word) + '\\s+(?:' + '|'.join((re.escape(f) for f in forms)) + ")(?:'s)?\\b"
    return word if re.search(pat, req) else ''

def story_guess(req_line, request, refs=()):
    try:
        if ':' not in req_line:
            return []
        prop, value = req_line.split(':', 1)
        value = strip_marks(value).strip()
        if not value or req_line in refs or is_reference(value) or STORY_REF_CUES.search(value.lower()):
            return []
        low = value.lower()
        if 'refer' in low or 'listing' in low:
            return []
        if re.sub('[^a-z]', '', low) in YES_NO - {'none'}:
            value = prop.split('(')[0]
            low = value.lower()
        low = VALUE_LINKS_RX.sub(' ', low)
        req = (request or '').lower()
        figs = re.findall('\\d+(?:\\.\\d+)?', low)
        if figs and all((figure_written(f, req) for f in figs)):
            return []
        req_words = re.findall('[a-z]+', req)
        absent = []
        present_count = 0
        cws = [w for w in content_words(low) if not COMPARISON_RX.fullmatch(w)]
        for w in cws:
            stem = w[:-2] if len(w) > 4 and w.endswith('es') else w[:-1] if len(w) > 3 and w.endswith('s') else w
            found = False
            for r in req_words:
                if r == w or (r.startswith(stem) and len(r) - len(stem) <= 3) or (stem.startswith(r) and len(r) >= max(4, len(stem) - 2)) or (len(w) >= 4 and r.startswith(w)) or (len(r) >= 4 and len(w) >= 4 and (r[0] == w[0]) and edit_le(r, w, 1)):
                    found = True
                    break
                rs = r[:-1] if r.endswith('s') else r
                if len(rs) >= 3 and stem.startswith(rs) and (len(stem) - len(rs) <= 3):
                    found = True
                    break
            if found:
                present_count += 1
            else:
                absent.append(w)
        if len(cws) <= 2 and any((len(w) >= 4 for w in cws if w not in absent)) and present_count:
            return []
        for run in re.findall('\\d+(?:\\.\\d+)?(?:\\s*[:x/]\\s*\\d+(?:\\.\\d+)?)+', low):
            fs = re.findall('\\d+(?:\\.\\d+)?', run)
            pat = '(?<![\\d.])' + '\\D{0,4}'.join((re.escape(f) for f in fs)) + '(?![\\d.])'
            if not re.search(pat, req):
                absent.append(run)
        return absent
    except Exception:
        return []

def is_reference(value):
    return bool(REF_RX.search((value or '').lower()))

def ref_relation(value):
    v = (value or '').lower()
    if DIFFER_RX.search(v):
        return ('differ', None, '')
    m = re.search('\\bwithin\\s+(\\d+(?:\\.\\d+)?)\\s*([a-z%]*)', v)
    if m:
        return ('within', float(m.group(1)), m.group(2))
    if re.search('\\b(?:up to|no more than|at most|not exceeding|or less|no larger than)\\b', v):
        return ('at most', None, '')
    if re.search('\\b(?:at least|or more|or larger|no less than|no smaller than)\\b', v):
        return ('at least', None, '')
    return ('same', None, '')
_PRICE_WORDS = ('amount', 'fee', 'charge', 'rate', 'value', 'usd')
_PRICE_BAD = ('total', 'rating', 'weight', 'count', 'score', 'qty', 'quantity', 'size', 'length', 'width', 'height', 'depth', 'capacity', 'volume', 'stars', 'review', 'page', 'limit', 'number', 'year', 'watt', 'age', 'percent')
_STOCK_WORDS = ('hand', 'orderable', 'ship', 'buy', 'purchas', 'sellable', 'ready', 'instock', 'in_stock')
_TITLE_WORDS = ('headline', 'caption', 'label', 'name')
_TITLE_BAD = ('seller', 'brand', 'category', 'shelf', 'vendor', 'shop', 'store', 'maker', 'id', 'file')

def lens(obs):
    try:
        out = json.loads(json.dumps(obs))
    except Exception:
        return obs

    def fix(d):
        if not any(('id' in k.lower() and id_like(v) for k, v in d.items())):
            return
        keys = list(d.keys())
        if not any((('price' in k.lower() or 'cost' in k.lower()) and is_num(d[k]) for k in keys)):
            c = [k for k in keys if is_num(d[k]) and any((w in k.lower() for w in _PRICE_WORDS)) and (not any((b in k.lower() for b in _PRICE_BAD)))]
            if len(c) == 1:
                d['price'] = d[c[0]]
        if not any(('stock' in k.lower() or 'avail' in k.lower() for k in keys)):
            c = [k for k in keys if isinstance(d[k], bool) and any((w in k.lower() for w in _STOCK_WORDS))]
            if len(c) == 1:
                d['in_stock'] = d[c[0]]
        if not any(('title' in k.lower() or 'heading' in k.lower() for k in keys)):
            c = [k for k in keys if isinstance(d[k], str) and any((w in k.lower() for w in _TITLE_WORDS)) and (not any((b in k.lower() for b in _TITLE_BAD)))]
            if len(c) == 1:
                d['title'] = d[c[0]]
    scan(out, fix)
    return out
_STOCK_NEG = re.compile('\\b(?:out|sold|unavailable|no|not|none|false)\\b')

def field_stock(key, value):
    k = (key or '').lower()
    if value is None:
        return None
    has_sa = 'stock' in k or 'avail' in k
    has_hand = bool(re.search('on.?hand', k))
    if not has_sa and (not has_hand):
        return None
    if isinstance(value, str) and re.search('back.?order|pre.?order|discontinu', value.lower()):
        return False
    if has_sa:
        if 'out' in k or 'sold' in k:
            return not value is True
        if value is False or (is_num(value) and value <= 0):
            return False
        if isinstance(value, str) and _STOCK_NEG.search(value.lower()):
            return False
        return True
    if value is False or (is_num(value) and value <= 0):
        return False
    return True

def record_stock(d):
    verdicts = [field_stock(k, v) for k, v in d.items() if not isinstance(v, (dict, list))]
    verdicts = [v for v in verdicts if v is not None]
    if not verdicts:
        return None
    return all(verdicts)

def record_label(d):
    for k, v in d.items():
        kl = k.lower()
        if 'id' in kl:
            continue
        if any((f in kl for f in ('categor', 'shelf', 'aisle', 'department'))) and isinstance(v, str):
            t = v.strip()
            if 0 < len(t) <= 60:
                return t.lower()
    return ''

def record_price(d):
    for k, v in d.items():
        kl = k.lower()
        if ('price' in kl or 'cost' in kl) and is_num(v):
            return v
    return None

def record_title(d):
    if isinstance(d.get('title'), str):
        return d['title']
    for k, v in d.items():
        kl = k.lower()
        if isinstance(v, str) and any((w in kl for w in ('title', 'heading', 'headline'))):
            return v
    for k, v in d.items():
        kl = k.lower()
        if isinstance(v, str) and 'name' in kl and ('id' not in kl) and (not any((b in kl for b in _TITLE_BAD))):
            return v
    return ''

def prop_map(d):
    for v in d.values():
        if isinstance(v, dict):
            scal = [x for x in v.values() if isinstance(x, (str, int, float, bool)) and x is not None]
            if len(scal) >= 2:
                return v
    return None

def money(v):
    if not is_num(v):
        return str(v)
    return str(int(v)) if float(v).is_integer() else ('%.2f' % v).rstrip('0').rstrip('.')
PROMPT_ACTION = 'You are a careful buying assistant acting for one shopper inside a simulated shop. The tools you are given are the only way you can act.\n\nSCORING. The result is all or nothing. You earn the point only with a fully correct finish: the item bought meets every requirement the request writes down, every hidden need the shopper discloses when asked, and every change made during the task; the quantity is right; the budget is respected; when the request asks for the lowest price (or the best by some measure) it is exactly that item; and every procedure the request lays down was followed. If nothing in the shop satisfies everything, the correct finish is a decline carried out exactly the way the request prescribes.\n\nHOW THE SHOP BEHAVES\n1. Every tool call costs one step, including calls that fail or are invalid. Steps are scarce; thinking is free.\n2. You may put several calls in one reply when none depends on another\'s result (for example opening the details of several listings); they run one after another. An order or a decline must always be sent alone.\n3. When a details tool accepts a list of ids, one step returns all of those records (up to that tool\'s own limit). Prefer it whenever more than one listing needs opening.\n4. Each shop names its tools, arguments and result fields differently. Read every tool description and schema, copy names and allowed values exactly, and learn from the first results which tool shows price, which shows stock and which shows the detailed properties.\n5. Early on, find out which category label this shop gives to listings of the wanted kind. Among neighbouring labels the right one matches the whole phrase of the request; search and buy inside that label.\n6. Search behaviour differs from shop to shop; learn it from the first results:\n   a. results may be ranked by relevance or listed in a fixed id order;\n   b. several words may all be required (adding a word lowers the count) or any one may be enough (an extra word raises the count, and more matched words rank higher);\n   c. you may only see a few rows or pages;\n   d. filters may work in one mode only (a keyword mode that ignores category and price filters, or a filter mode that ignores words);\n   e. a reported count larger than the visible rows means the fit, or the cheapest, may sit on later pages or beyond what is shown;\n   f. in all-words shops, narrow with more precise words (or category plus price filters with a lower price ceiling) until the count fits in the pages you can read;\n   g. keep queries to 1-3 words with the most distinctive word first; some shops read only the first word or two and silently drop the rest, so a long query may in effect search only its start (test it: compare a shorter query with the same start against a longer one, by counts and rows; code notes report the result);\n   h. in all-words shops, a required value followed by the kind noun (a colour plus a furniture noun, say) returns exactly the listings carrying that value;\n   i. in any-word shops, generic words (the kind noun, a unit) hit almost the whole shelf and drown everything else; a rare word or figure from a required value, searched alone (a weave-pattern name, a depth rating with its unit attached, a decimal figure), returns just the listings carrying it, so search it alone and also in other spellings listings might use (the unit split off by a space, the same rating in another unit); when that count fits in the readable pages, read them all; add the kind noun only to rank the closest matches first;\n   j. keyword hits can also come from category names or property values;\n   k. after each search, code notes say what the counts so far reveal about this shop\'s search.\n\nTHE SHOPPER AND HIDDEN NEEDS\n7. The shopper keeps some needs unwritten and reveals one only to a message that asks about exactly that property. The first question in a message is answered most reliably, so ask about one property per message (two at most, the likelier first), short, plain and friendly, naming the property and optionally offering one or two example values. Put nothing else in a message: no instructions, lists, labels, positions or rules.\n8. A "don\'t mind" answer (no preference, anything is fine) frees that property. Any preference or limit the shopper states binds from then on, exactly as worded, numbers included.\n9. In this agent the code plans and sends the questions. They go out after you have opened the details of 2-3 listings of the wanted kind, so make such openings among your first moves. Answers come back in the conversation with their meaning attached; treat every disclosed need as binding and search again using it together with the stated values.\n10. If you ever ask by yourself: first open the details of 2-3 listings; then ask about what the story hints at (who it is for, the look wanted, where or how it will be used); then about properties the listings carry that vary between candidates. Never ask about anything the request already settles and never exceed the message allowance.\n\nPROCEDURE RULES\n11. A request may set working rules: how many listings to view or compare, checks to make, what to tell the shopper, what a remark must contain, how to decline. They are graded like requirements; follow each literally and completely. An instruction to view (or open) N listings means N complete fits, each opened through the details tool.\n12. When the shopper asks you something (to confirm a change, say), the code answers while the message allowance lasts; the change binds either way.\n13. A mere liking (marked by "ideally", "if possible" and similar) is not a requirement: it only separates items that already satisfy everything and never outranks a cheapest rule.\n14. If the request leaves open a point only the shopper can decide (no price ceiling given; two wishes that may not both be achievable, so the shopper must rank them), ask before choosing.\n\nCHANGES DURING THE TASK\n15. The shopper may change things: a new budget (higher or lower), a requirement removed, replaced or added, a price rise, an item going out of stock. One line may hold several. They can appear inside any tool result (as an extra field of sentences) or come with an order attempt. They bind from the moment they arrive: re-check your pick and your backups, and under cheapest or best choose again across the changed set of acceptable items.\n16. When an order is refused the episode continues and the cart stays as it was. Read the refusal, apply every change it states, remove whatever no longer fits (a sold-out line, say), put in the next verified fit, refresh that one\'s stock and price, and order again.\n17. Items sell out and prices move silently, so read stock and price afresh before every order attempt. Once current price times quantity exceeds the current budget, that item no longer fits.\n\nSEVERAL PRODUCTS\n18. A request may want several different products, each with its own quantity, requirements and hidden needs. Handle them all: for each product search its category and verify a fit; buy them together, one cart line per product at that product\'s quantity. A cart tool that takes a list may replace the whole cart with that list, so every such call must include every product\'s line. Cheapest or best applies to each product separately.\n19. A requirement\'s value may be defined by another listing or page (for example the same colour as a garment a named seller offers). Find that listing (search with words that identify it, such as the shop or brand plus its kind), open its record, copy the value from the named field, and only then search for the wanted product with that value.\n\nCART\n20. Some cart tools add units to a line already held instead of setting them: putting one more unit of a held line gives two. Never put a held line in again; to change units, remove it or add only the difference. The code reports its count of the cart after every cart call.\n\nJUDGING A LISTING (this decides the score)\n21. The deciding field: the grader takes each requirement from the field for that property in the listing and compares the wording as this shop spells it there. Variants of the name still count as that property\'s field: different separators, a prefix, a shortened or longer name, a common alias (a trouser fit type under "fit", a collar type under "neckline", a watch\'s dial measurement under its dial-size key, a colour under "colour family"). Precedence: a field carrying exactly the property\'s name decides, and so does that name with one qualifier word in front (length under "trouser length", material under "clothing material"); the value in either one satisfies it. Names that differ only by separators (space or underscore) are one field; the value in either satisfies it. Without those, an alias field decides. Only when no field bears the property\'s name may the value appear in an options list or in the title instead. Descriptions are never evidence.\n22. Matching values: compare words in this shop\'s spelling; case, separators, singular or plural and spelling variants are all equal. A more specific value that contains the required one satisfies it (a fibre blend with a percentage of that fibre; a qualified green for green; a graphics-card model with its memory size for the bare model). In a list, one element is enough. A synonym, abbreviation, related word or shade (a shortened compound for the full form, a compact operating-system name for the spelled-out one, a craft adjective for the craft noun, a no-pattern word for plain, two pale shades for off-white) satisfies it only if no listing on that shelf spells that exact value in this field: if other listings write matte as their finish, satin fails matte. A plus-joined value (fibre + fibre) fails either part whenever the same field also carries the partner part on its own elsewhere. Field-evidence lines, when given, are computed by code over the shop\'s records and settle which case holds. Another value of the same property fails (quartz movement against mechanical; one wood species against another), and so do look-alike words. Values claiming to cover everything (every occasion, all seasons, universal) name nothing and fail, but a list naming the required value among others passes. Association or implication is no match (zip-up is not met by pullover, lace-up not by slip-on); beyond exact words only two extras count: a shade normally sold as that colour, or a narrower value containing the required one.\n23. Neighbouring properties never stand in (lining material for material, length for style, strap length for strap type): such a field never replaces a property for which this shelf\'s listings carry a dedicated field.\n24. Only stated or disclosed wishes count. Never reject over matters nobody asked about (age group, gender, occasion, season, style, use, recipient) guessed from the story or a description. The story is context; if something there seems to matter, ask the shopper rather than assume. Your own questions create no requirements; the shopper\'s answers do.\n25. Numbers: compare amounts after converting units (4.5 centimetres equals 45 millimetres); limits are inclusive; a range counts only when it lies wholly inside the limit. When the title is the evidence (no field names the property), a precise required value passes when the title contains its words and numbers in order, trailing detail allowed (a rug size in feet is still shown when extra inches follow its second figure, but not when that second figure differs).\n26. Figures in a title (two dimension pairs in inches and centimetres, say) do not tell which dimension each figure is: never reject or leave a listing unopened because of title figures; read the detail fields instead.\n27. Honour exclusions (drop anything with some feature) and closed lists (one of these and nothing else).\n28. The kind comes from the category alone: being on the shelf makes a listing that kind for grading, whatever other object its title or description names (a container, a refill, an accessory, a set, a case). Do not skip, reject or down-rank it for appearances; judge it on requirements, price and stock only. No property has to repeat the category\'s name. With labelled categories the item belongs under the label matching the whole phrase of the request (a request for "X or Y" belongs on a joint X-and-Y label, not on a narrower neighbouring Y label). Stock is required when on-hand goods are demanded. Price times quantity must not exceed the budget (equal passes). The price unit is the budget\'s unit unless the record names another (a price of 45 against a budget in US cents means 45 cents); rescale neither.\n29. Declining is rarely right. If no listing matches the strict wording but one passes every requirement under these rules (a more specific value, an alias field, a synonym or shade with no shelf listing spelling the exact value in this field), that listing fits; a decline is justified only after searching genuinely found nothing that passes every requirement under these rules.\n\nCHEAPEST OR BEST\n30. Cheapest: buy the lowest-priced item among all fits; finding one fit proves nothing. Method:\n   a. an all-words search that also matches property values: a value-plus-kind query lists the listings bearing that value; go through every page and open them from the lowest price up;\n   b. a category browse under a price ceiling shows prices across the shelf without values: use it to spot which cheaper listings exist (keep lowering the ceiling until the count fits), then inspect their properties;\n   c. for each required value, a short query with the value first and the kind noun; read all pages if the count fits the readable pages (otherwise add another required value or a narrower kind noun);\n   d. repeat with other kind nouns (two words for a seat, say) and with the value alone;\n   e. for a value the shop does not index (a figure it strips), also search the product line behind every fit and near miss with two words, the maker plus one distinctive title word, all pages: sibling sizes and colours of a line are separate listings with the same property values and sometimes lower prices;\n   f. rank the visible candidates by price yourself and open the lowest first; the winner is the first whose property fields pass everything once every cheaper one was opened and rejected; keep the later fits as backups.\n31. Best by a measure (largest, longest and similar): compare all fits on that measure the same way and pick the extreme one; an item stating nothing about the measure is out.\n\nFINISHING\n32. Keep your verified fits ranked with at least one backup: the top pick may be gone at checkout.\n33. Before each order attempt: the cart holds exactly one line per wanted product (a single line for a single product), each the chosen listing at its required number of units, nothing else; the chosen listing\'s stock and price were read after the latest cart change; it is in stock and affordable.\n34. Decline only once a thorough search shows nothing passes every requirement, and follow literally whatever decline procedure was prescribed. Where substitutes are forbidden, the decline text mentions no listing in any way: no ids, no titles, no prices.\n35. Plan your steps: make the first order attempt with 7 or more steps still unused so that a refusal can be repaired. From 7 steps left, the tools shrink to the cart, stock reading and finishing. Decide once; do not flip the cart back and forth.\n36. Every reply must call a tool. Keep your text short: the current requirements, the verified fits (price, stock) and your next move.'
PROMPT_READ = 'You will receive a shopper\'s request and the shop\'s tool list. Answer with a single compact JSON object (no indentation, nothing else) holding these fields:\n\n"product": the kind of item wanted, in the request\'s words; several products joined with "and".\n"items": a list with one entry per distinct product to buy, in request order; each entry is {"product": kind in the request\'s words, "quantity": as in "quantity", "requirements": firm requirements as in "requirements"}; a single-product request gets one entry.\n"quantity": whole number of units of one listing to buy; 1 unless several are explicitly wanted. Anything sold as one unit (a pair of shoes, a set) counts as 1. Change it only for a plain instruction to buy several units (wanting three, ordering two packs). How many people use or share it, or how many old items it replaces, is story and stays 1.\n"requirements": list of strings, each a firm requirement written "property: constraint", holding exactly the value, limit, exclusion or closed list the shopper gave; with several products put the product in brackets after the property, "property (product): constraint". Leave out budgets, price, stock, quantity and simple wishes; never repeat the kind or category; words that are part of the category name (a season, occasion or audience inside that name) belong to the category, not to a requirement. A requirement is a product property plus the value the request demands (a weave pattern that must be a named one; a mechanical movement; at least four drawers; an occasion that must be party). The story never gives a requirement: the recipient, the event of wear or use (a wedding, a staff party), the place, time or manner of use, how many share it, replacing an old one (asking for a new watch is a replacement, not a requirement of newness). Occasion, use or audience counts only when the request states it as the item\'s own attribute (an occasion that must be business; a jacket made for business). The value must be words or figures literally written in the request as this item\'s value. Qualities, reasons or wishes taken from the story (natural and breathable; thin enough to fit a case; compatible with owned gear; long enough for the full drop) are hints only.\n"price_cap": the most the shopper will spend in total, the figure exactly as written in the request\'s unit with no conversion (a cap in US cents keeps the cent figure); null if absent.\n"currency": the request\'s price unit, or "".\n"in_stock_only": true if the request accepts only items available now.\n"priority": "cheapest" only for a plain demand for the lowest-priced fit; "best" only for a plain demand to maximise some measure; otherwise "none". Remarks about tight money or saving do not make a cheapest demand; they only explain the cap.\n"priority_detail": the measure for "best", else "".\n"message_budget": whole number of messages the shopper allows, or null if not stated.\n"step_budget": whole number limit on moves or steps, or null if not stated.\n"rules": every working rule imposed on the assistant (how many listings to view, checks or reports to make, how to decline, prohibitions, things that must be included), copied word for word from the request. Actions the shopper will take personally are story.\n"soft_wants": likings without strict need (ideally, preferably and similar), kept apart from requirements.\n"ask_first": points that need the shopper\'s own answer before any choice: a budget when the request gives none; how to rank two wishes the request admits may conflict (both wishes then go to soft_wants, not requirements); anything the request says must be confirmed by the shopper. Otherwise [].\n"hints": story details aimed at a product property without giving its value (purpose, companion items, place, user, things to fit, match or withstand), each written "property (product): brief detail" (no bracket with a single product), the property worded the way listings probably word it. When a detail bears on several properties, put the likeliest first and the runner-up after a slash (a chair meant to blend with a room whose one wall is painted gives "colour / style"). Matching or suiting surroundings (a room, a wall, an outfit, companion items) and visual impression point first to colour, then to pattern or style. Skip details that only justify a stated requirement (the reason behind a stated size). Never hints: weather, job, schedule, travel, places where the item is not worn or used, and purposes every listing of the kind shares (a dresser stores things, a coat keeps warm). [] if none.\nCheck before you answer: every sentence of the request must land somewhere (requirement, priority, soft want, rule, budget, hint or plain story); a sentence that may or may not be a rule goes to rules.\n"references": requirements whose value is given only through another listing or page (a colour matching a garment some named seller offers; within some amount of a figure printed on another seller\'s page; any colour except the one on a third seller\'s page), each written "property (product): <phrase saying the value is as shown on that listing, named the way the request names it>" or, when a difference is demanded, a phrase saying it must differ from what that listing shows; each also appears under requirements in the same form. [] if none.\n"reference_targets": one object per reference describing the listing pointed at: {"property", "product", "seller": store or brand name verbatim, "kind": that listing\'s product kind, "details": further values the request gives for that listing (brief, request\'s words, e.g. a colour, a memory size, a net weight with many decimals), "relation": one of "same", "differ", "within", "at most", "at least", "tolerance": number for within else null, "unit": unit or ""}. [] if none.\n"decline_without_names": true if the request\'s decline rule forbids proposing alternatives in any form (offering, naming, suggesting); false if it wants something named; null if there is no decline rule.\n"roles": an object giving, for each of these eight roles, the exact tool name that serves it, or "" when no tool does: "search", "details", "view", "message" (writing to the shopper), "cart" (adding or removing cart lines), "cart_view", "order" (purchase or checkout), "decline" (finishing without a purchase).'
PROMPT_HEAR = 'You help a shopping assistant understand what the shopper has just said. Reply with one JSON object with exactly these ten fields:\n"meaning": one short sentence on what the shopper means.\n"needs": a JSON list of strings, the newly firm requirements, each "property: constraint", holding exactly what was given (a value, a limit, an exclusion or a closed list). After a comparison phrase (falling short of something, nothing below something, a floor), a value made of words rather than figures is itself the value, spelled as the shopper spells it: a line saying something falls short of the words "not available" gives that property the value "not available", because such words (and "none") can be real field values.\n"no_preference": a JSON list of the property names the shopper said they do not care about.\n"dropped": a JSON list of the earlier requirements that are no longer needed.\n"new_budget": a fresh maximum total price as a number, in the unit the original budget used (take the figure as written when the line uses that unit, e.g. cents for a budget in cents); otherwise null.\n"unavailable": true if the shopper reports that some item can no longer be bought (sold out, gone, not in stock, buy button greyed out); otherwise false.\n"meta_only": true only if the line just asks for fewer things per message (one question at a time, less at once, first this then the rest) and gives no value; such a line answers nothing yet, adds nothing to needs, no_preference or dropped, and its figurative words name no value. Lines saying only that the shopper did not follow or understand, or asking for a shorter or simpler rewording, are not meta_only.\n"requirements_now": a JSON list of strings, the full current requirement list, each "property: constraint": start from the requirements-so-far block, keep each one unless dropped or replaced (for a replaced one keep only the newest value), and add every need stated in the shopper\'s lines. Nothing may come in from the request\'s background story unless that block already holds it; additions come only from the shopper\'s lines. Leave out price, stock, quantity and product category.\n"reply": if the shopper asks the assistant a question or wants a confirmation, a short plain answer to send back that names the substance of the change (for example confirming its new value); otherwise "".\n"references": for each need whose value is given only through another listing or page, an object describing that listing: {"property", "seller" (as the line writes it), "kind", "details" (a list of other values the line gives for that listing, brief, the line\'s words), "relation" (one of "same", "differ" meaning the value must not equal that listing\'s, "within", "at most", "at least"), "tolerance" (number for within, else null), "unit" (or "")}; [] when none.\n\nRules:\n- One line may hold several changes; report all of them.\n- A need about a different property, related or not (seat height next to overall height; strap width next to plain width), adds a requirement and replaces none: an older requirement is dropped or replaced only when the line itself says so for the identical property.\n- A replacement is two entries: the old one under dropped and the new one under needs.\n- When the shopper ranks two wishes, the preferred one becomes a need.\n- Any budget stated goes under new_budget.\n- Mere wishes (ideally, preferably and similar) are not needs.\n- Read literally but kindly: homely, figurative or indirect wording that names a value is a firm wish for it; excluding a value needs the shopper\'s clear instruction to avoid it.\n- Words that only stress how firmly a value is held (strictly, only, exclusively, sticking with, nothing but, solely, purely, just) are not part of the value: write the value alone (material: jute; colour: white), never the value plus a trailing "only"; add an exclusion only when the shopper also explicitly rules out mixtures or other values.\n- A number with a comparison (ceiling, floor, below, above, not exceeding) is a limit; keep its unit.\n- A new budget replaces the previous figure, up or down, whether or not the shopper frames it as good news.\n- Pure indifference (does not mind, no preference, anything works): the property the assistant asked about goes under no_preference; for a message asking about two properties, only its first.\n- A reply saying only that the question went over the shopper\'s head (too much at once, lost, please rephrase) is this shopper\'s way of saying they have no preference, provided the question named which product owns the property: then the asked property also goes under no_preference; but with several products wanted and no product named in the question, the property stays open (record nothing).\n- With several products, write every need "property (product): constraint", taking the product from the line (a line about the toe style of some named women\'s boots gives that property tagged with those boots).\n- Questions written by the assistant never create requirements; only the shopper\'s statements do; example values inside a question never become needs.\n- Indifference frees only the property the assistant\'s message asked about; it never removes or changes an earlier stated need; if that message asked no question (it only informed), indifference changes nothing.\n- A pointing need keeps as its value the pointing phrase exactly as the line words it (a finish described as unlike the one on another seller\'s kettle page); values mentioned about the pointed-at listing describe that listing, not the wanted product, so they create no needs, drops or freed properties.'
PROMPT_READING_RULES = 'READING RULES (how the grader reads a listing)\n1. Deciding field: a requirement is judged from the listing\'s field for that property, in the shop\'s own wording. Name variants still count as that field: different separators, a prefix, a shortened or longer name, a common alias (a trouser fit type under "fit", a collar type under "neckline", a colour under "colour family"). A field with exactly the property\'s name, or that name with one qualifier word in front, decides; the value in either satisfies it. Separator-only variants are one field. Without those, an alias field decides. Only when no field names the property may the options list or the title supply the value. A description never counts.\n2. Value matching: words are compared in the shop\'s spelling; case, separators, singular or plural and spelling variants are equal. A more specific value containing the required one passes (a fibre blend with that fibre\'s percentage; a qualified green for green; a graphics-card model plus memory for the bare model). One element of a list is enough. A synonym, abbreviation, related word or shade (a shortened compound, a compact operating-system name, a craft adjective for the craft noun, a no-pattern word for plain, two pale shades for off-white) passes only where no listing on the shelf spells that exact value in this field: if others write matte, satin fails matte. A plus-joined value fails a part whose partner the same field carries alone elsewhere. Field-evidence lines from code show which case holds. Another value of the property, or a look-alike word, fails. Values claiming to cover everything (every occasion, all seasons, universal) name nothing and fail unless the list also contains the required value. Association or implication fails (zip-up is not pullover, lace-up is not slip-on); beyond exact words only a shade normally sold as that colour, or a narrower value containing the required one, counts.\n3. Neighbouring property fields (lining material for material, length for style, strap length for strap type) never count, and never stand in where the shelf\'s listings carry a dedicated field for the required property.\n4. Numbers: convert units before comparing (4.5 cm = 45 mm); limits are inclusive; a range counts only when it lies wholly inside the limit. A precise value shown in a title passes when the title has its words and numbers in order, trailing detail allowed (a rug size in feet still matches when inches follow its second figure, not when that figure differs).\n5. Prices are in the budget\'s unit unless a record names another currency or unit (a price of 45 against a budget in US cents means 45 cents, comfortably inside that budget); rescale neither.\n6. Only listed requirements count: never fail an item over matters nobody listed (age group, gender, occasion, season, style, use, recipient, theme).\n7. The category label alone settles the kind, even when a title names another object (a container, a refill, an accessory, a set, a case); judge on requirements, price and stock only.'
PROMPT_ITEM = 'You are the final gate before a shopping assistant commits to buying an item. The task scores 1 only if the item passes every listed requirement. Judge only from the records shown, by these rules.\n\n%s\n8. Words that are part of the shelf name are satisfied by the shelf itself.\n9. Respect exclusions and closed lists.\n10. The item must be of the wanted kind; with labelled categories its label must match the whole phrase of the request, never a narrower or neighbouring label.\n11. Price times quantity must not exceed the budget (equal passes); out of stock in the records means fail.\n12. A wish marked as optional is not a requirement.\n13. When records conflict (price or stock, say), trust the latest.\n\nAnswer with one JSON object only: {"id": "<item id>", "fit": true or false, "problems": ["one short string per requirement that is unmet or not shown"]}.\nYou may answer "fit": false in two cases only: some listed requirement is plainly violated by the item\'s fields under these rules (an evidence line from code reporting non-compliance counts), or the requirement is shown nowhere (in no field and, for a property without a field, not in the options or the title either). A missing stock or price record never justifies "fit": false.'
PROMPT_RANK = 'A shopping assistant is about to buy an item, and the request demands %s. Below are the requirements, the current pick and the records seen so far.\nTask: applying the reading rules (price times quantity within the budget; no record showing it unavailable), name the listings that may pass every requirement while beating the current pick on that priority (a lower price for cheapest; a better figure on the measure for best).\nLeave out listings whose records show a failed requirement, missing stock or a cost over the budget.\nNumbers inside a row title (a pair of dimensions, say) do not tell which dimension each figure is, so they can never rule a row out.\n\n%s\n\nAnswer with one JSON object only: {"better": ["ids, strongest first, at most 4"], "reason": "one sentence"}.'
PROMPT_AUDIT = 'A shopping assistant is about to tell the shopper that the shop has nothing that fits, which is rarely the correct finish.\nTask: test every listing record against the requirements under the reading rules (price within the budget; no record showing it unavailable).\n\n%s\n\nSome listings are known only as rows: their details were never opened, so their fields are unknown. Any such row on the wanted shelf (whatever object its title mentions) whose price, stock and title reveal no failed requirement is still unchecked (figures in a title, such as a pair of dimensions, never prove a failure): list it as unopened, the most promising first (a title mentioning a required value, or showing the wanted kind with nothing against it).\n\nAnswer with one JSON object only: {"possible_fits": ["ids that seem to pass every requirement, strongest first, at most 5"], "unopened": ["row-only ids worth opening before any decline, at most 6"], "reason": "one sentence"}.'
PROMPT_PLAN = 'You decide what a shopping assistant should ask the shopper next. Each hidden need is one value of one product property; the shopper reveals it only to a message aimed at exactly that property; answers to the first question of a message are the most reliable; asking about anything else only brings polite indifference.\nChoose up to %d distinct properties, strongest first, through these steps:\nStep 0, before anything else: points marked as needing the shopper\'s input and not yet raised, asked plainly. Budget: ask for the highest amount they would spend. Two competing wishes: name each value together with its property (for example: if no single listing manages both, would they rather have this value of one property or that value of the other).\nStep 1: properties that exist as keys in the listing details supplied. Start from the story (a story-hint block, if supplied, pairs story details with their likely property and takes precedence): the hidden need usually sits on whatever property concrete story details relate to while leaving its value unsaid. Story cues and the property they point to:\n - something the user handles or presses, looks at closely or in photos, listens to, measures, tries on for fit, reads on a label, or the way the item was made: that component or facet and its property;\n - tags, labels, fine print: material, fabric, ingredients;\n - photographs, looks: pattern, colour, visible part;\n - a body part or space needing a fit: shape or size of that part;\n - a gift for another person or a slight acquaintance: recipient (gender, age group);\n - how it should feel against hand, skin or foot: material or fabric first;\n - setting, time or activity of use (a fitness class, a workplace, outdoors): style or type, meaning the property whose values name such settings;\n - a part fastened or adjusted again and again: closure or mechanism;\n - an internal working part: the core component.\n Not cues (background): weather, an occupation, a timetable, any place other than where the item is worn or used. Lead with the property the story points to, using the listings\' term for it. Recipient competes equally: raise it if the story concerns another person or if no other hint exists. After that prefer descriptive properties (pattern, material, closure, colour, style, shape, finish) over logistical ones (warranty, pack size, years, delivery time, quantity).\nNever pick: properties the request already settles (its requirements, price, stock, quantity); anything asked before; two near-twins (two style variants, one property under two spellings); commercial or identity facts (brand, model name, delivery, seller, price, rating).\nKeys only: ask only about properties that exist as keys in the listing details supplied, using the listings\' wording. Exception: a hint property without a matching key may be asked under its everyday name, unless it is marked no-key (enough listings of that product were opened without such a field); then use whichever key varies the way the hint describes, or skip it.\nEvery question from step 1: one short, plain, friendly sentence of fifteen words or fewer, asking which option of that property the shopper wants for this kind of item, using the listings\' term for the property (sample values alone never surface a need); one or two example values may follow. No lead-in, no progress report, no instructions, no lists, no catch-all keys (options, variants, title, features, details).\nOne property of one product per question (the step-0 choice between two wishes excepted); two properties or two products in one question are not allowed, because a reply surfaces at most one need and a double question may get no answer at all.\n"second": the runner-up property of the same product (not a synonym of the first), or ""; the assistant may attach it to its last message.\nSeveral products: hidden needs may belong to any product; spread the questions so every product gets one before any gets two (skip products whose properties are used up); draw each product\'s candidates from that product\'s own category menu; write the property followed by the product in brackets, and mention the product in the question (ask about the toe style for the boots).\nAnswer with one JSON object only: {"questions": [{"property": "...", "second": "...", "question": "..."}]}.'
PROMPT_PLAN_ONE = '\nOnly one question goes out now; its reply will be read before the next one is chosen. %d questions remain in all and every one should be used: put the single strongest property first (one or two spares may follow). After a disclosed need, the next question moves to another property and another story detail.'
_TYPES = {'string': lambda v: isinstance(v, str), 'integer': lambda v: isinstance(v, int) and (not isinstance(v, bool)), 'number': lambda v: is_num(v), 'boolean': lambda v: isinstance(v, bool), 'object': lambda v: isinstance(v, dict), 'array': lambda v: isinstance(v, list), 'null': lambda v: v is None}

def _coerce(value, types):
    if not types:
        return value
    known = [t for t in types if t in _TYPES]
    if len(known) < len(types):
        return value
    if any((_TYPES[t](value) for t in types)):
        if 'integer' in types and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for t in types:
        if t == 'integer':
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and re.fullmatch('\\s*-?\\d+(?:\\.0+)?\\s*', value):
                return int(float(value.strip()))
        elif t == 'number':
            if isinstance(value, str) and re.fullmatch('\\s*-?\\d+(?:\\.\\d+)?\\s*', value):
                f = float(value.strip())
                return int(f) if '.' not in value else f
        elif t == 'string':
            if is_num(value):
                return str(value)
        elif t == 'boolean':
            if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
                return value.strip().lower() == 'true'
    return value

def check_args(value, schema, path='args'):
    faults = []
    if not isinstance(schema, dict):
        return (value, faults)
    types = schema.get('type')
    types = [types] if isinstance(types, str) else list(types) if isinstance(types, list) else []
    value = _coerce(value, types)
    if types and all((t in _TYPES for t in types)) and (not any((_TYPES[t](value) for t in types))):
        return (value, ['%s expected %s' % (path, '/'.join(types))])
    if 'enum' in schema and isinstance(schema['enum'], list):
        opts = schema['enum']
        if value not in opts:
            hit = None
            if isinstance(value, str):
                for o in opts:
                    if isinstance(o, str) and o.strip().lower() == value.strip().lower():
                        hit = o
                        break
            if hit is not None:
                value = hit
            else:
                faults.append('%s must be one of %s' % (path, ', '.join((str(o) for o in opts))))
    if 'const' in schema and value != schema['const']:
        faults.append('%s must equal %s' % (path, schema['const']))
    if isinstance(value, str):
        if is_num(schema.get('minLength')) and len(value) < schema['minLength']:
            faults.append('%s shorter than %s' % (path, schema['minLength']))
        if is_num(schema.get('maxLength')) and len(value) > schema['maxLength']:
            value = value[:int(schema['maxLength'])]
    if is_num(value):
        if is_num(schema.get('minimum')) and value < schema['minimum']:
            faults.append('%s below minimum %s' % (path, schema['minimum']))
        if is_num(schema.get('maximum')) and value > schema['maximum']:
            faults.append('%s above maximum %s' % (path, schema['maximum']))
        if is_num(schema.get('exclusiveMinimum')) and value <= schema['exclusiveMinimum']:
            faults.append('%s must exceed %s' % (path, schema['exclusiveMinimum']))
        if is_num(schema.get('exclusiveMaximum')) and value >= schema['exclusiveMaximum']:
            faults.append('%s must stay below %s' % (path, schema['exclusiveMaximum']))
    if isinstance(value, list):
        if is_num(schema.get('minItems')) and len(value) < schema['minItems']:
            faults.append('%s needs at least %s items' % (path, schema['minItems']))
        if is_num(schema.get('maxItems')) and len(value) > schema['maxItems']:
            faults.append('%s allows at most %s items' % (path, schema['maxItems']))
        if isinstance(schema.get('items'), dict):
            fixed = []
            for i, item in enumerate(value):
                v2, f2 = check_args(item, schema['items'], '%s[%d]' % (path, i))
                fixed.append(v2)
                faults.extend(f2)
            value = fixed
    if 'type' in schema and isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        for name in schema.get('required') or []:
            if name not in value:
                faults.append('%s.%s required' % (path, name))
        if schema.get('additionalProperties') is False:
            for name in value:
                if name not in props:
                    faults.append('%s.%s is not a known argument' % (path, name))
        out = {}
        for name, v in value.items():
            if name in props:
                v, f2 = check_args(v, props[name], '%s.%s' % (path, name))
                faults.extend(f2)
            out[name] = v
        value = out
    return (value, faults)

class Wire:

    def __init__(self, binding, finished):
        self.t0 = time.monotonic()
        self.binding = dict(binding or {})
        self.mclient = ProxyClient(timeout=70, max_retries=1)
        self.wclient = ProxyClient(timeout=95, max_retries=1)
        self.spend = 0.0
        self.scale = 1.0
        self.step = 0
        self.finished = bool(finished)
        self.blocked = False
        self.log = []
        self.last_send = 0.0
        self.slow_until = 0.0
        self.resume_at = 0.0
        self.last_reply = self.t0
        self.misses = 0
        self.no_model = False
        self.banned = set()
        self.tool_banned = set()
        self.spares = []
        self.tried = []
        self.mute = False
        self.fail_kind = ''

    def now(self):
        return time.monotonic()

    def time_left(self):
        return DEADLINE - (time.monotonic() - self.t0)

    def ceiling(self):
        return min(SPEND_CAP * self.scale, EPISODE_CAP)

    def spent_out(self):
        return self.spend >= self.ceiling()

    def down(self, at=None):
        at = time.monotonic() if at is None else at
        return self.no_model or (self.misses >= 1 and at - self.last_reply > DOWN_AFTER)

    def usable(self, models, tools=False):
        if self.no_model or self.mute:
            return False
        for m in list(models) + self.spares:
            if m not in self.banned and (not (tools and m in self.tool_banned)):
                return True
        return False

    def note(self, kind, **kw):
        rec = {'kind': kind, 't': round(time.monotonic() - self.t0, 1)}
        rec.update(kw)
        self.log.append(rec)

    def _pace(self, model):
        now = time.monotonic()
        gap = LOAD_GAP if now < self.slow_until else GAP
        wait = self.last_send + gap - now
        if model:
            wait = max(wait, self.resume_at - now)
        wait = min(wait, self.time_left() - 1)
        if wait > 0:
            time.sleep(wait)
        self.last_send = time.monotonic()

    def _backoff(self, seed, tryno, status):
        base = 2.0 if status == 429 else 1.0
        pause = min(base * 2 ** tryno, 8.0)
        f = (7 * len(seed) + sum((ord(c) for c in seed[-64:])) + 31 * tryno + 13 * self.step) % 101 / 100.0
        return pause * (0.75 + f / 2)

    @staticmethod
    def _flat(messages):
        out = []
        for m in messages:
            c = m.get('content')
            if isinstance(c, list):
                m = dict(m)
                m['content'] = '\n\n'.join((str(p.get('text') or '') for p in c if isinstance(p, dict)))
            out.append(m)
        return out

    @staticmethod
    def _cached(messages, model):
        if not model.startswith('anthropic/') or len(messages) <= 2:
            return messages
        out = list(messages)
        for idx in (0, len(out) - 1):
            m = out[idx]
            if isinstance(m.get('content'), str) and m['content']:
                m = dict(m)
                m['content'] = [{'type': 'text', 'text': m['content'], 'cache_control': {'type': 'ephemeral'}}]
                out[idx] = m
        return out

    def model(self, messages, models, cap, tools=None):
        self.fail_kind = ''
        self.tried = []
        if self.mute or self.no_model:
            return None
        ladder = list(models) + [s for s in self.spares if s not in models]
        refused = set()
        kinds = []
        had_empty = False
        seed = None

        def lone():
            return len({m for m in ladder if m not in self.banned}) <= 1

        def fail():
            self.fail_kind = 'empty' if had_empty else 'refused' if kinds and all((k == 'hard' for k in kinds)) else ''
            return None
        i = 0
        while i < len(ladder):
            m = ladder[i]
            pos = i
            i += 1
            if self.no_model or self.spent_out():
                return fail()
            if m in self.banned or m in refused or (tools and m in self.tool_banned):
                continue
            if 'sonnet' in m:
                cheaper_next = pos < len(models) and pos + 1 < len(models) and ('sonnet' not in models[pos + 1])
                try:
                    projected = len(json.dumps(messages)) / 2.3 * 2e-06 + min(cap, 1500) * 1e-05
                except Exception:
                    projected = 0.0
                if self.spent_out() or (cheaper_next and (self.spend >= SOFT_SPEND * self.scale or self.spend + projected > self.ceiling())):
                    continue
            attempts = 1 if self.down() else MODEL_TRIES
            flat = False
            a = 0
            while a < attempts:
                a += 1
                left = self.time_left()
                if left < LAST_CALL - 10:
                    return fail()
                self.mclient.timeout = int(max(8, min(30 if self.down() else 75, left - 13)))
                body = {'model': m, 'messages': self._flat(messages) if flat else self._cached(messages, m), 'temperature': 0, 'max_tokens': cap}
                if tools:
                    body['tools'] = tools
                    body['tool_choice'] = 'auto'
                self.tried.append(m)
                self._pace(True)
                started = time.monotonic()
                try:
                    res = self.mclient.post_verbose('/inference/chat/completions', body)
                    data, err = (res.data, res.error)
                except Exception as exc:
                    data, err = (None, {'status': None, 'body': str(exc)})
                took = time.monotonic() - started
                if isinstance(data, dict) and isinstance(data.get('choices'), list) and data['choices']:
                    self.last_reply = time.monotonic()
                    self.misses = 0
                    cost = (data.get('usage') or {}).get('cost') if isinstance(data.get('usage'), dict) else None
                    if is_num(cost):
                        self.spend += cost
                    msg = (data['choices'][0] or {}).get('message') or {}
                    if msg.get('content') or msg.get('tool_calls'):
                        return msg
                    had_empty = True
                    kinds.append('empty')
                    self.note('model_empty', model=m)
                    break
                self.misses += 1
                status = (err or {}).get('status') if isinstance(err, dict) else None
                if isinstance(data, dict) and status is None:
                    status = None
                text = str((err or {}).get('body') if isinstance(err, dict) else data)
                self.note('model_error', model=m, status=status, body=text[:300])
                if status in (429, 503):
                    self.slow_until = time.monotonic() + LOAD_WINDOW
                low = text.lower()
                if status == 402 or (status == 403 and re.search('key limit|insufficient|credit', low)):
                    self.no_model = True
                    return fail()
                if status in (401, 403, 404):
                    kinds.append('hard')
                    refused.add(m)
                    if status == 404 and tools and ('tool' in low):
                        self.tool_banned.add(m)
                    elif not (status == 403 and re.search('moderat|flag', low)):
                        self.banned.add(m)
                    if status == 403 and (not self.spares):
                        names = [n for n in re.findall('[\\"\']([^\\"\'\\s]{3,100})[\\"\']', text) if '/' in n and n not in self.banned and (n not in models)]
                        self.spares = names[:2]
                        for s in self.spares:
                            if s not in ladder:
                                ladder.append(s)
                    break
                if status in (400, 413, 422):
                    kinds.append('hard')
                    if flat:
                        refused.add(m)
                        break
                    flat = True
                    continue
                if status is not None and 400 <= status < 500 and (status not in (408, 425, 429)):
                    kinds.append('hard')
                    refused.add(m)
                    break
                kinds.append('soft')
                if took > 20:
                    if self.down() and (lone() or len(set(self.tried)) >= 2):
                        return fail()
                    break
                if seed is None:
                    try:
                        seed = json.dumps(messages[-1])[-600:]
                    except Exception:
                        seed = ''
                pause = self._backoff(seed + m, min(self.misses - 1, 6), status)
                if self.time_left() - pause < LAST_CALL - 10:
                    return fail()
                self.resume_at = time.monotonic() + pause
                if self.down(self.resume_at) and (lone() or len(set(self.tried)) >= 2):
                    return fail()
        return fail()

    def consult(self, content, models, cap):
        msg = self.model([{'role': 'user', 'content': content}], models, cap)
        if not msg:
            return None
        c = msg.get('content')
        if isinstance(c, list):
            c = ''.join((str(p.get('text') or '') for p in c if isinstance(p, dict)))
        return read_obj(c)

    def env(self, action):
        stepno = self.step + 1
        paused = 0
        tries = 0
        bumps = 0
        for _ in range(3 + WORLD_TRIES):
            left = self.time_left()
            if left < WORLD_FLOOR:
                return None
            self.wclient.timeout = int(max(3, min(95, left - 1)))
            body = dict(self.binding)
            body['step'] = stepno
            body['action'] = action
            self._pace(False)
            try:
                res = self.wclient.post_verbose('/environment/call', body)
                data, err = (res.data, res.error)
            except Exception as exc:
                data, err = (None, {'status': None, 'body': str(exc)})
            if isinstance(data, dict):
                st = data.get('step')
                self.step = st if isinstance(st, int) and (not isinstance(st, bool)) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                return data
            status = (err or {}).get('status') if isinstance(err, dict) else None
            text = str((err or {}).get('body') if isinstance(err, dict) else '')
            self.note('world_error', status=status, body=text[:300], step=stepno)
            if status in (429, 503):
                self.slow_until = time.monotonic() + LOAD_WINDOW
            low = text.lower()
            if status == 402:
                self.blocked = True
                return None
            if status == 409:
                if 'finish' in low:
                    self.finished = True
                    return None
                m = re.search('expect\\w*\\D{0,24}(\\d+)', low)
                if m:
                    self.step = int(m.group(1)) - 1
                    stepno = self.step + 1
                    continue
                if re.search('reus|already|duplicate', low) and bumps < 2:
                    bumps += 1
                    stepno += 1
                    continue
                return None
            if status in (403, 422):
                return None
            if status in (400, 401, 404) and paused:
                return None
            if paused >= 4:
                return None
            pause = self._backoff(text[-600:], tries, status)
            tries += 1
            paused += 1
            if self.time_left() - pause < WORLD_FLOOR:
                return None
            time.sleep(pause)
        return None

def _brief(v):
    return isinstance(v, str) and 1 <= len(v.strip()) <= 70

def _partcode(v):
    return bool(re.search('\\d', v)) and bool(CODE_SHAPE.match(v)) and ('-' in v or '_' in v)

class Records:

    def __init__(self):
        self.obs = []
        self.known = []
        self.known_set = set()
        self.read = set()
        self.labels = {}
        self._menu_cache = {}

    def intake(self, obs):
        self.obs.append(obs)
        self._menu_cache = {}

        def visit(d):
            for k, v in d.items():
                if 'id' in k.lower() and id_like(v) and (v not in self.known_set):
                    self.known_set.add(v)
                    self.known.append(v)
            oid = own_id(d)
            if oid and prop_map(d) is not None:
                self.read.add(oid)
            lab = record_label(d)
            if oid and lab and (oid not in self.labels):
                self.labels[oid] = lab
        scan(obs, visit)

    def carriers(self, lid):
        out = []
        for ob in self.obs:
            for d in dicts_in(ob):
                if any(('id' in k.lower() and v == lid for k, v in d.items())):
                    out.append(d)
        return out

    def price(self, lid):
        best = None
        for d in self.carriers(lid):
            p = record_price(d)
            if p is not None:
                best = p
        return best

    def label_of(self, lid):
        lab = ''
        for d in self.carriers(lid):
            l2 = record_label(d)
            if l2:
                lab = l2
        return lab or self.labels.get(lid, '')

    def stock_of(self, lid):
        state = None
        for ob in self.obs:
            for d in dicts_in(ob):
                if row_identity(d) == lid:
                    s = record_stock(d)
                    if s is not None:
                        state = s
        return state

    def mentions(self, lid, n=4):
        got = []
        for ob in self.obs:
            stack = [(ob, 0)]
            while stack and len(got) < 8:
                v, depth = stack.pop(0)
                if depth > 6:
                    continue
                if isinstance(v, dict):
                    if any((x == lid for x in v.values() if isinstance(x, str))):
                        text = compact(clip(v, 400), 2500)
                        if text not in got:
                            got.append(text)
                        continue
                    stack.extend(((x, depth + 1) for x in v.values()))
                elif isinstance(v, list):
                    stack.extend(((x, depth + 1) for x in v))
            if len(got) >= 8:
                break
        return got[-n:]

    def merged(self):
        cards = {}
        order = []
        for ob in self.obs:
            for d in dicts_in(ob):
                cid = None
                for k, v in d.items():
                    if 'id' in k.lower() and id_like(v) and (v in self.known_set):
                        cid = v
                        break
                if not cid:
                    continue
                pm = prop_map(d)
                if len(d) < 4 and pm is None:
                    continue
                if cid not in cards:
                    cards[cid] = {}
                    order.append(cid)
                cards[cid].update(d)
                if pm is not None:
                    self.read.add(cid)
        return (cards, order)

    def cards(self, budget=30000, ceiling_id=None, exclude=(), first=(), only=None):
        cards, order = self.merged()
        ceiling = None
        if ceiling_id:
            p = cards.get(ceiling_id, {})
            ceiling = record_price(p) if p else None
        ids = []
        for cid in order:
            if cid in exclude and cid != ceiling_id:
                continue
            if only is not None and cid not in only and (cid != ceiling_id):
                continue
            if ceiling is not None and cid != ceiling_id:
                p = record_price(cards[cid])
                if p is not None and p >= ceiling:
                    continue
            ids.append(cid)
        first = [x for x in first if x in ids]
        groups = [first, [x for x in ids if x in self.read and x not in first], [x for x in ids if x not in self.read and x not in first]]
        texts = {}
        for cid in ids:
            tag = 'DETAILS' if cid in self.read else 'ROW'
            texts[cid] = tag + ' ' + compact(clip(cards[cid], 160))
        if sum((len(t) + 1 for t in texts.values())) > budget:
            groups = [[x for x in g if record_stock(cards[x]) is not False] + [x for x in g if record_stock(cards[x]) is False] for g in groups]
        out, used = ([], 0)
        for g in groups:
            for cid in g:
                t = texts[cid]
                if used + len(t) + 1 > budget:
                    continue
                out.append(t)
                used += len(t) + 1
        return '\n'.join(out)

    def rich_reads(self, category=None):
        n = set()
        for ob in self.obs:
            for d in dicts_in(ob):
                rid = None
                for k, v in d.items():
                    if 'id' in k.lower() and isinstance(v, str) and (v in self.read):
                        rid = v
                        break
                if not rid:
                    continue
                if category is not None and record_label(d) != category.strip().lower() and (self.labels.get(rid, '') != category.strip().lower()):
                    continue
                pm = None
                for v in d.values():
                    if isinstance(v, dict) and len([x for x in v.values() if not isinstance(x, (dict, list))]) >= 3:
                        pm = v
                if pm is not None:
                    n.add(rid)
                elif not any((isinstance(v, dict) for v in d.values())) and len([v for v in d.values() if isinstance(v, str) and 1 <= len(v) <= 70]) >= 5:
                    n.add(rid)
        return len(n)

    def categories_read(self):
        tally = {}
        for ob in self.obs:
            for d in dicts_in(ob):
                if not any(('id' in k.lower() and v in self.read for k, v in d.items() if isinstance(v, str))):
                    continue
                lab = record_label(d)
                if lab and any((isinstance(v, dict) for v in d.values())):
                    tally[lab] = tally.get(lab, 0) + 1
        return [k for k, _ in sorted(tally.items(), key=lambda kv: -kv[1])]

    def menu(self, category=None):
        key = category or ''
        if key in self._menu_cache:
            return self._menu_cache[key]
        counts, rowcounts, samples = ({}, {}, {})
        any_map = [False]
        cat = (category or '').strip().lower()

        def walk(v, inside, depth):
            if depth > 6:
                return
            if isinstance(v, dict):
                here = inside or not cat or record_label(v) == cat or any((isinstance(x, str) and x.strip().lower() == cat and any((f in k.lower() for f in ('categor', 'shelf', 'aisle', 'department'))) for k, x in v.items()))
                if here:
                    brief = [(k, x.strip()) for k, x in v.items() if _brief(x) and (not non_property(k))]
                    inner = any((isinstance(x, dict) for x in v.values()))
                    if len(brief) >= 5 or (cat and len(brief) >= 2 and (not inner)):
                        is_row = len(brief) < 5 and any(('id' in k.lower() and isinstance(x, str) for k, x in v.items()))
                        if not is_row:
                            any_map[0] = True
                        for k, x in brief:
                            if _partcode(x):
                                continue
                            name = re.sub('[_.]', ' ', k).strip().lower()
                            counts[name] = counts.get(name, 0) + 1
                            if is_row:
                                rowcounts[name] = rowcounts.get(name, 0) + 1
                            s = samples.setdefault(name, [])
                            lx = x.lower()
                            if lx not in s and len(s) < 5:
                                s.append(lx)
                for x in v.values():
                    walk(x, here, depth + 1)
            elif isinstance(v, list):
                for x in v:
                    walk(x, inside, depth + 1)
        for ob in self.obs:
            walk(ob, False, 0)
        if any_map[0]:
            for name in list(counts):
                if rowcounts.get(name, 0) >= counts[name]:
                    del counts[name]
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:28]
        lines = [(n, c, samples.get(n, [])) for n, c in top]
        self._menu_cache[key] = lines
        return lines

    @staticmethod
    def menu_text(lines):
        return '\n'.join(('%s (%d listings): %s' % (n, c, ' | '.join(s)) for n, c, s in lines))

    def label_lines(self):
        per = {}
        for ob in self.obs:
            for d in dicts_in(ob):
                if len(d) < 4:
                    continue
                for k, v in d.items():
                    if isinstance(v, str) and 1 <= len(v.strip()) <= 40 and (not (re.search('\\d', v) and CODE_SHAPE.match(v))):
                        per.setdefault(k, {}).setdefault(v.strip(), 0)
                        per[k][v.strip()] += 1
        out = []
        for k, vals in per.items():
            total = sum(vals.values())
            if total >= 4 and 2 <= len(vals) <= 8 and (max(vals.values()) >= 2):
                items = sorted(vals.items(), key=lambda kv: -kv[1])
                out.append('%s: %s' % (k, ', '.join(('%s (%d)' % kv for kv in items))))
        return out[:12]

class CartBook:

    def __init__(self, tool=None, schema=None):
        self.tool = tool
        self.props = (schema or {}).get('properties') or {} if isinstance(schema, dict) else {}
        self.put_form = None
        self.units = {}
        self.mode = 'unknown'
        self.used = False
        self.change_step = -1
        self.view_step = -1
        self.view = None
        self.expect = {}

    def form(self, args):
        return compact({k: v for k, v in (args or {}).items() if isinstance(self.props.get(k), dict) and 'enum' in self.props[k]})

    def list_shape(self):
        for name, sch in self.props.items():
            if not isinstance(sch, dict) or sch.get('type') != 'array':
                continue
            items = sch.get('items') if isinstance(sch.get('items'), dict) else {}
            ip = items.get('properties') if isinstance(items.get('properties'), dict) else {}
            texts = [k for k, s in ip.items() if isinstance(s, dict) and s.get('type') == 'string' and ('enum' not in s)]
            idf = ([k for k in texts if 'id' in k.lower()] or texts or [None])[0]
            cnt = [k for k, s in ip.items() if isinstance(s, dict) and s.get('type') in ('integer', 'number')]
            if idf and cnt:
                return (name, idf, cnt[0])
        return None

    def lines_of(self, args):
        args = args or {}
        shape = self.list_shape()
        if shape and isinstance(args.get(shape[0]), list):
            out = []
            for row in args[shape[0]]:
                if isinstance(row, dict) and isinstance(row.get(shape[1]), str) and row[shape[1]].strip():
                    c = row.get(shape[2])
                    out.append((row[shape[1]].strip(), c if is_num(c) else None))
            return out
        aim = self.flat_aim(args)
        if aim is None:
            return []
        return [(aim, self.flat_count(args))]

    def flat_aim(self, args):
        for k, v in (args or {}).items():
            sch = self.props.get(k) if isinstance(self.props.get(k), dict) else {}
            if isinstance(v, str) and v.strip() and (len(v.split()) == 1) and ('enum' not in sch):
                return v.strip()
        return None

    def flat_count(self, args):
        for k, sch in self.props.items():
            if isinstance(sch, dict) and sch.get('type') in ('integer', 'number') and ('enum' not in sch):
                v = (args or {}).get(k, sch.get('default'))
                if is_num(v):
                    return v
        return None

    def aimed(self, args):
        ls = self.lines_of(args)
        for lid, c in ls:
            if c is None or c > 0:
                return lid
        return ls[0][0] if ls else None

    def count_of(self, args):
        shape = self.list_shape()
        if shape and isinstance((args or {}).get(shape[0]), list):
            aim = self.aimed(args)
            for lid, c in self.lines_of(args):
                if lid == aim:
                    return c
            return None
        return self.flat_count(args)

    def is_put(self, args):
        return self.put_form is None or self.form(args) == self.put_form

    def rows(self, obs):
        known = set(self.units) | set(self.expect)
        found = {}

        def visit(d):
            ids = [v for v in d.values() if isinstance(v, str) and (v in known or id_like(v)) and (' ' not in v)]
            cand = [(k, v) for k, v in d.items() if isinstance(v, int) and (not isinstance(v, bool)) and (0 < v < 1000) and (k.lower() != 'price')]
            if len(cand) > 1:
                q = [(k, v) for k, v in cand if any((f in k.lower() for f in ('qty', 'quant', 'count', 'units', 'unit', 'amount', 'num', 'pieces'))) and (not any((f in k.lower() for f in ('cost', 'fee', 'charge', 'price', 'total', 'sum'))))]
                if len(q) == 1:
                    cand = q
            if len(cand) == 1 and ids:
                pick = next((x for x in ids if x in known), ids[0])
                found[pick] = found.get(pick, 0) + cand[0][1]
        scan(obs, visit)
        if not found or (known and (not set(found) & known)):
            return None
        return found

    def learn(self, rows):
        for lid, exp in list(self.expect.items()):
            if lid in rows:
                if rows[lid] == exp and self.units.get(lid) != exp:
                    self.mode = 'overwrites'
                elif rows[lid] > exp:
                    self.mode = 'adds'
        self.expect = {}
        self.units = {k: v for k, v in rows.items() if v > 0}

    def bare(self, obs, lid):
        if not isinstance(obs, dict) or lid not in self.units:
            return False
        lists = [v for v in obs.values() if isinstance(v, list)]
        nums = [v for v in obs.values() if is_num(v)]
        if not lists or not nums or any(lists) or any((n != 0 for n in nums)):
            return False
        text = compact(obs)
        return not any((o in text for o in self.units if o != lid))

    def book(self, args, obs):
        self.used = True
        if isinstance(obs, dict) and obs.get('error'):
            return ''
        if self.put_form is None:
            self.put_form = self.form(args)
        shape = self.list_shape()
        if shape and isinstance((args or {}).get(shape[0]), list):
            self.mode = 'overwrites'
            new = {}
            for lid, c in self.lines_of(args):
                if c is None or c > 0:
                    new[lid] = new.get(lid, 0) + (c if c is not None else 1)
            self.units = new
            r = self.rows(obs)
            if r:
                self.learn(r)
            return self.summary()
        aim = self.aimed(args)
        if aim is None:
            return ''
        count = self.count_of(args)
        count = 1 if count is None else count
        prior = self.units.get(aim, 0)
        if self.is_put(args):
            if aim in self.units and self.mode != 'overwrites':
                self.units[aim] = prior + count
                self.expect = {aim: count}
            else:
                self.units[aim] = count if count > 0 else 0
        else:
            self.units[aim] = 0 if count <= 0 else max(prior - count, 0)
        if self.units.get(aim, 0) <= 0:
            self.units.pop(aim, None)
        r = self.rows(obs)
        if r:
            self.learn(r)
        elif (not self.is_put(args) or count == 0) and self.bare(obs, aim):
            self.units.pop(aim, None)
            self.expect.pop(aim, None)
        return self.summary()

    def see_view(self, obs, step):
        self.view = obs
        self.view_step = step
        r = self.rows(obs)
        if r:
            self.learn(r)
        elif isinstance(obs, (dict, list)):
            text = compact(obs)
            if not any((lid in text for lid in self.units)):
                self.units = {}

    def summary(self):
        if not self.units:
            return '[cart as counted by code: empty]'
        return '[cart as counted by code: %s]' % ', '.join(('%s x %s' % (k, money(v)) for k, v in self.units.items()))

    def faults(self, products, quantity):
        if not self.units:
            return []
        if products:
            want = [p['quantity'] for p in products]
            if len(self.units) != len(want):
                return ['the cart holds %s but the request wants %d products with exactly one line each' % (', '.join(('%s x %s' % (k, money(v)) for k, v in self.units.items())), len(want))]
            if sorted(self.units.values()) != sorted(want):
                return ['line units %s differ from the wanted quantities %s' % (sorted((money(v) for v in self.units.values())), sorted(want))]
            return []
        if len(self.units) > 1:
            return ['the cart holds %d lines (%s); remove every line except the chosen one' % (len(self.units), ', '.join(self.units))]
        lid, u = next(iter(self.units.items()))
        if u != quantity:
            return ['line %s holds %s units but %s are needed; remove or add the difference' % (lid, money(u), quantity)]
        return []
_MONEY_WORDS_RX = re.compile('\\b(?:price|stock|cents?)\\b')

def ptag(name):
    m = re.search('\\(([^)]*)\\)', (name or '').split(':')[0])
    return m.group(1).strip().lower() if m else ''

def pname(line):
    return (line or '').split(':')[0].split('(')[0].strip()

def pvalue(line):
    return strip_marks(line.split(':', 1)[1]).strip() if ':' in (line or '') else ''

def to_number(v):
    if is_num(v):
        return float(v)
    if isinstance(v, str):
        m = re.search('\\d[\\d,]*(?:\\.\\d+)?', v)
        if m:
            try:
                return float(m.group(0).replace(',', ''))
            except Exception:
                return None
    return None

class Needs:

    def __init__(self, wire, request):
        self.wire = wire
        self.request = request or ''
        self.reading = {}
        self.stated = []
        self.revealed = []
        self.dontmind = []
        self.dropped = []
        self.current = None
        self.budget_changed = None
        self.version = 0
        self.narrow_version = 0
        self.heard = set()
        self.history = []
        self.reply_due = ''
        self.widen_version = -1
        self.widen_step = -10 ** 6
        self.pageable = False
        self.resolved = []
        self.deferred = []
        self.meta_question = None
        self.absorbed_dontmind = 0
        self.last_needs = []
        self.fields_cb = None
        self.priority_cb = None
        self.borrowed = []

    def products(self):
        items = self.reading.get('items') if isinstance(self.reading, dict) else None
        out = []
        for row in items or []:
            if isinstance(row, dict) and str(row.get('product') or '').strip():
                q = row.get('quantity')
                q = int(q) if isinstance(q, (int, float)) and (not isinstance(q, bool)) and (q >= 1) and float(q).is_integer() else 1
                reqs = [str(r) for r in row.get('requirements') or [] if isinstance(r, str) and ':' in r]
                out.append({'product': str(row['product']).strip(), 'quantity': q, 'requirements': reqs})
        return out if len(out) >= 2 else []

    def quantity(self):
        if self.products():
            return 1
        q = self.reading.get('quantity') if isinstance(self.reading, dict) else None
        if isinstance(q, (int, float)) and (not isinstance(q, bool)) and (q >= 1) and float(q).is_integer():
            return int(q)
        return 1

    def line_counts(self):
        ps = self.products()
        return [p['quantity'] for p in ps] if ps else [self.quantity()]

    def priority(self):
        p = str((self.reading or {}).get('priority') or '').strip().lower()
        return p if p in ('cheapest', 'best') else 'none'

    def product_name(self):
        return str((self.reading or {}).get('product') or '').strip() or 'item'

    def currency(self):
        return str((self.reading or {}).get('currency') or '').strip()

    def budget(self):
        if self.budget_changed is not None:
            return self.budget_changed
        v = to_number((self.reading or {}).get('price_cap'))
        return v if v and v > 0 else None

    def _is_dropped(self, line):
        n = re.sub('\\s+', ' ', (line or '').lower()).strip()
        for d in self.dropped:
            dn = re.sub('\\s+', ' ', d.lower()).strip()
            if dn == n:
                return True
            if ':' not in d and loose_same(pname(d), pname(line)) and (ptag(d) == ptag(line)):
                return True
            if ':' in d and loose_same(pname(d), pname(line)) and (ptag(d) == ptag(line)) and (pvalue(d).lower() == pvalue(line).lower()):
                return True
        return False

    def base_list(self):
        return list(self.current) if self.current is not None else self.stated + [r for r in self.revealed if r not in self.stated]

    def binding(self):
        out = []
        for r in self.base_list():
            low = r.lower()
            if self._is_dropped(r) or _MONEY_WORDS_RX.search(low) or 'budget' in low or ('quantity' in low):
                continue
            out.append(r)
        return out

    def multi_note(self):
        ps = self.products()
        if not ps:
            return ''
        return 'SEVERAL PRODUCTS: the purchase is one cart line for each product at its quantity, nothing more; a listing is judged only against the requirements of its own product (its category decides); any cheapest or best priority applies per product. Products: ' + '; '.join(('%d) %s x %d' % (i + 1, p['product'], p['quantity']) for i, p in enumerate(ps)))

    def req_text(self):
        parts = []
        if self.current is not None:
            reqs = [r for r in self.current if not self._is_dropped(r)]
            if reqs:
                parts.append('; '.join(reqs))
        else:
            reqs = ['%s (stated)' % r for r in self.stated if not self._is_dropped(r)]
            reqs += ['%s (revealed)' % r for r in self.revealed if not self._is_dropped(r) and r not in self.stated]
            if reqs:
                parts.append('; '.join(reqs))
        if not parts:
            if self.dropped:
                parts.append('no property requirement remains, so any in-stock listing of the requested category qualifies')
            else:
                parts.append('requirements as written in the request')
        if self.budget_changed is not None:
            parts.append('total price ceiling now %s %s' % (money(self.budget_changed), self.currency()))
        mn = self.multi_note()
        parts.append(mn if mn else 'quantity %d' % self.quantity())
        if self.dropped:
            parts.append('dropped: ' + '; '.join(self.dropped))
        if self.dontmind:
            parts.append('shopper does not mind: ' + '; '.join(self.dontmind))
        return ' | '.join(parts)

    def changes_note(self):
        if not self.dropped:
            return ''
        return 'LATER CHANGES OVERRIDE THE REQUEST TEXT: the shopper dropped ' + '; '.join(self.dropped) + '. A dropped requirement must never fail, rank or favour a listing.'

    def overview(self):
        r = self.reading
        if not r:
            return ''
        out = ['PLAN READ FROM THE REQUEST (check it against what the shopper actually wrote):', '- product: ' + self.product_name()]
        ps = self.products()
        if ps:
            for p in ps:
                out.append('- %s: quantity %d, one cart line' % (p['product'], p['quantity']))
            out.append('- the purchase holds exactly these lines; when the cart tool takes a list of lines, send all of them in every cart call')
        else:
            out.append('- quantity: %d' % self.quantity())
        for s in self.stated:
            out.append('- requirement: ' + s)
        for ref in r.get('references') or []:
            if isinstance(ref, str):
                out.append('- value given by another listing: %s (open that listing first, then search with the value it shows)' % ref)
        b = to_number(r.get('price_cap'))
        if b:
            out.append('- total price cap: %s %s' % (money(b), self.currency()))
        if r.get('in_stock_only') is True:
            out.append('- in-stock items only')
        if self.priority() != 'none':
            det = str(r.get('priority_detail') or '').strip()
            out.append('- priority: %s%s, applied over all fitting items' % (self.priority(), ' (%s)' % det if det else ''))
        else:
            out.append('- no cheapest or best demand: any item meeting everything within budget is correct; do not spend steps hunting the minimum price')
        for s in r.get('soft_wants') or []:
            out.append('- optional (tie-breaker only): %s' % s)
        for s in r.get('ask_first') or []:
            out.append('- ask the shopper first: %s' % s)
        for s in r.get('rules') or []:
            out.append('- procedure rule: %s' % s)
        if self.ep is not None:
            if self.ep.allowance is not None:
                out.append('- message allowance: %d' % self.ep.allowance)
            out.append('- step allowance: %d' % self.ep.budget_steps)
        return '\n'.join(out)
    ep = None

    def _grounded(self, req, question):
        ws = [w for w in re.findall('[a-z]+', req.lower()) if len(w) >= 3]
        if not ws:
            return True
        pool = (self.req_text() + ' ' + ' '.join(self.history) + ' ' + (question or '')).lower()
        return any((w in pool for w in ws))

    def _spells_need(self, needs, lines):
        low = ' '.join(lines).lower()
        for n in needs or []:
            if not isinstance(n, str) or ':' not in n:
                continue
            pw = [w for w in prop_words(n.split(':')[0]) if len(w) >= 3]
            vw = [w for w in words_of(pvalue(n)) if w not in LINK]
            if pw and all((w in low for w in pw)) and all((w in low for w in vw)):
                return True
        return False

    def _renames(self, needs, lines):
        lw = re.findall('[a-z]+', ' '.join(lines).lower())
        joined = ' '.join(lw)
        out = {}
        for n in needs:
            m = re.match('\\s*([^:(]+?)\\s*(\\([^)]*\\))?\\s*:\\s*(.*)$', n)
            if not m:
                continue
            pw = re.findall('[a-z]+', m.group(1).lower())
            if not pw or any((len(w) < 3 for w in pw)) or ' '.join(pw) in joined:
                continue
            for i in range(len(lw) - len(pw) + 1):
                run = lw[i:i + len(pw)]
                if run != pw and all((run[j].startswith(pw[j]) for j in range(len(pw)))):
                    out[n] = '%s%s: %s' % (' '.join(run), ' ' + m.group(2) if m.group(2) else '', m.group(3))
                    break
        return out

    def _store_spelled(self, needs):
        out = {}
        if not self.fields_cb:
            return out
        for n in needs:
            if ':' not in n:
                continue
            prop, val = n.split(':', 1)
            val = strip_marks(val).strip()
            if re.search('\\d', val) or is_reference(val):
                continue
            low = val.lower()
            stripped = COMPARISON_RX.sub(' ', low)
            stripped = re.sub('\\b(?:below|under|less than|lower than|nothing short of|no less than)\\b', ' ', stripped)
            bare = ' '.join(words_of(stripped))
            if not bare:
                continue
            if bare == ' '.join(words_of(low)) and positive_value(val, prop):
                continue
            try:
                fields = self.fields_cb(n) or {}
            except Exception:
                fields = {}
            hit = None
            for key, vals in fields.items():
                if strict_same(key, prop):
                    for x in vals:
                        if ' '.join(words_of(x)) == bare:
                            hit = x
                            break
                if hit:
                    break
            if hit:
                LITERAL_VALUES.add(bare)
                out[n] = '%s: %s' % (prop.strip(), hit.lower())
        return out

    def _phrases(self, lines):
        toks = re.findall('[a-z]+', ' '.join(lines).lower())
        out = []
        for n in (1, 2, 3):
            for i in range(len(toks) - n + 1):
                ph = ' '.join(toks[i:i + n])
                if len(ph) >= 3 and ph not in LINK and (ph not in out):
                    out.append(ph)
        return out

    def drop_ok(self, entry, lines):
        try:
            name = pname(entry)
            tag = ptag(entry)

            def corr(r):
                return loose_same(pname(r), name) and (not tag or not ptag(r) or ptag(r) == tag)
            gathered = [r for r in self.stated if corr(r)]
            if not gathered:
                gathered = [r for r in self.revealed if corr(r) and r not in self.stated and (r not in self.last_needs)]
                if not gathered:
                    return True
                if any((corr(n) and n != entry for n in self.last_needs if isinstance(n, str))):
                    return True
            phrases = self._phrases(lines)
            for r in gathered:
                for ph in phrases:
                    if loose_same(pname(r), ph):
                        return True
            others = [b for b in self.binding()]
            for r in gathered:
                val = pvalue(r)
                if not val or is_reference(val) or (not alternatives(val)):
                    continue
                fields = self.fields_cb(r) if self.fields_cb else {}
                for ph in phrases:
                    if any((strict_same(ph, pname(b)) for b in others if b != r)):
                        continue
                    for key, vals in (fields or {}).items():
                        if strict_same(ph, key) and any((spells(val, x) for x in vals)):
                            self.wire.note('drop_by_field', req=r, field=key)
                            return True
            return False
        except Exception:
            return True

    def pinned(self, asked):
        try:
            atag = ptag(asked)
            refs = [x for x in self.reading.get('references') or [] if isinstance(x, str)] + self.resolved
            for r in self.stated:
                if self._is_dropped(r) or ':' not in r:
                    continue
                prop, val = r.split(':', 1)
                rtag = ptag(prop)
                if atag and rtag and (not kind_set(atag) & kind_set(rtag)):
                    continue
                if not positive_value(val, prop) or story_guess(r, self.request, refs):
                    continue
                if loose_same(asked, prop):
                    return True
                na, nr = (reduced(asked), reduced(prop))
                if len(nr) >= 4 and na.endswith(nr):
                    return True
            return False
        except Exception:
            return False

    def hear(self, lines, question=None):
        fresh = []
        for l in lines:
            if isinstance(l, str) and l.strip() and (l not in self.heard) and (l not in fresh):
                fresh.append(l)
        if not fresh:
            return ''
        for l in fresh:
            self.heard.add(l)
        if self.wire.time_left() < CLOSING:
            return ''
        parts = [PROMPT_HEAR, 'REQUEST (context only)\n' + self.request, 'REQUIREMENTS SO FAR\n' + self.req_text()]
        if self.history:
            parts.append('EARLIER EXCHANGES (oldest first)\n' + '\n'.join(self.history[-12:]))
        if question:
            parts.append('QUESTION THE ASSISTANT SENT\n' + question)
        parts.append('NEW SHOPPER LINES\n' + '\n'.join(fresh))
        self.history.append(('%s -> ' % question if question else '') + ' / '.join(fresh))
        v = self.wire.consult('\n\n'.join(parts), ROLE_LISTENER, CAP_LISTENER)
        if not isinstance(v, dict):
            if self.wire.mute or self.wire.misses > 0:
                for l in fresh:
                    self.heard.discard(l)
                self.history.pop()
                self.deferred.append((fresh, question, time.monotonic()))
            return ''

        def strs(key, keys_only=False):
            x = v.get(key)
            if isinstance(x, dict):
                out = []
                for kk, vv in x.items():
                    if keys_only or vv in (None, '', [], True):
                        out.append(str(kk).strip())
                    elif isinstance(vv, list):
                        out.append('%s: %s' % (kk, ', '.join((str(y) for y in vv))))
                    else:
                        out.append('%s: %s' % (kk, vv))
                return [s for s in out if s]
            if isinstance(x, str) and x.strip():
                return [x.strip()]
            return [str(s).strip() for s in x if isinstance(s, (str, int, float)) and str(s).strip()] if isinstance(x, list) else []
        needs = strs('needs')
        frees = strs('no_preference', keys_only=True)
        drops = strs('dropped')
        cur = v.get('requirements_now')
        if isinstance(cur, dict):
            cur = strs('requirements_now')
        nb = to_number(v.get('new_budget'))
        unavailable = v.get('unavailable') is True
        reply = str(v.get('reply') or '').strip()
        if v.get('meta_only') is True and question and (not self._spells_need(needs, fresh)):
            needs, frees, drops, cur, nb, unavailable, reply = ([], [], [], None, None, False, '')
            self.meta_question = question
            self.wire.note('meta_only', question=question)
        ren = self._renames(needs + ([c for c in cur if isinstance(c, str)] if isinstance(cur, list) else []), fresh)
        if ren:
            needs = [ren.get(n, n) for n in needs]
            if isinstance(cur, list):
                cur = [ren.get(c, c) if isinstance(c, str) else c for c in cur]
            self.wire.note('renamed', pairs=ren)
        masked = fresh
        try:
            sp = self._store_spelled(needs + ([c for c in cur if isinstance(c, str)] if isinstance(cur, list) else []))
            if sp:
                needs = [sp.get(n, n) for n in needs]
                if isinstance(cur, list):
                    cur = [sp.get(c, c) if isinstance(c, str) else c for c in cur]
        except Exception as exc:
            self.wire.note('store_spelled_error', error=str(exc)[:200])
        self.last_needs = list(needs)
        remarks = []
        widening = False
        tighter = False
        if isinstance(cur, list) and all((isinstance(c, str) for c in cur)):
            self.current = [c.strip() for c in cur if c.strip() and self._grounded(c, question)]
        pointing = []
        for n in needs:
            if n not in self.revealed and ':' in n and self._grounded(n, question):
                self.revealed.append(n)
                remarks.append('new requirement: ' + n)
                tighter = True
                if REF_RX.search(n.split(':', 1)[1].lower()):
                    pointing.append(n)
        if pointing:
            self.borrowed.append((pointing, v.get('references') if isinstance(v.get('references'), list) else [], fresh))
        held = [pname(r).lower() for r in self.base_list() if not self._is_dropped(r)]
        for f in frees:
            if f in self.dontmind or pname(f).lower() in held:
                continue
            if self.pinned(f):
                self.absorbed_dontmind += 1
                continue
            self.dontmind.append(f)
        for d in drops:
            if d in self.dropped:
                continue
            if nb is not None and re.search('\\b(?:price|budget|dollars?|cost|spend|cents?|euros?)\\b', d.lower()):
                continue
            if not self.drop_ok(d, masked):
                self.wire.note('drop_kept', entry=d)
                continue
            self.dropped.append(d)
            remarks.append('dropped: ' + d)
            if self.priority() in ('cheapest', 'best'):
                widening = True
        if nb is not None and nb > 0 and (nb != self.budget()):
            old_cap = self.budget()
            self.budget_changed = nb
            if old_cap is not None and nb < old_cap:
                tighter = True
            remarks.append('new budget: %s %s' % (money(nb), self.currency()))
        if unavailable:
            remarks.append('some item is no longer purchasable: take it out of the cart and do not order it')
            tighter = True
        if remarks:
            self.version += 1
            if tighter:
                self.narrow_version = self.version
            if widening:
                self.widen_version = self.version
                self.widen_step = self.wire.step
        if self.current is not None:
            low_cur = ' '.join(self.current).lower()
            low_drop = ' '.join(self.dropped).lower()
            for r in self.revealed:
                nm = pname(r).lower()
                if nm and nm not in low_cur and (nm not in low_drop) and (r not in self.current):
                    self.current.append(r)
            for r in self.stated:
                if self._is_dropped(r) or r in self.current:
                    continue
                try:
                    if self.drop_ok(r, masked):
                        continue
                    if any((loose_same(pname(d), pname(r)) and ptag(d) == ptag(r) for d in self.dropped)):
                        continue
                    if any((loose_same(pname(c), pname(r)) and ptag(c) == ptag(r) for c in self.current)):
                        continue
                    self.current.append(r)
                except Exception:
                    continue
        if widening:
            if not self.binding():
                remarks.append('WIDENED: no property requirement remains, so anything buyable on the requested shelf qualifies (accessories, parts and bundles stocked there included) and the lowest price across the whole shelf wins; cover the shelf with brief queries (the label, each object word in the label, product nouns recurring in titles) and pages of the category listing, then make the cheapest buyable row within budget the only cart line, recheck stock and price, and order')
            else:
                remarks.append('WIDENED: the drop enlarges the acceptable set; listings rejected only on it now pass and listings never fetched may undercut your choice; search the object type again without that requirement (brief queries pairing each remaining required value with an object word, value first, all pages), then pick the cheapest acceptable one')
        self.wire.note('heard', question=bool(question), needs=len(needs), frees=len(frees), drops=len(drops), budget=nb is not None)
        if not question and any(('?' in l for l in fresh)) and (not frees) and reply:
            self.reply_due = reply
        meaning = str(v.get('meaning') or '').strip()
        return 'Meaning: %s%s' % (meaning, ' (' + '; '.join(remarks) + ')' if remarks else '')
ROLE_KEYS = {'search': 'search', 'details': 'details', 'view': 'view', 'message': 'message', 'messaging': 'message', 'cart': 'cart', 'cart_view': 'cart_view', 'cartview': 'cart_view', 'cart viewing': 'cart_view', 'cart_viewing': 'cart_view', 'view_cart': 'cart_view', 'order': 'order', 'decline': 'decline'}

class EpisodeBase:

    def __init__(self, problem_data):
        env = problem_data.get('environment') if isinstance(problem_data, dict) else None
        env = env if isinstance(env, dict) else problem_data
        pv = env['policy_view']
        self.wire = Wire(env.get('binding') or {}, pv.get('finished'))
        self.log = self.wire.log
        q = pv.get('query')
        self.request = q if isinstance(q, str) else ''
        self.opening = pv.get('opening')
        self.tools = [t for t in pv.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict)]
        self.shapes = {}
        for t in self.tools:
            f = t['function']
            p = f.get('parameters')
            self.shapes[f.get('name')] = p if isinstance(p, dict) else {'type': 'object', 'properties': {}}
        ms = pv.get('max_steps')
        self.budget_steps = int(ms) if isinstance(ms, (int, float)) and (not isinstance(ms, bool)) and ms else 30
        self.world_steps = self.budget_steps
        self.records = Records()
        self.needs = Needs(self.wire, self.request)
        self.needs.ep = self
        self.needs.fields_cb = self.shelf_fields
        self.cart = CartBook()
        LITERAL_VALUES.clear()
        self.roles = {}
        self.sent_msgs = 0
        self.allowance = None
        self.orders_tried = 0
        self.refusals = 0
        self.vetted_gen = -1
        self.ranked = set()
        self.vetted_decline = False
        self.audit_done = False
        self.audits = 0
        self.audit_read_count = -1
        self.audit_read_set = set()
        self.audit_fits = []
        self.audit_reviewed = set()
        self.audit_answered = False
        self.early_audit_done = False
        self.qlog = []
        self.code_asks = True
        self.verdicts = {}
        self.put_checked = set()
        self.verified = {}
        self.refused_lines = set()
        self.refusal_at_put = {}
        self.warned = set()
        self.recent = []
        self.chat = []
        self.flow = None
        self.ask_base = None
        self.ask_reserve = 0
        self.saying_todo = False
        self.todos = []
        self.todo_holds = {}
        self.reviewed = False
        self.stuck = 0
        self.idle = 0
        self.hard_fails = 0
        self.seq = 0
        self.event = False
        self.dead = {}
        self.flagged = []
        self.flagged_pages = []
        self.reserve_reads = 0
        self.reply_admits = 0
        self.reply_admitted = set()
        self.backup_used = 0
        self.queue = []
        self.queue_meta = {}
        self.postput = []
        self.covering = False
        self.code_plan = False
        self.fallback_time = None
        self.reread_done = False
        self.setup_content = None
        self.latest_ranking = []
        self.meaning_ok = set()
        self.wanted_questions = []
        self.quiet_labels = []
        self.spent_labels = set()
        self.retry_queue = []
        self.retry_text = {}
        self.retried = set()
        self.open_pairs = []
        self.pairs_off = False
        self.empty_plans = 0
        self.plan_wait = None
        self.last_q_count = -1
        self.last_q_step = -100
        self.first_reads_tried = False
        self.menu_rounds = 0
        self.product_only_used = False
        self.twin_used = set()
        self.asking_label = ''
        self.tool_tally = {}
        self.row_avail = {}
        self.look_avail = {}
        self.stock_fields_seen = False
        self.rows_live = False
        self.rows_lag = False
        self.answer_count = 0
        self.change_counter = 0
        self.line_read = {}
        self.ever_put = set()
        self.cart_version = 0
        self.gen = 0
        self.needs_seen = 0
        self.vetted_gen = -1
        self.vetted_once = False
        self.face_read = set()
        self.algo = {}
        self.replans = 0
        self.plan_mark = None
        self.cart_shown_version = -1
        self.actor_calls_since_cart = 0
        self.multi = None
        self.multi_proven = False
        self.multi_retired = False
        self.refusal_texts = []
        self.search_log = {}
        self.totals = {}
        self.word_rule = 'unknown'
        self.kw_history = []
        self.code_searched = False
        self.next_pages = []
        self.hits = {}
        self.code_fit = {}
        self.order_holds = {}
        self.decline_once = set()
        self.swap_tried = set()
        self.cheaper_holds = {}
        self.sold_out_seen = False
        self.wanted_hint = None
        self._table_n = -1
        self._table = {}
        self._ev_cache = {}
        self._home_n = -1
        self._home = []
        self.closed = False

    def steps_left(self):
        return self.budget_steps - self.wire.step

    def role(self, r):
        return self.roles.get(r)

    def allowance_in_force(self):
        return self.allowance if self.allowance is not None else DEFAULT_ALLOWANCE

    def classify(self, name, args):
        o, d = (self.role('order'), self.role('decline'))
        if not name or name not in (o, d):
            return None
        if o != d:
            return 'order' if name == o else 'decline'
        for v in (args or {}).values():
            if isinstance(v, str) and len(v) <= 40:
                low = v.lower()
                if DECLINE_RX.search(low):
                    return 'decline'
                if ORDER_RX.search(low):
                    return 'order'
        return 'order'

    def finishers(self):
        return {x for x in (self.role('order'), self.role('decline')) if x}

    def clock_text(self):
        usable = self.wire.time_left() - LAST_CALL
        if usable > (DEADLINE - LAST_CALL) / 2:
            return ''
        if usable < CLOCK_STEP:
            return 'under 30 s'
        return 'about %d s' % (int(usable // CLOCK_STEP) * CLOCK_STEP)

    def status(self):
        left = self.steps_left()
        s = 'STATUS: steps used %d of %d, %d left; messages sent %d%s' % (self.wire.step, self.budget_steps, left, self.sent_msgs, ' of %d' % self.allowance if self.allowance is not None else '')
        ct = self.clock_text()
        if ct:
            s += '; time left ' + ct
        s += '\nREQUIREMENTS: ' + self.needs.req_text()
        s += self.ref_status()
        offer = self.offer()
        if len(offer) < len(self.shapes):
            s += '\nTOOLS USABLE NOW: %s (any other call is turned away at no step cost%s)' % (', '.join(sorted(offer)), '; the code itself plans and sends the shopper questions' if self.code_asks and self.role('message') else '')
        if self.cart.used:
            s += '\nCART: ' + self.cart.summary()
        res = self.reserve()
        both = bool(self.role('order') and self.role('cart'))
        if left <= 2:
            s += '\nLAST STEPS: if the cart already holds a candidate for each requested product, order now; if not, put in the best one (a verified fit; failing that the cheapest available affordable listing with no requirement visibly failing in its record) and order; decline only if the records prove nothing can fit.'
        elif left <= 5 or self.wire.time_left() < CLOSING:
            s += "\nFINISH NOW: trim the cart to a single line per product picked the same way, remove any sold-out line, read the kept line's stock again, then order; decline only if the records prove nothing can fit."
        elif self.wire.spent_out():
            s += '\nMODEL BUDGET USED: stop exploring; within a few steps put the top candidate for each product (one line each at its quantity; a verified listing where there is one, otherwise the nearest match the records support), refresh its stock and price, then order; declining is right only if the records prove some product has nothing able to satisfy its requirements.'
        elif both and self.orders_tried == 0 and (res < left <= res + 7):
            s += '\nSTEPS GETTING SCARCE: searches and detail reads stop once %d steps remain; mending a refused order takes about 4 steps.' % res
        elif both and self.orders_tried == 0 and (left <= res):
            s += "\nSEARCH PHASE CLOSED: details only for the cart's own line, for listings a check named, or (while a product still lacks a verified listing) three extra reads or searches in all; put your best verified listing in the cart, refresh its stock and price, and order; repairing a refusal takes about 4 steps."
        return s

    def tool_lines(self):
        rows = []
        for t in self.tools:
            f = t['function']
            rows.append(compact({'name': f.get('name'), 'description': f.get('description'), 'parameters': f.get('parameters')}))
        return '\n'.join(rows)[:VIEW_LEN]

    def setup(self):
        fixed = PROMPT_READ + '\n\nTOOLS\n' + self.tool_lines()
        var = 'REQUEST\n' + self.request
        if self.opening is not None:
            var += '\n\nOPENING OBSERVATION\n' + compact(self.opening)
        content = [{'type': 'text', 'text': fixed, 'cache_control': {'type': 'ephemeral'}}, {'type': 'text', 'text': var}]
        self.setup_content = content
        reading, partial = (None, None)
        for rnd in range(3):
            last = None
            for m in ROLE_READER_SETUP:
                last = self.wire.consult(content, [m], CAP_READER)
                if isinstance(last, dict) and isinstance(last.get('roles'), dict):
                    reading = last
                    break
                partial = last if isinstance(last, dict) else None
                self.wire.note('setup_retry', model=m)
            if reading is not None:
                break
            if not (last is None and self.wire.fail_kind == '' and (not self.wire.down()) and self.wire.usable(ROLE_READER_SETUP) and (self.wire.time_left() >= CLOSING + 60)):
                break
        if reading is None:
            try:
                code = self.code_reading()
            except Exception as exc:
                self.wire.note('code_reading_error', error=str(exc)[:300])
                code = {}
            if isinstance(partial, dict):
                code = dict(code)
                code.update({k: v for k, v in partial.items() if k != 'roles'})
            reading = code
            self.code_plan = True
            self.fallback_time = time.monotonic()
        self.apply_reading(reading)
        self.ref_register_setup()
        self.wire.note('plan', roles=self.roles, product=self.needs.product_name(), cap=self.needs.budget(), products=len(self.needs.products()), code=self.code_plan)

    def apply_reading(self, r, reread=False):
        r = dict(r or {})
        old_cart = self.roles.get('cart')
        old_budget = self.needs.budget()
        roles = {}
        for k, v in (r.get('roles') or {}).items() if isinstance(r.get('roles'), dict) else []:
            key = ROLE_KEYS.get(str(k).strip().lower().replace('-', '_'))
            if key and isinstance(v, str) and (v in self.shapes):
                roles[key] = v
        self.roles = roles
        if reread and self.cart.used and old_cart:
            self.roles['cart'] = old_cart
        if not (reread and self.cart.used):
            ct = self.roles.get('cart')
            self.cart = CartBook(ct, self.shapes.get(ct))
        self.needs.pageable = bool(self.role('search') and self.paging_param())
        mb = r.get('message_budget')
        if isinstance(mb, str) and re.fullmatch('\\s*\\d+\\s*', mb):
            mb = int(mb)
        if is_num(mb) and mb >= 0:
            self.allowance = int(mb)
        self.needs.reading = r
        try:
            self.wire.scale = 1.0 + 0.3 * min(2, max(0, len(self.needs.products()) - 1))
        except Exception:
            self.wire.scale = 1.0
        if isinstance(r.get('requirements'), dict):
            r['requirements'] = ['%s: %s' % (k, v) for k, v in r['requirements'].items() if v not in (None, '')]
        reqs = [str(x).strip() for x in r.get('requirements') or [] if isinstance(x, (str, int, float))]
        stated = [x for x in reqs if re.match('^\\s*[^:]*\\S[^:]*:\\s*\\S', x)]
        r['soft_wants'] = [str(x) for x in r.get('soft_wants') or [] if isinstance(x, (str, int, float))] + [x for x in reqs if x not in stated]
        r['requirements'] = stated
        self.needs.stated = list(stated)
        try:
            self.story_values(r)
        except Exception as exc:
            self.wire.note('story_error', error=str(exc)[:200])
        if reread and self.needs.version > 0 and (self.needs.budget_changed is None) and old_budget:
            self.needs.budget_changed = old_budget
        if r and self.needs.budget() is None and self.role('message'):
            af = [str(x) for x in r.get('ask_first') or [] if isinstance(x, (str, int, float))]
            af.insert(0, "the shopper's highest total spend, because the request sets no price limit (skip this if it is already listed)")
            r['ask_first'] = af
        if reread:
            self.budget_steps = self.world_steps
        sb = r.get('step_budget')
        if isinstance(sb, str) and re.fullmatch('\\s*\\d+\\s*', sb):
            sb = int(sb)
        if is_num(sb) and 0 < sb < self.budget_steps:
            self.budget_steps = int(sb)
        if reread:
            self.needs.version += 1

    def story_values(self, r):
        raw = r.get('hints')
        hints = []
        if isinstance(raw, dict):
            for k, v in raw.items():
                for x in v if isinstance(v, list) else [v]:
                    if x in (None, ''):
                        continue
                    x = str(x)
                    hints.append(x if ':' in x else '%s: %s' % (k, x))
        elif isinstance(raw, list):
            hints = [str(x).strip() for x in raw if isinstance(x, (str, int, float)) and str(x).strip()]
        hints = [single_prop_hint(h) for h in hints]
        own = list(hints)
        refs = [x for x in r.get('references') or [] if isinstance(x, str)]
        stated = list(self.needs.stated)
        guesses = [s for s in stated if s not in refs and story_guess(s, self.request, refs)]
        if guesses and len(stated) >= 2 and (len(guesses) <= len(stated) / 2.0):
            for g in guesses:
                self.needs.stated.remove(g)
                if g in r['requirements']:
                    r['requirements'].remove(g)
                for it in r.get('items') or []:
                    if isinstance(it, dict) and isinstance(it.get('requirements'), list):
                        it['requirements'] = [x for x in it['requirements'] if not (isinstance(x, str) and pname(x).lower() == pname(g).lower() and (pvalue(x).lower() == pvalue(g).lower()))]
                head = g.split(':', 1)[0].strip()
                hints.append('%s: the story suggests %s, but it is not given as a value, so ask' % (head, pvalue(g)))
            self.wire.note('story_guesses_moved', guesses=guesses)
        converted = self.written_hints(r, own)
        hints = [h for h in hints if h not in converted][:12]
        r['hints'] = hints

    def written_hints(self, r, own):
        done = []
        for h in own:
            if ':' not in h:
                continue
            head, detail = h.split(':', 1)
            prop = head.split('(')[0].strip()
            if not prop or non_property(prop):
                continue
            tag = ptag(head)
            names = [tag] if tag else [it.get('product') for it in r.get('items') or [] if isinstance(it, dict) and it.get('product')] or [r.get('product') or '']
            word, used = ('', '')
            for nm in names:
                word = lead_value(detail, self.request, str(nm))
                if word:
                    used = str(nm)
                    break
            if not word:
                continue
            covered = False
            for s in self.needs.stated:
                st = ptag(s)
                if st and (not kind_set(st) & kind_set(used)):
                    continue
                if loose_same(pname(s), prop) or re.search('\\b%s\\b' % re.escape(word), pvalue(s).lower()):
                    covered = True
                    break
            if covered:
                continue
            line = '%s: %s' % (head.strip(), word)
            self.needs.stated.append(line)
            r['requirements'].append(line)
            for it in r.get('items') or []:
                if isinstance(it, dict) and isinstance(it.get('requirements'), list) and (len(r.get('items')) == 1 or kind_set(str(it.get('product'))) & kind_set(used)):
                    it['requirements'].append(line)
                    break
            done.append(h)
            self.wire.note('hint_to_requirement', hint=h, line=line)
        return done

    def reserve(self):
        b = self.budget_steps
        base = min(max(b // 4 + 1, RESERVE_LOW), RESERVE_HIGH)
        due = max(min(b // 3, 16), base)
        if self.event or any((l in self.dead for l in self.cart.units)):
            try:
                due = max(due, 8 + max(1, len(self.needs.products())))
            except Exception:
                due = max(due, 9)
        return max(min(due, b // 2), base)

    def narrowing(self):
        return bool(self.role('order') and self.role('cart'))

    def reserve_phase(self):
        left = self.steps_left()
        return self.narrowing() and left > 3 and (self.orders_tried == 0 and left <= self.reserve() or (self.refusals > 0 and left <= REFUSAL_RESERVE))

    def verified_now(self):
        ver = self.needs.version
        return [lid for lid, v in self.verified if v == ver and lid not in self.refused_lines and (lid not in self.dead)]

    def no_verified(self):
        try:
            vs = self.verified_now()
            if not vs:
                return True
            ps = self.needs.products()
            if ps:
                homes = self.home_labels()
                labs = [self.category_of(x) for x in vs]
                if all(labs):
                    for hs in homes:
                        if hs and (not any((l in hs for l in labs))):
                            return True
            return False
        except Exception:
            return False

    def flagged_unread(self):
        return [x for x in self.flagged if x in self.records.known_set and x not in self.records.read and (x not in self.refused_lines)]

    def widened_late(self):
        return self.needs.widen_version >= 0 and self.wire.step - self.needs.widen_step <= self.reserve()

    def offer(self):
        names = set(self.shapes)
        out = set(names)
        if self.narrowing():
            left = self.steps_left()
            o, d, c = (self.role('order'), self.role('decline'), self.role('cart'))
            if left <= 1:
                out = {o, d}
            elif left <= 3:
                out = {o, d, c}
            elif self.reserve_phase():
                out = {o, d, c, self.role('cart_view'), self.role('view') or self.role('details'), self.stock_tool()}
                det = self.role('details')
                if self.reserve_reads < 3 and (self.no_verified() and left > 5 or (self.flagged_unread() and left >= min(9, self.reserve()))):
                    out.add(det)
                if self.reserve_reads < 3 and left > 4 and (self.widened_late() or self.flagged_pages or (self.no_verified() and left > 5)):
                    out.add(self.role('search'))
                if left >= 7 and self.backup_rows():
                    out.add(det)
                if left > 3 and self.wanted_hint and (self.wanted_hint not in self.records.read):
                    out.add(det)
            out = {x for x in out if x}
            if not out:
                out = set(names)
        if self.code_asks and self.role('message'):
            out.discard(self.role('message'))
        return out

    def backup_rows(self):
        try:
            if self.needs.priority() != 'cheapest' or self.backup_used >= 2:
                return []
            if not (self.event or self.sold_out_seen or self.flagged_unread()):
                return []
            tab = self.table()
            out = []
            cap = self.needs.budget()
            for k, homes in enumerate(self.home_labels() or [set()]):
                vs = [x for x in self.verified_now() if not homes or self.category_of(x) in homes]
                prices = [self.records.price(x) for x in vs if self.records.price(x) is not None]
                if not prices:
                    continue
                low = min(prices)
                units = self.units_for(k)
                for lid, row in tab.items():
                    p = row['price']
                    if lid in self.records.read or lid in self.dead or lid in self.refused_lines or (p is None):
                        continue
                    if homes and row['label'] not in homes:
                        continue
                    if cap is not None and p * units > cap:
                        continue
                    if row['stock'] is False or p >= low:
                        continue
                    out.append((p, lid))
            return [lid for _, lid in sorted(set(out))][:6]
        except Exception as exc:
            self.wire.note('backup_error', error=str(exc)[:200])
            return []

    def units_for(self, k):
        ps = self.needs.products()
        if ps and k is not None and (0 <= k < len(ps)):
            return ps[k]['quantity']
        return self.needs.quantity()

    def reserve_gate(self, name, args):
        det, srch = (self.role('details'), self.role('search'))
        lookups = self.lookup_tools()
        kind = None
        if name == srch:
            kind = 'search'
        elif name == det or name in lookups:
            kind = 'details'
        if kind is None or not self.reserve_phase():
            return ''
        ids = self.ids_named(args)
        if kind == 'details' and ids and all((i in self.cart.units or any((i == v[0] for v in self.verified)) for i in ids)):
            return ''
        left = self.steps_left() - self.reply_admits
        if self.reserve_reads + self.reply_admits < 3:
            ok = False
            if kind == 'details':
                if self.no_verified() and left > 5:
                    ok = True
                elif ids and any((i in self.flagged_unread() for i in ids)) and (left >= min(9, self.reserve())):
                    ok = True
            elif left > 4 and (self.widened_late() or compact(args) in self.flagged_pages or (self.no_verified() and left > 5)):
                ok = True
            if ok:
                self.reply_admits += 1
                self.reply_admitted |= set(ids)
                return ''
        if kind == 'details':
            backs = self.backup_rows()
            if ids and any((i in backs and i not in self.reply_admitted for i in ids)) and (left - 1 >= 6) and (self.backup_used < 2):
                self.backup_used += 1
                self.reply_admits += 1
                self.reply_admitted |= set(ids)
                return ''
            if self.wanted_hint and ids and (self.wanted_hint in ids) and (self.steps_left() > 2):
                self.wanted_hint = None
                self.reply_admits += 1
                return ''
        backs = self.backup_rows()
        return 'Not executed, no step used: the remaining steps belong to the order. From now on details are opened only for current cart lines, a listing a check named, or (with no verified listing) three listings overall. Put the best verified listing in the cart, refresh its stock and price, and order.' + (' Cheaper unread rows that could back up the choice: %s.' % ', '.join(backs) if backs else '')
_TOTALISH = ('qty', 'total', 'payable', 'subtotal', 'sum', 'count', 'quantity', 'shipping', 'tax')

class ChecksPart:

    def table(self):
        n = len(self.records.obs)
        if self._table_n == n:
            return self._table
        tab = {}
        for ob in self.records.obs:
            for d in dicts_in(ob):
                lid = None
                for k, v in d.items():
                    if k.lower() == 'id' and isinstance(v, (str, int)) and (not isinstance(v, bool)):
                        lid = str(v)
                        break
                if lid is None:
                    for k, v in d.items():
                        if k.lower().endswith('id') and isinstance(v, (str, int)) and (not isinstance(v, bool)):
                            lid = str(v)
                            break
                if not lid or lid not in self.records.known_set:
                    continue
                row = tab.setdefault(lid, {'label': '', 'title': '', 'price': None, 'stock': None, 'kinds': set(), 'fields': {}})
                lab = record_label(d)
                if lab:
                    row['label'] = lab
                for k, v in d.items():
                    kl = k.lower()
                    if isinstance(v, str) and v.strip() and ('title' in kl or 'name' in kl) and ('id' not in kl) and (not any((b in kl for b in ('brand', 'seller', 'vendor', 'shop', 'store', 'categor')))):
                        row['title'] = v
                        break
                for k, v in d.items():
                    kl = k.lower()
                    if is_num(v) and any((w in kl for w in ('price', 'cost', 'amount'))) and (not any((w in kl for w in _TOTALISH))):
                        row['price'] = v
                        break
                st = record_stock(d)
                if st is not None:
                    row['stock'] = st
                for v in d.values():
                    if isinstance(v, str):
                        row['kinds'] |= set(kind_list(v))
                    elif isinstance(v, dict):
                        for x in v.values():
                            if isinstance(x, str):
                                row['kinds'] |= set(kind_list(x))
                pm = prop_map(d)
                if pm is not None:
                    for k, v in pm.items():
                        if isinstance(v, (str, int, float)) and (not isinstance(v, bool)):
                            row['fields'][k] = v
                        elif isinstance(v, list) and all((isinstance(x, (str, int, float)) for x in v)):
                            row['fields'][k] = ', '.join((str(x) for x in v))
        self._table_n = n
        self._table = tab
        return tab

    def category_of(self, lid):
        return self.records.label_of(lid)

    def product_names(self):
        ps = self.needs.products()
        return [p['product'] for p in ps] if ps else [self.needs.product_name()]

    def home_labels(self):
        n = len(self.records.obs)
        if self._home_n == n:
            return self._home
        try:
            labels = set()
            for ob in self.records.obs:
                for d in dicts_in(ob):
                    if any(('id' in k.lower() and v in self.records.known_set for k, v in d.items() if isinstance(v, str))):
                        lab = record_label(d)
                        if lab:
                            labels.add(lab)
            names = self.product_names()
            out = []
            for i, nm in enumerate(names):
                kinds = kind_list(nm)
                shared = set()
                for j, other in enumerate(names):
                    if j != i:
                        shared |= set(kind_list(other))
                scores = {}
                for lab in sorted(labels):
                    toks = set(kind_list(lab))
                    scores[lab] = kind_score(kinds, toks, shared)
                top = max(scores.values()) if scores else 0
                out.append({l for l, s in scores.items() if top > 0 and s == top})
        except Exception:
            out = [set() for _ in self.product_names()]
        self._home_n = n
        self._home = out
        return out

    def own_label(self):
        if self.needs.products():
            return ''
        homes = self.home_labels()
        hs = homes[0] if homes else set()
        if not hs:
            return ''
        read = self.records.categories_read()
        if all((r in hs for r in read)):
            return ''
        for r in read:
            if r in hs:
                return r
        return sorted(hs)[0]

    def product_of_listing(self, lid):
        if not self.needs.products():
            return None
        lab = self.category_of(lid)
        for i, hs in enumerate(self.home_labels()):
            if lab and lab in hs:
                return i
        return None

    def product_index(self, text):
        ps = self.needs.products()
        if not ps:
            return None
        names = [p['product'] for p in ps]
        for piece in ([re.search('\\(([^)]*)\\)', text).group(1)] if re.search('\\(([^)]*)\\)', text or '') else []) + [text or '']:
            toks = set(kind_list(piece))
            scores = []
            for i, nm in enumerate(names):
                shared = set()
                for j, o in enumerate(names):
                    if j != i:
                        shared |= set(kind_list(o))
                scores.append(kind_score(kind_list(nm), toks, shared))
            top = max(scores)
            if top > 0 and scores.count(top) == 1:
                return scores.index(top)
        return None

    def shelf_fields(self, label_req):
        try:
            ps = self.needs.products()
            if ps:
                k = self.product_index(label_req)
                homes = self.home_labels()
                labs = homes[k] if k is not None and k < len(homes) else set()
            else:
                ol = self.own_label()
                labs = {ol} if ol else self.home_labels()[0] if self.home_labels() else set()
            out = {}
            for lid, row in self.table().items():
                if not row['fields'] or (labs and row['label'] not in labs):
                    continue
                for key, v in row['fields'].items():
                    if non_property(key) or not str(v).strip():
                        continue
                    out.setdefault(key, []).append(str(v))
            return out
        except Exception:
            return {}

    def evidence(self, lid):
        key = (lid, len(self.records.obs), self.needs.version, '|'.join(self.needs.binding()))
        if key in self._ev_cache:
            return self._ev_cache[key]
        try:
            out = self._evidence(lid)
        except Exception as exc:
            self.wire.note('evidence_error', error=str(exc)[:300])
            out = ([], [], [])
        self._ev_cache[key] = out
        return out

    def misses(self, lid):
        return self.evidence(lid)[2]

    def _evidence(self, lid):
        tab = self.table()
        row = tab.get(lid)
        if not row or not row['fields']:
            return ([], [], [])
        multi = bool(self.needs.products())
        lab = row['label']
        if multi and (not lab):
            return ([], [], [])
        prod = self.product_of_listing(lid) if multi else None
        peers = [x for x, r in tab.items() if r['fields'] and (r['label'] == lab if lab else True)]
        if len(peers) < 3:
            return ([], [], [])
        bearers, filed = ({}, {})
        for p in peers:
            for k, v in tab[p]['fields'].items():
                txt = str(v)
                if non_property(k) or not txt.strip():
                    continue
                bearers.setdefault(k, set()).add(p)
                els = list_elements(txt)
                for e in els:
                    filed.setdefault(k, []).append((p, e))
                if len(els) >= 2:
                    filed[k].append((p, txt.lower()))
        sentences, rules, misses = ([], [], [])
        fields = row['fields']
        for req in self.needs.binding()[:12]:
            if ':' not in req:
                continue
            head, val = req.split(':', 1)
            val = strip_marks(val).strip()
            prop = head.split('(')[0].strip()
            if not prop or not val or is_reference(val) or re.search('redirect|refer', val.lower()):
                continue
            lowv = val.lower()
            owner = self.product_index(head) if multi and ptag(head) else prod
            if re.search('\\d', lowv) and LIMIT_RX.search(lowv):
                keys2 = [k for k in bearers if len(bearers[k]) >= 2 and strict_same(k, prop)]
                pws = [w for w in prop_words(prop) if len(w) >= 3 and w not in LINK]
                if keys2 and pws and (not multi or (owner is not None and owner == prod)) and (not any((str(v).strip() and (strict_same(k, prop) or set(prop_words(k)) & set(pws)) for k, v in fields.items()))):
                    sentences.append("%s: %s -> the store's key for this bound is %s, borne by %d listings; this listing lacks it, so the bound is not shown satisfied: fails (another key's figure is no substitute)" % (prop, val[:70], '/'.join(keys2[:3]), len(set().union(*[bearers[k] for k in keys2]))))
                    misses.append('%s: it lacks the key %s, leaving the value unproven' % (prop, keys2[0]))
                continue
            if re.fullmatch('\\s*\\d+(?:\\.\\d+)?\\s*[a-z%\\"\']+\\.?\\s*', lowv):
                continue
            if multi and ptag(head) and (owner is None or prod is None or owner != prod):
                continue
            pos = positive_value(val, prop)
            excl = bool(EXCLUSION_RX.search(lowv))
            if not pos and (not excl):
                continue
            alts = alternatives(val)
            if not alts:
                continue
            named = sorted([k for k in bearers if strict_same(k, prop)], key=lambda k: -len(bearers[k]))
            by_name = bool(named)
            if not pos and named:
                vw = value_words(val)
                if any((value_words(e) == vw for k in named for _, e in filed.get(k, []))):
                    pos = True
                    LITERAL_VALUES.add(' '.join(vw))
            if not named:
                if not pos or (multi and prod is None):
                    continue
                cands = [k for k in bearers if len(bearers[k]) >= 2 and any((spells(val, e) for _, e in filed.get(k, [])))]
                if not cands:
                    continue
                top = max(cands, key=lambda k: len(bearers[k]))
                named = sorted([k for k in bearers if strict_same(k, top)], key=lambda k: -len(bearers[k]))
            nbear = len(set().union(*[bearers[k] for k in named]))
            if pos and (not (excl and (not positive_value(val, prop)))):
                spellers = {p for k in named for p, e in filed.get(k, []) if p != lid and spells(val, e)}
            else:
                bare = EXCLUSION_RX.sub(' ', lowv)
                spellers = {p for k in named for p, e in filed.get(k, []) if p != lid and spells(bare, e)}
            standalone = {e for k in named for p, e in filed.get(k, []) if p != lid}
            samples = []
            for k in named:
                for _, e in filed.get(k, []):
                    if e[:40] not in samples and len(samples) < 6:
                        samples.append(e[:40])
            opening = '%s: %s -> store key %s%s, borne by %d listings; ' % (prop, val[:70], '/'.join(named[:3]), " (no key bears the property's name; this key's texts spell the value)" if not by_name else '', nbear)
            is_excl = not pos
            opening += '%d other listings spell the excluded value there (any listing spelling it fails)' % len(spellers) if is_excl else '%d other listings spell the value itself there' % len(spellers)
            naming = []
            if not is_excl and (not spellers):
                kw = kind_list(val)
                if kw:
                    naming = [x for x, r in tab.items() if x != lid and r['label'] == lab and all_kinds(kw, r['kinds'])][:3]
            smp = 'samples: ' + ' | '.join(samples)
            if is_excl:
                rule = opening
            elif spellers:
                rule = opening + '. Only a field bearing those words satisfies it (containing them is enough, so is one list element; synonyms and related words are not)'
                if not by_name:
                    rule += "; a listing titled with the value satisfies it too unless its own field sharing the property's name contradicts"
            elif naming:
                rule = opening + ". %s. Listings %s name the value in titles or texts, so related wording satisfies it only once one of those listings' records shows this key spelling it otherwise" % (smp, ', '.join(naming))
            elif nbear < SYNONYM_MIN:
                rule = opening + '. %s. Too few bearers to know whether the store spells the value itself, so an equivalent term is not yet shown to satisfy it' % smp
            else:
                rule = opening + '. %s. Since nobody spells it, an equivalent term may satisfy it' % smp
            rules.append(rule)
            own = [(k, str(fields[k])) for k in named if k in fields and str(fields[k]).strip()]
            pw = prop_words(prop)
            qual = [(k, str(v)) for k, v in fields.items() if k not in named and str(v).strip() and (not non_property(k)) and (len(prop_words(k)) == len(pw) + 1) and (prop_words(k)[1:] == pw)]
            q_spells = [k for k, v in qual if spells(val, v)]
            q_near = bool(q_spells) or any((touches(val, v) for k, v in qual))
            qual_remark = "its field %s carries the property's name with one extra word in front, which names that property in fuller form, and spells the value: satisfied" % (q_spells[0] if q_spells else '')
            if self.redirects_property(lid, prop):
                sentences.append(opening + "; this listing's notes might redirect this property elsewhere, so its own field might not be the deciding one")
                continue
            if is_excl:
                sentences.append(opening + '; this listing shows %s; bearing the excluded value means failure' % (', '.join(('%s=%s' % (k, v[:90]) for k, v in own)) or 'no such field'))
                continue
            pws3 = {w for w in pw if len(w) >= 3 and w not in LINK}
            kin = [(k, v) for k, v in fields.items() if not non_property(k) and {w for w in prop_words(k) if len(w) >= 3} & pws3 and (not spells(val, str(v)))]
            title_speaks = not by_name and (not kin) and spells(val, row['title'])
            shown = '; this listing shows ' + ', '.join(('%s=%s' % (k, v[:90]) for k, v in own))
            if not own:
                if q_spells:
                    sentences.append(opening + '; it has no such field; ' + qual_remark)
                elif title_speaks:
                    sentences.append(opening + "; it has no such field, no key bears the property's name, and its title spells the value: satisfied")
                else:
                    others = [k for k, v in fields.items() if spells(val, str(v))] + [k for k, v in fields.items() if {w for w in prop_words(k) if len(w) >= 3} & pws3]
                    others = list(dict.fromkeys(others))[:4]
                    sentences.append(opening + "; it has no such field; other fields possibly naming the property: %s; an alternate label for this very property counts, a different property's field never substitutes" % (', '.join(others) or 'none'))
                continue
            els = []
            for k, v in own:
                e2 = list_elements(v)
                els.extend(e2)
                if len(e2) >= 2:
                    els.append(v.lower())

            def knocked(e):
                sides = plus_sides(e)
                return len(sides) >= 2 and any((spells(val, s) for s in sides)) and any((not spells(val, s) and s in standalone for s in sides))
            clean = [e for e in els if spells(val, e) and (not knocked(e))]
            joined = [e for e in els if spells(val, e) and knocked(e)]
            if clean:
                sentences.append(opening + shown + ': it spells the value, satisfied')
            elif q_spells:
                sentences.append(opening + shown + '; ' + qual_remark)
            elif title_speaks:
                sentences.append(opening + shown + "; no key bears the property's name and its title spells the value: satisfied")
            elif joined:
                sentences.append(opening + shown + ': the element pairs the value with another part that other listings bear standalone in this key: fails')
                if by_name and (not q_near):
                    misses.append('%s: its field pairs %s with a part the store also spells standalone there, hence not that value' % (prop, val))
            elif spellers:
                sentences.append(opening + shown + ': it does not spell the value, which others in this category spell in this key: fails (where the store spells the value itself, synonyms and related words are rejected)')
                if by_name and (not q_near) and (not any((touches(val, v) for k, v in own))):
                    misses.append("%s: field %s shows '%s', not '%s', which other listings spell" % (prop, own[0][0], own[0][1][:60], val))
            elif naming:
                sentences.append(opening + shown + ": nobody spells the value in this key yet (%s), but listings of this category name it in titles or texts, so the store owns a term for it and this listing's differing text stays unproven" % smp)
            elif nbear < SYNONYM_MIN:
                sentences.append(opening + shown + ': nobody spells it and too few listings bear the key to tell, so an equivalent term is not yet shown to satisfy it')
            else:
                sentences.append(opening + shown + ': nobody spells it, so an equivalent term may satisfy it')
        return (sentences, rules, misses)

    def ref_note(self):
        refs = [x for x in (self.needs.reading or {}).get('references') or [] if isinstance(x, str)]
        if not refs:
            return ''
        return "REFERENCES: these requirements take their value from another listing; until that listing's record is shown here and states the value, the value stays unknown and no item may fail on it alone: " + '; '.join(refs)

    def failed_now(self, lid):
        ver = self.needs.version
        return any((k[0] == lid and k[1] == ver and v for k, v in self.verdicts.items()))

    def item_check(self, lid, aim, extra=''):
        if self.wire.time_left() < CLOSING + 20:
            return ''
        ver = self.needs.version
        if (lid, ver) in self.meaning_ok:
            return ''
        refs = (self.needs.reading or {}).get('references') or []
        unresolved = any((t['status'] in ('open', 'stuck') for t in self.trackers))
        redirs = self.self_redirects(lid)
        memo = (lid, ver, len(self.records.read) if refs or unresolved else None, bool(extra), len(redirs))
        if memo in self.verdicts:
            return self.verdicts[memo]
        cr = self.chain_reason(lid)
        if cr:
            v = 'An independent check found %s failing: %s Pick another listing.' % (lid, cr)
            self.verdicts[memo] = v
            self.wire.note('item_check', id=lid, by='chain', fit=False)
            return v
        recs = self.records.mentions(lid)
        if not recs:
            return ''
        sent, rules, misses = self.evidence(lid)
        if misses:
            v = 'An independent check found %s failing: %s. Pick another listing.' % (lid, '; '.join(misses[:3]))
            self.verdicts[memo] = v
            self.wire.note('item_check', id=lid, by='evidence', fit=False)
            return v
        parts = [PROMPT_ITEM % PROMPT_READING_RULES, "THE SHOPPER'S REQUEST\n" + self.request]
        cn = self.needs.changes_note()
        if cn:
            parts.append(cn)
        parts.append('REQUIREMENTS AS THEY STAND (stated, revealed, changed)\n' + self.needs.req_text())
        if self.needs.products():
            k = self.product_of_listing(lid)
            units = self.cart.units.get(lid)
            parts.append("This listing belongs to one requested product (its category decides which); judge it only against that product's requirements and unit count%s%s." % (' (product: %s)' % self.needs.products()[k]['product'] if k is not None else '', '; the cart holds %s units of it' % money(units) if units else ''))
        rn = self.ref_block(lid)
        if rn:
            parts.append(rn)
        if extra:
            parts.append(extra)
        for r, p, sent in redirs:
            parts.append('NOTES THAT MAY REDIRECT %s: "%s". Should this sentence send readers to some other listing for this property (look to, take from, defer to), grading reads the property off that other listing, so this listing\'s own field is irrelevant for that requirement, which then holds only when that other record displays the value. Notes citing another listing only as an alternative version on sale elsewhere, or as something this one differs from, change nothing.' % (p, sent[:240]))
        parts.append('ITEM TO JUDGE (aim: %s): %s\n%s' % (aim, lid, '\n'.join(recs)))
        if self.cart.view is not None:
            parts.append('LATEST CART VIEW\n' + compact(self.cart.view)[:3000])
        fl = self.field_list()
        if fl:
            parts.append(fl)
        if sent:
            parts.append("FIELD EVIDENCE (derived by code from this store's records for the item's category: how the store spells each wanted value, and what this item shows)\n" + '\n'.join(sent))
        ll = self.records.label_lines()
        if ll:
            parts.append("RECURRING LABELS (the item's category label must agree with the request's whole phrase, never a narrower neighbour)\n" + '\n'.join(ll))
        ans = self.wire.consult('\n\n'.join(parts), ROLE_CHECKER, CAP_ITEM)
        self.wire.note('item_check', id=lid, aim=aim, answer=ans)
        if not isinstance(ans, dict) or 'fit' not in ans:
            return ''
        if ans.get('fit') is not False:
            self.verdicts[memo] = ''
            self.verified[lid, ver] = self.records.price(lid)
            return ''
        probs = [str(p) for p in ans.get('problems') or [] if str(p).strip()][:5]
        v = 'An independent check judged %s as failing: %s. Move to another fully verified candidate unless you are sure the check is wrong.' % (lid, '; '.join(probs) or 'a requirement is not shown')
        self.verdicts[memo] = v
        return v

    def field_list(self):
        lines = self.records.menu()
        if not lines:
            return ''
        return "THE STORE'S KEY NAMES (separator variants merged; how many opened listings bear each)\n" + self.records.menu_text(lines)

    def foreign(self, lid):
        return False

    def rank_check(self, lid):
        pr = self.needs.priority()
        if pr == 'none':
            return ''
        key = (lid, self.needs.version, self.refusals)
        if key in self.ranked or self.wire.time_left() < CLOSING + 30 or self.steps_left() < 4:
            return ''
        self.ranked.add(key)
        if pr == 'cheapest':
            phrase = 'the cheapest (lowest price)'
        else:
            phrase = 'the best (%s)' % (str((self.needs.reading or {}).get('priority_detail') or '').strip() or 'by the stated measure')
        parts = [PROMPT_RANK % (phrase, PROMPT_READING_RULES), "THE SHOPPER'S REQUEST\n" + self.request]
        cn = self.needs.changes_note()
        if cn:
            parts.append(cn)
        parts.append('CURRENT REQUIREMENTS\n' + self.needs.req_text())
        rn = self.ref_block(lid)
        if rn:
            parts.append(rn)
        if self.needs.products():
            parts.append('Only listings of the same product (category) as the current pick compete; other products never win.')
        cards = self.records.cards(20000, ceiling_id=lid if pr == 'cheapest' else None, first=[lid], exclude=self.off_category())
        parts.append('CURRENT PICK: %s\nLISTINGS SEEN (DETAILS = full record opened, ROW = search row only; a row card on the requested shelf, whatever object its title names, that beats the pick and is not ruled out by its title, price or stock may qualify and should be listed%s)\n%s' % (lid, '; only cheaper ones are shown' if pr == 'cheapest' else '', cards))
        fl = self.field_list()
        if fl:
            parts.append(fl)
        rules = self.evidence(lid)[1]
        if rules:
            parts.append("SHELF RULES (derived by code from this category's records)\n" + '\n'.join(rules))
        ans = self.wire.consult('\n\n'.join(parts), ROLE_CHECKER, CAP_RANK)
        self.wire.note('rank_check', id=lid, answer=ans)
        if not isinstance(ans, dict):
            return ''
        ids = []
        for x in ans.get('better') or []:
            if isinstance(x, (str, int)) and (not isinstance(x, bool)):
                x = str(x)
                if x != lid and x not in self.refused_lines and (x not in self.dead) and (x not in ids):
                    ids.append(x)
            if len(ids) >= 4:
                break
        ids = [x for x in ids if not self.misses(x)]
        self.latest_ranking = ids
        for x in ids:
            if x not in self.flagged:
                self.flagged.append(x)
        if not ids:
            return ''
        reason = str(ans.get('reason') or '').strip()
        return 'A rank check suspects these listings could also qualify and beat the current pick on the priority: %s%s. Check their details, stock and price before buying; if each fails, buy the current pick.' % (', '.join(ids), ' (%s)' % reason if reason else '')

    def rank_verify(self, lid):
        try:
            if self.needs.priority() != 'cheapest':
                return ''
            n = 0
            cap = self.needs.budget()
            for x in self.latest_ranking:
                if n >= 2 or self.wire.time_left() < CLOSING + 30:
                    break
                if x not in self.records.read or (x, self.needs.version) in self.verified or x in self.refused_lines or (x in self.dead) or (self.records.stock_of(x) is False) or self.misses(x):
                    continue
                k = self.product_of_listing(x)
                units = self.units_for(k)
                p = self.records.price(x)
                if p is None or (cap is not None and p * units > cap):
                    continue
                n += 1
                self.item_check(x, 'order')
            return self.cheaper_fit(lid)
        except Exception as exc:
            self.wire.note('rank_verify_error', error=str(exc)[:200])
            return ''

    def cheaper_fit(self, lid):
        if self.needs.priority() != 'cheapest':
            return ''
        p0 = self.records.price(lid)
        if p0 is None:
            return ''
        multi = bool(self.needs.products())
        lab = self.category_of(lid)
        best = None
        for (x, v), _ in sorted(self.verified.items(), key=lambda kv: kv[0][0]):
            if v != self.needs.version or x == lid or x in self.refused_lines or (x in self.dead):
                continue
            if multi and (not lab or self.category_of(x) != lab):
                continue
            p = self.records.price(x)
            if p is None or p >= p0 or self.misses(x):
                continue
            if best is None or p < best[0]:
                best = (p, x)
        return best[1] if best else ''

    def audit(self, early=False):
        read_now = len(self.records.read)
        if self.audit_done and (not (read_now > self.audit_read_count and self.audits < 2)):
            return ''
        if self.wire.time_left() < CLOSING + 30:
            return ''
        prev_set = set(self.audit_read_set)
        had = self.audit_done
        self.audit_done = True
        if not early:
            self.audits += 1
        fresh = sorted(self.records.read - prev_set) if had else []
        cards = self.records.cards(16000, first=fresh, exclude=self.off_category())
        self.audit_read_count = read_now
        self.audit_read_set = set(self.records.read)
        if not cards:
            return ''
        self.audit_reviewed = set(self.records.read)
        parts = [PROMPT_AUDIT % PROMPT_READING_RULES, "THE SHOPPER'S REQUEST\n" + self.request]
        cn = self.needs.changes_note()
        if cn:
            parts.append(cn)
        parts.append('CURRENT REQUIREMENTS\n' + self.needs.req_text())
        rn = self.ref_block(None)
        if rn:
            parts.append(rn)
        parts.append('LISTINGS SEEN (DETAILS = full record opened, ROW = search row only)\n' + cards)
        if fresh:
            parts.append('Opened since the previous review: %s. Judge their new detail cards first, field by field, under the reading rules.' % ', '.join(fresh))
        fl = self.field_list()
        if fl:
            parts.append(fl)
        ans = self.wire.consult('\n\n'.join(parts), ROLE_AUDITOR, CAP_AUDIT)
        self.wire.note('decline_audit', early=early, answer=ans)
        if not isinstance(ans, dict):
            return ''
        self.audit_answered = True
        fits = [str(x) for x in ans.get('possible_fits') or [] if isinstance(x, (str, int)) and (not isinstance(x, bool))][:5]
        fits = [x for x in fits if not self.misses(x)]
        self.audit_fits = fits
        unopened = []
        for x in ans.get('unopened') or []:
            x = str(x)
            if x in self.records.known_set and x not in self.records.read and (x not in unopened):
                unopened.append(x)
        unopened = unopened[:6]
        pages = self.ready_pages(2) if self.steps_left() > 5 else []
        if pages:
            unopened = unopened[:4]
        for x in fits + unopened:
            if x not in self.flagged:
                self.flagged.append(x)
        for p in pages:
            if p not in self.flagged_pages:
                self.flagged_pages.append(p)
        if not (fits or unopened or pages):
            return ''
        reason = str(ans.get('reason') or '').strip()
        out = []
        if fits:
            out.append('A separate check believes these listings may qualify: %s%s.' % (', '.join(fits), ' (%s)' % reason if reason else ''))
        if pages:
            out.append('These searches include wanted values and found more matches than were opened, so fits may hide on pages never fetched; fetch them first (one step returns a whole page) and open their affordable rows of the wanted kind: %s.' % '; '.join(('%s %s' % (self.role('search'), p) for p in pages)))
        if unopened:
            out.append('These appeared only as result rows and their fields are unexamined; open their details first, several per reply: %s.' % ', '.join(unopened))
        out.append('Vet each of them (details, price, stock), buy whichever qualifies, and decline only if every one of them truly fails.')
        return ' '.join(out)

    def roster(self):
        cap = self.needs.budget()
        out = []
        for x in self.flagged:
            if x in out or x not in self.records.known_set or x in self.records.read or (x in self.refused_lines):
                continue
            p = self.records.price(x)
            if cap is not None and p is not None and (p * self.needs.quantity() > cap):
                continue
            out.append(x)
        return out

    def key_arg(self, tool):
        sch = self.shapes.get(tool) or {}
        props = sch.get('properties') if isinstance(sch.get('properties'), dict) else {}
        req = [r for r in sch.get('required') or [] if isinstance(r, str)]
        texts = [k for k, s in props.items() if isinstance(s, dict) and (s.get('type') == 'string' or (isinstance(s.get('type'), list) and 'string' in s['type'])) and ('enum' not in s)]
        cand = [k for k in texts if k in req]
        key = cand[0] if cand else texts[0] if len(texts) == 1 else None
        if key is None or any((r != key for r in req)):
            return None
        return key

    def lookup_tools(self):
        busy = {self.role(r) for r in ('search', 'message', 'cart', 'cart_view', 'order', 'decline')}
        return [n for n in self.shapes if n not in busy and self.key_arg(n)]

    def ids_named(self, args):
        out = []
        for v in (args or {}).values():
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if isinstance(x, (str, int)) and (not isinstance(x, bool)):
                    x = str(x).strip()
                    if x in self.records.known_set and x not in out:
                        out.append(x)
        return out

    def last_action(self, tool):
        for rec in reversed(self.log):
            if rec.get('kind') == 'env' and rec.get('name') == tool:
                args = rec.get('args') or {}
                if any((isinstance(v, str) and v in self.records.known_set for v in args.values())):
                    return args
        return None

    def details_call(self, lid, tool=None):
        tool = tool or self.role('details')
        if not tool:
            return None
        prev = self.last_action(tool)
        if prev is not None:
            return {k: lid if isinstance(v, str) and v in self.records.known_set else v for k, v in prev.items()}
        key = self.key_arg(tool)
        if key:
            return {key: lid}
        sch = self.shapes.get(tool) or {}
        props = sch.get('properties') if isinstance(sch.get('properties'), dict) else {}
        texts = [k for k, s in props.items() if isinstance(s, dict) and s.get('type') == 'string' and ('enum' not in s)]
        req = [k for k in texts if k in (sch.get('required') or [])]
        if req:
            return {req[0]: lid}
        if len(texts) == 1:
            return {texts[0]: lid}
        return None

    def multi_tool(self):
        if self.multi is not None:
            return self.multi if self.multi and (not self.multi_retired) else None
        self.multi = ()
        try:
            busy = {self.role(r) for r in ('search', 'message', 'cart', 'cart_view', 'order', 'decline')}
            for t in self.tools:
                f = t['function']
                name = f.get('name')
                if name in busy:
                    continue
                txt = (split_camel(name or '').replace('_', ' ') + ' ' + str(f.get('description') or '')).lower()
                if re.search('\\b(?:remov|delet|checkout|check out|cart|basket|bag|purchas|buy|order|add|wishlist|list add|clear)', txt):
                    continue
                sch = self.shapes.get(name) or {}
                props = sch.get('properties') if isinstance(sch.get('properties'), dict) else {}
                arrs = [k for k, s in props.items() if isinstance(s, dict) and s.get('type') == 'array' and isinstance(s.get('items'), dict) and (s['items'].get('type') == 'string') and ('enum' not in s['items'])]
                if len(arrs) != 1:
                    continue
                if any((r != arrs[0] for r in sch.get('required') or [])):
                    continue
                mx = props[arrs[0]].get('maxItems')
                if not is_num(mx) or mx < 2:
                    continue
                self.multi = (name, arrs[0], int(min(mx, 10)))
                break
        except Exception:
            self.multi = ()
        return self.multi if self.multi and (not self.multi_retired) else None

    def lookup_args(self, tool, lid):
        if not tool:
            return None
        if tool == self.role('details'):
            return self.details_call(lid, tool)
        prev = self.last_action(tool)
        if prev is not None:
            return {k: lid if isinstance(v, str) and v in self.records.known_set else v for k, v in prev.items()}
        key = self.key_arg(tool)
        return {key: lid} if key else None

    def learn_answer(self, name, args, obs):
        newly = []
        try:
            t = self.tool_tally.setdefault(name, {'calls': 0, 'named': 0, 'avail': 0, 'map': 0, 'price': 0})
            t['calls'] += 1
            if isinstance(obs, dict) and obs.get('error'):
                return newly
            named = set(self.ids_named(args))
            is_cart = name in (self.role('cart'), self.role('cart_view'))
            flags = set()
            for d in dicts_in(obs):
                rid = None
                for k, v in d.items():
                    if k.lower() == 'id' and id_like(v):
                        rid = v
                        break
                if rid is None:
                    for k, v in d.items():
                        if k.lower().endswith('id') and id_like(v):
                            rid = v
                            break
                if rid is None:
                    continue
                st = record_stock(d)
                src = 'lookup' if rid in named or is_cart else 'row'
                if st is not None:
                    self.stock_fields_seen = True
                    other = self.look_avail if src == 'row' else self.row_avail
                    mine = self.row_avail if src == 'row' else self.look_avail
                    if rid in other and other[rid][0] == self.change_counter and (other[rid][1] != st):
                        self.rows_lag = True
                    if src == 'row' and rid in mine and (mine[rid][1] != st):
                        self.rows_live = True
                    mine[rid] = (self.change_counter, st)
                    if st is False:
                        self.sold_out_seen = True
                        if rid not in self.dead and rid in self.cart.units:
                            newly.append(rid)
                        if rid not in self.dead or src == 'lookup':
                            self.dead[rid] = src if rid not in self.dead else 'lookup' if src == 'lookup' else self.dead[rid]
                    elif src == 'lookup' or self.dead.get(rid) != 'lookup':
                        self.dead.pop(rid, None)
                if rid in named:
                    flags.add('named')
                    if st is not None:
                        flags.add('avail')
                    if prop_map(d) is not None:
                        flags.add('map')
                    if record_price(d) is not None:
                        flags.add('price')
            for f in flags:
                t[f] += 1
        except Exception as exc:
            self.wire.note('learn_error', error=str(exc)[:200])
        return newly

    def stock_tool(self):
        looks = self.lookup_tools()
        best = None
        for n in looks:
            a = self.tool_tally.get(n, {}).get('avail', 0)
            if a <= 0:
                continue
            rank = (a, n == self.role('view'), n == self.role('details'), n)
            if best is None or rank > best[0]:
                best = (rank, n)
        if best:
            return best[1]
        for n in (self.role('view'), self.role('details')):
            if n and n in looks:
                t = self.tool_tally.get(n, {})
                if not (t.get('named', 0) >= 2 and t.get('avail', 0) == 0):
                    return n
        for n in looks:
            if self.tool_tally.get(n, {}).get('named', 0) == 0:
                return n
        return None

    def stale(self, lid):
        return self.line_read.get(lid, -1) < self.change_counter

    def substitute(self, line):
        lab = self.category_of(line)
        multi = bool(self.needs.products())
        cap = self.needs.budget()
        units = max(1, self.cart.units.get(line, 1))
        others = 0.0
        for l, u in self.cart.units.items():
            if l != line and self.records.price(l) is not None:
                others += self.records.price(l) * u
        best = None
        for x in self.verified_now():
            if x == line or x in self.cart.units or self.misses(x):
                continue
            xl = self.category_of(x)
            if lab and (multi and xl != lab or (not multi and xl and (xl != lab))):
                continue
            p = self.records.price(x)
            if cap is not None and p is not None and (p * units + others > cap):
                continue
            key = (p if p is not None else 1e+18, x)
            if best is None or key < best:
                best = key
        if best:
            return (best[1], [])
        if multi and (not lab):
            return (None, [])
        rows = []
        tab = self.table()
        for x, row in tab.items():
            if x == line or x in self.dead or x in self.refused_lines or (x in self.cart.units):
                continue
            if row['stock'] is not True or row['price'] is None:
                continue
            if lab and row['label'] != lab:
                continue
            if cap is not None and row['price'] * units + others > cap:
                continue
            if self.failed_now(x) or (x in self.records.read and self.misses(x)):
                continue
            rows.append((0 if x in self.records.read else 1, row['price'], x))
        rows.sort()
        if self.needs.priority() == 'cheapest' and rows:
            un = [r for r in rows if r[0] == 1]
            op = [r for r in rows if r[0] == 0]
            if un and (not op or un[0][1] < min((r[1] for r in op))):
                rows.remove(un[0])
                rows.insert(0, un[0])
        top = rows[:3]
        swap = [x for _, _, x in top if x not in self.records.read]
        if swap:
            self.wanted_hint = swap[0]
        return (None, ['%s (%s)%s' % (x, money(p), ' [record shows every asked-for value]' if o == 0 else '') for o, p, x in top])

    def dead_remark(self, lines):
        self.event = True
        out = []
        for line in lines:
            sub, rows = self.substitute(line)
            s = '%s is now unavailable per its latest record; any purchase containing it will be refused, so remove it.' % line
            if sub:
                s += ' Put in %s instead (an item check judged it to satisfy every active need%s), refresh its availability and price, then buy.' % (sub, '; it is also the lowest-priced such listing' if self.needs.priority() == 'cheapest' else '')
            elif rows:
                s += ' No further verified fit exists, so inspect these cheapest available rows of that shelf under the cap (open an unopened one first), put in the earliest fitting one, refresh availability and price, then buy: %s.' % '; '.join(rows)
            else:
                s += ' Look for a different fitting listing.'
            out.append(s)
        return ' '.join(out)

    def scripted_reads(self, ids, sentence, tool=None):
        calls = []
        mt = self.multi_tool()
        if mt and tool is None and (len(ids) >= 2):
            for i in range(0, len(ids), mt[2]):
                calls.append((mt[0], {mt[1]: ids[i:i + mt[2]]}))
        else:
            reader = tool or self.role('details')
            for lid in ids:
                a = self.lookup_args(reader, lid)
                if a is not None:
                    calls.append((reader, a))
        if not calls:
            return None
        props = []
        for i, (n, a) in enumerate(calls):
            props.append({'id': 'code_read_%d_%d' % (i, self.wire.step), 'name': n, 'args': json.dumps(a)})
        self.chat.append({'role': 'assistant', 'content': sentence, 'tool_calls': [{'id': p['id'], 'type': 'function', 'function': {'name': p['name'], 'arguments': p['args']}} for p in props]})
        self.covering = True
        try:
            res = self.dispatch(props, scripted=True)
        finally:
            self.covering = False
        return res

    def enqueue(self, ids, purpose, sentence=None, tool=None, recheck=False, floor=None):
        for x in ids:
            if x not in self.queue:
                self.queue.append(x)
        self.queue_meta = {'purpose': purpose, 'sentence': sentence, 'tool': tool, 'recheck': recheck, 'floor': floor}

    def run_queue(self):
        ids = list(self.queue)
        meta = dict(self.queue_meta or {})
        self.queue, self.queue_meta = ([], {})
        if not ids and self.postput and (not (self.wire.finished or self.wire.blocked)):
            ids = [x for x in self.postput if x in self.cart.units]
            meta = {'purpose': 'cart', 'tool': self.stock_tool(), 'recheck': True, 'postput': True, 'sentence': 'I check the stock of what I just put in the cart.'}
            self.postput = []
        if not ids or self.wire.finished or self.wire.blocked:
            return False
        ids = list(dict.fromkeys(ids))
        purpose = meta.get('purpose') or 'other'
        recheck = bool(meta.get('recheck'))
        sentence = meta.get('sentence') or ('I re-read the cart lines before buying.' if recheck and purpose == 'cart' else 'I open the listings the checks named before declining.')
        mt = self.multi_tool()
        if purpose in ('menu', 'quota', 'walk') and self.narrowing() and (not self.wire.mute):
            try:
                keep = self.reserve() if self.orders_tried == 0 else REFUSAL_RESERVE
                room = self.steps_left() - keep - 1
                need = 1 if mt and len(ids) >= 2 else len(ids)
                if room < need and (self.cart.units or self.verified_now()):
                    ids = [] if mt and len(ids) >= 2 else ids[:max(0, room)]
                    self.wire.note('read_skip', purpose=purpose)
                    if not ids:
                        return False
            except Exception as exc:
                self.wire.note('queue_error', error=str(exc)[:200])
        if purpose == 'cart':
            ids = [x for x in ids if x in self.cart.units and self.stale(x) or (x not in self.cart.units and (not meta.get('postput')))]
            ids = [x for x in ids if x in self.cart.units]
            if not ids:
                return False
        tool = meta.get('tool')
        use_multi = mt and len(ids) >= 2 and (not (purpose == 'cart' and (not self.multi_proven))) and (not (purpose == 'cart' and self.tool_tally.get(mt[0], {}).get('avail', 0) == 0))
        before = len(self.records.obs)
        res = self.scripted_reads(ids, sentence, tool=None if use_multi else tool or None) if use_multi else self.scripted_reads_single(ids, sentence, tool)
        if use_multi and (not self.multi_proven):
            got = 0
            for ob in self.records.obs[before:]:
                for d in dicts_in(ob):
                    oid = own_id(d)
                    if oid in ids and (prop_map(d) is not None or len([v for v in d.values() if _brief(v)]) >= 5):
                        got += 1
            if got >= 1 and got * 2 >= len(ids):
                self.multi_proven = True
            else:
                self.multi_retired = True
                self.wire.note('multi_retired', tool=mt[0])
                if res:
                    self.post(res)
                limit = {'menu': 0, 'quota': 2, 'walk': 2}.get(purpose, 3)
                keep = meta.get('floor') or (self.reserve() if purpose in ('menu', 'quota') else 3)
                limit = min(limit, self.steps_left() - keep - 1)
                if limit > 0 and self.role('details'):
                    res = self.scripted_reads_single(ids[:limit], sentence, None)
                    if not res:
                        return True
                else:
                    return True
        if not res:
            return False
        if not recheck and self.wire.time_left() > CLOSING + 30:
            msg = self.audit()
            if msg:
                cid, text = res[-1]
                res[-1] = (cid, text + '\n[review] ' + msg)
        self.post(res)
        return True

    def scripted_reads_single(self, ids, sentence, tool):
        calls = []
        reader = tool or self.role('details')
        for lid in ids:
            a = self.lookup_args(reader, lid)
            if a is not None:
                calls.append((reader, a))
        if not calls:
            return None
        props = [{'id': 'code_read_%d_%d' % (i, self.wire.step), 'name': n, 'args': json.dumps(a)} for i, (n, a) in enumerate(calls)]
        self.chat.append({'role': 'assistant', 'content': sentence, 'tool_calls': [{'id': p['id'], 'type': 'function', 'function': {'name': p['name'], 'arguments': p['args']}} for p in props]})
        self.covering = True
        try:
            return self.dispatch(props, scripted=True)
        finally:
            self.covering = False

    def chosen_line(self):
        return list(self.cart.units)[-1] if self.cart.units else None

class QuestionsPart:

    def msg_arg(self):
        tool = self.role('message')
        if not tool:
            return None
        sch = self.shapes.get(tool) or {}
        props = sch.get('properties') if isinstance(sch.get('properties'), dict) else {}
        req = [r for r in sch.get('required') or [] if isinstance(r, str)]
        cand = [k for k, s in props.items() if isinstance(s, dict) and (s.get('type') == 'string' or (isinstance(s.get('type'), list) and 'string' in s['type'])) and ('enum' not in s)]
        pick = next((k for k in req if k in cand), None) or (cand[0] if len(cand) == 1 else None)
        if not pick or any((r != pick for r in req)):
            return None
        return pick

    def build_msg(self, text):
        arg = self.msg_arg()
        if not arg:
            return None
        args, faults = check_args({arg: text}, self.shapes.get(self.role('message')) or {})
        return None if faults else args

    def concerns(self, text):
        return self.product_index(text) if self.needs.products() else None

    def plain(self, label):
        prop = label.split('(')[0].strip()
        q = ptag(label) or (self.needs.product_name() if self.needs.product_name() != 'item' else 'item')
        if self.needs.products() and (not ptag(label)):
            q = 'item'
        return 'Which %s would you like for the %s?' % (prop, q)

    def samples_for(self, name, k):
        lines = self.menu_for(k)
        for n, c, s in lines:
            if strict_same(n, name):
                return [' '.join(words_of(x)) for x in s]
        return []

    def settles(self, req, samples=None):
        val = pvalue(req)
        if is_reference(val) or re.search('\\b(?:redirect|sold by|same as|as shown|page)\\b', val.lower()):
            return True
        if not positive_value(val, pname(req)):
            return False
        refs = [x for x in (self.needs.reading or {}).get('references') or [] if isinstance(x, str)]
        story_text = self.request + ' ' + ' '.join(self.needs.history)
        if req not in self.needs.resolved and story_guess(req, story_text, refs):
            return False
        if samples is not None and (not re.search('\\d', val)):
            pool = ' '.join(samples)
            pw = re.findall('[a-z]+', pool)
            ok = False
            for w in content_words(val):
                stem = w[:-1] if len(w) > 4 else w
                if re.search('\\b' + re.escape(stem), pool) or (len(w) >= 4 and any((len(p) >= 4 and edit_le(w, p, 1) for p in pw))):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def is_open(self, name, k=None, by_samples=False):
        try:
            nm = name.split('(')[0].strip()
            if not nm or non_property(nm):
                return False
            multi = bool(self.needs.products())

            def mine(text, tagged_only=False):
                if k is None or not multi:
                    return True
                c = self.concerns(text)
                return c == k or (c is None and (not tagged_only))
            for e in self.qlog:
                if 'not answered' in e.get('tags', ''):
                    continue
                for part in e['label'].split(';'):
                    if mine(part) and (strict_same(part, nm) or self.same_field(part, nm, k)):
                        return False
            samples = self.samples_for(nm, k) if by_samples else None
            for r in self.needs.binding():
                if not mine(r):
                    continue
                rp = pname(r)
                if strict_same(rp, nm):
                    if self.settles(r, samples):
                        return False
                    continue
                val = pvalue(r)
                if loose_same(rp, nm):
                    sm = samples if samples is not None else self.samples_for(nm, k)
                    if positive_value(val, rp) and sm and value_seen(val, sm):
                        return False
                    continue
                if samples and (not re.search('\\d', val)) and positive_value(val, rp):
                    cws = content_words(val)
                    pool = ' '.join(samples)
                    if cws and all((re.search('\\b' + re.escape(w), pool) for w in cws)):
                        return False
            for f in self.needs.dontmind:
                if mine(f, True) and (strict_same(f, nm) or self.same_field(f, nm, k)):
                    return False
            return True
        except Exception:
            return True

    def same_field(self, a, b, k=None):
        if strict_same(a, b):
            return True
        if loose_same(a, b):
            sa, sb = (set(self.samples_for(pname(a), k)), set(self.samples_for(pname(b), k)))
            return bool(sa & sb)
        return False

    def hints_for(self, k=None):
        out = []
        for h in (self.needs.reading or {}).get('hints') or []:
            if not isinstance(h, str) or ':' not in h:
                continue
            if k is not None and self.needs.products():
                c = self.concerns(h.split(':')[0]) if ptag(h) else self.concerns(h)
                if c is not None and c != k:
                    continue
            out.append(h)
        return out

    def plan_open(self, name, k=None):
        try:
            for h in self.hints_for(k):
                if strict_same(pname(h), name):
                    return self.is_open(name, k, by_samples=True)
            return self.is_open(name, k)
        except Exception:
            return self.is_open(name, k)

    def keyless(self, name, k=None):
        if self.product_reads(k) < 2:
            return False
        if not any((strict_same(pname(h), name) for h in self.hints_for(k))):
            return False
        return not any((strict_same(n, name) or loose_same(n, name) for n, _, _ in self.menu_for(k)))

    def open_hints(self, k=None):
        out = []
        for h in self.hints_for(k):
            prop = pname(h)
            if not prop:
                continue
            if not self.plan_open(prop, k):
                continue
            asked_inside = False
            for e in self.qlog:
                if 'not answered' in e.get('tags', ''):
                    continue
                for part in e['label'].split(';'):
                    pw = set(content_words(pname(part)))
                    hw = set(content_words(prop))
                    if pw and hw and (pw <= hw or hw <= pw):
                        asked_inside = True
            if not asked_inside:
                out.append(h)
        return out

    def menu_for(self, k=None):
        try:
            if self.needs.products():
                homes = self.home_labels()
                hs = homes[k] if k is not None and k < len(homes) else set()
                read = self.records.categories_read()
                for r in read:
                    if r in hs:
                        lines = self.records.menu(r)
                        return self.trim_menu(lines, r)
                return []
            ol = self.own_label()
            if ol:
                return self.trim_menu(self.records.menu(ol), ol)
            return self.trim_menu(self.records.menu(), None)
        except Exception:
            return []

    def trim_menu(self, lines, cat):
        if self.records.rich_reads(cat) >= 4:
            t = [x for x in lines if x[1] >= 2]
            return t or lines
        return lines

    def product_reads(self, k=None):
        if self.needs.products():
            homes = self.home_labels()
            hs = homes[k] if k is not None and k < len(homes) else set()
            return sum((self.records.rich_reads(h) for h in hs))
        ol = self.own_label()
        return self.records.rich_reads(ol if ol else None)

    def question_step(self, urged=False):
        msg_tool = self.role('message')
        if not msg_tool or self.sent_msgs >= self.allowance_in_force():
            self.code_asks = False
            return False
        ol = self.own_label() if not self.needs.products() else ''
        reads = self.records.rich_reads(ol if ol else None)
        pressing = bool((self.needs.reading or {}).get('ask_first')) and (not self.qlog)
        thin = reads < 2 and (not pressing) and (not urged)
        if thin and (self.wire.step < 5 or (self.wire.step < 8 and self.steps_left() > 20)):
            return False
        if self.steps_left() <= max(10, self.reserve()) or self.wire.time_left() < CLOSING + 40:
            self.code_asks = False
            return False
        if self.plan_wait is not None and reads <= self.plan_wait and (not urged):
            return False
        left_msgs = self.allowance_in_force() - self.sent_msgs
        if not self.needs.products() and self.last_q_count >= reads and (not urged) and (not pressing) and (self.steps_left() > self.reserve() + 2 * left_msgs + 14) and (self.wire.step - self.last_q_step < 4):
            return False
        if self.multi_tool():
            if self.menu_pre_read():
                return True
        elif not self.qlog and (not self.sent_msgs) and (not pressing) and (not urged):
            if self.pre_reads():
                return True
        entries = []
        retry = None
        self.retry_queue = [x for x in self.retry_queue if self.fresh(x)]
        if self.retry_queue:
            retry = self.retry_queue[0]
            entries = [{'label': retry, 'text': self.retry_text.get(retry) or self.plain(retry), 'second': ''}]
        else:
            plan = self.plan_call(left_msgs)
            if plan is None:
                if self.wire.fail_kind == '' and self.wire.misses > 0:
                    self.plan_wait = reads
                    return False
                plan = []
            entries = self.clean_plan(plan)
            if self.duty_waiting():
                entries = [e for e in entries if not DUTY_BASKET.search(e.get('text') or '')]
            if self.needs.products():
                entries = self.rotate(entries)
                if entries == 'reads':
                    return True
        if not entries:
            if reads < 2 and self.steps_left() > 12 and (not urged):
                self.plan_wait = reads
                return False
            self.empty_plans += 1
            if self.steps_left() > 12 and self.empty_plans == 1:
                self.plan_wait = reads
                return False
            self.code_asks = False
            return False
        for e in entries[:2]:
            label, text, second = (e['label'], e['text'], e.get('second') or '')
            k = self.concerns(label) if self.needs.products() else None
            if self.needs.products() and ptag(label):
                tw = [w for w in re.findall('[a-z]+', ptag(label)) if len(w) >= 3]
                if tw and (not any((_depl(w) in text.lower() or w.rstrip('s') in text.lower() for w in tw))):
                    text = self.plain(label)
            if retry is None and (not (pressing and (not self.qlog))) and self.second_subject(text, label, second, k):
                text = self.plain(label)
            files = label
            split_second = ''
            if second and left_msgs == 1 and (not self.pairs_off) and self.pair_fit(label, second, k) and self.hint_backed(pname(label), k):
                text = 'Which %s would you like for the %s, and which %s?' % (pname(label), ptag(label) or self.needs.product_name(), pname(second))
                files = '%s; %s' % (label, second)
            elif second:
                split_second = second
            args = self.build_msg(text)
            if args is None:
                continue
            return self.send_question(files, text, args, retry, split_second)
        if reads < 2 and self.steps_left() > 12 and (not urged):
            self.plan_wait = reads
            return False
        self.empty_plans += 1
        if self.steps_left() > 12 and self.empty_plans == 1:
            self.plan_wait = reads
            return False
        self.code_asks = False
        return False

    def menu_pre_read(self):
        try:
            mt = self.multi_tool()
            if not mt or self.menu_rounds >= 2 or len(self.qlog) > 1 or (self.wire.time_left() < CLOSING + 60):
                return False
            rnd = self.menu_rounds + 1
            if self.steps_left() <= (20 if rnd == 1 else 25):
                return False
            multi = bool(self.needs.products())
            names = self.product_names()
            ids = []
            zero_need = 0
            for k in range(len(names)):
                kk = k if multi else None
                homes = self.home_labels()
                hs = homes[k] if k < len(homes) else set()
                cov = self.product_reads(kk) if hs else 4
                if rnd == 1:
                    want = (2 if multi else 4) if cov == 0 else 4 - cov if cov < 4 else 0
                    if cov == 0:
                        zero_need += want
                else:
                    want = 4 - cov if cov < 3 else 0
                ids += self.menu_rows(kk, want, ids + list(self.menu_picked))
            cap = mt[2] * (2 if rnd == 1 and zero_need > mt[2] and (self.steps_left() > 25) else 1)
            ids = ids[:cap]
            if not ids:
                if rnd == 1:
                    return False
                self.menu_rounds = 2
                return False
            self.menu_rounds = rnd
            self.menu_picked |= set(ids)
            self.enqueue(ids, 'menu', 'I open some listings of each wanted product before asking anything, to learn which fields they carry.' if rnd == 1 else 'I open a few more listings because the first ones had sparse fields.', recheck=True)
            return self.run_queue()
        except Exception as exc:
            self.wire.note('menu_pre_read_error', error=str(exc)[:200])
            return False

    def pre_reads(self):
        try:
            if self.first_reads_tried or not self.role('details'):
                return False
            self.first_reads_tried = True
            room = self.steps_left() - self.reserve() - 2 * (self.allowance_in_force() - self.sent_msgs) - 8
            if room <= 0:
                return False
            picks = []
            names = self.product_names()
            order = sorted(range(len(names)), key=lambda k: self.product_reads(k if self.needs.products() else None))
            per = {}
            for k in order:
                kk = k if self.needs.products() else None
                have = self.product_reads(kk)
                if have >= 2:
                    continue
                per[k] = self.menu_rows(kk, 2 - have, picks)
            queue = []
            for rnd in range(2):
                for k in order:
                    if k in per and len(per[k]) > rnd:
                        queue.append(per[k][rnd])
            queue = queue[:room]
            if not queue:
                return False
            self.enqueue(queue, 'menu', 'I open a couple of listings per product to see which fields they carry before I message the shopper.', recheck=True)
            self.wire.note('first_reads', ids=queue)
            return self.run_queue()
        except Exception as exc:
            self.wire.note('pre_reads_error', error=str(exc)[:200])
            return False

    def menu_rows(self, k, n, taken):
        if n <= 0:
            return []
        homes = self.home_labels()
        if self.needs.products():
            hs = homes[k] if k is not None and k < len(homes) else set()
        else:
            hs = homes[0] if homes else set()
        if not hs:
            return []
        cap = self.needs.budget()
        units = self.units_for(k)
        out = []
        sold = {x for x, s in self.row_avail.items() if s[1] is False} | set(self.dead)
        for lid, row in self.table().items():
            if lid in self.records.read or lid in taken or lid in out or (lid in sold) or (lid in self.refused_lines):
                continue
            if row['label'] not in hs or row['stock'] is False:
                continue
            if cap is not None and row['price'] is not None and (row['price'] * units > cap):
                continue
            out.append(lid)
            if len(out) >= n:
                break
        return out

    def plan_call(self, left_msgs):
        parts = [PROMPT_PLAN % max(1, left_msgs) + PROMPT_PLAN_ONE % max(1, left_msgs), 'REQUEST\n' + self.request, 'REQUIREMENTS SO FAR\n' + self.needs.req_text()]
        if self.qlog:
            parts.append('EARLIER QUESTIONS AND WHAT CAME BACK\n' + '\n'.join(('- %s: "%s" -> %s%s' % (e['label'], e['text'], e['reply'][:400], ' [%s]' % e['tags'] if e.get('tags') else '') for e in self.qlog)))
        af = (self.needs.reading or {}).get('ask_first') or []
        if af:
            parts.append('POINTS TO LEARN FROM THE SHOPPER FIRST\n' + '; '.join((str(x) for x in af)))
        notes = self.planner_notes(None)
        if notes:
            parts.append(notes)
        hints = self.open_hints(None)[:8]
        if hints:
            parts.append('STORY DETAILS PAIRED WITH THE PROPERTY EACH PROBABLY CONCERNS (not requirements; hidden needs usually sit on one of these)\n' + '\n'.join((h + (' [no key]' if self.keyless(pname(h)) else '') for h in hints)))
        ps = self.needs.products()
        if ps:
            parts.append('PRODUCTS\n' + '\n'.join(('%d) %s x %d' % (i + 1, p['product'], p['quantity']) for i, p in enumerate(ps))))
            blocks = []
            homes = self.home_labels()
            read = self.records.categories_read()
            cats = [r for r in read if any((r in hs for hs in homes))] or read
            for c in cats[:4]:
                lines = self.trim_menu(self.records.menu(c), c)
                if lines:
                    blocks.append('CATEGORY %s\n%s' % (c, self.records.menu_text(lines)))
            parts.append('FIELDS SEEN SO FAR IN LISTING DETAILS (name, listing count, sample values)\n' + ('\n'.join(blocks) if blocks else self.records.menu_text(self.records.menu()) or 'none seen yet'))
        else:
            lines = self.menu_for(None)
            parts.append('FIELDS SEEN SO FAR IN LISTING DETAILS (name, listing count, sample values)\n' + (self.records.menu_text(lines) or 'none seen yet'))
        ans = self.wire.consult('\n\n'.join(parts), ROLE_PLANNER, CAP_PLANNER)
        self.wire.note('plan', answer=ans)
        if not isinstance(ans, dict):
            return None
        qs = ans.get('questions')
        return [q for q in qs if isinstance(q, dict)] if isinstance(qs, list) else []

    def planner_notes(self, k):
        out = []
        halves = [h for h in self.open_pairs if (k is None or self.concerns(h) in (k, None)) and (not any((e['label'] == h for e in self.qlog))) and self.plan_open(pname(h), k)][-3:]
        if halves:
            out.append("SECOND HALVES OF EARLIER PAIRS THE ANSWERS SKIPPED (still open; each was the plan's next choice)\n" + '\n'.join(halves))
        wq = [w for w in self.wanted_questions if k is None or self.concerns(w) in (k, None)][-3:]
        if wq:
            out.append("QUESTIONS THE ASSISTANT WANTED TO SEND AFTER READING RECORDS (an open property such a text names that is also a menu key goes first, asked under that key's exact name)\n" + '\n'.join(wq))
        return '\n\n'.join(out)

    def fresh(self, label):
        nm = pname(label)
        if not nm or non_property(nm) or label.strip().lower() in self.spent_labels:
            return False
        for e in self.qlog:
            for part in e['label'].split(';'):
                if part.strip().lower() == label.strip().lower():
                    return False
        k = self.concerns(label) if self.needs.products() else None
        if self.keyless(nm, k):
            return False
        return self.plan_open(nm, k)

    def clean_plan(self, plan):
        out = []
        for q in plan:
            text = str(q.get('question') or '').strip()
            if not text:
                continue
            prop = str(q.get('property') or '').strip()
            second = str(q.get('second') or '').strip()
            if 'none' in second.lower() or second in ('-', '') or 'n/a' in second.lower():
                second = ''
            if prop and self.fresh(prop):
                sec = second if second and self.fresh(second) else ''
                k = self.concerns(prop) if self.needs.products() else None
                if sec and self.keyless(pname(sec), k):
                    out.append({'label': prop, 'text': self.plain(prop), 'second': ''})
                else:
                    out.append({'label': prop, 'text': text, 'second': sec})
            elif second and self.fresh(second):
                out.append({'label': second, 'text': self.plain(second), 'second': ''})
        return out

    def rotate(self, entries):
        ps = self.needs.products()
        if not ps:
            return entries
        answered = [0] * len(ps)
        msgs = [0] * len(ps)
        for e in self.qlog:
            k = self.concerns(e['label'])
            if k is None:
                k = self.concerns(e['text'])
            if k is None:
                continue
            msgs[k] += 1
            if 'not answered' not in e.get('tags', ''):
                answered[k] += 1
        disc = [0] * len(ps)
        for n in self.needs.revealed:
            k = self.concerns(n)
            if k is not None:
                disc[k] += 1
        if not any(answered) and entries and (self.concerns(entries[0]['label']) is None):
            return entries
        hint_led = [any((not self.keyless(pname(h), k) for h in self.open_hints(k))) for k in range(len(ps))]

        def rank(k):
            other_none = any((answered[j] == 0 for j in range(len(ps)) if j != k))
            return (1 if disc[k] and other_none else 0, 0 if hint_led[k] else 1, answered[k] + disc[k], msgs[k] - answered[k], disc[k])
        levels = sorted(set((rank(k) for k in range(len(ps)))))
        for lv in levels:
            due = [k for k in range(len(ps)) if rank(k) == lv]
            for i, e in enumerate(entries):
                if self.concerns(e['label']) in due:
                    return [e] + entries[:i] + entries[i + 1:]
            for e in entries:
                if e.get('second') and self.concerns(e['second']) in due:
                    s = e['second']
                    return [{'label': s, 'text': self.plain(s), 'second': ''}] + entries
            for k in sorted(due, key=lambda j: (0 if hint_led[j] else 1, j)):
                pick = self.own_question(k, allow_fallback=answered[k] == 0)
                if pick == 'reads':
                    return 'reads'
                if pick:
                    return [pick] + entries
        return entries

    def own_question(self, k, allow_fallback):
        ps = self.needs.products()
        pname_k = ps[k]['product']
        homes = self.home_labels()
        hs = homes[k] if k < len(homes) else set()
        read = [r for r in self.records.categories_read() if r in hs]
        if not read:
            if hs and (not getattr(self, '_reads_for_%d' % k, False)):
                rows = self.menu_rows(k, min(4, (self.multi_tool() or (0, 0, 2))[2]), [])
                if rows and (self.role('details') or self.multi_tool()) and (self.steps_left() > 12):
                    setattr(self, '_reads_for_%d' % k, True)
                    self.enqueue(rows, 'menu', 'I open some listings of the %s to see which fields they carry before asking about it.' % pname_k, recheck=True)
                    return 'reads' if self.run_queue() else None
            return None

        def good(lbl):
            return self.fresh(lbl) and (not any((strict_same(pname(lbl), pname(p)) for e in self.qlog for p in e['label'].split(';') if self.concerns(p) == k)))
        if not self.product_only_used and self.wire.time_left() >= CLOSING + 40:
            self.product_only_used = True
            lines = self.menu_for(k)
            if lines:
                parts = [PROMPT_PLAN % max(1, min(3, self.allowance_in_force() - self.sent_msgs)), 'REQUEST\n' + self.request, 'REQUIREMENTS SO FAR\n' + self.needs.req_text()]
                if self.qlog:
                    parts.append('EARLIER QUESTIONS\n' + '\n'.join(('- %s: "%s" -> %s' % (e['label'], e['text'], e['reply'][:400]) for e in self.qlog)))
                parts.append('EVERY QUESTION THIS TIME CONCERNS THE %s: each other product was already asked about and this one never was, so choose only among its own listed fields, starting with whichever the story details target.' % pname_k.upper())
                oh = self.open_hints(k)[:6]
                if oh:
                    parts.append('ITS STORY DETAILS WITH LIKELY PROPERTIES (likeliest first; not requirements)\n' + '\n'.join((h + (' [no key]' if self.keyless(pname(h), k) else '') for h in oh)))
                notes = self.planner_notes(k)
                if notes:
                    parts.append(notes)
                parts.append('FIELDS SEEN IN ITS LISTINGS (name, count, samples)\n' + self.records.menu_text(lines))
                ans = self.wire.consult('\n\n'.join(parts), ROLE_PLANNER_ONE, CAP_PLANNER_ONE)
                self.wire.note('plan_one', product=k, answer=ans)
                for q in (ans or {}).get('questions') or [] if isinstance(ans, dict) else []:
                    if not isinstance(q, dict):
                        continue
                    first = pname(str(q.get('property') or ''))
                    sec = pname(str(q.get('second') or ''))
                    if 'none' in sec.lower() or sec in ('-',) or 'n/a' in sec.lower():
                        sec = ''
                    if first and self.keyless(first, k):
                        first = ''
                    if sec and self.keyless(sec, k):
                        sec = ''
                    text = str(q.get('question') or '')
                    if first and self.plan_open(first, k):
                        lbl = '%s (%s)' % (first, pname_k)
                        if not good(lbl):
                            continue
                        names = bool(set(kind_list(text)) & set(kind_list(pname_k)))
                        sec_l = '%s (%s)' % (sec, pname_k) if names and sec and (not strict_same(sec, first)) and self.plan_open(sec, k) else ''
                        return {'label': lbl, 'text': text if names else self.plain(lbl), 'second': sec_l}
                    if sec and self.plan_open(sec, k):
                        lbl = '%s (%s)' % (sec, pname_k)
                        if good(lbl):
                            return {'label': lbl, 'text': self.plain(lbl), 'second': ''}
        for h in self.open_hints(k):
            nm = pname(h)
            if nm and (not non_property(nm)) and (not self.keyless(nm, k)):
                lbl = '%s (%s)' % (nm, pname_k)
                if good(lbl):
                    return {'label': lbl, 'text': self.plain(lbl), 'second': ''}
        if not allow_fallback:
            return None
        for n, c, s in self.menu_for(k):
            lbl = '%s (%s)' % (n, pname_k)
            if self.is_open(n, k) and good(lbl):
                return {'label': lbl, 'text': self.plain(lbl), 'second': ''}
        return None

    def second_subject(self, text, label, second, k):
        tw = set(re.findall('[a-z]+', text.lower()))
        tw |= {_depl(w) for w in tw}
        own = set(prop_words(label)) | set(re.findall('[a-z]+', pname(label).lower())) | {'which', 'would', 'you', 'like'}
        for nm in self.product_names():
            for w in re.findall('[a-z]+', nm.lower()):
                own.add(w)
                own.add(_depl(w))

        def mentioned(name):
            cw = [w for w in re.findall('[a-z]+', name.lower()) if len(w) >= 3 and w not in LINK]
            return bool(cw) and all((w in tw or _depl(w) in tw for w in cw)) and any((w not in own and _depl(w) not in own for w in cw))
        if second and (not strict_same(second, label)) and mentioned(pname(second)):
            return True
        lines = self.records.menu(None) if k is None else self.records.menu(None)
        low = text.lower()
        for n, c, s in lines:
            if strict_same(n, pname(label)) or non_property(n):
                continue
            if mentioned(n):
                return True
            for v in s:
                toks = [w for w in re.findall('[a-z0-9]+', v) if w not in own and w not in LINK]
                if toks and len(''.join(toks)) >= 3 and re.search('\\b' + re.escape(v) + '\\b', low):
                    return True
        return False

    def pair_fit(self, label, second, k):
        if self.needs.products():
            if self.concerns(label) is None or self.concerns(second) != self.concerns(label):
                return False
        a, b = (pname(label), pname(second))
        if strict_same(a, b) or loose_same(a, b) or self.same_field(a, b, k):
            return False
        if set(content_words(a)) & set(content_words(b)):
            return False
        if set(self.samples_for(a, k)) & set(self.samples_for(b, k)):
            return False
        return True

    def hint_backed(self, name, k):
        for h in self.hints_for(k):
            if strict_same(pname(h), name):
                return True
            m = re.search('\\(or ([^)]*)\\)', h)
            if m and any((strict_same(x.strip(), name) for x in re.split(',|/|\\bor\\b', m.group(1)))):
                return True
            cw = content_words(name)
            if cw and all((re.search('\\b' + re.escape(w), h.lower()) for w in cw)):
                return True
        return False

    def send_question(self, label, text, args, retry, split_second):
        self.last_q_count = self.records.rich_reads(self.own_label() or None) if not self.needs.products() else self.records.rich_reads()
        self.last_q_step = self.wire.step
        cid = 'code_q_%d_%d' % (self.sent_msgs, self.wire.step)
        self.chat.append({'role': 'assistant', 'content': 'I ask the shopper about %s.' % label, 'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': self.role('message'), 'arguments': json.dumps(args)}}]})
        disc_before = len(self.needs.revealed)
        free_before = len(self.needs.dontmind) + self.needs.absorbed_dontmind
        self.asking_label = label
        self.current_question = text
        res = self.dispatch([{'id': cid, 'name': self.role('message'), 'args': json.dumps(args)}], scripted=True)
        self.asking_label = ''
        self.current_question = None
        reply = res[-1][1] if res else ''
        self.post(res)
        if retry and retry in self.retry_queue:
            self.retry_queue.remove(retry)
        meta = self.needs.meta_question
        self.needs.meta_question = None
        new_disc = len(self.needs.revealed) > disc_before
        new_free = len(self.needs.dontmind) + self.needs.absorbed_dontmind > free_before
        parts = [p.strip() for p in label.split(';')]
        tags = ''
        if meta:
            self.pairs_off = True
            first = parts[0]
            alone = len(parts) == 1
            pl = self.plain(first)
            if first.lower() not in self.retried and pl.strip().lower() != text.strip().lower():
                self.retried.add(first.lower())
                self.retry_queue.insert(0, first)
                self.retry_text.pop(first, None)
                tags = 'not answered; one by one; repeated alone next'
            elif first.lower() not in self.retried:
                self.retried.add(first.lower())
                vq = self.values_question(first)
                if vq:
                    self.retry_queue.insert(0, first)
                    self.retry_text[first] = vq
                    tags = 'not answered; one by one; asked again by its values'
                else:
                    self.spent_labels.add(first.lower())
                    tags = 'not answered; one by one; no further retry'
            else:
                self.spent_labels.add(first.lower())
                tags = 'not answered; one by one; no further retry'
            for p in parts[1:] + ([split_second] if split_second else []):
                if p and p not in self.open_pairs:
                    self.open_pairs.append(p)
            label = ' and '.join(parts)
        elif len(parts) == 2 and new_disc:
            sec = parts[1]
            label = parts[0]
            nw = [w for w in re.findall('[a-z]+', pname(sec).lower()) if len(w) >= 4]
            if not any((w in ' '.join(self.needs.revealed[disc_before:]).lower() for w in nw)):
                self.retry_queue.insert(0, sec)
                tags = 'second half unanswered, still open'
        elif len(parts) == 2 and new_free:
            sec = parts[1]
            label = parts[0]
            self.needs.dontmind = [d for d in self.needs.dontmind if not strict_same(d, sec)]
            if sec not in self.open_pairs:
                self.open_pairs.append(sec)
            tags = 'second half unanswered, still open'
        elif len(parts) == 2:
            self.pairs_off = True
            for p in parts:
                if p not in self.open_pairs:
                    self.open_pairs.append(p)
            label = ' and '.join(parts)
            tags = 'not answered'
        elif split_second:
            nw = [w for w in re.findall('[a-z]+', pname(split_second).lower()) if len(w) >= 4]
            if not any((w in ' '.join(self.needs.revealed[disc_before:]).lower() for w in nw)):
                if split_second not in self.open_pairs:
                    self.open_pairs.append(split_second)
        if not meta and (not new_disc):
            self.quiet_labels.append(parts[0])
        self.qlog.append({'label': label, 'text': text, 'reply': re.split('\\nSTATUS:', reply)[0][:400], 'tags': tags})
        self.wanted_questions = [w for w in self.wanted_questions if not any((all((re.search('\\b' + re.escape(_depl(x)), w.lower()) for x in prop_words(p) if x)) for p in re.split(';|\\band\\b', label) if prop_words(p)))]
        if self.sent_msgs >= self.allowance_in_force():
            self.code_asks = False
        return True

    def values_question(self, label):
        k = self.concerns(label) if self.needs.products() else None
        lines = self.menu_for(k)
        mine = None
        others = set()
        for n, c, s in lines:
            if strict_same(n, pname(label)):
                mine = s
            else:
                others |= set(s)
        if not mine:
            return ''
        vals = []
        for v in mine:
            if len(v) <= 30 and re.search('[a-z]', v) and (not re.search('[,;/]', v)) and (v not in others) and (v not in vals):
                vals.append(v)
        vals = vals[:3]
        if len(vals) < 2:
            return ''
        q = ptag(label) or self.needs.product_name()
        return 'For the %s, %s?' % (q, ', '.join(vals[:-1]) + ' or ' + vals[-1])

    def answer_shopper(self):
        text = self.needs.reply_due
        if not text:
            return False
        base = self.ask_base if self.ask_base is not None else self.allowance_in_force()
        left_msgs = base - self.sent_msgs
        if not self.role('message') or left_msgs <= 0 or self.steps_left() <= 4 or (self.wire.time_left() < LAST_CALL):
            return False
        self.needs.reply_due = ''
        if self.code_asks and left_msgs == 1 and (self.steps_left() > 10) and (self.wire.time_left() >= CLOSING + 40):
            return False
        args = self.build_msg(text)
        if args is None:
            return False
        cid = 'code_reply_%d' % self.wire.step
        self.chat.append({'role': 'assistant', 'content': "I answer the shopper's question.", 'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': self.role('message'), 'arguments': json.dumps(args)}}]})
        res = self.dispatch([{'id': cid, 'name': self.role('message'), 'args': json.dumps(args)}], scripted=True)
        self.post(res)
        return True

    def finish_args(self, tool, kind, text):
        sch = self.shapes.get(tool) or {}
        props = sch.get('properties') if isinstance(sch.get('properties'), dict) else {}
        args = {}
        rx = DECLINE_RX if kind == 'decline' else ORDER_RX
        for name in sch.get('required') or []:
            s = props.get(name) if isinstance(props.get(name), dict) else {}
            t = s.get('type')
            if isinstance(s.get('enum'), list) and s['enum']:
                hit = next((o for o in s['enum'] if isinstance(o, str) and rx.search(o.lower())), None)
                args[name] = hit if hit is not None else s['enum'][0]
            elif t == 'string' or (isinstance(t, list) and 'string' in t):
                args[name] = text[:int(s['maxLength'])] if is_num(s.get('maxLength')) else text
            elif t == 'boolean':
                args[name] = True
            elif t in ('integer', 'number'):
                mn = s.get('minimum')
                args[name] = max(1, mn) if is_num(mn) else 1
        if len(props) == 1:
            name, s = next(iter(props.items()))
            if name not in args and isinstance(s, dict) and (s.get('type') == 'string') and ('enum' not in s):
                args[name] = text[:int(s['maxLength'])] if is_num(s.get('maxLength')) else text
        return args

    def scrub_decline(self, name, args):
        flag = (self.needs.reading or {}).get('decline_without_names')
        if flag is False:
            return args
        sch = (self.shapes.get(name) or {}).get('properties') or {}
        found = False
        for k, v in args.items():
            if isinstance(v, str):
                toks = set(re.split('[\\s,;:.!?()\\[\\]\\"\']+', v))
                if toks & self.records.known_set:
                    found = True
        if flag is True or found:
            out = dict(args)
            for k, v in args.items():
                s = sch.get(k) if isinstance(sch.get(k), dict) else {}
                if isinstance(v, str) and 'enum' not in s:
                    out[k] = DECLINE_REMARK[:int(s['maxLength'])] if is_num(s.get('maxLength')) else DECLINE_REMARK
            return out
        return args
_NUMWORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'twenty': 20, 'thirty': 30, 'fifty': 50}
_SPEND_STEMS = ('spend', 'afford', 'cost', 'pay', 'budget', 'price', 'max', 'limit')
_LIMIT_BEFORE = re.compile('(?:under|below|no more than|not more than|up to|at most|maximum|max|within|less than|budget of|limit of|cap of|ceiling of|to|at|around|about|only|tops|spend|afford)\\s*$')
_COUNT_AFTER = re.compile('^\\s*(?:x|\\u00d7|%|[:\\-/]\\d)|^\\s*(?:\\w+\\s+){0,2}(?:messages?|questions?|notes?|steps?|moves?|tool calls?|calls?|turns?|pieces?|pairs?|units?|items?|packs?|sets?|of|times?)\\b')

def extract_budget(text):
    best = None
    low = (text or '').lower()
    for sent in re.split('(?<=[.!?;])\\s+', low):
        base = 1 if re.search('\\b(?:total|altogether|everything|overall|all in|combined|in all|whole|entire)\\b', sent) else 0
        for m in re.finditer('([$€£])?\\s*(\\d[\\d,]*(?:\\.\\d+)?)', sent):
            sign, fig = (m.group(1), m.group(2))
            after = sent[m.end():]
            if _COUNT_AFTER.match(after):
                continue
            before = sent[:m.start()]
            if not sign and re.search('[\\d.,:/\\-x]\\s*$', before):
                continue
            score = base
            if sign or any((re.search('\\b' + s, before[-60:]) and len(before[-60:].split()) >= 0 and re.search('\\b' + s + '\\w*(?:\\W+\\w+){0,4}\\W*$', before[-60:]) for s in _SPEND_STEMS)):
                score += 3
            if _LIMIT_BEFORE.search(before.rstrip()) or re.search('(?:under|below|no more than|up to|at most|max\\w*|within|less than)\\s+\\w*\\s*$', before):
                score += 2
            if re.match('^\\s*(?:\\S+\\s+){0,3}?(?:or (?:less|lower|under|below)|at (?:the very )?most|max\\w*|tops|not (?:one|a) \\w+ more)', after):
                score += 2
            if score < 2:
                continue
            try:
                val = float(fig.replace(',', ''))
            except Exception:
                continue
            rest = [w for w in re.findall('[a-z]+', after)[:3] if w not in LINK and w not in ('or', 'only', 'tops', 'max', 'total', 'maximum', 'less', 'at', 'most')]
            unit = rest[0] if rest else sign or ''
            key = (score, val)
            if best is None or key > best[0]:
                best = (key, int(val) if val.is_integer() else val, unit)
    return (best[1], best[2]) if best else (None, '')

def extract_allowance(text, family):
    nouns = '(?:messages?|questions?|notes?|texts?|replies)' if family == 'message' else '(?:steps?|moves?|tool calls?|actions?|calls?)'
    low = (text or '').lower()
    num = '(\\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|fifty)'
    for m in re.finditer(num + '\\s+(?:\\w+\\s+){0,2}?' + nouns + '\\b', low):
        before = low[max(0, m.start() - 60):m.start()]
        after = low[m.end():m.end() + 40]
        cue_before = re.search('(?:no more than|not more than|up to|at most|maximum of|max|only|limit\\w*|you have|you get|stick to|keep (?:it )?to|within|allowed|budget of|cap of|no more|use|just|under|fewer than|a maximum of)(?:\\W+\\w+){0,3}\\W*$', before)
        cue_after = re.search('^(?:\\W+\\w+){0,3}?\\W*(?:in total|or fewer|or less|allowed|max\\w*|at most|tops|total|limit|only|budget)', after)
        if cue_before or cue_after:
            w = m.group(1)
            return int(w) if w.isdigit() else _NUMWORDS.get(w)
    return None

class CodePart:

    def code_roles(self):
        roles = {}
        txt = {}
        for t in self.tools:
            f = t['function']
            n = f.get('name')
            txt[n] = (split_camel(n or '').replace('_', ' ') + ' | ' + str(f.get('description') or '')).lower()

        def req(n):
            return [r for r in (self.shapes.get(n) or {}).get('required') or []]

        def props(n):
            p = (self.shapes.get(n) or {}).get('properties')
            return p if isinstance(p, dict) else {}
        taken = set()
        for n in self.shapes:
            for r in req(n):
                en = props(n).get(r, {}).get('enum') if isinstance(props(n).get(r), dict) else None
                if isinstance(en, list) and any((ORDER_RX.search(str(e).lower()) for e in en)) and any((DECLINE_RX.search(str(e).lower()) for e in en)):
                    roles['order'] = roles['decline'] = n
                    taken.add(n)
                    break
            if n in taken:
                break
        for n in self.shapes:
            if n in taken:
                continue
            nm = split_camel(n).replace('_', ' ').lower()
            if 'decline' not in roles and DECLINE_RX.search(nm):
                roles['decline'] = n
                taken.add(n)
            elif 'order' not in roles and ORDER_RX.search(nm) and (not re.search('\\b(?:cart|bag|basket)\\b', nm)):
                roles['order'] = n
                taken.add(n)
        for n in self.shapes:
            if n in taken or req(n) or 'final' not in txt[n]:
                continue
            if 'decline' not in roles and DECLINE_RX.search(txt[n]):
                roles['decline'] = n
                taken.add(n)
            elif 'order' not in roles and ORDER_RX.search(txt[n]):
                roles['order'] = n
                taken.add(n)
        for n in self.shapes:
            if n in taken or not re.search('\\b(?:cart|bag|basket|trolley)\\b', txt[n]):
                continue
            arr = any((isinstance(s, dict) and s.get('type') == 'array' for s in props(n).values()))
            if req(n) or arr:
                if 'cart' not in roles:
                    roles['cart'] = n
                    taken.add(n)
            elif 'cart_view' not in roles:
                roles['cart_view'] = n
                taken.add(n)
        person = '\\b(?:shopper|customer|user|buyer|client)\\b'
        cues = [('search', '\\b(?:search|find|browse|query|look ?up|list|filter)'), ('message', '\\b(?:send|reply|tell|ask|message|contact|write|notify)'), ('details', '\\b(?:details?|spec|attribute|propert|inspect|describe|open)'), ('view', '\\b(?:offer|price|availab|stock)')]
        for role, rx in cues:
            for n in self.shapes:
                if n in taken or role in roles or (not re.search(rx, txt[n])):
                    continue
                pr = props(n)
                texts = [k for k, s in pr.items() if isinstance(s, dict) and s.get('type') == 'string' and ('enum' not in s)]
                if role == 'search':
                    if re.search(person, txt[n]) and re.search('\\b(?:send|message|tell|reply)', txt[n]):
                        continue
                    if not (texts or any((isinstance(s, dict) and 'enum' in s for s in pr.values()))):
                        continue
                elif role == 'message':
                    if not (re.search(person, txt[n]) or re.search('\\b(?:message|chat)', txt[n])):
                        continue
                    if len(req(n)) != 1 or req(n)[0] not in texts:
                        continue
                elif len(req(n)) != 1 or req(n)[0] not in texts:
                    continue
                roles[role] = n
                taken.add(n)
        return roles

    def shelf_choices(self):
        s = self.role('search') or self.code_roles().get('search')
        best = []
        for k, sch in ((self.shapes.get(s) or {}).get('properties') or {}).items():
            if isinstance(sch, dict) and isinstance(sch.get('enum'), list) and (len(sch['enum']) >= 4):
                if len(sch['enum']) > len(best):
                    best = [str(x) for x in sch['enum']]
        return best

    def nearest_label(self, words):
        best = None
        for lab in self.shelf_choices():
            kw = kind_set(lab)
            if not kw:
                continue
            hit = len(kw & words)
            if hit == 0:
                continue
            key = (hit, hit / float(len(kw)))
            if best is None or key > best[0]:
                best = (key, lab)
        return best[1] if best else ''

    def code_reading(self):
        low = self.request.lower()
        plan = {'code_made': True, 'requirements': [], 'rules': [], 'soft_wants': [], 'ask_first': [], 'hints': [], 'references': [], 'priority_detail': ''}
        self.roles = {}
        plan['roles'] = self.code_roles()
        cap, unit = extract_budget(self.request)
        plan['price_cap'] = cap
        plan['currency'] = unit
        plan['in_stock_only'] = bool(re.search('\\b(?:in[- ]stock|sold[- ]out|available now|on hand|ready to ship|currently available|ships now)\\b', low))
        plan['priority'] = 'cheapest' if re.search('\\b(?:cheapest|least expensive|lowest[- ]priced|lowest\\b[^.]{0,20}\\bprice|cheapest possible)\\b', low) else 'none'
        mb = extract_allowance(self.request, 'message')
        if mb is not None:
            plan['message_budget'] = mb
        sb = extract_allowance(self.request, 'step')
        if sb is not None and sb >= REFUSAL_RESERVE + 3:
            plan['step_budget'] = sb
        q = 1
        m = re.search('\\b(\\d+|one|two|three|four|five|six|seven|eight|nine|ten)\\s+(?:units?|copies|of them|of those|of these)\\b', low)
        if m:
            w = m.group(1)
            q = int(w) if w.isdigit() else _NUMWORDS.get(w, 1)
            q = q if 1 <= q <= 99 else 1
        plan['quantity'] = q
        words = {re.sub('s+$', '', w) for w in content_words(low)}
        plan['product'] = self.nearest_label(words) or (' '.join(kind_list(self.request)[-1:]) or 'item')
        return plan

    def judged_values(self, k=None):
        out = []
        for r in self.needs.binding():
            if ':' not in r:
                continue
            if k is not None and self.needs.products() and ptag(r):
                if self.product_index(r) not in (k, None):
                    continue
            val = pvalue(r)
            if not val or is_reference(val):
                continue
            short = len(re.findall('[a-z0-9]+', val.lower())) <= 5 and (not EXCLUSION_RX.search(val.lower()))
            if positive_value(val, pname(r)) or short:
                out.append((pname(r), val))
        return out

    @staticmethod
    def carries(text, value):
        if not text:
            return False
        t = str(text).lower()
        v = VALUE_LINKS_RX.sub(' ', LIMIT_RX.sub(' ', value.lower()))
        tw = re.findall('[a-z0-9]+', t)
        for alt in re.split('\\bor\\b|[/,;]', v):
            ws = [w for w in re.findall('[a-z]+', alt) if len(w) >= 2 and w not in LINK]
            figs = re.findall('\\d+(?:\\.\\d+)?', alt)
            if not ws and (not figs):
                continue
            ok = all((any((x.startswith(w[:-1] if len(w) > 4 else w) for x in tw)) for w in ws))
            ok = ok and all((re.search('(?<![\\d.])' + re.escape(f) + '(?![\\d])', t) for f in figs))
            if ok:
                return True
        return False

    def shows(self, lid, prop, value):
        row = self.table().get(lid) or {}
        fields = row.get('fields') or {}
        match = [str(v) for k, v in fields.items() if loose_same(k, prop)]
        if match:
            return any((self.carries(v, value) for v in match))
        if self.carries(row.get('title'), value):
            return True
        if lid not in self.records.read:
            kw = kind_list(value)
            return bool(kw) and all_kinds(kw, self.hits.get(lid, set()))
        return False

    def clashes(self, lid, prop, value):
        fields = (self.table().get(lid) or {}).get('fields') or {}
        match = [str(v) for k, v in fields.items() if loose_same(k, prop)]
        return bool(match) and (not any((value_seen(value, [v]) for v in match)))

    def rank_pair(self, lid, prop, value):
        row = self.table().get(lid) or {}
        best = None
        forms = [' '.join(value_words(a)) for a in re.split('\\bor\\b|/', value.lower()) if value_words(a)]
        for k, v in (row.get('fields') or {}).items():
            if not loose_same(k, prop) or not self.carries(v, value):
                continue
            a = 0 if reduced(k) == reduced(prop) else 1
            vv = ' '.join(value_words(str(v)))
            if vv in forms:
                b = 0
            elif any((' '.join(value_words(e)) in forms for e in list_elements(str(v)))):
                b = 1
            else:
                b = 3
            if best is None or (a, b) < best:
                best = (a, b)
        if best is None:
            best = (2, 2) if self.carries(row.get('title'), value) else (2, 3)
        return best

    def picks(self, avoid=(), keep=()):
        ps = self.needs.products()
        prods = [(p['product'], p['quantity']) for p in ps] if ps else [(self.needs.product_name(), self.needs.quantity())]
        homes = self.home_labels()
        if not ps:
            off = self.off_category()
            offlabs = {self.category_of(x) for x in off}
            read = [r for r in self.records.categories_read() if r not in offlabs] if off else self.records.categories_read()
            homes = [{read[0]} if read else homes[0] if homes else set()]
        ver = self.needs.version
        failed = {k[0] for k, v in self.verdicts.items() if k[1] == ver and v}
        verified = {x for x, v in self.verified if v == ver}
        cap = self.needs.budget()
        avoid = set(avoid) | set(self.dead) | self.chain_listings()
        tab = self.table()
        allpicks = []
        cands = []
        req_kinds = set(kind_list(self.request))
        for k, (name, units) in enumerate(prods):
            kk = k if ps else None
            wanted = set(kind_list(name))
            vals = self.judged_values(kk)
            vwords = set()
            for _, v in vals:
                vwords |= set(kind_list(v))
            judged = [(p, v) for p, v in vals if not LIMIT_RX.search(v.lower()) and (not re.search('\\d\\s*(?:-|to)\\s*\\d|[+<>]|\\b(?:about|roughly|approximately|around)\\b', v.lower()))]
            hs = homes[k] if k < len(homes) else set()
            rows = []
            for lid, row in tab.items():
                if lid in failed or lid in self.refused_lines or lid in avoid or (row['stock'] is False):
                    continue
                if hs and row['label'] not in hs:
                    continue
                if not hs and (not wanted & row['kinds']):
                    continue
                if any((self.clashes(lid, p, v) for p, v in judged)):
                    continue
                price = row['price'] if row['price'] is not None else self.records.price(lid)
                if cap is not None and price is not None and (price * units > cap):
                    continue
                fits = all((self.shows(lid, p, v) for p, v in judged))
                opened = lid in self.records.read
                pairs = [self.rank_pair(lid, p, v) for p, v in judged] if opened and judged else []
                kinds = row['kinds']
                score = 3 * len(vwords & kinds) + 2 * len(wanted & kinds) + len(req_kinds & kinds) + 4 * len(vwords & self.hits.get(lid, set()))
                s1 = sum((a for a, b in pairs)) if pairs else 9
                s2 = sum((b for a, b in pairs)) if pairs else 9
                key = (0 if lid in keep else 1, 0 if lid in verified else 1, 0 if fits else 1, (0 if price is not None else 1) if cap is not None else 0, (price if price is not None else 1e+18) if self.needs.priority() == 'cheapest' else 0, 0 if opened else 1, 0 if row['stock'] is True else 1, s1, s2, -score, price if price is not None else 1e+18, lid)
                rows.append((key, lid, fits, price))
            rows.sort()
            if not rows:
                return []
            cands.append((rows, units))
        for pass_no in (1, 2):
            chosen = []
            fitmap = {}
            for rows, units in cands:
                order = rows if pass_no == 1 else sorted(rows, key=lambda r: (0 if r[2] else 1, (0 if r[3] is not None else 1) if cap is not None else 0, r[3] if r[3] is not None else 1e+18, r[1]))
                got = None
                for key, lid, fits, price in order:
                    if lid in [c[0] for c in chosen]:
                        continue
                    if lid in self.records.read and self.misses(lid):
                        continue
                    got = (lid, units, price)
                    fitmap[lid] = fits
                    break
                if got is None:
                    return []
                chosen.append(got)
            total = sum(((p or 0) * u for _, u, p in chosen))
            if cap is None or total <= cap:
                self.code_fit = fitmap
                return [(lid, u) for lid, u, _ in chosen]
        return []

    def cart_schema(self):
        tool = self.role('cart')
        sch = self.shapes.get(tool) or {}
        props = sch.get('properties') if isinstance(sch.get('properties'), dict) else {}
        req = [r for r in sch.get('required') or [] if r in props]
        add, rem = ({}, {})
        for r in req:
            s = props[r]
            if not (isinstance(s, dict) and isinstance(s.get('enum'), list)):
                continue
            opts = [str(o) for o in s['enum']]
            pick = None
            if self.cart.put_form:
                try:
                    pf = json.loads(self.cart.put_form)
                    if pf.get(r) in opts:
                        pick = pf[r]
                except Exception:
                    pass
            if pick is None:
                for o in opts:
                    ws = set(re.findall('[a-z]+', split_camel(o).lower()))
                    if ws & PUT_WORDS and (not ws & TAKE_WORDS):
                        pick = o
                        break
            add[r] = pick if pick is not None else opts[0]
            rm = None
            for o in opts:
                if o != add[r] and set(re.findall('[a-z]+', split_camel(o).lower())) & TAKE_WORDS:
                    rm = o
                    break
            if rm is None and len(opts) == 2:
                rm = [o for o in opts if o != add[r]][0]
            rem[r] = rm
        texts = [k for k, s in props.items() if isinstance(s, dict) and s.get('type') == 'string' and ('enum' not in s)]
        idarg = next((k for k in texts if re.search('id|product|sku|item|listing|offer|code|ref', k.lower())), texts[0] if texts else None)
        qargs = [k for k, s in props.items() if isinstance(s, dict) and s.get('type') in ('integer', 'number') and ('enum' not in s)]
        return (tool, props, add, rem, idarg, qargs)

    def cart_calls(self, picks):
        tool, props, add, rem, idarg, qargs = self.cart_schema()
        if not tool:
            return (None, None)
        shape = self.cart.list_shape()
        adds, rems = ([], [])
        if shape:
            args = dict(add)
            args[shape[0]] = [{shape[1]: lid, shape[2]: u} for lid, u in picks]
            a, f = check_args(args, self.shapes.get(tool) or {})
            if f:
                return (None, None)
            adds.append(a)
            return (adds, [])
        if not idarg:
            return (None, None)
        want = dict(picks)
        for lid, u in picks:
            if self.cart.units.get(lid) == u:
                continue
            args = dict(add)
            args[idarg] = lid
            if qargs:
                args[qargs[0]] = u
            a, f = check_args(args, self.shapes.get(tool) or {})
            if f:
                return (None, None)
            adds.append(a)
        for lid, held in self.cart.units.items():
            if lid in want:
                continue
            if add:
                if any((v is None for v in rem.values())):
                    return (None, None)
                args = dict(rem)
                args[idarg] = lid
                if qargs:
                    args[qargs[0]] = max(1, int(held))
            elif qargs and (not (is_num(props[qargs[0]].get('minimum')) and props[qargs[0]]['minimum'] > 0)):
                args = {idarg: lid, qargs[0]: 0}
            else:
                return (None, None)
            a, f = check_args(args, self.shapes.get(tool) or {})
            if f:
                return (None, None)
            rems.append(a)
        return (adds, rems)

    def send_code(self, calls):
        out = []
        for name, args in calls:
            if self.wire.finished or self.wire.blocked or self.steps_left() <= 0:
                break
            res = self.send_one(name, args, last=True)
            out.append(res)
            if res is None:
                break
        return out

    def cart_match(self, picks):
        adds, rems = self.cart_calls(picks)
        if adds is None:
            return None
        calls = [(self.role('cart'), a) for a in rems] + [(self.role('cart'), a) for a in adds]
        if not calls:
            return []
        if self.steps_left() < len(calls) + 1:
            return None
        obs = self.send_code(calls)
        if len(obs) < len(calls):
            return None
        rejected = set()
        shown = None
        for (name, args), ob in zip(calls, obs):
            o = (ob or {}).get('observation')
            r = self.cart.rows(o)
            if r:
                shown = r
            if self.is_fault(o):
                text = compact(o)
                hit = [lid for lid, _ in picks if lid in text]
                rejected |= set(hit or self.ids_named(args) or ([lid for lid, _ in picks] if self.cart.list_shape() else []))
        if shown is None and (not rejected) and self.role('cart_view') and (self.steps_left() >= 3):
            v = self.send_code([(self.role('cart_view'), {})])
            if v and v[0]:
                o = v[0].get('observation')
                r = self.cart.rows(o)
                if r:
                    shown = r
                else:
                    rejected |= {lid for lid, _ in picks if lid not in self.cart.units}
        if shown is not None:
            rejected |= {lid for lid, _ in picks if lid not in shown}
        for lid in rejected:
            self.cart.units.pop(lid, None)
        return sorted(rejected)

    @staticmethod
    def is_fault(o):
        for d in dicts_in(o, 2):
            for k, v in d.items():
                if any((s in k.lower() for s in FAULT_STEMS)) and v not in (None, '', False, 0, [], {}):
                    return True
        return False

    def stock_reread(self, lines):
        tool = self.stock_tool() or self.role('view') or self.role('details')
        if not lines or not tool or self.steps_left() < len(lines) + 2 or (self.wire.time_left() < 8):
            return None
        calls = []
        for lid in lines:
            a = self.lookup_args(tool, lid)
            if a is not None:
                calls.append((tool, a))
        if not calls:
            return None
        self.send_code(calls)
        return [lid for lid in lines if lid in self.dead or self.records.stock_of(lid) is False]

    def queries(self, k=None):
        tool = self.role('search')
        sch = self.shapes.get(tool) or {}
        props = sch.get('properties') if isinstance(sch.get('properties'), dict) else {}
        page = self.paging_param()
        texts = [n for n, s in props.items() if isinstance(s, dict) and s.get('type') == 'string' and ('enum' not in s) and (not page or n != page[0])]
        kw = next((n for n in texts if n.lower() == 'q' or re.search('query|keyword|search|term|text|words|phrase|name', n.lower())), texts[0] if texts else None)
        enums = [(n, s['enum']) for n, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list)]
        shelf = max([e for e in enums if len(e[1]) >= 4], key=lambda e: len(e[1]), default=None)
        name = self.needs.products()[k]['product'] if k is not None and self.needs.products() else self.needs.product_name()
        label = self.nearest_label(kind_set(name)) or self.nearest_label(kind_set(self.request))
        toks = [w for w in re.findall('[a-z0-9]+', name.lower()) if len(w) >= 3 and w not in LINK]
        typ = toks[-1] if toks else ''
        firm = [v for p, v in self.judged_values(k) if not re.search('\\bor\\b|[/,;]', v) and (not LIMIT_RX.search(v.lower()))]
        qs = []
        if firm:
            ws = [w for w in re.findall('[a-z0-9]+', firm[0].lower()) if w not in LINK]
            qs.append(' '.join(dict.fromkeys(ws + ([typ] if typ else []))))
            if len(ws) >= 2:
                qs.append(' '.join(dict.fromkeys([max(ws, key=len)] + ([typ] if typ else []))))
        else:
            qs.append(' '.join(dict.fromkeys(toks[:1] + toks[-1:])) if toks else ' '.join(sorted(kind_set(label))[:2]))
        calls = []
        if kw and qs and qs[0]:
            calls.append({kw: qs[0]})
        if shelf and label:
            calls.append({shelf[0]: label})
        if kw and len(qs) > 1:
            calls.append({kw: qs[1]})
        out = []
        for a in calls:
            for n in sch.get('required') or []:
                if n in a or n not in props:
                    continue
                s = props[n]
                if isinstance(s.get('enum'), list) and s['enum']:
                    a[n] = s['enum'][0]
                elif s.get('type') in ('integer', 'number'):
                    a[n] = s.get('default', s.get('minimum', 1))
                elif s.get('type') == 'string':
                    a[n] = qs[0] if qs else label
            a2, f = check_args(a, sch)
            if not f and a2 not in out:
                out.append(a2)
        return out[:3]

    def code_search(self):
        tool = self.role('search')
        if not tool:
            return False
        n = max(1, len(self.needs.products()))
        calls = []
        per = [self.queries(k if self.needs.products() else None) for k in range(n)]
        for i in range(3):
            for qs in per:
                if i < len(qs) and qs[i] not in calls:
                    calls.append(qs[i])
        calls = calls[:max(0, self.steps_left() - (2 + 2 * n))]
        if not calls:
            return False
        self.code_searched = True
        self.send_code([(tool, a) for a in calls])
        return True

    def open_picks(self, picks, avoid, keep):
        for _ in range(self.cap_reads()):
            un = [lid for lid, _ in picks if lid not in self.records.read]
            if not un or self.wire.time_left() <= 8:
                break
            mt = self.multi_tool()
            if mt and len(un) >= 2:
                calls = [(mt[0], {mt[1]: un[:mt[2]]})]
            elif self.role('details'):
                calls = [(self.role('details'), self.details_call(lid)) for lid in un]
                calls = [c for c in calls if c[1] is not None]
            else:
                break
            if not calls or self.steps_left() < len(calls) + 2 * len(picks) + 3:
                break
            self.send_code(calls)
            new = self.picks(avoid, keep)
            if new == picks:
                break
            picks = new
        return picks

    def code_shop(self, avoid=(), keep=()):
        if not self.role('cart') or self.wire.finished or self.wire.blocked or (self.steps_left() < 2) or (self.wire.time_left() < 4):
            return False
        self.wire.mute = True
        avoid = set(avoid)
        try:
            for _ in range(4):
                p = self.picks(avoid, keep)
                unfit = not p or any((not self.code_fit.get(lid, False) for lid, _ in p))
                if self.role('search') and (not self.code_searched) and (self.steps_left() >= 4) and unfit:
                    self.code_search()
                    p = self.picks(avoid, keep)
                p = self.open_picks(p, avoid, keep) if p else p
                if not p:
                    self.wire.note('code_shop', result='nothing purchasable')
                    return False
                rej = self.cart_match(p)
                if rej is None:
                    self.wire.note('code_shop', result='cart match failed')
                    return False
                if not rej and self.steps_left() >= len(p) + 3:
                    gone = self.stock_reread([lid for lid, _ in p]) or []
                    rej = list(gone)
                if not rej:
                    self.wire.note('code_shop', result='cart', cart=dict(self.cart.units))
                    return bool(self.cart.units)
                avoid |= set(rej)
                self.wire.note('code_shop', result='swap', avoid=sorted(avoid))
            return False
        except Exception as exc:
            self.wire.note('code_shop_error', error=str(exc)[:300])
            return False

    def resend_ok(self, avoid):
        try:
            if not self.cart.units:
                return False
            if any((l in avoid or l in self.dead for l in self.cart.units)):
                return False
            return bool(self.picks(set(avoid)))
        except Exception:
            return False

    def budget_scan(self, lines):
        for l in lines + [x for batch in self.needs.deferred for x in batch[0]]:
            if l in getattr(self, '_scanned', set()):
                continue
            self._scanned = getattr(self, '_scanned', set()) | {l}
            v, _ = extract_budget(l)
            if v and v > 0:
                self.needs.budget_changed = float(v)
                self.wire.note('code_budget', value=v)

    def forced_finish(self):
        if self.wire.finished or self.wire.blocked or self.steps_left() <= 0 or (self.wire.time_left() < 2):
            return
        if self.todos and self.role('order') and self.cart.units and (not any((l in self.dead for l in self.cart.units))):
            try:
                self.duty_before_order()
            except Exception as exc:
                self.wire.note('duty_failed', error=(type(exc).__name__ + ': ' + str(exc))[:200])
        o, d = (self.role('order'), self.role('decline'))
        if not o and (not d):
            cr = self.code_roles()
            o, d = (cr.get('order'), cr.get('decline'))
        avoid, keep = (set(), set())
        last_kind, last_refused, prepared = (None, False, None)
        left_by_guard = False
        for turn in range(CODE_FINISH_CALLS):
            if self.wire.finished or self.wire.blocked or self.steps_left() <= 0 or (self.wire.time_left() < 2):
                left_by_guard = True
                break
            if prepared is None:
                if turn == 0 and self.cart.units:
                    self.wire.mute = True
                    try:
                        stale = [l for l in self.cart.units if self.stale(l) and l not in self.dead]
                        if stale and self.stock_tool() and (self.steps_left() >= len(stale) + 2):
                            self.stock_reread(stale)
                        gone = {l for l in self.cart.units if l in self.dead or self.records.stock_of(l) is False}
                    except Exception:
                        gone = set()
                    avoid |= gone
                    keep = set(self.cart.units) - gone
                else:
                    gone = set()
                view_rows = self.cart.rows(self.cart.view) if self.cart.view is not None else None
                if turn == 0 and o and (not gone) and self.cart.used and (self.cart.units or not view_rows):
                    prepared = ('order', o)
                elif o and self.code_shop(avoid, keep):
                    prepared = ('order', o)
                elif o and turn > 0 and self.resend_ok(avoid):
                    prepared = ('order', o)
                elif d:
                    prepared = ('decline', d)
                else:
                    break
            kind, tool = prepared
            text = 'Please place the order for the items in the cart.' if kind == 'order' else DECLINE_REMARK
            args = self.finish_args(tool, kind, text)
            self.wire.mute = True
            lines_before = list(self.cart.units)
            res = self.send_one(tool, args, last=True)
            last_kind = kind
            self.wire.note('code_finish', finish=kind, ok=res is not None)
            if res is None or self.wire.finished or turn == CODE_FINISH_CALLS - 1 or (self.steps_left() <= 1):
                last_refused = res is not None and (not self.wire.finished)
                break
            last_refused = True
            prepared = None
            o_obs = res.get('observation')
            self.budget_scan(long_lines(o_obs if not isinstance(o_obs, dict) else list(o_obs.values()), []))
            if kind == 'decline' or not lines_before:
                continue
            gone = self.stock_reread(lines_before)
            if gone is None:
                continue
            for l in lines_before:
                if l not in gone:
                    self.refused_lines.discard(l)
            total = sum(((self.records.price(l) or 0) * u for l, u in self.cart.units.items()))
            cap = self.needs.budget()
            if not gone and (cap is None or total <= cap):
                prepared = ('order', o)
            elif cap is not None and total > cap:
                avoid = set(gone)
                keep = set()
            else:
                avoid = set(gone)
                keep = set(self.cart.units) - set(gone)
        if left_by_guard or self.wire.finished or (not last_refused) or (not d) or (last_kind == 'decline') or (self.steps_left() < 1):
            return
        try:
            again = last_kind == 'order' and self.resend_ok(avoid)
        except Exception:
            again = False
        kind, tool = ('order', o) if again else ('decline', d)
        text = 'Please place the order for the items in the cart.' if kind == 'order' else DECLINE_REMARK
        self.send_one(tool, self.finish_args(tool, kind, text), last=True)

    def catch_up(self):
        try:
            if self.wire.mute or self.wire.time_left() < CLOSING:
                return
            notes = []
            if self.code_plan and (not self.reread_done) and (self.fallback_time is not None) and (self.wire.last_reply > self.fallback_time) and (self.wire.time_left() > CLOSING + 60):
                self.reread_done = True
                r = self.wire.consult(self.setup_content, ROLE_READER, CAP_READER)
                if isinstance(r, dict) and isinstance(r.get('roles'), dict):
                    self.apply_reading(r, reread=True)
                    self.code_plan = False
                    notes.append('Models are back; this new reading of the request replaces the earlier one:\n' + self.needs.overview())
            if self.needs.deferred and self.wire.last_reply > self.needs.deferred[-1][2]:
                batches = self.needs.deferred[-3:]
                self.needs.deferred = []
                for lines, q, _ in batches:
                    m = self.needs.hear(lines, q)
                    if m:
                        notes.append('From earlier shopper lines: ' + m)
            if notes:
                text = '\n'.join(notes) + '\n' + self.status()
                last = self.chat[-1] if self.chat else None
                if last and last.get('role') in ('user', 'tool') and isinstance(last.get('content'), str):
                    last['content'] += '\n' + text
                else:
                    self.chat.append({'role': 'user', 'content': text})
        except Exception as exc:
            self.wire.note('catch_up_error', error=str(exc)[:200])

    def is_refusal(self, name, obs, finished):
        if isinstance(obs, dict) and 'refusal' in obs:
            return True
        if name != self.role('order') and (not (name == self.role('decline') and self.role('decline') == self.role('order'))):
            return False
        if finished:
            return False
        if isinstance(obs, dict):
            err = str(obs.get('error') or '').lower()
            if err and re.search('invalid arg|unknown tool|not a tool|unrecognised|unrecognized|schema|missing required', err):
                return False
        return True

    def paging_param(self):
        tool = self.role('search')
        props = (self.shapes.get(tool) or {}).get('properties') or {}
        for n, s in props.items():
            if not isinstance(s, dict) or s.get('type') not in ('integer', 'number'):
                continue
            nl = n.lower()
            if any((f in nl for f in ('page', 'offset', 'skip', 'start', 'from', 'cursor', 'after'))) and (not any((f in nl for f in ('size', 'limit', 'price', 'per', 'max')))):
                is_page = 'page' in nl and (not any((f in nl for f in ('offset', 'skip', 'start', 'from', 'after'))))
                first = s.get('minimum', s.get('default', 1 if is_page else 0))
                return (n, 'page' if is_page else 'offset', first, s.get('maximum'))
        for n, s in props.items():
            if isinstance(s, dict) and s.get('type') == 'string' and ('enum' not in s) and re.search('cursor|token|next|after|continu|bookmark|page', n.lower()):
                return (n, 'token', '', None)
        return None

    def log_search(self, args, obs):
        pg = self.paging_param()
        rows = []
        for v in obs.values() if isinstance(obs, dict) else [obs]:
            if isinstance(v, list) and v and all((isinstance(x, dict) for x in v)):
                if len(v) > len(rows):
                    rows = v
        if not pg or not rows:
            return rows
        key = compact({k: v for k, v in sorted((args or {}).items()) if k != pg[0]})
        e = self.search_log.setdefault(key, {'args': {k: v for k, v in (args or {}).items() if k != pg[0]}, 'visited': set(), 'size': 0, 'total': 0, 'labels': set(), 'first': [], 'sorted': None, 'tokens': {}, 'seen': [], 'row_labels': {}, 'order': len(self.search_log)})
        val = (args or {}).get(pg[0], pg[2])
        if pg[1] == 'token':
            val = str(val or '').strip()
            tok = None
            for k, v in obs.items() if isinstance(obs, dict) else []:
                if isinstance(v, str) and v.strip() and re.search('next|cursor|more|token|continu|follow|after', k.lower()):
                    tok = v
            if tok:
                e['tokens'][val] = tok
        elif not is_num(val):
            return rows
        e['visited'].add(val)
        e['size'] = max(e['size'], len(rows))
        tot = max([v for v in (obs.values() if isinstance(obs, dict) else []) if isinstance(v, int) and (not isinstance(v, bool))] + [0])
        e['total'] = max(e['total'], tot if tot > len(rows) else len(rows))
        for r in rows:
            lab = record_label(r)
            rid = row_identity(r)
            if lab:
                e['labels'].add(lab)
                e['row_labels'].setdefault(rid, lab)
            if rid not in e['seen']:
                e['seen'].append(rid)
        self.totals[key] = e['total']
        ids = [row_identity(r) for r in rows]
        if val in (pg[2], '') and (not e['first']):
            e['first'] = ids
        if len(ids) >= 5:

            def nat(s):
                return [int(t) if t.isdigit() else t for t in re.split('(\\d+)', s)]
            asc = ids == sorted(ids) or ids == sorted(ids, key=nat)
            if not asc:
                e['sorted'] = False
            elif e['sorted'] is None:
                e['sorted'] = True
        return rows

    def next_page(self, e):
        pg = self.paging_param()
        if not pg:
            return None
        if pg[1] == 'token':
            for v, tok in e['tokens'].items():
                if tok not in e['visited']:
                    return tok
            return None
        nums = [v for v in e['visited'] if is_num(v)]
        base = min([pg[2]] + nums) if is_num(pg[2]) else min(nums) if nums else 0
        inc = 1 if pg[1] == 'page' else max(1, e['size'])
        for i in range(1, 400):
            v = base + i * inc
            if pg[3] is not None and is_num(pg[3]) and (v > pg[3]):
                return None
            if i * max(1, e['size']) >= e['total']:
                return None
            if v not in e['visited']:
                return v
        return None

    def value_score(self, args):
        qt = ' '.join((str(v) for v in (args or {}).values() if isinstance(v, str))).lower()
        qtok = [t for t in re.findall('[a-z0-9]+', qt) if len(t) >= 2]
        n = 0
        for r in self.needs.binding():
            if ':' not in r or is_reference(pvalue(r)):
                continue
            vt = [t for t in re.findall('[a-z0-9]+', r.split(':', 1)[1].lower()) if len(t) >= 3]
            if any((len(a) >= 3 and (a.startswith(b) or b.startswith(a)) for a in qtok for b in vt)):
                n += 1
        return n

    def ready_pages(self, count=2):
        pg = self.paging_param()
        if not pg:
            return []
        wanted = set()
        for hs in self.home_labels():
            wanted |= hs
        cands = []
        for key, e in self.search_log.items():
            nxt = self.next_page(e)
            if nxt is None:
                continue
            if wanted and e['labels'] and (not e['labels'] & wanted):
                continue
            sc = self.value_score(e['args'])
            if sc > 0:
                a = dict(e['args'])
                a[pg[0]] = nxt
                cands.append((-sc, e['total'], compact(a)))
        cands.sort()
        return [c[2] for c in cands[:count]]

    def search_note(self, args, obs):
        try:
            rows = self.log_search(args, obs)
            for r in rows:
                rid = row_identity(r)
                qw = ' '.join((str(v) for v in (args or {}).values() if isinstance(v, str)))
                self.hits.setdefault(rid, set()).update(kind_list(qw))
            tot = max([v for v in (obs.values() if isinstance(obs, dict) else []) if isinstance(v, int) and (not isinstance(v, bool))] + [0])
            total = tot if tot > len(rows) else len(rows)
            notes = []
            self.learn_words(args, rows, total)
            if self.read_width:
                toks = [t for t in re.findall('[a-z0-9]+', self.free_text(args).lower())]
                notes.append('search here uses only the first %d query words%s; lead with the %d most telling words (value before kind) and query other values separately' % (self.read_width, ' (counted: %s)' % ' '.join(toks[:self.read_width]) if len(toks) > self.read_width else '', self.read_width))
            if self.word_rule != 'unknown':
                notes.append("so far this shop's keyword search looks %s (%s)" % (self.word_rule, 'every word is required' if self.word_rule == 'strict' else 'any one word is enough'))
            if rows and tot > len(rows):
                hint = {'loose': "a required value's distinctive term or number on its own lists only its bearers, the kind plus values ranks closer matches higher, and category browsing or a working price filter covers the rest", 'strict': 'extra exact words plus category and price filters narrow it'}.get(self.word_rule, 'add distinctive value words or filters to narrow it')
                notes.append('this search reports %d matches but shows %d rows; the fit or the cheapest may sit on later pages: page this very search before anything else; if the total exceeds what can be read, %s' % (tot, len(rows), hint))
            pg = self.paging_param()
            limits = [(k, v) for k, v in (args or {}).items() if is_num(v) and v > 1 and (not pg or k != pg[0]) and (not re.search('page|size|offset|skip|min|lower|from|count|limit', k.lower()))]
            if limits and rows:
                top = max((v for _, v in limits))
                over = [r for r in rows if record_price(r) is not None and record_price(r) > top + 1e-09]
                if over:
                    notes.append('%d rows cost more than the numeric limit %s, so this mode ignored that filter and might ignore others such as the category; switch to a mode whose filters work, or verify each row' % (len(over), money(top)))
            return '; '.join(notes)
        except Exception as exc:
            self.wire.note('search_note_error', error=str(exc)[:200])
            return ''

    def learn_words(self, args, rows, total):
        pg = self.paging_param()
        s = self.role('search')
        props = (self.shapes.get(s) or {}).get('properties') or {}
        if pg:
            v = (args or {}).get(pg[0])
            if pg[1] == 'token':
                if str(v or '').strip():
                    return
            elif is_num(v) and v > (pg[2] if is_num(pg[2]) else 0):
                return
        enum_keys = {k for k in args or {} if isinstance(props.get(k), dict) and 'enum' in props[k]}
        texts = {k: v for k, v in (args or {}).items() if isinstance(v, str) and k not in enum_keys and (not pg or k != pg[0])}
        mode = compact(sorted(((k, str(args[k])) for k in enum_keys)))
        rest = compact(sorted(((k, str(v)) for k, v in (args or {}).items() if k not in texts and k not in enum_keys and (not pg or k != pg[0]))))
        ids = [row_identity(r) for r in rows]
        for param, text in texts.items():
            toks = []
            for t in text.lower().split():
                t = t.strip('"\'()[]{}.,;:!?|/')
                if re.search('[a-z0-9]', t) and t not in toks:
                    toks.append(t)
            if not toks or not all((re.fullmatch('[a-z]+|[0-9]+', t) and len(t) >= 2 for t in toks)):
                continue
            other_texts = compact(sorted(((k, v) for k, v in texts.items() if k != param)))
            ctx = (param, mode, rest, other_texts)
            hint = False
            for h in self.kw_history:
                if h['ctx'] != ctx:
                    continue
                if h['total'] != total:
                    a, b = (h['tokens'], toks)
                    if len(b) > len(a) and b[:len(a)] == a:
                        self.word_rule = 'loose' if total > h['total'] else 'strict'
                    elif len(a) > len(b) and a[:len(b)] == b:
                        self.word_rule = 'loose' if h['total'] > total else 'strict'
                    else:
                        ka = a[:self.read_width] if self.read_width else a
                        kb = b[:self.read_width] if self.read_width else b
                        if set(kb) > set(ka) and total < h['total']:
                            hint = True
                        elif set(ka) > set(kb) and h['total'] < total:
                            hint = True
                if h['tokens'] != toks:
                    c = 0
                    while c < min(len(h['tokens']), len(toks)) and h['tokens'][c] == toks[c]:
                        c += 1
                    key = (param, mode)
                    fl = self.k_floor.get(key, -1)
                    if h['total'] != total or h['ids'] != ids:
                        self.k_floor[key] = max(fl, c)
                    elif c >= 1 and c == min(len(h['tokens']), len(toks)) and (len(set(ids)) >= 3):
                        self.k_ceil.setdefault(key, set()).add(c)
            if hint and self.word_rule == 'unknown':
                self.word_rule = 'strict'
            self.kw_history.append({'ctx': ctx, 'tokens': toks, 'total': total, 'ids': ids})
        claims = []
        for key, ceils in self.k_ceil.items():
            fl = self.k_floor.get(key, -1)
            above = sorted((c for c in ceils if c > fl))
            if above and above[0] == fl + 1:
                claims.append((above[0], key))
        if claims:
            claims.sort()
            self.read_width = claims[0][0]
        else:
            self.read_width = 0
PROMPT_HOP = 'A shopper\'s request takes the %s of the product it wants from a different shop listing, described as: %s. The records below may include that listing. Reply with one JSON object only, holding:\n"target": the id of the record that really matches the description: same seller, same kind, every named detail present, compared by sense (a figure in another unit counts if the amount is equal); null if none matches.\n"missing": described details absent from the target ([] if none).\n"go_to": if the target\'s text sends the reader elsewhere for the %s (take it from, use, see, consult, follow some other listing), that other listing as {"seller": verbatim, "kind": ..., "details": [values the text gives for it, brief and verbatim: a colour, a size with unit, a material]}; otherwise null.\n"value": when there is no go_to, the %s as the target\'s record spells it.\n"goes_with": values the target\'s own wording declares as accompanying, paired with or equivalent to that value ([] if none).\nNever treat as a go_to a listing mentioned only as something the target differs from (for example a remark that it is not some seller\'s green item), or as another version (another size or colour) sold by someone else, or a listing mentioned about a different property.\nA record lacking the described details cannot be the target, whatever its wording claims.'
_SELLER_STEMS = ('seller', 'vendor', 'merchant', 'sold by', 'shop', 'store', 'brand', 'maker', 'manufactur', 'supplier', 'retailer')
_TEXTISH = ('note', 'desc', 'message', 'overview', 'summary', 'comment', 'detail', 'about')

def seller_key(name):
    t = (name or '').lower().replace("'s", '')
    return re.sub('[^a-z0-9]', '', t)

def ref_tokens(text):
    t = (text or '').lower()
    t = re.sub('(\\d)\\s*-?\\s*([a-z]{1,8})\\b', lambda m: m.group(1) + m.group(2) if m.group(2) not in LINK else m.group(0), t)
    toks = re.findall("\\d[\\d.,:/]*[a-z]*|[a-z][a-z0-9'\\-]*", t)
    out = []
    for w in toks:
        if w in LINK or (len(w) == 1 and (not w.isdigit())):
            continue
        if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) and (not w[-2].isdigit()):
            w = w[:-2] if re.search('(?:ch|sh|x|z)es$', w) else w[:-1]
        out.append(w)
    return out

def parse_ref_phrase(text, accept=None):
    try:
        t = re.split('\\(\\s*or\\b', text or '')[0]
        seller, pos = (None, None)
        for m in re.finditer("(?<![A-Za-z])([A-Z][a-z]{3,}(?:'s)?)(?![A-Za-z])", t):
            if m.start() == 0:
                continue
            nm = m.group(1).replace("'s", '')
            if accept and (not accept(nm)):
                continue
            seller, pos = (nm, m)
            break
        if not seller:
            return None
        before = t[:pos.start()]
        after = t[pos.end():]
        kind = ''
        mb = re.search('\\b(?:the|a|an|for)\\s+([a-z][a-z \\-]{1,60}?)\\s+(?:sold|made|offered|listed)\\s+by\\s*$', before, re.I)
        if mb:
            kind = mb.group(1)
            rest = after
        elif after.lstrip().startswith('('):
            kind, rest = ('', after)
        else:
            mk = re.match('\\s*([^,;]*?)(?=\\s*(?:[,;]|$|\\b(?:having|from|sold|with|which|that|whose|in|at|by)\\b))', after)
            kind = mk.group(1) if mk else ''
            rest = after[mk.end():] if mk else after
        details = []
        for piece in re.split('[,;]|\\band\\b|\\bwhich\\b|\\bthat (?:is|are|has|have|was)\\b(?:\\s+as)?', rest):
            toks = [w for w in re.findall("[A-Za-z0-9][A-Za-z0-9.%:/&'\\-]*", piece) if w.lower() not in LINK]
            if toks:
                details.append(' '.join(toks))
            if len(details) >= 6:
                break
        return (seller, kind.strip()[:80], details)
    except Exception:
        return None

class RefsPart:

    def kind_vocab(self):
        voc = set()
        for nm in self.product_names():
            voc |= set(kind_list(nm))
        s = self.role('search')
        for sch in ((self.shapes.get(s) or {}).get('properties') or {}).values():
            if isinstance(sch, dict) and isinstance(sch.get('enum'), list):
                for e in sch['enum']:
                    voc |= set(kind_list(str(e)))
        return voc

    def seller_ok(self, name, text):
        if not name or len(name) < 3:
            return False
        if not re.search('\\b' + re.escape(name.lower()) + "(?:'s)?\\b", (text or '').lower()):
            return False
        kw = set(kind_list(name))
        voc = self.kind_vocab()
        return bool(kw) and (not all((kind_in(w, voc) for w in kw)))

    def rec_sellers(self, d):
        out = set()
        for k, v in d.items():
            kl = k.lower()
            if isinstance(v, str) and v.strip() and (len(v) <= 60) and ('id' not in kl) and any((s in kl for s in _SELLER_STEMS)) and (not any((t in kl for t in _TEXTISH))):
                out.add(seller_key(v))
        return out

    def rec_notes(self, d):
        notes, desc = ([], [])
        for k, v in d.items():
            kl = k.lower()
            if not isinstance(v, str) or len(v.split()) < 3 or kl == 'id' or kl.endswith('id'):
                continue
            if any((s in kl for s in ('title', 'name', 'categor', 'shelf', 'heading', 'label', 'caption', 'headline', 'aisle', 'department', 'sku'))):
                continue
            if any((s in kl for s in ('desc', 'overview', 'summary', 'about'))):
                desc.append(v)
            else:
                notes.append(v)
        return (' '.join(notes), ' '.join(desc))

    def ref_index(self):
        cards, _ = self.records.merged()
        return cards

    def shows_detail(self, d, detail):
        pm = prop_map(d) or {}
        names = set()
        for k in pm:
            names |= set(re.findall('[a-z]+', split_camel(k).lower().replace('_', ' ')))
        notes, _ = self.rec_notes(d)
        notes = re.sub('\\b(?:not|never|without|no|nor)\\b(?:\\s+\\S+){1,3}', ' ', notes.lower())
        vals = ' '.join([record_title(d), notes] + [str(v) for v in pm.values() if not isinstance(v, (dict, list))])
        vt = set(ref_tokens(vals))
        words = [w for w in ref_tokens(detail) if len(w) >= 3 and (not re.search('\\d', w)) and (w not in names)]
        if words and (not any((w in vt for w in words))):
            return False
        pairs = re.findall('(\\d+(?:\\.\\d+)?)\\s*-?\\s*([a-z]{1,8})\\b', detail.lower())
        if pairs:
            src = ' '.join([record_title(d)] + [str(v) for v in pm.values() if not isinstance(v, (dict, list))]).lower()
            have = re.findall('(\\d+(?:\\.\\d+)?)\\s*-?\\s*([a-z]{1,8})\\b', src)
            for num, unit in pairs:
                same_unit = [n for n, u in have if u.rstrip('s') == unit.rstrip('s')]
                if same_unit and (not any((float(n) == float(num) for n in same_unit))):
                    return False
        return True

    def full_count(self, d, details):
        pm = prop_map(d) or {}
        pool = set(ref_tokens(' '.join([record_title(d)] + [str(k) + ' ' + str(v) for k, v in pm.items()])))
        return sum((1 for x in details if ref_tokens(x) and all((t in pool for t in ref_tokens(x)))))

    def rec_value(self, d, prop):
        pm = prop_map(d) or {}
        for k, v in pm.items():
            if strict_same(k, prop) and str(v).strip() and (not isinstance(v, (dict, list))):
                return str(v)
        for k, v in pm.items():
            if loose_same(k, prop) and str(v).strip() and (not isinstance(v, (dict, list))):
                return str(v)
        return ''

    def ref_register_setup(self):
        try:
            r = self.needs.reading or {}
            refs = [x for x in r.get('references') or [] if isinstance(x, str)]
            entries = [e for e in r.get('reference_targets') or [] if isinstance(e, dict)]
            for s in list(self.needs.stated):
                val = s.split(':', 1)[1] if ':' in s else ''
                if REF_RX.search(val.lower()) or (s in refs and DIFFER_RX.search(val.lower())):
                    self.open_tracker(s, entries, self.request, 'request')
        except Exception as exc:
            self.wire.note('ref_setup_error', error=str(exc)[:200])

    def open_tracker(self, req, entries, source, origin):
        head, val = req.split(':', 1)
        prop = head.split('(')[0].strip()
        tag = ptag(head)
        if not prop or non_property(prop):
            return None
        for t in self.trackers:
            if t['req'] == req or (strict_same(t['prop'], prop) and kind_set(t['tag']) == kind_set(tag)):
                return None
        entry = None
        for e in entries:
            ep = str(e.get('property') or '')
            eprod = str(e.get('product') or '')
            if (strict_same(ep, prop) or loose_same(ep, prop)) and (not tag or not eprod or (not self.needs.products()) or kind_set(eprod) & kind_set(tag)):
                entry = e
                break
        target = self.make_target(entry, source, val)
        if not target:
            self.wire.note('ref_skip', req=req)
            return None
        rel, tol, unit = ref_relation(val)
        if rel == 'same' and entry and (str(entry.get('relation') or '').lower() == 'within') and is_num(entry.get('tolerance')):
            rel, tol, unit = ('within', float(entry['tolerance']), str(entry.get('unit') or ''))
        if rel == 'same':
            etxt = str((entry or {}).get('relation') or '').lower()
            span = source or ''
            i = span.lower().find(target['seller'].lower())
            if re.search('differ|exclu|unlike|other than|apart|not the same', etxt) or (i > 0 and DIFFER_RX.search(span[:i].lower())):
                rel = 'differ'
        t = {'req': req, 'head': head.strip(), 'prop': prop, 'tag': tag, 'relation': rel, 'tol': tol, 'unit': unit, 'target': target, 'first': dict(target), 'chain': [], 'seen': {self.target_key(target)}, 'judged': set(), 'queries': [], 'pages': 0, 'hops': 0, 'fails': 0, 'status': 'open', 'value': '', 'paired': [], 'resolved': '', 'origin': origin, 'not_indexed': False}
        self.trackers.append(t)
        self.wire.note('ref_open', req=req, target=target, relation=rel)
        return t

    def target_key(self, target):
        toks = set()
        for d in target.get('details') or []:
            toks |= set(ref_tokens(d))
        return seller_key(target.get('seller')) + '|' + ' '.join(sorted(toks))

    def make_target(self, entry, source, value):
        seller, kind, details = ('', '', [])
        if entry:
            seller = str(entry.get('seller') or '').strip()
            kind = str(entry.get('kind') or '').strip()
            raw = entry.get('details') or []
            if isinstance(raw, str):
                raw = [x for x in re.split('[,;]', raw) if x.strip()]
            details = [str(x).strip() for x in raw if isinstance(x, (str, int, float)) and (not isinstance(x, bool)) and str(x).strip()]
        parsed = None
        for src in (source, value):
            p = parse_ref_phrase(src, accept=lambda n, s=src: self.seller_ok(n, s))
            if p:
                parsed = p
                break
        if not self.seller_ok(seller, source):
            if not parsed:
                return None
            seller, kind, details = (parsed[0], kind or parsed[1], details or parsed[2])
        elif not details and parsed and (seller_key(parsed[0]) == seller_key(seller)):
            details = parsed[2]
        src_toks = set(ref_tokens(source + ' ' + value))
        keep = []
        for d in details:
            if d not in keep and set(ref_tokens(d)) & src_toks:
                keep.append(d)
        return {'seller': seller, 'kind': (kind or '')[:80], 'details': keep[:6]}

    def phrase(self, target):
        return '%s sold by %s%s' % (target.get('kind') or 'listing', target.get('seller'), ' (%s)' % '; '.join(target['details']) if target.get('details') else '')

    def ref_intake(self, needs, entries, lines):
        for n in needs:
            if ':' not in n:
                continue
            if REF_RX.search(n.split(':', 1)[1].lower()):
                t = self.open_tracker(n, [e for e in entries if isinstance(e, dict)], ' '.join(lines), 'reply')
                if t:
                    refs = self.needs.reading.setdefault('references', [])
                    if isinstance(refs, list) and n not in refs:
                        refs.append(n)

    def resolver_room(self):
        opened = [t for t in self.trackers if t['status'] == 'open']
        if not opened:
            return False
        differ_only = all((t['relation'] == 'differ' for t in opened))
        if differ_only and all((not t['chain'] for t in opened)) and (self.ref_steps >= 5):
            return False
        cap = max(6, int(self.budget_steps * self.share_refs(differ_only))) if differ_only else max(8, int(self.budget_steps * self.share_refs(differ_only)))
        if self.ref_steps >= cap or self.wire.time_left() <= CLOSING + 60 or self.wire.spent_out() or self.wire.finished or self.wire.blocked:
            return False
        base = min(max(self.budget_steps // 4 + 1, RESERVE_LOW), RESERVE_HIGH)
        floor = max(base + 10, self.reserve() + 7)
        return self.steps_left() > floor

    def hops_allowed(self):
        return self.hop_calls < self.cap_hops() and self.wire.time_left() > CLOSING + 40 and (not self.wire.spent_out()) and (self.wire.spend < 0.8 * self.wire.ceiling())

    def ref_turn(self):
        if not self.trackers or self.wire.finished or self.wire.blocked:
            return False
        try:
            for t in self.trackers:
                if t['status'] == 'open' and (not self.tracker_applies(t)):
                    t['status'] = 'dropped'
                    self.wire.note('ref_dropped', req=t['req'])
            opened = sorted([t for t in self.trackers if t['status'] == 'open'], key=lambda t: t['hops'])
            if not opened:
                return False
            before = self.wire.step
            for t in opened:
                self.judge(t)
                if self.wire.step != before:
                    return True
            opened = [t for t in self.trackers if t['status'] == 'open']
            if not opened or not self.resolver_room():
                return False
            mt = self.multi_tool()
            batch = mt[2] if mt else 2
            queue = []
            pools = {}
            for t in opened:
                pool = [c for c in self.candidates(t) if c not in self.records.read and c not in t['judged']]
                ready = bool(t['queries']) or not self.planned_queries(t) or len(pool) <= batch or (pool and t.get('shown', {}).get(pool[0], 0) > 0)
                if pool and ready:
                    pools[id(t)] = pool
            while len(queue) < batch and any(pools.values()):
                for k in list(pools):
                    if pools[k]:
                        c = pools[k].pop(0)
                        if c not in queue and len(queue) < batch:
                            queue.append(c)
            if queue:
                self.ref_read(queue, opened)
                return True
            tmpl = self.search_template()
            if tmpl is None:
                return False
            for t in sorted(opened, key=lambda t: (t['hops'], len(t['queries']))):
                qs = self.planned_queries(t)
                if qs:
                    q = qs[0]
                    t['queries'].append(q.lower())
                    args = dict(tmpl[0])
                    args[tmpl[1]] = q
                    self.ref_search(args, t, q)
                    return True
            for t in opened:
                if not self.candidates(t) or all((c in t['judged'] for c in self.candidates(t))):
                    self.stop(t, "its listing never appeared in the store's search results")
            return False
        except Exception as exc:
            self.wire.note('ref_error', error=(type(exc).__name__ + ': ' + str(exc))[:300])
            for t in self.trackers:
                if t['status'] == 'open':
                    self.stop(t, 'resolver failure')
            return False

    def tracker_applies(self, t):
        for r in self.needs.binding():
            if r == t['req'] or r == t['resolved'] or strict_same(pname(r), t['prop']):
                return True
        return not any((strict_same(pname(d), t['prop']) for d in self.needs.dropped))

    def candidates(self, t):
        sk = seller_key(t['target']['seller'])
        out = []
        cards = self.ref_index()
        kinds = set(kind_list(t['target'].get('kind') or ''))
        shown = {}
        for lid, d in cards.items():
            if sk and sk in self.rec_sellers(d):
                lab = record_label(d)
                score = 2 if kinds and lab and set(kind_list(lab)) & kinds else 0
                _, desc = self.rec_notes(d)
                pool = set(ref_tokens(record_title(d) + ' ' + desc))
                n = sum((1 for x in t['target']['details'] if ref_tokens(x) and all((w in pool for w in ref_tokens(x)))))
                shown[lid] = n
                out.append((-(score + n), lid))
        t['shown'] = shown
        return [lid for _, lid in sorted(out)]

    def search_template(self):
        s = self.role('search')
        if not s:
            return None
        props = (self.shapes.get(s) or {}).get('properties') or {}
        pg = self.paging_param()
        labels = set(self.records.labels.values())
        for rec in reversed(self.log):
            if rec.get('kind') != 'env' or rec.get('name') != s:
                continue
            args = rec.get('args') or {}
            texts = [k for k, v in args.items() if isinstance(v, str) and v.strip() and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and (not pg or k != pg[0])]
            if len(texts) != 1:
                continue
            tmpl = {k: v for k, v in args.items() if k != texts[0] and (not pg or k != pg[0]) and (not (isinstance(v, str) and v.strip().lower() in labels and (k not in ((self.shapes.get(s) or {}).get('required') or []))))}
            return (tmpl, texts[0])
        plain = [k for k, v in props.items() if isinstance(v, dict) and v.get('type') == 'string' and ('enum' not in v) and (not pg or k != pg[0])]
        req = (self.shapes.get(s) or {}).get('required') or []
        if len(plain) == 1 and all((r == plain[0] for r in req)):
            return ({}, plain[0])
        return None

    def planned_queries(self, t):
        tg = t['target']
        seller = tg.get('seller') or ''
        fieldvoc = set()
        for lid, row in self.table().items():
            for k in row['fields']:
                fieldvoc |= set(re.findall('[a-z]+', split_camel(k).lower().replace('_', ' ')))
        for r in self.needs.stated + self.needs.revealed:
            fieldvoc |= set(re.findall('[a-z]+', pname(r).lower()))
        fused, nums, words = ([], [], [])
        sk = set(re.findall('[a-z]+', seller.lower()))
        for d in tg.get('details') or []:
            prev_num = False
            for piece in d.lower().split():
                bit = re.sub('[^a-z0-9.]', '', piece)
                if not bit or bit in LINK or bit in sk:
                    prev_num = False
                    continue
                if re.fullmatch('\\d+(?:\\.\\d+)?', bit):
                    if len(re.sub('\\D', '', bit)) >= 2:
                        nums.append(bit)
                    prev_num = True
                    continue
                if re.search('\\d', bit) and re.search('[a-z]', bit):
                    fused.append(bit)
                elif bit.isalpha() and (not prev_num) and (len(bit) >= 3) and (bit not in fieldvoc) and (bit.rstrip('s') not in fieldvoc):
                    words.append(bit)
                prev_num = False
        words.sort(key=len, reverse=True)
        fieldnames = set()
        for lid, row in self.table().items():
            for k in row['fields']:
                fieldnames.add(' '.join(re.findall('[a-z0-9]+', split_camel(k).lower().replace('_', ' '))))
        dets = [[w for w in re.findall('[a-z0-9.]+', d.lower()) if w not in LINK] for d in tg.get('details') or []]
        all_long = bool(dets) and all((len(x) >= 2 for x in dets))
        for toks in dets:
            if len(toks) < 2 or toks[-1] not in words:
                continue
            lead = any((' '.join(toks[:n]) in fieldnames for n in range(1, len(toks))))
            if lead or (t['hops'] == 0 and all_long):
                words.remove(toks[-1])
                words.insert(0, toks[-1])
        cand = list(dict.fromkeys(fused + nums + words))
        out = []
        if t['not_indexed']:
            kw = [w for w in kind_list(tg.get('kind') or '') if w not in LINK]
            if cand and kw:
                out.append(' '.join([cand[0], kw[-1]]))
        else:
            if cand and len(seller.split()) < 3:
                out.append('%s %s' % (seller, cand[0]))
            out.append(seller)
        return [q for q in out if q.strip() and q.lower() not in t['queries']]

    def ref_search(self, args, t, q):
        s = self.role('search')
        args, faults = check_args(args, self.shapes.get(s) or {})
        if faults:
            t['queries'].append('__fault__')
            return
        cid = 'code_ref_s_%d' % self.wire.step
        self.chat.append({'role': 'assistant', 'content': 'I search for the listing that %s refers to.' % t['head'], 'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': s, 'arguments': json.dumps(args)}}]})
        before = self.wire.step
        self.covering = True
        try:
            res = self.dispatch([{'id': cid, 'name': s, 'args': json.dumps(args)}], scripted=True)
        finally:
            self.covering = False
        self.ref_steps += self.wire.step - before
        self.post(res)
        rows = []
        ob = self.records.obs[-1] if self.records.obs else None
        for d in dicts_in(ob, 3):
            if own_id(d) in self.records.known_set and len(d) >= 4:
                rows.append(d)
        sk = seller_key(t['target']['seller'])
        if rows and q.strip().lower() == t['target']['seller'].strip().lower() and (not any((sk in self.rec_sellers(r) for r in rows))):
            t['not_indexed'] = True

    def ref_read(self, ids, opened):
        before = self.wire.step
        sentence = 'Some requirements (%s) take their value from another listing, so I open possible matches.' % '; '.join((t['head'] for t in opened))
        res = self.scripted_reads(ids, sentence)
        self.ref_steps += self.wire.step - before
        if res:
            self.post(res)
        self.ref_faces(ids, opened)
        self.cand_read |= set(ids)
        for x in ids:
            if x not in self.records.read:
                for t in opened:
                    t['judged'].add(x)

    def ref_faces(self, ids, opened):
        face = self.role('view')
        if not face or face == self.role('details'):
            return
        want = []
        det = [x for t in opened for x in t['target']['details']]
        for lid in ids:
            if lid in self.face_read:
                continue
            card = self.ref_index().get(lid) or {}
            if not det or not card or any((self.shows_detail(card, x) for x in det)):
                want.append(lid)
        want = want[:2]
        if not want or self.steps_left() <= len(want) + 6 or self.wire.time_left() < CLOSING:
            return
        before = self.wire.step
        res = self.scripted_reads(want, 'A referenced value is often written on the listing itself rather than in its fields, so I open the listing.', tool=face)
        self.ref_steps += self.wire.step - before
        self.face_read |= set(want)
        if res:
            self.post(res)

    def judge(self, t):
        cards = self.ref_index()
        cands = [c for c in self.candidates(t) if c in self.records.read and c not in t['judged']]
        if not cands:
            return
        det = t['target']['details']
        ranked = []
        for i, c in enumerate(cands):
            d = cards.get(c, {})
            wm = all((self.shows_detail(d, x) for x in det))
            ranked.append((0 if wm else 1, -self.full_count(d, det), i, c))
        ranked.sort()
        best_wm = ranked[0][0] == 0
        more = [c for c in self.candidates(t) if c not in self.records.read and c not in t['judged']]
        if not best_wm and self.resolver_room() and (more or self.planned_queries(t)):
            return
        wms = [c for w, _, _, c in ranked if w == 0]
        pick = wms[:4] if wms else [c for _, _, _, c in ranked[:4]]
        exact = [c for c in pick if det and self.full_count(cards.get(c, {}), det) == len(det)]
        if len(exact) == 1 and self.plain_record(cards.get(exact[0], {}), t):
            c = exact[0]
            t['judged'] |= set(pick)
            self.cand_read.add(c)
            self.resolve(t, c, self.rec_value(cards.get(c, {}), t['prop']), [])
            return
        if not self.hops_allowed():
            t['judged'] |= set(pick)
            self.stop(t, 'hop check budget exhausted')
            return
        ans, problem = self.hop_call(t, pick, cards, None)
        if problem and self.hops_allowed():
            ans, problem = self.hop_call(t, pick, cards, problem)
        if ans is None:
            t['fails'] += 1
            if t['fails'] >= 2:
                t['judged'] |= set(pick)
                self.stop(t, 'hop check returned nothing')
            return
        tid, goto, value, paired = ans
        self.cand_read |= set(pick)
        if problem:
            self.wire.note('ref_hop_problem', problem=problem)
            if problem.startswith('target'):
                t['judged'].add(tid)
            else:
                t['judged'] |= set(pick)
                if tid and tid not in t['chain']:
                    t['chain'].append(tid)
                self.stop(t, problem)
            return
        if tid is None:
            t['judged'] |= set(pick)
            return
        t['judged'] |= set(pick)
        if tid in t['chain']:
            self.stop(t, 'notes loop back')
            return
        if goto:
            self.follow(t, tid, goto)
        else:
            self.resolve(t, tid, value, paired)

    def plain_record(self, d, t):
        if not self.rec_value(d, t['prop']):
            return False
        notes, _ = self.rec_notes(d)
        low = notes.lower()
        pw = [w for w in prop_words(t['prop']) if len(w) >= 3]
        if any((re.search('\\b' + re.escape(w[:4]), low) for w in pw)):
            return False
        own = self.rec_sellers(d)
        for sent in re.split('(?<=[.!?])\\s+', notes):
            for w in re.findall('[A-Za-z][a-z]+', sent)[1:]:
                k = seller_key(w)
                if w[0].isupper() and len(k) >= 4 and (k not in own):
                    return False
        return True

    def hop_call(self, t, pick, cards, reason):
        self.hop_calls += 1
        recs = []
        for c in pick:
            d = cards.get(c, {})
            pm = prop_map(d) or {}
            notes, desc = self.rec_notes(d)
            card = {'id': c, 'sold_by': '/'.join((str(v) for k, v in d.items() if isinstance(v, str) and any((s in k.lower() for s in _SELLER_STEMS)) and ('id' not in k.lower()))), 'category': record_label(d), 'title': record_title(d)[:160], 'facts': {k: str(v)[:90] for k, v in list(pm.items())[:40] if not isinstance(v, (dict, list))}, 'notes': (notes or desc)[:900]}
            recs.append(compact(card, 5000))
        text = PROMPT_HOP % (t['prop'], self.phrase(t['target']), t['prop'], t['prop']) + '\n\nRECORDS\n' + '\n'.join(recs)
        if reason:
            text += '\n\nYour previous answer was unusable (%s). Answer again from the records.' % reason
        ans = self.wire.consult(text, ROLE_HOP, CAP_HOP)
        self.wire.note('ref_hop', req=t['req'], answer=ans)
        if not isinstance(ans, dict):
            return (None, None)
        tid = ans.get('target')
        tid = str(tid).strip() if isinstance(tid, (str, int)) and (not isinstance(tid, bool)) and str(tid).strip() else None
        missing = [str(m) for m in ans.get('missing') or [] if str(m).strip()] if isinstance(ans.get('missing'), list) else []
        if tid is None or tid not in pick or missing:
            return ((None, None, '', []), None)
        d = cards.get(tid, {})
        if not all((self.shows_detail(d, x) for x in t['target']['details'])):
            return ((tid, None, '', []), 'target: some described detail is absent, so it is the wrong listing')
        mine = self.full_count(d, t['target']['details'])
        for c in pick:
            if c != tid and self.full_count(cards.get(c, {}), t['target']['details']) > mine:
                return ((tid, None, '', []), 'target: %s shows more of the described details' % c)
        notes, desc = self.rec_notes(d)
        texts = notes or desc
        goto = ans.get('go_to') if isinstance(ans.get('go_to'), dict) else None
        if goto and str(goto.get('seller') or '').strip():
            gs = str(goto['seller']).strip()
            if not re.search('\\b' + re.escape(gs.lower()) + "(?:'s)?\\b", texts.lower()):
                return ((tid, None, '', []), 'answer: the redirect seller does not appear in the notes')
            ok = False
            for sent in re.split('(?<=[.!?])\\s+', texts):
                if gs.lower() in sent.lower() and any((re.search('\\b' + re.escape(w[:4]), sent.lower()) for w in prop_words(t['prop']) if len(w) >= 3)):
                    ok = True
            if not ok:
                return ((tid, None, '', []), 'answer: the sentence naming that seller concerns a different property')
            dets = []
            tt = set(ref_tokens(texts))
            for x in (goto.get('details') or [])[:6]:
                xt = ref_tokens(str(x))
                if xt and sum((1 for w in xt if w in tt)) * 2 >= len(xt):
                    dets.append(str(x))
            g = {'seller': gs, 'kind': str(goto.get('kind') or '')[:80], 'details': dets}
            if seller_key(gs) in self.rec_sellers(d) and self.target_key(g) == self.target_key(t['target']):
                return ((tid, None, '', []), 'answer: the redirect points back at the listing itself')
            return ((tid, g, '', []), None)
        value = str(ans.get('value') or '').strip()
        pm = prop_map(d) or {}
        pool = set(ref_tokens(' '.join((str(k) + ' ' + str(v) for k, v in pm.items())) + ' ' + record_title(d)))
        if not value or not all((w in pool for w in ref_tokens(value))):
            return ((tid, None, '', []), 'answer: the value does not appear in the record')
        paired = []
        tt = set(ref_tokens(texts))
        for x in ans.get('goes_with') or [] if isinstance(ans.get('goes_with'), list) else []:
            xt = set(ref_tokens(str(x)))
            if xt and xt <= tt and (xt != set(ref_tokens(value))) and (str(x) not in paired):
                paired.append(str(x))
        return ((tid, None, value, paired[:3]), None)

    def follow(self, t, tid, goto):
        t['chain'].append(tid)
        key = self.target_key(goto)
        if key in t['seen']:
            self.stop(t, 'the notes loop back to an earlier listing')
            return
        if t['hops'] + 1 >= 8:
            self.stop(t, 'too many redirects in a row')
            return
        t['seen'].add(key)
        t['hops'] += 1
        t['target'] = goto
        t['judged'], t['queries'], t['pages'], t['fails'], t['not_indexed'] = (set(), [], 0, 0, False)
        self.wire.note('ref_follow', req=t['req'], to=goto)
        self.ref_tell('%s is the listing named for %s; its notes send the reader on to %s, which the code now pursues.' % (tid, t['head'], self.phrase(goto)))

    def resolve(self, t, tid, value, paired):
        if not value:
            self.stop(t, 'the target listing shows no value')
            return
        t['chain'].append(tid)
        t['status'] = 'resolved'
        t['value'] = value
        t['paired'] = paired
        if t['relation'] == 'within' and t['tol'] is not None:
            m = re.search('-?\\d[\\d,]*(?:\\.\\d+)?', value)
            if m:
                x = float(m.group(0).replace(',', ''))
                unit = (re.match('\\s*([a-z%]+)', value[m.end():].lower()) or [None, None])[1] or t['unit'] or ''
                txt = '%s to %s %s (within %s of %s)' % ('%g' % round(x - t['tol'], 6), '%g' % round(x + t['tol'], 6), unit, '%g' % t['tol'], value)
            else:
                txt = value
        elif t['relation'] in ('at most', 'at least'):
            txt = '%s %s' % (t['relation'], value)
        elif t['relation'] == 'differ':
            vals = [value] + [p for p in paired if p.lower() != value.lower()]
            txt = ' and '.join(('not %s' % v for v in vals))
        else:
            vals = [value] + [p for p in paired if p.lower() != value.lower()]
            txt = ' or '.join(vals)
        t['resolved'] = '%s: %s' % (t['head'], txt)
        self.rewrite(t)
        self.needs.resolved.append(t['resolved'])
        self.needs.version += 1
        self.wire.note('ref_resolved', req=t['req'], text=t['resolved'], chain=t['chain'])
        msg = 'resolved: %s comes from another listing; %s shows %s%s; the requirement now reads "%s".' % (t['head'], tid, value, ' (reached from the first listing through its notes)' if len(t['chain']) > 1 else '', t['resolved'])
        if t['relation'] == 'differ':
            msg += " The shopper's value has to DIFFER from it; any other value is fine."
        self.ref_tell(msg)

    def stop(self, t, reason):
        if t['status'] != 'open':
            return
        t['status'] = 'stuck'
        t['reason'] = reason
        self.wire.note('ref_stuck', req=t['req'], reason=reason)
        self.ref_tell('unresolved: %s (%s); its value is unknown, so never decline and never reject a listing on this ground.' % (t['head'], reason))

    def rewrite(self, t):

        def belongs(r):
            if r == t['req'] or r == t['resolved']:
                return True
            val = r.split(':', 1)[1].lower() if ':' in r else ''
            sk = seller_key(t['first']['seller'])
            return strict_same(pname(r), t['prop']) and (not ptag(r) or not t['tag'] or kind_set(ptag(r)) & kind_set(t['tag'])) and (REF_RX.search(val) or (sk and sk in re.sub('[^a-z0-9]', '', val)))
        changed = False
        for lst in (self.needs.stated, self.needs.revealed):
            new = []
            for r in lst:
                if belongs(r):
                    changed = True
                    r = t['resolved']
                if r not in new:
                    new.append(r)
            lst[:] = new
        if self.needs.current is not None:
            self.needs.current = [t['resolved'] if belongs(r) else r for r in self.needs.current]
            if t['resolved'] not in self.needs.current:
                self.needs.current.append(t['resolved'])
        if not changed and t['resolved'] not in self.needs.revealed:
            self.needs.revealed.append(t['resolved'])
        self.needs.reading['requirements'] = list(self.needs.stated)
        for it in self.needs.reading.get('items') or []:
            if not isinstance(it, dict) or not isinstance(it.get('requirements'), list):
                continue
            if len(self.needs.reading.get('items')) >= 2 and t['tag'] and (not kind_set(str(it.get('product'))) & kind_set(t['tag'])):
                continue
            it['requirements'] = ['%s: %s' % (r.split(':', 1)[0], t['resolved'].split(':', 1)[1].strip()) if belongs(r) else r for r in it['requirements']]

    def ref_tell(self, text):
        if len(self.chat) >= 3 and self.chat[-1].get('role') in ('tool', 'user') and isinstance(self.chat[-1].get('content'), str):
            self.chat[-1]['content'] += '\n[reference] ' + text
        else:
            self.ref_pending.append(text)

    def ref_status(self):
        try:
            parts = []
            for t in self.trackers:
                if t['status'] == 'resolved':
                    parts.append('%s (from %s)%s' % (t['resolved'], t['chain'][-1] if t['chain'] else '?', '; must differ' if t['relation'] == 'differ' else ''))
                elif t['status'] == 'open':
                    parts.append('%s: code is pursuing %s' % (t['head'], self.phrase(t['target'])))
                elif t['status'] == 'stuck':
                    parts.append('%s: value unknown; it justifies no decline or rejection' % t['head'])
            out = '\n[references] ' + '; '.join(parts) if parts else ''
            for x in self.ref_pending:
                out += '\n[reference] ' + x
            self.ref_pending = []
            return out
        except Exception:
            return ''

    def ref_block(self, lid):
        try:
            if not self.trackers:
                return self.ref_note()
            lines = ["BORROWED REQUIREMENTS (values taken from other listings; code traced those listings' notes in the store records):"]
            remember = []
            for t in self.trackers:
                if t['status'] == 'dropped':
                    continue
                if t['status'] == 'resolved' and t['relation'] == 'differ':
                    lines.append("- %s: the shopper's %s must differ from what %s shows (%s); every other value satisfies it; all chain listings (%s) carry it." % (t['resolved'], t['prop'], t['chain'][-1], t['value'], ', '.join(t['chain'])))
                elif t['status'] == 'resolved':
                    lines.append('- %s: equals what %s shows%s.' % (t['resolved'], t['chain'][-1], '; each paired value satisfies it' if t['paired'] else ''))
                else:
                    lines.append("- %s is not settled by code (%s); should a record shown here be that very listing (or the chain's last one) and state its own value, take it; otherwise treat it as unknown and never let it alone fail an item." % (t['head'], ' -> '.join(t['chain']) if t['chain'] else 'its listing never turned up'))
                    remember += [c for c in t['chain'][-2:] if c != lid]
            out = '\n'.join(lines)
            for c in remember[:2]:
                recs = self.records.mentions(c, 1)
                if recs:
                    out += '\nRECORD OF %s: %s' % (c, recs[-1])
            return out
        except Exception:
            return self.ref_note()

    def chain_reason(self, lid):
        for t in self.trackers:
            if t['relation'] != 'differ' or t['status'] == 'dropped' or lid not in t['chain']:
                continue
            if self.needs.products() and t['tag']:
                k = self.product_of_listing(lid)
                tk = self.product_index(t['tag'])
                if k is not None and tk is not None and (k != tk):
                    continue
            return '%s is the listing the reference for %s points at (or one its notes pass through), so its value equals the very value%s the shopper rules out; it cannot satisfy that requirement.' % (lid, t['head'], ' (%s)' % t['value'] if t['value'] else '')
        return ''

    def chain_listings(self):
        out = set()
        for t in self.trackers:
            if t['relation'] == 'differ' and t['status'] != 'dropped':
                out |= set(t['chain'])
        return out

    def off_category(self):
        try:
            if not self.cand_read:
                return set()
            homes = self.home_labels()
            if not homes or any((not h for h in homes)):
                return set()
            allh = set().union(*homes)
            out = set()
            for lid in self.cand_read:
                lab = self.category_of(lid)
                if lab and lab not in allh and (lid not in self.cart.units):
                    out.add(lid)
            return out
        except Exception:
            return set()

    def open_differ(self):
        return [t for t in self.trackers if t['relation'] == 'differ' and t['status'] in ('open', 'stuck') and self.tracker_applies(t)]

def exact_run_named(prop, text, exact=False):
    pw = [w for w in re.findall('[a-z0-9]+', split_camel((prop or '').split('(')[0]).lower().replace('_', ' '))]
    if sum((len(w) for w in pw)) < 3:
        return -1
    tw = re.findall('[a-z0-9]+', (text or '').lower())

    def eq(a, b):
        if a == b:
            return True
        if exact or len(a) < 3 or len(b) < 3:
            return False
        return a + 's' == b or b + 's' == a or a + 'es' == b or (b + 'es' == a)
    for i in range(len(tw) - len(pw) + 1):
        if all((eq(tw[i + j], pw[j]) for j in range(len(pw)))):
            return i
    return -1

class SelfRedirectPart:

    def self_redirects(self, lid):
        key = (lid, len(self.records.obs), self.needs.version)
        if key in self._redir_cache:
            return self._redir_cache[key]
        out = []
        try:
            cards = self.ref_index()
            d = cards.get(lid)
            if d:
                notes, _ = self.rec_notes(d)
                if notes:
                    own = self.rec_sellers(d)
                    others = set()
                    for x, dd in cards.items():
                        if x != lid:
                            others |= self.rec_sellers(dd)
                    pm = prop_map(d) or {}
                    ftoks = set()
                    for k, v in pm.items():
                        ftoks |= set(re.findall('[a-z0-9]+', split_camel(k).lower().replace('_', ' ')))
                        ftoks |= set(re.findall('[a-z0-9]+', str(v).lower()))
                    ftoks |= set(re.findall('[a-z0-9]+', (record_title(d) + ' ' + record_label(d)).lower()))
                    ps = self.needs.products()
                    k_own = self.product_of_listing(lid) if ps else None
                    sentences = [s for s in re.split('(?<=[.!?;])\\s+', notes) if s.strip()]
                    for r in self.needs.binding():
                        if ':' not in r:
                            continue
                        if ps and ptag(r) and (k_own is not None) and (self.product_index(r) not in (k_own, None)):
                            continue
                        prop = pname(r)
                        found = None
                        for sent in sentences:
                            masked = re.sub('\\b\\w+(?=\\s+by\\s+[A-Z])', ' ', sent)
                            words = re.findall("[A-Za-z0-9']+", sent)
                            mwords = re.findall("[A-Za-z0-9']+", masked)
                            pos = exact_run_named(prop, ' '.join(mwords))
                            if pos < 0:
                                continue
                            start = pos + len(re.findall('[a-z0-9]+', prop.lower()))
                            mlow = [w.lower() for w in mwords]
                            for i in range(max(1, start), len(mwords)):
                                w = mwords[i]
                                red = seller_key(w)
                                foreign = red in others or (w[0].isupper() and len(red) >= 4 and (red not in own) and (red not in ftoks) and (w.lower() not in ftoks))
                                if not foreign:
                                    continue
                                if any((n in mlow[max(0, i - 5):i] for n in ('nor', 'neither', 'not', 'never', 'no'))):
                                    continue
                                found = (r, prop, sent.strip())
                                break
                            if found:
                                break
                        if found:
                            out.append(found)
        except Exception:
            out = []
        self._redir_cache[key] = out
        return out

    def redirect_remark(self, name, args):
        R = self.role
        if name in (R('search'), R('cart'), R('cart_view'), R('message')) or name in self.finishers():
            return ''
        out = []
        ver = self.needs.version
        for lid in self.ids_named(args):
            if not (lid in self.cart.units or (lid, ver) in self.verified):
                continue
            if (lid, ver) in self.redirect_warned:
                continue
            self.redirect_warned.add((lid, ver))
            found = self.self_redirects(lid)
            if not found:
                continue
            self.verified.pop((lid, ver), None)
            self.wire.note('self_redirect', id=lid, props=[f[1] for f in found])
            out.append('[notes] %s\'s notes may route %s to some other listing ("%s"); if so, this listing\'s field no longer decides that requirement because grading reads the other listing, and it qualifies only if that other record shows the needed value; the item check reruns before ordering%s.' % (lid, ', '.join((f[1] for f in found)), found[0][2][:160], '; as a cart line, swap it for another verified fit in that case' if lid in self.cart.units else ''))
        return ' '.join(out)

    def redirects_property(self, lid, prop):
        try:
            for r, p, s in self.self_redirects(lid):
                if loose_same(pname(r), prop) or loose_same(p, prop):
                    return True
            return False
        except Exception:
            return True
_SIMPLE_BAD = re.compile('\\b(?:at least|at most|up to|between|no more than|no less than|or more|or less|under|over|below|above|less than|more than|within|minimum|maximum|about|roughly|approximately|around)\\b|\\d\\s*(?:-|to)\\s*\\d|[+<>]')

def simple_value(v):
    return not _SIMPLE_BAD.search((v or '').lower())

def wordy_value(v):
    low = (v or '').lower()
    return bool(re.search('[a-z]{3}', low)) and (not re.search('\\d', low)) and simple_value(v) and (not EXCLUSION_RX.search(low)) and (not is_reference(low))

class MeaningPart:

    def product_needs(self, k):
        out = []
        ps = self.needs.products()
        for r in self.needs.binding():
            if ':' not in r:
                continue
            if ps and ptag(r) and (self.product_index(r) not in (k, None)):
                continue
            out.append((pname(r), pvalue(r), r not in self.needs.stated))
        return out

    def pool(self, k):
        ps = self.needs.products()
        if ps:
            homes = self.home_labels()
            shelves = homes[k] if k < len(homes) else set()
        else:
            read = self.records.categories_read()
            shelves = {read[0]} if read else self.home_labels()[0] if self.home_labels() else set()
        if not shelves:
            return {}
        units = self.units_for(k if ps else None)
        judged = [(p, v) for p, v in self.judged_values(k if ps else None) if simple_value(v)]
        revealed = [(p, v) for p, v, rv in self.product_needs(k) if rv]
        cap = self.needs.budget()
        ver = self.needs.version
        chain = self.chain_listings()
        out = {}
        for lid, row in self.table().items():
            if row['label'] not in shelves:
                continue
            price = row['price'] if row['price'] is not None else self.records.price(lid)
            opened = lid in self.records.read
            gone = lid in self.dead or row['stock'] is False or lid in self.refused_lines
            over = cap is not None and price is not None and (price * units > cap + 1e-09)
            conflicts = [p for p, v in judged if self.clashes(lid, p, v)]
            gaps = [p for p, v in judged if not self.shows(lid, p, v)]
            objected = self.failed_now(lid) or (opened and bool(self.misses(lid)))
            verified = (lid, ver) in self.verified and (not gone)
            fit = (verified or (opened and (not gaps) and (not conflicts))) and (not (objected or lid in chain or gone or over))
            out[lid] = {'price': price, 'opened': opened, 'gone': gone, 'over': over, 'chain': lid in chain, 'conflicts': conflicts, 'gaps': gaps, 'objected': objected, 'verified': verified, 'fit': fit, 'excluded': gone or over or lid in chain or objected or bool(conflicts), 'revealed_shown': all((self.shows(lid, p, v) for p, v in revealed))}
        return out

    def rows_lacking(self):
        try:
            ps = self.needs.products()
            if not ps:
                return []
            homes = self.home_labels()
            covered = {self.category_of(x) for x in self.verified_now()} | {self.category_of(x) for x in self.audit_fits}
            lacking = [k for k, h in enumerate(homes) if h and (not h & covered)]
            if not lacking:
                return []
            mt = self.multi_tool()
            want = mt[2] if mt else 3
            seen, order, gone = ({}, [], set())
            for ob in reversed(self.records.obs):
                for d in dicts_in(ob):
                    rid = own_id(d)
                    if not rid or rid not in self.records.known_set:
                        continue
                    if rid not in seen:
                        seen[rid] = record_label(d)
                        order.append(rid)
                    elif not seen[rid]:
                        seen[rid] = record_label(d)
                    if record_stock(d) is False:
                        gone.add(rid)
            cap = self.needs.budget()
            per = {k: [] for k in lacking}
            for rid in order:
                if rid in self.records.read or rid in self.refused_lines or rid in gone or (rid in self.dead):
                    continue
                lab = seen.get(rid) or self.category_of(rid)
                for k in lacking:
                    if lab in homes[k]:
                        p = self.records.price(rid)
                        if cap is None or p is None or p * self.units_for(k) <= cap:
                            per[k].append(rid)
                        break
            out = []
            while len(out) < want and any(per.values()):
                for k in lacking:
                    if per[k] and len(out) < want:
                        out.append(per[k].pop(0))
            return out
        except Exception as exc:
            self.wire.note('rows_lacking_error', error=str(exc)[:200])
            return []

    def slack_rows(self):
        try:
            ps = self.needs.products()
            cap = self.needs.budget()
            if not ps or cap is None:
                return ([], 0)
            floors, pools = ([], [])
            for k in range(len(ps)):
                tab = self.pool(k)
                fits = [r['price'] for r in tab.values() if r['fit'] and r['price'] is not None]
                if not fits:
                    return ([], 0)
                floors.append(min(fits) * self.units_for(k))
                pools.append(tab)
            if sum(floors) <= cap:
                return ([], 0)
            per = []
            for k, tab in enumerate(pools):
                units = self.units_for(k)
                slack = (cap - (sum(floors) - floors[k])) / float(units)
                c = sorted(((r['price'], lid) for lid, r in tab.items() if not r['opened'] and (not r['gone']) and (not r['chain']) and (not r['objected']) and (not r['conflicts']) and (r['price'] is not None) and (slack > 0) and (r['price'] <= slack)))
                per.append([lid for _, lid in c])
            out = []
            while len(out) < 3 and any(per):
                for p in per:
                    if p and len(out) < 3:
                        out.append(p.pop(0))
            return (out, sum(floors))
        except Exception as exc:
            self.wire.note('slack_rows_error', error=str(exc)[:200])
            return ([], 0)

    def news(self, k):
        if self.refusals:
            return True
        ps = self.needs.products()
        for n in self.needs.revealed + self.needs.dropped:
            if not ps:
                return True
            c = self.product_index(n) if ptag(n) else None
            if c is None or c == k:
                return True
        return False

    def latest_problems(self, lid):
        for rec in reversed(self.log):
            if rec.get('kind') == 'item_check' and rec.get('id') == lid and isinstance(rec.get('answer'), dict):
                a = rec['answer']
                if a.get('fit') is False:
                    return [str(p) for p in a.get('problems') or [] if str(p).strip()]
                return []
        return []

    def meaning_check(self):
        if self.meaning_done or not (self.role('cart') and self.role('order')):
            return ''
        if self.steps_left() < 4 or self.wire.time_left() <= CLOSING + 30 or self.wire.mute or self.covering or self.wire.finished or self.wire.spent_out() or (not self.wire.usable(ROLE_CHECKER)):
            return ''
        ps = self.needs.products()
        n = len(ps) if ps else 1
        chosen, cands = ({}, {})
        for k in range(n):
            tab = self.pool(k)
            fits = sorted(((r['price'], lid) for lid, r in tab.items() if r['fit'] and r['price'] is not None))
            if fits:
                chosen[k] = fits[0][1]
                continue
            if not self.news(k):
                return ''
            cl = []
            for lid, r in tab.items():
                if not r['opened'] or r['gone'] or r['over'] or r['chain'] or (r['price'] is None):
                    continue
                fields = (self.table().get(lid) or {}).get('fields') or {}
                lines, ok = ([], True)
                for prop, val, rv in self.product_needs(k if ps else None):
                    if self.shows(lid, prop, val):
                        continue
                    got = [str(v) for kk, v in fields.items() if loose_same(kk, prop) and str(v).strip()]
                    if rv or not got or (not wordy_value(val)) or any((re.search('\\d', g) for g in got)):
                        ok = False
                        break
                    lines.append((prop, val, '; '.join(dict.fromkeys(got))[:120]))
                if not ok or not lines:
                    continue
                if r['objected']:
                    qprops = [p for p, _, _ in lines]
                    marks = set()
                    for p, v, _ in lines:
                        marks |= {w for w in re.findall('[a-z]+', (p + ' ' + v).lower()) if len(w) >= 3 and w not in LINK}
                    if not all((any((loose_same(m.split(':')[0], q) for q in qprops)) for m in self.misses(lid))):
                        continue
                    probs = self.latest_problems(lid)
                    if not all((any((re.search('\\b' + re.escape(w), pr.lower()) for w in marks)) for pr in probs)):
                        continue
                cl.append((len(lines), r['price'], lid, lines))
            cl.sort()
            cands[k] = cl[:6 if n == 1 else 3]
            if not cands[k]:
                return ''
        if not cands:
            return ''
        cap = self.needs.budget()
        if cap is not None:
            total = sum(((self.records.price(l) or 0) * self.units_for(k if ps else None) for k, l in chosen.items()))
            total += sum((cl[0][1] * self.units_for(k if ps else None) for k, cl in cands.items()))
            if total > cap:
                return ''
        flat = []
        for k, cl in cands.items():
            for _, _, lid, lines in cl:
                for prop, val, ftxt in lines:
                    flat.append((lid, prop, val, ftxt))
        flat = flat[:12]
        self.meaning_done = True
        text = "A shopping assistant is about to decline because the listings below spell a requested value differently from the request.\n\nTHE SHOPPER'S REQUEST\n" + self.request[:1500] + "\n\nEach numbered line names a listing, a property with the shopper's wanted value, and the text in that listing's field.\n" + '\n'.join(('%d. %s | %s: "%s" | field text: %s' % (i + 1, lid, prop, val[:60], ftxt) for i, (lid, prop, val, ftxt) in enumerate(flat))) + '\n\nFor each line decide whether the listing\'s text expresses the requested value by other means (a synonym, another spelling or phrasing, or a narrower, sharper form of that very thing) or names some other value for that property; when in doubt, answer false.\nAnswer with one JSON object only: {"lines": [{"line": <number>, "same": true or false}]}.'
        ans = self.wire.consult(text, ROLE_CHECKER, CAP_MEANING)
        conf = set()
        if isinstance(ans, dict) and isinstance(ans.get('lines'), list):
            for e in ans['lines']:
                if isinstance(e, dict) and e.get('same') is True and is_num(e.get('line')):
                    conf.add(int(e['line']))
        self.wire.note('meaning_check', lines=len(flat), confirmed=sorted(conf))
        picks = dict(chosen)
        for k, cl in cands.items():
            got = None
            for _, _, lid, lines in cl:
                idx = [i + 1 for i, f in enumerate(flat) if f[0] == lid]
                if len(idx) == len(lines) and all((i in conf for i in idx)):
                    got = lid
                    break
            if got is None:
                return ''
            picks[k] = got
        lines = [(picks[k], self.units_for(k if ps else None)) for k in sorted(picks)]
        if cap is not None and sum(((self.records.price(l) or 0) * u for l, u in lines)) > cap:
            return ''
        if self.steps_left() < len(lines) + 2:
            return ''
        ver = self.needs.version
        for k, cl in cands.items():
            lid = picks[k]
            for key in list(self.verdicts):
                if key[0] == lid and key[1] == ver:
                    self.verdicts[key] = ''
            self.verified[lid, ver] = self.records.price(lid)
            self.meaning_ok.add((lid, ver))
        saved = self.wire.mute
        self.wire.mute = True
        try:
            rej = self.cart_match(lines)
            ok = rej == [] and all((l in self.cart.units for l, _ in lines))
            if ok:
                stale = [l for l, _ in lines if self.stale(l)]
                if stale and self.steps_left() >= len(stale) + 2:
                    gone = self.stock_reread(stale) or []
                    ok = not gone
            ok = ok and bool(self.cart.units)
        finally:
            self.wire.mute = saved
        self.wire.note('meaning_cart', ok=ok, lines=lines)
        if not ok:
            return ''
        self.meaning_cart = (ver, self.refusals, [l for l, _ in lines])
        return "the meaning check found that each chosen listing's field text expresses the asked-for value in different words; each such listing satisfies every revealed need, is available and within the budget, so the code loaded this cart and checked availability: %s. Purchase this cart now and give up only when an already-read record proves a line breaks an active need." % self.cart.summary()

    def near_match_hold(self):
        try:
            mc = self.meaning_cart
            if not mc or (mc[0], mc[1]) != (self.needs.version, self.refusals) or (not self.role('order')):
                return ''
            if not all((l in self.cart.units and l not in self.dead and (l not in self.refused_lines) for l in mc[2])):
                return ''
            if self.steps_left() < 1 or self.wire.time_left() < 2 or self.wire.finished or self.wire.blocked:
                return ''
            self.meaning_cart = None
            self.wire.note('near_match_buy', lines=mc[2])
            saved = self.wire.mute
            self.forced_finish()
            self.wire.mute = saved
            return 'an earlier meaning check found the carried lines express the asked-for values in different words, with no change since then; therefore the code bought instead of giving up.'
        except Exception as exc:
            self.wire.note('near_match_error', error=str(exc)[:200])
            return ''

def stem_word(w):
    return w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w

class CheapPart:

    def free_text(self, args):
        s = self.role('search')
        props = (self.shapes.get(s) or {}).get('properties') or {}
        pg = self.paging_param()
        return ' '.join((str(v) for k, v in (args or {}).items() if isinstance(v, str) and v.strip() and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and (not pg or k != pg[0])))

    def lead_tokens(self, args, width=None):
        toks = [t for t in re.findall('[a-z0-9]+', self.free_text(args).lower()) if len(t) >= 2]
        w = width or (self.read_width or None)
        return toks[:w] if w else toks

    def mode_of(self, args):
        s = self.role('search')
        props = (self.shapes.get(s) or {}).get('properties') or {}
        return tuple(sorted(((k, str(v)) for k, v in (args or {}).items() if isinstance(props.get(k), dict) and isinstance(props[k].get('enum'), list) and (len(props[k]['enum']) <= 4))))

    def already_run(self, text):
        w = self.read_width or 3
        mine = {stem_word(t) for t in re.findall('[a-z0-9]+', text.lower())[:w]}
        for e in self.search_log.values():
            if {stem_word(t) for t in re.findall('[a-z0-9]+', self.free_text(e['args']).lower())[:w]} == mine:
                return True
        for rec in self.log:
            if rec.get('kind') == 'env' and rec.get('name') == self.role('search'):
                t = self.free_text(rec.get('args') or {})
                if {stem_word(x) for x in re.findall('[a-z0-9]+', t.lower())[:w]} == mine:
                    return True
        return False

    def id_world(self):
        flags = [e['sorted'] for e in self.search_log.values() if e['sorted'] is not None]
        return flags.count(True) >= 2 and False not in flags

    def strict_world(self):
        return self.word_rule == 'strict' or (self.read_width > 0 and self.word_rule != 'loose')

    def walk_labels(self, k):
        homes = self.home_labels()
        if self.needs.products():
            return homes[k] if k is not None and k < len(homes) else set()
        hs = homes[0] if homes else set()
        if hs:
            return hs
        read = self.records.categories_read()
        return {read[0]} if read else set()

    def carry(self, args, k, pairs=False):
        words = self.lead_tokens(args)
        alltok = re.findall('[a-z0-9]+(?:\\.[0-9]+)?', self.free_text(args).lower())
        if self.read_width:
            alltok = alltok[:self.read_width]
        out = set()
        ps = self.needs.products()
        for i, r in enumerate(self.needs.binding()):
            if ':' not in r:
                continue
            if ps and k is not None and ptag(r) and (self.product_index(r) not in (k, None)):
                continue
            val = pvalue(r).lower()
            for j, alt in enumerate(re.split('[,;/]|\\bor\\b|\\beither\\b|\\bnor\\b', val)):
                alt = alt.strip()
                if not alt:
                    continue
                m = re.fullmatch('(\\d+)\\s+([a-z]+)', alt)
                if m and (not LIMIT_RX.search(alt)):
                    if m.group(1) in alltok and any((t == m.group(2) or t.startswith(m.group(2)[:3]) or m.group(2).startswith(t) for t in alltok if t.isalpha())):
                        out.add((i, j) if pairs else i)
                    continue
                if re.search('\\d', alt) and (LIMIT_RX.search(alt) or re.match('\\d+(?:\\.\\d+)?\\s*[a-z]', alt)):
                    continue
                aw = [w for w in re.findall('[a-z0-9]+', alt) if len(w) >= 3]
                if any((len(min(a, q, key=len)) >= 3 and (a.startswith(q) or q.startswith(a)) for a in aw for q in words)):
                    out.add((i, j) if pairs else i)
        return out

    def pages_left(self, e):
        if not e['size']:
            return 0
        return max(0, -(-e['total'] // e['size']) - len(e['visited']))

    def exhausted(self, e):
        return e['sorted'] is True and e['first'] and (self.next_page(e) is None) and self.lead_tokens(e['args']) and (len(e['seen']) >= e['total'])

    def walk_options(self, k):
        try:
            labels = self.walk_labels(k)
            if self.needs.products() and (not labels):
                return []
            vstems, lstems = (set(), set())
            for r in self.needs.binding():
                if ':' in r and (not self.needs.products() or not ptag(r) or self.product_index(r) in (k, None)):
                    vstems |= {stem_word(w) for w in re.findall('[a-z0-9]+', r.split(':', 1)[1].lower()) if len(w) >= 2}
            for lab in labels:
                lstems |= {stem_word(w) for w in re.findall('[a-z0-9]+', lab) if len(w) >= 2}
            exh = [e for e in self.search_log.values() if self.exhausted(e)]
            opts = []
            for key, e in self.search_log.items():
                if e['sorted'] is not True or not e['first'] or self.next_page(e) is None:
                    continue
                cc = len(self.carry(e['args'], k))
                if cc == 0 or not any((l in labels for l in e['row_labels'].values())):
                    continue
                words = {stem_word(w) for w in self.lead_tokens(e['args'])}
                covered = False
                for x in exh:
                    if self.mode_of(x['args']) != self.mode_of(e['args']):
                        continue
                    xw = {stem_word(w) for w in self.lead_tokens(x['args'])}
                    if xw and xw <= words:
                        covered = True
                    elif xw and xw <= vstems | lstems and (self.carry(e['args'], k, True) <= self.carry(x['args'], k, True)):
                        covered = True
                if covered:
                    continue
                pl = self.pages_left(e)
                if 1 <= pl <= 12:
                    opts.append({'key': key, 'e': e, 'pages': pl, 'carry': cc, 'words': words})
            uniq = []
            for o in sorted(opts, key=lambda o: (o['pages'], o['e']['order'])):
                if any((u['e']['total'] == o['e']['total'] and u['e']['first'] == o['e']['first'] for u in uniq)):
                    continue
                uniq.append(o)
            if self.strict_world():
                drop = set()
                for a in uniq:
                    for b in uniq:
                        if a is b or not a['words'] < b['words']:
                            continue
                        extra = b['words'] - a['words']
                        if extra <= vstems | lstems:
                            drop.add(a['key'])
                        elif a['pages'] <= 6:
                            drop.add(b['key'])
                uniq = [o for o in uniq if o['key'] not in drop]
            uniq.sort(key=lambda o: (o['pages'], -o['carry'], compact(o['e']['args'])))
            return uniq
        except Exception as exc:
            self.wire.note('walk_options_error', error=str(exc)[:200])
            return []

    def page_plan(self, budget, only=None):
        try:
            if budget <= 0 or self.needs.priority() != 'cheapest' or (not self.role('search')) or (not self.paging_param()) or (not self.id_world()):
                return []
            ps = self.needs.products()
            prods = list(range(len(ps))) if ps else [None]
            bids = []
            for k in prods:
                if k in self.walked or (only is not None and k not in only):
                    continue
                opts = self.walk_options(k)
                if opts:
                    bids.append((opts[0]['pages'], 0 if k is None else 1, -1 if k is None else k, opts[0]))
            bids.sort(key=lambda b: b[:3])
            plan, remaining = ([], min(budget, 8))
            for pages, _, k, opt in bids:
                take = min(pages, 6, remaining)
                if take <= 0:
                    break
                plan.append((None if k == -1 else k, opt['key'], take))
                remaining -= take
            return plan
        except Exception as exc:
            self.wire.note('page_plan_error', error=str(exc)[:200])
            return []

    def pick_of(self, k):
        ps = self.needs.products()
        cands = [l for l in self.cart.units if l not in self.dead and (not ps or self.product_of_listing(l) == k)]
        if not cands:
            cands = [x for x in self.verified_now() if not ps or self.product_of_listing(x) == k]
        priced = sorted(((self.records.price(x), x) for x in cands if self.records.price(x) is not None))
        return priced[0] if priced else (None, None)

    def keyword_template(self):
        s = self.role('search')
        props = (self.shapes.get(s) or {}).get('properties') or {}
        pg = self.paging_param()
        for key, e in sorted(self.search_log.items(), key=lambda kv: -kv[1]['order']):
            a = e['args']
            texts = [k for k, v in a.items() if isinstance(v, str) and v.strip() and (not (isinstance(props.get(k), dict) and 'enum' in props[k])) and (not pg or k != pg[0])]
            if len(texts) == 1:
                return ({k: v for k, v in a.items() if k != texts[0]}, texts[0])
        return None

    def scripted_search(self, args, reason):
        s = self.role('search')
        args, faults = check_args(args, self.shapes.get(s) or {})
        if faults:
            return False
        cid = 'code_s_%d' % self.wire.step
        self.chat.append({'role': 'assistant', 'content': reason, 'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': s, 'arguments': json.dumps(args)}}]})
        n = len(self.records.obs)
        self.covering = True
        try:
            res = self.dispatch([{'id': cid, 'name': s, 'args': json.dumps(args)}], scripted=True)
        finally:
            self.covering = False
        ok = len(self.records.obs) > n
        if ok and res:
            res = [(res[-1][0], self.digest(self.records.obs[-1]) + self._tail_notes(res[-1][1]))]
        self.post(res)
        return ok

    @staticmethod
    def _tail_notes(text):
        i = text.find('\nnote: ')
        return text[i:] if i >= 0 else ''

    def digest(self, ob):
        rows = []
        for d in dicts_in(ob, 3):
            if len(d) >= 4 and any(('id' in k.lower() and v in self.records.known_set for k, v in d.items() if isinstance(v, str))):
                rows.append(d)
            if len(rows) >= 12:
                break
        if not rows:
            return '[page digest] zero rows'
        out = ['[page digest] id | price | title | [category]']
        for d in rows:
            p = record_price(d)
            out.append('%s | %s | %s"%s" [%s]' % (own_id(d) or row_identity(d), money(p) if p is not None else '?', 'SOLD OUT ' if record_stock(d) is False else '', record_title(d)[:70], record_label(d)))
        return '\n'.join(out)

    def line_query(self, k):
        price, pick = self.pick_of(k)
        tmpl = self.keyword_template()
        if not pick or not tmpl:
            return None
        rec = None
        for ob in self.records.obs:
            for d in dicts_in(ob):
                if any((v == pick for v in d.values())) and record_title(d):
                    rec = d
        if not rec:
            return None
        maker = ''
        for kk, v in rec.items():
            if isinstance(v, str) and v.strip() and re.search('brand|make|manufactur', kk.lower()) and ('id' not in kk.lower()):
                maker = v.strip()
                break
        avoid = set(LINK)
        for lab in self.walk_labels(k):
            avoid |= set(re.findall('[a-z]+', lab))
        for r in self.needs.binding():
            avoid |= set(re.findall('[a-z]+', r.lower()))
        for nm in self.product_names():
            avoid |= set(re.findall('[a-z]+', nm.lower()))
        title = record_title(rec)
        if not maker:
            toks = re.findall('[A-Za-z0-9]+', title)
            for i, tk in enumerate(toks):
                low = tk.lower()
                if tk.isalpha() and len(tk) >= 3 and (low not in avoid) and (stem_word(low) not in avoid):
                    q = low
                    if i + 1 < len(toks) and toks[i + 1].lower() not in avoid:
                        q += ' ' + toks[i + 1].lower()
                    return None if self.already_run(q) else q
            return None
        mw = maker.split()[0].lower()
        avoid |= set(re.findall('[a-z]+', maker.lower()))
        words = re.findall("[A-Za-z][A-Za-z'\\-]{3,}", title)
        low = [w.lower() for w in words]
        if mw in low:
            words = words[low.index(mw) + 1:]
        words = [w for w in words if w.lower().strip("'-") not in avoid and "'" not in w]
        if not words:
            return None
        q = '%s %s' % (mw, words[0].lower())
        return None if self.already_run(q) else q

    def walk_rows(self, k, count, extra_keys=()):
        labels = self.walk_labels(k)
        top = 0
        for e in self.search_log.values():
            if e['sorted'] is True:
                top = max(top, len(self.carry(e['args'], k)))
        rows = []
        for key, e in self.search_log.items():
            if e['sorted'] is True and top and (len(self.carry(e['args'], k)) == top) or key in extra_keys:
                rows += [r for r, lab in e['row_labels'].items() if lab in labels]
        price, _ = self.pick_of(k)
        if price is None:
            opened = [self.records.price(r) for r in rows if r in self.records.read and self.records.price(r) is not None]
            price = min(opened) if opened else None
        cap = self.needs.budget()
        units = self.units_for(k)
        keep = []
        for r in dict.fromkeys(rows):
            p = self.records.price(r)
            if r in self.records.read or r in self.refused_lines or p is None or (self.newest_stock(r) is False):
                continue
            if cap is not None and p * units > cap:
                continue
            if price is not None and p >= price:
                continue
            keep.append((p, r))
        return [r for _, r in sorted(keep)][:count]

    def newest_stock(self, lid):
        st = None
        for ob in self.records.obs:
            for d in dicts_in(ob):
                if row_identity(d) == lid:
                    st = record_stock(d)
        return st

    def run_walk(self, plan, floor):
        pg = self.paging_param()
        if not plan or not pg or (not self.role('search')):
            return False
        sent = False
        report = []
        line_keys = {}
        for k, key, take in plan:
            self.walked.add(k)
            e = self.search_log.get(key)
            if not e:
                continue
            got = 0
            while got < take:
                nxt = self.next_page(e)
                if nxt is None or self.steps_left() <= floor + 1 or self.wire.finished or self.wire.blocked or (self.wire.time_left() < CLOSING + 30):
                    break
                args = dict(e['args'])
                args[pg[0]] = nxt
                before = len(e['visited'])
                ok = self.scripted_search(args, 'The lowest price is wanted and hits here come in id order rather than price order, so I fetch another page of this query.')
                if not ok:
                    break
                sent = True
                got += 1
                if len(e['visited']) == before:
                    break
            report.append('%s: %d more pages, %d of %d rows seen' % (self.free_text(e['args']) or compact(e['args']), got, len(e['seen']), e['total']))
        for k, key, take in plan:
            if self.steps_left() <= floor + 2 or self.wire.finished or self.wire.time_left() < CLOSING + 30:
                break
            q = self.line_query(k)
            tmpl = self.keyword_template()
            if not q or not tmpl:
                continue
            args = dict(tmpl[0])
            args[tmpl[1]] = q
            if self.scripted_search(args, "I search the pick's product line (maker plus line name), since sibling sizes or colours are separate listings and might cost less."):
                sent = True
                lk = compact({kk: v for kk, v in sorted(args.items()) if kk != pg[0]})
                line_keys[k] = lk
                e = self.search_log.get(lk)
                extra = 0
                while e and e['size'] and (e['total'] <= 3 * e['size']) and (extra < 2) and (self.next_page(e) is not None) and (self.steps_left() > floor + 3) and (self.wire.time_left() > CLOSING + 30):
                    a2 = dict(e['args'])
                    a2[pg[0]] = self.next_page(e)
                    if not self.scripted_search(a2, 'I read the rest of that short product line.'):
                        break
                    extra += 1
                report.append('line query: %s' % q)
        if not sent:
            return False
        mt = self.multi_tool()
        count = mt[2] if mt else 2
        pools = []
        for k, key, take in plan:
            pools.append(self.walk_rows(k, count, [line_keys[k]] if k in line_keys else []))
        queue = []
        while len(queue) < count and any(pools):
            for p in pools:
                while p and p[0] in queue:
                    p.pop(0)
                if p and len(queue) < count:
                    queue.append(p.pop(0))
        while queue and (1 if mt and len(queue) >= 2 else len(queue)) >= self.steps_left() - floor:
            queue.pop()
        labs = set()
        for k, _, _ in plan:
            labs |= self.walk_labels(k)
        self.ranked = {x for x in self.ranked if self.category_of(x[0]) not in labs}
        self.vetted_gen = -1
        summary = '[cheapest walk] ' + '; '.join(report)
        if queue:
            summary += '; the cheapest unopened available affordable rows under each pick are opened next: %s' % ', '.join(('%s (%s)' % (q, money(self.records.price(q))) for q in queue))
        else:
            summary += '; no unopened available affordable row cheaper than the pick appeared'
        summary += '. Choose, per product, the cheapest listing that satisfies all binding requirements by its own fields; the rank check repeats at cart time and at order time.'
        if self.chat and self.chat[-1].get('role') == 'tool':
            self.chat[-1]['content'] += '\n' + summary + ('' if queue else '\n' + self.status())
        self.wire.note('walk', plan=[(k, take) for k, _, take in plan], opened=queue)
        if queue:
            self.enqueue(queue, 'walk', 'I open the cheapest unopened rows this walk revealed.', recheck=True, floor=floor)
            self.run_queue()
        return True

    def walk_hold(self, ids):
        try:
            if self.needs.priority() != 'cheapest' or self.walk_queued or self.wire.time_left() < CLOSING + 60:
                return ''
            if self.refusals and self.steps_left() <= 15:
                return ''
            ps = self.needs.products()
            prods = set()
            for i in ids:
                if ps:
                    k = self.product_of_listing(i)
                    if k is not None:
                        prods.add(k)
                else:
                    prods.add(None)
            if not prods:
                return ''
            floor = REFUSAL_RESERVE + 1 if self.orders_tried == 0 else 4
            half = (self.steps_left() - floor - 1) // 2
            room = half - (1 if self.multi_tool() else 2)
            plan = self.page_plan(min(8, room), only=prods)
            if not plan:
                return ''
            self.walk_queued = (plan, max(floor, self.steps_left() - half - 1))
            return 'rows in this store come in fixed id order rather than price order, so a cheaper fit could hide on unread pages; the code now pages the value queries and opens the lowest-priced unread rows under the pick; afterwards resend this call, or resend it naming a cheaper fit.'
        except Exception as exc:
            self.wire.note('walk_hold_error', error=str(exc)[:200])
            return ''

    def product_not_found(self):
        try:
            homes = self.home_labels()
            read = set(self.records.categories_read())
            if any((not h or not h & read for h in homes)):
                return True
            vs = [x for x, v in self.verified if v == self.needs.version]
            if vs and all((x in self.refused_lines or x in self.dead for x in vs)):
                return True
            for k, h in enumerate(homes):
                if not any((self.category_of(x) in h for x in self.verified_now())):
                    return True
            return False
        except Exception:
            return False

    def walk_step(self):
        try:
            if self.decline_reads:
                return self.run_decline_reads()
            if self.walk_queued:
                plan, floor = self.walk_queued
                self.walk_queued = None
                return self.run_walk(plan, floor)
            if self.orders_tried or self.needs.priority() != 'cheapest' or self.wire.time_left() < CLOSING + 60 or self.wire.finished:
                return False
            if self.needs.products() and self.product_not_found():
                return False
            reserved = 1 if self.multi_tool() else 2
            plan = self.page_plan(8)
            if not plan:
                return False
            pages = sum((t for _, _, t in plan)) + sum((1 for k, _, _ in plan if self.pick_of(k)[1]))
            if self.steps_left() > self.reserve() + 2 + pages + reserved:
                return False
            plan = self.page_plan(min(8, self.steps_left() - self.reserve() - 1 - reserved))
            if not plan:
                return False
            return self.run_walk(plan, self.reserve())
        except Exception as exc:
            self.wire.note('walk_step_error', error=str(exc)[:200])
            return False

    def decline_read_hold(self):
        try:
            if 'dreads' in self.decline_once or self.wire.time_left() < CLOSING + 60 or self.cart.units or (not (self.role('details') or self.multi_tool())):
                return ''
            mt = self.multi_tool()
            cost = 1 if mt else 2
            room = min(3, self.steps_left() - cost - 6)
            searches = []
            if room > 0 and self.paging_param() and self.role('search') and self.id_world():
                ps = self.needs.products()
                fits = set(self.verified_now()) | {x for x in self.audit_fits if not self.failed_now(x) and x not in self.refused_lines}
                prods = list(range(len(ps))) if ps else [None]
                for k in prods:
                    labs = self.walk_labels(k)
                    if not labs or any((self.category_of(f) in labs for f in fits)):
                        continue
                    opts = [o for o in self.walk_options(k) if self.stated_carry(o['e']['args'], k)]
                    opts.sort(key=lambda o: (-self.stated_carry(o['e']['args'], k), -o['carry'], o['pages'], compact(o['e']['args'])))
                    for o in opts:
                        if o['key'] not in [s[1] for s in searches]:
                            searches.append((k, o['key']))
                            break
                searches = searches[:2]
            if not searches:
                return ''
            self.decline_once.add('dreads')
            self.decline_reads = {'searches': searches, 'room': room}
            return 'these value-carrying queries had more matches than were opened, and since this store orders hits by id, ignoring fit and price, fits are just as likely on unseen pages; the code now fetches up to %d of them, then opens the lowest-priced available affordable rows found. Weigh those records against the binding requirements, buy whatever fits (a line per requested product at its own quantity, or the single item), and decline only if all of them fail.' % room
        except Exception as exc:
            self.wire.note('decline_read_error', error=str(exc)[:200])
            return ''

    def stated_carry(self, args, k):
        binding = self.needs.binding()
        n = 0
        for i in self.carry(args, k):
            if i < len(binding) and (not re.search('redirect|refer|sold by|same as|as shown|the page', binding[i].lower())):
                n += 1
        return n

    def run_decline_reads(self):
        dr = self.decline_reads
        self.decline_reads = None
        pg = self.paging_param()
        mt = self.multi_tool()
        cost = 1 if mt else 2
        fetched = 0
        seen = set()
        for k, key in dr['searches']:
            e = self.search_log.get(key)
            if not e or not pg:
                continue
            while fetched < dr['room'] and self.next_page(e) is not None and (self.steps_left() > 6 + cost) and (not self.wire.finished) and (self.wire.time_left() > CLOSING + 30):
                a = dict(e['args'])
                a[pg[0]] = self.next_page(e)
                if not self.scripted_search(a, 'I fetch the next page before declining, since unseen pages of an id-ordered store can hold fits.'):
                    break
                fetched += 1
            seen |= set(e['seen'])
        count = mt[2] if mt else 2
        cap = self.needs.budget()
        pools = []
        for k, _ in dr['searches']:
            labs = self.walk_labels(k)
            units = self.units_for(k)
            rows = []
            for r in seen:
                p = self.records.price(r)
                if r in self.records.read or r in self.refused_lines or p is None or (self.newest_stock(r) is False):
                    continue
                if self.category_of(r) not in labs or (cap is not None and p * units > cap):
                    continue
                rows.append((p, r))
            pools.append([r for _, r in sorted(rows)])
        queue = []
        while len(queue) < count and any(pools):
            for p in pools:
                if p and len(queue) < count:
                    x = p.pop(0)
                    if x not in queue:
                        queue.append(x)
        while queue and (1 if mt and len(queue) >= 2 else len(queue)) >= self.steps_left() - 6:
            queue.pop()
        self.wire.note('decline_reads', pages=fetched, opened=queue)
        if queue:
            self.enqueue(queue, 'decline', 'I open the cheapest unopened affordable rows from the fetched pages.', floor=6)
            self.run_queue()
            return True
        if self.chat and self.chat[-1].get('role') == 'tool':
            self.chat[-1]['content'] += '\nno unopened available affordable row turned up.\n' + self.status()
        return fetched > 0

    def g_values(self, k):
        out = []
        ps = self.needs.products()
        for r in self.needs.binding():
            if ':' not in r:
                continue
            if ps and k is not None and ptag(r) and (self.product_index(r) not in (k, None)):
                continue
            prop, val = (pname(r), pvalue(r))
            short = len(re.findall('[a-z0-9]+', val.lower())) <= 5 and (not EXCLUSION_RX.search(val.lower()))
            if not (positive_value(val, prop) or short):
                continue
            v = re.sub('\\([^)]*\\)', ' ', val.lower()).strip()
            if not v or is_reference(v):
                continue
            if re.search('\\bor\\b|\\beither\\b|/', v):
                pieces = [p for p in re.split('[,;/]|\\bor\\b|\\beither\\b', v) if p.strip()][:3]
            else:
                pieces = [re.split('[,;&+]|\\band\\b', v)[0]]
            pw = set(re.findall('[a-z]+', prop.lower()))
            alts = []
            for p in pieces:
                if EXCLUSION_RX.search(p):
                    continue
                amt = re.fullmatch('\\s*(\\d+)\\s*([a-z]+)\\s*', p)
                if re.search('\\d', p) and (not (amt and (not LIMIT_RX.search(p)))):
                    continue
                p = VALUE_LINKS_RX.sub(' ', p).replace("'s", '')
                toks = []
                for t in re.findall('[a-z0-9]+(?:\\.[0-9]+)?', p):
                    if '.' in t or t in LINK or (len(t) < 2 and (not t.isdigit())):
                        continue
                    if t in pw and (not (amt and t == amt.group(2))):
                        continue
                    toks.append(t)
                if toks and toks not in alts:
                    alts.append(toks)
            if alts and alts not in out:
                out.append(alts)
        return out

    def row_spelling(self, labels):
        tally = {}
        for ob in self.records.obs:
            for d in dicts_in(ob):
                if not any(('id' in kk.lower() and v in self.records.known_set for kk, v in d.items() if isinstance(v, str))):
                    continue
                for kk, v in d.items():
                    if 'id' in kk.lower() or not isinstance(v, str):
                        continue
                    if any((f in kk.lower() for f in ('categor', 'shelf', 'aisle', 'department'))) and 1 <= len(v.strip()) <= 60 and (v.strip().lower() in labels):
                        tally[v.strip()] = tally.get(v.strip(), 0) + 1
        if not tally:
            return ''
        return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def composed_query(self):
        msgs_left = self.allowance_in_force() - self.sent_msgs if self.code_asks and self.role('message') else 0
        if self.steps_left() - self.reserve() - 2 * max(0, msgs_left) <= 4:
            return False
        ps = self.needs.products()
        prods = list(range(len(ps))) if ps else [None]
        for k in prods:
            if self.g_tries.get(k, 0) >= 4:
                continue
            vals = self.g_values(k)
            sig = (k, compact(vals))
            if sig in self.g_settled or not vals:
                continue
            labels = self.walk_labels(k)
            spelling = self.row_spelling(labels)
            if not spelling:
                continue
            qs = self.queries(k)
            tmpl = None
            s = self.role('search')
            props = (self.shapes.get(s) or {}).get('properties') or {}
            req = (self.shapes.get(s) or {}).get('required') or []
            for a in qs:
                texts = [kk for kk, v in a.items() if isinstance(v, str) and v.strip() and (not (isinstance(props.get(kk), dict) and 'enum' in props[kk]))]
                if len(texts) == 1:
                    tmpl = ({kk: v for kk, v in a.items() if kk != texts[0] and (kk in req or (isinstance(props.get(kk), dict) and 'enum' in props[kk]))}, texts[0])
                    break
            if not tmpl:
                self.g_settled.add(sig)
                continue
            maxlen = (props.get(tmpl[1]) or {}).get('maxLength')
            label_words = [w for w in dict.fromkeys(re.findall('[a-z0-9]+', spelling.lower().replace("'s", ''))) if len(w) >= 2 and w not in LINK]
            if self.labels_indexed is False:
                titles = [record_title(d).lower() for lid, d in self.ref_index().items() if record_label(d) in labels and record_title(d)]
                lw2 = [w for w in label_words if titles and sum((1 for t in titles if stem_word(w) in t)) * 2 >= len(titles)]
                if not lw2:
                    nm = self.product_names()[k] if ps else self.needs.product_name()
                    cw = [w for w in re.findall('[a-z]+', nm.lower()) if len(w) >= 3 and w not in LINK]
                    lw2 = cw[-1:]
                label_words = lw2

            def form(alt_lists):
                words, stems = ([], set())
                for ws in alt_lists + [label_words]:
                    for w in ws:
                        if stem_word(w) not in stems:
                            stems.add(stem_word(w))
                            words.append(w)
                if self.read_width:
                    words = words[:self.read_width]
                text = ' '.join(words)
                if is_num(maxlen):
                    while len(text) > maxlen and ' ' in text:
                        text = text.rsplit(' ', 1)[0]
                    text = text[:int(maxlen)]
                return text
            cheapest = self.needs.priority() == 'cheapest'
            if len(vals) == 1:
                l1 = [form([a]) for a in (vals[0] if cheapest else vals[0][:1])]
                l2 = []
            else:
                multi_i = next((i for i, v in enumerate(vals) if len(v) > 1), None)
                l1 = []
                variants = vals[multi_i] if multi_i is not None and cheapest else [None]
                for var in variants:
                    chosen = [var if i == multi_i and var is not None else v[0] for i, v in enumerate(vals)]
                    l1.append(form(chosen))
                l2 = [form([a]) for a in (vals[0] if cheapest else vals[0][:1])]
            self.g_sets[k] = [{stem_word(w) for w in f.split()} for f in l1 + l2]
            nxt, from_one = self.next_form(l1, l2, tmpl[0])
            if nxt == 'all empty':
                if self.labels_indexed is None and self.value_led_rows(k):
                    self.labels_indexed = False
                    continue
                nxt = None
            if not nxt:
                self.g_settled.add(sig)
                continue
            args = dict(tmpl[0])
            args[tmpl[1]] = nxt
            a2, faults = check_args(args, self.shapes.get(s) or {})
            if faults:
                self.g_settled.add(sig)
                continue
            self.g_tries[k] = self.g_tries.get(k, 0) + 1
            ok = self.scripted_search(a2, "Values plus the category name as rows spell it should list that category's listings bearing the values.")
            if not ok:
                return True
            pg = self.paging_param()
            key = compact({kk: v for kk, v in sorted(a2.items()) if not pg or kk != pg[0]})
            e = self.search_log.get(key)
            rows = [r for d in dicts_in(self.records.obs[-1], 3) for r in [own_id(d)] if r in self.records.known_set and len(d) >= 4]
            total = self.totals.get(key, len(rows))
            shows_all = bool(rows) and len(rows) >= total and any((self.category_of(r) in labels for r in rows))
            queue = []
            if shows_all and cheapest and pg:
                self.value_queries.setdefault(k, set()).add(key)
                if e is not None:
                    e['value_done'] = True
                if not ps:
                    price, _ = self.pick_of(k)
                    queue = self.bargain(rows, k, price)[:max(0, min(3, 3 - self.k_reads.get(k, 0)))]
                    self.k_reads[k] = self.k_reads.get(k, 0) + len(queue)
            elif shows_all and (not cheapest) and from_one:
                queue = self.bargain([r for r in rows if self.category_of(r) in labels], k, None)[:3]
            floor = self.reserve() + 2
            mt = self.multi_tool()
            while queue and (1 if mt and len(queue) >= 2 else len(queue)) >= self.steps_left() - floor:
                queue.pop()
            note = '[value query] %s: %s' % (nxt, 'zero matches' if not total else '%d matches%s' % (total, ', all on this page' if shows_all else ''))
            if queue:
                note += '; opening %s' % ', '.join(('%s (%s)' % (q, money(self.records.price(q))) for q in queue))
            if self.chat and self.chat[-1].get('role') == 'tool':
                self.chat[-1]['content'] += '\n' + note
            self.wire.note('composed_query', query=nxt, total=total, queued=queue)
            if queue:
                self.ranked = {x for x in self.ranked if self.category_of(x[0]) not in labels}
                self.vetted_gen = -1
                self.enqueue(queue, 'walk', 'I open the listings this value query surfaced.', recheck=True, floor=floor)
                self.run_queue()
            return True
        return False

    def next_form(self, l1, l2, mode_args):
        w = self.read_width or None
        sent = {}
        for key, e in self.search_log.items():
            st = frozenset((stem_word(x) for x in re.findall('[a-z0-9]+', self.free_text(e['args']).lower())[:w]))
            sent[st] = max(sent.get(st, 0), e['total'])
        for rec in self.log:
            if rec.get('kind') == 'env' and rec.get('name') == self.role('search'):
                st = frozenset((stem_word(x) for x in re.findall('[a-z0-9]+', self.free_text(rec.get('args') or {}).lower())[:w]))
                sent.setdefault(st, 0)
        for lst, first in ((l1, True), (l2, False)):
            any_rows = False
            for f in lst:
                st = frozenset((stem_word(x) for x in f.split()[:w]))
                if st not in sent:
                    return (f, first)
                if sent[st]:
                    any_rows = True
            if lst and any_rows:
                return (None, first)
        return ('all empty', False) if l1 or l2 else (None, False)

    def value_led_rows(self, k):
        labels = self.walk_labels(k)
        for e in self.search_log.values():
            if self.carry(e['args'], k) and any((l in labels for l in e['row_labels'].values())):
                return True
        return False

    def bargain(self, ids, k, ceiling):
        cap = self.needs.budget()
        units = self.units_for(k)
        out = []
        for r in dict.fromkeys(ids):
            p = self.records.price(r)
            if r in self.records.read or r in self.refused_lines or p is None or (self.newest_stock(r) is False):
                continue
            if cap is not None and p * units > cap:
                continue
            if ceiling is not None and p >= ceiling:
                continue
            out.append((p, r))
        return [r for _, r in sorted(out)]

    def value_pages(self):
        try:
            if any((t['status'] == 'open' for t in self.trackers)) and self.resolver_room():
                return False
            if self.orders_tried or self.refusals or self.wire.finished or self.wire.blocked or (self.wire.time_left() < CLOSING + 60) or (not self.role('search')):
                return False
            if self.composed_query():
                return True
            pg = self.paging_param()
            if self.needs.priority() != 'cheapest' or self.word_rule == 'loose' or (not pg):
                return False
            ps = self.needs.products()
            prods = list(range(len(ps))) if ps else [None]
            cap_k, cap_all = (4, max(8, 5 * len(prods)))
            floor = self.reserve() + 2
            reserved = 1 if self.multi_tool() else 2 if ps else 3
            best = None
            for k in prods:
                labels = self.walk_labels(k)
                if not labels:
                    continue
                for key, e in self.search_log.items():
                    if e.get('value_done') or e['sorted'] is not True or (not e['first']) or (self.next_page(e) is None) or (not self.carry(e['args'], k)) or (not any((l in labels for l in e['row_labels'].values()))):
                        continue
                    need = self.pages_left(e)
                    allowed = min(cap_k - self.k_pages.get(k, 0), cap_all - sum(self.k_pages.values()))
                    composed = {stem_word(w) for w in self.lead_tokens(e['args'])} in self.g_sets.get(k, [])
                    take = need if need <= allowed else allowed if composed else 0
                    if take <= 0 or self.steps_left() - take - reserved <= floor:
                        continue
                    msgs_left = self.allowance_in_force() - self.sent_msgs if self.code_asks and self.role('message') else 0
                    slack = self.steps_left() - self.reserve() - 2 * max(0, msgs_left)
                    if (take > 2 or ps) and msgs_left > 0 and (slack > take + reserved + 4):
                        continue
                    cand = (self.k_pages.get(k, 0), take, -len(self.carry(e['args'], k)), -1 if k is None else k, key)
                    if best is None or cand < best[0]:
                        best = (cand, k, key, take)
            if not best:
                return False
            _, k, key, take = best
            e = self.search_log[key]
            e['value_done'] = True
            got = 0
            while got < take and self.next_page(e) is not None and (self.steps_left() > floor + 1) and (not self.wire.finished) and (self.wire.time_left() >= CLOSING + 30):
                a = dict(e['args'])
                a[pg[0]] = self.next_page(e)
                self.k_pages[k] = self.k_pages.get(k, 0) + 1
                before = len(e['visited'])
                if not self.scripted_search(a, 'The cheapest is wanted, every word is required and rows come in fixed id order, so further pages reveal all listings bearing these words.'):
                    break
                got += 1
                if len(e['visited']) == before:
                    break
            if not got:
                return False
            self.value_queries.setdefault(k, set()).add(key)
            complete = self.next_page(e) is None and len(e['seen']) >= e['total']
            price, _ = self.pick_of(k)
            labels = self.walk_labels(k)
            queue = self.bargain([r for r in e['seen'] if self.category_of(r) in labels], k, price)
            queue = queue[:max(0, min(3, 3 - self.k_reads.get(k, 0)))]
            self.k_reads[k] = self.k_reads.get(k, 0) + len(queue)
            mt = self.multi_tool()
            while queue and (1 if mt and len(queue) >= 2 else len(queue)) >= self.steps_left() - floor:
                queue.pop()
            note = '[value pages] %s: %d of %d rows seen; %s' % (self.free_text(e['args']), len(e['seen']), e['total'], 'all listings bearing these words are among them' if complete else 'the remainder is unreachable, so something cheaper that fits could lie beyond')
            note += '; opening %s' % ', '.join(('%s (%s)' % (q, money(self.records.price(q))) for q in queue)) if queue else '; none unread, available and affordable remains there'
            note += '. Per product choose the cheapest listing whose fields themselves satisfy all binding requirements; a pricier fit is acceptable only once each cheaper candidate was read and rejected.'
            if self.chat and self.chat[-1].get('role') == 'tool':
                self.chat[-1]['content'] += '\n' + note + ('' if queue else '\n' + self.status())
            self.wire.note('value_pages', query=self.free_text(e['args']), pages=got, queued=queue)
            if queue:
                self.ranked = {x for x in self.ranked if self.category_of(x[0]) not in labels}
                self.vetted_gen = -1
                self.enqueue(queue, 'walk', 'I open the cheapest unread rows these pages revealed.', recheck=True, floor=floor)
                self.run_queue()
            return True
        except Exception as exc:
            self.wire.note('value_pages_error', error=(type(exc).__name__ + ': ' + str(exc))[:200])
            return False

def name_matcher(name):
    kinds = kind_list(name)
    if not kinds:
        return None
    parts = []
    for k in kinds:
        forms = sorted(plural_forms(k) | {k}, key=len, reverse=True)
        parts.append('(?:' + '|'.join((re.escape(f) for f in forms)) + ")(?:'s)?")
    return '\\b' + "[\\s'\\-]+".join(parts) + '\\b'

class JoinedPart:

    def joined_labels_from(self, obs):
        for d in dicts_in(obs):
            if not any(('id' in k.lower() and id_like(v) for k, v in d.items())):
                continue
            lab = record_label(d)
            if not lab:
                continue
            for j in JOINERS:
                pieces = lab.split(j)
                if len(pieces) == 2 and all((p.strip() for p in pieces)):
                    self.joined.setdefault(lab, (lab, [p.strip() for p in pieces], j))
                    break

    def asked_about_product(self):
        names = [p['product'] for p in self.needs.products()]
        for e in self.qlog:
            for tag in re.findall('\\(([^)]*)\\)', e['label']):
                if any((kind_in(w, set(kind_list(nm))) for w in kind_list(tag) for nm in names)):
                    return True
            q = e['text'].lower()
            for nm in names:
                m = name_matcher(nm)
                if m and re.search(m, q):
                    return True
        return False

    def fusion_gate(self):
        return bool(self.needs.products()) and bool(self.joined) and (not self.cart.used) and (not self.cart.units) and (not self.needs.revealed) and (not self.needs.dontmind) and (not self.asked_about_product())

    def collide(self, a, b):

        def red(reqs):
            out = []
            for r in reqs:
                if ':' in r:
                    out.append((pname(r), ' '.join(re.findall('[a-z0-9]+', pvalue(r).lower()))))
            return out
        for pa, va in red(a['requirements']):
            for pb, vb in red(b['requirements']):
                if (strict_same(pa, pb) or loose_same(pa, pb)) and va != vb:
                    return True
        return False

    def wording(self, n1, n2, label, pieces, joiner):
        req = re.sub('\\s+', ' ', self.request.lower().replace('’', "'"))
        joins = [re.escape(j.strip()) for j in JOINERS if j.strip()] + ['and', 'or']
        m1, m2 = (name_matcher(pieces[0]), name_matcher(pieces[1]))
        if m1 and m2:
            hit = re.search(m1 + '\\s*(?:' + '|'.join(joins) + ')\\s*' + m2, req)
            if hit:
                return (hit.group(0), True)
        k1, k2 = (kind_list(n1), kind_list(n2))
        if not k1 or not k2 or same_kind(k1[-1], k2[-1]):
            return (None, False)
        for a, b in ((n1, n2), (n2, n1)):
            ma, mb = (name_matcher(a), name_matcher(b))
            if not ma or not mb:
                continue
            for m in re.finditer(ma, req):
                tail = req[m.end():m.end() + 40 + 60]
                mm = re.match('([^.;:!?]{0,40}?)(' + mb + ')', tail)
                if not mm:
                    continue
                between = mm.group(1)
                if len(re.findall('[a-z0-9]+', between)) > 3:
                    continue
                if not re.search('\\b(?:and|or)\\b|&|/|,|\\+', between):
                    continue
                if re.search('\\d|\\bx\\b', between):
                    continue
                phrase = req[m.start():m.end() + mm.end()]
                rest = req[:m.start()] + ' ' + req[m.end() + mm.end():]
                if re.search(ma, rest) and re.search(mb, rest):
                    continue
                return (phrase, False)
        return (None, False)

    def try_fusion(self):
        notes = []
        try:
            for _ in range(2):
                if not self.fusion_gate():
                    break
                ps = self.needs.products()
                done = False
                for key in sorted(self.joined):
                    lab, pieces, joiner = self.joined[key]
                    halves = [kind_list(p) for p in pieces]
                    if not all(halves):
                        continue
                    lab_kinds = set(kind_list(lab))
                    for i in range(len(ps)):
                        for j in range(i + 1, len(ps)):
                            ki, kj = (kind_list(ps[i]['product']), kind_list(ps[j]['product']))
                            if not ki or not kj:
                                continue
                            if all_kinds(ki, set(kj)) and all_kinds(kj, set(ki)):
                                continue
                            if not all_kinds(ki + kj, lab_kinds):
                                continue
                            if ps[i]['quantity'] != ps[j]['quantity']:
                                continue
                            pi = [w for w in ki if not kind_in(w, set(kj))]
                            pj = [w for w in kj if not kind_in(w, set(ki))]
                            ok = False
                            for P, Q in ((halves[0], halves[1]), (halves[1], halves[0])):
                                if all((kind_in(w, set(P)) and (not kind_in(w, set(Q))) for w in pi)) and all((kind_in(w, set(Q)) and (not kind_in(w, set(P))) for w in pj)) and any((kind_in(w, set(P)) for w in ki)) and any((kind_in(w, set(Q)) for w in kj)):
                                    ok = True
                                    break
                            if not ok or self.collide(ps[i], ps[j]):
                                continue
                            phrase, flag = self.wording(ps[i]['product'], ps[j]['product'], lab, pieces, joiner)
                            if not phrase:
                                continue
                            note = self.fuse(i, j, phrase, flag, lab)
                            if note:
                                notes.append(note)
                                done = True
                            break
                        if done:
                            break
                    if done:
                        break
                if not done:
                    break
        except Exception as exc:
            self.wire.note('fusion_error', error=str(exc)[:200])
        return ' '.join(notes)

    def fuse(self, i, j, phrase, flag, label):
        r = self.needs.reading
        items = [it for it in r.get('items') or [] if isinstance(it, dict) and str(it.get('product') or '').strip()]
        ps = self.needs.products()
        if len(items) != len(ps) or not 0 <= i < j < len(items):
            return ''
        remaining = len(ps) - 1
        new_tag = phrase if remaining >= 2 else None
        names = [ps[i]['product'], ps[j]['product']]
        lab_kinds = set(kind_list(label))
        others = [p['product'] for x, p in enumerate(ps) if x not in (i, j)]

        def belongs(tag):
            kw = kind_list(tag)
            return bool(kw) and all_kinds(kw, lab_kinds) and any((kind_in(w, set(kind_list(n))) for w in kw for n in names)) and (not any((all_kinds(kw, set(kind_list(o))) and all_kinds(kind_list(o), set(kw)) for o in others)))

        def retag(text):
            if not isinstance(text, str) or ':' not in text:
                head, rest = (text, '') if isinstance(text, str) else ('', '')
                sep = ''
            else:
                head, rest = text.split(':', 1)
                sep = ':'

            def rep(m):
                if belongs(m.group(1)):
                    return '(%s)' % new_tag if new_tag else ''
                return m.group(0)
            head = re.sub('\\s*\\(([^)]*)\\)', lambda m: ' ' + rep(m) if rep(m) else '', head)
            return re.sub('\\s+', ' ', head).strip() + sep + rest

        def dedupe(lst):
            out = []
            for x in lst:
                if x.lower() not in [y.lower() for y in out]:
                    out.append(x)
            return out
        reqs = []
        for it in (items[i], items[j]):
            for q in it.get('requirements') or []:
                if isinstance(q, str) and ':' in q:
                    h, v = q.split(':', 1)
                    reqs.append(re.sub('\\s*\\([^)]*\\)', '', h).strip() + ':' + v)
        fused = {'product': phrase, 'quantity': items[i].get('quantity') or 1, 'requirements': dedupe(reqs)}
        new_items = [fused if x == i else it for x, it in enumerate(items) if x != j]
        n = self.needs
        n.stated = dedupe([retag(x) for x in n.stated])
        n.revealed = dedupe([retag(x) for x in n.revealed])
        n.dropped = dedupe([retag(x) for x in n.dropped])
        n.dontmind = dedupe([retag(x) for x in n.dontmind])
        if n.current is not None:
            n.current = dedupe([retag(x) for x in n.current])
        n.resolved = [retag(x) for x in n.resolved]
        for key in ('requirements', 'hints', 'references', 'soft_wants'):
            if isinstance(r.get(key), list):
                r[key] = dedupe([retag(x) if isinstance(x, str) else x for x in r[key]])
        for tgt in r.get('reference_targets') or []:
            if isinstance(tgt, dict) and belongs(str(tgt.get('product') or '')):
                tgt['product'] = phrase
        for t in self.trackers:
            t['req'] = retag(t['req'])
            t['head'] = retag(t['head'] + ':').rstrip(':')
            if t['tag'] and belongs(t['tag']):
                t['tag'] = new_tag or ''
        self.retry_queue = [retag(x + ':').rstrip(':') for x in self.retry_queue]
        self.open_pairs = [retag(x + ':').rstrip(':') for x in self.open_pairs]
        if remaining >= 2:
            r['items'] = new_items
            r['product'] = ' and '.join((it['product'] for it in new_items))
        else:
            r['items'] = [fused]
            r['product'] = phrase
            r['quantity'] = fused['quantity']
        n.version += 1
        self._home_n = -1
        self.walked = set()
        self.k_reads, self.k_pages = ({}, {})
        try:
            self.wire.scale = 1.0 + 0.3 * min(2, max(0, len(self.needs.products()) - 1))
        except Exception:
            self.wire.scale = 1.0
        self.wire.note('fusion', label=label, names=names, phrase=phrase, label_wording=flag, remaining=remaining)
        return "This store treats '%s' as a single category: the shopper wants one item from it (%s), hence %sa single cart line for it, not one line per kind the label combines. Requirements now: %s" % (label, phrase, 'ONLY ' if remaining <= 1 else '', n.req_text())
HOLD = 'Not executed and no step used: '
HOLD_LATER = 'Not run yet (no step used): '
CAP_FLOW = 1200
DEFAULT_FLOW = ('queue', 'pages', 'asks', 'refs', 'walk')
ALGO_RANGE = {'reads': (2, 24), 'hops': (0, 20), 'refs_share': (0.1, 0.8)}
STAGE_FNS = {'queue': 'run_queue', 'pages': 'value_pages', 'asks': 'asks_stage', 'refs': 'ref_turn', 'walk': 'walk_step'}
ACT_FNS = {'message': 'act_message', 'read': 'act_read', 'recheck': 'act_recheck', 'total': 'act_total', 'use': 'act_use', 'none': 'act_none'}
LOOP_ANCHORS = ('first', 'after_needs', 'before_cart', 'after_cart')
ALL_ANCHORS = LOOP_ANCHORS + ('before_order', 'before_finish')
DUTY_FIRST = re.compile('\\b(?:before|beforehand|prior|first|then|ahead)\\b', re.I)
DUTY_ORDER = re.compile('\\b(?:order|orders|ordering|ordered|checkout|buying|purchase)\\b', re.I)
DUTY_BASKET = re.compile('\\b(?:basket|cart)\\b', re.I)
DUTY_FLOOR = 8.0
DUTY_LINES = 4
PROMPT_FLOW = ("You plan how a shopping assistant will work through one task before it starts. Read the request and "
               "decide the order of work, which routines to use, and anything the request demands that a standard "
               "shopping run would miss.\n"
               "Reply with one JSON object and nothing else:\n"
               '{"stages": [names in the order you want them], "asks": <whole number or null>, '
               '"todos": [{"what": "...", "act": "message|read|none", "content": "...", "when": "first|before_order|before_finish"}]}\n'
               "stages: choose from queue, pages, asks, refs, walk. Name the ones this task needs first, in the order "
               "they should run; any you leave out still runs, but only after those you name. queue runs reads the code already lined up. pages "
               "reads the listings found so far. asks lets the code plan and send questions to the shopper. refs "
               "follows a named product or seller the request points at. walk follows a chain of references between "
               "listings. Leave the list empty to keep the standard order.\n"
               "asks: the number of messages to the shopper the task can afford. Lower it when the request obliges a "
               "message of its own so there is room for it; null keeps the allowance as it is.\n"
               "todos: anything the request requires beyond finding and ordering a fitting product. act 'message' "
               "sends content to the shopper, 'total' works out what the basket costs and sends that, 'read' reads "
               "what content names, 'recheck' re-reads what is in the basket, 'use' calls one of this world's own "
               "tools that the code has no routine for, written as the tool name then a JSON object of its arguments, "
               "'none' is a reminder for later. Use "
               "'total' whenever the request wants the sum, the price or the total said before the order, so the "
               "figure is worked out rather than guessed. when "
               "says the earliest point the step may run: 'first' straight away, 'after_needs' once the shopper's hidden "
               "needs are settled, 'before_cart' while nothing is in the cart yet, 'after_cart' once something is in "
               "it, 'before_order' just before the order is placed, 'before_finish' at the very end. Use an empty "
               "list when the request asks for nothing of the kind.\n"
               "algorithm: how hard to work each part, as whole numbers, any you leave out keeps its default. "
               "reads: how many listings the fallback picker opens. hops: how many times a reference may point on "
               "to another listing. refs_share: the share of the step budget the reference chase may use, between "
               "0.1 and 0.8. Raise reads and hops for a task that hangs on one referenced value, and leave the whole "
               "field out when the request is plain.")
PROMPT_FLOW_AGAIN = ("You planned how a shopping assistant would work through this task. It has since learned things it "
                     "did not know when you planned. Revise the plan for the rest of the run.\n"
                     "Reply with one JSON object and nothing else, in the same shape as before: "
                     '{"stages": [...], "asks": <whole number or null>, "algorithm": {...}, '
                     '"todos": [{"what": "...", "act": "...", "content": "...", "when": "..."}]}\n'
                     "Only the obligations you name now are added; the ones already carried out stay done. Keep what "
                     "still fits and change what the new information has made wrong.")
PROMPT_FLOW_REVIEW = ("A shopping assistant is about to finish a task. You are given what the request obliged it to do and "
                 "what it actually did. Say which obligations are still unmet.\n"
                 "Reply with one JSON object and nothing else: {\"missing\": [the what text of each unmet obligation]}. "
                 "Reply with an empty list when every obligation was met.")

class LoopPart:

    def run(self):
        if self.wire.finished:
            return self.log
        self.setup()
        opening = 'REQUEST\n' + self.request
        if self.opening is not None:
            opening += '\n\nOPENING OBSERVATION\n' + compact(self.opening)
            self.records.intake(lens(self.opening))
        ov = self.needs.overview()
        if ov:
            opening += '\n\n' + ov
        opening += '\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': PROMPT_ACTION}, {'role': 'user', 'content': opening}]
        self.plan_flow()
        self.last_good = time.monotonic()
        self.missed = 0
        for _ in range(LOOP_LIMIT):
            if self.wire.finished or self.wire.blocked or self.steps_left() <= 0:
                break
            if self.wire.time_left() < LAST_CALL:
                break
            if self.needs.version != self.needs_seen:
                self.needs_seen = self.needs.version
                self.gen += 1
            self.catch_up()
            if self.answer_shopper():
                continue
            if self.plan_mark is not None:
                self.replan()
            if self.todos and self.todo_run():
                continue
            staged = False
            for _nm in self.flow or DEFAULT_FLOW:
                _fn = self.stage_fn(_nm)
                if _fn is not None and _fn():
                    staged = True
                    break
            if staged:
                continue
            models = ROLE_ACTOR_ROTATED if self.stuck >= STUCK_LIMIT else ROLE_ACTOR
            msg = self.wire.model(self.chat, models, CAP_ACTOR, tools=self.tools)
            if msg is None:
                self.missed += 1
                if not self.wire.usable(ROLE_ACTOR, tools=True):
                    break
                if time.monotonic() - self.last_good > DOWN_AFTER and (self.missed >= 2 or (self.wire.down() and len(set(self.wire.tried)) >= 2)):
                    break
                if self.wire.fail_kind in ('empty', 'refused'):
                    self.hard_fails += 1
                    if self.hard_fails >= 2:
                        break
                continue
            self.last_good = time.monotonic()
            self.missed = 0
            props = self.normalise(msg)
            text = msg.get('content')
            if not isinstance(text, str):
                text = compact(text) if text else ''
            if self.code_asks and any((p['name'] in (self.role('cart'), self.role('order'), self.role('decline')) for p in props if p['name'])):
                if self.question_step(urged=True):
                    continue
            if not props:
                self.idle += 1
                self.chat.append({'role': 'assistant', 'content': text or '(no tool call)'})
                self.chat.append({'role': 'user', 'content': 'Every reply must call one of the tools.\n' + self.status()})
                if self.idle >= IDLE_LIMIT:
                    break
                continue
            self.idle = 0
            self.wire.note('actor', text=text[:2000], calls=[[p['name'], p['args']] for p in props])
            self.chat.append({'role': 'assistant', 'content': text or '', 'tool_calls': [{'id': p['id'], 'type': 'function', 'function': {'name': p['name'] or '', 'arguments': p['args']}} for p in props]})
            before = self.wire.step
            res = self.dispatch(props, scripted=False)
            self.post(res)
            self.stuck = self.stuck + 1 if self.wire.step == before else 0
        if self.todos and (not self.wire.finished) and (not self.wire.blocked):
            try:
                self.review_todos()
            except Exception as exc:
                self.wire.note('review_failed', error=(type(exc).__name__ + ': ' + str(exc))[:200])
        if not self.wire.finished and (not self.wire.blocked):
            self.forced_finish()
        self.wire.note('closing', finished=self.wire.finished, steps=self.wire.step, spend=round(self.wire.spend, 5), seconds=round(time.monotonic() - self.wire.t0, 1))
        return self.log

    def normalise(self, msg):
        out = []
        for c in msg.get('tool_calls') or []:
            if not isinstance(c, dict):
                continue
            self.seq += 1
            f = c.get('function') if isinstance(c.get('function'), dict) else {}
            args = f.get('arguments')
            if not isinstance(args, str):
                args = json.dumps(args if args is not None else {})
            out.append({'id': c.get('id') or 'call_%d' % self.seq, 'name': f.get('name'), 'args': args})
        return out

    def dispatch(self, props, scripted):
        self.reply_admits = 0
        self.reply_admitted = set()
        usable = self.offer()
        cart_tool = self.role('cart')
        puts = []
        for p in props:
            if p['name'] == cart_tool:
                a = read_obj(p['args']) if isinstance(p['args'], str) else p['args']
                if isinstance(a, dict) and self.cart.is_put(a):
                    aim = self.cart.aimed(a)
                    if aim:
                        puts.append(aim)
        results = {}
        send = []
        removal_held = False
        msgs_ok = 0
        for p in props:
            if removal_held and p['name'] == cart_tool:
                results[p['id']] = HOLD + 'a prior change to the cart within this same reply was stopped; learn the reason, then send those cart changes again.'
                continue
            verdict, args = self.screen(p, scripted, usable, len(props), msgs_ok, puts)
            if verdict is not None:
                results[p['id']] = verdict
                if p['name'] == cart_tool and isinstance(args, dict) and args and (not self.cart.is_put(args)):
                    removal_held = True
                continue
            if p['name'] == self.role('message'):
                msgs_ok += 1
            send.append((p, args))
        if send and send[0][0]['name'] in self.finishers() and self.classify(send[0][0]['name'], send[0][1]):
            p, args = send[0]
            held, args = self.finish_screen(p['name'], args)
            if held is not None:
                results[p['id']] = held
                return [(q['id'], results.get(q['id'], HOLD + 'not sent.')) for q in props]
            send[0] = (p, args)
        elif send and self.finishers():
            room = max(0, self.steps_left() - 1)
            for p, a in send[room:]:
                results[p['id']] = HOLD + 'the single remaining step is reserved for ordering or declining.'
            send = send[:room]
        for i in range(0, len(send), CHUNK):
            chunk = send[i:i + CHUNK]
            for p, a in chunk:
                if self.wire.finished or self.wire.blocked or self.steps_left() <= 0:
                    results[p['id']] = 'Not run: the episode has ended or no steps remain.'
                    continue
                res = self.send_one(p['name'], a)
                if res is None:
                    results[p['id']] = 'The request failed on the network or server side; it may be retried.' if not self.wire.finished else 'Not run: the episode has ended.'
                else:
                    results[p['id']] = self._last_text
        return [(p['id'], results.get(p['id'], HOLD + 'not sent.')) for p in props]

    def screen(self, p, scripted, usable, batch_n, msgs_ok, puts):
        name = p['name']
        if name not in self.shapes:
            return (HOLD + 'unknown tool. Usable tools: %s.' % ', '.join(sorted(usable)), None)
        if not scripted and name not in usable:
            if name == self.role('message') and self.code_asks:
                a = read_obj(p['args']) if isinstance(p['args'], str) else p['args']
                txt = ' '.join((str(v) for v in (a or {}).values() if isinstance(v, str) and v.strip()))[:300] if isinstance(a, dict) else ''
                if txt and txt.lower() not in [w.lower() for w in self.wanted_questions]:
                    self.wanted_questions = (self.wanted_questions + [txt])[-3:]
                return (HOLD + 'the code plans and sends the shopper questions itself once a few listings have been opened; carry on exploring.', None)
            return (HOLD + 'the remaining steps do not allow this tool now. Usable tools: %s.' % ', '.join(sorted(usable)), None)
        raw = p['args']
        if isinstance(raw, dict):
            args = raw
        elif not (raw or '').strip():
            args = {}
        else:
            args = read_obj(raw)
            if args is None:
                return (HOLD + 'the arguments are not a valid JSON object.', None)
        args, faults = check_args(args, self.shapes.get(name) or {})
        if faults:
            return (HOLD + 'argument problems: %s.' % '; '.join(faults[:6]), None)
        if name in self.finishers() and batch_n >= 2 and self.classify(name, args):
            return (HOLD + 'an order or decline must be sent by itself; send it alone after reading what the other calls returned.', None)
        if name == self.role('message') and self.allowance is not None and (self.sent_msgs + msgs_ok >= self.allowance):
            return (HOLD + 'the message allowance is used up; stop messaging.', None)
        if name == self.role('message') and (not scripted) and (not self.cart.units) and self.duty_waiting() and DUTY_BASKET.search(' '.join((str(v) for v in (args or {}).values() if isinstance(v, str)))):
            return (HOLD + 'nothing is in the cart yet, so there is no basket to put to the shopper and this message would spend an allowance the request needs for the real one; the code shows the basket itself in the step before the order.', None)
        if not scripted and name == self.role('cart_view') and (name != self.role('cart')) and (self.role('cart') not in [q['name'] for q in [p]]) and self.cart.units and (self.cart_shown_version == self.cart_version) and (self.actor_calls_since_cart == 0) and (('view', self.cart_version) not in self.warned):
            self.warned.add(('view', self.cart_version))
            st = self.stock_tool()
            return (HOLD + "the counted cart is %s and the latest cart reply already listed its lines, so a view adds nothing. To refresh a line's stock use %s." % (self.cart.summary(), st or "a tool that shows that listing's stock"), None)
        sig = (name or '') + compact(args)
        if name not in self.finishers() and name not in (self.role('message'), self.role('cart')) and (sig in self.recent[-2:]):
            return (HOLD + 'identical call just made with nothing changed; its result already shows above; alter the arguments or move on.', None)
        if not scripted:
            g = self.reserve_gate(name, args)
            if g:
                return (HOLD + g, None)
        if name == self.role('cart'):
            v = self.cart_screen(args, puts)
            if v:
                return (HOLD + v, args)
        return (None, args)

    def cart_screen(self, args, puts):
        aim = self.cart.aimed(args)
        put = self.cart.is_put(args)
        ver = self.needs.version
        left = self.steps_left()
        if put and aim:
            cr = self.chain_reason(aim)
            if cr and self.chain_holds.get(aim, 0) < 2:
                self.chain_holds[aim] = self.chain_holds.get(aim, 0) + 1
                return cr + ' Pick another listing.'
        if not put and aim and ((aim, ver) in self.verified):
            key = ('rm', aim, self.put_count.get(aim, 0))
            cheaper = self.needs.priority() == 'cheapest' and (any((x != aim and self.records.price(x) is not None and (self.records.price(aim) is not None) and (self.records.price(x) < self.records.price(aim)) for x in puts)) or bool(self.cheaper_fit(aim)))
            same_price = self.records.price(aim) == self.verified.get((aim, ver))
            refusal_since = self.refusals > self.refusal_at_put.get(aim, self.refusals)
            if not cheaper and same_price and (not refusal_since) and (key not in self.warned):
                self.warned.add(key)
                return 'a separate check judged %s to satisfy all current requirements under the reading rules; only the category label settles what kind of item it is;%s nothing has changed since. Resend the removal only if some change or refusal excludes it, a field plainly fails, or a cheaper listing replaces it.' % (aim, ' it is the lowest-priced verified option;' if self.needs.priority() == 'cheapest' else '')
        if not put or aim is None:
            return ''
        if aim in self.cart.units and self.cart.mode != 'overwrites':
            key = ('stack', compact(args))
            if key not in self.warned:
                self.warned.add(key)
                return '%s is already in the cart with %s units and this cart tool adds units onto a held line (or that is not settled yet), so this call would increase it. To buy it as it stands, leave the cart alone; to change units, put in or take out the difference. Resend only for a deliberate add.' % (aim, money(self.cart.units[aim]))
        lines = self.cart.lines_of(args)
        wh = self.walk_hold([lid for lid, c in lines if c is None or c > 0])
        if wh:
            return wh
        if left > 3:
            for lid, c in lines[1:]:
                if (c is None or c > 0) and (lid, ver) not in self.put_checked:
                    self.put_checked.add((lid, ver))
                    v = self.item_check(lid, 'adding to the cart')
                    if v:
                        return v + ' Resend only if you are certain.'
            if (aim, ver) not in self.put_checked:
                self.put_checked.add((aim, ver))
                v = self.item_check(aim, 'adding to the cart')
                if v:
                    return v + ' Resend the same call only if you are certain the check is wrong.'
            for lid, c in lines:
                if c is not None and c <= 0:
                    continue
                r = self.rank_check(lid)
                if r:
                    cf = self.rank_verify(lid)
                    r = r.replace('before buying; if each fails, buy the current pick.', 'ahead of this put; if each fails, resend the same call.')
                    if cf:
                        r += ' An item check approved %s on every requirement at a lower price: put that one in instead, refresh its price and stock, then buy.' % cf
                    return r
        if self.needs.priority() == 'cheapest':
            for lid, c in lines:
                if c is not None and c <= 0:
                    continue
                cf = self.cheaper_fit(lid)
                if cf and cf not in self.cart.units and (('cheaper', lid, cf) not in self.warned):
                    self.warned.add(('cheaper', lid, cf))
                    return "%s passed the item check and costs less than %s; the lowest price was requested and the checker's reading outranks yours, so put it in instead. Resend only if some record proves it fails." % (cf, lid)
        ps = self.needs.products()
        key = ('qty', compact(args))
        if ps:
            qs = [x['quantity'] for x in ps]
            for lid, c in lines:
                cc = 1 if c is None else c
                if cc > 0 and cc not in qs and (key not in self.warned):
                    self.warned.add(key)
                    return 'line %s carries %s units, which matches none of the wanted quantities (%s; one line per product). Resend if you are sure.' % (lid, money(cc), ', '.join(('%s x %d' % (x['product'], x['quantity']) for x in ps)))
        else:
            c = self.cart.count_of(args)
            c = 1 if c is None else c
            want = self.needs.quantity()
            exp = want if self.cart.mode == 'overwrites' else want - self.cart.units.get(aim, 0)
            if c != exp and key not in self.warned:
                self.warned.add(key)
                return 'this call puts %s units but %s are expected (quantity %d). Resend if you are sure.' % (money(c), money(exp), want)
        return ''

    def finish_screen(self, name, args):
        kind = self.classify(name, args)
        if kind == 'order':
            return (self.order_holds_chain(), args)
        if kind == 'decline':
            return self.decline_holds_chain(name, args)
        return (None, args)

    def order_holds_chain(self):
        if self.todos:
            th = self.todo_hold()
            if th:
                return HOLD_LATER + th
            for _w in ALL_ANCHORS:
                for _t in self.due_todos(_w):
                    _fn = self.act_fn(_t['act'])
                    if _fn is not None and _t['act'] != 'none':
                        _fn(_t)
        left = self.steps_left()
        tl = self.wire.time_left()
        deadl = [l for l in self.cart.units if l in self.dead]
        if deadl and left > 2:
            state = (tuple(sorted(deadl)), self.reread_mark)
            if state not in self.swap_tried:
                n = self.order_holds.get(state, 0)
                if n < 2:
                    self.order_holds[state] = n + 1
                    return HOLD_LATER + self.dead_remark(deadl)
                if self.role('cart') and left >= 6 and (tl > CLOSING):
                    self.swap_tried.add(state)
                    saved = self.wire.mute
                    ok = self.code_shop(set(deadl), set(self.cart.units) - set(deadl))
                    self.wire.mute = saved
                    if ok and (not any((l in self.dead for l in self.cart.units))):
                        return HOLD_LATER + ('the code replaced the sold-out line and re-read stock; review the cart and order: ' + self.cart.summary())
        if left > 3:
            ch = self.chosen_line()
            for lid in ([ch] if ch else []) + [l for l in self.cart.units if l != ch]:
                cf = self.cheaper_fit(lid)
                if cf and cf not in self.cart.units:
                    n = self.cheaper_holds.get((lid, cf, self.gen), 0)
                    if n < 2:
                        self.cheaper_holds[lid, cf, self.gen] = n + 1
                        return HOLD_LATER + '%s passed the item check and costs less than %s: swap it in, recheck availability and cost, then order.' % (cf, lid)
        if self.cart.units:
            wh = self.walk_hold(list(self.cart.units))
            if wh:
                return HOLD_LATER + wh
        st = self.stock_tool()
        stale = [l for l in self.cart.units if self.stale(l)]
        if st and stale and (left > len(self.cart.units) + 2) and (tl > CLOSING):
            key = (tuple(sorted(self.cart.units)), self.reread_mark)
            if key not in self.rechecked:
                self.rechecked.add(key)
                self.enqueue(stale[:4], 'cart', "I re-read the cart lines' stock and price before ordering.", tool=st, recheck=True)
                return HOLD_LATER + 'availability and cost can shift silently and nobody looked at these lines after the latest cart change, so the code fetches them now; then order if each is available and the sum stays within budget, otherwise swap the line.'
        if self.vetted_gen != self.gen and left > 2 and (tl > CLOSING or not self.vetted_once):
            self.vetted_gen = self.gen
            self.vetted_once = True
            if left > 3:
                f = self.cart.faults(self.needs.products(), self.needs.quantity())
                if f:
                    return HOLD_LATER + 'cart problems: ' + '; '.join(f) + '.'
            ch = self.chosen_line()
            lines = list(self.cart.units) if self.needs.products() else [ch] if ch else []
            for lid in lines:
                v = self.item_check(lid, 'ordering')
                if v:
                    return HOLD_LATER + v
            for lid in lines:
                r = self.rank_check(lid)
                if r:
                    cf = self.rank_verify(lid)
                    if cf:
                        r += ' An item check approved %s on every requirement at a lower price: swap it in, re-read it, then order.' % cf
                    return HOLD_LATER + r
            return HOLD_LATER + "final checklist before ordering: (1) the cart holds a single line per requested product, each at its own quantity, and nothing else; (2) availability and cost were rechecked after the latest cart change through a tool showing both, the item is available, and price times quantity fits the current budget; (3) the item's own properties satisfy every stated requirement, revealed need and change; (4) when the lowest price or the best was requested, nothing better that fits exists among the listings found; (5) every procedural rule of the request was obeyed. Verify anything unverified, otherwise resend the order."
        return None

    def decline_holds_chain(self, name, args):
        nm = self.near_match_hold()
        if nm:
            return (HOLD_LATER + nm, args)
        left = self.steps_left()
        tl = self.wire.time_left()
        ver = self.needs.version
        ps = self.needs.products()
        vset = {x for x, v in self.verified if v >= self.needs.narrow_version}
        vlines = [l for l in self.cart.units if l in vset and l not in self.refused_lines and (l not in self.dead)]
        cap = self.needs.budget()
        if vlines and cap is not None and sum(((self.records.price(l) or 0) * max(1, u) for l, u in self.cart.units.items())) > cap:
            vlines = []
        if vlines and left > 1 and ('verified_cart' not in self.decline_once):
            self.decline_once.add('verified_cart')
            return (HOLD_LATER + "verified lines sit in the cart (%s); declining would waste them. Recheck their availability and cost, then order (with each product's line when several); declining is right only when a later change clearly excludes them." % ', '.join(vlines), args)
        if ps and 'slack' not in self.decline_once:
            self.decline_once.add('slack')
            rows, floor_sum = self.slack_rows()
            reads = self.reads_needed(rows)
            if rows and left > reads + 4 and (tl > CLOSING + 40):
                self.enqueue(rows, 'decline', 'I open affordable rows that fit inside what the budget leaves each product before declining.')
                return (HOLD_LATER + 'each product owns a fit, but the cheapest fits add up to %s, above the budget; these unopened rows sit within what the cap leaves each product and are being read now: %s. Assess every row for its own product under the reading rules and order an affordable combination; declining stays open only if nothing of that kind exists.' % (money(floor_sum), ', '.join(rows)), args)
        if ps and 'lacking' not in self.decline_once:
            self.decline_once.add('lacking')
            rows = self.rows_lacking()
            reads = self.reads_needed(rows)
            if rows and left > reads + 3 and (tl > CLOSING + 40):
                self.enqueue(rows, 'decline', 'I open rows of products that still lack a verified listing before declining.')
                return (HOLD_LATER + 'some products still lack a verified listing, so the code opens their unread affordable rows (%s); judge each for its own product under the reading rules, order an affordable combination, and decline only if nothing of that kind exists.' % ', '.join(rows), args)
        if 'named' not in self.decline_once and (self.role('details') or self.multi_tool()):
            self.decline_once.add('named')
            mt = self.multi_tool()
            rows = self.roster()[:mt[2] if mt else 3]
            reads = self.reads_needed(rows)
            if rows and left > reads + 3 and (tl > CLOSING + 40):
                self.enqueue(rows, 'decline', 'I open the listings the checks named before declining.')
                return (HOLD_LATER + 'listings named by the checks are still unopened (%s); the code opens them now. Judge each by the field named after the property (where the store spells the value itself only its words count; failing such a field another field, the listed options or the title can reveal it), buy one that qualifies, and decline only when all fail.' % ', '.join(rows), args)
        dr = self.decline_read_hold()
        if dr:
            return (HOLD_LATER + dr, args)
        if self.vetted_decline and len(self.records.read) > self.audit_read_count and (self.audits < 2) and (left > 3) and (tl > CLOSING + 30):
            a = self.audit()
            if a:
                return (HOLD_LATER + a, args)
        if not self.vetted_decline and left > 3 and (tl > CLOSING + 30):
            self.vetted_decline = True
            a = self.audit()
            if a:
                return (HOLD_LATER + a, args)
            return (HOLD_LATER + "before declining confirm: searches were varied (other words, filters, pages); each candidate's own property fields were inspected; every one breaks a stated requirement, a revealed need, the budget or availability; a property field counts under any spelling of its key. Buy whatever qualifies, otherwise resend the decline.", args)
        sugg = [f for f in self.audit_fits if f in self.records.known_set and f not in self.refused_lines]
        if 'audit_fits' not in self.decline_once and (not self.cart.units) and (left > 4) and (tl >= CLOSING + 30) and sugg:
            self.decline_once.add('audit_fits')
            passed = []
            for x in sugg[:2]:
                if x not in self.records.read:
                    continue
                v = self.item_check(x, 'ordering')
                if v and self.wire.time_left() > CLOSING + 30:
                    v = self.arbitrate(x, v)
                if not v and (x, ver) in self.verified:
                    passed.append(x)
                    if not ps:
                        break
            covered = not ps or all((any((self.category_of(y) in hs for y in self.verified_now())) for hs in self.home_labels() if hs))
            if passed and covered:
                return (HOLD_LATER + 'the audit named %s as possible fits, and the item checker, reading their full records, found every live requirement met; weigh them first: if they fit, put %s into the cart, re-fetch stock and price, then place the order; decline only if one turns out to fail a live requirement.' % (', '.join(passed), 'a verified line per product' if ps else passed[0]), args)
        if 'unrefuted' not in self.decline_once and left > 4:
            self.decline_once.add('unrefuted')
            cands = []
            for _ in range(3):
                cands = self.unrefuted()
                todo = [c for c in cands if (c, ver) not in self.verified]
                if not todo or tl < CLOSING + 30:
                    break
                self.item_check(todo[0], 'ordering')
                tl = self.wire.time_left()
            cands = self.unrefuted()
            if cands and left > len(cands) + 3 and (self.wire.time_left() > CLOSING + 20):
                return (HOLD_LATER + 'a decline fits only a shop with no qualifying listing, yet these candidates are available, affordable and unrefuted: %s. Put them in the cart (a single line per product, each at its own quantity), recheck, and order; decline only when every one clearly fails.' % ', '.join(cands), args)
        if 'differ' not in self.decline_once and left > 3 and self.open_differ():
            self.decline_once.add('differ')
            return (HOLD_LATER + '%s depends on another listing that the code could not settle; its value is therefore unknown, excludes nothing and gives no ground for declining. Put in whichever listing satisfies all other requirements (in stock, within budget; a line for each product when several), refresh stock and price, and order; decline only when nothing satisfies the remaining requirements.' % self.open_differ()[0]['head'], args)
        if left >= 3 and tl > CLOSING and (self.refusals or self.reserve_phase() or ps) and (('repair', self.refusals) not in self.decline_once):
            self.decline_once.add(('repair', self.refusals))
            saved = self.wire.mute
            try:
                p = self.picks()
                if p and all((self.code_fit.get(lid) for lid, _ in p)):
                    rej = self.cart_match(p)
                    if rej == [] and self.cart.units:
                        stale = [l for l in self.cart.units if self.stale(l)]
                        if stale:
                            self.stock_reread(stale)
                        return (HOLD_LATER + 'the stored records hold a fitting pick for each product, so the code put them in the cart: %s. Order now.' % self.cart.summary(), args)
            except Exception as exc:
                self.wire.note('repair_error', error=str(exc)[:200])
            finally:
                self.wire.mute = saved
        try:
            mc = self.meaning_check()
        except Exception as exc:
            self.wire.note('meaning_error', error=str(exc)[:200])
            mc = ''
        if mc:
            return (HOLD_LATER + mc, args)
        return (None, self.scrub_decline(name, args))

    def reads_needed(self, rows):
        mt = self.multi_tool()
        if mt and len(rows) >= 2:
            return -(-len(rows) // mt[2])
        return len(rows)

    def arbitrate(self, lid, objection):
        try:
            reason = ''
            for rec in reversed(self.log):
                if rec.get('kind') == 'decline_audit' and isinstance(rec.get('answer'), dict):
                    reason = str(rec['answer'].get('reason') or '')[:400]
                    break
            extra = 'ARBITRATION: two readings conflict over this item. The decline review, from the same records, judged it to meet all requirements (%s). An earlier check ruled: %s. Decide which is right by reading the records: a field giving the demanded value in different wording, or naming a qualified form, a shade or a variant spelling of that same value, satisfies it; a field stating some other value does not.' % (reason, objection[:500])
            v = self.item_check(lid, 'ordering', extra=extra)
            if v:
                return v
            ver = self.needs.version
            for key in list(self.verdicts):
                if key[0] == lid and key[1] == ver:
                    self.verdicts[key] = ''
            return ''
        except Exception:
            return objection

    def unrefuted(self):
        ps = self.needs.products()
        n = len(ps) if ps else 1
        labels = [self.walk_labels(k if ps else None) for k in range(n)]
        if not labels or any((not l for l in labels)):
            return []
        cap = self.needs.budget()
        if self.audit_answered:
            reviewed = set(self.records.read) if len(self.records.read) <= self.audit_read_count or not self.audit_reviewed else set(self.audit_reviewed)
            excluded_review = reviewed - set(self.audit_fits)
        else:
            excluded_review = set()
        chain = self.chain_listings()
        out, total = ([], 0.0)
        for k in range(n):
            units = self.units_for(k if ps else None)
            best = None
            for lid in sorted(self.records.read):
                if lid in out or lid in self.refused_lines or lid in self.dead or (lid in chain) or (self.records.stock_of(lid) is False) or self.failed_now(lid) or (lid in excluded_review):
                    continue
                if self.category_of(lid) not in labels[k]:
                    continue
                p = self.records.price(lid)
                if p is None or (cap is not None and p * units > cap):
                    continue
                key = (0 if (lid, self.needs.version) in self.verified else 1, p, lid)
                if best is None or key < best[0]:
                    best = (key, lid, p * units)
            if not best:
                return []
            out.append(best[1])
            total += best[2]
        if cap is not None and total > cap:
            return []
        return out

    def send_one(self, name, args, last=True):
        res = self.wire.env({'name': name, 'args': args})
        self.log.append({'kind': 'env', 'name': name, 'args': args, 'step': self.wire.step, 'reply': clip(res, 300) if res is not None else None})
        if res is None:
            self._last_text = 'The request failed on the network or server side; it may be retried.'
            return None
        obs = res.get('observation')
        try:
            self._last_text = self.absorb(name, args, obs, bool(res.get('finished')), last)
        except Exception as exc:
            self.wire.note('absorb_error', error=(type(exc).__name__ + ': ' + str(exc))[:300])
            self._last_text = compact(clip(obs, RECORD_FIELD_LEN))
        return res

    def absorb(self, name, args, obs, finished, last):
        work = lens(obs)
        remarks = []
        R = self.role
        if name == R('message'):
            self.sent_msgs += 1
            if self.todos and (not self.saying_todo):
                self.duty_credit(args)
        kind = self.classify(name, args)
        if kind == 'order':
            self.orders_tried += 1
        sig = (name or '') + compact(args)
        if name in (R('cart'), R('message')) or name in self.finishers():
            self.recent = []
        else:
            self.recent = (self.recent + [sig])[-2:]
        self.records.intake(work)
        self.answer_count += 1
        if name != R('cart') and (not self.covering):
            self.actor_calls_since_cart += 1
        newly_dead = self.learn_answer(name, args, work)
        if self.cart.units:
            named = set(self.ids_named(args))
            for d in dicts_in(work):
                rid = row_identity(d)
                if rid not in self.cart.units:
                    continue
                if self.stock_fields_seen:
                    if record_stock(d) is None:
                        continue
                    if not (self.rows_live and (not self.rows_lag)) and (not (rid in named or name in (R('cart'), R('cart_view')))):
                        continue
                elif prop_map(d) is None:
                    continue
                self.line_read[rid] = self.answer_count
        err = isinstance(work, dict) and bool(work.get('error'))
        if name == R('details') and (not err):
            for i in self.ids_named(args):
                self.records.read.add(i)
        if self.steps_left() < self.reserve() and (not self.covering) and (name in (R('details'), R('search')) or name in self.lookup_tools()):
            ids = self.ids_named(args)
            if name == R('details'):
                exempt = any((i in self.cart.units for i in ids))
            elif name == R('search'):
                exempt = False
            else:
                exempt = any((i in self.cart.units or any((i == v[0] for v in self.verified)) for i in ids))
            if not exempt:
                self.reserve_reads += 1
        if newly_dead:
            remarks.append('STOCK: ' + self.dead_remark(newly_dead))
        try:
            rr = self.redirect_remark(name, args)
            if rr:
                remarks.append(rr)
        except Exception as exc:
            self.wire.note('redirect_error', error=str(exc)[:200])
        if name == R('cart'):
            self.vetted_gen = -1
            self.cart.change_step = self.wire.step
            before = dict(self.cart.units)
            summary = self.cart.book(args, work)
            self.cart_version += 1
            self.gen += 1
            self.actor_calls_since_cart = 0
            if any((l not in before for l in self.cart.units)):
                self.change_counter = self.answer_count
                self.reread_mark += 1
            for l, u in self.cart.units.items():
                if l in before and u > before[l]:
                    self.line_read.pop(l, None)
            if self.stock_fields_seen:
                for d in dicts_in(work):
                    rid = row_identity(d)
                    if rid in self.cart.units and record_stock(d) is not None:
                        self.line_read[rid] = self.answer_count
            if self.cart.rows(work):
                self.cart_shown_version = self.cart_version
            is_put = self.cart.is_put(args)
            fresh = [l for l, c in self.cart.lines_of(args) if (c is None or c > 0) and l in self.cart.units and (l not in self.ever_put)]
            if is_put and fresh and (not self.is_fault(work)):
                self.ever_put |= set(fresh)
                st = [l for l in fresh if self.stale(l)]
                if st and self.stock_tool() and (not self.wire.mute) and (not self.covering) and (self.steps_left() >= 4) and (self.wire.time_left() >= CLOSING):
                    self.postput = (self.postput + st)[:4]
            if is_put:
                aim = self.cart.aimed(args)
                if aim:
                    self.refusal_at_put[aim] = self.refusals
                    self.put_count[aim] = self.put_count.get(aim, 0) + 1
                    if (aim, self.needs.version) in self.verified and aim in self.cart.units and (aim not in self.dead):
                        remarks.append('the checker accepted %s under all current requirements; keep it unless a later change excludes it.' % aim)
            if summary:
                remarks.append(summary)
        if name == R('cart_view') and name != R('cart'):
            self.cart.see_view(work, self.wire.step)
            self.cart_shown_version = self.cart_version
            remarks.append(self.cart.summary())
        refused = isinstance(work, dict) and 'refusal' in work or (name in self.finishers() and self.is_refusal(name, work, finished))
        if refused and kind == 'order':
            self.change_counter = self.answer_count
            self.reread_mark += 1
            self.gen += 1
        remarks += self.result_notes(name, args, work, refused, kind)
        if name == R('search'):
            n = self.search_note(args, work)
            if n:
                remarks.append(n)
            c = compact(args)
            if c in self.flagged_pages:
                self.flagged_pages.remove(c)
            try:
                self.joined_labels_from(work)
                fn = self.try_fusion()
                if fn:
                    remarks.append(fn)
            except Exception as exc:
                self.wire.note('fusion_error', error=str(exc)[:200])
        if last and name not in (R('cart'), R('message')) and (name not in self.finishers()) and (not self.early_audit_done) and (7 < self.steps_left() <= self.budget_steps // 3 + 2) and (not self.cart.units) and (not self.verified_now()) and (not self.wire.mute):
            self.early_audit_done = True
            a = self.audit(early=True)
            if a:
                remarks.append('No verified fit exists so far. ' + a)
        lines = []
        if name == R('message'):
            if isinstance(obs, str):
                lines = [obs]
            elif not err:
                for d in dicts_in(obs):
                    for v in d.values():
                        long_lines(v, lines)
        elif isinstance(obs, dict):
            for v in obs.values():
                if isinstance(v, list) and v and all((isinstance(x, str) for x in v)):
                    long_lines(v, lines)
            if refused:
                for d in dicts_in(obs):
                    for v in d.values():
                        if isinstance(v, str) and len(v.split()) >= 4 and (v not in lines):
                            lines.append(v)
        if lines and (not self.wire.mute):
            note = self.needs.hear(lines, self.current_question if name == R('message') else None)
            if note:
                remarks.append(note)
            while self.needs.borrowed:
                try:
                    self.ref_intake(*self.needs.borrowed.pop(0))
                except Exception as exc:
                    self.wire.note('ref_intake_error', error=str(exc)[:200])
        elif lines and self.wire.mute:
            self.needs.deferred.append((lines, None, time.monotonic()))
        text = compact(clip(obs, SEARCH_FIELD_LEN if name == R('search') else RECORD_FIELD_LEN), VIEW_LEN)
        if finished:
            text += '\n[the episode has finished]'
        remarks = [r for r in remarks if r]
        if remarks:
            text += '\nnote: ' + ' '.join(remarks)
        return text

    def result_notes(self, name, args, obs, refused, kind):
        out = []
        R = self.role
        if isinstance(obs, dict):
            err = str(obs.get('error') or '') + ' ' + str(obs.get('code') or '')
            if re.search('budget|allowance|limit|quota', err.lower()) and re.search('exceed|exhaust|used up|reached|over|no more|spent', err.lower()):
                if name == R('message') or self.allowance is not None:
                    self.allowance = self.sent_msgs if self.allowance is None else min(self.allowance, self.sent_msgs)
                    if name == R('message'):
                        self.code_asks = False
                out.append("this tool's allowance is used up; do not use it again.")
        if refused and name == R('decline') and (kind == 'decline'):
            self.audit_done = False
            self.vetted_decline = False
            self.event = True
            out.append("DECLINE REFUSED: the world rejected the decline and the episode continues; it maintains that something fitting can still be ordered. Study the refusal and act on it: re-examine the closest candidates against the current requirements (each one's own properties, availability and price) and order whichever fits; a second decline is acceptable only when every candidate genuinely fails.")
        elif refused:
            self.refusals += 1
            self.vetted_gen = -1
            self.event = True
            extra = ''
            try:
                extra = self.after_refusal(obs)
            except Exception as exc:
                self.wire.note('after_refusal_error', error=str(exc)[:200])
            self.refused_lines |= set(self.cart.units)
            texts = []
            for d in dicts_in(obs):
                for v in d.values():
                    for s in long_lines(v, []):
                        texts.append(re.sub('\\s+', ' ', s.lower()).strip())
            self.refusal_texts = texts
            out.append("ORDER REFUSED: the order was rejected, the episode continues and the cart is untouched. Carry out each change the refusal names, fix the cart by removing whatever no longer qualifies (typically an item that went unavailable), verify the replacement's availability and price afresh, then order again." + (' ' + extra if extra else ''))
        elif isinstance(obs, dict) and len(obs) >= 2 and (name != R('message')):
            got = []
            for v in obs.values():
                if isinstance(v, list) and v and all((isinstance(x, str) for x in v)):
                    long_lines(v, got)
            if got:
                texts = self.refusal_texts
                self.refusal_texts = []
                norm = [re.sub('\\s+', ' ', g.lower()).strip() for g in got]
                if texts and all((any((g in t for t in texts)) for g in norm)):
                    out.append('this result only repeats what the refused finish already carried; that was already dealt with and says nothing about this call.')
                else:
                    self.event = True
                    out.append('The shopper sent a change alongside this result; apply it: ' + ' | '.join(got[:4]))
        return out

    def after_refusal(self, obs):
        if len(self.cart.units) == 1 and (not self.wire.mute) and (not self.covering):
            line = next(iter(self.cart.units))
            text = []
            for d in dicts_in(obs):
                for v in d.values():
                    if isinstance(v, list):
                        long_lines(v, text)
                    elif isinstance(v, str) and len(v.split()) >= 4:
                        text.append(v)
            if text:
                snap = (self.needs.version, len(self.needs.dropped), len(self.needs.revealed), self.needs.budget(), tuple(self.needs.binding()))
                note = self.needs.hear(text, None)
                snap2 = (self.needs.version, len(self.needs.dropped), len(self.needs.revealed), self.needs.budget(), tuple(self.needs.binding()))
                if note and snap2[0] != snap[0] and (snap2[1:] == snap[1:]):
                    self.dead[line] = 'lookup'
                    self.wire.note('refusal_line_dead', line=line)
                    return note + ' The refusal reports an unavailable item while the cart had a single line, so the code counts %s as dead without a refresh. ' % line + self.dead_remark([line])
                return note
        st = self.stock_tool()
        lines = list(self.cart.units)[:4]
        if st and lines and (not self.wire.mute) and (not self.covering) and (not self.queue) and (self.steps_left() >= len(lines) + 3) and (self.wire.time_left() >= CLOSING):
            self.enqueue(lines, 'cart', "I re-read the cart lines' availability after the refusal in case one has gone.", tool=st, recheck=True)
            return 'The availability of the cart lines is being fetched now with %s.' % st
        return ''

    def post(self, res):
        if not res:
            return
        for i, (cid, text) in enumerate(res):
            if i == len(res) - 1:
                text = text + '\n' + self.status()
            self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

class PlanPart:

    def duty_reserve(self):
        return min(2, len([t for t in getattr(self, 'todos', []) if t['state'] == 'open' and t['act'] in ('message', 'total')]))

    def allowance_in_force(self):
        base = self.allowance if self.allowance is not None else DEFAULT_ALLOWANCE
        if self.saying_todo:
            return base
        hold = max(self.ask_reserve, self.duty_reserve())
        return max(0, base - hold) if hold else base

    def algo_num(self, key, default):
        v = self.algo.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return default
        lo, hi = ALGO_RANGE[key]
        return type(default)(max(lo, min(hi, v)))

    def cap_reads(self):
        return self.algo_num('reads', SCRIPTED_READS)

    def cap_hops(self):
        return self.algo_num('hops', 12)

    def share_refs(self, differ_only):
        return self.algo_num('refs_share', 0.3 if differ_only else 0.4)

    def stage_fn(self, name):
        fn = STAGE_FNS.get(name)
        return getattr(self, fn, None) if fn else None

    def act_fn(self, act):
        fn = ACT_FNS.get(act)
        return getattr(self, fn, None) if fn else None

    def act_message(self, t):
        return self.say_todo(t)

    def act_none(self, t):
        t['state'] = 'noted'
        return False

    def basket_total(self):
        if not self.cart.units:
            return None
        tot = 0.0
        for l, u in self.cart.units.items():
            p = self.records.price(l)
            if p is None:
                return None
            tot += p * max(1, u)
        return tot

    def act_total(self, t):
        if self.basket_total() is None:
            return False
        keep, keepc = (t['content'], t.get('cart'))
        t['cart'] = True
        if self.say_todo(t, late=True):
            return True
        t['content'], t['cart'] = (keep, keepc)
        return False

    def act_use(self, t):
        txt = (t['content'] or '').strip()
        if not txt or self.steps_left() <= 4 or self.wire.time_left() < CLOSING:
            return False
        i = txt.find('{')
        name = (txt[:i] if i > 0 else txt).strip().strip(':').strip()
        if name not in self.shapes:
            return False
        if name in {self.role(r) for r in ('order', 'decline', 'cart', 'message')}:
            return False
        try:
            raw = json.loads(txt[i:]) if i > 0 else {}
        except ValueError:
            return False
        if not isinstance(raw, dict):
            return False
        args, faults = check_args(raw, self.shapes.get(name) or {})
        if faults:
            return False
        cid = 'code_use_%d' % self.wire.step
        self.chat.append({'role': 'assistant', 'content': 'The request needs this store tool first.',
                          'tool_calls': [{'id': cid, 'type': 'function',
                                          'function': {'name': name, 'arguments': json.dumps(args)}}]})
        before = self.wire.step
        res = self.dispatch([{'id': cid, 'name': name, 'args': json.dumps(args)}], scripted=True)
        self.post(res)
        if self.wire.step <= before:
            return False
        t['state'] = 'done'
        self.gen += 1
        return True

    def act_recheck(self, t):
        st = self.stock_tool()
        if not st or not self.cart.units or self.steps_left() <= 4:
            return False
        lines = [l for l in self.cart.units if self.stale(l)] or list(self.cart.units)
        self.enqueue(lines[:3], 'cart', 'I re-read what is in the basket before going on.', tool=st, recheck=True)
        t['state'] = 'noted'
        return True

    def act_read(self, t):
        if self.ref_lookup(t['content'] or t['what']):
            t['state'] = 'noted'
            return True
        if 'refs' not in (self.flow or DEFAULT_FLOW):
            self.flow = list(self.flow or DEFAULT_FLOW) + ['refs']
            self.wire.note('flow_reopened', stage='refs', why=t['what'][:120])
            t['state'] = 'noted'
        return False

    def asks_stage(self):
        return bool(self.code_asks and self.question_step())

    def plan_flow(self):
        if not self.wire.usable(ROLE_PLANNER_ONE) or self.wire.time_left() < CLOSING + 30:
            return
        if self.wire.spend > PLAN_SHARE * self.wire.ceiling():
            return
        parts = [PROMPT_FLOW, 'REQUEST\n' + self.request]
        if self.opening is not None:
            parts.append('OPENING OBSERVATION\n' + compact(self.opening)[:2000])
        lines = []
        for t in self.tools or []:
            f = t.get('function') if isinstance(t, dict) else None
            f = f if isinstance(f, dict) else t
            if not isinstance(f, dict) or not f.get('name'):
                continue
            known = [r for r in ROLE_KEYS.values() if self.role(r) == f['name']]
            tail = ' [the code already uses this as %s]' % known[0] if known else ''
            lines.append('- %s(%s): %s%s' % (f['name'], ', '.join(sorted(((f.get('parameters') or {}).get('properties') or {}))),
                                             str(f.get('description') or '')[:200], tail))
        if lines:
            parts.append('TOOLS THIS WORLD OFFERS\n' + '\n'.join(lines))
        parts.append('BUDGETS\nsteps left %d; messages to the shopper allowed %d; seconds left %d'
                     % (self.steps_left(), self.allowance_in_force(), int(self.wire.time_left())))
        ans = self.wire.consult('\n\n'.join(parts), ROLE_PLANNER_ONE, CAP_FLOW)
        if not isinstance(ans, dict):
            return
        self.apply_plan(ans)
        self.plan_mark = self.state_mark()
        self.wire.note('flowplan', stages=self.flow, asks=self.allowance_in_force(), algo=self.algo,
                       todos=[t['what'] for t in self.todos])

    def apply_plan(self, ans, again=False):
        flow = []
        for nm in ans.get('stages') or []:
            nm = str(nm).strip().lower()
            if nm in DEFAULT_FLOW and nm not in flow:
                flow.append(nm)
        if flow:
            for nm in DEFAULT_FLOW:
                if nm not in flow:
                    flow.append(nm)
        self.flow = flow or None
        if self.ask_base is None:
            self.ask_base = self.allowance if self.allowance is not None else DEFAULT_ALLOWANCE
        asks = ans.get('asks')
        if isinstance(asks, bool):
            asks = None
        if isinstance(asks, (int, float)) and 0 <= int(asks) <= self.ask_base:
            self.ask_reserve = max(0, self.ask_base - int(asks) - self.sent_msgs)
        algo = ans.get('algorithm')
        if isinstance(algo, dict):
            for k, v in algo.items():
                if str(k) in ALGO_RANGE and isinstance(v, (int, float)) and (not isinstance(v, bool)):
                    self.algo[str(k)] = v
        known = {t['what'] for t in self.todos}
        for t in ans.get('todos') or []:
            if not isinstance(t, dict):
                continue
            what = str(t.get('what') or '').strip()
            if not what or what[:300] in known:
                continue
            act = str(t.get('act') or 'none').strip().lower()
            when = str(t.get('when') or 'before_finish').strip().lower()
            act = act if act in ACT_FNS else 'none'
            when = when if when in ALL_ANCHORS else 'before_finish'
            content = str(t.get('content') or '').strip()[:400]
            if what[:300] in {x['what'] for x in self.todos}:
                continue
            cart = False
            if act == 'message':
                both = what + ' ' + content
                if DUTY_FIRST.search(both) and DUTY_ORDER.search(both):
                    when = 'before_order'
                    cart = bool(DUTY_BASKET.search(both))
            self.todos.append({'what': what[:300], 'act': act, 'content': content,
                               'when': when, 'cart': cart, 'tries': 0, 'state': 'open'})
        self.wire.note('flowplan', stages=self.flow, asks=self.allowance, todos=[t['what'] for t in self.todos])

    def state_mark(self):
        done = len([t for t in self.trackers if t['status'] == 'resolved']) if hasattr(self, 'trackers') else 0
        vals = tuple(sorted((str(t['head']), str(t.get('value') or '')) for t in getattr(self, 'trackers', []) if t['status'] == 'resolved'))
        return (done, vals, self.needs.version, tuple(sorted(self.cart.units.items())), bool(self.event))

    def replan(self):
        if self.replans >= 2 or self.plan_mark is None:
            return False
        mark = self.state_mark()
        if mark == self.plan_mark:
            return False
        if not self.wire.usable(ROLE_PLANNER_ONE) or self.wire.time_left() < CLOSING + 20 or self.steps_left() <= 6:
            return False
        if self.wire.spend > REPLAN_SHARE * self.wire.ceiling():
            return False
        self.plan_mark = mark
        self.replans += 1
        parts = [PROMPT_FLOW_AGAIN, 'REQUEST\n' + self.request, 'WHAT IS KNOWN NOW\n' + self.needs.req_text()]
        res = [t for t in self.trackers if t['status'] == 'resolved'] if hasattr(self, 'trackers') else []
        if res:
            parts.append('REFERENCED VALUES NOW SETTLED\n' + '; '.join((str(t['head']) + ' = ' + str(t.get('value') or '?') for t in res)))
        if self.cart.units:
            parts.append('BASKET\n' + self.cart.summary())
        open_t = [t['what'] for t in self.todos if t['state'] == 'open']
        if open_t:
            parts.append('OBLIGATIONS STILL OPEN\n' + '; '.join(open_t))
        parts.append('BUDGETS\nsteps left %d; messages left %d; seconds left %d'
                     % (self.steps_left(), max(0, self.allowance_in_force() - self.sent_msgs), int(self.wire.time_left())))
        ans = self.wire.consult('\n\n'.join(parts), ROLE_PLANNER_ONE, CAP_FLOW)
        if not isinstance(ans, dict):
            return False
        self.apply_plan(ans, again=True)
        self.wire.note('flowreplan', n=self.replans, stages=self.flow, algo=self.algo,
                       todos=[t['what'] for t in self.todos if t['state'] == 'open'])
        return False

    def due_todos(self, when):
        return [t for t in self.todos if t['state'] == 'open' and t['when'] == when]

    def anchor_ready(self, when):
        if when == 'first':
            return True
        if when == 'after_needs':
            return not self.open_hints()
        if when == 'before_cart':
            return not self.cart.units
        if when == 'after_cart':
            return bool(self.cart.units)
        return False

    def duty_waiting(self):
        return any((t['state'] == 'open' and t['act'] == 'message' and t['when'] == 'before_order' and t.get('cart') for t in self.todos))

    def duty_mark(self):
        return tuple(sorted(self.cart.units.items()))

    def duty_text(self, t):
        base = re.sub('\\s{2,}', ' ', re.sub('\\[[^\\]]*\\]', '', t['content'] or t['what'])).strip()
        if not t.get('cart') or not self.cart.units:
            return base or t['what']
        lines, total = ([], 0)
        for lid, u in self.cart.units.items():
            title = ''
            for d in self.records.carriers(lid):
                got = record_title(d)
                if got:
                    title = got
            p = self.records.price(lid)
            if total is not None:
                total = None if p is None else total + p * u
            if len(lines) < DUTY_LINES:
                head = ' '.join(words_of(title)[:6]) or lid
                lines.append('%s x %s%s' % (money(u), head, '' if p is None else ' at ' + money(p)))
        out = 'Here is the final basket: ' + '; '.join(lines)
        if len(self.cart.units) > len(lines):
            out += ' and %d more' % (len(self.cart.units) - len(lines))
        if total is not None:
            out += '. Total ' + money(total)
        return out + '. Reply yes and I will place the order.'

    def duty_credit(self, args):
        txt = ' '.join((str(v) for v in (args or {}).values() if isinstance(v, str)))
        if not self.cart.units or not DUTY_BASKET.search(txt):
            return
        tot = self.basket_total()
        if tot is not None and money(tot) not in txt:
            return
        for t in self.todos:
            if t['act'] == 'message' and t['when'] == 'before_order' and t.get('cart') and t['state'] != 'done':
                t['state'] = 'done'
                t['at'] = self.duty_mark()
                t['said'] = t.get('said', 0) + 1

    def duty_due(self):
        for t in self.todos:
            if t['state'] == 'done' and t['act'] == 'message' and t['when'] == 'before_order' and t.get('cart') and (t.get('at') != self.duty_mark()) and (t.get('said', 0) < 2):
                t['state'] = 'open'
        return self.due_todos('before_order')

    def duty_before_order(self):
        for t in self.duty_due():
            if t['act'] == 'message' and self.say_todo(t, late=True):
                return True
        return False

    def say_todo(self, t, late=False):
        cap = self.ask_base if self.ask_base is not None else self.allowance_in_force()
        if late and self.allowance is not None:
            cap = max(cap, self.allowance)
        if not self.role('message') or cap - self.sent_msgs <= 0:
            return False
        if self.steps_left() <= (1 if late else 2) or self.wire.time_left() < (DUTY_FLOOR if late else LAST_CALL):
            return False
        if t.get('cart') and not self.cart.units:
            return False
        args = self.build_msg(self.duty_text(t))
        if args is None:
            return False
        cid = 'code_todo_%d' % self.wire.step
        self.chat.append({'role': 'assistant', 'content': 'I do what the request asked for.',
                          'tool_calls': [{'id': cid, 'type': 'function',
                                          'function': {'name': self.role('message'), 'arguments': json.dumps(args)}}]})
        before = self.sent_msgs
        self.saying_todo = True
        try:
            res = self.dispatch([{'id': cid, 'name': self.role('message'), 'args': json.dumps(args)}], scripted=True)
            self.post(res)
        finally:
            self.saying_todo = False
        if self.sent_msgs <= before:
            return False
        t['state'] = 'done'
        t['at'] = self.duty_mark()
        t['said'] = t.get('said', 0) + 1
        self.ask_reserve = max(0, self.ask_reserve - (self.sent_msgs - before))
        self.gen += 1
        return True

    def todo_run(self):
        for when in LOOP_ANCHORS:
            if self.anchor_ready(when) and self.todo_step(when):
                return True
        return False

    def todo_step(self, when):
        for t in self.due_todos(when):
            fn = self.act_fn(t['act'])
            if fn is not None and fn(t):
                return True
        return False

    def ref_lookup(self, text):
        want = [w for w in re.findall(r'[A-Za-z]*[0-9][A-Za-z0-9_-]*', str(text)) if len(w) >= 3]
        ids = []
        for w in want:
            for k in (w, w.upper()):
                if k in self.records.known_set and k not in ids:
                    ids.append(k)
        ids = [k for k in ids if k not in self.records.read]
        if not ids or not self.steps_left() > 3:
            return False
        self.enqueue(ids[:3], 'menu', 'I read the listing the request points at before shopping.', recheck=True)
        return True

    def todo_hold(self):
        for t in self.duty_due():
            fn = self.act_fn(t['act'])
            if fn is not None and t['act'] != 'none' and (self.say_todo(t, late=True) if t['act'] == 'message' else fn(t)):
                return None
            t['tries'] = t.get('tries', 0) + 1
            key = t['what']
            n = self.todo_holds.get(key, 0)
            if n >= 2:
                t['state'] = 'given_up'
                continue
            self.todo_holds[key] = n + 1
            return 'the request asks for this before the order: ' + t['what'] + (
                (' Say this to the shopper: ' + t['content']) if t['content'] else '')
        return None

    def review_todos(self):
        if self.reviewed or not self.todos:
            return
        self.reviewed = True
        for t in self.due_todos('before_finish'):
            fn = self.act_fn(t['act'])
            if fn is not None and t['act'] != 'none':
                fn(t)
        left = [t for t in self.todos if t['state'] in ('open', 'given_up')]
        if not left or not self.wire.usable(ROLE_PLANNER_ONE) or self.wire.time_left() < CLOSING:
            return
        if self.wire.spend > REVIEW_SHARE * self.wire.ceiling():
            return
        did = []
        for e in self.log[-60:]:
            if isinstance(e, dict) and e.get('kind') == 'action':
                did.append(str(e.get('name') or '') + ' ' + compact(e.get('args'))[:120])
        parts = [PROMPT_FLOW_REVIEW, 'OBLIGATIONS\n' + '\n'.join(('- ' + t['what']) for t in left)]
        if did:
            parts.append('WHAT THE ASSISTANT DID\n' + '\n'.join(did[-30:]))
        ans = self.wire.consult('\n\n'.join(parts), ROLE_PLANNER_ONE, CAP_FLOW)
        if not isinstance(ans, dict):
            return
        miss = [str(m).strip().lower() for m in (ans.get('missing') or []) if str(m).strip()]
        if not miss:
            return
        self.wire.note('review', missing=miss)
        for t in left:
            if any((m in t['what'].lower() or t['what'].lower() in m for m in miss)):
                fn = self.act_fn(t['act'])
                if fn is not None and t['act'] != 'none':
                    fn(t)


class Episode(LoopPart, PlanPart, SelfRedirectPart, JoinedPart, CheapPart, MeaningPart, RefsPart, CodePart, QuestionsPart, ChecksPart, EpisodeBase):

    def __init__(self, problem_data):
        EpisodeBase.__init__(self, problem_data)
        self.put_count = {}
        self.rechecked = set()
        self.reread_mark = 0
        self.current_question = None
        self._last_text = ''
        self.menu_picked = set()
        self.trackers = []
        self.ref_steps = 0
        self.hop_calls = 0
        self.cand_read = set()
        self.ref_pending = []
        self.chain_holds = {}
        self.meaning_done = False
        self.meaning_cart = None
        self.walked = set()
        self.walk_queued = None
        self.decline_reads = None
        self.read_width = 0
        self.g_tries = {}
        self.g_settled = set()
        self.g_sets = {}
        self.labels_indexed = None
        self.value_queries = {}
        self.k_reads = {}
        self.k_pages = {}
        self.k_floor = {}
        self.k_ceil = {}
        self.joined = {}
        self._redir_cache = {}
        self.redirect_warned = set()

def _safe(log):
    try:
        return json.loads(json.dumps(log, default=str))
    except Exception:
        return [{'kind': 'log_unserialisable'}]

def agent_main(problem_data):
    ep = Episode(problem_data)
    try:
        ep.run()
    except Exception as exc:
        ep.wire.note('crash', error=(type(exc).__name__ + ': ' + str(exc))[:500])
        if not ep.wire.finished and (not ep.wire.blocked):
            try:
                ep.forced_finish()
            except Exception as exc2:
                ep.wire.note('crash', error=(type(exc2).__name__ + ': ' + str(exc2))[:500])
    return _safe(ep.log)
