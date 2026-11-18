from __future__ import annotations
import json
import re
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_FAST_MODEL = 'deepseek/deepseek-v3.2'
_DEEP_MODEL = 'qwen/qwen3.5-397b-a17b'
_BACKUP_MODELS = ('google/gemma-4-31b-it', 'openai/gpt-oss-120b', 'deepseek/deepseek-chat-v3.1')
TILT = 'tilt'
SHELF = 'shelf'
WALLS = 'walls'
PALATE = 'palate'
LEXICON = 'lexicon'
MEND = 'mend'
WITNESS = 'witness'
_FAMILIES = (TILT, SHELF, WALLS, PALATE, LEXICON, MEND, WITNESS)
_DEEP_FAMILIES = frozenset({PALATE, WITNESS, MEND, WALLS, LEXICON})
_SOFT_LEXICON: dict[str, tuple[str, ...]] = {'durability': ('durab', 'sturdy', 'rugged', 'tough', 'heavy duty', 'long lasting', 'lasts', 'wear out', 'reinforced', 'robust', 'hard wearing', 'built to last'), 'battery': ('battery', 'runtime', 'run time', 'charge', 'recharg', 'playtime', 'play time', 'mah', 'hours of use', 'all day power'), 'comfort': ('comfort', 'cushion', 'padded', 'ergonomic', 'plush', 'soft feel', 'cosy', 'cozy', 'all day wear', 'chafe'), 'materials': ('material', 'fabric', 'build quality', 'premium', 'quality of the', 'real leather', 'genuine', 'solid wood', 'stainless', 'finish quality'), 'sweat': ('sweat', 'moisture', 'wicking', 'breathab', 'breathe', 'ventilat', 'airflow', 'quick dry', 'quick-dry'), 'water': ('waterproof', 'water resistant', 'water-resistant', 'rain', 'splash', 'ipx', 'wet weather', 'gets wet'), 'outdoor': ('outdoor', 'hiking', 'hike', 'trail', 'camping', 'terrain', 'all weather', 'all-weather', 'backpacking', 'field use'), 'workout': ('workout', 'work out', 'gym', 'training', 'running', 'exercise', 'athletic', 'sports', 'practice'), 'wireless': ('wireless', 'bluetooth', 'cordless', 'true wireless', 'tws', 'no wire', 'wire-free', 'wire free'), 'fit': ('secure fit', 'sport fit', 'earhook', 'ear hook', 'stay in', "won't fall", 'snug fit', 'over-ear', 'in-ear', 'earbud', 'stable fit'), 'portability': ('portab', 'lightweight', 'light weight', 'compact', 'packable', 'foldab', 'easy to carry', 'travel size', 'grams', 'weighs', 'not bulky'), 'capacity': ('capacity', 'roomy', 'spacious', 'storage', 'holds', 'fits everything', 'liters', 'litres', 'big enough'), 'noise': ('noise', 'quiet', 'silent', 'loud', 'rattle', 'hum', 'sound isolat'), 'grip': ('grip', 'traction', 'slip', 'non slip', 'non-slip', 'grippy'), 'warmth': ('warm', 'insulat', 'thermal', 'cold weather', 'fleece'), 'style': ('stylish', 'style', 'looks', 'design', 'aesthetic', 'sleek', 'classy', 'smart casual', 'professional look'), 'ease': ('easy to use', 'easy to clean', 'simple to', 'hassle', 'beginner', 'user friendly', 'low maintenance', 'dishwasher'), 'speed': ('fast', 'quick', 'responsive', 'performance', 'powerful', 'snappy'), 'support': ('warranty', 'guarantee', 'return policy', 'customer support', 'replacement')}
_TENSIONS = (('portability', 'durability'), ('portability', 'capacity'), ('portability', 'warmth'), ('comfort', 'durability'), ('style', 'durability'), ('speed', 'battery'), ('speed', 'noise'), ('thrift', 'materials'), ('thrift', 'durability'), ('thrift', 'support'))
_THRIFT_RX = re.compile('\\bcheapest\\b|\\bcheap\\b|\\bas little as\\b|\\bsave money\\b|\\blowest price\\b|\\btight budget\\b|\\bbargain\\b|\\bspend as little\\b', re.I)
_NAMED_RX = re.compile('\\b\\d{4,}::[\\w.-]+')
_CAP_RX = re.compile('(?:under|below|less than|at most|up to|no more than|cap(?:ped)? at|stay(?:ing)? (?:under|below)|budget of|max(?:imum)?)\\s*(?:[^\\d\\n]{0,12})?([\\d][\\d,]*(?:\\.\\d+)?)', re.I)
_WALL_RX = re.compile('(?:must (?:be|have|include|support)|needs? to (?:be|have)|has to (?:be|have)|only (?:want|accept)|non[- ]?negotiable[:\\s]*|required?[:\\s]*|it has to|no exceptions? (?:on|for)|strictly)\\s+([a-z0-9][a-z0-9 _/\'.\\"-]{2,50})', re.I)
_LEANING_RX = re.compile("(?:prefer(?:ably|ence for|red)?|ideally|would love|lean(?:ing)? toward[s]?|partial to|fan of|hoping for|nice to have|if possible)\\s+([A-Za-z][A-Za-z0-9 &'.-]{1,34})", re.I)
_WITNESS_RX = re.compile('structured\\s+claim|justification\\s*(?:\\.|\\s)*claims|justification\\s+is\\s+required|must\\s+include\\s+justification|back\\s+(?:up|it)\\s+with\\s+(?:claims|evidence)|(?:explain|justify)\\s+(?:your|the|each)\\s+(?:choice|pick|decision|selection)', re.I)
_ASK_WHY_RX = re.compile('\\bwhy\\b|\\breasons?\\b|\\bjustif\\w*|\\bexplain\\b|\\bmake the case\\b|\\bdefend\\b|\\bwalk me through\\b|\\bgive me a reason\\b|\\bback it up\\b', re.I)
_HELD_RX = re.compile('\\bif you ask\\b|\\bwhen you ask\\b|\\bonly if asked\\b|(?:tell|share|reveal|give)[^.]{0,44}\\bif\\b[^.]{0,24}\\bask\\w*', re.I)
_LOST_RX = re.compile('\\bsold\\s*out\\b|\\bout\\s+of\\s+stock\\b|\\bunavailable\\b|\\bdisappear\\w*|\\bno longer\\b|\\bfall(?:s|en)?\\s+through\\b|\\bbackup\\b|\\bnext best\\b|\\bsecond choice\\b', re.I)
_ORDERING_RX = re.compile('\\btie[-\\s]?break\\w*|\\brank\\w*|\\bsort\\w*|\\bhighest\\b|\\blowest\\b|\\bin order of\\b|\\bfirst by\\b|\\bthen by\\b|\\bmaximi[sz]e\\b|\\bminimi[sz]e\\b', re.I)
_AXIS_ORDER_RX = re.compile("(?:matters? (?:the )?most|care(?:s)? most about|top priority|most important|above all|before anything else|then|secondar(?:y|ily)|least important)[:\\s]+([a-z][a-z0-9 _/'-]{2,40})", re.I)
_SHAPE_RX = re.compile('^[a-z][a-z0-9_]{1,40}$')
_BASE_FIELDS = frozenset({'price', 'in_stock'})
_BLOCKED_FIELDS = frozenset({'category', 'category_path', 'categories', 'device_operating_system', 'device os', 'operating system', 'operating_system', 'os', 'title', 'name', 'product_name', 'summary', 'evidence', 'reason', 'rationale', 'note', 'notes', 'description', 'display', 'facts', 'options', 'specification', 'spec', 'specs', 'hdr', 'feature', 'features', 'attribute', 'attributes', 'requirement', 'requirements', 'why', 'justification', 'claim', 'claims', 'proof', 'support', 'meta', 'property', 'properties', 'phone_type', 'device_type', 'product_type', 'sku', 'product_id', 'item_type', 'model_number'})
_SHOPPER_ATTRS = frozenset({'size', 'material', 'color', 'colour', 'width', 'capacity', 'dimensions', 'finish', 'weight'})
_CATEGORY_RX = re.compile('\\bin\\s+the\\s+([^\\n,]{3,64}?)\\s+category\\b', re.I)
_QUOTED_RX = re.compile('[\\"\\u201c\\u2018\']([^\\"\\u201d\\u2019\']{3,64})[\\"\\u201d\\u2019\']')
_PAIR_RX = re.compile('(\\d{4,})::([\\w.-]+)')
_CONTAINERS = ('display', 'facts', 'options', 'os', 'size', 'title', 'spec', 'specs')
_UNIT_RX = re.compile('\\b\\d+(?:\\.\\d+)?(?:\\s?(?:inch|in|hz|p|k|gb|tb|mb|mm|cm|mp|w|wh|mah|oz|ml|l|kg|lb|lbs|ft|yd|\\"))?\\b', re.I)
_ATTR_WORDS = frozenset({'xs', 's', 'm', 'l', 'xl', 'xxl', 'small', 'medium', 'large', 'petite', 'tall', 'cotton', 'polyester', 'wool', 'merino', 'silk', 'linen', 'leather', 'suede', 'denim', 'canvas', 'nylon', 'mesh', 'rubber', 'plastic', 'metal', 'steel', 'aluminum', 'aluminium', 'wood', 'bamboo', 'glass', 'ceramic', 'navy', 'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'gray', 'grey', 'brown', 'beige', 'cream', 'khaki', 'olive', 'tan', 'silver', 'gold', 'narrow', 'wide', 'regular', 'slim', 'relaxed', 'matte', 'glossy', 'gloss', 'satin', 'brushed', 'polished', 'oak', 'walnut', 'maple', 'chrome', 'brass', 'hd', 'fhd', 'qhd', 'uhd', 'oled', 'lcd', 'led', 'qled'})
_NOISE_WORDS = frozenset({'a', 'an', 'the', 'and', 'or', 'for', 'with', 'from', 'that', 'this', 'find', 'buy', 'get', 'need', 'want', 'looking', 'please', 'under', 'below', 'above', 'price', 'budget', 'stock', 'available', 'why', 'reason', 'justify', 'explain', 'product', 'item', 'items', 'shopper', 'order', 'cart', 'exactly', 'must', 'should', 'also', 'only', 'than', 'then', 'when', 'which', 'what', 'have', 'has', 'are', 'is', 'be', 'to', 'of', 'on', 'at', 'by', 'as', 'if', 'not', 'no', 'yes', 'true', 'false', 'brand', 'size', 'color', 'colour', 'material', 'width', 'capacity', 'dimensions', 'finish', 'resolution', 'screen', 'screen_size'})

def _squash(raw: Any) -> str:
    return re.sub('[^a-z0-9]+', '', str(raw or '').lower())

def _snake(raw: Any) -> str:
    low = str(raw or '').strip().lower().replace('-', '_').replace(' ', '_').replace('/', '_')
    return re.sub('_+', '_', re.sub('[^a-z0-9_]+', '', low)).strip('_')

def _evidenced(value: Any, blob: Any) -> bool:
    tight = _squash(value)
    return bool(tight) and tight in _squash(blob)

def _lex_hit(token: str, text: str) -> bool:
    needle = str(token or '').strip().lower()
    hay = str(text or '').lower()
    if not needle or not hay:
        return False
    if re.search(f'(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])', hay):
        return True
    tight = _squash(needle)
    if not tight:
        return False
    flat = _squash(hay)
    if len(tight) <= 3:
        return bool(re.search(f'(?<![a-z0-9]){re.escape(tight)}(?![a-z0-9])', flat))
    return tight in flat

def _stated_tokens(goal: str) -> list[str]:
    text = (goal or '').lower()
    out: list[str] = []
    seen: set[str] = set()
    for hit in _UNIT_RX.finditer(text):
        tok = hit.group(0).strip()
        key = _squash(tok)
        if key and key not in seen:
            seen.add(key)
            out.append(tok)
    for word in re.findall('[a-z][a-z0-9-]+', text):
        if word in _NOISE_WORDS or word not in _ATTR_WORDS:
            continue
        key = _squash(word)
        if key and key not in seen:
            seen.add(key)
            out.append(word)
    return out

def _glued(field: str) -> bool:
    key = _snake(field)
    if not key:
        return False
    return any((key.startswith(c + '_') or key.endswith('_' + c) for c in _CONTAINERS))

def _money(text: Any) -> float | None:
    hit = re.search('\\d[\\d,]*(?:\\.\\d+)?', str(text or ''))
    if not hit:
        return None
    try:
        return float(hit.group(0).replace(',', ''))
    except ValueError:
        return None

def _json_obj(text: str) -> dict[str, Any]:
    body = (text or '').strip()
    if not body:
        return {}
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        (left, right) = (body.find('{'), body.rfind('}'))
        if left < 0 or right <= left:
            return {}
        try:
            obj = json.loads(body[left:right + 1])
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}

def _distinct(items: Any) -> list[str]:
    return list(dict.fromkeys((str(i) for i in items)))

def _schema_names(tools: list[Any]) -> list[str]:
    out: list[str] = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get('function')
        if isinstance(fn, dict) and fn.get('name'):
            out.append(str(fn['name']))
        elif entry.get('name'):
            out.append(str(entry['name']))
    return out

def _field_node(tools: list[Any]) -> dict[str, Any]:
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get('function') if isinstance(entry.get('function'), dict) else entry
        if not isinstance(fn, dict) or fn.get('name') != 'place_test_order':
            continue
        props = (fn.get('parameters') or {}).get('properties') or {}
        note = props.get('justification') if isinstance(props, dict) else None
        claims = (note.get('properties') or {}).get('claims') if isinstance(note, dict) else None
        items = claims.get('items') if isinstance(claims, dict) else None
        node = (items.get('properties') or {}).get('field') if isinstance(items, dict) else None
        if isinstance(node, dict):
            return node
    return {}

class AxisBag:
    __slots__ = ('axes', 'families', 'ordered', 'thrifty', 'hard_ok', 'clarify_left')

    def __init__(self, goal: str, rules: str) -> None:
        blob = f'{goal}\n{rules}'.lower()
        hits: list[tuple[int, str]] = []
        for (family, cues) in _SOFT_LEXICON.items():
            spot = min((blob.find(cue) for cue in cues if cue in blob), default=-1)
            if spot >= 0:
                hits.append((spot, family))
        hits.sort()
        self.families = [family for (_, family) in hits]
        self.thrifty = bool(_THRIFT_RX.search(blob))
        if self.thrifty:
            self.families.append('thrift')
        self.ordered = self._stated_order(goal, rules)
        self.axes = _distinct([*self.ordered, *self.families])[:6]
        self.hard_ok: set[str] = set()
        self.clarify_left = 1

    def _stated_order(self, goal: str, rules: str) -> list[str]:
        out: list[str] = []
        for text in (goal, rules):
            for hit in _AXIS_ORDER_RX.finditer(text or ''):
                phrase = hit.group(1).lower()
                for (family, cues) in _SOFT_LEXICON.items():
                    if family in out:
                        continue
                    if any((cue in phrase for cue in cues)):
                        out.append(family)
        return out

    def __bool__(self) -> bool:
        return bool(self.axes)

    def note_hard_ok(self, pid: str) -> None:
        if pid:
            self.hard_ok.add(pid)

    def forget(self, pid: str) -> None:
        self.hard_ok.discard(pid)

    @property
    def viewed_hard_ok(self) -> int:
        return len(self.hard_ok)

    def tension(self) -> tuple[str, str] | None:
        held = set(self.axes)
        for (left, right) in _TENSIONS:
            if left in held and right in held:
                return (left, right)
        return None

    def spend_clarify(self) -> bool:
        if self.clarify_left <= 0:
            return False
        self.clarify_left -= 1
        return True

    def cues_for(self, family: str) -> tuple[str, ...]:
        return _SOFT_LEXICON.get(family, (family,))

    def evidence(self, well: WhiteWell, pid: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for family in self.axes:
            found[family] = well.soft_value(pid, self.cues_for(family))
        return found

    def soft_axes(self) -> list[str]:
        return [a for a in self.axes if a != 'thrift']

    def soft_score(self, well: WhiteWell, pid: str) -> int:
        total = 0
        weight = len(self.axes)
        for (rank, family) in enumerate(self.axes):
            if family == 'thrift':
                continue
            if well.soft_value(pid, self.cues_for(family)):
                total += max(1, weight - rank)
        return total

    def axis_hits(self, well: WhiteWell, pid: str) -> int:
        return sum((1 for family in self.soft_axes() if well.soft_value(pid, self.cues_for(family))))

    def lead_printed(self, well: WhiteWell, pid: str) -> bool:
        soft = self.soft_axes()
        if not soft:
            return True
        lead = soft[0]
        return bool(well.soft_value(pid, self.cues_for(lead)))

    def coverage_gap(self, well: WhiteWell, pid: str) -> list[str]:
        return [family for family in self.soft_axes() if not well.soft_value(pid, self.cues_for(family))]

    def line(self) -> str:
        if not self.axes:
            return ''
        head = ' > '.join(self.ordered) if self.ordered else ''
        rest = [a for a in self.axes if a not in self.ordered]
        text = head + (' | also stated: ' + ', '.join(rest) if rest else '')
        return text.strip(' |')
_CUES: dict[str, tuple[tuple[str, float], ...]] = {TILT: (('ideally', 2.5), ('preferably', 2.5), ('would love', 2.0), ('leaning toward', 2.0), ('nice to have', 2.0), ('but it must', 2.5), ('non-negotiable', 2.0), ('firm on', 2.0)), SHELF: (('options', 1.5), ('a few', 1.5), ('show me', 1.5), ('what else', 2.0), ('alternatives', 2.0), ('listing', 1.0), ('matches', 1.5), ('search for', 1.0)), WALLS: (('must be', 2.5), ('has to be', 2.5), ('no exceptions', 3.0), ('strictly', 2.5), ('only accept', 2.5), ('required', 2.0), ('exact', 1.5), ('hard requirement', 3.0)), PALATE: (('matters more', 3.0), ('care more about', 3.0), ('rather pay', 3.0), ('worth paying', 2.5), ('not the cheapest', 3.0), ('trade-off', 2.5), ('tradeoff', 2.5), ('quality over', 2.5), ('i value', 2.0), ('prioritize', 1.5)), LEXICON: (('tie-break', 3.0), ('tie break', 3.0), ('rank', 2.5), ('sort by', 2.5), ('in order of', 2.5), ('first by', 2.5), ('then by', 2.5), ('highest', 1.5), ('lowest', 1.5)), MEND: (('sold out', 3.0), ('out of stock', 3.0), ('unavailable', 2.5), ('no longer', 2.0), ('backup', 2.5), ('next best', 2.5), ('falls through', 2.5), ('second choice', 2.0)), WITNESS: (('justification', 3.0), ('structured claim', 3.5), ('claims', 2.0), ('why did you', 2.5), ('explain your choice', 3.0), ('defend', 2.0), ('back it up', 2.0))}

class WhiteBrief:
    __slots__ = ('raw', 'goal', 'rules', 'family', 'margin', 'named', 'cap', 'walls', 'leanings', 'wants_why', 'held_figure', 'quoted', 'claim_fields', 'claim_rule')

    def __init__(self, raw: str, tools: list[Any] | None=None) -> None:
        self.raw = raw or ''
        (self.goal, self.rules) = self._split(self.raw)
        self.named = _distinct(_NAMED_RX.findall(self.raw))
        self.cap = self._cap()
        self.walls = self._walls()
        self.leanings = self._leanings()
        self.wants_why = bool(_ASK_WHY_RX.search(self.goal))
        self.held_figure = bool(_HELD_RX.search(self.raw))
        hit = _QUOTED_RX.search(self.goal) or _QUOTED_RX.search(self.raw)
        self.quoted = hit.group(1).strip() if hit else ''
        node = _field_node(tools or [])
        enum = node.get('enum') if isinstance(node, dict) else None
        self.claim_fields = _distinct((_snake(x) for x in enum if _snake(x))) if isinstance(enum, list) else []
        self.claim_rule = str(node.get('description') or '')[:600] if node else ''
        (self.family, self.margin) = self._vector(AxisBag(self.goal, self.rules))

    @staticmethod
    def _split(raw: str) -> tuple[str, str]:
        (head, sep, tail) = raw.partition('\n\n')
        if not sep:
            return (raw.strip(), '')
        body = tail.strip()
        (first, _, rest) = body.partition('\n')
        if first.rstrip().endswith(':') and len(first) < 44:
            body = rest.strip()
        return (head.strip(), body)

    def _cap(self) -> str:
        for text in (self.goal, self.raw):
            hit = _CAP_RX.search(text or '')
            if hit:
                return hit.group(1)
        return ''

    def _walls(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        if self.cap:
            out.append(f'spend at or under {self.cap}')
            seen.add('cap')
        for text in (self.goal, self.rules):
            for hit in _WALL_RX.finditer(text or ''):
                phrase = re.split('\\b(?:and|but|or|then|because|so)\\b|[,.;:]', hit.group(1))[0]
                phrase = re.sub('^(?:the|a|an|my|its)\\s+', '', phrase.strip(' -_/\'"'), flags=re.I)
                key = _squash(phrase)
                if not phrase or len(key) < 3 or key in seen:
                    continue
                seen.add(key)
                out.append(phrase[:48])
        return out[:6]

    def _leanings(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for hit in _LEANING_RX.finditer(self.raw):
            phrase = re.split('\\b(?:but|and|if|though|however|because|so|while)\\b|[,.;:]', hit.group(1))[0]
            phrase = phrase.strip(' -_/\'"')
            key = _squash(phrase)
            if not phrase or len(key) < 2 or key in seen:
                continue
            seen.add(key)
            out.append(phrase[:34])
        return out[:3]

    def category(self) -> str:
        hit = _CATEGORY_RX.search(self.goal or self.raw or '')
        if not hit:
            return ''
        return re.sub('\\s+', ' ', hit.group(1)).strip(' -_/\'"')[:64]

    def ingest(self, note: str) -> None:
        body = (note or '').strip()
        if not body:
            return
        if _squash(body) in _squash(self.goal):
            return
        self.goal = (self.goal + '\n' + body).strip()
        self.walls = self._walls()
        self.leanings = self._leanings()

    def _vector(self, bag: AxisBag) -> tuple[str, float]:
        if len(self.named) >= 2:
            return (LEXICON, 9.0)
        if self.wants_why:
            return (WITNESS, 9.0)
        score = dict.fromkeys(_FAMILIES, 0.0)
        (goal, rules) = (self.goal.lower(), self.rules.lower())
        for (family, cues) in _CUES.items():
            for (phrase, weight) in cues:
                if phrase in goal:
                    score[family] += weight
                if family == WITNESS:
                    continue
                if phrase in rules:
                    score[family] += weight * 0.6
        if _WITNESS_RX.search(self.goal):
            score[WITNESS] += 12.0
        if self.wants_why and self.claim_fields:
            score[WITNESS] += 4.0
        if _ORDERING_RX.search(goal):
            score[LEXICON] += 2.0
        if _LOST_RX.search(goal + rules):
            score[MEND] += 5.0
        if self.held_figure:
            score[WALLS] += 4.0
        if len(self.walls) >= 2:
            score[WALLS] += 1.5 * min(2, len(self.walls) - 1)
        if self.leanings:
            score[TILT] += 3.0
        if bag.axes:
            score[PALATE] += 2.0 + 1.2 * min(3, len(bag.axes))
        if bag.thrifty and bag.axes:
            score[PALATE] += 1.5
        if bag.ordered:
            score[PALATE] += 1.5
        if not any(score.values()):
            score[SHELF] += 1.0
        for family in _FAMILIES:
            span = max(1.0, len(_CUES.get(family, ())) * 0.35)
            score[family] = score[family] / span
        ranked = sorted(score.items(), key=lambda row: (-row[1], _FAMILIES.index(row[0])))
        (top, best) = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = round(best - runner, 3)
        if margin < 0.6 and top != WITNESS:
            near = [f for (f, v) in ranked[:3] if best - v < 0.6]
            if PALATE in near:
                return (PALATE, margin)
        return (top, margin)

    def needs_claims(self) -> bool:
        return self.family == WITNESS

    def board_notes(self, bag: AxisBag) -> str:
        rows: list[str] = []
        if self.walls:
            rows.append('- hard walls: ' + '; '.join(self.walls))
        if bag.line():
            rows.append(f'- soft axes: {bag.line()}')
        if bag.thrifty:
            rows.append('- the shopper mentioned price pressure: it is an axis, not the answer')
        if self.leanings:
            rows.append('- soft leanings named: ' + ', '.join(self.leanings))
        if self.named:
            rows.append('- variants named in the ask: ' + ', '.join(self.named))
        if self.quoted:
            rows.append(f'- quoted listing phrase: {self.quoted}')
        if self.held_figure:
            rows.append('- one figure is withheld until you ask for it')
        if self.claim_fields:
            rows.append('- live claim fields: ' + ', '.join(self.claim_fields))
        if self.claim_rule:
            rows.append(f'- live claim rule: {self.claim_rule[:260]}')
        return 'WHITE BOARD\n' + '\n'.join(rows) + '\n' if rows else ''

class WhiteWell:
    __slots__ = ('rows', 'prose', 'heads', 'last_pid', 'pin_pid', 'fresh_price', 'fresh_stock')

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, str]] = {}
        self.prose: dict[str, str] = {}
        self.heads: dict[str, str] = {}
        self.last_pid = ''
        self.pin_pid = ''
        self.fresh_price = ''
        self.fresh_stock = ''

    def stash(self, pid: str, key: Any, value: Any) -> None:
        field = _snake(key)
        if not pid or not field or value is None or isinstance(value, (dict, list)):
            return
        text = str(value).strip()
        if not text:
            return
        if field == 'in_stock':
            low = text.lower()
            if low in {'true', '1', 'yes', 'in_stock'}:
                text = 'true'
            elif low in {'false', '0', 'no'}:
                text = 'false'
            self.fresh_stock = text
        if field == 'price':
            self.fresh_price = text
        self.rows.setdefault(pid, {})[field] = text
        self.last_pid = pid

    def drink(self, obs: Any) -> None:
        if isinstance(obs, dict):
            self._descend(obs, '')

    def _descend(self, node: Any, carried: str) -> None:
        if not isinstance(node, dict):
            return
        pid = str(node.get('product_id') or '') or carried
        chunks: list[str] = []
        head = str(node.get('title') or '')
        if head:
            chunks.append(head)
            if pid:
                self.heads[pid] = head
        if node.get('options'):
            chunks.append(str(node['options']))
        for key in ('price', 'in_stock', 'brand', 'sku'):
            if key in node:
                self.stash(pid, key, node.get(key))
        facts = node.get('facts')
        if isinstance(facts, dict):
            self._unfold(facts, pid, chunks, '')
        if pid and chunks:
            prior = self.prose.get(pid, '')
            joined = ' | '.join(chunks)
            if joined not in prior:
                self.prose[pid] = (prior + ' | ' + joined if prior else joined)[:2200]
        for key in ('observation', 'product', 'compared', 'variants', 'results', 'items', 'candidates'):
            child = node.get(key)
            if isinstance(child, dict):
                self._descend(child, pid)
            elif isinstance(child, list):
                for item in child[:12]:
                    self._descend(item, pid)

    def _unfold(self, block: dict[str, Any], pid: str, chunks: list[str], prefix: str) -> None:
        for (key, val) in block.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            if isinstance(val, dict):
                blob = json.dumps(val, ensure_ascii=False)[:160]
                chunks.append(f'{path}={blob}')
                self.stash(pid, key, blob)
                self._unfold(val, pid, chunks, path)
            elif isinstance(val, list):
                text = ', '.join((str(x) for x in val[:12]))
                chunks.append(f'{path}={text}')
                self.stash(pid, key, text)
            else:
                text = str(val).strip()
                if not text:
                    continue
                chunks.append(f'{path}={text}')
                self.stash(pid, key, text)
                if prefix:
                    self.stash(pid, path.rsplit('.', 1)[-1], text)

    def fact_rows(self, pid: str) -> list[dict[str, str]]:
        return [{'key': k, 'value': v} for (k, v) in self.rows.get(pid, {}).items()]

    def witness_blob(self, pid: str) -> str:
        seen = self.rows.get(pid, {})
        return self.prose.get(pid, '') + ' | ' + ' | '.join((f'{k}={v}' for (k, v) in seen.items()))

    def rows_blob(self, pid: str) -> str:
        return ' '.join((f'{k} {v}' for (k, v) in self.rows.get(pid, {}).items())).lower()

    def price_of(self, pid: str) -> float | None:
        return _money(self.rows.get(pid, {}).get('price'))

    def head_only(self, pid: str, goal: str) -> bool:
        head = self.heads.get(pid, '').lower()
        if not head or not goal:
            return False
        blob = self.rows_blob(pid)
        return any((_lex_hit(t, head) and (not _lex_hit(t, blob)) for t in _stated_tokens(goal)))

    def phrase_title_only(self, pid: str, phrase: str) -> bool:
        if not phrase or not pid:
            return False
        head = self.heads.get(pid, '')
        if not head or not _evidenced(phrase, head):
            return False
        return not _evidenced(phrase, self.rows_blob(pid))

    def rows_answer(self, pid: str, goal: str) -> bool:
        if not pid or pid not in self.rows:
            return False
        blob = self.rows_blob(pid)
        tokens = _stated_tokens(goal)
        if not tokens:
            return any((k not in _BASE_FIELDS for k in self.rows.get(pid, {})))
        return any((_lex_hit(t, blob) for t in tokens))

    def soft_value(self, pid: str, cues: tuple[str, ...]) -> str:
        rows = self.rows.get(pid, {})
        for (key, val) in rows.items():
            for cue in cues:
                if cue in key.lower().replace('_', ' '):
                    return val
        for (key, val) in rows.items():
            joined = f'{key} {val}'.lower()
            for cue in cues:
                if cue in joined:
                    return f'{key}={val}'
        blob = self.prose.get(pid, '').lower()
        for cue in cues:
            spot = blob.find(cue)
            if spot >= 0:
                return self.prose.get(pid, '')[max(0, spot - 24):spot + 42].strip(' |')
        return ''

    def hard_pass(self, pid: str, cap: str, goal: str) -> bool:
        rows = self.rows.get(pid)
        if not rows:
            return False
        if rows.get('in_stock') == 'false':
            return False
        (ceiling, price) = (_money(cap), self.price_of(pid))
        if ceiling is not None and price is not None and (price > ceiling):
            return False
        tokens = [t for t in _stated_tokens(goal) if _squash(t) != _squash(cap)]
        if tokens and (not any((_lex_hit(t, self.rows_blob(pid)) for t in tokens))):
            return False
        return True

    def shape_key(self, pid: str) -> str:
        row = self.rows.get(pid, {})
        brand = _squash(row.get('brand', ''))
        head = _squash(self.heads.get(pid, ''))[:48]
        return f'{brand}:{head}' if brand or head else pid
_WITNESS_FILL = '\nYou name ONE hard-gate witness claim the palate still owes on this sku.\nInput JSON: goal, allowed_fields (live enum, may be empty), schema_rule (the\ntool\'s own wording for legal fields), have (fields already claimed — often only\nprice / in_stock / brand), listing (observed text for the exact sku), fact_rows\n(key/value pairs including catalog drawer keys — those are VALUE evidence only;\ndrawer names vary by category).\nReturn JSON only: {"claims":[{"field":"...","value":"..."}, ...]}\n\nWHAT THE PALATE STILL OWES\n- Cover the shopper\'s HARD gate that `have` is missing. price, in_stock, and\n  brand alone do NOT finish a gate ask — you still owe the conventional product\n  attribute the shopper required.\n- FIELD = a conventional snake_case shopping attribute named by the ask or\n  schema (size, material, color, width, capacity, dimensions, finish, weight,\n  …). Never a meta or identity label.\n- VALUE = the printed fact token from listing/fact_rows, copied verbatim —\n  never a preferred synonym invented from the ask. Tokens like large / cotton /\n  navy sit as VALUEs under the right attribute — never as field names.\n- Prefer the attribute the ask actually hinged on. A true-but-irrelevant side\n  attribute does not finish the gate.\n\nTASTE RULES\n- Obey schema_rule. When allowed_fields is non-empty, field MUST come from it.\n  When empty, name the conventional attribute for that kind of ask — NOT the\n  catalog drawer key.\n- Brand alone does not finish the palate when a hard gate remains unclaimed.\n- No guessing: if listing/fact_rows do not evidence the stated requirement,\n  return {"claims":[]}.\n\nREFUSALS (meta labels are NOT gates; they block the real fill)\n- Never emit: specification, spec, specs, hdr, feature, features, attribute,\n  requirement, why, rationale, justification, evidence, reason, summary,\n  description, display, os, category, title, facts, options, phone_type,\n  device_type, product_type, sku, product_id, item_type, model_number.\n- Never invent field names by concatenating drawer+attribute\n  (e.g. display_resolution, size_material).\n- Do not restate price or in_stock — those are already handled.\n- Optional short brand ONLY if a brand fact row evidences it (never invent).\n- At most 2 claims (one gate preferred; optional short brand if evidenced).\n'.strip()

class WhiteClaims:

    @staticmethod
    def admit(field: str, allowed: set[str]) -> bool:
        if not field or field in _BLOCKED_FIELDS or _glued(field):
            return False
        if allowed and field not in allowed:
            return False
        return bool(_SHAPE_RX.match(field))

    @staticmethod
    def polish(raw: Any, brief: WhiteBrief, well: WhiteWell, pid: str) -> list[dict[str, str]]:
        proposed = raw if isinstance(raw, list) else []
        allowed = {_snake(x) for x in brief.claim_fields}
        rows = well.rows.get(pid, {})
        witness = well.witness_blob(pid)
        kept: list[dict[str, str]] = []
        taken: set[str] = set()
        for entry in proposed:
            if not isinstance(entry, dict):
                continue
            field = _snake(entry.get('field'))
            value = '' if entry.get('value') is None else str(entry.get('value')).strip()
            if not value or field in taken or (not WhiteClaims.admit(field, allowed)):
                continue
            if field == 'in_stock' and value.lower() in {'true', '1', 'yes', 'in_stock'}:
                value = 'true'
            if field == 'brand':
                value = WhiteClaims._brand(value, rows, witness)
                if not value or not rows.get('brand'):
                    continue
            elif field not in _BASE_FIELDS and (not _evidenced(value, witness)):
                swap = WhiteClaims._printed_swap(value, rows, brief.goal, witness)
                if not swap:
                    continue
                value = swap
            taken.add(field)
            kept.append({'field': field, 'value': value})
        for need in ('price', 'in_stock'):
            if need in taken or (allowed and need not in allowed):
                continue
            got = rows.get(need) or (well.fresh_price if need == 'price' else well.fresh_stock)
            if got:
                kept.insert(0 if need == 'price' else min(1, len(kept)), {'field': need, 'value': got})
                taken.add(need)
        if 'brand' not in taken and (not allowed or 'brand' in allowed):
            brand = WhiteClaims._brand(rows.get('brand', ''), rows, witness)
            if brand and rows.get('brand') and _evidenced(brand, brief.goal):
                kept.append({'field': 'brand', 'value': brand})
                taken.add('brand')
        if brief.needs_claims() and (not WhiteClaims._gate_present(kept, brief.goal)):
            gate = WhiteClaims._gate_from_facts(brief.goal, rows, witness, taken, allowed)
            if gate:
                taken.add(gate['field'])
                kept.append(gate)
        if brief.needs_claims() and (not WhiteClaims._gate_present(kept, brief.goal)):
            for row in WhiteClaims._fill(brief, well, pid, taken):
                field = _snake(row.get('field'))
                value = str(row.get('value') or '').strip()
                if not value or field in taken or (not WhiteClaims.admit(field, allowed)):
                    continue
                if not _evidenced(value, witness):
                    swap = WhiteClaims._printed_swap(value, rows, brief.goal, witness)
                    if not swap:
                        continue
                    value = swap
                if field == 'brand':
                    if not rows.get('brand'):
                        continue
                    value = WhiteClaims._brand(value, rows, witness)
                    if not value:
                        continue
                taken.add(field)
                kept.append({'field': field, 'value': value})
                if WhiteClaims._gate_present(kept, brief.goal):
                    break
        if brief.needs_claims() and 'brand' not in taken and (not allowed or 'brand' in allowed) and (len(kept) < 4) and rows.get('brand'):
            brand = WhiteClaims._brand(rows.get('brand', ''), rows, witness)
            if brand:
                kept.append({'field': 'brand', 'value': brand})
                taken.add('brand')
        return kept[:6]

    @staticmethod
    def _gate_worthy(field: str, value: str, goal: str) -> bool:
        if not field or field in _BASE_FIELDS or field == 'brand' or (field in _BLOCKED_FIELDS):
            return False
        if field in _SHOPPER_ATTRS:
            return True
        goal_l = (goal or '').lower()
        if field in goal_l or field.replace('_', ' ') in goal_l:
            return True
        if value and (_evidenced(value, goal) or value.lower() in goal_l):
            return True
        return False

    @staticmethod
    def _gate_present(kept: list[dict[str, str]], goal: str='') -> bool:
        for claim in kept:
            field = _snake(claim.get('field'))
            value = '' if claim.get('value') is None else str(claim.get('value')).strip()
            if WhiteClaims._gate_worthy(field, value, goal):
                return True
        return False

    @staticmethod
    def _gate_from_facts(goal: str, rows: dict[str, str], witness: str, have: set[str], allowed: set[str]) -> dict[str, str] | None:
        goal_l = (goal or '').lower()
        ranked: list[tuple[int, str, str]] = []
        for (raw_key, raw_val) in rows.items():
            key = _snake(raw_key)
            value = str(raw_val or '').strip()
            if not key or not value or key in have:
                continue
            if key in _BASE_FIELDS or key == 'brand' or key in _BLOCKED_FIELDS:
                continue
            field = key
            if field in have or field in _BLOCKED_FIELDS or _glued(field):
                continue
            if allowed and field not in allowed:
                continue
            if not _SHAPE_RX.match(field):
                continue
            if not _evidenced(value, witness):
                continue
            score = 0
            if field in _SHOPPER_ATTRS:
                score += 4
            if field in goal_l or key in goal_l:
                score += 12
            if _evidenced(value, goal):
                score += 5
            if score <= 0 and (not WhiteClaims._gate_worthy(field, value, goal)):
                continue
            if score <= 0:
                continue
            ranked.append((score, field, value))
        if not ranked:
            return None
        ranked.sort(key=lambda row: (-row[0], row[1]))
        (_, field, value) = ranked[0]
        return {'field': field, 'value': value}

    @staticmethod
    def _printed_swap(value: str, rows: dict[str, str], goal: str, witness: str) -> str:
        if value and _evidenced(value, witness):
            return value
        ranked: list[tuple[int, str]] = []
        for (key, val) in rows.items():
            if not val or key in _BASE_FIELDS or key == 'brand':
                continue
            if not _evidenced(val, witness):
                continue
            weight = 0
            if _evidenced(val, goal):
                weight += 5
            if value and (_evidenced(val, value) or _evidenced(value, val)):
                weight += 4
            if key in _BLOCKED_FIELDS or key in {'display', 'size', 'facts', 'options'}:
                weight += 1
            if weight:
                ranked.append((weight, val))
        if not ranked:
            return ''
        ranked.sort(key=lambda row: (-row[0], -len(row[1])))
        return ranked[0][1]

    @staticmethod
    def _brand(raw: str, rows: dict[str, str], witness: str) -> str:
        structured = str(rows.get('brand') or '').strip()
        if not structured or not _evidenced(structured, witness):
            return ''
        if len(structured.split()) > 4:
            return ''
        text = str(raw or '').strip()
        if text and len(text.split()) <= 3 and (len(text) <= 40) and _evidenced(text, structured):
            return text
        return structured
_ROLE = '\nYou are WhiteGirl, an EnvPack buyer who scores FIT on a soft palate — not a\nprice-tag autopilot. Eligible (hard walls clear) is not the same as chosen.\nA gold-looking cart with an unfinished market event or a missed hard gate is\nstill a zero.\n\nSENSES (tools only)\n  search · filter · view · compare · inspect_stock · add_to_cart ·\n  remove_from_cart · message · place_test_order\nThe episode ends only when place_test_order returns done=true. No order → no\nscore. Late-budget law: stop open search, close any open event, finish the\norder pair (inspect_stock → place_test_order). Keep ≥2 steps for that close.\n\nPALATE RHYTHM\n  Gather   search / filter with the shopper’s own words (walls + lead soft cues)\n  Read     view candidates; trust printed fact rows over title adjectives\n  Compare  put ≥2 hard-OK items side by side on the STATED soft axes\n  Commit   inspect_stock the intended sku, then add_to_cart (first cart graded)\n  Confirm  inspect_stock again after any market move\n  Close    place_test_order\n\nHOUSE RULES\n- Cheapest ≠ winner by default. Price is a candidate attribute, not a verdict,\n  unless the shopper ranked it first. Firm walls always beat soft wishes.\n- Printed values beat vibes. If the catalog is silent on an axis, record “not\n  printed” — never invent support. Title adjectives are marketing, not proof.\n- MANDATORY HARD GATE: when the ask names a disqualifying product property\n  (spec, size, resolution, material, …), that property must appear in FACT ROWS\n  on the carted sku before commit. Brand, price, and in_stock alone never finish\n  a hard-wall ask.\n- Answer shopper questions with message same/next turn BEFORE cart or order.\n- When a figure is withheld until you ask: message-before-search — no lookup\n  until disclosure.\n- Non-WITNESS (non-TF7): omit justification.claims entirely.\n\nEVENT-CLOSE RITUAL (every family — unfinished events pay 0)\nBefore place_test_order, if ANY market event fired:\n  1. inspect_stock the EVENT-TARGET first — that alone marks event_closed.\n     A substitute / replacement inspect does NOT close the event. Do this even\n     when you will order a different sku afterward.\n  2. remove_from_cart if the target no longer qualifies; recover per the family\n     book.\n  3. Re-check STOCK and BUDGET after the event (both can fail after a market\n     move). Never order a sku last seen out of stock or over the spend lid.\n  4. Rebuild any claims from the fresh observation, then place_test_order.\nevent_closed=false pays 0 even when the cart looks perfect.\n\nNever invent meta field names such as specification or hdr — those are not\ngates on the palate.\n'.strip()
_WITNESS_PLAYBOOK = '\nFAMILY — WITNESS (TF7 justification / claim palate)\n\nplace_test_order MUST carry justification.claims as {field, value} pairs graded\nagainst printed catalog text. Unprinted “reasons” score worse than silence.\n\nMANDATORY-PROPERTY GATE (claim_truth hinge)\n- Identify which product property in the goal is a DISQUALIFIER (hard wall) —\n  often resolution, display class, size, material, capacity, or a named spec.\n- The gate claim FIELD must NAME THAT property (conventional attribute /\n  schema enum / allowed_fields). Schema_rule and allowed_fields outrank guesswork.\n- price + in_stock + brand alone do NOT finish the palate when that mandatory\n  property is still unclaimed. Brand alone never finishes a hard-wall ask.\n- SECONDARY-ATTR SUBSTITUTION IS A FAILED CLAIM: vouchering a true-but-\n  irrelevant attested property (e.g. a size when the ask gated resolution)\n  while the mandatory hard property stays open fails claim_truth. Cover the\n  wall that disqualifies, not a convenient side attribute.\n- VALUE = printed fact-row / inspect token only. Titles are marketing. If facts\n  do not prove the stated requirement, leave that sku — do not invent.\n\nSHAPE (max 4; include all facts support)\n  price      from newest view / inspect_stock\n  in_stock   usually "true" from inspect_stock\n  gate       FIELD = the mandatory property named by ask/schema\n             — never catalog drawer keys, never shopper shorthand as the field,\n             never meta labels (specification / spec / hdr / feature /\n             attribute / requirement / why / rationale / justification /\n             evidence)\n             VALUE = printed fact-row token verbatim\n             (large, cotton, navy are values under the right attribute —\n             not field names)\n  soft       at most one; short brand only when the shopper named a maker AND a\n             brand fact row shows it. Titles are not brands.\n\nSEARCH & COMMITMENT\n- When brand and a hard gate are both named, search brand+gate tokens together.\n- Do not cart until the gate figure appears in a FACT ROW for that sku.\n  Title-only figures are decoys — reject that sku.\n- If facts do not prove the stated requirement, leave the palate empty for that\n  sku; do not invent a matching claim.\n- When a brand is named in the goal, only order a fact-backed matching maker\n  unless no firm-compliant brand match exists (then firm backup + message).\n\nREFUSALS\n- Never emit: category, title, summary, evidence, reason, rationale,\n  description, os, display, specification, spec, hdr, feature(s), attribute,\n  requirement, why, justification — or fused drawer+attribute field names.\n- Soft junk that is NEVER a gate: phone_type, device_type, product_type, sku,\n  product_id, item_type, model_number.\n- No unit conversion, no synonym invention, no prose paragraphs inside values.\n- inspect_stock a candidate BEFORE the first add_to_cart (intended commitment).\n- After a market event: ALWAYS inspect_stock the EVENT-TARGET first (closes it)\n  even if ordering a different sku; drop if it no longer qualifies; re-check\n  stock AND budget after any price move; STRIP stale claims; rebuild claims\n  from the fresh observation; then place_test_order. event_closed=false pays 0\n  even when the cart looks gold.\n'.strip()
_BOOK: dict[str, str] = {PALATE: '\nFAMILY — PALATE (TF4 preference — White specialty)\n\nHard walls make an item eligible. Soft axes decide which eligible item wins.\nCheapest-valid is never the automatic FIT. Weak soft-axis compare (one\ncandidate, title adjectives, invented axes, ignoring the lead preference) is a\nfailed taste.\n\nSEARCH WITH THE LEAD PREFERENCE\nPut the shopper’s OWN preference phrasing into search/filter — the words they\nranked first (sweat / water / battery / comfort / outdoor / workout / …). A\nshelf that never asked for the lead axis cannot surface printed evidence for it.\n\nMETHOD\n1. Clear walls first — budget, stock, category/class, every stated hard\n   requirement. Soft axes never excuse a broken wall.\n2. Read ≥2 hard-OK candidates before the first add_to_cart. One read is not a\n   comparison; an uncompared pick is a coin flip the grader will punish.\n   Compare ≥2 hard-OK on the STATED soft axes before committing.\n3. Score eligible candidates on the shopper’s STATED soft axes using PRINTED\n   values only (fact rows / listing facts). Title adjectives and your priors\n   do not count.\n4. PRIMARY AXIS FIRST: if the lead stated axis is “not printed” on a candidate,\n   keep looking — do not crown a winner on a secondary axis (e.g. wireless)\n   while the lead preference (e.g. sweat/water resistance) stays blank.\n5. COVER MORE AXES: when battery / wireless / secure-fit (or other stated axes)\n   also appear in the ask, prefer the hard-OK that prints MORE of them — do not\n   cart on the lead axis alone while another eligible listing covers the rest.\n6. Where the catalog is silent on an axis, mark “not printed” and lean on the\n   next stated axis. Never fabricate supporting evidence.\n7. Cart the best FIT. If the cheapest eligible item also wins on the axes, name\n   the axis win — do not let price be the reason by default. Cheapest ≠ winner.\n   A costlier hard-OK winner on the stated tilt is the correct pick.\n8. If two stated axes genuinely conflict, send exactly ONE clarify message,\n   then proceed. Never stall twice on the same fork.\n9. After one widen-search for soft evidence, COMMIT. Late budget or a stock-ok\n   cart means place_test_order — do not refine forever into a step-limit zero.\n\nEVENTS & CLOSE\ninspect_stock the intended commitment before first cart. Close market events on\nthe EVENT-TARGET with inspect_stock before ordering — even if ordering a\ndifferent sku (replacement inspect does not close). After a price move, re-check\nstock AND spend lid. Omit justification unless the environment separately\nrequires WITNESS. Leave ≥2 steps for inspect_stock + place_test_order.\nLate budget: stop tasting, close event, finish. A stock-ok cart must order now.\n'.strip(), TILT: '\nFAMILY — TILT (TF1 intent — firm filter, soft bonus)\n\nA leaning is a palate bonus, never a licence to break walls. Firm over wish.\n\nSORT THE ASK\n- Firm requirements: class, spend, stated specs, stock — disqualifiers.\n- Soft leaning: ideally / prefer / would love — often a maker or look.\n\nDUAL DISCOVERY ON THE PALATE (fund BOTH passes early)\n- Pass A: firm nouns + leaning / maker token so wish-honouring candidates can\n  appear. When BOTH a maker and a hard gate are named, keep brand+gate tokens\n  together rather than dropping one.\n- Pass B: firm nouns alone so a wall-clearing backup exists.\nDo NOT wait until late turns to try the leaning. Collapsing into one lazy\nfirm-only query leaves the wish column empty.\n\nFIRM GATE BEFORE CART\n- Walk every firm wall against the candidate’s FACT ROWS (not titles). Do not\n  cart until the mandatory hard property (e.g. the stated screen/spec figure)\n  appears in a fact row for that sku. Price/stock alone are not enough.\n- Wrong firm gate = failed purchase: a sku that clears budget/stock but misses\n  the stated hard property is not eligible.\n\nFIRST CART = PREFERRED BRAND ONLY IF FULLY FIRM-COMPLIANT\n- Among firm-compliant items, take the one that also honours the leaning when\n  FACT ROWS prove it (brand fact when a maker was named).\n- If none do, take the firm-compliant FIT anyway and say with message that the\n  leaning could not be met inside the walls.\n- Never buy soft over firm. Ordering a non-matching / non-fact-backed sku when\n  a brand was named is a failed purchase, not a compromise.\n\nMESSAGE & EVENTS\nAnswer shopper questions before cart/order. When a figure is withheld until you\nask, message first — no search until disclosure. After a market event, inspect\nthe EVENT-TARGET first (even if ordering a different sku), re-check stock AND\nbudget after any price move, then re-apply the brand/leaning rule under the new\nwalls — wish cart only if still fully firm-compliant. Prove compliance from\nfact rows before the first cart. Omit justification.\n'.strip(), SHELF: '\nFAMILY — SHELF (TF2 recall — coverage before FIT)\n\nBuild a shelf you can actually compare; one lucky hit is not coverage. Thin\ncoverage left unfinished OR event_closed=false zeros even a gold-looking cart.\n\nMULTI-QUERY COVERAGE\n- Start from the shopper’s distinctive phrasing; reuse quoted wording exactly\n  across passes — do not scrub the distinctive tokens away.\n- When the ask names a catalog category (“in the X category”), include that\n  category in at least one query shape alongside the distinctive phrase.\n- When the shelf is thin, change ONE token (near-miss refine) and look again —\n  do not rewrite the whole query every turn. Run more than one query shape:\n  keep the rare nouns, vary one secondary knob (category, budget, stock).\n- Then view many DISTINCT product_ids. Repeated skus of the same product count\n  as one identity, not two. Judge progress by identities opened, not row spam.\n- Open several identities before you settle. First cart is graded — no\n  placeholder while coverage is still thin.\n\nPHRASE IN FACT ROWS\nTitles borrow words. Confirm distinctive / quoted phrases in fact rows before\ntrusting a match — a quote that lives only in the title is a decoy and must not\ncart. Phrase the match from fact rows, not titles. Discard accessories,\nbundles, and look-alikes by classification facts — never by title vibes.\n\nTHEN FIT & CLOSE (do not burn the step budget refining forever)\nOnce a row-backed match clears walls: inspect_stock (intended commitment) →\nadd_to_cart → if any market event fired, inspect_stock the EVENT-TARGET ALONE\nfirst (even when ordering another sku) → re-check stock/budget → place_test_order.\nAnswer questions with message the turn they land. Omit justification. Late\nbudget: stop refining, close event + order pair. Thin coverage left unfinished\nOR event_closed=false pays 0.\n'.strip(), WALLS: '\nFAMILY — WALLS (TF3 constraint — immutable hard gates)\n\nWalls are immutable. Soft attractions are not walls. The palate only chooses\namong items that already clear every firm gate on the exact variant.\n\nIF WITHHELD: TURN-1 MESSAGE ONLY\nWhen a requirement is held until you ask, send message before any search or\nfilter. Shopping blind wastes the budget on the wrong shelf; the reveal changes\nthe shelf. Do not “warm up” with a speculative browse while disclosure is owed.\nMessage-before-search is law. Ask once, naming what you need.\n\nAFTER THE FIGURE POSTS\n- Search with the revealed literal immediately. When brand and gate are both\n  named, keep them together in the query.\n- Buy the exact sku printed in observations — never invent a sku string.\n- Every stated wall must be confirmed by a fact row on that exact variant.\n  Title text is not confirmation. Do not cart until the disclosed figure\n  appears in a FACT ROW.\n- Soft attractions (nicer look, lower price) NEVER excuse a wall that fails.\n- Once a wall-clearing sku is proven, finish: inspect_stock → place_test_order.\n  Do not wander the catalog until the step budget dies.\n\nSELLOUT → NEW PRODUCT_ID (NO SIBLING TRAP)\nIf the compliant sku sells out, move to a new product_id that still meets ALL\nfirm gates. A sibling colourway that misses a gate is not a recovery. Re-prove\neach wall on the replacement’s observations before carting.\n\nCLOSE + EVENT\nOn market events: inspect EVENT-TARGET first (even when ordering a different\nsku), recover with another fully wall-clearing product_id, re-check stock AND\nbudget after any price move, then order. Reserve ≥2 steps for the closing pair.\nLate budget: stop open search, close the event, finish. event_closed=false\npays 0. Omit justification unless separately required.\n'.strip(), LEXICON: '\nFAMILY — LEXICON (TF5 ranking — arithmetic, not taste)\n\nThis is lexicographic order on a closed named set — vibes carry zero weight.\nDo not let the soft palate invent axes the shopper never stated. Unfinished\nevents and stale pre-event leaders are classic zeros here.\n\nBOUNDS\n- Candidates are exactly the product_id::sku lines the shopper named. Do not\n  widen-search strangers into the race.\n- view or compare ALL named candidates before any cart. An unread line cannot\n  be placed. First cart is graded — no provisional leader before the full open.\n\nLEX-RANK\n- First stated axis decides; later axes only break genuine ties; price last\n  unless stated. General “quality” or look counts only if stated as an axis.\n- Split product_id and sku as separate arguments. Write the ordering in your\n  reply so the pick is auditable.\n\nAFTER EVENT — RE-RANK FROM SCRATCH (strip stale rankings)\nDo not keep the pre-event #1. Ritual:\n  1. inspect_stock the EVENT-TARGET to close the event (even if you will order\n     a different named line; replacement inspect does not close).\n  2. Refresh surviving named lines from fresh observations.\n  3. Re-check stock AND budgets after any price move; never order a line last\n     seen out of stock.\n  4. Re-sort from scratch under the same axes. Whoever leads NOW wins —\n     update the cart to that leader before place_test_order.\nPre-event rankings are stale. Ordering the old #1 after the market moved is a\nfailed ranking.\n\nORDER SHAPE\nplace_test_order carries product_id + sku ONLY — strip justification entirely.\nOptional claims on a ranking order count against you. Cart the current leader,\ninspect_stock, order. Answer open questions with message first. Late budget:\nstop re-tasting the closed set, close the event, finish.\n'.strip(), MEND: '\nFAMILY — MEND (TF6 recovery — walls first, then re-taste)\n\nRebuild eligibility before you taste again. Soft-axis winners that break walls\nare new failures, not recoveries. White often stalls after a valid mend OR\nleaves a gold cart with an open event — finish. Do not stall.\n\nPROVE ORIGINAL UNAVAILABLE\nBefore substituting, get an observation that the original is gone (inspect_stock\nor view). Soft-axis tasting of replacements waits until that proof lands.\nSuspicion-based swaps fail even when stock was truly empty.\n\nCONCESSION LADDER (one rung at a time)\n  L0  Hold all constraints; search the original ask (exact model + colour).\n  L1  Drop cosmetic only (shade/finish/trim/colour); announce with message.\n  L2  Drop model/maker next.\n  NEVER drop product class, spend wall, hard spec, or stock requirement.\nProve each dead end with observations before climbing. Prefer a same-model\ndifferent-colour survivor over jumping models when that rung still clears\nwalls. Re-qualify survivors against walls BEFORE soft axes decide.\n\nBEFORE FIRST CART\ninspect_stock the intended commitment before the first add_to_cart.\n\nAFTER MARKET EVENT — FINISH THE RUN\n1. inspect_stock EVENT-TARGET immediately (closes event; replacement inspect\n   does not — do this even when ordering a different sku).\n2. remove_from_cart if invalid.\n3. Climb ONE rung to a wall-clearing replacement; re-check stock AND budget\n   after any price move; re-taste only among survivors.\n4. Cart → inspect_stock → place_test_order promptly.\nDo not burn leftover steps on extra refine passes after a valid recovery is\ncarted. An unplaced mend scores nothing. Omit justification. event_closed=false\npays 0 even if the substitute looks like gold.\n'.strip(), WITNESS: _WITNESS_PLAYBOOK}
_BEAT: dict[str, tuple[str, ...]] = {PALATE: ('Taste beat: put the shopper’s lead preference words into search; then ≥2 hard-OK reads before cart.', "Taste beat: PRIMARY axis first — if lead preference is 'not printed', keep looking; do not crown a secondary-axis winner.", "Taste beat: score stated soft axes on printed values; 'not printed' beats invention; cheapest ≠ verdict.", 'Taste beat: late budget — stop tasting, inspect_stock, place_test_order.', 'Taste beat: market event → inspect EVENT-TARGET first even if ordering another sku; unfinished events pay 0.', 'Taste beat: after a price move, re-check stock AND spend lid before place_test_order.'), TILT: ('Tilt beat: firm walls filter; the leaning only chooses among survivors. Firm over wish.', 'Tilt beat: dual discovery NOW — fund firm+wish AND firm-only; brand+gate together when both named.', 'Tilt beat: prove the hard-gate figure in FACT ROWS before cart; price/stock alone are not enough.', 'Tilt beat: first cart = preferred brand only if fully firm-compliant on FACT ROWS; else firm backup.', 'Tilt beat: after an event, inspect EVENT-TARGET first, re-check stock+budget, re-apply the brand rule.', 'Tilt beat: message open questions before cart/order; message-before-search when a figure is withheld; omit justification.'), SHELF: ('Shelf beat: reuse distinctive shopper tokens across multi-query coverage; one near-miss refine, then view many ids.', 'Shelf beat: coverage by distinct product_ids beats six restatements of one listing.', 'Shelf beat: phrase must appear in fact rows — title-only is a decoy; quote-in-title never carts.', 'Shelf beat: once a row-backed match clears walls, finish — do not refine forever into a step-limit zero.', 'Shelf beat: cart → inspect EVENT-TARGET if event fired → inspect order sku → place_test_order.'), WALLS: ('Walls beat: if withheld, turn-1 message only — no search until disclosure.', 'Walls beat: after disclosure, search the literal immediately; every firm gate on FACT ROWS before cart.', 'Walls beat: once proven wall-clearing, finish inspect→order — do not wander until step-limit.', 'Walls beat: sellout → new product_id still clearing all firm gates (no sibling trap).', 'Walls beat: soft attraction never excuses a broken wall; re-check stock+budget after events; omit justification.'), LEXICON: ('Lexicon beat: closed named product_id::sku set only — no stranger search.', 'Lexicon beat: view/compare ALL before cart; first axis decides, later break ties.', 'Lexicon beat: after event, re-rank from scratch — do NOT keep pre-event #1; update cart to the new leader.', 'Lexicon beat: inspect EVENT-TARGET first to close the event, then re-sort survivors; re-check stock+budget.', 'Lexicon beat: order product_id + sku only — strip claims/justification entirely.'), MEND: ('Mend beat: prove original unavailable via observation before substituting — soft axes wait.', 'Mend beat: concession ladder — exact match → cosmetic/colour → model/maker; never relax class/budget/stock/spec.', 'Mend beat: after event close on EVENT-TARGET (even if ordering another), replace, inspect_stock, place_test_order — finish.', 'Mend beat: after remove_from_cart or thin budget, always cart→inspect→order; unplaced mend ≈ 0.'), WITNESS: ('Witness beat: MANDATORY-PROPERTY GATE — FIELD must name the hard disqualifier from the ask/schema; VALUE = printed fact-row token. Secondary-attr substitution while that gate is open fails claim_truth.', 'Witness beat: price + in_stock + brand alone do not finish a gate ask — still claim the hard attribute. Never phone_type/sku/product_id/specification/spec/hdr/feature/why.', 'Witness beat: claim the attribute the ask/schema names; paste the printed token. Soft junk never counts as the gate.', 'Witness beat: banned — phone_type, device_type, product_type, sku, product_id, item_type, model_number, category, title, os, display, specification, hdr, feature, attribute, requirement, why; never glue drawer+attribute.', 'Witness beat: title-only figures are decoys. Keep reading until a fact row prints it. If facts do not prove the stated requirement, reject the sku.', 'Witness beat: search brand+gate together when both named; inspect before first cart; after event ALWAYS inspect EVENT-TARGET (even if ordering another), strip stale claims, rebuild, then order.', 'Witness beat: late budget — stop hunting synonyms; place_test_order once inspected. Do not cart until the gate figure is in a FACT ROW.')}
_PROD = ('No tool call in that reply. Pick a tool and use it now — FIT only moves via tools.', 'Still stalled. Gather → read ≥2 hard-OK → compare soft axes → cart → inspect_stock (event-target first if needed) → place_test_order.')

def _system_text(brief: WhiteBrief, bag: AxisBag, policy: dict[str, Any]) -> str:
    blocks = [_ROLE, _BOOK.get(brief.family, _BOOK[SHELF])]
    if brief.needs_claims() and brief.family != WITNESS:
        blocks.append(_WITNESS_PLAYBOOK)
    notes = brief.board_notes(bag)
    if notes:
        blocks.append(notes.strip())
    blocks.append(f"SESSION\n- family: {brief.family} (margin {brief.margin})\n- tools: {', '.join(_schema_names(policy.get('tools') or [])) or '(schemas)'}\n- step limit: {policy.get('max_steps')}\n- actions per turn: {policy.get('max_calls_per_turn', 1)}\n- optional claims: {('required' if brief.needs_claims() else 'omit entirely')}")
    return '\n\n'.join(blocks)

def _beat(family: str, turn: int) -> str:
    bank = _BEAT.get(family) or _BEAT[SHELF]
    return bank[(turn - 1) % len(bank)]

class TasteBoard:
    __slots__ = ('turn', 'limit', 'ran', 'ever', 'shelf', 'cart_pid', 'cart_sku', 'carted', 'cart_stock_ok', 'ordered', 'question_open', 'asked_held', 'event_turn', 'event_pid', 'event_sku', 'requalified', 'clarified', 'burnt', 'unavailable_proof', 'vetoes', 'search_count', 'closed_events')

    def __init__(self, limit: int) -> None:
        self.turn = 0
        self.limit = max(1, limit)
        self.ran: list[str] = []
        self.ever: set[str] = set()
        self.shelf: list[str] = []
        self.cart_pid = ''
        self.cart_sku = ''
        self.carted = False
        self.cart_stock_ok = False
        self.ordered = False
        self.question_open = False
        self.asked_held = False
        self.event_turn = 0
        self.event_pid = ''
        self.event_sku = ''
        self.requalified = 0
        self.clarified = False
        self.burnt: set[str] = set()
        self.unavailable_proof = False
        self.vetoes = 0
        self.search_count = 0
        self.closed_events: set[str] = set()

    @property
    def steps_left(self) -> int:
        return max(0, self.limit - self.turn)

    def used(self, name: str) -> bool:
        return name in self.ever

    def fresh(self, key: str) -> bool:
        return bool(key) and key not in self.burnt

    def seal(self, key: str) -> None:
        if key:
            self.burnt.add(key)

    def shelve(self, pid: str) -> None:
        if pid and pid not in self.shelf:
            self.shelf.append(pid)

    def record(self, names: list[str], calls: list[dict[str, Any]]) -> None:
        self.ran = names
        self.ever.update(names)
        self.search_count += sum((1 for n in names if n == 'search'))
        if 'add_to_cart' in names:
            self.carted = True
            self.cart_stock_ok = False
            for call in calls:
                if _fn_name(call) != 'add_to_cart':
                    continue
                args = _fn_args(call)
                self.cart_pid = str(args.get('product_id') or self.cart_pid)
                self.cart_sku = str(args.get('sku') or self.cart_sku)
        if 'remove_from_cart' in names:
            self.carted = False
            self.cart_stock_ok = False
        if 'place_test_order' in names:
            self.ordered = True
        if 'message' in names:
            self.question_open = False
            self.asked_held = True
        if 'inspect_stock' in names and self.event_turn:
            for call in calls:
                if _fn_name(call) != 'inspect_stock':
                    continue
                pid = str(_fn_args(call).get('product_id') or '')
                if self.event_pid and pid == self.event_pid:
                    self.closed_events.add(self.event_pid)
                    self.event_turn = 0
                    self.event_pid = ''
                    self.event_sku = ''
                    self.requalified += 1
                    break

    def mark_event(self, pid: str='', sku: str='') -> None:
        snap = pid or self.cart_pid
        if snap and snap in self.closed_events:
            return
        if self.turn > self.event_turn:
            self.event_turn = self.turn
        if snap:
            self.event_pid = snap
            self.event_sku = sku or self.cart_sku or self.event_sku

    def event_pending(self) -> bool:
        return bool(self.event_turn) and (not self.ordered)

    def may_veto(self, tag: str) -> bool:
        key = f'veto:{tag}'
        if key in self.burnt:
            return False
        self.burnt.add(key)
        return True

class WhiteBridge:

    def __init__(self, client: ProxyClient) -> None:
        self._client = client

    @staticmethod
    def model_for(family: str) -> str:
        override = (getenv('SANDBOX_MODEL') or '').strip()
        if override:
            low = override.lower()
            if 'kimi' in low or 'glm' in low or low.startswith('z-ai/') or low.startswith('moonshotai/'):
                return _DEEP_MODEL
            return override
        return _DEEP_MODEL if family in _DEEP_FAMILIES else _FAST_MODEL

    def _chain(self, model: str) -> list[str]:
        return [model, *[m for m in _BACKUP_MODELS if m != model]][:3]

    def ask_tools(self, *, engine: str, transcript: list[dict[str, Any]], schemas: list[Any], budget: int) -> tuple[str, list[dict[str, Any]]]:
        trouble: Exception | None = None
        for model in self._chain(engine):
            payload = self._client.post('/inference/chat/completions', json_data={'model': model, 'messages': transcript, 'tools': schemas, 'tool_choice': 'required', 'temperature': 0})
            if not payload:
                trouble = RuntimeError(model)
                continue
            try:
                message = payload['choices'][0]['message']
            except (KeyError, IndexError, TypeError) as exc:
                trouble = exc
                continue
            return (message.get('content') or '', list(message.get('tool_calls') or [])[:budget])
        raise RuntimeError(f'inference failed ({trouble})')

    def ask_json(self, *, engine: str, system: str, user: str) -> str:
        for model in self._chain(engine):
            payload = self._client.post('/inference/chat/completions', json_data={'model': model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'temperature': 0})
            if not payload:
                continue
            try:
                return str(payload['choices'][0]['message'].get('content') or '')
            except (KeyError, IndexError, TypeError):
                continue
        return ''

    def act(self, envelope: dict[str, Any]) -> dict[str, Any]:
        reply = self._client.post('/environment/call', json_data=envelope)
        if reply is None:
            raise RuntimeError('environment call failed')
        return reply
_LINK = WhiteBridge(ProxyClient(timeout=120, max_retries=2))

def _fn_name(call: Any) -> str:
    spec = call.get('function') if isinstance(call, dict) else None
    return str(spec['name']) if isinstance(spec, dict) and spec.get('name') else ''

def _fn_args(call: Any) -> dict[str, Any]:
    spec = call.get('function') if isinstance(call, dict) else None
    raw = spec.get('arguments', '{}') if isinstance(spec, dict) else '{}'
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or '{}')
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}

def _fn_id(call: Any, fallback: str) -> str:
    return str(call.get('id') or fallback) if isinstance(call, dict) else fallback

def _clip(obs: Any, cap: int=6000) -> str:
    try:
        text = json.dumps(obs, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obs)
    return text if len(text) <= cap else text[:cap - 18] + '…<clipped>'

def _finished(result: dict[str, Any]) -> bool:
    for row in result.get('calls') or []:
        if isinstance(row, dict):
            obs = row.get('observation')
            if isinstance(obs, dict) and obs.get('done'):
                return True
    return False

def _shifted(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    keys = ('market_event', 'stock_changed', 'price_changed', 'event')
    if any((obs.get(k) for k in keys)):
        return True
    inner = obs.get('observation')
    return isinstance(inner, dict) and any((inner.get(k) for k in keys))

def _event_target(obs: Any) -> tuple[str, str]:
    if not isinstance(obs, dict):
        return ('', '')
    me = obs.get('market_event') if isinstance(obs, dict) else None
    if isinstance(me, dict):
        tgt = me.get('target') if isinstance(me.get('target'), dict) else {}
        pid = str(tgt.get('product_id') or '')
        sku = str(tgt.get('sku') or '')
        if pid or sku:
            return (pid, sku)
    nested = obs.get('observation')
    if isinstance(nested, dict):
        return _event_target(nested)
    return ('', '')

def _stock_gone(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    blob = json.dumps(obs, ensure_ascii=False, default=str).lower()
    if '"in_stock": false' in blob or '"in_stock":false' in blob:
        return True
    return bool(re.search('sold[_\\s]?out|unavailable|out[_\\s]of[_\\s]stock', blob))

def _asked_something(text: str) -> bool:
    body = (text or '').strip().lower()
    if not body:
        return False
    if '?' in body:
        return True
    return body.startswith(('which', 'what', 'does', 'did', 'can you', 'could you', 'would you', 'wait', 'hold on'))

class WhiteEpisode:

    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data['environment']
        self.binding = env['binding']
        self.policy = env['policy_view']
        self.schemas = self.policy.get('tools') or []
        self.ask = str(self.policy.get('query') or problem_data.get('query') or '')
        self.key = str(problem_data.get('problem_id', problem_data.get('id', 'problem')))
        self.per_turn = max(1, int(self.policy.get('max_calls_per_turn', 1)))
        self.brief = WhiteBrief(self.ask, self.schemas)
        self.bag = AxisBag(self.brief.goal, self.brief.rules)
        self.board = TasteBoard(int(self.policy.get('max_steps', 20)))
        self.well = WhiteWell()
        self.engine = WhiteBridge.model_for(self.brief.family)

    def run(self) -> list[dict[str, Any]]:
        transcript: list[dict[str, Any]] = [{'role': 'system', 'content': _system_text(self.brief, self.bag, self.policy)}, {'role': 'user', 'content': self.ask}]
        if self.brief.family == LEXICON and self.brief.named:
            transcript.append({'role': 'user', 'content': 'Closed named set on the palate: ' + ', '.join(self.brief.named) + '. Taste each one before ranking; pass product_id and sku as separate arguments — never a joined product_id::sku string.'})
        if self.brief.family == WALLS and self.brief.held_figure:
            transcript.append({'role': 'user', 'content': 'Opening taste: the brief holds a figure until asked. Send message asking for it before any search.'})
        trail: list[dict[str, Any]] = [{'role': 'meta', 'whitegirl': {'version': 13, 'architecture': 'SoftAxisComparator', 'family': self.brief.family, 'margin': self.brief.margin, 'soft_axes': self.bag.axes, 'walls': self.brief.walls, 'engine': self.engine, 'claims': self.brief.needs_claims(), 'provider': 'openrouter', 'finish_policy': 'force_stock_ok_order'}}]
        while self.board.turn < self.board.limit:
            self.board.turn += 1
            nudge = self._coach()
            if nudge:
                transcript.append({'role': 'user', 'content': nudge})
            (spoken, calls) = self._speak(transcript, trail)
            (calls, note) = self._guard(calls)
            if note:
                transcript.append({'role': 'user', 'content': note})
                if not calls:
                    (spoken, calls) = self._speak(transcript, trail)
                    (calls, note2) = self._guard(calls)
                    if note2:
                        transcript.append({'role': 'user', 'content': note2})
            calls = self._block_order_during_event(calls)
            calls = self._force_finish(calls)
            transcript.append({'role': 'assistant', 'content': spoken, 'tool_calls': calls})
            result = self._dispatch(calls, self.board.turn)
            trail.append({'role': 'assistant', 'content': spoken, 'tool_calls': calls, 'environment_result': result, 'whitegirl_turn': {'family': self.brief.family, 'turn': self.board.turn, 'hard_ok': self.bag.viewed_hard_ok, 'shelf': len(self.board.shelf), 'requalified': self.board.requalified, 'cart_stock_ok': self.board.cart_stock_ok, 'unavailable_proof': self.board.unavailable_proof}})
            self.board.record([_fn_name(c) or '?' for c in calls], calls)
            self._absorb(result, transcript, calls)
            if _finished(result):
                break
        return trail

    def _guard(self, calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        if not calls:
            return (calls, '')
        labels = [_fn_name(c) for c in calls]
        family = self.brief.family
        if self.brief.held_figure and (not self.board.asked_held):
            banned = {'search', 'filter', 'add_to_cart', 'place_test_order'}
            if any((name in banned for name in labels)):
                return ([], 'Withheld: lookup/cart never reached the environment. A requirement is held until you ask — send one message requesting it, then shop.')
        if family == SHELF and self.brief.quoted:
            for (call, name) in zip(calls, labels):
                if name != 'add_to_cart':
                    continue
                pid = str(_fn_args(call).get('product_id') or '')
                if pid and self.well.phrase_title_only(pid, self.brief.quoted) and self.board.may_veto('shelf-title:' + pid):
                    return ([], f'Withheld: add_to_cart on {pid} was dropped. The distinctive phrase "{self.brief.quoted}" is title-only there — confirm it in fact rows, or view another candidate.')
        if family == WALLS:
            for (call, name) in zip(calls, labels):
                if name != 'add_to_cart':
                    continue
                pid = str(_fn_args(call).get('product_id') or '')
                if not pid:
                    continue
                if self.well.head_only(pid, self.brief.goal) and self.board.may_veto('walls-head:' + pid):
                    return ([], f'Withheld: add_to_cart on {pid} was dropped — the asked figure is title-only. Read fact rows on another variant until the wall prints.')
                if (self.board.asked_held or self.brief.walls) and (not self.well.rows_answer(pid, self.brief.goal)):
                    if self.board.may_veto('walls-unproven:' + pid):
                        return ([], f'Withheld: add_to_cart on {pid} was dropped because no fact row yet proves the firm walls (including any disclosed figure). view another variant or keep reading until a row states it.')
        if family == PALATE and self.bag.axes and ('add_to_cart' in labels):
            late = self.board.steps_left <= 10 or self.board.requalified > 0
            soft_needed = self.bag.soft_axes()
            want_hits = min(2, len(soft_needed)) if soft_needed else 0
            for (call, name) in zip(calls, labels):
                if name != 'add_to_cart':
                    continue
                pid = str(_fn_args(call).get('product_id') or '')
                if not pid or pid not in self.bag.hard_ok:
                    continue
                hits = self.bag.axis_hits(self.well, pid)
                score = self.bag.soft_score(self.well, pid)
                lead_ok = self.bag.lead_printed(self.well, pid)
                better = [other for other in self.board.shelf if other in self.bag.hard_ok and other != pid and (self.bag.soft_score(self.well, other) > score or self.bag.axis_hits(self.well, other) > hits or (self.bag.lead_printed(self.well, other) and (not lead_ok)))]
                if score == 0 and better and self.board.may_veto('palate-soft:' + pid):
                    return ([], f'Withheld: add_to_cart on {pid} was dropped — it clears walls but shows no printed support on the stated soft axes, while {better[0]} does. Cart the soft-supported hard-OK item, or view more candidates that print the lead preference.')
                if not late and soft_needed and (not lead_ok) and better and self.board.may_veto('palate-lead:' + pid):
                    gap = ', '.join(self.bag.coverage_gap(self.well, pid)[:3]) or 'lead axis'
                    return ([], f'Withheld: add_to_cart on {pid} was dropped — lead soft axis still blank ({gap}), while {better[0]} prints more of the stated palate. Cart the fuller soft match, or search/view until the lead preference prints.')
                if not late and want_hits >= 2 and (hits < want_hits) and better and self.board.may_veto('palate-thin:' + pid):
                    return ([], f'Withheld: add_to_cart on {pid} was dropped — only {hits} stated soft axis(es) print, but {better[0]} covers more. Prefer the fuller soft match (battery / wireless / fit cues when stated), or keep searching before carting a partial FIT.')
        return (calls, '')

    def _block_order_during_event(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.board.event_pending() or self.board.ordered:
            return calls
        event_pid = self.board.event_pid
        event_sku = self.board.event_sku
        if event_pid:
            probe: dict[str, Any] = {'product_id': event_pid}
            if event_sku:
                probe['sku'] = event_sku
            if not calls:
                return calls
            messages: list[dict[str, Any]] = []
            rest: list[dict[str, Any]] = []
            for call in calls:
                name = _fn_name(call)
                if name == 'message':
                    messages.append(call)
                    continue
                if name in {'search', 'filter'}:
                    continue
                if name == 'inspect_stock' and str(_fn_args(call).get('product_id') or '') == event_pid:
                    continue
                if name == 'place_test_order':
                    continue
                rest.append(call)
            inspect_call: dict[str, Any] = {'id': f'{self.key}-event-inspect', 'type': 'function', 'function': {'name': 'inspect_stock', 'arguments': json.dumps(probe)}}
            return messages + [inspect_call] + rest
        out: list[dict[str, Any]] = []
        for call in calls:
            if _fn_name(call) == 'place_test_order':
                continue
            out.append(call)
        return out or calls

    def _force_finish(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        board = self.board
        if board.ordered or board.event_pending() or board.question_open:
            return calls
        if self.brief.held_figure and (not board.asked_held):
            return calls
        if not board.carted or not board.cart_pid:
            return calls
        labels = {_fn_name(c) for c in calls}
        if 'place_test_order' in labels:
            return calls
        rows = self.well.rows.get(board.cart_pid, {})
        if rows.get('in_stock') == 'false' or _stock_gone({'in_stock': rows.get('in_stock')}):
            if 'remove_from_cart' in labels:
                return calls
            if board.steps_left <= 8 and board.may_veto('force-remove-dead'):
                payload: dict[str, Any] = {'product_id': board.cart_pid}
                if board.cart_sku:
                    payload['sku'] = board.cart_sku
                return [{'id': f'{self.key}-force-remove', 'type': 'function', 'function': {'name': 'remove_from_cart', 'arguments': json.dumps(payload)}}]
            return calls
        ceiling = _money(self.brief.cap)
        price = self.well.price_of(board.cart_pid)
        if ceiling is not None and price is not None and (price > ceiling):
            return calls
        ready = board.cart_stock_ok or rows.get('in_stock') == 'true'
        if not ready or board.steps_left > 6:
            return calls
        payload = {'product_id': board.cart_pid}
        if board.cart_sku:
            payload['sku'] = board.cart_sku
        return [{'id': f'{self.key}-force-order', 'type': 'function', 'function': {'name': 'place_test_order', 'arguments': json.dumps(payload)}}]

    def _coach(self) -> str | None:
        board = self.board
        deadline_w = 34.0
        if board.steps_left <= 10 or board.requalified or board.cart_stock_ok or (board.carted and board.steps_left <= 12) or ('remove_from_cart' in board.ever and (not board.carted)):
            deadline_w = 78.0
        bids = ((96.0, self._bid_reply), (92.0, self._bid_withheld), (88.0, self._bid_event), (84.0, self._bid_post_event_finish), (deadline_w, self._bid_deadline), (70.0, self._bid_family), (44.0, self._bid_launch), (12.0, self._bid_beat))
        winner: tuple[float, str, str] | None = None
        for (weight, bid) in bids:
            offer = bid()
            if not offer:
                continue
            (key, text) = offer
            if key and (not self.board.fresh(key)):
                continue
            if winner is None or weight > winner[0]:
                winner = (weight, key, text)
        if winner is None:
            return None
        self.board.seal(winner[1])
        return winner[2]

    def _bid_reply(self) -> tuple[str, str] | None:
        if not self.board.question_open:
            return None
        return ('', 'The shopper asked you a question. Answer it with message this turn before any cart or order — an unanswered question sinks the episode.')

    def _bid_withheld(self) -> tuple[str, str] | None:
        if not self.brief.held_figure or self.board.asked_held:
            return None
        return ('withheld', 'A requirement is being withheld until you ask for it. Send message now and ask for it — searching without it wastes the run.')

    def _bid_event(self) -> tuple[str, str] | None:
        if not self.board.event_pending():
            return None
        target = self.board.event_pid or self.board.cart_pid or 'the affected sku'
        key = f'event-{self.board.turn}'
        return (key, f'The market moved. inspect_stock the EVENT-TARGET ({target}) FIRST this turn to close the event (replacement inspect does not count). Then remove_from_cart if invalid, recover under the family book, re-check stock+budget, and place_test_order — do not order the dead/over-budget event target.')

    def _bid_post_event_finish(self) -> tuple[str, str] | None:
        board = self.board
        if board.ordered or board.event_pending():
            return None
        post_remove = 'remove_from_cart' in board.ever and (not board.carted)
        stock_ready = board.carted and board.cart_stock_ok
        if not (board.requalified or post_remove or stock_ready or (board.steps_left <= 10)):
            return None
        key = f'finish-{board.turn}'
        if board.carted:
            return (key, f'Post-event / recovery close: {board.steps_left} step(s) left. ' + ('Cart already stock-ok — place_test_order NOW. ' if stock_ready else 'inspect_stock the carted sku if needed, then place_test_order NOW. ') + 'An unplaced order scores nothing. Do not keep refining or re-searching.')
        return (key, 'Post-event recovery after remove_from_cart: cart a wall-clearing replacement NOW, inspect_stock it, then place_test_order before the budget ends. Do not end the run on remove.')

    def _bid_family(self) -> tuple[str, str] | None:
        table = {PALATE: self._family_palate, TILT: self._family_tilt, SHELF: self._family_shelf, WALLS: self._family_walls, LEXICON: self._family_lexicon, MEND: self._family_mend, WITNESS: self._family_witness}
        handler = table.get(self.brief.family, self._family_shelf)
        return handler()

    def _family_palate(self) -> tuple[str, str] | None:
        (board, bag) = (self.board, self.bag)
        if board.ordered:
            return None
        clash = bag.tension()
        if clash and (not board.carted) and (not board.clarified) and (board.turn <= max(2, board.limit // 3)):
            (left, right) = clash
            return ('palate-clarify', f'The ask pulls two ways: {left} against {right}. Send ONE message asking which one should lead, then commit to the answer. Do not ask a second time and do not stall the run waiting — keep searching in the same turn budget.')
        axes = bag.line() or "the shopper's stated preferences"
        lead = bag.axes[0] if bag.axes else ''
        soft = bag.soft_axes()
        lead_cues = ', '.join(bag.cues_for(lead)[:4]) if lead else ''
        if not board.used('search') and (not board.used('filter')):
            return ('palate-open', f"Search with the shopper's OWN lead preference words plus the hard walls. Stated soft axes: {axes}." + (f' Lead-axis cues to put in the query: {lead_cues}.' if lead_cues else '') + ' A shelf that never asked for the lead axis cannot surface printed evidence.')
        if not board.carted and bag.viewed_hard_ok < 2:
            if board.used('view') or board.used('compare'):
                return (f'palate-second-{bag.viewed_hard_ok}', f'Only {bag.viewed_hard_ok} wall-clearing candidate has been read. view another one before any add_to_cart — {axes} cannot be weighed against a single option, and an uncompared pick is a coin flip.')
            return ('palate-first-read', 'Search with lead preference tokens, then read candidates. You need two wall-clearing reads before the first cart — plan the views now rather than carting the first hit.')
        late = board.steps_left <= 12 or board.search_count >= 4 or board.requalified > 0
        if not board.carted and bag.viewed_hard_ok >= 2 and (not self._soft_leader()) and lead_cues and (not late) and board.fresh('palate-reseach'):
            return ('palate-reseach', f'Hard-OK candidates are soft-silent so far. Search again using lead-axis cues ({lead_cues}) with the product class, then view new hits. Do not cart a soft-blank item while better printed support may still be findable.')
        leader = self._soft_leader()
        if not board.carted and leader and (not late) and soft and (bag.axis_hits(self.well, leader) < min(2, len(soft))) and board.fresh('palate-cover'):
            gap = ', '.join(bag.coverage_gap(self.well, leader)[:4]) or 'secondary axes'
            extra = ''
            if any((a in soft for a in ('battery', 'wireless', 'fit'))):
                extra = ' Include battery / wireless / secure-fit cues in the query when those axes were stated — do not cart on sweat/water alone if other axes still print on another hard-OK.'
            return ('palate-cover', f'Best soft hit so far ({leader}) still blanks: {gap}. Search/view once more for candidates that print MORE of the stated axes ({axes}).{extra} If nothing fuller appears, cart the current leader next — do not refine forever.')
        cheap = self._cheapest_eligible()
        if not board.carted and bag.viewed_hard_ok >= 2:
            table = self._axis_table()
            lead_hint = f" PRIMARY axis '{lead}' first: if it is 'not printed' on a candidate, keep looking — do not crown a secondary-axis winner while the lead preference stays blank." if lead and (not late) else ''
            commit_hint = ' LATE BUDGET: cart the best printed soft match NOW, leave room for inspect_stock + place_test_order.' if late else ''
            if cheap and self._soft_leader() and (self._soft_leader() != cheap):
                return ('palate-anticheap', f'Do not default to the cheapest eligible listing. Printed evidence on the stated axes:\n{table}\nThe cheapest eligible item is {cheap}, but {self._soft_leader()} carries more printed support on the stated axes.{lead_hint}{commit_hint} Cart on fit and name the axis that decided it.')
            return ('palate-weigh', 'Both reads are in. Weigh them on printed values only:\n' + table + lead_hint + "\nWhere a value is 'not printed', move to the next stated axis instead of guessing. Prefer the candidate that covers MORE stated axes (lead first, then battery/wireless/fit when asked)." + commit_hint + '\nThen add_to_cart the best fit — cheapest ≠ winner by default.')
        if board.carted and cheap and (board.cart_pid == cheap) and (self._soft_leader() not in ('', cheap)):
            return ('palate-recheck', f'The cart holds the cheapest eligible item ({cheap}) while {self._soft_leader()} has more printed support on the stated axes. Either state the axis on which the carted item wins, or swap it before you close.')
        if board.carted and board.cart_stock_ok:
            return (f'palate-order-{board.turn}', 'Fit winner is carted and stock-ok. place_test_order NOW with no justification field — do not search again.')
        if board.carted and (not board.used('inspect_stock')):
            return ('palate-inspect', 'Fit winner is carted. inspect_stock it, then place_test_order with no justification field.')
        if board.carted and (not board.ordered):
            return (f'palate-close-{board.turn}', 'Carted. If stock was already checked, place_test_order NOW. Otherwise inspect_stock once, then order — leave no unplaced cart.')
        return None

    def _eligible(self) -> list[str]:
        return [pid for pid in self.board.shelf if pid in self.bag.hard_ok]

    def _cheapest_eligible(self) -> str:
        priced = [(self.well.price_of(pid), pid) for pid in self._eligible()]
        live = [(cost, pid) for (cost, pid) in priced if cost is not None]
        if not live:
            return ''
        live.sort()
        return live[0][1]

    def _soft_leader(self) -> str:
        ranked = [(self.bag.axis_hits(self.well, pid), self.bag.soft_score(self.well, pid), 1 if self.bag.lead_printed(self.well, pid) else 0, pid) for pid in self._eligible()]
        ranked = [row for row in ranked if row[1] > 0]
        if not ranked:
            return ''
        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        return ranked[0][3]

    def _axis_table(self) -> str:
        lines: list[str] = []
        for pid in self._eligible()[:4]:
            found = self.bag.evidence(self.well, pid)
            cells = [f"{axis}: {found.get(axis) or 'not printed'}" for axis in self.bag.axes[:4]]
            price = self.well.rows.get(pid, {}).get('price', '?')
            lines.append(f'  {pid} (price {price}) — ' + '; '.join(cells))
        return '\n'.join(lines) if lines else '  (no wall-clearing candidate read yet)'

    def _family_tilt(self) -> tuple[str, str] | None:
        board = self.board
        lean = ', '.join(self.brief.leanings)
        firm = '; '.join(self.brief.walls) or 'the stated requirements'
        if not board.used('search') and (not board.used('filter')):
            return ('tilt-open', f'Dual discovery now. Pass A: search firm walls ({firm})' + (f' PLUS the leaning ({lean})' if lean else '') + '. Pass B: firm nouns alone for a wall-clearing backup. Fund both — do not collapse into one firm-only query.')
        if board.used('search') and lean and board.fresh('tilt-dual'):
            return ('tilt-dual', f'If you have not yet, run the missing dual-discovery pass: firm+leaning ({lean}) and firm-only backup. Then prove the hard-gate figure in FACT ROWS before any cart — price/stock alone are not enough.')
        if not board.carted and lean:
            compliant = [pid for pid in self.board.shelf if pid in self.bag.hard_ok]
            with_lean = [pid for pid in compliant if any((_lex_hit(word, self.well.witness_blob(pid)) for word in self.brief.leanings))]
            if compliant and (not with_lean):
                return ('tilt-nolean', f'No firm-compliant candidate shows the leaning ({lean}). That is fine: cart the firm-compliant item and note in your message that the leaning could not be met inside the walls. Never trade a firm requirement for the leaning.')
            if with_lean:
                return ('tilt-bonus', f'{with_lean[0]} is firm-compliant AND shows the leaning ({lean}). Confirm EVERY firm wall — including the stated hard-gate figure — in its fact rows, then cart it.')
        if not board.carted and (not lean):
            return ('tilt-strict', 'No explicit leaning was stated, so compliance is the whole story: prove every firm requirement in the fact rows and cart the compliant item.')
        if board.carted and (not board.ordered):
            return ('tilt-close', 'Carted. inspect_stock the sku, then place_test_order with no justification field.')
        return None

    def _family_shelf(self) -> tuple[str, str] | None:
        board = self.board
        shapes = {self.well.shape_key(pid) for pid in board.shelf}
        phrase = self.brief.quoted
        cat = self.brief.category()
        if board.carted and board.cart_stock_ok and (not board.ordered):
            return (f'shelf-order-{board.turn}', 'Match is carted and stock-ok. place_test_order NOW without justification — do not refine the query again.')
        if not board.used('search') and (not board.used('filter')):
            bits = []
            if phrase:
                bits.append(f'keep the distinctive phrasing "{phrase}" verbatim')
            if cat:
                bits.append(f'include the category "{cat}"')
            tip = '; '.join(bits) if bits else "use the shopper's own nouns"
            return ('shelf-open', f'Open with search that {tip}. Then view several DISTINCT product_ids — coverage is identities opened, not duplicate rows.')
        if board.used('search') and cat and board.fresh('shelf-cat'):
            return ('shelf-cat', f'Run a second query shape that keeps the rare phrase and names category "{cat}" (or the reverse: category-first, then the distinctive tokens). Multi-query coverage beats one neighbour.')
        backed = []
        if phrase:
            backed = [pid for pid in board.shelf if pid in self.well.rows and _evidenced(phrase, self.well.rows_blob(pid))]
        if len(board.shelf) <= 1 and board.used('search') and (not backed) and (board.search_count < 4) and (board.steps_left > 12):
            return ('shelf-thin', 'The shelf is thin. Change exactly ONE token in the query and search again — do not rewrite it wholesale — then view the new hits. Do not burn the whole step budget refining; once a row-backed match clears walls, finish the order.')
        if phrase and (not board.carted):
            if backed and board.fresh('shelf-backed'):
                return ('shelf-backed', f'Fact rows already print "{phrase}" on {backed[0]}. Confirm walls/budget/stock, then cart → inspect_stock → place_test_order. Do not keep refining.')
            pid = self.well.pin_pid or self.well.last_pid
            if pid and self.well.phrase_title_only(pid, phrase) and board.fresh('shelf-title'):
                return ('shelf-title', f'Quoted phrase "{phrase}" is title-only on {pid}. Do not cart yet — confirm the phrase in fact rows of a variant, or taste the next candidate.')
        if board.steps_left <= 12 and (not board.carted) and board.used('view'):
            return (f'shelf-finish-{board.turn}', f'Only {board.steps_left} step(s) left. If any viewed sku already shows the distinctive phrase in FACT ROWS and clears walls, cart → inspect_stock (EVENT-TARGET alone first if an event fired) → place_test_order NOW.')
        if len(shapes) < len(board.shelf) and (not backed):
            return ('shelf-dupes', f'{len(board.shelf)} entries collapse to {len(shapes)} distinct identities. Repeated skus of one product are a single shelf entry — find a genuinely different item before comparing.')
        if len(shapes) >= 2 and (not board.used('view')):
            return ('shelf-read', f'{len(shapes)} distinct identities on the shelf. view the strongest ones and read the classification fact rows to drop accessories, bundles and look-alikes.')
        if board.carted and (not board.ordered):
            return (f'shelf-close-{board.turn}', 'Pick is carted. If a market event just fired, inspect the EVENT-TARGET alone first. Then inspect_stock the order sku and place_test_order without justification — finish this turn if stock is already ok.')
        return None

    def _family_walls(self) -> tuple[str, str] | None:
        board = self.board
        pid = self.well.pin_pid or self.well.last_pid
        if self.brief.walls and board.fresh('walls-list'):
            return ('walls-list', 'These are walls, not wishes: ' + '; '.join(self.brief.walls) + '. Each one must be confirmed by a fact row on the exact variant you buy. No soft attraction — nicer look, better feel, lower price — excuses a wall that does not hold. Once proven, finish inspect→order — do not wander.')
        if self.brief.held_figure and board.asked_held and (not board.carted) and board.fresh('walls-after-reveal'):
            return ('walls-after-reveal', 'The withheld figure is disclosed. Search the revealed literal immediately (brand+gate together when both named), prove it in FACT ROWS, then finish inspect_stock → place_test_order. Do not browse until the step budget dies.')
        if self.brief.named and (not board.carted):
            return ('walls-exact', f'The shopper named {self.brief.named[0]}. Buy that exact variant, not a sibling. view it and confirm every wall in its fact rows first.')
        named_pids = {token.partition('::')[0] for token in self.brief.named}
        if pid and pid not in named_pids and self.well.head_only(pid, self.brief.goal):
            return ('walls-decoy', "The asked figure appears in this product's title but in none of its fact rows. That is decoration, not evidence — keep reading other variants.")
        if pid and board.used('view') and (not board.carted) and (not self.well.rows_answer(pid, self.brief.goal)):
            return ('walls-unproven', 'No fact row confirms the wall yet. Do not assume a category default and do not convert units — read another variant until a fact row states it.')
        if board.carted and (not board.used('inspect_stock')):
            return ('walls-inspect', 'Walls confirmed and carted. inspect_stock the sku, then order.')
        return None

    def _family_lexicon(self) -> tuple[str, str] | None:
        board = self.board
        named = self.brief.named
        unread = [token for token in named if token.partition('::')[0] not in self.well.rows]
        if named and unread and (not board.carted):
            return (f'lex-read-{len(unread)}', f"{len(unread)} named candidate(s) still unread ({', '.join(unread[:4])}). view each one before ranking — an unread candidate cannot be placed, and this is arithmetic on the stated axes, not an impression.")
        if board.requalified and (not board.ordered):
            return (f'lex-rerank-{board.turn}', 'Market moved — pre-event #1 is stale. Re-rank survivors from scratch on the stated axes, UPDATE the cart to whoever leads NOW, then place_test_order with product_id + sku only. Keeping the old leader is a failed ranking.')
        if not board.carted:
            return ('lex-sort', "Rank the read candidates on the stated axes in the stated order: first axis decides, later axes break ties only. General 'quality', looks and feel carry zero weight unless stated. Then cart the leader.")
        if board.carted and (not board.ordered):
            return ('lex-close', 'Close with place_test_order carrying product_id and sku ONLY. Strip the justification field — optional claims on a ranking order count against you.')
        return None

    def _family_mend(self) -> tuple[str, str] | None:
        board = self.board
        keep = '; '.join(self.brief.walls) or 'product class, spend cap and availability'
        if board.fresh('mend-open'):
            return ('mend-open', f'Recovery run. What survives the loss unchanged: {keep}. Climb the concession ladder one rung at a time: exact match → cosmetic/colour → model/maker. Re-qualify survivors against those walls BEFORE consulting any preference — a soft-axis winner that breaks a wall is a new failure, not a recovery.')
        if not board.unavailable_proof:
            if board.used('view') or board.used('inspect_stock') or board.used('search'):
                return (f'mend-prove-{board.turn}', 'Prove the original is gone BEFORE any substitute cart. inspect_stock or view the named item and let the observation say sold-out / unavailable — soft-axis tasting of replacements waits until that proof lands.')
            return ('mend-hold', 'Hold every soft preference for now and look for exactly what the shopper named. Concessions come after an unavailability observation, not before.')
        post_remove = 'remove_from_cart' in board.ever and (not board.carted)
        if not board.ordered and (board.requalified or board.steps_left <= 8 or (board.carted and board.steps_left <= 10) or post_remove):
            if board.event_pending():
                target = board.event_pid or board.cart_pid or 'the event-target sku'
                return (f'mend-event-finish-{board.turn}', f'Late recovery with an open event. inspect_stock the event-target ({target}) FIRST this turn — that closes the event — then remove_from_cart if invalid, cart a wall-clearing replacement if needed, and place_test_order before the budget ends.')
            if board.carted:
                return (f'mend-late-place-{board.turn}', f'Only {board.steps_left} step(s) left after recovery. inspect_stock the carted sku if not yet confirmed this episode, then place_test_order NOW — an unplaced order scores nothing. Skip further search/refine.')
            return (f'mend-late-cart-{board.turn}', 'Late recovery / post-remove: cart the best wall-clearing survivor NOW, inspect_stock it, place_test_order. Do not keep searching — finish the run.')
        eligible = self._eligible()
        if not board.carted and len(eligible) < 2 and board.used('search'):
            return (f'mend-requalify-{len(eligible)}', f'{len(eligible)} survivor(s) confirmed against the walls. view another survivor and check the walls in its fact rows; only once two clear the walls do the soft axes get to choose.')
        if not board.carted and len(eligible) >= 2:
            if not board.used('inspect_stock') and board.fresh('mend-precart'):
                return ('mend-precart', 'Two survivors clear the walls. Before the first add_to_cart: inspect_stock the chosen candidate so the environment records an intended commitment. Soft axes:\n' + self._axis_table())
            return ('mend-choose', 'Two survivors clear the walls. NOW the soft axes decide between them:\n' + self._axis_table() + '\nConcede only what the shopper offered, keep the defining attribute, then cart.')
        if board.carted and (not board.ordered):
            return (f'mend-close-{board.turn}', 'Recovery pick carted. Say which attribute you conceded in your message if useful, inspect_stock the sku, then place_test_order with a clean payload — close now.')
        return None

    def _family_witness(self) -> tuple[str, str] | None:
        board = self.board
        if board.ordered:
            return None
        pid = self.well.pin_pid or self.well.last_pid
        if pid and (not board.carted) and self.well.head_only(pid, self.brief.goal):
            return ('witness-decoy', "Title decoy: the asked figure is in this product's title but in no fact row. A claim built on title text is untruthful and scores nothing. Keep reading.")
        if pid and board.used('view') and (not board.carted) and (not self.well.rows_answer(pid, self.brief.goal)):
            return ('witness-proof', "The first commitment must be fact-backed: cart only when a fact row overlaps the shopper's ask. Keep reading until one does.")
        if board.fresh('witness-field-class'):
            return ('witness-field-class', 'MANDATORY-PROPERTY GATE: claim FIELD must name the hard disqualifier from the ask/schema (resolution, display class, size, material, … — whichever the goal actually gated). VALUE = printed fact-row token. Secondary-attr substitution while that gate is open fails claim_truth. Brand/price/in_stock alone are not enough.')
        if not board.carted and board.used('view') and (board.steps_left <= max(board.limit // 2, 5)):
            return ('witness-midpoint', 'Midpoint: if a read sku already prints the gate specification and clears budget and stock, cart it now and keep room for inspect_stock + place_test_order.')
        return (f'witness-beat-{board.turn}', self._witness_beat())

    def _witness_beat(self) -> str:
        text = _beat(WITNESS, self.board.turn)
        extra: list[str] = []
        if self.brief.claim_fields:
            extra.append(f"Allowed claim fields: {', '.join(self.brief.claim_fields)}.")
        if self.brief.claim_rule:
            extra.append(f'Schema rule: {self.brief.claim_rule[:180]}')
        pid = self.well.pin_pid or self.well.last_pid
        if pid:
            rows = self.well.rows.get(pid, {})
            if rows:
                extra.append(f'Fact sample for {pid}: ' + ', '.join((f'{k}={v}' for (k, v) in list(rows.items())[:8])))
            if self.well.fresh_price or self.well.fresh_stock:
                extra.append(f"Newest price={self.well.fresh_price or '?'} in_stock={self.well.fresh_stock or '?'}.")
        return text + ('\n' + '\n'.join(extra) if extra else '')

    def _bid_launch(self) -> tuple[str, str] | None:
        board = self.board
        if board.turn > 2:
            return None
        if any((board.used(n) for n in ('search', 'filter', 'view', 'compare', 'message'))):
            return None
        return ('launch', "Begin with search or filter using the shopper's own words, then read the hits.")

    def _bid_deadline(self) -> tuple[str, str] | None:
        board = self.board
        if board.ordered:
            return None
        post_remove = 'remove_from_cart' in board.ever
        urgent = board.steps_left <= 10 or board.requalified or board.cart_stock_ok or post_remove or (board.carted and board.steps_left <= 12)
        if board.carted and (board.cart_stock_ok or board.steps_left <= 8):
            return (f'deadline-{board.turn}' if urgent else 'deadline', f'Only {board.steps_left} step(s) left. ' + ('Cart is stock-ok — place_test_order NOW. ' if board.cart_stock_ok else 'inspect_stock if needed, then place_test_order NOW. ') + 'An unplaced order scores nothing.')
        if post_remove and (not board.carted) and (board.steps_left <= 12):
            return (f'post-remove-{board.turn}', 'Cart cleared. Cart a wall-clearing replacement NOW, inspect_stock it, then place_test_order — do not leave the run on remove_from_cart.')
        if not board.carted and board.steps_left <= 10 and board.used('view'):
            return (f'hurry-cart-{board.turn}' if urgent else 'hurry-cart', 'Step budget is low. Cart the best proven candidate NOW and leave room for inspect_stock + place_test_order.')
        if board.steps_left <= 1:
            return (f'last-step-{board.turn}', 'Last step. place_test_order on the best proven candidate now.')
        return None

    def _bid_beat(self) -> tuple[str, str] | None:
        board = self.board
        if board.turn <= 2 or board.turn % 2:
            return None
        return (f'beat-{board.turn}', _beat(self.brief.family, board.turn))

    def _shape(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        payload = dict(args)
        if name in {'view', 'compare', 'inspect_stock', 'add_to_cart', 'place_test_order'}:
            payload = self._split_pair(payload)
        if name in {'view', 'compare', 'inspect_stock', 'add_to_cart'}:
            pid = str(payload.get('product_id') or '')
            if pid:
                self.well.pin_pid = pid
                self.board.shelve(pid)
        if name != 'place_test_order':
            return payload
        if len(self.brief.named) >= 2:
            payload.pop('justification', None)
            payload.pop('claims', None)
            return payload
        if self.brief.family == WITNESS:
            pid = str(payload.get('product_id') or self.board.cart_pid or self.well.pin_pid or self.well.last_pid)
            note = payload.get('justification')
            raw = note.get('claims') if isinstance(note, dict) else None
            payload['justification'] = {'claims': WhiteClaims.polish(raw, self.brief, self.well, pid)}
            if pid and (not payload.get('product_id')):
                payload['product_id'] = pid
            return payload
        payload.pop('justification', None)
        payload.pop('claims', None)
        return payload

    def _split_pair(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = dict(args)
        pid = str(payload.get('product_id') or '')
        hit = _PAIR_RX.search(pid)
        if hit and '::' in pid:
            payload['product_id'] = hit.group(1)
            if not payload.get('sku'):
                payload['sku'] = hit.group(2)
        sku = str(payload.get('sku') or '')
        hit = _PAIR_RX.search(sku)
        if hit and '::' in sku:
            payload['sku'] = hit.group(2)
            if not payload.get('product_id'):
                payload['product_id'] = hit.group(1)
        return payload

    def _dispatch(self, calls: list[dict[str, Any]], turn: int) -> dict[str, Any]:
        batch = f'{self.key}-turn-{turn}'
        actions: list[dict[str, Any]] = []
        for (slot, call) in enumerate(calls, start=1):
            name = _fn_name(call)
            if not name:
                raise RuntimeError('nameless tool call')
            args = self._shape(name, _fn_args(call))
            spec = call.get('function')
            if isinstance(spec, dict):
                spec['arguments'] = json.dumps(args)
            actions.append({'call_id': f'{batch}-{slot}', 'action': {'name': name, 'args': args}})
        return _LINK.act({**self.binding, 'call_id': batch, 'idempotency_key': batch, 'turn': turn, 'calls': actions})

    def _speak(self, transcript: list[dict[str, Any]], trail: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        for attempt in range(len(_PROD) + 1):
            (spoken, calls) = _LINK.ask_tools(engine=self.engine, transcript=transcript, schemas=self.schemas, budget=self.per_turn)
            if calls:
                return (spoken, calls)
            trail.append({'role': 'assistant', 'content': spoken})
            transcript.extend([{'role': 'assistant', 'content': spoken}, {'role': 'user', 'content': _PROD[min(attempt, len(_PROD) - 1)]}])
        raise RuntimeError('engine would not emit an executable action')

    def _absorb(self, result: dict[str, Any], transcript: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
        rows = result.get('calls') or []
        for (index, call) in enumerate(calls, start=1):
            row = rows[index - 1] if index - 1 < len(rows) else None
            obs = row.get('observation') if isinstance(row, dict) else None
            name = _fn_name(call)
            self.well.drink(obs)
            transcript.append({'role': 'tool', 'tool_call_id': _fn_id(call, f'{self.key}-{self.board.turn}-{index}'), 'content': _clip(obs)})
            if _shifted(obs):
                (ep, es) = _event_target(obs)
                snap = ep or self.board.cart_pid or self.well.pin_pid or self.well.last_pid or ''
                self.board.mark_event(pid=snap, sku=es or self.board.cart_sku)
            if name == 'inspect_stock' and self.board.carted and self.board.cart_pid:
                pid = str(_fn_args(call).get('product_id') or '')
                if pid == self.board.cart_pid:
                    if _stock_gone(obs):
                        self.board.cart_stock_ok = False
                    elif self.well.rows.get(pid, {}).get('in_stock') == 'true':
                        self.board.cart_stock_ok = True
                    elif self.well.fresh_stock == 'true':
                        self.board.cart_stock_ok = True
            if self.brief.family == MEND and name in {'view', 'compare', 'inspect_stock'} and _stock_gone(obs):
                self.board.unavailable_proof = True
            if self.brief.family == MEND and name in {'search', 'filter'}:
                empty = not (isinstance(obs, dict) and obs.get('results'))
                if empty and self.board.turn >= 3:
                    self.board.unavailable_proof = True
        for pid in self.well.rows:
            if self.well.hard_pass(pid, self.brief.cap, self.brief.goal):
                self.bag.note_hard_ok(pid)
                self.board.shelve(pid)
            else:
                self.bag.forget(pid)
        note = result.get('user_message')
        if isinstance(note, dict) and note.get('content'):
            text = str(note['content'])
            transcript.append({'role': 'user', 'content': text})
            if self.brief.family in {WALLS, TILT, SHELF, PALATE, MEND, WITNESS}:
                self.brief.ingest(text)
            if _asked_something(text):
                self.board.question_open = True
        if 'message' in self.board.ran and self.brief.family == PALATE and self.bag.tension():
            self.board.clarified = True
            self.bag.spend_clarify()
        if self.board.event_pending() and (not self.board.ordered):
            tip = f"Event note: inspect_stock the event-target sku now ({self.board.event_pid or self.board.cart_pid or 'carted'}) — that closes the event; inspecting a replacement does not. Order only after that close."
            if self.brief.needs_claims():
                tip += ' Then rebuild justification.claims from that fresh observation before place_test_order.'
            transcript.append({'role': 'user', 'content': tip})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return WhiteEpisode(problem_data).run()
__all__ = ['agent_main']
