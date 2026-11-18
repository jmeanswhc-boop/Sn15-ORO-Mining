from __future__ import annotations
import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_ROSTER = {'chutes': ('deepseek-ai/DeepSeek-V3.2-TEE', 'Qwen/Qwen3-32B-TEE'), 'openrouter': ('deepseek/deepseek-v3.2', 'qwen/qwen3-32b')}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_SECONDS_PER_PROBLEM = 270.0
_OBSERVATION_CHARS = 7000
_REPROMPTS = 2
_EPS = 1e-09

class Spec:
    INCH = re.compile('(\\d+(?:\\.\\d+)?)\\s*[-–]?\\s*(?:(?:inch(?:es)?|in)\\b|\'{2}|[\\"“”])', re.I)
    HZ = re.compile('(\\d+(?:\\.\\d+)?)\\s*hz', re.I)
    GB = re.compile('(\\d+(?:\\.\\d+)?)\\s*(tb|gb)\\b', re.I)
    WXH = re.compile('(\\d{3,5})\\s*[x×]\\s*(\\d{3,5})')
    PAIR = re.compile('(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*(?:g(?:b)?)?\\s*\\+\\s*(\\d+(?:\\.\\d+)?)\\s*g(?:b)?\\b', re.I)
    PAIR_HINT = re.compile('(?<!\\d)\\d+(?:\\.\\d+)?\\s*(?:g(?:b)?)?\\s*\\+\\s*\\d+(?:\\.\\d+)?(?:\\s*g(?:b)?)?(?!\\d)', re.I)
    MEMORY = re.compile('(\\d+(?:\\.\\d+)?)\\s*gb\\s*(?:ram|memory|unified memory)\\b', re.I)
    QUARTER_HD = re.compile('\\bqHD\\b')
    CLASS_WORDS = re.compile('\\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\\b', re.I)
    FIELDS = ('screen_size', 'refresh_rate', 'resolution', 'ram')
    LABEL = {'screen_size': 'size', 'resolution': 'display', 'refresh_rate': 'refresh_rate', 'ram': 'storage'}
    UNIT = {'screen_size': 'in', 'refresh_rate': 'hz', 'ram': 'gb'}

    @staticmethod
    def inches(text: str) -> set:
        return {'%gin' % float(m.group(1)) for m in Spec.INCH.finditer(text or '')}

    @staticmethod
    def hertz(text: str) -> set:
        return {'%ghz' % float(m.group(1)) for m in Spec.HZ.finditer(text or '')}

    @staticmethod
    def gigabytes(text: str) -> set:
        return {'%ggb' % (float(m.group(1)) * (1024 if m.group(2).lower() == 'tb' else 1)) for m in Spec.GB.finditer(text or '')}

    @staticmethod
    def display_class(text: str) -> set:
        text = text or ''
        if Spec.QUARTER_HD.search(text):
            return set()
        low = text.lower()
        out = set()
        for m in Spec.WXH.finditer(low):
            long_side, short_side = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
            if long_side >= 3840 and short_side >= 2160:
                out.add('4k')
            elif long_side >= 2560 and short_side >= 1440:
                out.add('qhd')
            elif long_side >= 1920 and short_side >= 1080:
                out.add('fhd')
            elif long_side >= 1280 and short_side >= 720:
                out.add('hd')
        if re.search('\\b(?:4k|uhd|ultra hd|2160p?)\\b', low):
            out.add('4k')
        if re.search('\\b(?:qhd|1440p?|2k)\\b', low):
            out.add('qhd')
        if re.search('\\b(?:fhd|1080p?|full ?hd)\\b', low):
            out.add('fhd')
        if re.search('\\b(?:hd|720p?)\\b', re.sub('\\b(?:full|ultra) hd\\b', '', low)):
            out.add('hd')
        return out

    @staticmethod
    def read(field: str, text: str) -> set:
        text = text or ''
        if field == 'screen_size':
            return Spec.inches(text)
        if field == 'refresh_rate':
            return Spec.hertz(text)
        if field == 'resolution':
            return Spec.display_class(text)
        paired = {'%ggb' % float(a) for a, _b in Spec.PAIR.findall(text)}
        return paired or {'%ggb' % float(n) for n in Spec.MEMORY.findall(text)}

    @staticmethod
    def mentions(field: str, text: str) -> bool:
        text = text or ''
        if field == 'ram':
            return bool(Spec.PAIR_HINT.search(text) or Spec.MEMORY.search(text))
        if field == 'screen_size':
            return bool(Spec.INCH.search(text))
        if field == 'refresh_rate':
            return bool(Spec.HZ.search(text))
        return bool(Spec.WXH.search(text) or Spec.CLASS_WORDS.search(text))

    @staticmethod
    def one(values: set) -> str | None:
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def said(text: str) -> str:
        return re.sub('(gb)\\s+of\\s+(ram|memory)\\b', '\\1 \\2', text or '', flags=re.I)

    @staticmethod
    def spoken(field: str, value: str) -> str:
        if field == 'resolution':
            return value
        number = re.sub('[a-z]+$', '', value)
        return number + {'screen_size': ' inch', 'refresh_rate': 'Hz', 'ram': 'GB RAM'}[field]

def _call_key(name: str, args: dict | None) -> str:
    norm = dict(args or {})
    if name in ('search', 'filter'):
        norm.setdefault('in_stock', True)
        norm.setdefault('k', 10)
    if name == 'filter':
        for field in ('category', 'brand'):
            if norm.get(field):
                norm[field] = str(norm[field]).strip().lower().rstrip('s')
        for field in ('max_price', 'min_price'):
            if norm.get(field) is not None:
                try:
                    norm[field] = round(float(norm[field]), 2)
                except (TypeError, ValueError):
                    pass
    return name + json.dumps(norm, sort_keys=True)

def _join_or(items: list) -> str:
    items = list(items)
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1]) + ' or ' + items[-1]

def _words(text: str) -> set:
    return set(re.findall('[a-z0-9]+', (text or '').lower()))

def _stems(text: str) -> set:
    return {w.rstrip('s') for w in re.findall('[a-z]+', (text or '').lower())}

def _same_noun(a: str, b: str) -> bool:
    x = re.sub('[^a-z0-9]+', '', (a or '').lower()).rstrip('s')
    y = re.sub('[^a-z0-9]+', '', (b or '').lower()).rstrip('s')
    return bool(x) and x == y

def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(str(value).replace(',', ''))
    except (TypeError, ValueError):
        return None

class Canon:
    ALIASES = {'video cards': 'graphics_card', 'graphics cards': 'graphics_card', 'video graphics cards': 'graphics_card', 'gpu': 'graphics_card', 'gpus': 'graphics_card', 'mobile phones': 'smartphone', 'cell phones': 'smartphone', 'cellphones': 'smartphone', 'cellphone': 'smartphone', 'phones': 'smartphone', 'smartphones': 'smartphone', 'unlocked phones': 'smartphone', 'smart phones': 'smartphone', 'wi fi': 'networking', 'wifi': 'networking', 'smartwatches': 'smartwatch', 'smart watches': 'smartwatch', 'headphones headsets': 'headphones'}
    HIGH_REFRESH = re.compile('(\\d{2,3}(?:\\.\\d+)?)\\s*-?\\s*hz', re.I)

    @staticmethod
    def norm(text: Any) -> str:
        return ' '.join(re.findall('[a-z0-9]+', str(text or '').casefold()))

    @staticmethod
    def stems(text: Any) -> set:
        out: set = set()
        for token in re.findall('[a-z0-9]+', str(text or '').casefold()):
            out.add(token)
            if len(token) > 3 and token.endswith('s'):
                out.add(token[:-1])
        return out

    @classmethod
    def labels(cls, values: Any) -> set:
        out: set = set()
        for value in values:
            low = cls.norm(value)
            if low:
                out.add(low)
                out.add(cls.ALIASES.get(low, low))
        return out

    @classmethod
    def category_matches(cls, expected: str | None, path: Any, title: str='') -> bool:
        if not expected:
            return True
        path = list(path or [])
        if cls.labels((expected,)) & cls.labels(path):
            return True
        exp = cls.norm(expected)
        if exp in ('gaming monitors', 'gaming monitor'):
            tokens = cls.stems(title)
            fast = any((float(m.group(1)) >= 120 for m in cls.HIGH_REFRESH.finditer(str(title or ''))))
            return bool(cls.labels(path) & cls.labels(('Monitors', 'Computer Monitors'))) and ({'gaming', 'monitor'} <= tokens or ('monitor' in tokens and fast))
        if exp in ('over ear', 'over-ear', 'over ear headphones', 'over-ear headphones'):
            tokens = cls.stems(title)
            return bool(cls.labels(path) & cls.labels(('Headphones',))) and ({'over', 'ear'} <= tokens or 'overear' in tokens or 'circumaural' in tokens)
        return False

    @classmethod
    def brand_matches(cls, expected: str | None, actual: str | None) -> bool:
        if not expected:
            return True
        exp, act = (cls.norm(expected), cls.norm(actual))
        if not exp or exp == act:
            return True
        wanted = set(exp.split())
        return bool(wanted) and wanted <= set(act.split())

class Taste:
    GOOD = 0.65
    WATER = '\\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\\b'
    SPORT = '\\b(sports?|running|gym|workout|fitness|ear ?hook|secure fit)\\b'
    WIRELESS = '\\b(wireless|bluetooth|true wireless)\\b'
    KID = '\\b(kids?|child|children|drop ?proof|rugged case|bumper)\\b'
    SIZE = '\\b(8|10|10\\.\\d|11)\\s*(?:inch|in\\b|\\")'
    WIFI = '\\b(wi-?fi|wifi)\\b'
    NAV = '\\b(gps|gnss|maps?|compass|barometer|altitude|tactical|trail|hiking|outdoor)\\b'
    RUGGED = '\\b(10\\s*atm|5\\s*atm|50m|100m|ip68|ip69|mil-std|waterproof|water resistant|sapphire|fiber[- ]reinforced|titanium|rugged|tactical|tank)\\b'
    RUGGED_HALF = '\\b(water|swim|steel|aluminum)\\b'
    KID_PENALTY = '\\b(kid|kids|child|children|myfirst|watch phone)\\b'
    OUTDOOR_BRANDS = {'garmin', 'amazfit', 'fitbit', 'kospet'}
    HOURS = re.compile('(?<![\\w.])(\\d+(?:\\.\\d+)?)\\s*\\+?\\s*(?:hours?|hrs?|h)\\b', re.I)
    DAYS = re.compile('(\\d+(?:\\.\\d+)?)\\s*days?', re.I)
    LONG_HOURS = re.compile('(\\d+(?:\\.\\d+)?)\\s*hours?', re.I)
    PROFILES = {'headphones': {'signals': ('headphone', 'headphones', 'earbud', 'earbuds', 'earphone', 'earphones', 'headset'), 'accessories': ('case for', 'ear pads', 'earpads', 'replacement pad', 'cable for', 'adapter', 'stand for'), 'reach': ('headphones ipx4 sport running 20h', 'neckband sports earphones ipx5 battery hours', 'sweatproof earbuds hours'), 'more': ('tws earbuds sports waterproof 40 hours', 'earhook sport earbuds waterproof playtime'), 'axes': ('sweat_water_resistance', 'sport_secure_fit', 'battery_life', 'wireless_freedom')}, 'smartwatches': {'signals': ('smartwatch', 'smart watch', 'watch phone', 'apple watch', 'galaxy watch', 'pixel watch', 'garmin', 'fitbit', 'amazfit', 'kospet'), 'accessories': ('case for', 'screen protector', 'band for', 'strap for', 'charger for', 'mudra band', 'replacement band', 'watch band'), 'reach': ('rugged gps smartwatch', 'long battery waterproof smartwatch', 'outdoor fitness gps watch'), 'more': ('garmin gps watch 5 atm days battery', 'amazfit outdoor smartwatch compass barometer'), 'axes': ('navigation', 'rugged_water_resistance', 'battery_life', 'outdoor_watch_credibility')}, 'tablets': {'signals': ('tablet', 'ipad', 'galaxy tab', 'fire hd', 'fire max'), 'accessories': ('case for', 'keyboard for', 'screen protector', 'stylus', 'pen for', 'stand for', 'charger for'), 'reach': ('kids tablet 10 inch wifi hours battery', 'kids android tablet wifi 12 hours', 'rugged tablet kids 8 inch'), 'more': ('kids tablet drop proof bumper case wifi hours', 'tablet 10 inch wifi 10 hours battery kids'), 'axes': ('kid_ready_durability', 'battery_life', 'kid_friendly_size', 'wifi_ready')}}

    @classmethod
    def profile(cls, category: str | None) -> str | None:
        low = (category or '').lower()
        if any((w in low for w in ('headphone', 'earphone', 'earbud', 'headset'))):
            return 'headphones'
        if 'watch' in low:
            return 'smartwatches'
        if 'tablet' in low:
            return 'tablets'
        return None

    @staticmethod
    def _has(text: str, pattern: str) -> bool:
        return re.search(pattern, text, flags=re.I) is not None

    @classmethod
    def short_battery(cls, text: str) -> float:
        if cls.DAYS.search(text):
            return 1.0
        hours = [float(m.group(1)) for m in cls.HOURS.finditer(text)]
        if hours:
            best = max(hours)
            if best >= 15:
                return 1.0
            if best >= 10:
                return 0.75
            if best >= 6:
                return 0.5
        return 0.25 if 'battery life' in text else 0.0

    @classmethod
    def long_battery(cls, text: str) -> float:
        days = [float(m.group(1)) for m in cls.DAYS.finditer(text)]
        if days:
            best = max(days)
            if best >= 10:
                return 1.0
            if best >= 5:
                return 0.75
            if best >= 2:
                return 0.5
        hours = [float(m.group(1)) for m in cls.LONG_HOURS.finditer(text)]
        if hours:
            best = max(hours)
            if best >= 120:
                return 1.0
            if best >= 72:
                return 0.75
            if best >= 36:
                return 0.5
        return 0.5 if 'solar' in text or 'battery life' in text else 0.0

    @staticmethod
    def facts_text(facts: dict) -> str:
        return ' '.join(('%s %s' % (str(k).replace('_', ' '), v) for k, v in (facts or {}).items())).casefold()

    @classmethod
    def score(cls, profile: str, title: str, facts: dict, path: list, brand: str) -> tuple:
        title = (title or '').casefold()
        ftext = cls.facts_text(facts)
        text = ftext + ' ' + title
        if profile == 'headphones':
            water = 1.0 if cls._has(text, cls.WATER) else 0.0
            sport = 1.0 if cls._has(text, cls.SPORT) else 0.0
            battery = cls.short_battery(text)
            wireless_text = ftext + ' ' + ' '.join((str(p) for p in path or [])).casefold() + ' ' + title
            wireless = 1.0 if cls._has(wireless_text, cls.WIRELESS) else 0.0
            raw = 0.3 * water + 0.28 * sport + 0.22 * battery + 0.2 * wireless
            axes = {'sweat_water_resistance': water, 'sport_secure_fit': sport, 'battery_life': battery, 'wireless_freedom': wireless}
        elif profile == 'tablets':
            kid = 1.0 if cls._has(text, cls.KID) else 0.0
            battery = cls.short_battery(text)
            size = 1.0 if cls._has(text, cls.SIZE) else 0.0
            wifi = 1.0 if cls._has(text, cls.WIFI) else 0.0
            raw = 0.4 * kid + 0.25 * battery + 0.2 * size + 0.15 * wifi
            axes = {'kid_ready_durability': kid, 'battery_life': battery, 'kid_friendly_size': size, 'wifi_ready': wifi}
        else:
            nav = 1.0 if cls._has(ftext, cls.NAV) else 0.0
            rugged = 1.0 if cls._has(ftext, cls.RUGGED) else 0.5 if cls._has(ftext, cls.RUGGED_HALF) else 0.0
            battery = cls.long_battery(ftext)
            credible = 1.0 if (brand or '').casefold() in cls.OUTDOOR_BRANDS else 0.0
            penalty = 0.2 if cls._has(ftext, cls.KID_PENALTY) else 0.0
            raw = 0.32 * nav + 0.26 * rugged + 0.26 * battery + 0.16 * credible - penalty
            axes = {'navigation': nav, 'rugged_water_resistance': rugged, 'battery_life': battery, 'outdoor_watch_credibility': credible}
        return (round(max(0.0, min(1.0, raw)), 3), axes)
    AXIS_LABELS = {'headphones': (('sweat_water_resistance', 'sweat and water resistance'), ('sport_secure_fit', 'secure sport fit'), ('battery_life', 'battery life'), ('wireless_freedom', 'wireless freedom')), 'tablets': (('kid_ready_durability', 'kid-ready durability'), ('battery_life', 'battery life'), ('kid_friendly_size', 'kid-friendly size'), ('wifi_ready', 'Wi-Fi')), 'smartwatches': (('navigation', 'navigation'), ('rugged_water_resistance', 'rugged water resistance'), ('battery_life', 'battery life'), ('outdoor_watch_credibility', 'outdoor watch brand'))}
    AXIS_KEYWORDS = {'sweat_water_resistance': ('sweat', 'water', 'resistance', 'resistant', 'waterproof', 'sweatproof'), 'sport_secure_fit': ('sport', 'sports', 'fit', 'secure', 'running', 'gym', 'workout'), 'battery_life': ('battery', 'playtime', 'hours', 'charge', 'life'), 'wireless_freedom': ('wireless', 'bluetooth', 'freedom', 'cordless'), 'kid_ready_durability': ('kid', 'kids', 'ready', 'durability', 'durable', 'rugged', 'drop', 'tough'), 'kid_friendly_size': ('size', 'sized', 'inch', 'screen', 'friendly', 'small', 'compact'), 'wifi_ready': ('wifi', 'wi', 'fi', 'internet'), 'navigation': ('navigation', 'gps', 'maps', 'map', 'trail', 'compass'), 'rugged_water_resistance': ('rugged', 'water', 'resistance', 'resistant', 'waterproof', 'durable', 'tough'), 'outdoor_watch_credibility': ('brand', 'credibility', 'credible', 'outdoor', 'maker', 'reputation')}

    @classmethod
    def evidence(cls, profile: str, title: str, facts: dict, path: list, brand: str) -> dict:
        title = (title or '').casefold()
        ftext = cls.facts_text(facts)
        text = ftext + ' ' + title
        _score, axes = cls.score(profile, title, facts, path, brand)

        def first(pattern: str, where: str) -> str | None:
            m = re.search(pattern, where, flags=re.I)
            return m.group(0) if m else None

        def battery(where: str, long_scale: bool) -> str | None:
            days = [(float(m.group(1)), m.group(0)) for m in cls.DAYS.finditer(where)]
            if days and (not long_scale or max(days)[0] >= 2):
                return max(days)[1]
            finder = cls.LONG_HOURS if long_scale else cls.HOURS
            hours = [(float(m.group(1)), m.group(0)) for m in finder.finditer(where)]
            if hours and max(hours)[0] >= (36 if long_scale else 6):
                return max(hours)[1]
            if long_scale and 'solar' in where:
                return 'solar'
            return 'battery life' if 'battery life' in where else None
        if profile == 'headphones':
            wireless_text = ftext + ' ' + ' '.join((str(p) for p in path or [])).casefold() + ' ' + title
            picks = {'sweat_water_resistance': first(cls.WATER, text), 'sport_secure_fit': first(cls.SPORT, text), 'battery_life': battery(text, False), 'wireless_freedom': first(cls.WIRELESS, wireless_text)}
        elif profile == 'tablets':
            picks = {'kid_ready_durability': first(cls.KID, text), 'battery_life': battery(text, False), 'kid_friendly_size': first(cls.SIZE, text), 'wifi_ready': first(cls.WIFI, text)}
        else:
            picks = {'navigation': first(cls.NAV, ftext), 'rugged_water_resistance': first(cls.RUGGED, ftext) or first(cls.RUGGED_HALF, ftext), 'battery_life': battery(ftext, True), 'outdoor_watch_credibility': (brand or '').casefold() or None}
        return {axis: phrase for axis, phrase in picks.items() if axes.get(axis, 0.0) > 0 and phrase}

    @classmethod
    def is_device(cls, profile: str, title: str) -> bool:
        low = ' ' + (title or '').casefold() + ' '
        spec = cls.PROFILES[profile]
        if any((phrase in low for phrase in spec['accessories'])) or any((phrase in low for phrase in Gate.ACCESSORY_PHRASES)):
            return False
        return any((signal in low for signal in spec['signals']))

class Labels:
    COLOURS = ('midnight blue', 'titanium black', 'matte black', 'triple black', 'frost white', 'white smoke', 'cloud grey', 'moonlight grey', 'space gray', 'street gray', 'driftwood sand', 'cypress green', 'tidal blue', 'light blue', 'sky blue', 'indigo blue', 'canvas white', 'platinum silver', 'starlight', 'graphite', 'sandstone', 'transparent', 'silver', 'beige', 'black', 'blue', 'white', 'gray', 'grey', 'green', 'pink', 'violet', 'purple', 'red', 'sage', 'gold', 'khaki', 'peach')
    NAMED = (('wh[-\\s]?1000xm\\s*([456])', lambda m: 'WH-1000XM' + m.group(1)), ('wf[-\\s]?1000xm\\s*([45])', lambda m: 'WF-1000XM' + m.group(1)), ('quietcomfort ultra(?: headphones)?[^\\n]{0,80}\\b2nd gen\\b', lambda _m: 'QuietComfort Ultra 2nd Gen'), ('quietcomfort ultra', lambda _m: 'QuietComfort Ultra'), ('quietcomfort(?: headphones| wireless)?', lambda _m: 'QuietComfort'), ('headphones 700', lambda _m: 'Headphones 700'), ('beats studio pro|studio pro', lambda _m: 'Studio Pro'), ('beats studio buds \\+|studio buds \\+', lambda _m: 'Studio Buds Plus'), ('beats studio buds|studio buds', lambda _m: 'Studio Buds'), ('beats fit pro', lambda _m: 'Fit Pro'), ('momentum true wireless 4|mtw4|momentum 4', lambda _m: 'Momentum 4'), ('accentum', lambda _m: 'Accentum'), ('px7\\s*s2e', lambda _m: 'Px7 S2e'), ('px7\\s*s3', lambda _m: 'Px7 S3'), ('elite 10', lambda _m: 'Elite 10'), ('jbuds lux anc|go lux', lambda _m: 'JBuds Lux ANC'))
    TITLE_CODE = re.compile('\\b[a-z]*\\d{2,5}[a-z0-9]*(?:-[a-z0-9]+)*\\b', re.I)
    NOT_CODE = re.compile('^(?:android|bluetooth|wifi|ips|ddr|lpddr|mtk)|(?:gb|tb|mb|mah|hz|khz|mhz|ghz|mp|fps|inch|inches|p|g)$|^\\d+x\\d+$', re.I)
    FEATURE_PHONE = re.compile('\\b(?:feature|basic|dumb|button|keypad) ?phones?\\b')
    NOT_HEADPHONES = ('join bild expo', 'earbud', 'in-ear', 'in ear', 'earphone', 'open-ear', 'true wireless')
    OVER_OR_ON_EAR = re.compile('\\b(?:over|on)[-\\s]?ear\\b', re.I)

    @staticmethod
    def phrase_in(text: str, phrase: str) -> bool:
        words = re.findall('[a-z0-9]+', (phrase or '').casefold())
        if not words:
            return False
        pattern = '(?<![a-z0-9])' + '\\W+'.join((re.escape(w) for w in words)) + '(?![a-z0-9])'
        return re.search(pattern, (text or '').casefold()) is not None

    @classmethod
    def code(cls, raw: str) -> str | None:
        words = re.findall('[a-z0-9]+', (raw or '').casefold())
        while words and words[-1] in cls.COLOURS:
            words.pop()
        joined = ''.join(words)
        return joined.upper() if 3 <= len(joined) <= 24 and any((c.isdigit() for c in joined)) else None

    @classmethod
    def named(cls, text: str) -> str | None:
        low = (text or '').casefold()
        for pattern, label in cls.NAMED:
            m = re.search(pattern, low)
            if m:
                return label(m)
        return None

    @classmethod
    def model(cls, title: str, brand: str, facts: dict) -> str | None:
        raw = str((facts or {}).get('model') or '')
        if raw:
            if ';' in raw or '/' in raw:
                return None
            named = cls.named(raw)
            if named:
                return named
            words = raw.casefold().split()
            prefix = []
            while len(words) > 1 and words[0].isalpha() and any((c.isdigit() for c in ' '.join(words[1:]))):
                prefix.append(words.pop(0))
            identity = (title or '') + ' ' + (brand or '')
            if prefix and (not cls.phrase_in(identity, ' '.join(prefix))):
                return None
            if raw.isalpha() and 3 <= len(raw) <= 24:
                return raw.upper() if cls.phrase_in(identity, raw) else None
            return cls.code(' '.join(words))
        named = cls.named(title)
        if named:
            return named
        labels: set = set()
        for m in cls.TITLE_CODE.finditer(title or ''):
            token = m.group()
            if cls.NOT_CODE.search(token):
                continue
            suffix = re.match('(?:\\s+(?:pro|max|ultra|plus)\\b)+', title[m.end():], re.I)
            code = cls.code(token + (suffix.group() if suffix else ''))
            if code:
                labels.add(code)
        return next(iter(labels)) if len(labels) == 1 else None

    @classmethod
    def colour(cls, option: str, facts: dict, title: str) -> str | None:
        for text in (option or '', str((facts or {}).get('color') or ''), title or ''):
            low = text.casefold()
            matches = []
            for name in cls.COLOURS + ('sage green',):
                pattern = '(?<![a-z0-9])' + '\\W+'.join((re.escape(w) for w in name.split())) + '(?![a-z0-9])'
                for m in re.finditer(pattern, low):
                    matches.append((m.start(), m.end(), 'green' if name == 'sage green' else name))
            if not matches:
                continue
            longest = [m for m in matches if not any((o[0] <= m[0] and o[1] >= m[1] and (o[1] - o[0] > m[1] - m[0]) for o in matches))]
            labels = {m[2] for m in longest}
            return next(iter(labels)) if len(labels) == 1 else None
        return None

    @classmethod
    def wanted(cls, text: str) -> str | None:
        named = cls.named(text or '')
        if named:
            return named
        words = (text or '').casefold().split()
        while len(words) > 1 and words[0].isalpha() and any((c.isdigit() for c in ' '.join(words[1:]))):
            words.pop(0)
        return cls.code(' '.join(words))

    @classmethod
    def level(cls, model: str | None, colour: str | None, want_model: str | None, want_colour: str | None) -> int | None:
        if not model or not colour or (not want_model) or (not want_colour):
            return None
        same_model = model.casefold() == want_model.casefold() or (cls.code(model) is not None and cls.code(model) == cls.code(want_model)) or (cls.wanted(want_model) is not None and cls.wanted(want_model) == model)
        same_colour = colour.casefold() == want_colour.casefold()
        if same_model and same_colour:
            return 0
        if same_model:
            return 1
        return 2 if same_colour else 3

class Identity:
    OS = re.compile('\\b(?:android(?:\\s*\\d+(?:\\.\\d+)*)?|ios|blackberry(?:\\s*os)?|windows\\s*phone|symbian|harmony\\s*os|hyper\\s*os|sailfish|ubuntu\\s*touch|tizen)\\b', re.I)
    HANDSET = re.compile('\\bsmart\\s*phones?\\b', re.I)
    NOT_HANDSET = re.compile('\\btablets?\\b|\\b(?:feature|basic)\\s*phones?\\b|\\b(?:mystery|blind|lucky)\\s*box(?:es)?\\b|\\b(?:accessor(?:y|ies)|screen\\s*protectors?|phone\\s*cases?|case\\s*(?:cover\\s*)?for|replacement\\s*(?:screen|battery)|smart\\s*watches?|earbuds?|headphones?)\\b', re.I)
    PART_ONLY = re.compile('\\bonly\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b|\\b(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\s+only\\b|^\\s*replacement\\s+(?:(?:lcd|oled|amoled)\\s+)?(?:screen|display)\\b', re.I)
    NEGATED = re.compile('\\bnot\\s+(?:an?\\s+)?(?:smart\\s*phones?|feature\\s*phones?|basic\\s*phones?|tablets?|accessor(?:y|ies))\\b', re.I)
    BUNDLE = re.compile('\\b(?:(?:with\\s+(?:free|bundled)|bundled\\s+with|free)\\s+|(?:with|includes?)\\s+(?:an?\\s+)?(?=(?:headphones?|earbuds?|phone\\s*case|screen\\s*protector|charger|accessor(?:y|ies))\\b))', re.I)
    COMPATIBLE = re.compile('\\bcompatib', re.I)

    @staticmethod
    def _own_words(text: str, title: bool=False) -> str:
        text = Identity.NEGATED.sub('', text or '')
        return Identity.BUNDLE.split(text, maxsplit=1)[0] if title else text

    @staticmethod
    def handset(facts: dict, title: str, option: str) -> str:
        if Identity.PART_ONLY.search(option or ''):
            return 'no'
        facts = facts if isinstance(facts, dict) else {}
        raw_title = str(title or '')
        head = Identity._own_words(raw_title, title=True)
        kind = ' '.join((Identity._own_words(str(facts.get(k) or '')) for k in ('phone_type', 'device_type', 'description_device_type')))
        blob = head + ' ' + kind
        if Identity.NOT_HANDSET.search(blob):
            return 'no'
        if any(('[truncated]' in str(facts.get(k) or '') for k in ('phone_type', 'device_type', 'device_operating_system'))):
            return 'unclear'
        if Identity.HANDSET.search(blob):
            return 'yes'
        os_text = ' '.join((Identity._own_words(str(facts.get(k) or '')) for k in ('device_operating_system', 'description_operating_system') if not Identity.COMPATIBLE.search(str(facts.get(k) or ''))))
        if Identity.OS.search(os_text):
            return 'yes'
        if Identity.OS.search(head) and (not Identity.COMPATIBLE.search(raw_title)):
            return 'yes'
        return 'unclear'

class Request:
    NUMBER = '(\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.(\\d+))?'
    BUDGET = re.compile("(?:at or under|at or below|at most|up to|under|below|no more than|not more than|no higher than|not higher than|within|not exceeding|budget (?:is|of|now)(?: at| only| just)?|budget(?: is)? now(?: only| at)?|can (?:only )?(?:spend|go|pay|do|afford)(?: up to| to)?|can(?:'|’|no)?t go (?:above|over|past|beyond)|(?:the )?most i can (?:do|spend|pay|go|afford)(?: is| to)?|max(?:imum)?(?: is| of|:)?|cap(?:ped)?(?: it| that)? at|(?:my )?(?:limit|ceiling|cap)(?: is|:)|(?:drop|lower|reduce|bring|cut)(?:ped|ed)?(?: it| that| the budget| my budget)?(?: down)? to|down to|stick(?:ing)? to|keep it (?:under|to|at|below)|now (?:only )?(?:up to|at most|under|below)|tops? out at)\\s*(?:php\\s*)?" + NUMBER, re.I)
    CANDIDATE = re.compile('(\\d{6,})::([A-Za-z0-9._\\-]+)')
    QUOTED = re.compile('\\"([^\\"]{2,80})\\"')
    PHRASE = re.compile('matches (?:the phrase )?\\"?([A-Za-z0-9][^\\",]{1,80}?)\\"?\\s*,', re.I)
    SURFACE = re.compile('surface at least (\\d+) distinct', re.I)
    COLOURS = ('midnight blue', 'titanium black', 'matte black', 'triple black', 'frost white', 'white smoke', 'cloud grey', 'moonlight grey', 'space gray', 'street gray', 'driftwood sand', 'cypress green', 'tidal blue', 'light blue', 'sky blue', 'indigo blue', 'canvas white', 'platinum silver', 'sage green', 'starlight', 'graphite', 'sandstone', 'transparent', 'silver', 'beige', 'black', 'blue', 'white', 'gray', 'grey', 'green', 'pink', 'violet', 'purple', 'red', 'sage', 'gold', 'khaki', 'peach')
    WANTED_ITEM = re.compile('prefer the (' + '|'.join(COLOURS) + ') (?=[A-Za-z0-9\\-+ ]*\\d|[A-Z])([A-Za-z0-9][A-Za-z0-9\\-+ ]{0,40}?)(?=\\s*(?:\\.|,|;|!| specifically\\b| if\\b| when\\b|$))', re.I)
    GAMING = re.compile('\\bgaming\\s+monitors?\\b', re.I)
    SOFT_BRAND = re.compile("\\bprefer(?:ably)?\\s+(?:the\\s+)?([A-Za-z][\\w&.'\\- ]{1,38}?)\\s*(?:,|\\.|;|\\bbut\\b|\\bwhen\\b|\\bif\\b|$)")
    SHELF = re.compile("in the ([A-Za-z][\\w&'\\- ]{1,48}?) category", re.I)
    SHELF_SAID = re.compile("\\bcatalog category is\\s+([A-Za-z][\\w&'\\- ]{1,48}?)\\s*(?:\\.|,|;|$)", re.I)
    SHELF_BEFORE = re.compile('\\b([a-z][a-z\\-]{2,20})\\s*(?:,|\\s(?:at or (?:under|below)|at most|under|below|within|with)\\b)', re.I)
    SHELF_PHRASE = re.compile("\\b(?:an?|the)\\s+(?:(?:[A-Z][\\w&.'\\-]*|[A-Za-z]*\\d[\\w.\\-]*)\\s+){0,3}((?:[a-z][a-z\\-]{2,20}\\s+){1,2}[a-z][a-z\\-]{2,20})\\s*(?:,|\\.|\\s(?:at or (?:under|below)|at most|under|below|within|with|device|in stock)\\b)")
    SHELF_FILLER = ('in-stock', 'in stock', 'genuine', 'actual', 'real', 'proper', 'new', 'used', 'good', 'best', 'cheap', 'affordable', 'reliable', 'decent', 'solid', 'nice', 'great', 'one', 'test')
    SPEC_WORD = re.compile('\\b(?:\\d+(?:\\.\\d+)?\\s*(?:-?\\s*inch(?:es)?|in|\\"|hz|gb|tb)|full hd|ultra hd|hd|fhd|qhd|uhd|4k|2k|720p|1080p|1440p|in-stock|in stock|refurbished|new|used|gaming|curved|wireless|smart)\\b', re.I)
    REASON = re.compile('\\b(?:why|reason|reasons|explain|justif\\w*|walk me through)\\b', re.I)
    CURRENCIES = ('PHP', 'USD', 'EUR', 'SGD', 'MYR', 'IDR', 'THB', 'VND')
    MARKERS = (('preference', 'listing_id'), ('constraint', 'exact variant'), ('justification', 'trustworthy reason'), ('intent', 'nice-to-haves'))

    def __init__(self, query: str) -> None:
        self.raw = query or ''
        goal, _, rules = self.raw.partition('\n\nTask rules:')
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self.ceiling(self.goal)
        self.currency = 'PHP'
        m = re.search('\\b([A-Z]{3})\\b', self.goal)
        if m and m.group(1) in self.CURRENCIES:
            self.currency = m.group(1)
        self.candidates = [(pid, sku.rstrip('.,;:')) for pid, sku in self.CANDIDATE.findall(self.goal)]
        phrase = self.PHRASE.search(self.goal)
        quoted = self.QUOTED.findall(self.goal)
        self.theme = re.findall('[a-z0-9]+', (phrase.group(1) if phrase else quoted[0] if quoted else '').lower())
        sm = self.SURFACE.search(self.rules)
        self.need_identities = int(sm.group(1)) if sm else 0
        wanted = self.WANTED_ITEM.search(self.goal)
        self.wanted_color = wanted.group(1).lower() if wanted else None
        self.wanted_model = wanted.group(2).strip() if wanted else None
        self.category = self._shelf(self.goal)
        self.said_shelf = None
        self.brand = self._brand(self.goal, self.category)
        self.soft_brand = None if wanted else self._soft_brand(self.goal)
        self.qualifier = self._qualifier(self.goal)
        self.specs = {f: v for f in Spec.FIELDS if (v := Spec.one(Spec.read(f, Spec.said(self.goal))))}
        self.gate_field, self.gate_value = next(iter(self.specs.items()), (None, None))
        self.sealed = self.gate_field is None and 'firm requirement' in self.goal.lower()
        self.family = self._family()

    def _family(self) -> str:
        if self.candidates:
            return 'ranking'
        if self.need_identities:
            return 'recall'
        if self.wanted_model:
            return 'recovery'
        for name, marker in self.MARKERS:
            if marker in self.rules:
                return name
        if self.theme:
            return 'recall'
        if self.REASON.search(self.goal):
            return 'justification'
        if self.soft_brand:
            return 'intent'
        if self.sealed or self.gate_field or self.brand:
            return 'constraint'
        return 'preference'

    @classmethod
    def ceiling(cls, text: str) -> float | None:
        best = None
        for m in cls.BUDGET.finditer(text or ''):
            try:
                value = float(m.group(1).replace(',', '') + '.' + (m.group(2) or '0'))
            except ValueError:
                continue
            if best is None or value < best:
                best = value
        return best

    @classmethod
    def _shelf(cls, goal: str) -> str | None:
        m = cls.SHELF.search(goal or '')
        if m:
            return m.group(1).strip()
        if cls.GAMING.search(goal or ''):
            return 'gaming monitor'
        m = cls.SHELF_BEFORE.search(goal or '')
        return m.group(1) if m else None

    def say_shelf(self, text: str) -> str | None:
        m = self.SHELF_SAID.search(text or '')
        if not m:
            return None
        said = m.group(1).strip()
        if not said or said.lower() == (self.category or '').lower():
            return None
        self.category = said
        self.said_shelf = said
        return said

    @property
    def filter_category(self) -> str | None:
        return re.sub('^gaming\\s+', '', self.category, flags=re.I) if self.category else None

    def shelf_phrases(self) -> list:
        out = []
        for m in self.SHELF_PHRASE.finditer(self.goal or ''):
            words = [w for w in m.group(1).split() if w.lower() not in self.SHELF_FILLER]
            for n in range(len(words), 1, -1):
                phrase = ' '.join(words[-n:])
                if phrase.lower() not in [o.lower() for o in out]:
                    out.append(phrase)
        return out

    def shelf_forms(self) -> list:
        out = []
        dropped = ''
        if self.category and self.filter_category and (self.filter_category.lower() != self.category.lower()):
            dropped = self.category[:len(self.category) - len(self.filter_category)].strip().lower()
        for word in [self.said_shelf, self.filter_category] + self.shelf_phrases():
            if not word or (dropped and word.lower().startswith(dropped)):
                continue
            for form in (word, word + 's', word[:-1] if word.endswith('s') else None, word.split()[-1] if ' ' in word else None, word.split()[-1] + 's' if ' ' in word else None):
                form = (form or '').strip()
                if form and form.lower() not in [o.lower() for o in out]:
                    out.append(form)
        return out

    @classmethod
    def _brand(cls, goal: str, category: str | None) -> str | None:
        if not category:
            return None
        m = re.search('\\b(?:an?|the)\\s+([^,.;]{0,60}?)\\b' + re.escape(category), goal or '', re.I)
        if not m:
            return None
        words = [w for w in re.findall("[\\w&.'\\-]+", cls.SPEC_WORD.sub(' ', m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and (not any((c.isdigit() for c in w))) and (not Spec.display_class(w))]
        return ' '.join(keep) or None

    @classmethod
    def _soft_brand(cls, goal: str) -> str | None:
        m = cls.SOFT_BRAND.search(goal or '')
        if not m:
            return None
        name = m.group(1).strip(' .,')
        parts = name.split()
        if not parts or len(parts) > 4 or (not any((w[:1].isupper() for w in parts))):
            return None
        return name

    @staticmethod
    def _qualifier(goal: str) -> str | None:
        if re.search('\\bgaming\\s+monitors?\\b', goal or '', re.I):
            return 'gaming'
        if re.search('\\bover[\\s-]?ear\\b', goal or '', re.I) and (not re.search('\\bon[\\s-]?ear\\b', goal or '', re.I)):
            return 'over ear'
        return None
    REVISION = re.compile('\\b(?:must now have|now needs?|change one requirement|restate my requirement|instead of)\\b', re.I)

    def learn(self, text: str, replace: bool=False) -> str | None:
        text = Spec.said(text)
        if replace:
            text = re.split('\\binstead of\\b', text, maxsplit=1, flags=re.I)[0]
        for field in ('screen_size', 'refresh_rate', 'ram', 'resolution'):
            value = Spec.one(Spec.read(field, text))
            if value and (replace or not self.gate_value):
                self.gate_field, self.gate_value = (field, value)
                self.specs[field] = value
                self.sealed = False
                return field + ' = ' + value
        return None

class Facts:

    def __init__(self) -> None:
        self.products: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.state: dict = {}
        self.seen_at: dict = {}
        self.inspected: set = set()
        self.cart: list = []
        self.adds = 0
        self.carted_at: int | None = None
        self.listings: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.surfaced_by: dict = {}
        self.empty_shelves: set = set()
        self.shelves: dict = {}
        self.dead: set = set()
        self.dead_products: set = set()
        self.event_target: str | None = None
        self.calls_made: set = set()
        self.seq = 0

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + '::' + str(sku or '')

    def absorb(self, action: str, observation: dict, request: Request, args: dict | None=None) -> None:
        self.seq += 1
        if not isinstance(observation, dict) or observation.get('error'):
            return
        if action in ('search', 'filter'):
            query = str((args or {}).get('query') or '') if action == 'search' else ''
            rows = observation.get('results') or []
            if action == 'filter' and (not rows) and (not (args or {}).get('brand')):
                word = str((args or {}).get('category') or '').strip().lower()
                if word:
                    self.empty_shelves.add(word)
            for row in rows:
                if isinstance(row, dict):
                    self._row(row, request, query)
        elif action == 'view':
            self._page(observation, request)
        elif action == 'compare':
            for page in observation.get('compared') or []:
                if isinstance(page, dict):
                    self._page(page, request)
        elif action == 'inspect_stock':
            k = self.key(observation.get('product_id'), observation.get('sku'))
            self.state[k] = {'in_stock': bool(observation.get('in_stock')), 'price': _number(observation.get('price'))}
            self.seen_at[k] = self.seq
            self.inspected.add(k)
        elif action == 'add_to_cart':
            added = observation.get('added') or {}
            pid, sku = (added.get('product_id'), added.get('sku'))
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                self.carted_at = self.seq
            self._cart(observation)
        elif action == 'remove_from_cart':
            self._cart(observation)
            removed = observation.get('removed') or {}
            pair = (removed.get('product_id'), removed.get('sku'))
            if pair in self.cart:
                self.cart.remove(pair)
        elif action == 'inspect_cart':
            self._cart(observation)

    def _cart(self, observation: dict) -> None:
        rows = observation.get('cart')
        if not isinstance(rows, list):
            return
        pairs = [(r.get('product_id'), r.get('sku')) for r in rows if isinstance(r, dict) and r.get('product_id')]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get('product_id'):
                continue
            k = self.key(r.get('product_id'), r.get('sku'))
            price = _number(r.get('current_price'))
            stock = r.get('current_in_stock')
            if price is None and stock is None:
                continue
            prev = self.state.get(k, {})
            self.state[k] = {'in_stock': bool(stock) if stock is not None else prev.get('in_stock', True), 'price': price if price is not None else prev.get('price')}
            self.seen_at[k] = self.seq

    def _row(self, row: dict, request: Request, query: str='') -> None:
        pid = row.get('product_id')
        if not pid:
            return
        title = str(row.get('title') or '')
        if not request.theme or all((t in _words(title) for t in request.theme)):
            self.identities.add(str(pid))
        if not row.get('sku'):
            return
        k = self.key(pid, row.get('sku'))
        self.rows[k] = row
        if query and query not in self.surfaced_by.get(k, ''):
            self.surfaced_by[k] = (self.surfaced_by.get(k, '') + ' | ' + query).strip(' |')
        if row.get('price') is not None:
            self.state.setdefault(k, {'in_stock': bool(row.get('in_stock', True)), 'price': _number(row.get('price'))})

    def _page(self, page: dict, request: Request) -> None:
        pid = page.get('product_id')
        if not pid:
            return
        self.products[str(pid)] = page
        path = page.get('category_path')
        if isinstance(path, list) and path:
            leaf = str(path[-1] or '').strip()
            if leaf:
                self.shelves[leaf] = self.shelves.get(leaf, 0) + 1
        if page.get('listing_id'):
            self.listings.add(str(page['listing_id']))
            self.listing_of[str(pid)] = str(page['listing_id'])
        if request.theme and all((t in _words(str(page.get('title') or '')) for t in request.theme)):
            self.identities.add(str(pid))
        for v in page.get('variants') or []:
            if not isinstance(v, dict) or not v.get('sku'):
                continue
            k = self.key(pid, v.get('sku'))
            self.state[k] = {'in_stock': bool(v.get('in_stock')), 'price': _number(v.get('price'))}
            self.seen_at[k] = self.seq
            self.options[k] = str(v.get('options') or '')

    def status(self, pid: str, sku: str) -> dict | None:
        return self.state.get(self.key(pid, sku))

    def fresh(self, pid: str, sku: str) -> bool:
        k = self.key(pid, sku)
        return k in self.inspected and (self.carted_at is None or self.seen_at.get(k, 0) > self.carted_at)

class Reader:

    def __init__(self, facts: Facts) -> None:
        self.facts = facts

    def page(self, pid: str) -> dict:
        return self.facts.products.get(str(pid)) or {}

    def raw(self, pid: str) -> dict:
        f = self.page(pid).get('facts')
        return f if isinstance(f, dict) else {}

    def title(self, pid: str) -> str:
        title = str(self.page(pid).get('title') or '')
        if title:
            return title
        for key, row in self.facts.rows.items():
            if key.startswith(str(pid) + '::'):
                return str(row.get('title') or '')
        return ''

    def brand(self, pid: str) -> str:
        brand = self.page(pid).get('brand')
        if brand:
            return str(brand)
        for key, row in self.facts.rows.items():
            if key.startswith(str(pid) + '::') and row.get('brand'):
                return str(row['brand'])
        return ''

    def shelf(self, pid: str) -> list:
        path = self.page(pid).get('category_path')
        return list(path) if isinstance(path, list) else []

    def option(self, pid: str, sku: str) -> str:
        return self.facts.options.get(Facts.key(pid, sku), '')

    def _without_model_codes(self, pid: str, title: str) -> str:
        for token in re.split('[;,]', str(self.raw(pid).get('model') or '')):
            token = token.strip()
            stem = re.split('[-_]', token)[0] if token else ''
            if token and re.fullmatch('[a-z0-9]+(?:[-_][a-z0-9]+)+', token, re.I) and re.search('[a-z]', stem, re.I) and re.search('\\d', stem) and (not re.fullmatch('(?:[48]k|[ufq]?hd|fullhd|ultrahd|\\d+p)\\d*', stem, re.I)):
                title = re.sub('(?<![\\w-])' + re.escape(token) + '(?![\\w-])', '', title, flags=re.I)
        return title

    @staticmethod
    def fact_values(field: str, text: str) -> set:
        if field == 'ram':
            return Spec.gigabytes(text)
        return Spec.read(field, text)

    def fact_value(self, pid: str, field: str) -> str | None:
        raw = self.raw(pid).get(Spec.LABEL[field])
        if not raw:
            return None
        return Spec.one(self.fact_values(field, str(raw).strip()))

    def claim_value(self, pid: str, field: str) -> str | None:
        raw = self.raw(pid).get(Spec.LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        if field == 'resolution':
            return Spec.one(Spec.display_class(text))
        found = Spec.hertz(text) if field == 'refresh_rate' else Spec.inches(text) if field == 'screen_size' else Spec.gigabytes(text)
        if not found and field in ('refresh_rate', 'screen_size') and re.fullmatch('\\d+(?:\\.\\d+)?', text):
            found = {'%g' % float(text) + Spec.UNIT[field]}
        return Spec.one(found)

    def product_value(self, pid: str, field: str) -> str | None:
        values: set = set()
        conflict = False
        raw = self.raw(pid).get(Spec.LABEL[field])
        if raw:
            found = self.fact_values(field, str(raw))
            conflict = conflict or len(found) > 1
            if field == 'resolution' and Spec.QUARTER_HD.search(str(raw)):
                conflict = True
            values |= found
        title = self.title(pid)
        if field == 'resolution':
            title = self._without_model_codes(pid, title)
            if Spec.QUARTER_HD.search(title):
                conflict = True
        values |= Spec.read(field, title)
        return None if conflict else Spec.one(values)

    def value(self, pid: str, sku: str, field: str, loose: bool=False) -> str | None:
        option = self.option(pid, sku)
        values = Spec.read(field, option)
        if len(values) == 1:
            return next(iter(values))
        if not loose and (values or Spec.mentions(field, option)):
            return None
        return self.product_value(pid, field)

    def measure(self, pid: str, sku: str, field: str, loose: bool=False) -> float | None:
        value = self.value(pid, sku, field, loose)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    def is_resale(self, pid: str) -> bool:
        return bool(re.search('customer (?:rating|review)|\\bcompare \\b|refurb|pre-owned|restored|renewed', self.title(pid), re.I))

    def model_words(self, pid: str) -> set:
        title = self.title(pid).lower()
        title = re.sub('\\(.*?\\)|\\[.*?\\]', ' ', title)
        words = [w for w in re.findall('[a-z0-9]+', title) if not re.fullmatch('\\d+(?:gb|tb)?|ram|rom|\\d+g', w) and w not in _COLOURS and (len(w) > 1)]
        return set(words[:6])
_COLOURS = {'black', 'white', 'blue', 'green', 'red', 'pink', 'purple', 'gold', 'silver', 'gray', 'grey', 'yellow', 'orange', 'brown', 'beige', 'navy', 'teal', 'cyan', 'violet', 'graphite', 'midnight', 'starlight', 'titanium', 'lavender', 'mint', 'cream', 'ivory', 'coral', 'rose', 'bronze', 'copper'}

class Shopper:
    QUESTION = re.compile('\\?|？|\\b(?:which one|what are you|are you going|does it|is that|can you|could you|did the|do you|have you|will you|is it)\\b|(?:^|[.!?;]\\s*)(?:please\\s+)?(?:confirm|reply|respond|answer|tell me|let me know)\\b', re.I)

    def __init__(self, request: Request) -> None:
        self.request = request
        self.budget = request.budget
        self.owes_answer = False
        self.answer_waits = 0
        self.questions_asked = 0
        self.updated = False
        self.asked_opening = False

    def opening_question(self) -> str | None:
        if self.request.family == 'constraint' or self.request.sealed:
            return 'Before I search: which hardware specification is your firm requirement (screen size, resolution class, refresh rate, or memory), and what value do you need?'
        if self.request.family == 'preference':
            return 'Quick question so I choose well: which of your preferences counts most, and in what order should I weigh the others? Let me know your priority.'
        return None

    def hear(self, text: str, facts: Facts) -> list:
        notes = []
        ceiling = Request.ceiling(text)
        if ceiling is not None and (self.budget is None or ceiling < self.budget):
            self.budget = ceiling
            self.updated = True
            notes.append('Budget updated by the shopper: at most %.2f %s. Every later choice must fit it.' % (ceiling, self.request.currency))
            for pid, sku in list(facts.cart):
                price = (facts.status(pid, sku) or {}).get('price')
                if price is not None and price > ceiling + _EPS:
                    facts.dead.add(Facts.key(pid, sku))
                    notes.append('The carted item %s costs %.2f, above the new budget; it will be replaced.' % (Facts.key(pid, sku), price))
        said = self.request.say_shelf(text)
        if said:
            notes.append('The shopper named the catalog category: %s. Filter and judge the shelf by that label.' % said)
        if self.request.family in ('constraint', 'intent'):
            replacing = bool(self.request.gate_value) and Request.REVISION.search(text) is not None
            if replacing or not self.request.gate_value:
                learned = self.request.learn(text, replace=replacing)
                if learned:
                    self.updated = True
                    if replacing:
                        for pid, sku in list(facts.cart):
                            facts.dead.add(Facts.key(pid, sku))
                    notes.append(('Firm requirement replaced: ' if replacing else 'Firm requirement recorded: ') + learned + '. Only a variant that resolves to it may be bought.')
        if self.QUESTION.search(text):
            self.owes_answer = True
            notes.append('The shopper asked a question: answer it with message in your next turn, then continue.')
        return notes

    def answered(self) -> None:
        self.owes_answer = False
        self.answer_waits = 0

class Verdict(dict):
    __getattr__ = dict.__getitem__

class Gate:
    DEVICE_SHELVES = ('smartphone', 'phone', 'tablet', 'laptop', 'monitor', 'camera', 'console')

    def __init__(self, request: Request, facts: Facts, reader: Reader, shopper: Shopper) -> None:
        self.request, self.facts, self.reader, self.shopper = (request, facts, reader, shopper)

    def shelf_known(self) -> bool:
        want = self.request.category
        if not want or self.request.said_shelf:
            return True
        if not self.facts.empty_shelves & {f.lower() for f in self.request.shelf_forms()}:
            return True
        pages = [pid for pid in self.facts.products if self.reader.shelf(pid)]
        if len(pages) < 3:
            return True
        forms = {want, want + 's', want[:-1] if want.lower().endswith('s') else want}
        for pid in pages:
            path = self.reader.shelf(pid)
            if any((Canon.category_matches(form, path, self.reader.title(pid)) for form in forms if form)):
                return True
        return False

    def shelf_ok(self, pid: str) -> bool:
        want = self.request.category
        path = self.reader.shelf(pid)
        if not want or not path or (not self.shelf_known()):
            return True
        title = self.reader.title(pid)
        forms = {want, want + 's', want[:-1] if want.lower().endswith('s') else want}
        if not any((Canon.category_matches(form, path, title) for form in forms if form)):
            return False
        if self.request.qualifier != 'over ear':
            return True
        words = _stems(title) | _words(title)
        return 'over' in words and 'ear' in words or 'overear' in words or 'circumaural' in words

    def brand_ok(self, pid: str) -> bool:
        return Canon.brand_matches(self.request.brand, self.reader.brand(pid))

    def named_brand(self, pid: str) -> bool:
        return bool(self.request.soft_brand) and Canon.brand_matches(self.request.soft_brand, self.reader.brand(pid))

    def device_verdict(self, pid: str, sku: str) -> str:
        if not _same_noun(self.request.category or '', 'smartphone'):
            return 'n/a'
        return Identity.handset(self.reader.raw(pid), self.reader.title(pid), self.reader.option(pid, sku))

    def device_ok(self, pid: str, sku: str) -> bool:
        verdict = self.device_verdict(pid, sku)
        if verdict == 'n/a':
            return True
        if self.request.family in ('constraint', 'justification', 'recovery'):
            return verdict == 'yes'
        return verdict != 'no'
    ACCESSORY_PHRASES = ('case for', 'cover for', 'cases for', 'covers for', 'sleeve for', 'pouch for', 'skin for', 'bag for', 'ear pads', 'earpads', 'ear pad ', 'ear cushions', 'earcushions', 'replacement pad', 'replacement band', 'replacement screen', 'replacement battery', 'replacement', 'spare part', 'cable for', 'cables for', 'adapter', 'stand for', 'holder for', 'mount for', 'band for', 'strap for', 'charger for', 'keyboard for', 'stylus', 'pen for', 'screen protector', 'protector', 'protective cover', 'protective case', 'tempered glass', 'watch band', 'mudra band', 'carabiner', 'compatible with')
    RECOVERY_ACCESSORY = re.compile('\\b(?:case|cases|cover|covers|sleeve|sleeves|skin|skins|pouch|protector|cable|cables|strap|straps|ear ?tips?|ear ?hooks?|adapter|holder|stand|replacement|carabiner)\\b')
    GADGET_SHELVES = ('headphone', 'earphone', 'earbud', 'headset', 'speaker', 'monitor', 'laptop', 'tablet', 'camera', 'watch', 'console', 'keyboard', 'mouse', 'printer', 'router', 'phone')

    def accessory(self, pid: str) -> bool:
        shelf = (self.request.category or '').lower()
        if not any((word in shelf for word in self.GADGET_SHELVES)):
            return False
        title = self.reader.title(pid)
        low = ' ' + Identity._own_words(title, title=True).casefold() + ' '
        if self.request.family == 'recovery':
            return self.RECOVERY_ACCESSORY.search(low) is not None
        if re.match('\\s*(?:for|fits?)\\b', low):
            return True
        if any((phrase in low for phrase in self.ACCESSORY_PHRASES)):
            return True
        noun = re.sub('s$', '', (self.request.filter_category or '').lower().split()[-1] if self.request.filter_category else '')
        if not noun:
            return False
        if re.search('\\bfor\\s+(?:\\w+\\s+){0,2}' + re.escape(noun) + 's?\\b', low):
            return True
        return re.search(re.escape(noun) + 's?\\s+(?:' + '|'.join(self.CARRIED_FOR) + ')\\b', low) is not None
    CARRIED_FOR = ('bag', 'bags', 'sleeve', 'sleeves', 'pouch', 'case', 'cases', 'casing', 'cover', 'covers', 'stand', 'stands', 'arm', 'mount', 'holder', 'strap', 'straps', 'band', 'bands', 'charger', 'chargers', 'adapter', 'adaptor', 'cable', 'cables', 'protector', 'protectors', 'skin', 'dock', 'pad', 'pads', 'cushion', 'cushions', 'tips', 'filter', 'filters', 'bracket')

    def taste(self, pid: str, sku: str) -> tuple:
        profile = Taste.profile(self.request.category)
        if not profile:
            return (0.0, {})
        return Taste.score(profile, self.reader.title(pid), self.reader.raw(pid), self.reader.shelf(pid), self.reader.brand(pid))

    def price_plausible(self, pid: str, sku: str) -> bool:
        price = (self.facts.status(pid, sku) or {}).get('price')
        budget = self.shopper.budget
        if price is None or budget is None or (not self.request.category):
            return True
        shelf = self.request.category.lower()
        if not any((word in shelf for word in self.DEVICE_SHELVES)):
            return True
        return not (price < 100.0 and price < 0.01 * budget)

    def relax(self, pid: str, sku: str) -> int:
        model, colour = (self.request.wanted_model, self.request.wanted_color)
        if not model:
            return 0
        if str(pid) not in self.facts.products:
            return 3
        level = self.recovery_level(pid, sku)
        return 3 if level is None else level

    def recovery_level(self, pid: str, sku: str) -> int | None:
        facts = self.reader.raw(pid)
        title = self.reader.title(pid)
        got_model = Labels.model(title, self.reader.brand(pid), facts)
        got_colour = Labels.colour(self.reader.option(pid, sku), facts, title)
        return Labels.level(got_model, got_colour, self.request.wanted_model, self.request.wanted_color)

    def recovery_candidate(self, pid: str, sku: str) -> bool:
        title = self.reader.title(pid)
        low = title.casefold().strip()
        if not low or low.startswith(('customer ratings', 'customer reviews')):
            return False
        if self.RECOVERY_ACCESSORY.search(low):
            return False
        if _same_noun(self.request.category or '', 'smartphone') or 'phone' in (self.request.category or '').lower():
            return Labels.FEATURE_PHONE.search(low) is None
        if any((w in (self.request.category or '').lower() for w in ('headphone', 'earphone', 'headset'))):
            surface = title + ' ' + self.reader.option(pid, sku)
            return not any((fragment in low for fragment in Labels.NOT_HEADPHONES)) and Labels.OVER_OR_ON_EAR.search(surface) is not None
        return True

    def verdict(self, pid: str, sku: str) -> Verdict:
        fam = self.request.family
        key = Facts.key(pid, sku)
        st = self.facts.status(pid, sku) or {}
        price, stock = (st.get('price'), st.get('in_stock'))
        budget = self.shopper.budget
        dead = key in self.facts.dead or (str(pid) in self.facts.dead_products and fam != 'recovery')
        live = bool(stock) and price is not None and (budget is None or price <= budget + _EPS) and (not dead)
        read = str(pid) in self.facts.products
        field, want = (self.request.gate_field, self.request.gate_value)
        spec = 2
        if field and want:
            got = self.reader.value(pid, sku, field)
            spec = 2 if got == want else -1 if got else 0
        shelf_ok = self.shelf_ok(pid)
        ok = bool(live and read and (spec == 2) and shelf_ok and self.brand_ok(pid) and self.device_ok(pid, sku) and self.price_plausible(pid, sku) and (not self.accessory(pid)))
        if fam == 'justification':
            ok = ok and (not self.reader.is_resale(pid)) and bool(self.reader.brand(pid) or (read and self.soft_feature(pid)))
            if field and want:
                ok = ok and self.reader.claim_value(pid, field) == want and (self.reader.product_value(pid, field) == want)
        if fam == 'preference':
            profile = Taste.profile(self.request.category)
            if profile and read:
                ok = ok and Taste.is_device(profile, self.reader.title(pid))
            if self.facts.event_target:
                dead_listing = self.facts.listing_of.get(self.facts.event_target.split('::', 1)[0])
                if dead_listing and self.facts.listing_of.get(str(pid)) == dead_listing:
                    ok = False
        if fam == 'recall' and self.request.theme:
            ok = ok and all((t in _words(self.reader.title(pid)) for t in self.request.theme))
        if fam == 'recovery' and read and self.request.wanted_model:
            ok = ok and self.recovery_candidate(pid, sku) and (self.recovery_level(pid, sku) is not None)
        if fam == 'ranking':
            ok = ok and (str(pid), str(sku)) in self.request.candidates
            ok = ok and self.reader.measure(pid, sku, 'refresh_rate', loose=True) is not None and (self.reader.measure(pid, sku, 'screen_size', loose=True) is not None)
        return Verdict(pid=str(pid), sku=str(sku), key=key, price=price, live=live, read=read, spec=spec, shelf_ok=shelf_ok, ok=ok, relax=self.relax(pid, sku) if fam == 'recovery' else 0)

    def candidates(self) -> list:
        seen: set = set()
        out = []
        for key in list(self.facts.state):
            pid, _, sku = key.partition('::')
            if sku and key not in seen:
                seen.add(key)
                out.append(self.verdict(pid, sku))
        return out

    def order_ok(self, pid: str, sku: str) -> tuple:
        st = self.facts.status(pid, sku)
        k = Facts.key(pid, sku)
        if st is None:
            return (False, 'inspect_stock the exact product_id and sku first')
        if (pid, sku) not in self.facts.cart:
            return (False, 'add that exact variant to the cart first')
        if not self.facts.fresh(pid, sku):
            return (False, 're-check the variant with inspect_stock after the cart addition')
        if not st.get('in_stock'):
            return (False, 'that variant is out of stock; choose a different product')
        price = st.get('price')
        budget = self.shopper.budget
        if budget is not None and price is not None and (price > budget + _EPS):
            return (False, 'that variant costs %.2f, above the budget of %.2f' % (price, budget))
        if k in self.facts.dead:
            return (False, 'that variant was invalidated by the market change; choose a different product')
        return (True, '')

    def claims(self, pid: str, sku: str) -> list:
        field, want = (self.request.gate_field, self.request.gate_value)
        if not field and self.request.family == 'justification':
            field, want = ('resolution', self.reader.claim_value(pid, 'resolution'))
        if not field or not want:
            return []
        stated = dict(self.request.specs) or {field: want}
        stated.setdefault(field, want)
        out = [{'field': name, 'value': value} for name, value in stated.items() if self.reader.claim_value(pid, name) == value and self.reader.product_value(pid, name) == value]
        if not out:
            return []
        st = self.facts.status(pid, sku) or {}
        budget = self.shopper.budget
        if budget is not None:
            out.append({'field': 'price', 'value': 'at most %.2f' % budget})
        elif st.get('price') is not None:
            out.append({'field': 'price', 'value': '%.2f' % st['price']})
        out.append({'field': 'in_stock', 'value': 'true'})
        brand = self.reader.brand(pid)
        if brand:
            out.append({'field': 'brand', 'value': brand})
        else:
            feature = self.soft_feature(pid)
            if feature:
                out.append({'field': feature, 'value': 'true'})
        return out
    SOFT_FEATURES = (('streaming', '\\b(streaming|conference|conferencing|zoom|teams)\\b|\\bvideo[- ]?call(?:ing|s)?\\b'), ('microphone', '\\b(?:mic|microphone)\\b'), ('privacy_cover', '\\bprivacy (?:cover|shutter)\\b'), ('autofocus', '\\bauto[- ]?focus\\b'), ('hdr', '\\b(hdr|high dynamic range)\\b'))
    NEGATED_BEFORE = re.compile("\\b(?:no|without|lack(?:s|ing)?|missing|exclude[ds]?|unsupported|not|cannot|can['’]t|doesn['’]t|optional)\\b(?:[\\s-]+[\\w'’]+){0,5}[\\s-]*$", re.I)
    NEGATED_AFTER = re.compile('\\s*(?:(?:support|feature|function|mode|capability)\\s*)?[:=]?\\s*(?:(?:is|are|was|were)\\s+)?(?:not|no|false|none|unsupported|absent|unavailable|missing|excluded|disabled|optional|0)\\b|^\\s*[- ]free\\b|^\\s+sold\\s+separately\\b', re.I)

    def soft_feature(self, pid: str) -> str | None:
        texts = [str(v) for v in self.reader.raw(pid).values()] + [self.reader.title(pid)]
        for field, pattern in self.SOFT_FEATURES:
            supported = False
            defeated = False
            for text in texts:
                for m in re.finditer(pattern, text, re.I):
                    before = re.split('[.;!?\\n]', text[:m.start()])[-1]
                    after = re.split('[.;!?\\n]', text[m.end():])[0]
                    if self.NEGATED_BEFORE.search(before) or self.NEGATED_AFTER.match(after):
                        defeated = True
                        break
                    supported = True
                if defeated:
                    break
            if supported and (not defeated):
                return field
        return None

    def order_args(self, pid: str, sku: str) -> dict:
        out = {'product_id': pid, 'sku': sku}
        if self.request.family == 'justification':
            claims = self.claims(pid, sku)
            if claims:
                out['justification'] = {'claims': claims}
        return out

class Planner:

    def __init__(self, request: Request, facts: Facts, reader: Reader, gate: Gate, shopper: Shopper, tools: set, max_steps: int, clock: 'Stopwatch') -> None:
        self.request, self.facts, self.reader, self.gate, self.shopper = (request, facts, reader, gate, shopper)
        self.tools, self.max_steps, self.clock = (tools, max_steps, clock)
        self.turn = 0
        self.stuck = False
        self.notes: list = []
        self.probes: set = set()
        self.forced_reads = 0
        self.probes_since_ok = 0
        self.exact_reads = 0
        self.searches = 0
        self.searched_twins = False
        self.sibling_forms = 0
        self.hinted_brand = False
        self.hinted_exact = False
        self.reads_done = False
        self.unreadable: set = set()
        self._wishes: set | None = None
        self.wish_searches = 0
        self.model_queries: list = []
        self.queries: list = []
        self.suggested: list | None = None
        self.axis_searches = 0
        self.replayed = False
        self.taste_filtered = False
        self.reach_done = 0
        self.taste_compares = 0

    def note(self, text: str) -> None:
        self.notes.append(text)

    def rank_key(self, v: Verdict, first: bool) -> tuple:
        fam = self.request.family
        price = v.price if v.price is not None else 1000000000000.0
        if fam == 'ranking':
            hz = self.reader.measure(v.pid, v.sku, 'refresh_rate', loose=True) or 0.0
            inch = self.reader.measure(v.pid, v.sku, 'screen_size', loose=True) or 0.0
            return (-hz, -inch, price, v.key)
        if fam == 'recovery':
            family = 1
            if self.facts.event_target and '-' in self.facts.event_target:
                family = 0 if v.sku.split('-')[0] == self.facts.event_target.split('::', 1)[1].split('-')[0] else 1
            return (v.relax, family, price, v.key)
        if fam == 'preference' and Taste.profile(self.request.category):
            return (-self.gate.taste(v.pid, v.sku)[0], price, v.key)
        missed = sum((1 for f, want in self.request.specs.items() if f != self.request.gate_field and self.reader.value(v.pid, v.sku, f) != want))
        proven = 0 if Identity.handset(self.reader.raw(v.pid), self.reader.title(v.pid), self.reader.option(v.pid, v.sku)) == 'yes' else 1
        if fam == 'intent' and first and self.request.soft_brand:
            near = 0 if self.gate.named_brand(v.pid) else 1
            return (near, missed, proven, price, v.key)
        return (missed, proven, price, v.key)

    def target(self) -> Verdict | None:
        first = self.facts.adds == 0
        fam = self.request.family
        good = [v for v in self.gate.candidates() if v.ok]
        if fam == 'preference':
            carted = [v for v in good if (v.pid, v.sku) in self.facts.cart]
            if carted:
                return carted[0]
            if Taste.profile(self.request.category) and good:
                good.sort(key=lambda v: self.rank_key(v, first))
                best = good[0]
                if self.explored() or self.gate.taste(best.pid, best.sku)[0] >= 1.0 - _EPS or self.turn >= self.max_steps * 0.6:
                    return best
                return None
            if good and self.turn >= self.max_steps - 8:
                good.sort(key=lambda v: (-self.wish_hits(v.pid, v.sku), -(v.price or 0.0)))
                return good[0]
            return None
        if fam == 'intent' and first and self.request.soft_brand:
            named = [v for v in good if self.gate.named_brand(v.pid)]
            if named:
                good = named
            elif self.turn < self.max_steps * 0.6 and (not self.brand_probed()):
                self.hint_brand()
                return None
        if fam == 'recovery':
            if first:
                exact = [v for v in good if v.relax == 0]
                if exact:
                    good = exact
                elif self.turn < self.max_steps * 0.6 or (self.model_leads() and self.exact_reads < 3):
                    self.hint_exact()
                    return None
            if self.sibling_leads() and self.forced_reads < 8:
                return None
            if not first and self.facts.event_target and (not self.searched_twins) and ('search' in self.tools) and good and (min((v.relax for v in good)) >= 2):
                return None
        if not good:
            return None
        good.sort(key=lambda v: self.rank_key(v, first))
        return good[0]

    def recall_need(self) -> int:
        early = self.searches < 8 and self.turn < self.max_steps * 0.5
        return self.request.need_identities + (2 if early else 0)

    def matches_theme(self, pid: str) -> bool:
        theme = self.request.theme
        if not theme:
            return True
        titles = [self.reader.title(pid)] + [str(row.get('title') or '') for key, row in self.facts.rows.items() if key.startswith(str(pid) + '::')]
        return any((all((t in _words(title) for t in theme)) for title in titles if title))

    def ready(self) -> bool:
        fam = self.request.family
        if self.facts.adds:
            return True
        if self.request.sealed:
            if self.turn < self.max_steps * 0.6:
                return False
            self.request.sealed = False
            self.note('The firm requirement was never stated, so the remaining requirements decide the choice.')
        if fam in ('intent', 'constraint') and self.turn < 4 and (not self.shopper.updated):
            return False
        if fam == 'preference' and (len(self.facts.listings) < 2 or not self.explored()):
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if fam == 'recall':
            return not (roomy and len({q.strip().lower() for q in self.queries}) < 3)
        hunt = 0 if self.max_steps < 25 else 1
        if roomy and (not self.reads_done) and (self.probes_since_ok < hunt) and (len({v.pid for v in self.gate.candidates() if v.ok}) < 2):
            return False
        return True

    def next(self) -> list | None:
        if self.shopper.owes_answer and (not self.facts.event_target):
            if self.shopper.answer_waits < (1 if self.max_steps < 25 else 2) or 'message' not in self.tools:
                self.shopper.answer_waits += 1
                return None
            return [('message', {'content': self.answer_text()})]
        fam = self.request.family
        if fam == 'preference' and Taste.profile(self.request.category):
            move = self.taste_explore()
            if move is not None:
                return move
        if fam == 'ranking' and 'compare' in self.tools:
            missing = [p for p, _s in self.request.candidates if p not in self.facts.products and p not in self.unreadable]
            if missing:
                return [('compare', {'product_ids': missing[:5]})]
            unchecked = [(p, s) for p, s in self.request.candidates if Facts.key(p, s) not in self.facts.inspected and p in self.facts.products]
            if unchecked and 'inspect_stock' in self.tools:
                return [('inspect_stock', {'product_id': p, 'sku': s}) for p, s in unchecked[:5]]
        if fam == 'recovery' and (not self.facts.adds) and self.request.wanted_model and ('search' in self.tools):
            forced = self.sibling_search()
            if forced is not None:
                return [forced]
        if fam == 'preference' and self.facts.event_target and (not self.replayed) and self.model_queries and ('search' in self.tools) and (self.turn < self.max_steps * 0.7):
            self.replayed = True
            query = self.model_queries[0]
            key = 'search' + json.dumps({'in_stock': True, 'k': 10, 'query': query}, sort_keys=True)
            self.facts.calls_made.discard(key)
            self.facts.calls_made.add(key)
            self.searches += 1
            return [('search', {'query': query, 'in_stock': True, 'k': 10})]
        if fam == 'preference' and (not Taste.profile(self.request.category)) and (self.turn < self.max_steps * 0.6) and (self.facts.adds == 0 or self.facts.event_target):
            forced = self.lacking_axis_search()
            if forced is not None:
                return [forced]
        target = self.target()
        if target is None or not self.ready():
            forced = self.read_more()
            if forced is not None:
                return [forced]
            if target is None:
                return None
        pid, sku = (target.pid, target.sku)
        if (pid, sku) in self.facts.cart:
            if not self.facts.fresh(pid, sku):
                return [('inspect_stock', {'product_id': pid, 'sku': sku})]
            order = [('place_test_order', self.gate.order_args(pid, sku))]
            note = self.order_message(pid, sku)
            if note and 'message' in self.tools:
                return [('message', {'content': note})] + order
            return order
        if not self.ready():
            return None
        swap = [('remove_from_cart', {'product_id': cp, 'sku': cs}) for cp, cs in self.facts.cart if Facts.key(cp, cs) in self.facts.dead and 'remove_from_cart' in self.tools]
        if not self.facts.fresh(pid, sku):
            return swap + [('inspect_stock', {'product_id': pid, 'sku': sku})]
        return swap + [('add_to_cart', {'product_id': pid, 'sku': sku})]

    def reach_queries(self) -> tuple:
        profile = Taste.profile(self.request.category)
        if not profile:
            return ()
        spec = Taste.PROFILES[profile]
        if self.max_steps < 25:
            return spec['reach']
        wish = self.wish_query()
        return spec['reach'] + spec.get('more', ()) + ((wish,) if wish else ())

    def explored(self) -> bool:
        if not Taste.profile(self.request.category):
            return True
        return self.taste_filtered and self.reach_done >= len(self.reach_queries()) and (self.taste_compares >= 2 or not self.taste_rows())

    def taste_rows(self) -> list:
        profile = Taste.profile(self.request.category)
        if not profile:
            return []
        budget = self.shopper.budget
        scored: dict = {}
        for key, row in self.facts.rows.items():
            pid, _, sku = key.partition('::')
            if not sku or pid in self.facts.products or pid in self.facts.dead_products or (pid in self.unreadable):
                continue
            st = self.facts.state.get(key) or {}
            price = st.get('price')
            if not st.get('in_stock') or price is None or (budget is not None and price > budget + _EPS):
                continue
            title = str(row.get('title') or '')
            if not Taste.is_device(profile, title):
                continue
            preview = Taste.score(profile, title, {}, [], str(row.get('brand') or ''))[0]
            if pid not in scored or scored[pid][0] < preview:
                scored[pid] = (preview, price)
        return [pid for pid, _v in sorted(scored.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))]

    def taste_explore(self) -> list | None:
        req = self.request
        profile = Taste.profile(req.category)
        if not profile or (self.facts.adds and (not self.facts.event_target)):
            return None
        if not self.taste_filtered and 'filter' in self.tools and req.category:
            self.taste_filtered = True
            shelf = self.shelf_word()
            if not shelf:
                self.taste_filtered = True
                return None
            args = {'category': shelf, 'in_stock': True, 'k': 10}
            if self.shopper.budget is not None:
                args['max_price'] = self.shopper.budget
            key = _call_key('filter', args)
            if key not in self.facts.calls_made:
                self.facts.calls_made.add(key)
                self.probes.add(key)
                return [('filter', args)]
        reach = self.reach_queries()
        while self.reach_done < len(reach) and 'search' in self.tools:
            query = reach[self.reach_done]
            self.reach_done += 1
            args = {'query': query, 'in_stock': True, 'k': 10}
            key = 'search' + json.dumps(args, sort_keys=True)
            if key in self.facts.calls_made:
                continue
            self.facts.calls_made.add(key)
            return [('search', args)]
        limit = 3 if self.facts.event_target else 1 if self.max_steps < 25 else 2
        if self.taste_compares < limit and 'compare' in self.tools:
            good = [v for v in self.gate.candidates() if v.ok]
            if self.facts.event_target and good:
                return None
            rows = self.taste_rows()
            if rows:
                self.taste_compares += 1
                return [('compare', {'product_ids': rows[:5]})]
        return None

    def read_more(self) -> tuple | None:
        fam = self.request.family
        if fam == 'recall' and self.request.theme and ('search' in self.tools):
            query = self.recall_query()
            if query is not None:
                key = 'search' + json.dumps({'in_stock': True, 'k': 10, 'query': query}, sort_keys=True)
                self.facts.calls_made.add(key)
                return ('search', {'query': query, 'in_stock': True, 'k': 10})
        if fam == 'recall' and len({q.strip().lower() for q in self.queries}) < 3 and (self.turn < self.max_steps * 0.5):
            return None
        twin = self.twin_search()
        if twin is not None:
            return twin
        exact = self.model_leads()
        if exact and self.exact_reads < 3:
            self.exact_reads += 1
            if len(exact) > 1 and 'compare' in self.tools:
                return ('compare', {'product_ids': exact[:5]})
            return ('view', {'product_id': exact[0]})
        leads = self.leads()
        if leads and self.forced_reads < 8 and (self.turn >= 2 or self.stuck):
            self.forced_reads += 1
            if fam == 'recall':
                return ('view', {'product_id': leads[0]})
            if len(leads) > 1 and 'compare' in self.tools:
                return ('compare', {'product_ids': leads[:5]})
            return ('view', {'product_id': leads[0]})
        if self.turn < 2 and (not self.stuck) and (fam not in ('intent', 'constraint', 'justification')):
            return None
        for kind, args in self.ladder():
            key = _call_key(kind, args)
            if key in self.probes or key in self.facts.calls_made:
                continue
            self.probes.add(key)
            self.facts.calls_made.add(key)
            if any((v.ok for v in self.gate.candidates())):
                self.probes_since_ok += 1
            return (kind, args)
        self.reads_done = True
        return None

    def ladder(self) -> list:
        req = self.request
        rungs = []
        shelf = self.shelf_word()
        if 'filter' in self.tools and shelf:
            base = {'category': shelf, 'in_stock': True, 'k': 10}
            if self.shopper.budget is not None:
                base['max_price'] = self.shopper.budget
            dead_brand = self.reader.brand(self.facts.event_target.split('::', 1)[0]) if self.facts.event_target else ''
            seen_brands: list = []
            for brand in (dead_brand, req.brand, req.soft_brand):
                if brand and brand.lower() not in seen_brands:
                    seen_brands.append(brand.lower())
                    rungs.append(('filter', dict(base, brand=brand)))
            plain = not req.brand
            if plain and req.family == 'intent' and req.soft_brand and (not self.facts.adds) and (not self.brand_probed()):
                plain = False
            if plain:
                rungs.append(('filter', dict(base)))
        if 'search' in self.tools:
            spoken = Spec.spoken(req.gate_field, req.gate_value) if req.gate_field and req.gate_value else None
            model = req.wanted_model
            forms = [(req.brand, spoken, req.category), (req.soft_brand, spoken, req.category)]
            if not req.brand:
                forms += [(spoken, req.category), (model, req.category)]
            forms.append((req.brand, req.category))
            for words in forms:
                query = ' '.join((w for w in words if w))
                if query and query != req.category:
                    rungs.append(('search', {'query': query, 'in_stock': True, 'k': 10}))
        if 'filter' in self.tools and shelf:
            counts: dict = {}
            if self.max_steps >= 25 and (req.family == 'intent' or (req.family in ('constraint', 'justification') and (not req.brand))):
                for row in self.facts.rows.values():
                    brand = str(row.get('brand') or '').strip().lower()
                    if re.fullmatch("[a-z][a-z .&'\\-]{2,}", brand):
                        counts[brand] = counts.get(brand, 0) + 1
            filtered = set()
            for key in self.facts.calls_made:
                if key.startswith('filter'):
                    m = re.search('"brand": "([^"]*)"', key)
                    if m:
                        filtered.add(m.group(1).lower())
            for brand, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]:
                if brand not in seen_brands and brand not in filtered:
                    seen_brands.append(brand)
                    rungs.append(('filter', dict(base, brand=brand)))
        return rungs

    def model_leads(self) -> list:
        req = self.request
        if req.family != 'recovery' or self.facts.adds or (not req.wanted_model):
            return []
        code = re.sub('[^a-z0-9]+', '', req.wanted_model.lower())
        label = (Labels.wanted(req.wanted_model) or '').lower()
        cat = (req.category or '').lower()
        nouns = {cat, cat + 's', cat[:-1] if cat.endswith('s') else cat}
        if 'phone' in cat:
            nouns |= {'phone', 'phones', 'mobile', 'smartphone', 'smartphones'}
        if any((w in cat for w in ('headphone', 'earphone', 'earbud', 'headset'))):
            nouns |= {'headphone', 'headphones', 'earphone', 'earphones', 'earbud', 'earbuds', 'headset'}
        nouns.discard('')
        out = []
        for key, row in self.facts.rows.items():
            pid, _, sku = key.partition('::')
            if not sku or pid in self.facts.products or pid in self.facts.dead_products or (pid in self.unreadable):
                continue
            if not (self.facts.state.get(key) or {}).get('in_stock'):
                continue
            title = str(row.get('title') or '')
            if not nouns & _words(title):
                continue
            flat = re.sub('[^a-z0-9]+', '', title.lower())
            surfaced = re.sub('[^a-z0-9]+', '', self.facts.surfaced_by.get(key, '').lower())
            named = code and code in flat or (label and label in flat)
            by_search = code and code in surfaced
            if (named or by_search) and pid not in out and (not self.gate.accessory(pid)):
                out.append(pid)
        return out

    def sibling_leads(self) -> list:
        if self.facts.event_target and '-' in self.facts.event_target.split('::', 1)[1]:
            family = self.facts.event_target.split('::', 1)[1].split('-')[0]
            kin = []
            for key, row in self.facts.rows.items():
                pid, _, sku = key.partition('::')
                if not sku or pid in self.facts.products or pid in self.facts.dead_products:
                    continue
                if sku.split('-')[0] == family and (self.facts.state.get(key) or {}).get('in_stock') and (pid not in kin):
                    kin.append(pid)
            if kin:
                return kin
        anchor = self.facts.event_target.split('::', 1)[0] if self.facts.event_target else None
        words = self.reader.model_words(anchor) if anchor else set()
        if not words:
            for pid in self.facts.products:
                if self.request.wanted_model and re.sub('[^a-z0-9]+', '', self.request.wanted_model.lower()) in re.sub('[^a-z0-9]+', '', self.reader.title(pid).lower()) and self.gate.shelf_ok(pid):
                    words = self.reader.model_words(pid)
                    break
        if not words:
            return []
        out = []
        for key, row in self.facts.rows.items():
            pid, _, sku = key.partition('::')
            if not sku or pid in self.facts.products or pid in self.facts.dead_products:
                continue
            st = self.facts.state.get(key) or {}
            if not st.get('in_stock'):
                continue
            if words <= _words(str(row.get('title') or '')) and pid not in out:
                out.append(pid)
        return out

    def family_leads(self) -> list:
        if not self.facts.event_target or '-' not in self.facts.event_target.split('::', 1)[1]:
            return []
        family = self.facts.event_target.split('::', 1)[1].split('-')[0]
        out = []
        for key, row in self.facts.rows.items():
            pid, _, sku = key.partition('::')
            if not sku or sku.split('-')[0] != family or pid in self.facts.products or (pid in out):
                continue
            title = str(row.get('title') or '')
            if self.request.theme and (not all((t in _words(title) for t in self.request.theme))):
                continue
            if self.gate.accessory(pid):
                continue
            out.append(pid)
        return out

    def leads(self) -> list:
        siblings = self.sibling_leads() if self.request.family == 'recovery' else self.family_leads()
        leads = []
        for key, row in self.facts.rows.items():
            pid, _, sku = key.partition('::')
            if not sku or pid in self.facts.products or pid in self.facts.dead_products or (pid in siblings):
                continue
            st = self.facts.state.get(key) or {}
            price = st.get('price')
            budget = self.shopper.budget
            if not st.get('in_stock') or price is None or (budget is not None and price > budget + _EPS):
                continue
            title = str(row.get('title') or '')
            if self.request.theme and (not all((t in _words(title) for t in self.request.theme))):
                continue
            if self.gate.accessory(pid):
                continue
            if self.request.brand and (not Canon.brand_matches(self.request.brand, str(row.get('brand') or ''))):
                continue
            named = 0
            if self.request.wanted_model and self.facts.adds == 0 and (not self.facts.event_target):
                code = re.sub('[^a-z0-9]+', '', self.request.wanted_model.lower())
                surfaced = re.sub('[^a-z0-9]+', '', self.facts.surfaced_by.get(key, '').lower())
                if code not in re.sub('[^a-z0-9]+', '', title.lower()) and code not in surfaced:
                    continue
            states = 0 if self.states_requirement(title, row) else 1
            leads.append((named, states, price, pid))
        leads.sort()
        out = list(siblings)
        for _named, _states, _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def states_requirement(self, title: str, row: dict) -> bool:
        req = self.request
        if not req.gate_field or not req.gate_value:
            return False
        text = ' '.join((str(row.get(k) or '') for k in ('title', 'options', 'variant'))) + ' ' + (title or '')
        want = Spec.one(Spec.read(req.gate_field, req.gate_value))
        got = Spec.read(req.gate_field, text)
        return bool(want) and want in got
    PRIORITY = re.compile('([a-z/\\- ]{3,40}?) counts most,? with (?:a |an |the )?([a-z/\\- ]{3,40}?) close behind,? (?:a |an |the )?([a-z/\\- ]{3,40}?) next,? (?:a |an |the )?([a-z/\\- ]{3,40}?) counts least', re.I)
    AXIS_WORDS = {'sweat': {'sweat', 'sweatproof', 'waterproof', 'water', 'resistant', 'resistance', 'ipx4', 'ipx5', 'ipx6', 'ipx7', 'ipx8', 'ip67', 'ip68'}, 'water': {'waterproof', 'water', 'resistant', 'resistance', 'ipx4', 'ipx5', 'ipx6', 'ipx7', 'ipx8', 'ip67', 'ip68', 'sweatproof'}, 'sport': {'sport', 'sports', 'running', 'gym', 'workout', 'fitness', 'earhook', 'neckband', 'secure'}, 'battery': {'battery', 'mah', 'hours', 'hour', 'hrs', 'playtime', 'playback', 'endurance'}, 'wireless': {'wireless', 'bluetooth', 'tws', 'cordless'}, 'noise': {'noise', 'anc', 'cancelling', 'canceling', 'cancellation'}, 'comfort': {'comfort', 'comfortable', 'lightweight', 'ergonomic'}, 'sound': {'sound', 'bass', 'hifi', 'stereo', 'audio'}}

    def axes(self) -> list:
        m = self.PRIORITY.search(self.shopper_said)
        if not m:
            return []
        out = []
        for i in (1, 2, 3, 4):
            phrase = re.sub('^php\\s+', '', m.group(i).strip().lower())
            words = set(re.findall('[a-z0-9]+', phrase))
            for key, extra in self.AXIS_WORDS.items():
                if key in words:
                    words |= extra
            out.append((phrase, words))
        return out
    AXIS_QUERY = {'battery': 'long battery playtime hours', 'sweat': 'sweatproof waterproof IPX', 'water': 'waterproof sweatproof IPX', 'sport': 'sports running earhook', 'wireless': 'wireless bluetooth', 'noise': 'noise cancelling ANC', 'comfort': 'comfortable lightweight', 'sound': 'bass stereo sound'}

    def lacking_axis_search(self) -> tuple | None:
        axes = self.axes()
        good = [v for v in self.gate.candidates() if v.ok]
        if not axes or not good or self.axis_searches >= 2 or ('search' not in self.tools):
            return None
        for label, words in axes:
            if any((words & _words(self.reader.title(v.pid) + ' ' + ' '.join((str(x) for x in self.reader.raw(v.pid).values()))) for v in good)):
                continue
            key = next((k for k in self.AXIS_QUERY if k in words), None)
            terms = self.AXIS_QUERY.get(key, label)
            query = terms + (' ' + self.request.category if self.request.category else '')
            call = 'search' + json.dumps({'in_stock': True, 'k': 10, 'query': query}, sort_keys=True)
            if call in self.facts.calls_made:
                continue
            self.facts.calls_made.add(call)
            self.axis_searches += 1
            self.searches += 1
            self.note("Every verified candidate so far lacks '%s'; searching for one that states it." % label)
            return ('search', {'query': query, 'in_stock': True, 'k': 10})
        return None

    def evidence_note(self, pid: str, sku: str) -> str:
        axes = self.axes()
        if not axes:
            return ''
        text = self.reader.title(pid) + ' ' + ' '.join((str(v) for v in self.reader.raw(pid).values())) + ' ' + self.reader.option(pid, sku)
        have = _words(text)
        stated_hours = bool(re.search('\\d+\\s*(?:hours?|hrs?|h\\b|mah)', text, re.I)) or 'playtime' in have or 'playback' in have
        has, lacks = ([], [])
        for label, words in axes:
            hit = bool(words & have)
            if 'battery' in words and (not stated_hours):
                hit = False
            (has if hit else lacks).append(label)
        return ' [evidences: %s; lacks: %s]' % (', '.join(has) or 'none', ', '.join(lacks) or 'none')
    STOP = {'the', 'and', 'for', 'with', 'of', 'in', 'on', 'to', 'by', 'a', 'an', 'new', 'original', 'free', 'shipping', 'sale', 'hot', 'best', 'pcs', 'pc', 'set', 'ml', 'g', 'kg', 'cm', 'mm'}

    def expansion_words(self) -> list:
        counts: dict = {}
        theme = set(self.request.theme)
        brands: set = set()
        for row in self.facts.rows.values():
            brands |= _words(str(row.get('brand') or ''))
        for key, row in self.facts.rows.items():
            title = str(row.get('title') or '')
            words = _words(title)
            if not theme <= words:
                continue
            for w in words:
                if w in theme or w in self.STOP or w in brands or (len(w) < 3) or w.isdigit() or re.fullmatch('\\d+(?:g|kg|ml|l|cm|mm|pcs?|pc|pack|x)', w) or any((w in t or t in w for t in theme)):
                    continue
                counts[w] = counts.get(w, 0) + 1
        return [w for w, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def expansion_words_apart(self, used: set) -> list:
        counts: dict = {}
        theme = set(self.request.theme)
        ranked = self.expansion_words()
        for key, row in self.facts.rows.items():
            words = _words(str(row.get('title') or ''))
            if not theme <= words or words & used:
                continue
            for w in words:
                if w in ranked:
                    counts[w] = counts.get(w, 0) + 1
        order = {w: i for i, w in enumerate(ranked)}
        apart = [w for w, _n in sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))]
        return apart + [w for w in ranked if w not in counts]

    def recall_query(self) -> str | None:
        phrase = ' '.join(self.request.theme)
        made = {q.strip().lower() for q in self.queries}
        if len(made) >= 3:
            return None
        if not made:
            return phrase
        used: set = set()
        for q in made:
            used |= _words(q)
        extra = used - set(self.request.theme)
        ranked = self.expansion_words_apart(extra) if extra else self.expansion_words()
        options = [phrase + ' ' + w for w in ranked if w not in used] if not extra else []
        options += [phrase + ' ' + w for w in self.suggested or [] if w not in used]
        options += [phrase + ' ' + w for w in ranked if w not in used]
        if self.request.category:
            options.append(phrase + ' ' + self.request.category)
            options.append(max(self.request.theme, key=len) + ' ' + self.request.category)
        for query in options:
            if query.strip().lower() not in made:
                return query
        return None

    def twin_search(self) -> tuple | None:
        dead = self.facts.event_target
        if self.searched_twins or not dead or 'search' not in self.tools or (self.request.family in ('preference', 'recall')):
            return None
        dead_pid = dead.split('::', 1)[0]
        title = self.reader.title(dead_pid)
        if not title or not self.gate.shelf_ok(dead_pid):
            return None
        self.searched_twins = True
        words = title.split()[:8]
        req = self.request
        if req.gate_field and req.gate_value:
            words.append(Spec.spoken(req.gate_field, req.gate_value))
        self.note('The item that died is one of several listings sharing a title; searching that title surfaces its other colors and sizes, which meet the same requirements.')
        return ('search', {'query': ' '.join(words), 'k': 10, 'in_stock': True})

    def sibling_search(self) -> tuple | None:
        if self.sibling_forms >= 2 or not self.request.wanted_color:
            return None
        cands = self.gate.candidates()
        if any((v.ok and v.relax == 0 for v in cands)):
            return None
        if self.sibling_forms and self.model_leads():
            return None
        near = [v for v in cands if v.read and v.relax == 1 and self.gate.shelf_ok(v.pid)]
        by_title = [w for w in self.reader.title(near[0].pid).split()[:6] if w.lower() not in _COLOURS] if near else None
        if not near and self.turn < 2 and (not self.stuck):
            return None
        by_code = [self.request.wanted_model] + ([self.request.filter_category] if self.request.filter_category else [])
        for words in (f for f in (by_title, by_code) if f):
            query = ' '.join(words + [self.request.wanted_color])
            key = 'search' + json.dumps({'in_stock': True, 'k': 10, 'query': query}, sort_keys=True)
            if key in self.facts.calls_made:
                continue
            self.sibling_forms += 1
            self.facts.calls_made.add(key)
            return ('search', {'query': query, 'k': 10, 'in_stock': True})
        self.sibling_forms = 2
        return None

    def brand_probed(self) -> bool:
        brand = (self.request.soft_brand or '').lower()
        if not brand:
            return True
        filtered = searched = False
        for key in self.facts.calls_made:
            if brand not in key.lower():
                continue
            if key.startswith('filter'):
                filtered = True
            elif key.startswith('search'):
                searched = True
        return filtered and searched

    def hint_brand(self) -> None:
        if not self.hinted_brand:
            self.hinted_brand = True
            self.note("No %s product that meets every firm requirement has been read yet. The shopper's first choice is graded on that brand, so filter by brand=%s within the budget and read those products before any other." % (self.request.soft_brand, self.request.soft_brand))

    def hint_exact(self) -> None:
        if not self.hinted_exact:
            self.hinted_exact = True
            self.note('The shopper\'s first choice is graded on the exact model %s in %s. Search "%s %s" (the model code with the category word, never the code alone), read the listings that carry it, and cart only the one in that color.' % (self.request.wanted_model, self.request.wanted_color or 'the named color', self.request.wanted_model, self.request.filter_category or self.request.category or ''))

    def shortlist(self) -> None:
        cands = self.gate.candidates()
        if self.request.family == 'preference' and (not self.facts.cart):
            good = [v for v in cands if v.ok]
            if Taste.profile(self.request.category):
                good.sort(key=lambda v: self.rank_key(v, True))
                if good:
                    self.note('Candidates that meet every firm requirement, best fit first: ' + '; '.join(('%s::%s at %.2f (fit %.2f)' % (v.pid, v.sku, v.price or 0.0, self.gate.taste(v.pid, v.sku)[0]) for v in good[:6])) + '. The controller will inspect, cart and order the best fit.')
                return
            good.sort(key=lambda v: (-self.wish_hits(v.pid, v.sku), v.price if v.price is not None else 1000000000000.0))
            if good:
                self.note('Products that meet every firm requirement so far: ' + '; '.join(('%s::%s at %.2f%s' % (v.pid, v.sku, v.price or 0.0, self.evidence_note(v.pid, v.sku)) for v in good[:6])) + ". Pick the one whose own title and facts evidence the most of the shopper's ranked wants, the top-ranked want first; inspect it, then add it.")
                return
        best = self.target()
        if best:
            self.note('Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; read other candidates only if one could rank better.' % (best.pid, best.sku, best.price or 0.0))
            return
        field, want = (self.request.gate_field, self.request.gate_value)
        blocked = [v for v in cands if v.read and v.live and (not v.ok)]
        if blocked and field:
            self.note("None of the products read so far can be verified. For each, the evidence for %s must resolve to exactly %s from the variant's own option text or the product's raw facts: %s. Look at different products." % (field, want, ', '.join((v.key for v in blocked[:3]))))
        elif blocked:
            self.note('The products read so far fail a firm requirement (category, brand, stock, budget or the kind of device requested). Search for different products.')

    def answer_text(self) -> str:
        target = self.target() or (self.gate.candidates() or [None])[0]
        if not target:
            return 'I am still checking the options against your requirements; nothing is decided yet.'
        parts = ['I am going with ' + (self.reader.title(target.pid) or target.pid)[:90]]
        field, want = (self.request.gate_field, self.request.gate_value)
        if field and want and (self.reader.value(target.pid, target.sku, field) == want):
            parts.append('its listed ' + field.replace('_', ' ') + ' is ' + want)
        if target.price is not None:
            parts.append('it is %.2f %s and in stock' % (target.price, self.request.currency))
        return ', '.join(parts) + '.'

    def order_message(self, pid: str, sku: str) -> str | None:
        parts = []
        if self.shopper.owes_answer:
            parts.append(self.answer_text())
        note = self.decision_note(pid, sku, with_head=not parts)
        if note:
            parts.append(note)
        return ' '.join(parts) or None

    def shelf_word(self) -> str | None:
        dead = self.facts.empty_shelves
        if self.request.said_shelf:
            return self.request.said_shelf
        own = self.catalog_spelling()
        if own and own.lower() not in dead:
            return own
        for form in self.request.shelf_forms():
            if form.lower() not in dead:
                return form
        for leaf, _n in sorted(self.facts.shelves.items(), key=lambda kv: (-kv[1], kv[0])):
            if leaf.lower() not in dead:
                return leaf
        return None

    def catalog_spelling(self) -> str | None:
        want = self.request.filter_category
        if not want:
            return None
        forms = {want.lower(), want.lower() + 's', want.lower().rstrip('s')}
        best, seen = (None, -1)
        for pid in self.facts.products:
            for tok in self.reader.shelf(pid):
                t = str(tok or '').strip()
                low = t.lower()
                if not t or low in forms:
                    continue
                if Canon.labels((t,)) & Canon.labels((want,)):
                    n = self.facts.shelves.get(t, 0)
                    if n > seen:
                        best, seen = (t, n)
        return best

    def listing_of(self, pid: str) -> str:
        page = self.facts.products.get(str(pid)) or {}
        if page.get('listing_id'):
            return str(page['listing_id'])
        for key, row in self.facts.rows.items():
            if key.startswith(str(pid) + '::'):
                lid = row.get('source_listing_id') or row.get('listing_id')
                if lid:
                    return str(lid)
        return ''

    def alternative(self, pid: str) -> Verdict | None:
        first = self.facts.adds == 0
        others = [v for v in self.gate.candidates() if v.pid != pid and v.pid in self.facts.products]
        good = [v for v in others if v.ok]
        if good:
            mine = self.listing_of(pid)
            good.sort(key=lambda v: (1 if mine and self.listing_of(v.pid) == mine else 0,) + tuple(self.rank_key(v, first)))
            return good[0]
        gone = [v for v in others if v.read and self.gate.device_verdict(v.pid, v.sku) != 'no' and ((v.pid, v.sku) in self.facts.cart)]
        if gone:
            gone.sort(key=lambda v: self.rank_key(v, first))
            return gone[0]
        return None

    def decision_note(self, pid: str, sku: str, with_head: bool=True) -> str | None:
        if self.request.family != 'preference':
            return None
        cur = self.request.currency
        st = self.facts.status(pid, sku) or {}
        price = st.get('price')
        parts = []
        if with_head:
            head = 'Ordering ' + (self.reader.title(pid) or pid)[:80] + ' (' + pid + ')'
            if price is not None:
                head += ' at %.2f %s' % (price, cur)
            if st.get('in_stock'):
                head += ', in stock'
            parts.append(head)
        else:
            parts.append('That is product ' + pid)
        profile = Taste.profile(self.request.category)
        alt = self.alternative(pid)
        if profile:
            labels = dict(Taste.AXIS_LABELS[profile])
            own = self._evidence(profile, pid)
            if own:
                parts.append('Its own listing shows ' + ', '.join(("%s ('%s')" % (labels[a], ph) for a, ph in own.items())) + ', which is what your priorities ask for')
            else:
                parts.append('Its listing is the closest to your priorities among the items read')
            if alt is not None:
                parts.append(self._contrast(profile, labels, own, pid, sku, alt))
        else:
            carried = self._carried_words(pid, sku)
            if carried:
                parts.append('Its listing carries your words ' + ', '.join(("'" + w + "'" for w in carried)) + ', which is what you asked for')
            if alt is not None:
                line = 'Compared with the alternative ' + (self.reader.title(alt.pid) or alt.pid)[:60].strip() + ' (' + alt.pid + ')'
                theirs = self._carried_words(alt.pid, alt.sku)
                if len(theirs) < len(carried):
                    line += ', whose listing carries fewer of them (%d against %d), the chosen one is the better fit' % (len(theirs), len(carried))
                else:
                    line += ', the chosen one is the one verified in stock at the checked price'
                parts.append(line)
        return '. '.join(parts) + '.'

    def _evidence(self, profile: str, pid: str) -> dict:
        found = Taste.evidence(profile, self.reader.title(pid), self.reader.raw(pid), self.reader.shelf(pid), self.reader.brand(pid))
        return {axis: found[axis] for axis in self.axis_order(profile) if axis in found}

    def axis_order(self, profile: str) -> list:
        default = [axis for axis, _label in Taste.AXIS_LABELS[profile]]
        stated = self.axes()
        if len(stated) < 2:
            return default
        out = []
        for _phrase, words in stated:
            best, hits = (None, 0)
            for axis in default:
                if axis in out:
                    continue
                n = sum((1 for w in Taste.AXIS_KEYWORDS[axis] if w in words))
                if n > hits:
                    best, hits = (axis, n)
            if best is not None:
                out.append(best)
        return out + [axis for axis in default if axis not in out]

    def _contrast(self, profile: str, labels: dict, own: dict, pid: str, sku: str, alt: Verdict) -> str:
        cur = self.request.currency
        line = 'Compared with the alternative ' + (self.reader.title(alt.pid) or alt.pid)[:60].strip() + ' (' + alt.pid
        if alt.price is not None:
            line += ', %.2f %s' % (alt.price, cur)
        line += ')'
        other = self._evidence(profile, alt.pid)
        lacking = [labels[a] for a in own if a not in other]
        mine, theirs = (self.gate.taste(pid, sku)[1], self.gate.taste(alt.pid, alt.sku)[1])
        stronger = [a for a in own if a in other and mine.get(a, 0.0) > theirs.get(a, 0.0) + _EPS]
        price = (self.facts.status(pid, sku) or {}).get('price')
        budget = self.shopper.budget
        if not alt.ok and budget is not None and (alt.price is not None) and (alt.price > budget + _EPS):
            line += ', now %.2f %s above your %.2f %s budget, the chosen one is the fit still within budget' % (alt.price, cur, budget, cur)
        elif not alt.ok and (not alt.live):
            line += ', which is no longer available, the chosen one is the fit still in stock'
        elif lacking:
            line += ', whose listing shows no ' + _join_or(lacking) + ', the chosen one is the better fit'
        elif stronger:
            a = stronger[0]
            line += ", which shows only '%s' for %s, the chosen one is stronger on %s" % (other[a], labels[a], labels[a])
        elif price is not None and alt.price is not None and (price < alt.price - _EPS):
            line += ', which matches it on ' + (', '.join((labels[a] for a in own)) or 'the counted axes') + ' at a higher price, the chosen one is the better value'
        else:
            line += ', which matches it on ' + (', '.join((labels[a] for a in own)) or 'the counted axes') + ', the chosen one is the one verified in stock at the checked price'
        return line

    def _carried_words(self, pid: str, sku: str) -> list:
        wants = self.wish_words()
        have = _words(' '.join((str(v) for v in self.reader.raw(pid).values())) + ' ' + self.reader.title(pid) + ' ' + self.reader.option(pid, sku))
        out = []
        for want in sorted(wants):
            stem = want[:5] if len(want) > 5 else want
            token = next((h for h in sorted(have) if h.startswith(stem)), None)
            if token and token not in out:
                out.append(token)
        return out[:6]
    WANT_VOCAB = {'sweat', 'water', 'resistant', 'resistance', 'waterproof', 'sweatproof', 'sport', 'sports', 'running', 'gym', 'workout', 'wireless', 'bluetooth', 'battery', 'playtime', 'noise', 'cancelling', 'canceling', 'anc', 'comfortable', 'comfort', 'lightweight', 'bass', 'microphone', 'mic', 'gaming', 'durable', 'fast', 'charging', 'secure', 'fit', 'earhook', 'neckband', 'portable', 'compact', 'foldable', 'stereo', 'hifi', 'long', 'lasting'}

    def wish_query(self, second: bool=False) -> str | None:
        wants = [w for w in sorted(self.wish_words()) if w in self.WANT_VOCAB]
        if not wants:
            wants = sorted(self.wish_words())[:4]
        if not wants:
            return None
        words = wants[:5] if not second else wants[2:7] or wants[:5]
        if self.request.category and self.request.category.lower() not in words:
            words = words + [self.request.category]
        return ' '.join(words)

    def wish_hits(self, pid: str, sku: str) -> int:
        wants = self.wish_words()
        if not wants:
            return 0
        have = _words(' '.join((str(v) for v in self.reader.raw(pid).values())) + ' ' + self.reader.title(pid) + ' ' + self.reader.option(pid, sku))
        stems = {w[:5] if len(w) > 5 else w for w in wants}
        return sum((1 for stem in stems if any((h.startswith(stem) for h in have))))

    def wish_words(self) -> set:
        if self._wishes is None:
            said = self.request.goal + ' ' + self.shopper_said
            drop = _words(self.request.category or '') | {'the', 'and', 'for', 'with', 'that', 'this', 'you', 'your', 'find', 'need', 'want', 'order', 'stock', 'place', 'under', 'over', 'one', 'php', 'test', 'buy', 'rather', 'than', 'pay', 'bit', 'more', 'settle', 'something', 'its', 'would', 'like', 'have'}
            self._wishes = {w for w in _words(said) if len(w) > 3 and (not w.isdigit()) and (w not in drop)}
        return self._wishes
    shopper_said = ''

class Stopwatch:

    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)

class Session:

    def __init__(self, binding: dict, problem_id: str, clock: Stopwatch) -> None:
        self.binding = dict(binding)
        self.problem_id = problem_id
        self.clock = clock
        self.turn = 0
        self.done = False
        self.pending: dict | None = None

    def _post(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            result = _PROXY.post('/environment/call', json_data=envelope)
            if isinstance(result, dict) and result.get('calls'):
                return result
            if self.clock.remaining() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def send(self, calls: list) -> tuple:
        if self.pending is not None:
            replay = self._post(self.pending['envelope'])
            if replay is None:
                return ([None] * len(calls), None)
            self.pending = None
        turn = self.turn + 1
        group = self.problem_id + '-turn-' + str(turn)
        envelope = {**self.binding, 'call_id': group, 'idempotency_key': group, 'turn': turn, 'calls': [{'call_id': '%s-%d' % (group, i), 'action': {'name': name, 'args': args}} for i, (name, args) in enumerate(calls, start=1)]}
        result = self._post(envelope)
        if result is None:
            self.pending = {'envelope': envelope, 'turn': turn}
            return ([None] * len(calls), None)
        self.turn = turn
        observations = []
        for entry in result.get('calls') or []:
            wrap = (entry.get('observation') if isinstance(entry, dict) else None) or {}
            obs = wrap.get('observation') if isinstance(wrap.get('observation'), dict) else wrap
            if isinstance(obs, dict) and wrap.get('error') and (not obs.get('error')):
                obs = dict(obs, error=wrap.get('error'))
            if wrap.get('done'):
                self.done = True
            observations.append(obs)
        while len(observations) < len(calls):
            observations.append(None)
        user = result.get('user_message')
        return (observations, str(user['content']) if isinstance(user, dict) and user.get('content') else None)

class Model:

    def __init__(self, clock: Stopwatch) -> None:
        provider = (getenv('INFERENCE_PROVIDER') or '').strip().lower()
        self.roster = _ROSTER.get(provider) or _ROSTER['chutes']
        pinned = getenv('SANDBOX_MODEL')
        if pinned:
            self.roster = (pinned,) + tuple((m for m in self.roster if m != pinned))
        self.index = 0
        self.clock = clock

    def ask(self, system: str, prompt: str) -> str:
        for _attempt in range(2):
            model = self.roster[min(self.index, len(self.roster) - 1)]
            resp = _PROXY.post('/inference/chat/completions', json_data={'model': model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}], 'temperature': 0, 'max_tokens': 60})
            if resp and isinstance(resp, dict) and resp.get('choices'):
                msg = resp['choices'][0].get('message') or {}
                text = str(msg.get('content') or '').strip()
                if text:
                    return text
            self.index = min(self.index + 1, len(self.roster) - 1)
        return ''

    def propose(self, messages: list, tools: list, max_calls: int) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.index, len(self.roster) - 1)]
            resp = _PROXY.post('/inference/chat/completions', json_data={'model': model, 'messages': messages, 'tools': tools, 'tool_choice': 'required', 'temperature': 0})
            if resp and isinstance(resp, dict) and resp.get('choices'):
                msg = resp['choices'][0].get('message') or {}
                calls = msg.get('tool_calls') or []
                if calls:
                    return {'content': msg.get('content') or '', 'tool_calls': calls[:max_calls]}
                messages.append({'role': 'assistant', 'content': msg.get('content') or ''})
                messages.append({'role': 'user', 'content': '[note] No tool call was made. Choose exactly one of the supplied tools now.'})
                continue
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.index = min(self.index + 1, len(self.roster) - 1)
        return None

def _arguments(tool_call: dict) -> dict:
    raw = (tool_call.get('function') or {}).get('arguments') or '{}'
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}

def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + '... [truncated]"}'
_PLAYBOOK = {'intent': "Firm requirements: the category, the named hardware specification, the budget and stock. A named brand is only a preference: take it when a fully compliant item of that brand exists, otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view or compare to read the exact variant's specification from raw facts. Answer the shopper with message; a budget the shopper lowers later binds every later choice.", 'recall': 'Run several differently worded searches (k=10, in_stock=true) built from the requested words and close synonyms until enough distinct matching products have appeared; every requested word must appear as a whole word of the title. Then view the strongest candidates so the controller can verify one and buy it.', 'constraint': "The shopper holds one firm hardware requirement and states it when asked; meet it exactly for the specific variant, reading the variant's own option text before any product-wide value. Use filter(category, brand, max_price, in_stock=true) and view candidates. If the carted variant sells out, move to a different product that still meets every requirement.", 'preference': 'You already asked which preference matters most; use the answer. Search or filter the category within budget, then compare candidates with different listing_id values, reading raw facts. The first cart item is graded, so make it the best genuine match. If it is repriced above budget, recover to a different listing_id.', 'ranking': "Only the named candidate variants count, and the controller ranks them: compare all of them first so their facts are on the record. Highest advertised maximum refresh rate wins, then the largest screen, then the lowest price, reading the selected variant's own value before a product-wide one. After the market moves, the remaining candidates are re-ranked.", 'recovery': 'Find the preferred model and colour first: search the model code, then view it so every variant and colour is on the record. The first cart item must be exactly that model and colour. If it disappears, relax as little as possible -- the same model in another colour before any other model -- and never relax the category, budget or stock.', 'justification': 'Find a product whose own raw facts state the requested specification, in the requested category and brand, in stock and within budget. View or compare it so its facts are on the record, then the controller inspects it, carts it and orders it with claims taken only from that evidence.', 'generic': 'Understand the firm requirements, filter or search the catalog, view candidates, and let the controller verify and buy the best one.'}
_RULES = "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the cart and the order once a candidate is fully verified, so your job is to surface and read candidates: search or filter, then view or compare so their raw facts, variant options and prices are on record. Never add an item to the cart as a placeholder -- the first cart addition is graded. Prices and stock change while you shop; the controller re-checks the carted item and tells you when to move to a different product. Answer every direct question from the shopper with one short message before anything else, and never send progress or status updates. Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments, and do not repeat filters the controller has already run; one compare of up to five products beats several view calls. A search query is words a product title would contain; never put a price or a budget figure in it -- the budget belongs to filter's max_price."

class Agent:

    def __init__(self, problem: dict) -> None:
        env = problem['environment']
        self.policy = env['policy_view']
        self.tools = list(self.policy.get('tools') or [])
        self.tool_names = {(t.get('function') or {}).get('name') for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get('max_steps') or 30)
        self.max_calls = max(1, int(self.policy.get('max_calls_per_turn') or 1))
        self.clock = Stopwatch(_SECONDS_PER_PROBLEM)
        self.request = Request(str(self.policy.get('query') or ''))
        self.facts = Facts()
        self.reader = Reader(self.facts)
        self.shopper = Shopper(self.request)
        self.gate = Gate(self.request, self.facts, self.reader, self.shopper)
        self.planner = Planner(self.request, self.facts, self.reader, self.gate, self.shopper, self.tool_names, self.max_steps, self.clock)
        self.session = Session(env['binding'], str(problem.get('problem_id') or problem.get('id') or 'problem'), self.clock)
        self.model = Model(self.clock)
        self.messages: list = [{'role': 'system', 'content': self._system_prompt()}, {'role': 'user', 'content': self.policy.get('query') or ''}]
        self.dialogue: list = []
        self.pending_recheck: tuple | None = None
        self.rejected_streak = 0
        self.rejected_total = 0
        self.cart_inspections = 0
        self.silent_rounds = 0

    def _system_prompt(self) -> str:
        play = _PLAYBOOK.get(self.request.family, _PLAYBOOK['generic'])
        extra = ''
        if self.request.family == 'recall' and self.request.need_identities:
            extra = ' You must surface at least %d distinct matching products in search or filter results before ordering; the controller reports the count after each search.' % self.request.need_identities
        return _RULES + '\n\nFamily playbook: ' + play + extra

    def _flush_notes(self) -> None:
        for text in self.planner.notes:
            self.messages.append({'role': 'user', 'content': '[note] ' + text})
            self.dialogue.append({'role': 'assistant', 'content': '[note] ' + text[:300]})
        self.planner.notes = []

    def _execute(self, calls: list, content: str='', ids: list | None=None, from_model: bool=False) -> list:
        ids = list(ids or [])
        while len(ids) < len(calls):
            ids.append('agent-%d-%d' % (self.session.turn + 1, len(ids) + 1))
        tool_calls = [{'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}} for cid, (name, args) in zip(ids, calls)]
        self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
        observations, said = self.session.send(calls)
        self.planner.turn = self.session.turn
        self.dialogue.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls, 'environment_result': {'observations': observations, 'user_message': said}})
        for cid, (name, args), obs in zip(ids, calls, observations):
            if obs is None:
                self.facts.calls_made.discard(_call_key(name, args))
                self.messages.append({'role': 'tool', 'tool_call_id': cid, 'content': json.dumps({'error': 'environment call failed; it will be retried'})})
                continue
            self.facts.absorb(name, obs, self.request, args)
            self.messages.append({'role': 'tool', 'tool_call_id': cid, 'content': _trim(json.dumps(obs), _OBSERVATION_CHARS)})
            if name == 'add_to_cart' and isinstance(obs, dict) and (not obs.get('error')):
                added = obs.get('added') or {}
                if added.get('product_id'):
                    self.pending_recheck = (str(added['product_id']), str(added['sku']))
            if name in ('search', 'filter'):
                self.planner.searches += 1
                if name == 'search' and args.get('query'):
                    self.planner.queries.append(str(args['query']))
                if self.request.family == 'recall' and self.request.need_identities:
                    have, need = (len(self.facts.identities), self.planner.recall_need())
                    self.planner.note('Distinct matching products surfaced so far: %d of the required %d.' % (have, need) + ('' if have >= need else ' Keep searching with different wording.'))
            if name == 'inspect_cart':
                self.cart_inspections += 1
            if name == 'message':
                self.shopper.answered()
        if said:
            self.messages.append({'role': 'user', 'content': said})
            self.planner.shopper_said += ' ' + said
            for text in self.shopper.hear(said, self.facts):
                self.planner.note(text)
        if from_model and any((name in ('view', 'compare') for name, _a in calls)):
            self.planner.shortlist()
        self._flush_notes()
        self.session.done = self.session.done or any((isinstance(o, dict) and o.get('done') for o in observations if o))
        return observations

    def _suggest_expansions(self) -> list:
        theme = self.request.theme
        titles = []
        for row in self.facts.rows.values():
            title = str(row.get('title') or '')
            if all((t in _words(title) for t in theme)) and title not in titles:
                titles.append(title)
        brands: set = set()
        for row in self.facts.rows.values():
            brands |= _words(str(row.get('brand') or ''))
        prompt = 'A shopper wants a product whose listing title contains the words "%s"%s. The catalog search is a bag of words and shows ten listings per query. These matching titles came back for the phrase alone:\n- %s\n\nSuggest three single words to add to the phrase, each likely to surface a DIFFERENT kind of matching product not listed above -- a product type, form, material or use that other such products carry, never a brand, never a word from the phrase, and not a word most of the titles above already use. Answer with the three words only, separated by commas.' % (' '.join(theme), ' in the %s category' % self.request.category if self.request.category else '', '\n- '.join((t[:90] for t in titles[:10])))
        answer = self.model.ask('You expand product-catalog searches with one word at a time.', prompt)
        counts: dict = {}
        for title in titles:
            for w in _words(title):
                counts[w] = counts.get(w, 0) + 1
        common = {w for w, c in counts.items() if c >= max(3, len(titles) // 2)}
        out = []
        for token in re.findall('[A-Za-z][A-Za-z0-9\\-]{2,}', answer):
            w = token.lower().strip('-')
            if w in theme or w in brands or w in Planner.STOP or (w in common) or (w in out) or any((w in t or t in w for t in theme)):
                continue
            out.append(w)
            if len(out) >= 3:
                break
        if out:
            self.planner.note('Expansion words for the remaining searches: ' + ', '.join(out) + '.')
        return out

    def _recheck_after_add(self) -> None:
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._execute([('inspect_stock', {'product_id': pid, 'sku': sku})])[0]
        if not isinstance(obs, dict) or obs.get('error'):
            return
        stock = bool(obs.get('in_stock'))
        price = _number(obs.get('price'))
        budget = self.shopper.budget
        over = budget is not None and price is not None and (price > budget + _EPS)
        if stock and (not over):
            self.planner.note('Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it.' % (Facts.key(pid, sku), price))
            self._flush_notes()
            return
        self.facts.event_target = Facts.key(pid, sku)
        self.facts.dead.add(Facts.key(pid, sku))
        self.facts.dead_products.add(str(pid))
        self.planner.forced_reads = 0
        reason = 'went out of stock' if not stock else 'was repriced to %.2f, above the budget' % (price or 0.0)
        guidance = {'recovery': 'Recover with the smallest possible change: the same model in another color before any other model.', 'ranking': 'The remaining named candidates are re-ranked; the controller takes the new best.', 'preference': "Recover to a product with a different listing_id that still fits the shopper's priorities.", 'constraint': 'Choose a different product that still meets every requirement, not another variant of the same one.'}.get(self.request.family, 'Choose a different product that meets every firm requirement.')
        self.planner.note('Market change: the carted item %s %s. ' % (Facts.key(pid, sku), reason) + guidance)
        self.planner.shortlist()
        self._flush_notes()

    def _finish(self) -> None:
        for pid, sku in reversed(self.facts.cart):
            ok, _why = self.gate.order_ok(pid, sku)
            if ok and (not self.session.done):
                self._order(pid, sku)
                return
        for pid, sku in reversed(self.facts.cart):
            if self.session.done:
                return
            st = self.facts.status(pid, sku) or {}
            price = st.get('price')
            budget = self.shopper.budget
            if st.get('in_stock') and Facts.key(pid, sku) not in self.facts.dead and (budget is None or price is None or price <= budget + _EPS):
                obs = self._execute([('inspect_stock', {'product_id': pid, 'sku': sku})])[0]
                if self.session.done or not isinstance(obs, dict) or (not obs.get('in_stock')):
                    continue
                self._order(pid, sku)
                return
        if self.session.done:
            return
        if self.session.turn > self.max_steps - 3:
            self._last_resort()
            return
        good = [v for v in self.gate.candidates() if v.ok]
        if not good and self.request.family != 'justification':
            good = [v for v in self.gate.candidates() if v.live and v.spec == 2 and v.shelf_ok and self.gate.brand_ok(v.pid) and self.gate.price_plausible(v.pid, v.sku) and (not self.gate.accessory(v.pid)) and (not self.reader.is_resale(v.pid)) and (self.gate.device_verdict(v.pid, v.sku) != 'no') and (not self.request.candidates or (v.pid, v.sku) in self.request.candidates) and self.planner.matches_theme(v.pid)]
        if not good:
            self._last_resort()
            return
        good.sort(key=lambda v: self.planner.rank_key(v, self.facts.adds == 0))
        for v in good[:3]:
            if self.session.done or self.session.turn > self.max_steps - 3:
                break
            pid, sku = (v.pid, v.sku)
            if (pid, sku) not in self.facts.cart:
                obs = self._execute([('add_to_cart', {'product_id': pid, 'sku': sku})])[0]
                if self.session.done:
                    return
                if not isinstance(obs, dict) or obs.get('error'):
                    continue
            obs = self._execute([('inspect_stock', {'product_id': pid, 'sku': sku})])[0]
            if self.session.done:
                return
            price = _number(obs.get('price')) if isinstance(obs, dict) else None
            budget = self.shopper.budget
            if not isinstance(obs, dict) or not obs.get('in_stock') or (budget is not None and price is not None and (price > budget + _EPS)):
                self.facts.dead.add(Facts.key(pid, sku))
                continue
            self._order(pid, sku)
            return
        self._last_resort()

    def _order(self, pid: str, sku: str) -> None:
        calls = []
        note = self.planner.order_message(pid, sku)
        if note and 'message' in self.tool_names and (self.max_calls > 1):
            calls.append(('message', {'content': note}))
            self.shopper.answered()
        calls.append(('place_test_order', self.gate.order_args(pid, sku)))
        self._execute(calls)

    def _last_resort(self) -> None:
        live_cart = [(pid, sku) for pid, sku in reversed(self.facts.cart) if Facts.key(pid, sku) not in self.facts.dead and (self.facts.status(pid, sku) or {}).get('in_stock') is not False]
        for pid, sku in live_cart:
            if self.session.done:
                return
            self._order(pid, sku)
        if self.session.done or self.session.turn > self.max_steps - 2:
            return
        rows = []
        for key, row in self.facts.rows.items():
            pid, _, sku = key.partition('::')
            st = self.facts.state.get(key) or {}
            price = st.get('price')
            budget = self.shopper.budget
            if not sku or not st.get('in_stock') or price is None or (budget is not None and price > budget + _EPS):
                continue
            if key in self.facts.dead or pid in self.facts.dead_products:
                continue
            if self.gate.accessory(pid) or self.reader.is_resale(pid) or self.gate.device_verdict(pid, sku) == 'no' or (not self.planner.matches_theme(pid)) or (not self.gate.price_plausible(pid, sku)):
                continue
            if self.request.gate_field and self.gate.verdict(pid, sku).get('spec') != 2:
                continue
            rows.append((0 if self.gate.shelf_ok(pid) and self.gate.brand_ok(pid) else 1, price, pid, sku))
        if not rows:
            for pid, sku in reversed(self.facts.cart):
                if self.session.done:
                    return
                self._order(pid, sku)
            return
        _rank, _price, pid, sku = min(rows)
        obs = self._execute([('add_to_cart', {'product_id': pid, 'sku': sku})])[0]
        if not self.session.done and isinstance(obs, dict) and (not obs.get('error')):
            self._order(pid, sku)

    def _refuse(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return 'unknown tool; choose one of ' + str(sorted((n for n in self.tool_names if n)))
        if name == 'place_test_order':
            pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
            if not pid and self.facts.cart:
                pid, sku = self.facts.cart[-1]
            ok, why = self.gate.order_ok(pid, sku)
            if not ok:
                return 'order refused: ' + why
            if self.request.family == 'recall' and len({q.strip().lower() for q in self.planner.queries}) < 3 and (self.planner.turn < self.max_steps * 0.5):
                return 'order refused: the three differently worded searches come first'
        if name == 'search' and self.request.family == 'recall':
            made = {q.strip().lower() for q in self.planner.queries}
            if len(made) >= 3 and str(args.get('query') or '').strip().lower() not in made:
                return 'search refused: three distinct searches are already on record and the retrieval score is divided by any beyond three; view the matching products instead'
        if name in ('view', 'compare') and self.request.family == 'recall' and self.request.theme:
            pids = [str(args.get('product_id') or '')] if name == 'view' else [str(p) for p in args.get('product_ids') or []]
            off = [p for p in pids if p and p not in self.facts.products and (not self.planner.matches_theme(p))]
            if off:
                return 'read refused: the retrieval score counts every product read that does not match the phrase against you; read only titles carrying every requested word'
            if self.request.family == 'preference' and len(self.facts.listings) < 2:
                return 'order refused: compare at least two products with different listing_id values first'
        if name == 'add_to_cart':
            pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
            if not pid or not sku:
                return 'add_to_cart needs both product_id and sku'
            v = self.gate.verdict(pid, sku)
            if not v.live:
                return 'that variant is out of stock, over budget, or already invalidated'
            if not v.read:
                return 'view or compare that product first so its raw facts are on record'
            if Facts.key(pid, sku) not in self.facts.inspected:
                return 'inspect_stock that exact variant first: its current price and stock decide whether it can be bought'
            if v.spec != 2 and self.request.gate_field:
                return 'that variant does not resolve to %s = %s from its own option text or raw facts' % (self.request.gate_field, self.request.gate_value)
            if not v.ok:
                return 'that product fails a firm requirement (category, brand, price, or the kind of device requested)'
            if self.request.family == 'preference' and len(self.facts.listings) < 2:
                return 'compare two products with different listing_id values before the first cart addition'
            if self.request.family == 'recovery' and (not self.facts.adds) and self.request.wanted_model and v.relax:
                return 'the first cart item must be the exact model and color the shopper named; the other colors are separate listings sharing the same title, so search that title and read them first'
            fam = self.request.family
            if not self.facts.adds and (fam == 'ranking' or (fam == 'preference' and Taste.profile(self.request.category)) or (fam == 'intent' and self.request.soft_brand)):
                target = self.planner.target()
                if target is None:
                    return 'hold: the first cart addition is graded, and the candidates it depends on are still being read'
                if (target.pid, target.sku) != (pid, sku):
                    return 'the first cart addition must be %s::%s, the best-verified candidate on record' % (target.pid, target.sku)
            if not self.planner.ready():
                return 'hold: read one more product that also meets every requirement first, so there is a fallback when the market moves right after the cart addition'
        if name == 'message':
            text = str(args.get('content') or '')
            if not text.strip():
                return 'message content is required'
            if not self.shopper.owes_answer and '?' not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper.owes_answer and self.shopper.questions_asked >= 2:
                return 'the shopper has already answered; decide from the evidence you have rather than asking again'
        if name == 'inspect_cart' and self.cart_inspections >= 1:
            return 'the cart contents are already known from earlier observations'
        key = _call_key(name, args)
        if key in self.facts.calls_made and name in ('search', 'filter', 'view', 'compare'):
            return 'identical call already made; change the arguments'
        self.facts.calls_made.add(key)
        return None

    def run(self) -> list:
        question = self.shopper.opening_question()
        last_move, repeats = (None, 0)
        while not self.session.done and self.session.turn < self.max_steps:
            if self.clock.remaining() < 25 or self.session.turn >= self.max_steps - 3:
                self._finish()
                break
            if question and (not self.shopper.asked_opening) and ('message' in self.tool_names):
                self.shopper.asked_opening = True
                self._execute([('message', {'content': question})])
                continue
            if self.pending_recheck and 'inspect_stock' in self.tool_names:
                self._recheck_after_add()
                continue
            if self.request.family == 'recall' and self.request.theme and (self.planner.suggested is None) and (len(self.planner.queries) == 1) and self.facts.rows:
                self.planner.suggested = self._suggest_expansions()
            move = self.planner.next()
            if move is not None:
                signature = json.dumps(move, sort_keys=True)
                repeats = repeats + 1 if signature == last_move else 1
                last_move = signature
                if repeats > 2:
                    for name, args in move:
                        if name in ('inspect_stock', 'add_to_cart'):
                            self.facts.dead.add(Facts.key(args.get('product_id'), args.get('sku')))
                        if name == 'compare':
                            self.planner.unreadable.update((str(p) for p in args.get('product_ids') or []))
                        if name == 'message':
                            self.shopper.answered()
                    last_move, repeats = (None, 0)
                    self.planner.note('That step changed nothing, so the controller moved on to other candidates.')
                    self._flush_notes()
                    continue
                if any((name == 'message' for name, _a in move)):
                    self.shopper.answered()
                self._execute(move[:self.max_calls])
                continue
            last_move, repeats = (None, 0)
            if self.max_steps < 25 and self.planner.reads_done and (self.facts.cart or any((v.ok for v in self.gate.candidates()))):
                self._finish()
                break
            self._flush_notes()
            proposal = self.model.propose(self.messages, self.tools, self.max_calls)
            if proposal is None:
                self.silent_rounds += 1
                self.planner.stuck = True
                if self.shopper.owes_answer:
                    self.shopper.answer_waits = 2
                forced = self.planner.read_more() if self.silent_rounds <= 12 else None
                if forced is not None and (not self.session.done):
                    self._execute([forced])
                    continue
                if self.shopper.owes_answer and self.silent_rounds <= 12:
                    continue
                self._finish()
                break
            call = proposal['tool_calls'][0]
            name = (call.get('function') or {}).get('name') or ''
            args = _arguments(call)
            content = proposal.get('content') or ''
            reprompts = 0
            while True:
                why = self._refuse(name, args)
                if why is None:
                    break
                if reprompts >= _REPROMPTS:
                    self.rejected_streak += 1
                    self.rejected_total += 1
                    self.planner.note('That call was not made: ' + why + '. Choose a different, valid action.')
                    if self.rejected_streak >= 3:
                        self.rejected_streak = 0
                        self.planner.stuck = True
                        self.planner.shortlist()
                        forced = self.planner.read_more()
                        if forced is not None and (not self.session.done):
                            self._execute([forced])
                        if self.rejected_total >= 9 and self.facts.cart:
                            self._finish()
                    self._flush_notes()
                    why = 'skip'
                    break
                reprompts += 1
                self.messages.append({'role': 'assistant', 'content': content, 'tool_calls': [call]})
                self.messages.append({'role': 'tool', 'tool_call_id': call.get('id') or 'x', 'content': json.dumps({'rejected': why})})
                self.dialogue.append({'role': 'assistant', 'content': '[rejected ' + name + '] ' + why[:200]})
                again = self.model.propose(self.messages, self.tools, self.max_calls)
                if again is None:
                    why = 'skip'
                    break
                call = again['tool_calls'][0]
                name = (call.get('function') or {}).get('name') or ''
                args = _arguments(call)
                content = again.get('content') or ''
            if why == 'skip':
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            calls = []
            if name in ('search', 'filter'):
                args = dict(args)
                args.setdefault('in_stock', True)
                args.setdefault('k', 10)
                if name == 'search' and args.get('query'):
                    self.planner.model_queries.append(str(args['query']))
            if name == 'place_test_order':
                pid, sku = (str(args.get('product_id') or ''), str(args.get('sku') or ''))
                if not pid and self.facts.cart:
                    pid, sku = self.facts.cart[-1]
                args = self.gate.order_args(pid, sku)
                note = self.planner.order_message(pid, sku)
                if note and 'message' in self.tool_names and (self.max_calls > 1):
                    calls.append(('message', {'content': note}))
                    self.shopper.answered()
            if name == 'message':
                if not self.shopper.owes_answer:
                    self.shopper.questions_asked += 1
                self.shopper.answered()
            calls.append((name, args))
            ids = [None] * (len(calls) - 1) + [call.get('id') or 'x']
            self._execute(calls, content, [i or 'agent-%d-%d' % (self.session.turn + 1, n + 1) for n, i in enumerate(ids)], from_model=True)
        return self.dialogue

def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:
        return [{'role': 'assistant', 'content': 'agent error: ' + type(exc).__name__ + ': ' + str(exc)[:200]}]
__all__ = ['agent_main']
