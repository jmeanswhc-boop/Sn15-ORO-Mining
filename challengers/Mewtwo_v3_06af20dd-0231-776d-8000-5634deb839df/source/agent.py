import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

TURN_CAP = 42
CALL_CAP = 60
SOFT_SPEND = 0.055
HARD_SPEND = 0.078
HEAVY_ROOM = 0.02
BRIEF_DEADLINE = 120.0
OUTAGE_PAUSE = 10.0
OUTAGE_CUTOFF = 150.0

RETRY_WAITS = (4, 10, 20, 35)
WAIT_BUDGET = 45.0

EPISODE_LEN = 300.0
EPISODE_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_DEADLINE = 165.0
LATE_ASK_DEADLINE = 190.0
WIND_DOWN = 200.0
HUNT_DEADLINE = 200.0
HUNT_GRACE = 25.0
QUESTION_DEADLINE = 210.0
LAST_TURN = 255.0
EMPTY_CART_ROOM = 12.0

STORE_PAUSE = 0.25
OPENING_PAUSE = 0.5
ROW_CLIP = 450
TEXT_CLIP = 3000
OBS_CAP = 15000
HISTORY_CAP = 150000
PAGE_ATTR_CLIP = 700
PAGE_REST_CLIP = 300
TITLE_LIMIT = 200
KEY_TABLE_SIZE = 18
CRASH_RETRIES = 3

OPEN_MIN_STEPS = 20
OPEN_MIN_ITEMS = 2
OPEN_LATER_STEPS = 2
OPEN_ROWS_PER_ITEM = 6
OPEN_WIDEN_SHARE = 5
OPEN_ROWS = 5

SHORTER_RESERVE = 8
WIDEN_PER_ITEM = 6
AUTO_PAGES = 6
HUNT_CAP = 24
HUNT_RESERVE = 10
HUNT_ROUNDS = 2
HUNT_IDLE = 4
HUNT_YIELD = 5
HUNT_WHOLE = 49
HUNT_VALUE_LEN = 30
HUNT_COMMON = 0.6
FIT_WORDS = 4
FIT_OPENS = 4
FIT_PAGES = 2
CHEAPER_SHOWN = 4
CHEAPER_OPENED = 3
VALUE_ROOM = 12
VALUE_RESERVE = 8
LIMIT_CHANCE = 0.05
WALK_RESERVE = 10
WALK_PER_LINE = 4

TURNS_BEFORE_ASK = 2
SCARCE_MESSAGES = 2
LIST_ONE = 12
LIST_MANY = 8
RAMP_ONE = (1, 3)
RAMP_MANY = (3, 3)
MIXED_LABELS = 3
LOOKALIKE_KEYS = 2
QUIET_WORD_LEN = 4
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34

GATE_ROUNDS = 3
TRAIL_ROOM = 15
EARLY_TRAIL_ROUNDS = 3
CHEAP_SETS = 2
SERIES_HUNTED = 3
SERIES_ROWS = 3
SERIES_BREADTH = 12
RECORD_MIN = 3
NOTE_MIN = 5
FIELD_SHARE = 0.15
PAIR_SHARE = 0.7
BLOCK_NAME_LEN = 60
BLOCK_VALUE_LEN = 40
TAGGED_LIMIT = 20000

PAIR_MARKS = (": ", " = ", "=", " -> ", " - ")
FIELD_MARKS = (" | ", "\n", " || ", " ; ", "; ", "\t", " // ", " · ", " • ", "|", ";", ", ")

_MODEL_CLIENT = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_SHORT_CLIENTS: dict = {}
_ENV_CLIENT = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

V1_MSG = re.compile(r"(?:^|_)(?:text|texts|body|message|messages|note|notes|content|contents|question|questions|msg|say)(?:_|$)")
V2_NONMSG = re.compile(r"(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$")
V3_OPEN_EXCL = re.compile(r"\b(?:not|no|other\s+than|unlike|at\s+most|at\s+least|under|over|below|above|less|more|between|within|up\s+to|or|same|listing)\b", re.I)
V4_SOLD_WHOLE = re.compile(r"^(?:no|none|false|0|n)$", re.I)
V4_SOLD_PART = re.compile(r"out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available", re.I)
V5_HOWTO = re.compile(r"\b(?:one|each|single|first|once|several|many|much|all|lot|too|list|separate\w*|split|short\w*|time|turn|question|questions|ask\w*)\b", re.I)
V6_CAP = re.compile(r"(?:^|_)(?:cap|max|maximum|ceiling|under|up_to|upto)(?:_|$)")
V7_NONPRICE = re.compile(r"result|row|page|count|size|num|item|step", re.I)
V8_PRICE = re.compile(r"price|cost|budget|spend|money|amount")
V9_LOWER = re.compile(r"(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)")
V10_SHELF = re.compile(r"section|department|categor|label|shelf|aisle|(?:^|(?<!age)_)group(?:_(?:name|label))?$", re.I)
V11_LISTMODE = re.compile(r"field|filter|facet|brows|categor|list|structured", re.I)
V12_AMOUNT = r"\d[\d.,]*\s*[A-Za-z\"']{1,12}"
V12_AMOUNT_RE = re.compile(r"^\s*" + V12_AMOUNT + r"\s*$")
V13_FLOOR = re.compile(r"\b(?:at\s+least|no\s+less\s+than|no\s+fewer\s+than|min|minimum)\b\s+(" + V12_AMOUNT + r")\s*$", re.I)
V14_QUERY = re.compile(r"(?:^|_)(?:q|query|search|text|phrase|keyword|keywords|term|terms|word|words)(?:_|$)|query|phrase|keyword", re.I)
V15_NONPROP = re.compile(r"^\W*(?:options?|variants?|models?|model\W+numbers?|part\W+numbers?|item\W+(?:numbers?|ids?|codes?)|(?:\w+\W+)?(?:ids?|skus?|asin|upc|ean|barcodes?)|codes?|warrant\w*(?:\W+\w+)?|url|links?|images?|photos?)\W*$", re.I)
V15_MODELS_KEEP = re.compile(r"[a-z0-9]+[ _]models?\b")
V16_NOTCATCH = re.compile(r"\b(?:type|types|style|styles|kind|kinds|shape|shapes)\b")
V16_CATCH = re.compile(r"\b(?:feature|features|function|functions|highlight|highlights|spec|specs|specification|specifications|description|descriptions|extra|extras|includ\w*|accessor\w*|misc\w*)\b")
V17_CODE = r"[A-Z][A-Za-z]{0,5}[ -]?(?:\d{3,6}|\d{1,5}[xX]{1,3})"
V17_RE = re.compile(r"\b" + V17_CODE + r"\b")
V17_ANYCASE = re.compile(r"[A-Za-z][A-Za-z]{0,5}[ -]?(?:\d{3,6}|\d{1,5}[xX]{1,3})\b")
V18_PART = r"[A-Za-z]{1,5}\d{0,3}-?\d{3,7}"
V19_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_ ./&'()\-]{0,38}$")
V20_BULLET = re.compile(r"^\s*(?:[-*•·]|\d{1,3}[.)])\s+", re.M)
V21_PAIRKEY = re.compile(r"name|key|label|attr|prop|field|spec|title|^k$", re.I)
V22_ID_A = re.compile(r"(?:^|[ _-])(?:id|idx|sku|pid|ref|asin|ident[A-Za-z0-9]*)(?:$|[ _-])", re.I)
V22_ID_B = re.compile(r"^[iI][dD][A-Z0-9]")
V22_ID_C = re.compile(r"Ident")
V22_ID_D = re.compile(r"(?<![A-Za-z])([A-Za-z]*id)$", re.I)
V22_ID_DX = re.compile(r"^(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$", re.I)
V23_FAIL = re.compile(r"error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid", re.I)
V24_PLACEHOLDER = re.compile(r"^\s*(?:none|null|n/a|unknown)\s*$", re.I)

A12_SEASON = re.compile(r"^[A-Z][a-z]+ (?:19|20)\d\d$")
A12_GAP = re.compile(r"[\s-]+")
A12_OWN = re.compile(r"[\[(]\s*(?:part\s*)?(" + V18_PART + r")\s*[\])]", re.I)
A12_MODEL = re.compile(r"\bmodel\s*(?:number|no\.?|#)?\s*[:.]?\s*(?!(?:no|number)\b)(" + r"[A-Za-z][A-Za-z]{0,5}[ -]?(?:\d{3,6}|\d{1,5}[xX]{1,3})" + r")\b", re.I)
A12_ALSO = re.compile(r"\b(?:also\s+sold\s+as|also\s+known\s+as|sold\s+also\s+as)\b([^.;]*)", re.I)
A12_WORKS = re.compile(r"(?<!not )(?<!never )\b(?:works\s+with|compatible\s+with|fits)\s*:?\s+([^.;]*)", re.I)
A12_WORKS_SKIP = re.compile(r"part\s*" + V18_PART + r"\s*also\s*$", re.I)
A12_CUT = re.compile(r"\b(?:except|excluding|but\s+not)\b", re.I)
A12_NOTE = re.compile(r"\bpart\s*(" + V18_PART + r")\s*also\s+fits\s*:?([^.;]*)", re.I)
A12_POINTER = re.compile(r"\b(?:more|details|see|read)\s+(?:on|at|under|about)?\s*(?:the\s+)?part\s+(" + V18_PART + r")", re.I)
A12_SERIES_OWN = re.compile(r"\b([A-Za-z]{1,5}\d{1,3})-\d{2,7}\b")
A12_SERIES_EXCL = re.compile(r"\b(?:except|excluding|but\s+not|not\s+for|not\s+compatible\s+with|does\s+not\s+fit|does\s+not\s+take)\b[^.;]*", re.I)
A12_SERIES_NAMED = re.compile(r"\b([A-Za-z]{1,5}\d{1,3})\b")


def find_codes(text):
    out = []
    for m in V17_RE.finditer(text or ""):
        c = m.group(0)
        if A12_SEASON.match(c):
            continue
        out.append(A12_GAP.sub(" ", c).strip().lower())
    return out


def code_covers(part_code, device_code):
    p = re.sub(r"[\s-]+", "", part_code or "").lower()
    d = re.sub(r"[\s-]+", "", device_code or "").lower()
    m = re.match(r"^(.*\d)(x+)$", p)
    if m:
        return d.startswith(m.group(1)) and len(d) == len(p)
    return p == d


def fit_marks(text):
    text = text or ""
    own = ""
    m = A12_OWN.search(text)
    if m:
        own = m.group(1).upper()
    models = []
    for m in A12_MODEL.finditer(text):
        models.extend(find_codes(m.group(1)))
    for m in A12_ALSO.finditer(text):
        models.extend(find_codes(m.group(1)))
    works = []
    barred = []
    for m in A12_WORKS.finditer(text):
        before = text[max(0, m.start() - 40):m.start()]
        if A12_WORKS_SKIP.search(before):
            continue
        seg = m.group(1)
        c = A12_CUT.search(seg)
        if c:
            works.extend(find_codes(seg[:c.start()]))
            barred.extend(find_codes(seg[c.end():]))
        else:
            works.extend(find_codes(seg))
    notes = []
    for m in A12_NOTE.finditer(text):
        notes.append((m.group(1).upper(), find_codes(m.group(2))))
    pointer = ""
    m = A12_POINTER.search(text)
    if m:
        pointer = m.group(1).upper()
    own_series = sorted(set(x.group(1).upper() for x in A12_SERIES_OWN.finditer(text)))
    rest = A12_SERIES_OWN.sub(" ", text)
    rest = A12_SERIES_EXCL.sub(" ", rest)
    named = sorted(set(x.group(1).upper() for x in A12_SERIES_NAMED.finditer(rest)))
    return {"own": own, "models": models, "works": works, "barred": barred, "notes": notes,
            "pointer": pointer, "own_series": own_series, "named_series": named}


H_BOOL = re.compile(r"^(?:true|false)$")
H_EMPTY = re.compile(r"^(?:none|null|)$")
H_MONEY = re.compile(r"^[$€£]?\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)$")
H_LEADZERO = re.compile(r"^0\d")
H_LOOSE = re.compile(r"['\"]?([A-Za-z][A-Za-z0-9_ ./\-]{0,38})['\"]?\s*[:=]\s*['\"]?([^,'\"}\]]*)")
H_BLOCK_CUT = re.compile(r";\s|;$|\s\|\s|\n")
H_NV_PART = re.compile(r"^[^=:]{1,40}=[^=]{0,200}$")
H_RULE_LINE = re.compile(r"^[\s|:+=\-]*$")
H_BLANK_CUT = re.compile(r"\n\s*\n")
H_TAG = re.compile(r"<[A-Za-z/][^<>]{0,40}>")
H_INLINE_TAG = re.compile(r"</?(?:b|i|em|strong|span|a|u|small)\b[^<>]{0,40}>", re.I)
H_CELL_TAG = re.compile(r"</t[dh]>\s*<t[dh][^<>]{0,40}>", re.I)
H_TAG_RUN = re.compile(r"\s*(?:<[^<>]{1,40}>\s*)+")
H_COL_GAP = re.compile(r" {2,}|\t")
H_TWIN_CUT = re.compile(r"(?<!\d),|,(?!\d)")
H_GLUE = re.compile(r"(\d)\s+([a-z]{1,3})(?![a-z0-9])")
H_RUNS = re.compile(r"[a-z0-9]+")
H_TOK1 = re.compile(r"[a-z0-9]{3,}")
H_TOK2 = re.compile(r"\d+[a-z]|\d*[a-z][a-z0-9]+")
H_TOK3 = re.compile(r"\d+")
H_FENCE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.S)
H_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
H_SEPS = re.compile(r"[\s_.\-]+")
H_ES_END = re.compile(r"(?:ch|sh|ss|x|z)es$")
H_STOP = re.compile(r"^(?:and|for|the|with)$")
H_OUR = re.compile(r"our$")
H_SIGNED = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")
H_COUNT = re.compile(r"^\s*(\d+)(?:\.0*)?\s*$")
H_DIGIT = re.compile(r"\d")
H_FILLER = "question"
H_TF = re.compile(r"^\s*(true|false)\s*$", re.I)

FIRST_SPELLING: dict = {}


def is_id_name(k):
    k = str(k)
    if V22_ID_A.search(k) or V22_ID_B.search(k) or V22_ID_C.search(k):
        return True
    m = V22_ID_D.search(k)
    if m and not V22_ID_DX.match(m.group(1)):
        return True
    return False


def fail_only(obs):
    if not isinstance(obs, dict) or not (1 <= len(obs) <= 3):
        return False
    for k, v in obs.items():
        if V23_FAIL.search(str(k)) and v:
            return True
    if len(obs) == 1:
        v = list(obs.values())[0]
        if isinstance(v, str) and 1 <= len(v) <= 40 and " " not in v:
            return True
    return False


def plain(s):
    t = s.strip()
    low = t.lower()
    if H_BOOL.match(low):
        return low == "true"
    if H_EMPTY.match(low):
        return None
    m = H_MONEY.match(t)
    if m and not H_LEADZERO.match(m.group(1)):
        num = m.group(1).replace(",", "")
        if "." in num:
            return float(num)
        return int(num)
    return t


def block_val(s):
    v = plain(s)
    if isinstance(v, bool) or (v is None and s.strip()):
        return s.strip()
    return v


def pair_under(text, mark):
    ix = text.find(mark)
    if ix >= 0:
        name, value = text[:ix], text[ix + len(mark):]
    else:
        bare = mark.strip()
        if bare and text.endswith(bare):
            name, value = text[:-len(bare)], ""
        else:
            return None
    name = V20_BULLET.sub("", name, count=1).strip()
    if not V19_FIELD.match(name) or len(name.split()) > 5:
        return None
    return name, value.strip()


def split_fields(text, sep, mark):
    parts = [p for p in text.split(sep) if p.strip()]
    if not parts or pair_under(parts[0].strip(), mark) is None:
        return [], 0
    fields = []
    named = 0
    heading = None
    for p in parts:
        s = p.strip()
        if sep == "\n" and heading is not None and (V20_BULLET.match(p) or p[:1] in (" ", "\t")):
            v = fields[heading][1]
            fields[heading][1] = (v + "\n" + s) if v else s
            continue
        heading = None
        if sep == "\n" and s.endswith(":") and V19_FIELD.match(s[:-1].strip()):
            fields.append([s[:-1].strip(), ""])
            heading = len(fields) - 1
            named += 1
            continue
        pr = pair_under(s, mark)
        if pr:
            fields.append([pr[0], pr[1]])
            named += 1
        elif fields:
            fields[-1][1] = fields[-1][1] + sep + s
    return fields, named


def attr_text(text):
    s = text.strip()
    if len(s) < 5:
        return None
    if s[0] in "{[" and s[-1] in "}]":
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else None
        except Exception:
            pass
        out = {}
        for m in H_LOOSE.finditer(s):
            out[m.group(1).strip()] = plain(m.group(2))
        return out if len(out) >= 2 else None
    if "=" in s:
        block = {}
        last = None
        for part in H_BLOCK_CUT.split(s):
            if not part.strip():
                continue
            if "=" in part:
                name, rest = part.split("=", 1)
                name = name.strip()
                if 1 <= len(name) <= 60 and name not in block:
                    block[name] = block_val(rest)
                    last = name
                    continue
            if last is not None:
                block[last] = str(block[last]) + "; " + part.strip()
            else:
                return None
        if len(block) >= 2:
            return block
        if len(block) == 1:
            n = list(block)[0]
            if ":" not in n and len(n) <= 30 and len(s) <= 90:
                return block
        return None
    lines = [l for l in s.split("\n") if l.strip()]
    if len(lines) >= 2 and all(V20_BULLET.match(l) for l in lines):
        prs = [pair_under(l.strip(), ": ") for l in lines]
        if all(prs):
            return {p[0]: plain(p[1]) for p in prs}
    for sep in (", ", "; "):
        if sep in s and ": " in s:
            parts = [p.strip() for p in s.split(sep)]
            prs = [pair_under(p, ": ") for p in parts]
            if len(parts) >= 2 and all(prs) and not any("." in p[0] for p in prs) and all(len(p[1]) <= 40 for p in prs):
                return {p[0]: plain(p[1]) for p in prs}
    return None


def rec_text(text):
    kept = None
    for sep in FIELD_MARKS:
        if sep not in text:
            continue
        pieces = len([p for p in text.split(sep) if p.strip()])
        best = None
        for mark in PAIR_MARKS:
            fields, named = split_fields(text, sep, mark)
            share = PAIR_SHARE if sep == ", " else FIELD_SHARE
            if named < RECORD_MIN or named < share * pieces:
                continue
            if len(set(f[0] for f in fields)) * 2 <= len(fields):
                continue
            if best is None or named > best[1]:
                best = (fields, named)
        if best is not None:
            kept = best[0]
            break
    if kept is None:
        return None
    out = {}
    for name, val in kept:
        if name in out:
            continue
        a = attr_text(val)
        if a is not None:
            out[name] = a
        elif is_id_name(name):
            out[name] = val.strip()
        else:
            out[name] = plain(val)
    return out


def table_text(text):
    lines = [l for l in text.split("\n") if l.strip() and not H_RULE_LINE.match(l)]
    lines = [l.strip().strip("|").strip() for l in lines]
    if len(lines) < 2:
        return None
    for sep in ("|", "\t", ";", ","):
        cells = [[c.strip() for c in l.split(sep)] for l in lines]
        head = cells[0]
        if len(head) < 3 or any(len(c) != len(head) for c in cells):
            continue
        if not all(V19_FIELD.match(c) for c in head):
            continue
        if any(re.match(r"^[\d.,$]+$", c) for c in head):
            continue
        return [{h: plain(v) for h, v in zip(head, row)} for row in cells[1:]]
    return None


def records_from_text(text):
    s = text.strip()
    if len(s) < 12:
        return None
    if s[0] in "{[":
        try:
            return restore(json.loads(s), True, 0, False)
        except Exception:
            pass
    for parts in (H_BLANK_CUT.split(s), s.split("\n")):
        if len(parts) < 2:
            continue
        recs = [rec_text(p) for p in parts]
        good = [r for r in recs if r]
        if len(good) >= 2 and len(good) * 5 >= len(parts) * 4 and len(set(list(r)[0] for r in good)) == 1:
            return good
    found = rec_text(s)
    if found is None and "\n" in s:
        found = table_text(s)
    if found is None and len(s) <= TAGGED_LIMIT and H_TAG.search(s):
        t = H_INLINE_TAG.sub("", s)
        t = H_CELL_TAG.sub(": ", t)
        t = H_TAG_RUN.sub("\n", t).strip()
        if t != s and len(t) >= 12:
            return records_from_text(t)
    if found is None and "\n" in s:
        t = "\n".join(H_COL_GAP.sub(": ", l.strip(), count=1) for l in s.split("\n") if l.strip())
        if t != s:
            return rec_text(t)
    return found


def block_text(text):
    if len(text) < 5 or "=" not in text or not ("; " in text or " | " in text or "\n" in text):
        return text
    parts = [p for p in H_BLOCK_CUT.split(text.strip()) if p.strip()]
    if len(parts) < 2 or not all(H_NV_PART.match(p.strip()) for p in parts):
        return text
    r = attr_text(text)
    return r if r is not None else text


def attr_list(lst, allowed):
    if len(lst) < 2:
        return None
    if isinstance(lst[0], dict):
        keys = list(lst[0].keys())
        if len(keys) != 2 or not V21_PAIRKEY.search(keys[0]):
            return None
        out = {}
        for e in lst:
            if not isinstance(e, dict) or set(e.keys()) != set(keys):
                return None
            n, v = e[keys[0]], e[keys[1]]
            if not isinstance(n, str) or not (1 <= len(n) <= 60) or isinstance(v, (dict, list)):
                return None
            out[n.strip()] = v
        return out
    if isinstance(lst[0], list):
        out = {}
        for e in lst:
            if not isinstance(e, list) or len(e) != 2:
                return None
            if not isinstance(e[0], str) or not (1 <= len(e[0]) <= 60) or isinstance(e[1], (dict, list)):
                return None
            out[e[0].strip()] = e[1]
        return out
    if allowed and all(isinstance(e, str) for e in lst):
        out = {}
        for e in lst:
            pr = pair_under(e.strip(), ": ") or pair_under(e.strip(), "=")
            if not pr or len(pr[1]) > 40 or len(pr[0].split()) > 3:
                return None
            out[pr[0]] = plain(pr[1])
        return out
    return None


def column_rows(obj):
    for hk, hv in obj.items():
        if not (isinstance(hv, list) and len(hv) >= 2 and all(isinstance(x, str) and V19_FIELD.match(x) for x in hv)):
            continue
        for rk, rv in obj.items():
            if rk == hk:
                continue
            if isinstance(rv, list) and rv and all(isinstance(r, list) and len(r) == len(hv) for r in rv):
                out = {}
                for k, v in obj.items():
                    if k == hk:
                        continue
                    out[k] = [dict(zip(hv, r)) for r in rv] if k == rk else v
                return out
    return obj


def id_texts(obj):
    if not any(isinstance(v, str) and " " in v for v in obj.values()):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(v, int) and not isinstance(v, bool) and is_id_name(k) and abs(v) >= 100:
            out[k] = str(v)
        else:
            out[k] = v
    return out


def keyed_rows(obj):
    if len(obj) < 2:
        return None
    if not all(isinstance(v, dict) and len(v) >= 3 for v in obj.values()):
        return None
    if not all(H_DIGIT.search(str(k)) and " " not in str(k) for k in obj):
        return None
    shared = None
    for v in obj.values():
        shared = set(v) if shared is None else shared & set(v)
    if len(shared) < 3:
        return None
    out = []
    for k, v in obj.items():
        if holds_id(v):
            out.append(v)
        else:
            r = {"id": str(k)}
            r.update(v)
            out.append(r)
    return out


def holds_id(obj):
    return isinstance(obj, dict) and any(is_id_name(k) for k in obj)


def flat_block(obj):
    if not any(isinstance(v, dict) for v in obj.values()):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            for ik, iv in v.items():
                if not isinstance(iv, (dict, list)):
                    out[str(k) + " " + str(ik)] = iv
        else:
            out[k] = v
    return out


def id_like(v, named=False):
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not (3 <= len(s) <= 64) or " " in s:
        return False
    return True if named else bool(H_DIGIT.search(s))


def field_rule(v, depth, inside, page):
    r = restore(v, page, depth + 1, inside)
    if isinstance(v, str) and isinstance(r, dict) and len(r) < 5 and not holds_id(r) \
            and not any(id_like(x) for x in r.values()):
        return v
    return r


def restore(v, page=True, depth=0, inside=False):
    if depth > 7:
        return v
    if isinstance(v, str):
        r = records_from_text(v)
        return v if r is None else r
    if isinstance(v, list):
        a = attr_list(v, inside and page)
        if a is not None:
            return a
        return [field_rule(x, depth, False, page) for x in v]
    if isinstance(v, dict):
        first_text = bool(v) and isinstance(list(v.values())[0], str)
        o = id_texts(column_rows(v))
        carded = len(o) >= 3 or holds_id(o)
        o = dict(o)
        for k, val in list(o.items()):
            if isinstance(val, dict):
                kr = keyed_rows(val)
                if kr is not None:
                    o[k] = kr
        listing = holds_id(o) or any(id_like(x) for x in o.values())
        new = {}
        for k, val in o.items():
            if not isinstance(val, str) or len(o) <= 2:
                new[k] = field_rule(val, depth, carded, page)
            else:
                new[k] = block_text(val) if listing else val
        if depth <= 3 and len(new) >= 3:
            for k, val in list(new.items()):
                if not isinstance(val, dict) or not val:
                    continue
                inner = list(val.values())
                if not any(isinstance(x, dict) for x in inner) or any(isinstance(x, list) for x in inner):
                    continue
                f = inner[0]
                if not (isinstance(f, (dict, str, int, float)) and f != {}):
                    continue
                if holds_id(val):
                    continue
                new[k] = flat_block(val)
        if len(new) == 1:
            only = list(new.values())[0]
            if isinstance(only, dict) and len(only) >= 3 and first_text:
                return only
        return new
    return v


def join_twin(held, new):
    got = set(p.strip().lower() for p in H_TWIN_CUT.split(held))
    extras = []
    for p in H_TWIN_CUT.split(new):
        q = p.strip()
        if q and q.lower() not in got:
            got.add(q.lower())
            extras.append(q)
    if not extras:
        return held
    joined = held + (", " if held.strip() else "") + ", ".join(extras)
    FIRST_SPELLING[joined] = FIRST_SPELLING.get(held, held)
    return joined


def fold_twins(v, depth=0):
    if depth > 5:
        return v
    if isinstance(v, list):
        return [fold_twins(x, depth + 1) for x in v]
    if not isinstance(v, dict):
        return v
    out = {}
    names = {}
    for k, val in v.items():
        n = word_form(str(k))
        fv = fold_twins(val, depth + 1)
        if n not in names:
            names[n] = k
            out[k] = fv
            continue
        kk = names[n]
        if fv == out[kk]:
            continue
        if isinstance(fv, str) and isinstance(out[kk], str) and not is_id_name(n):
            out[kk] = join_twin(out[kk], fv)
        else:
            out[k] = fv
    if len(out) < 2 and len(v) >= 2:
        return {k: fold_twins(val, depth + 1) for k, val in v.items()}
    return out


def cjson(v):
    try:
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(v)


def clip_deep(v, n, depth=0):
    if depth > 7:
        return "..."
    if isinstance(v, str):
        return v[:n] + "..." if len(v) > n else v
    if isinstance(v, dict):
        return {k: clip_deep(x, n, depth + 1) for k, x in v.items()}
    if isinstance(v, list):
        return [clip_deep(x, n, depth + 1) for x in v]
    return v


def clip(s, n):
    s = s if isinstance(s, str) else str(s)
    return s[:n]


def json_obj(text):
    if not isinstance(text, str):
        return None
    t = text.strip()
    m = H_FENCE.match(t)
    if m:
        t = m.group(1)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try:
            v = json.loads(t[a:b + 1])
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    return None


def word_form(s):
    s = H_CAMEL.sub(r"\1 \2", str(s))
    return H_SEPS.sub(" ", s).strip().lower()


def under_form(s):
    return word_form(s).replace(" ", "_")


def stems(s):
    return set(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in word_form(s).split())


def roots(s):
    out = set()
    for w in word_form(s).split():
        if H_ES_END.search(w):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if len(w) >= 3 and not H_STOP.match(w):
            out.add(w)
    return out


def spelled(s):
    out = set()
    for w in word_form(s).split():
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(H_OUR.sub("or", w))
    return out


def names_test(p, k):
    if not p or not k:
        return False
    if p == k or p <= k:
        return True
    return len(k) >= 2 and k <= p


def glued(s):
    return H_GLUE.sub(r"\1\2", str(s).lower())


def text_words(s):
    s = str(s)
    return set(H_RUNS.findall(s.lower())) | set(H_RUNS.findall(glued(s)))


def value_tokens(v):
    g = glued(str(v))
    for pat in (H_TOK1, H_TOK2, H_TOK3):
        t = pat.findall(g)
        if t:
            return t
    return []


def kinds(schema):
    t = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def kind_fit(v, k):
    if k == "string":
        return isinstance(v, str)
    if k == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if k == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if k == "boolean":
        return isinstance(v, bool)
    if k == "array":
        return isinstance(v, list)
    if k == "object":
        return isinstance(v, dict)
    if k == "null":
        return False
    return True


def shape(v, schema, path, faults):
    if not isinstance(schema, dict):
        return v
    ks = kinds(schema)
    if ks and not any(kind_fit(v, k) for k in ks):
        for k in ks:
            if k == "integer" and isinstance(v, float) and v == int(v):
                v = int(v)
                break
            if k in ("integer", "number") and isinstance(v, str) and H_SIGNED.match(v):
                f = float(v.strip())
                v = int(f) if (k == "integer" or f == int(f)) else f
                break
            if k == "string" and isinstance(v, (int, float)) and not isinstance(v, bool):
                v = str(v)
                break
            if k == "boolean" and isinstance(v, str) and H_TF.match(v):
                v = v.strip().lower() == "true"
                break
            if k == "array" and v is not None and not isinstance(v, list):
                v = [v]
                break
        if not any(kind_fit(v, k) for k in ks):
            faults.append("%s should be %s" % (path, "/".join(ks)))
            return v
    en = schema.get("enum")
    if isinstance(en, list) and en and v not in en:
        if isinstance(v, str):
            low = v.strip().lower()
            for o in en:
                if isinstance(o, str) and o.lower() == low:
                    v = o
                    break
        if v not in en:
            faults.append("%s must be one of %s" % (path, cjson(en)))
    if isinstance(v, str):
        ml = schema.get("maxLength")
        if isinstance(ml, int) and not isinstance(ml, bool) and len(v) > ml:
            v = v[:ml]
        mn = schema.get("minLength")
        if isinstance(mn, int) and not isinstance(mn, bool) and len(v) < mn:
            faults.append("%s is too short" % path)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if isinstance(lo, (int, float)) and not isinstance(lo, bool) and v < lo:
            faults.append("%s is below the minimum %s" % (path, lo))
        if isinstance(hi, (int, float)) and not isinstance(hi, bool) and v > hi:
            faults.append("%s exceeds the maximum %s" % (path, hi))
    if isinstance(v, list):
        mx, mi = schema.get("maxItems"), schema.get("minItems")
        if isinstance(mx, int) and not isinstance(mx, bool) and len(v) > mx:
            faults.append("%s allows at most %d entries" % (path, mx))
        if isinstance(mi, int) and not isinstance(mi, bool) and len(v) < mi:
            faults.append("%s needs at least %d entries" % (path, mi))
        it = schema.get("items")
        if isinstance(it, dict):
            v = [shape(x, it, "%s[%d]" % (path, i), faults) for i, x in enumerate(v)]
    if isinstance(v, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        if schema.get("additionalProperties") is False:
            for k in v:
                if k not in props:
                    faults.append("%s.%s is not allowed by the schema" % (path, k))
        req = schema.get("required")
        if isinstance(req, list):
            for k in req:
                if k not in v:
                    faults.append("%s.%s is required" % (path, k))
        out = {}
        for k, x in v.items():
            out[k] = shape(x, props[k], path + "." + k, faults) if k in props else x
        v = out
    return v


def shape_args(schema, args):
    faults = []
    out = shape(args, schema, "args", faults)
    return out, faults


def items_of(v):
    if isinstance(v, list):
        return v
    if (isinstance(v, str) and v) or (isinstance(v, dict) and v):
        return [v]
    return []


def count_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v == int(v) else None
    if isinstance(v, str):
        m = H_COUNT.match(v)
        if m:
            return int(m.group(1))
    return None


def learn_id(rows):
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return None
    best, score = None, None
    for k in rows[0]:
        named = is_id_name(k)
        vals = [r.get(k) for r in rows]
        if not all(id_like(x, named) for x in vals):
            continue
        st = [x.strip() for x in vals]
        if len(rows) >= 2 and len(set(st)) < len(st):
            continue
        sc = (1 if named else 0, sum(len(x) for x in st) / float(len(st)))
        if score is None or sc > score:
            best, score = k, sc
    return best


def row_has(ident, v, depth=0, inlist=False):
    if depth > 5:
        return False
    if isinstance(v, list):
        return any(row_has(ident, x, depth + 1, True) for x in v)
    if isinstance(v, dict):
        for x in v.values():
            if inlist and isinstance(x, str) and x.strip() == ident:
                return True
            if isinstance(x, (dict, list)) and row_has(ident, x, depth + 1, False):
                return True
    return False


def model_client(room):
    if room >= 50:
        return _MODEL_CLIENT
    t = max(5, int(room // 5) * 5)
    if t not in _SHORT_CLIENTS:
        _SHORT_CLIENTS[t] = ProxyClient(timeout=t, max_retries=1, retry_delay=1)
    return _SHORT_CLIENTS[t]


def attr_block(rec):
    if not isinstance(rec, dict):
        return None
    best, n = None, 1
    for v in rec.values():
        if isinstance(v, dict) and len(v) >= 2:
            sc = {k: x for k, x in v.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)}
            if len(sc) > n:
                best, n = sc, len(sc)
    return best


def soft(fallback):
    def wrap(fn):
        def inner(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception as e:
                self.trace.add({"kind": "soft_crash", "method": fn.__name__,
                                "error": clip(type(e).__name__ + ": " + str(e), 300)})
                return fallback() if callable(fallback) else fallback
        inner.__name__ = fn.__name__
        return inner
    return wrap


class Trace:
    def __init__(self, t0):
        self.t0 = t0
        self.items = []

    def add(self, e):
        if isinstance(e, dict) and "t" not in e:
            e["t"] = round(time.monotonic() - self.t0, 1)
        self.items.append(e)


def filler(schema):
    if not isinstance(schema, dict):
        return None
    en = schema.get("enum")
    if isinstance(en, list) and en:
        return en[0]
    if "const" in schema:
        return schema["const"]
    if "default" in schema:
        return schema["default"]
    ks = kinds(schema)
    if "integer" in ks or "number" in ks:
        mn = schema.get("minimum")
        return mn if isinstance(mn, (int, float)) and not isinstance(mn, bool) else 1
    if "boolean" in ks:
        return False
    if "string" in ks:
        ml = schema.get("maxLength")
        return H_FILLER[:ml if isinstance(ml, int) and ml else 8]
    return None


def strings_of(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        out = []
        for x in v.values():
            out.extend(strings_of(x))
        return out
    if isinstance(v, list):
        out = []
        for x in v:
            out.extend(strings_of(x))
        return out
    return []


def says(v, text):
    return any(s == text for s in strings_of(v))


def make_item(name, shelf, qty, musts):
    return {"name": name, "shelf": shelf, "qty": qty, "musts": list(musts), "told": [], "dropped": [], "free": []}


GUIDE = """You are Cairn, a careful purchaser who serves one single shopper inside a web shop. Your only way to act is to call one of the store's tools. The episode is graded all or nothing: it scores one only when the finish is exactly correct, and anything else scores zero. A correct purchase must hold every requested item, each in the quantity asked, each satisfying every requirement still in force at the moment of checkout (changes that arrived later count), and it must cost no more than the budget. There is a single other correct finish: when the shop truly holds nothing that satisfies everything, you must instead close the episode through the ending tool, in the manner the request specifies.

HOW THE STORE OPERATES
- Every call to a tool spends one step, and steps are finite; thinking costs nothing. Every store has its own tool names, parameter names and result layouts, so you must study the schema and the description of each tool and pass parameter names and permitted values exactly as given. Never guess a parameter name or a value.
- A search can be narrow: it may filter by category, a keyword mode may read only the first words of a query, pages may hold only a handful of rows, and the next page may need a token or an offset. Compare the total the store reports with the number of rows you have seen; if the answer hinges on seeing every row, fetch the further pages, passing back unchanged whatever token or offset came with the result. Keep keyword queries brief and lead with the value: the wanted value first, then the product noun. A store's search may match attribute values and seller names as well as titles, and may demand that every query word be present, so a few required values together with the noun can return almost the whole fitting set. A search by a part number or a model code can come back empty; a single attempt shows whether this store's search finds codes at all. When a filtered search returns zero rows, the usual cause is a wrong filter value or a wrong filter parameter: try again with the label copied letter for letter as the rows of a result show it, or switch to another filter parameter.
- Rows in search results are summaries only. Shops divide a listing's record across views differently: one view may expose the attribute block (attributes and properties are the same thing here), another only the offer and a description, and either kind may appear in the side-by-side compare view. Once Cairn has learned which tool exposes attributes, the ledger says so. Each property requirement is judged on those attributes, so you must inspect the attributes of every listing you might purchase. One call that displays several listings at once saves steps, provided it displays what is required. Never reopen a listing whose state has not changed since you read it.
- Asking the shopper is the job of Cairn's interviewer: you never send the shopper a message yourself. The replies appear in your conversation, and the ledger records them.
- A request may also demand intermediate steps: obtaining an approval, sending a message, verifying or reading something beforehand. Each appears in the ledger. Those due right ahead of the order or of a no-purchase ending are carried out by Cairn itself, at the moment you call checkout or the ending tool, so never hold off waiting for them and never decline on their account. The remaining steps, those not due at the finish, you perform yourself with the shop's tools; questions about products are the exception and go through the interviewer.

WHAT COUNTS AS FITTING
- The category label of a listing settles its kind of product: only a label that matches the requested item in its entirety qualifies; an adjacent label never does.
- A property requirement is satisfied when the attribute the listing has for that property gives the value. How the key is written does not matter: spacing, underscore separators, prefixes or camel-cased joins. When the attribute is a list split by commas or slashes, one matching element suffices. Match word by word; differences of letter case, of singular versus plural, and ordinary synonyms are tolerated. Containment counts: a value that includes all the words of the required value meets it despite extra words, so a shade word in front of a colour still gives that colour, and a fit word with an intensifier before it and a noun after it still gives that fit. A value made of other words fails: an abbreviation or a code does not equal the word it abbreviates, and two colour names that share one word but differ in the other (a dark green against a sea green) are different values. For numbers, convert the units first, then compare; a limit includes its boundary, and a value given as a range must sit entirely within the limit.
- An attribute keyed by the requirement's property (that exact name or an ordinary synonym of it) is decisive: if it states another value, the listing fails, whatever the title or any other key says. Exception: when a key puts the item's name in front of the property name (a bag listing whose key means the bag's shape), its value counts even if the plain property key says otherwise. A key about another property, even a closely related one, settles nothing for this requirement: the colour of a handle tells nothing about the colour of the body. After listings are opened, Cairn appends [evidence] lines that name, for each requirement, the attribute keys whose name matches it.
- When a listing has no attribute at all for the property, the value sought still counts when it appears under some other attribute (a styles list or a features list that contains it), or in its description or title. It does not count when the figure is a different kind of quantity (weight is no substitute for capacity), or when a bare number appears with no text saying what it measures.
- When the required value is written into the category label itself (a season or a target audience inside the label), the category satisfies it. Conversely, a listing filed under the label that was asked for counts as a product of the requested kind, even when its description or title mentions something different; judge such a listing only on its requirements, its price and its stock.
- A requirement may refer to another listing instead of giving its value, for example "the X of the <property> sold by <shop>..." or "within N units of the value shown on that listing's page". Resolve such a requirement first: locate the referenced listing (search on its words, brand, shop or category) and open its whole page with the single-item tool or the compare tool; a row from a search does not suffice. If the opened page sends you onward to one more listing that holds the property, locate and open that listing by the same method, and so on: the final page in the chain supplies the value. If the requirement wants whatever goes with the referenced value (its partner or its counterpart), and that page says what its own value is paired with, then that paired value becomes the requirement. An order is valid only if every page along the chain was opened. If the requirement accepts anything other than the referenced listing's value, it removes only the listings that share exactly that value, and that rule by itself must never cause a decline.
- If a listing you intend to buy now points to a further page with more detail about itself, check that page before the purchase.
- If one requested item has to suit another, as an accessory suits a device, pick both jointly, based on the listings' own wording: a compatibility list, or a series or model code that appears on both pages. If no accessory meeting its own requirements is compatible with the chosen device, change to a different device that satisfies its own requirements and that one of those accessories does fit. Never decline while that device switch has not been tried.
- Exclusions must be honoured, and so must closed lists. Wishes (softly worded preferences such as ideally or nice to have) serve only to break ties.
- Sold-out listings must never be bought. The total (each line's price times its quantity, added over all lines) must not go over the budget; landing exactly on it is allowed.

CHOOSING
- If the request wants the cheapest choice (the smallest price or the lowest total), the order must contain the lowest-priced listing out of ALL fitting listings. Establish the complete fitting set by paging through it (rows often arrive unsorted by price, and Cairn fetches further pages of short result lists on its own), examine the candidates from the cheapest upward, and pick the earliest one that fits fully. If a search tool accepts a maximum price, use it: list that category with the cap set below what your present best pick costs, so that only the cheaper ones remain to be checked.
- If the request wants the greatest or the smallest amount of some measure, compare every fitting listing on that measure and take the extreme one; a listing silent on the measure is never the winner.
- Otherwise any listing that satisfies everything is correct: do not burn steps looking for a better one.
- Put each item in the cart the moment an opened listing satisfies all of that item's requirements (for a price-ranked or measure-ranked pick, only after the fitting set has been compared), then go on to the next item. A store often sells out a line shortly after it enters the cart, and carting early keeps steps in hand for a replacement.
- When several distinct items are requested, each needs its own cart line, every line at that item's requested quantity.

CHANGES DURING THE WORK
- Both the shopper and the store may change the task: a new budget, an added or removed requirement, a new price, a listing gone out of stock. Such changes show up inside tool results, often on a checkout attempt in the form of a refusal. A note counts whichever tool's result carries it. The ledger shows the requirements currently in force. A refusal does not end the episode: apply all the changes, correct the cart and try checkout again.
- Prices and stock may also shift without any note. When you call checkout, Cairn re-checks the cart lines and views the cart itself, so do not spend a turn on either.

FINISHING
- Make the cart contain only the chosen lines, exactly those, removing everything else; then send checkout as a call by itself, never together with other calls in one turn.
- A no-purchase ending is allowed only once a genuine search has found that no listing satisfies everything. Being low on steps proves no such thing: in that situation, check out with the best cart you hold. The ending follows the request's own wording; when a remark has to go with it, keep the remark brief and name no listing.
- Every reply must include one or more tool calls. Keep your text brief: the known facts, then the next action."""


BRIEF_PROMPT = """You set up a buying job that an agent will carry out in a web store. You get the shopper's text and the list of the store's tools. Reply with EXACTLY ONE JSON object, compact and unindented, with no text before or after it, holding these keys:

"items": a list with one entry for each distinct product to be bought. Each entry is an object with four fields: "name": what sort of product it is, worded as the request words it; "shelf": if some tool exposes the store's category labels, the matching label copied exactly; else a category the request gives the product (a product wanted "in" a named department takes that department's name); else ""; "quantity": the number of units to buy of that single listing, an integer, 1 when no number is stated; pairs and sets that come as one listing are quantity 1; "musts": the list of the product's firm requirements, each written as property: constraint, stating the precise value, bound, excluded value or permitted set of values.
Name the property inside a must with the very word the request used for it: wording that the product is intended for some kind of occasion gives the occasion property with that kind as its value; a measure placed before a part of the product (a twelve-inch blade) gives that part's length, with number and unit. Never trade the property for a neighbouring one.
A requirement whose value has to come from a different listing is written as ONE must that carries the other listing's full description, for example "color: different from the color of the backpack sold by Shop Z with a laptop sleeve". All the words that describe the other listing (its shop, its product kind and any trailing "with" clause of properties and values) stay inside that single must; none of them ever becomes a must of this product.
Price, budget, stock and quantity never go into musts. Narrative around the purchase (the intended user, the place the product goes, earlier trouble, the occasion that prompted the purchase) is context and never a must, unless it explicitly gives some property's value or bound. Words that are part of the product's own name, such as a season or a target group mentioned with the product, stay in "name" and never become musts.
Exception: when the request treats a purpose or an event as a property of the product, that IS a requirement on the very property the request turns the purpose into (for example "occasion: business"), unless the store's category label already names it. This covers something said to be for a kind of wear, something said to be made for a kind of event, a product followed by a separate clause stating it is for an event, and a plain statement that the occasion must be a given value. In contrast these stay context: one specific person's own event (wearing it at their own wedding), the recipient or the place the purchase is meant for (a present for a co-worker), and how it should sit, feel or look in use.
When one phrase names products joined by "and" (a single "I want" clause with a single quantity, often the title of one department of the store), that is ONE item. Split into separate items only if every product receives its own quantity or its own requirements.

"budget": the shopper's ceiling for the whole spend, as a number in whatever unit the request uses; null if absent. "budget_unit": that unit written out (for example cents); null if absent. "budget_scope": "total" if the cap applies to the whole order, "each" if it applies per item, null if there is no cap.
"in_stock_only": true by default; false only when the request plainly allows unavailable listings.
"pick": "cheapest" when the request wants the minimum price or the least total cost; "most" or "least" when it wants the maximum or the minimum of some measure; "any" otherwise. Saying that money is short does not make a pick rule. "pick_measure": for "most" or "least", the quantity pushed to its extreme; otherwise "".
"messages": the number of shopper messages the request permits the agent, an integer; null if unstated. "steps": the request's allowance of steps (it may call them moves or tool calls), an integer, else null.

"lookups": exact copies of the musts whose value must first be fetched from a listing the request refers to; [] if none.
"rules": every instruction in the request about the working method or the way to finish, each at most 12 words. "wishes": nice-to-haves, preferences the shopper voices without requiring them. "ask_first": whatever the request explicitly wants confirmed with the shopper before choosing; [] if none.

"process": duties the request imposes on top of locating and buying the right items (getting the shopper's approval, telling or showing the shopper something, confirming a point with the shopper, reading something before acting, using a specific tool). Each entry is an object: "step": the duty in at most 15 words; "when": exactly one of "choose" (before any product is chosen), "order" (just before checking out), "end" (just before finishing with no purchase); "messages": the number of shopper messages the duty uses, 0 if none. [] if the request imposes no such duty. These are NOT duties: item requirements, budget, stock, the message and step limits, and what to do if nothing fits.
"quiet_decline": true if the request forbids suggesting or naming alternatives when nothing fits; false if it wants alternatives; null if it is silent.
"fits": one entry for every requested item that, by the request, has to work with, fit or suit some OTHER requested item (whichever listing the agent picks for that other item), as an object {"item": the dependent item's number, "with": the other item's number}; [] if none.

"roles": an object mapping each of nine jobs to the exact tool name, or null when no tool does it: "search"; "open": shows the properties or attributes of a single listing (of two tools that each show one listing, choose the one whose description offers specifications, properties or attributes); "offer": shows one listing's live offer (such as price and stock), only if a tool other than the "open" one does that; "compare": shows several listings in full; "ask": messages the shopper; "cart": changes the cart; "cart_view": shows the cart; "buy": performs the checkout; "refuse": finishes with no purchase.
"search_words": an object {"tool": the tool that takes free-text words to search listings (if one tool browses a category and another takes words, choose the word-taking one), "field": the parameter of that tool that receives the words, "args": constant arguments needed to switch it to word search (a mode tool needs its mode value here), else {}}.
"buy_args": constant arguments the buy tool needs to perform the checkout (a mode tool needs its mode value here), else {}. "refuse_args": constant arguments the refuse tool needs to finish with no purchase, else {}.
"""

PLAN_PROMPT = """You pick which questions an agent will put to the person it shops for. Every shopper keeps a small number of firm, unwritten preferences, and each such preference is about a single property of a single requested product. Asking reveals a preference only if the question names exactly that property. Where the shopper holds no opinion the answer is "don't mind"; where the property never crossed the shopper's mind the answer is puzzled. So each property you put into a message spends a share of that message.
You receive the request, the items with the requirements the request already fixes, and a numbered table of the attribute keys found on opened listings in each item's category, giving for each key how many of those listings carry it and some sample values. Beware: the unwritten preference may concern a property that none of the listed keys covers.

Choose in three tiers, in this order.
Tier 0: anything the request wants checked with the shopper (shown under ASK FIRST, if present), each named after the property involved.
Tier 1, hinted properties: a request's story often hints at an unwritten preference while leaving its value unsaid. Look for sentences about a matter the shopper cannot judge or leaves to the agent; about a person whose taste decides; about how the product looks or the surroundings it belongs in; about the intended wearer or user; about its contents. For each such sentence give the single property it hints at, with its item, in the story's own order. Whenever a key listed for the same item matches the hint, use that key: a hint about the wearer maps to the gender or audience key; a hint about the product's looks maps to whichever colour, finish or pattern key is carried by the largest share of that item's listings. A hinted property that no listing of this item carries ranks below that item's three most widely carried keys, so never above fourth.
Tier 2, after the hinted ones: the listed keys in rank order. Prefer the properties that shoppers for that sort of product decide on earliest (commonly the type or style, the form or fit, the dimensions, size or capacity, the material, the pattern or colour: whichever that sort of product is chosen by), and keys present on most of that item's listings. Once at least five listings have been read, steer clear of keys found on only one or two listings.

Skip, for each item: any property the request has already settled for that item, that is, one a requirement already names (a key whose values merely echo a required value, such as a pattern key whose values show the decoration the request demands, is NOT settled and stays askable); a further name for a property already chosen; a key in which each value bundles several purchase options together; identifiers; model names; warranty and any other logistics. Do not skip pack sizes, per-pack quantities or counts: they are ordinary properties. A key marked [not the stated ...] is a different property from the requirement it mentions, however many words the two names share.
Rank every item separately, its hinted properties before its listed keys.

"label" holds the plain name of the property. From a listed key it takes every word of the key in plain spelling: a key whose words run together with a capital at each join, or are joined by underscores, is written as separate lower-case words, and it is never cut down to its last word (the label for a lid's colour is "lid colour", never just "colour"; for a rim's finish it is "rim finish", never only "finish"). Only the name: no sample values, no bracketed text, never a second property.
Pick at most COUNT properties per item, strongest first. Reply with ONE JSON object only: {"ask": [{"item": item number, "label": "property name", "hint": "at most eight words of the story that led to this pick, or empty"}]}, with each item's picks in rank order."""

REPLY_PROMPT = """An agent has put some questions to a shopper. Read every answer and reply with ONE JSON object only, nothing besides it, with five keys:
"needs": a list of {"item": the item's number, "need": "property: constraint", "points": true if the value has to be taken from a different listing, else false};
"drop": a list of {"item": number, "property": a requirement the shopper has now given up};
"free": a list of {"item": number, "property": the property the shopper does not mind about};
"crowded": a list of the question numbers whose answer objects that the message covered too many subjects, or wants the subjects asked one at a time (not answers that merely say the question was not understood or ask for plainer wording);
"budget": a revised overall ceiling as a number, else null.
"needs" covers every firm preference any answer expresses, with its exact value or bound and its unit (for example "length: at least 30 cm", "shape: round", "material: not plastic"). Name the need after the property the answer itself mentions, even when that differs from the property the question was about. Several answers on one property give one need. Copy the value word for word from the answer, nouns and units included ("shelves: at least 4 shelves", not only the number). Keep only what a listing could state: start at the value wanted (or at the bound word before it), end where the value ends, and leave out the answer's framing ("matching navy" gives "navy"; "it has to be oak, that's settled" gives "oak"; "steel and nothing else" gives "steel"; "equal to at least 2 kg" gives "at least 2 kg").
An answer may give the value by pointing at another listing: a value like, unlike or close to the value of a product from some shop, the product identified by a few of its properties. That makes ONE single need for the property the answer names, with "points": true. Its value is the exact text of the answer from the comparison word through to where that description ends, together with anything said about a listing that page might name; never shorten or reword it. Properties that describe the other product are never needs of this item.
"free": for each question, the property the shopper declared indifference to.
Folksy or indirect phrasing that still names a value does state that value. Phrasing that refuses to negotiate a value (it is not to be traded away, given up or altered) states that value as wanted; write a need as "not" a value only when the answer turns that very value down."""


NOTE_PROMPT = """While an agent was working in a web store, the store displayed the notice below; its author may be the store itself or the shopper. Reply with ONE JSON object, nothing besides it, with these keys:
"kind": "change" if the notice alters the job at hand (a budget, requirements, some price, availability, or refusing the order); "error" if it only says some call was badly formed or turned down; "other" in every other case;
"budget": the new limit on total spend as a figure in the unit the request used, else null;
"add": a list of new requirements, each {"item": an item number, "need": "property: constraint"};
"drop": a list of requirements no longer wanted, each {"item": an item number, "need": the wording of the withdrawn requirement};
"gone": a list of the ids of listings the notice calls unavailable;
"sold_out": true if the notice says a picked listing has vanished, sold out or is no longer obtainable, even when no id is named;
"price_moved": true if it reports some price moving;
"summary": one brief sentence.
A requirement replaced by another must appear twice: as a drop of the old wording and as an add of the new one. Report every change the notice contains, not only the first."""

CHECK_PROMPT = """You are the final gate ahead of checkout. From the records below, decide whether this cart is exactly right.
Reading rules, for a requirement R on property P with value V: R holds when the listing's field for P states V; the key may be spelled any way, a single member of a list suffices, capitalisation, singular or plural and simple synonyms are all fine, a value that contains V's words counts (a shade word in front of the wanted colour still gives that colour), units are converted, a limit includes its own boundary, and a range has to fall wholly inside. A field whose key names P itself decides: if it states something other than V, the listing fails; but a value filed under a key that puts the item's own kind in front of P (on a backpack, a key for the backpack's colour) still counts even though the bare key for P disagrees. A key for another property, however related, decides nothing (a strap's colour says nothing about the main body's colour). Where no field for P exists at all, V stated under some other field, or in the title or description, is enough, except a figure of a different sort (a weight is no capacity) and a lone figure whose text never says what it measures. If the category label itself states R, being in that category satisfies R. The category label has to match the whole requested item, not just part of it. A requirement to differ from another listing's value is taken as met, except where the records reveal that very value.
Cart rules: every requested item must have its own cart line carrying the quantity asked; the cart must hold nothing beyond those lines; no line may be shown as unavailable through a stock-out; and each line's price times its quantity, added up, must give a total no higher than the budget.
When there are several records of one listing, the newest governs.
If other listings are given below and the pick rule is cheapest, most or least, weigh them too: if one plainly satisfies all requirements and beats the cart's choice under that rule, this cart is wrong.
Reply with ONE JSON object only, nothing besides it: {"ok": true or false, "problems": [short, concrete faults], "better": [ids of listings to put in place of some cart line, strongest first]}. Set "ok" to false only for a fault the records plainly show, or when some requirement is addressed by no record."""

READ_RULES = """for a requirement R on property P with value V: a field of the listing settles P whenever its key K names P; K may be spelled any way, one member of a list suffices, synonyms are fine, units may be converted, a value that contains V's words is fine (a shade word in front of the wanted colour still gives that colour), and a limit includes its own boundary; a value filed under a key that puts the item's own kind in front of P (on a backpack, a key for the backpack's colour) still counts even though the bare key for P disagrees; a key for another property, however related, decides nothing (a strap's colour says nothing about the main body's colour); with no field naming P, V stated under some other field, or in the title or description, is enough, but a lone figure that never says what it measures is worth nothing; if the category label states R, that category meets R; the category label has to match the item; a requirement to differ from another listing's value is taken as met, except where the records reveal that very value; a listing that is out of stock, or that goes over the budget, is not a fit."""

AUDIT_PROMPT = ("""An agent means to stop without buying, believing that nothing fits. Check the records below against the requirements, using these reading rules: """
                + READ_RULES + """
A ROW record comes only from a search row: nobody has read its page (its fields), so on any property the row itself does not show, it may be judged neither a pass nor a fail.
Reply with ONE JSON object only: {"fits": {item number: [ids that seem to satisfy all requirements, lowest price first, at most 4]}, "open": [ids of ROW listings of a requested kind, within budget, whose page might prove a fit, likeliest first, at most 6], "reason": "one sentence"}.""")

UNMIX_PROMPT = """Someone turned a shopping request into firm requirements for one product. Some requirements take their value from a different listing the request refers to, and the words that pick out that listing (the kind of product it is, the shop it comes from, the properties a "with" phrase after it names) are at times mistaken for requirements on the product being bought. Going by the request's own phrasing, say which entries of the list headed OTHER REQUIREMENTS below do nothing but describe the referenced listing; the requirements that point to the other listing are context only, not candidates. Reply with ONE JSON object only: {"describe_other": [such requirements, each copied word for word]}, an empty list when all of them are genuine requirements of the purchase."""

CHAIN_PROMPT = """An agent is near checkout. Some requirements draw their value off a second listing instead of stating it, and one listing's page may refer the reader onward to yet another listing for that same property. An order only counts once the agent has fully opened each listing along every such chain.
Trace the chain of every requirement, starting at its own description, link after link, through what the pages themselves say, using the records after this text. OPENED means the agent opened that page; ROW means the page is unread and the listing is known only from a search row or through its fields.
A record counts as a link only if it fits the link's entire description: the shop or brand, the product kind AND each property value the description lists. A description of a multipack holding some count excludes a listing from that shop with the same count of another unit and no multipack. If no record fits all of it, the link is unlocated, even when that shop carries further listings of the same kind. For an unlocated link, give a handful of search words: the shop's name plus the values in the description.
Give the links of every chain in sequence, and end that list at the earliest link that is unlocated or lacks an OPENED page, because what that page says is not yet known.
Reply with ONE JSON object only: {"chains": [{"need": "the requirement, copied word for word", "links": [{"id": "the listing's id if some record is this link, else empty", "about": "how this link is described, by the requirement or by the preceding page", "find": "search words, only where the link is not yet located"}], "done": true if the final link is OPENED and its page itself states the property's value (true as well when the requirement wants a value that differs from that one or is unlike it, because the value on that page is the yardstick), "value": "the property's value as the final page states it; if the requirement wants whatever pairs with that value and the page names its partner, that partner; else empty"}]}."""

GATE_PROMPT = """The agent will shortly STAGE. The request wants the steps listed below performed on the way. Using the RECORD (the numbered lines recording the agent's actions), decide for each step owed ahead of this STAGE whether it has been completed for the cart in its current state.
A step is complete only if some RECORD line records its performance; for a step concerning the cart or the order, that same line must also come later than the latest cart change, and the reply recorded on that same line must grant it. Questions about a product's properties never perform a step.
For a step still missing, name the single store call that performs it now, choosing a tool from the TOOLS list and supplying its arguments. Shopper messages use the messaging tool and consist of one brief sentence or question about that step alone, plainly worded, mentioning no product property or value beyond what the step needs. Give no call if all owed steps are complete, or if the remaining messages or steps cannot achieve the missing one.
Reply with ONE JSON object only: {"steps": [{"step": "the step", "done": true or false, "lines": [numbers of the RECORD lines that prove it]}], "call": {"tool": "name", "args": {}} or null, "hold": "a change that some reply asks for before the agent goes on, else empty", "needs": [{"item": an item number, "need": "property: constraint"} for each firm requirement such a reply states]}."""

SWAP_PROMPT = ("""A listing in the agent's cart sold out a moment ago. Pick substitutes for it from the other records shown, using these reading rules: """
               + READ_RULES + """
Reply with ONE JSON object only: {"fits": [{"id": "a listing id", "price": its price as a number, the way the record states it}] covering every listing that seems to satisfy all requirements of the lost line's item, at most 6, "reason": "one sentence"}.""")

SHAPE_PROMPT = """A store tool produced the output shown after this text. Label its parts, using the exact keys found in that result and in the tool's parameters. Reply with ONE JSON object only: {"rows": the key where the array of listings sits, "count": the key holding the total number of matches (not the number displayed) or null, "next": the key holding the value that requests the following page or null, "next_arg": the tool parameter that accepts that value or null, "price": the key inside a row holding the listing's price or null, "stock": the key inside a row telling whether it can be bought right now or null}."""

SPLIT_PROMPT_A = "The store tool "
SPLIT_PROMPT_B = (" can either complete the purchase or close the episode with no purchase, according to the arguments it"
                  " is given.")
SPLIT_PROMPT_C = ("Reply with a single JSON object only, holding two fields: \"buy_args\", the constant arguments that make"
                  " this tool buy, and \"refuse_args\", the constant arguments that make it finish with no purchase."
                  " Include in both only the arguments that select the mode.")


E1_FINISH409 = re.compile(r"finish|quarantin|finaliz")
E1_EXPECTED = re.compile(r"expected (\d+)")
E1_REUSE = re.compile(r"reuse|different action")
E1_POINTER_WORDS = re.compile(r"\b(?:listing|page|shown|other\s+than|unlike|same\s+as)\b", re.I)
E1_WORD_RUNS = re.compile(r"[a-z0-9]+")
E1_PLAIN_BUDGET = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")
E1_ANTHROPIC = re.compile(r"^anthropic/")


class EpisodeCore:
    def __init__(self, problem_data):
        env = problem_data["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        q = self.policy.get("query")
        self.request = q if isinstance(q, str) and q else ""
        self.tools = []
        self.schemas = {}
        for t in self.policy.get("tools") or []:
            if not isinstance(t, dict):
                continue
            fn = t.get("function")
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                continue
            self.tools.append(t)
            ps = fn.get("parameters")
            self.schemas[fn["name"]] = ps if isinstance(ps, dict) else {"type": "object", "properties": {}}
        ms = self.policy.get("max_steps")
        budget = count_int(ms)
        if budget is None and isinstance(ms, float) and not isinstance(ms, bool):
            try:
                budget = int(ms)
            except Exception:
                budget = None
        if not budget or budget <= 0:
            budget = 30
        self.max_steps = budget
        self.t0 = time.monotonic()
        self.steps = 0
        self.finished = bool(self.policy.get("finished"))
        self.store_broke = False
        self.no_credit = False
        self.waited = 0.0
        self.misses = 0
        self.spend = 0.0
        self.calls = 0
        self.turns = 0
        self.trace = Trace(self.t0)
        self.gap = STORE_PAUSE
        self.store_secs = 0.0
        self.store_calls = 0
        self.last_store = 0.0
        self.convo = []
        self.brief = {}
        self.roles = {}
        self.items = []
        self.budget = None
        self.msg_limit = 0
        self.msgs_used = 0
        self.duties = []
        self.msg_reserve = 0
        self.gate_msgs = 0
        self.gate_passes = set()
        self.done_log = []
        self.done_steps = []
        self.asked = False
        self.question_log = []
        self.heard = set()
        self.changes = []
        self.version = 0
        self.queued = []
        self.rows = {}
        self.full = {}
        self.seen = []
        self.opened = set()
        self.views = {}
        self.facts = {}
        self.bare = {}
        self.factless = set()
        self.id_key = None
        self.row_keys = set()
        self.tool_keys = {}
        self.view_keys = {}
        self.shapes = {}
        self.price_key = None
        self.stock_key = None
        self.read_cache = {}
        self.pages_read = 0
        self.paged = []
        self.mem_results = {}
        self.mem_args = {}
        self.page_arg = {}
        self.page_queries = {}
        self.searched = []
        self.value_rows = set()
        self.cmp_ok = 0
        self.cmp_fail = 0
        self.code_tries = []
        self.codes_dead = False
        self.cart_touched = False
        self.cart_changed_at = -1
        self.cart_seen_at = -1
        self.cart_view = None
        self.cart = {}
        self.cart_said = ""
        self.id_args = set()
        self.built = []
        self.dropped = set()
        self.checkouts = 0
        self.refused_at = -1
        self.refusals = 0
        self.recent = []
        self.sold_holds = 0
        self.line_holds = 0
        self.fit_holds = 0
        self.checked = set()
        self.verdicts = {}
        self.audited = False
        self.leads = []
        self.lead_items = {}
        self.decline_blocks = 0
        self.pushed = False
        self.late_held = False
        self.confirmed = None
        self.completed = False
        self.offer_seen = {}
        self.gone = {}
        self.unavailable = set()
        self.sold_lines = set()
        self.note_sold = False
        self.note_named = False
        self.refs = []
        self.ref_values = {}
        self.chain_rounds = 0
        self.chain_holds = 0
        self.chain_room = None
        self.decline_chain_done = False
        self.link_tried = set()
        self.link_searches = 0
        self.link_seen = {}
        self.fit_pairs = []
        self.fit_cache = None
        self.owners_seen = {}
        self.adopted = {}
        self.plan_order = []
        self.lanes = {}
        self.lane_found = {}
        self.lane_spent = {}
        self.lane_used = {}
        self.lane_misses = {}
        self.late_asked = False
        self.one_product = set()
        self.list_asks = 0
        self.apart = {}
        self.follow_ups = []
        self.listing = True
        self.reply_reads = 0
        self.capped = set()
        self.cheaper_marks = set()
        self.wider_held = False
        self.hunts = 0
        self.hunt_mark = ""
        self.hunt_tried = set()
        self.hunt_stems = {}
        self.hunt_nonoun = set()
        self.fit_hunted = False
        self.facts_order = []
        self.title_key_cache = None
        self.lookalike_done = False
        self.bare_tools = {}
        self.cart_seen_version = -1

    def left(self):
        return self.max_steps - self.steps

    def clock(self):
        return time.monotonic() - self.t0

    def room(self):
        return EPISODE_LEN - EPISODE_MARGIN - self.clock()

    def rushed(self):
        pace = (self.store_secs / self.store_calls) if self.store_calls else 1.0
        return self.room() <= 2 * max(pace, 1.0)

    def msgs_free(self):
        return self.msg_limit - self.msgs_used - max(0, self.msg_reserve - self.gate_msgs)

    def cart_mark(self):
        return cjson(sorted([[k, v] for k, v in self.cart.items()], key=lambda p: str(p[0])))

    def gate_due(self, stage):
        if not any(d["stage"] == stage for d in self.duties):
            return False
        return (stage, self.cart_mark() + "#" + str(self.version)) not in self.gate_passes

    def reserve(self):
        if (any(d["stage"] == "order" and d["messages"] > 0 for d in self.duties)
                and self.msgs_used < self.msg_limit and self.gate_due("order")):
            return 2
        return 1

    def judges(self, limit=SOFT_SPEND):
        if self.spend <= limit and self.spend + HEAVY_ROOM <= HARD_SPEND:
            return [HEAVY, LIGHT]
        if self.spend <= HARD_SPEND:
            return [LIGHT]
        return []

    def page_tools(self):
        out = []
        if self.roles.get("offer"):
            out.append(self.roles["offer"])
        elif self.roles.get("open"):
            out.append(self.roles["open"])
        if self.roles.get("compare"):
            out.append(self.roles["compare"])
        return out

    def facts_tool(self):
        for t in self.facts_order:
            if self.facts.get(t) is True and self.single_target(t):
                return t
        op = self.roles.get("open")
        if op and self.facts.get(op) is not False and self.single_target(op):
            return op
        return None

    def compare_facts(self):
        c = self.roles.get("compare")
        if not c:
            return None
        return self.facts.get(c)

    def compare_bare(self):
        cf = self.compare_facts()
        if cf is False:
            return True
        if cf is None:
            ft = self.facts_tool()
            if (ft and self.facts.get(ft) is True) or self.roles.get("offer"):
                return True
        return False

    def map_row_ids(self, tool, args):
        if tool not in [self.roles.get(r) for r in ("open", "offer", "compare", "cart") if self.roles.get(r)]:
            return args
        if not self.rows:
            return args
        tk = self.title_key()
        if not tk:
            return args
        props = (self.schemas.get(tool) or {}).get("properties") or {}

        def fix(name, v):
            if not isinstance(v, str):
                return v
            s = v.strip()
            if s in self.rows or not (3 <= len(s) <= 30) or not re.search(r"\d", s):
                return v
            ps = props.get(name) if isinstance(props, dict) else None
            if isinstance(ps, dict) and (ps.get("enum") or (isinstance(ps.get("items"), dict) and ps["items"].get("enum"))):
                return v
            pat = re.compile(r"(?<![a-z0-9])" + re.escape(s.lower()) + r"(?![a-z0-9])")
            hits = [i for i, r in self.rows.items() if isinstance(r, dict) and pat.search(str(r.get(tk, "")).lower())]
            if len(hits) == 1:
                self.trace.add({"kind": "row_id", "value": s, "id": hits[0]})
                return hits[0]
            return v
        out = {}
        for k, v in args.items():
            if isinstance(v, list):
                out[k] = [fix(k, x) for x in v]
            else:
                out[k] = fix(k, v)
        return out

    def think(self, messages, models, cap, tools=None):
        for model in models:
            if self.calls >= CALL_CAP:
                return None
            if self.room() < CALL_FLOOR:
                self.trace.add({"kind": "late_skip", "model": model})
                return None
            self.calls += 1
            tries = 0
            while True:
                msg, status = self.one_call(messages, model, cap, tools)
                if msg is not None:
                    return msg
                retryable = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if status == 0:
                    retryable = False
                if not retryable:
                    break
                most = 1 if (status is None or status == 408) else len(RETRY_WAITS)
                if tries >= most:
                    break
                wait = RETRY_WAITS[tries]
                if self.waited + wait > WAIT_BUDGET:
                    break
                if self.room() - 1.2 * wait < CALL_FLOOR:
                    break
                time.sleep(wait * random.uniform(0.8, 1.2))
                self.waited += wait
                tries += 1
            if status == 402:
                self.no_credit = True
                return None
        return None

    def one_call(self, messages, model, cap, tools=None):
        msgs = messages
        if E1_ANTHROPIC.match(model) and len(messages) >= 2:
            msgs = [dict(m) for m in messages]
            for ix in (0, len(msgs) - 1):
                c = msgs[ix].get("content")
                if isinstance(c, str) and c:
                    msgs[ix]["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
        body = {"model": model, "messages": msgs, "max_tokens": cap, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            res = model_client(self.room()).post_verbose("/inference/chat/completions", body)
        except Exception as e:
            self.trace.add({"kind": "model_fail", "model": model, "status": None, "body": clip(str(e), 200)})
            return None, None
        data = res.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage")
            if isinstance(usage, dict):
                cost = usage.get("cost")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    self.spend += cost
            try:
                msg = data["choices"][0].get("message") or {}
            except Exception:
                msg = {}
            if msg.get("content") or msg.get("tool_calls"):
                return msg, 200
            self.trace.add({"kind": "model_empty", "model": model})
            return None, 0
        err = res.error or {}
        status = err.get("status")
        body_text = err.get("body")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = count_int(data["error"].get("code"))
            body_text = data["error"].get("message")
        self.trace.add({"kind": "model_fail", "model": model, "status": status, "body": clip(str(body_text), 200)})
        return None, status

    def ask_json(self, prompt, models, cap=1800):
        for model in models:
            msg = self.think([{"role": "user", "content": prompt}], [model], cap)
            if msg is not None:
                obj = json_obj(msg.get("content") or "")
                if obj is not None:
                    return obj
            if self.store_broke or self.no_credit or self.calls >= CALL_CAP:
                return None
        return None

    def env_call(self, tool, args):
        body = dict(self.binding)
        body["step"] = self.steps + 1
        body["action"] = {"name": tool, "args": args}
        since = time.monotonic() - self.last_store
        if since < self.gap:
            time.sleep(self.gap - since)
        self.last_store = time.monotonic()
        for _ in range(4):
            t = time.monotonic()
            try:
                res = _ENV_CLIENT.post_verbose("/environment/call", body)
            except Exception as e:
                self.store_secs += time.monotonic() - t
                self.store_calls += 1
                self.trace.add({"kind": "env_fail", "status": None, "body": clip(str(e), 200)})
                continue
            self.store_secs += time.monotonic() - t
            self.store_calls += 1
            data = res.data
            if isinstance(data, dict):
                st = data.get("step")
                if isinstance(st, int) and not isinstance(st, bool):
                    self.steps = st
                else:
                    self.steps += 1
                if data.get("finished") is True:
                    self.finished = True
                obs = data.get("observation")
                if tool != self.roles.get("ask"):
                    obs = self.unspell(tool, obs)
                    data["observation"] = obs
                self.note_read(tool, args, obs)
                ft = self.facts_tool()
                if tool == ft and isinstance(obs, dict) and not fail_only(obs):
                    ident = self.listing_id(obs)
                    if ident is not None and not attr_block(obs):
                        self.factless.add((tool, ident))
                if tool in self.page_tools():
                    bad = isinstance(obs, dict) and obs.get("error")
                    if not bad and not fail_only(obs):
                        cj = None
                        for v in args.values():
                            vals = []
                            if isinstance(v, str):
                                vals = [v.strip()]
                            elif isinstance(v, list):
                                if cj is None:
                                    cj = cjson(obs)
                                vals = [x.strip() for x in v if isinstance(x, str) and x.strip() in cj]
                            for x in vals:
                                self.opened.add(x)
                                self.views.setdefault(x, set()).add(tool)
                return data
            err = res.error or {}
            status = err.get("status")
            btxt = err.get("body")
            btxt = btxt if isinstance(btxt, str) else cjson(btxt)
            self.trace.add({"kind": "env_fail", "status": status, "body": clip(btxt, 200)})
            if status == 409:
                low = btxt.lower()
                if E1_FINISH409.search(low):
                    self.finished = True
                    return None
                m = E1_EXPECTED.search(btxt)
                if m:
                    n = int(m.group(1))
                    self.steps = n - 1
                    body["step"] = n
                    continue
                if E1_REUSE.search(low):
                    self.steps += 1
                    body["step"] = self.steps + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.store_broke = True
                return None
        return None

    def unspell(self, tool, obs):
        views = [self.roles.get(r) for r in ("open", "offer", "compare") if self.roles.get(r)]
        page = tool != self.roles.get("search") or tool in views
        obs = fold_twins(restore(obs, page))
        if tool not in views:
            return obs
        for i, r in self.rows.items():
            if i not in self.opened and isinstance(r, dict):
                self.row_keys.update(r.keys())
        if not self.row_keys:
            return obs

        def gather(rec, plain):
            if not isinstance(rec, dict) or attr_block(rec):
                return rec
            if self.listing_id(rec, not plain, 0 if plain else 1) is None:
                return rec
            extras = {}
            for k, v in rec.items():
                if k not in self.row_keys and isinstance(v, (str, int, float)) and not isinstance(v, bool):
                    extras[k] = v
            if len(extras) < 3:
                return rec
            out = {k: v for k, v in rec.items() if k not in extras}
            out["attributes"] = extras
            return out
        if isinstance(obs, dict):
            if self.listing_id(obs) is not None:
                return gather(obs, True)
            out = {}
            for k, v in obs.items():
                if isinstance(v, list):
                    out[k] = [gather(x, False) for x in v]
                else:
                    out[k] = v
            return out
        return obs

    def read_brief(self):
        lines = []
        for t in self.tools:
            fn = t["function"]
            lines.append("- " + fn["name"] + ": " + str(fn.get("description") or "") + " | parameters: "
                         + cjson(fn.get("parameters") if fn.get("parameters") is not None else {}))
        prompt = BRIEF_PROMPT + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.policy.get("opening") is not None:
            prompt += "\n\nSTORE OPENING NOTE:\n" + cjson(clip_deep(self.policy.get("opening"), 1500))
        b = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit:
            if not self.brief_ok(b):
                b2 = self.ask_json(prompt, [HEAVY, LIGHT], 2500)
                if b2:
                    b = b2
            elif self.shared_shelf(b):
                b2 = self.ask_json(prompt, [HEAVY], 2500)
                if b2 is not None and self.brief_ok(b2):
                    b = b2
        while not b and not self.no_credit and self.calls < CALL_CAP and self.clock() < BRIEF_DEADLINE:
            time.sleep(OUTAGE_PAUSE)
            b = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        if b:
            self.waited = 0.0
        self.brief = b
        roles = b.get("roles")
        names = set(self.schemas)
        if isinstance(roles, dict):
            self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in names}
        for k in ("buy_args", "refuse_args"):
            if not isinstance(b.get(k), dict):
                b[k] = {}
        buy, ref = self.roles.get("buy"), self.roles.get("refuse")
        if buy and buy == ref and (not (b["buy_args"] and b["refuse_args"]) or b["buy_args"] == b["refuse_args"]):
            p = (SPLIT_PROMPT_A + buy + SPLIT_PROMPT_B + "\nSCHEMA: " + cjson(self.schemas.get(buy) or {})
                 + "\n" + SPLIT_PROMPT_C)
            sp = self.ask_json(p, [LIGHT, HEAVY], 1500)
            if isinstance(sp, dict) and isinstance(sp.get("buy_args"), dict) and isinstance(sp.get("refuse_args"), dict) \
                    and sp["buy_args"] != sp["refuse_args"]:
                b["buy_args"] = sp["buy_args"]
                b["refuse_args"] = sp["refuse_args"]
        raw = b.get("items")
        if isinstance(raw, (dict, str)):
            raw = [raw]
        raw = raw if isinstance(raw, list) else []
        items = []
        for e in raw:
            if not isinstance(e, dict):
                continue
            q = count_int(e.get("quantity"))
            q = q if q and q > 0 else 1
            ms = e.get("musts")
            ms = ms if isinstance(ms, list) else ([ms] if ms is not None else [])
            ms = [str(x) for x in ms if x]
            name = str(e["name"]) if e.get("name") is not None else "the product"
            shelf = str(e["shelf"]) if e.get("shelf") is not None else ""
            items.append(make_item(name, shelf, q, ms))
        merged = []
        renum = {}
        for oi, it in enumerate(items, 1):
            target = None
            lab = word_form(it["shelf"])
            if lab:
                for mi, m in enumerate(merged, 1):
                    if m["qty"] != it["qty"] or word_form(m["shelf"]) != lab:
                        continue
                    if self.conflict(m["musts"], it["musts"]):
                        continue
                    target = mi
                    break
            if target is None:
                merged.append(it)
                renum[oi] = len(merged)
            else:
                m = merged[target - 1]
                for r in it["musts"]:
                    if r not in m["musts"]:
                        m["musts"].append(r)
                renum[oi] = target
                self.trace.add({"kind": "merge", "item": oi, "into": target})
        if not merged:
            merged.append(make_item("the requested product", "", 1, []))
        self.items = merged
        fits = b.get("fits")
        fits = fits if isinstance(fits, list) else ([fits] if isinstance(fits, dict) else [])
        for f in fits:
            if not isinstance(f, dict):
                continue
            a, w = count_int(f.get("item")), count_int(f.get("with"))
            if a not in renum or w not in renum:
                continue
            pa, pw = renum[a], renum[w]
            if pa != pw and 1 <= pa <= len(self.items) and 1 <= pw <= len(self.items) and (pa, pw) not in self.fit_pairs:
                self.fit_pairs.append((pa, pw))
        looks = b.get("lookups")
        looks = looks if isinstance(looks, list) else ([looks] if looks else [])
        for lk in looks:
            t = str(lk).strip() if lk is not None else ""
            if not t:
                continue
            if not any(t in it["musts"] for it in self.items):
                pp = word_form(t.split(":", 1)[0])
                for it in self.items:
                    hit = next((r for r in it["musts"] if word_form(r.split(":", 1)[0]) == pp), None)
                    if hit:
                        t = hit
                        break
            n = next((ix for ix, it in enumerate(self.items, 1) if t in it["musts"]), 1)
            if t and (n, t) not in self.refs:
                self.refs.append((n, t))
        self.unmix()
        bud = b.get("budget")
        if isinstance(bud, str) and E1_PLAIN_BUDGET.match(bud):
            bud = float(bud.strip())
        if isinstance(bud, (int, float)) and not isinstance(bud, bool) and bud > 0:
            self.budget = bud
        ml = count_int(b.get("messages"))
        if ml is not None and ml >= 0:
            self.msg_limit = ml
        elif self.roles.get("ask"):
            self.msg_limit = 3
        sl = count_int(b.get("steps"))
        if sl is not None and sl >= 10 and sl < self.max_steps:
            self.max_steps = sl
        proc = b.get("process")
        proc = proc if isinstance(proc, list) else ([proc] if isinstance(proc, dict) else [])
        for d in proc:
            if not isinstance(d, dict):
                continue
            st = d.get("step")
            if st is None or not str(st).strip():
                continue
            when = str(d.get("when") or "").strip().lower()
            when = when if when in ("choose", "order", "end") else "order"
            mc = count_int(d.get("messages"))
            mc = max(0, mc) if mc is not None else 0
            self.duties.append({"text": str(st).strip(), "stage": when, "messages": mc})
        if self.roles.get("ask"):
            counts = [d["messages"] for d in self.duties if d["stage"] in ("order", "end") and d["messages"] > 0]
            self.msg_reserve = min(max(counts), self.msg_limit) if counts else 0
        self.trace.add({"kind": "brief", "items": len(self.items), "roles": self.roles, "budget": self.budget,
                        "messages": self.msg_limit, "steps": self.max_steps, "duties": len(self.duties)})

    def brief_ok(self, b):
        if not isinstance(b, dict) or not b.get("items"):
            return False
        r = b.get("roles")
        return isinstance(r, dict) and bool(r.get("buy"))

    def shared_shelf(self, b):
        raw = b.get("items")
        if isinstance(raw, (dict, str)):
            raw = [raw]
        if not isinstance(raw, list):
            return False
        seen = set()
        for e in raw:
            if isinstance(e, dict):
                s = word_form(str(e.get("shelf") or ""))
                if s:
                    if s in seen:
                        return True
                    seen.add(s)
        return False

    def conflict(self, a, b):
        for x in a:
            for y in b:
                if word_form(x.split(":", 1)[0]) == word_form(y.split(":", 1)[0]):
                    vx = word_form(x.split(":", 1)[1]) if ":" in x else ""
                    vy = word_form(y.split(":", 1)[1]) if ":" in y else ""
                    if vx != vy:
                        return True
        return False

    def unmix(self):
        for ix, it in enumerate(self.items, 1):
            ptrs = [t for (n, t) in self.refs if n == ix and t in it["musts"]]
            rest = [r for r in it["musts"] if r not in ptrs]
            if not ptrs or not rest:
                continue
            prompt = (UNMIX_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nPRODUCT: " + it["name"]
                      + "\nTHESE LINES REFER TO A DIFFERENT LISTING:\n" + "\n".join(ptrs)
                      + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            ans = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
            lst = ans.get("describe_other") if isinstance(ans, dict) else None
            lst = lst if isinstance(lst, list) else []
            texts = set(x.strip() for x in lst if isinstance(x, str))
            moved = [r for r in rest if r in texts]
            if not moved:
                continue
            first = ptrs[0]
            new = first + " (that listing: " + "; ".join(moved) + ")"
            it["musts"] = [new if r == first else r for r in it["musts"] if r not in moved]
            self.refs = [(n, new) if (n == ix and t == first) else (n, t) for (n, t) in self.refs]
            self.trace.add({"kind": "unmix", "item": ix, "moved": moved})

    def described(self, ix, req):
        if any(req == t for (_, t) in self.refs):
            return False
        looks = self.brief.get("lookups")
        looks = looks if isinstance(looks, list) else ([looks] if looks else [])
        if any(req == str(x) for x in looks):
            return False
        if ":" not in req:
            return False
        prop, val = req.split(":", 1)
        if not prop.strip():
            return False
        if len(E1_WORD_RUNS.findall(val.lower())) < 4:
            return False
        if E1_POINTER_WORDS.search(val):
            return False
        vst = set(stems(val))
        for j, other in enumerate(self.items, 1):
            if j == ix:
                continue
            if vst & set(s for s in stems(other["name"]) if len(s) >= 4):
                return False
        ck = self.cat_key()
        blocks = []
        for i, rec in self.full.items():
            ab = attr_block(rec)
            if not ab:
                continue
            if ck is not None:
                row = self.rows.get(i) or {}
                lab = (rec.get(ck) if isinstance(rec, dict) else None) or row.get(ck)
                if not self.owns_label(ix, lab):
                    continue
            blocks.append(ab)
        if len(blocks) < 3:
            return False
        pst = set(stems(prop))
        for ab in blocks:
            for k in ab:
                if pst & set(stems(str(k))):
                    return False
        return True

    def ledger(self):
        out = []
        several = len(self.items) > 1
        for ix, it in enumerate(self.items, 1):
            head = ("item %d: %s" % (ix, it["name"])) if several else ("item: %s" % it["name"])
            if it["shelf"]:
                head += " (store label: %s)" % it["shelf"]
            head += ", quantity %d" % it["qty"]
            live = self.live_reqs(ix)
            deciding, desc = [], []
            for r in live:
                t = r
                if r in self.ref_values:
                    t = r + " (value read on the referenced page: %s)" % self.ref_values[r]
                if self.described(ix, r):
                    desc.append(t)
                else:
                    deciding.append(t)
            if deciding:
                head += "; requirements: " + "; ".join(deciding)
            else:
                head += "; requirements: none beyond the product type itself"
            if desc:
                head += ("; shopper's own phrasing that no listing attribute records, so these decide nothing: "
                         + "; ".join(desc))
            out.append(head)
        if self.budget is not None:
            unit = str(self.brief.get("budget_unit") or "")
            scope = "for each item" if self.brief.get("budget_scope") == "each" else "in total"
            line = ("budget: at most %s %s %s" % (self.budget, unit, scope)).replace("  ", " ", 1)
            out.append(line)
        pick = self.brief.get("pick")
        meas = str(self.brief.get("pick_measure") or "")
        if pick == "cheapest":
            out.append("pick: the lowest-priced listing that matches (lowest price)")
        elif pick in ("most", "least"):
            out.append("pick: the %s %s among the matching listings" % (pick, meas))
        else:
            out.append("pick: any listing that meets every requirement")
        if self.brief.get("in_stock_only") is not False:
            out.append("in stock only")
        for key, lab in (("wishes", "wish (breaks ties only)"), ("rules", "rule the request gives")):
            v = self.brief.get(key)
            v = v if isinstance(v, list) else ([v] if isinstance(v, str) and v else [])
            for w in v:
                out.append("%s: %s" % (lab, w))
        if self.brief.get("quiet_decline") is True:
            out.append("if nothing fits, end the episode without naming or proposing any listing")
        for d in self.duties:
            if d["stage"] == "choose":
                tm = "before choosing"
            elif d["stage"] == "order":
                tm = "at the purchase (Cairn carries it out when you order)"
            else:
                tm = "at a decline (Cairn carries it out at that moment)"
            out.append("step the request requires, %s: %s" % (tm, d["text"]))
        ft = self.facts_tool()
        if ft and (self.facts.get(ft) is True or self.roles.get("offer")):
            line = "listing attributes come from %s" % ft
            pts = list(self.page_tools())
            if self.roles.get("offer") and self.roles["offer"] not in pts:
                pts.append(self.roles["offer"])
            none = sorted(set(t for t in pts if self.facts.get(t) is False))
            if none:
                line += "; " + ", ".join(none) + " show none"
            others = sorted(set(t for t in self.page_tools() if t != ft))
            if others:
                line += "; %s display a listing's page, the page meant whenever a chain is followed or a page is checked" % " or ".join(others)
            out.append(line)
        return "\n".join("- " + x for x in out)

    def status(self):
        s, m, l = self.steps, self.max_steps, self.left()
        res = self.reserve()
        parts = ["[Cairn] step %d of %d used, %d left." % (s, m, l)]
        if l <= res + 1:
            parts.append(" FINAL STEPS: order now if every item already has a carted line; decline only when searching"
                         " has proved that some item has no matching listing at all, and running short of steps proves"
                         " no such thing.")
        elif l <= 6 or (l <= len(self.items) + 5 and not self.checkouts):
            parts.append(" Few steps remain: make the cart hold exactly the picked lines and order at once.")
        elif not self.checkouts and l <= 12:
            parts.append(" Steps are getting scarce: decide the picks, put them in the cart and order within a few"
                         " more steps.")
        elif not self.checkouts and not self.cart_touched and s >= m // 2:
            parts.append(" Half the step budget is spent with nothing in the cart: put the matching listing of every"
                         " item in it now, because the store may yet sell a listing out.")
        if self.clock() >= WIND_DOWN and l > res + 1:
            parts.append(" TIME IS ALMOST OVER: make the cart hold the picked lines and order within two more turns;"
                         " an episode cut off by the clock scores zero.")
        text = "".join(parts)
        live = set()
        for ix in range(1, len(self.items) + 1):
            live.update(self.live_reqs(ix))
        pend = [t for (n, t) in self.refs if t in live and t not in self.ref_values]
        if pend and not self.checkouts:
            text += ("\n[settle these first: visit the listing each one points at, and also every listing that this"
                     " listing's page refers onward to (only the ones the page directs onward, not every listing it"
                     " mentions), since the final page holds the value] " + " | ".join(pend[:3]))
        if self.leads:
            text += "\n[review candidates] " + ", ".join(str(x) for x in self.leads)
        if self.fit_pairs and not self.checkouts:
            fp = self.fit_plan()
            if fp:
                text += "\n" + fp
        if self.changes:
            text += "\n[changes so far] " + " | ".join(self.changes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text


E2_THREE_DIGITS = re.compile(r"\d{3}")
E2_SLASH = re.compile(r" / ")
E2_LETTERS3 = re.compile(r"[a-z]{3,}")
E2_ALNUM3 = re.compile(r"[a-z0-9]{3,}")
E2_ALNUM = re.compile(r"[a-z0-9]+")
E2_PRICE_TEXT = re.compile(r"^\s*[^\d\s]{0,3} ?(\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,6}\.?\s*$")
E2_COMPARE_WORDS = re.compile(r"\b(?:not|no|other\s+than|unlike|under|less|below|up\s+to|at\s+most|more|above|over|at\s+least|between|within|or)\b", re.I)
E2_PRICED = re.compile(r"price|money|cost|amount|spend|budget", re.I)
E2_CAPWORD = re.compile(r"max|most|under|below|limit|cap|upto|up_to", re.I)
E2_ID_LETTERS = re.compile(r"id", re.I)


def gnum(x):
    return "%g" % x if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


class EpisodeRows:
    def listing_id(self, obj, inrows=False, depth=0):
        if not isinstance(obj, dict) or len(obj) < 3:
            return None
        if self.id_key is not None and id_like(obj.get(self.id_key), True):
            return obj[self.id_key].strip()
        if not (inrows or depth == 0 or any(isinstance(v, dict) for v in obj.values())):
            return None
        k = learn_id([obj])
        if k is not None:
            v = obj[k].strip()
            if E2_ID_LETTERS.search(str(k)) or v in self.rows:
                return v
        return None

    def is_rows(self, lst):
        cands = [e for e in lst if isinstance(e, dict) and len(e) >= 3]
        if not cands or len(cands) * 2 < len(lst):
            return False
        if self.id_key is not None and all(id_like(c.get(self.id_key), True) for c in cands):
            return True
        k = learn_id(cands)
        if k is not None:
            self.id_key = k
            return True
        return False

    def absorb(self, obs, depth=0, inrows=False):
        if depth > 6:
            return
        if isinstance(obs, dict):
            lid = self.listing_id(obs, inrows, depth)
            if lid is not None:
                if lid not in self.rows:
                    self.seen.append(lid)
                    self.rows[lid] = {}
                row = self.rows[lid]
                if any(row.get(k) is True and v is False for k, v in obs.items()):
                    self.unavailable.add(lid)
                elif any(row.get(k) is False and v is True for k, v in obs.items()):
                    self.unavailable.discard(lid)
                row.update(obs)
                if attr_block(obs):
                    self.full[lid] = dict(row)
                elif lid in self.full:
                    self.full[lid].update(obs)
                return
            for v in obs.values():
                self.absorb(v, depth + 1, False)
        elif isinstance(obs, list):
            r = self.is_rows(obs)
            for e in obs:
                self.absorb(e, depth + 1, r)

    def enum_labels(self):
        out = set()

        def walk(v, d):
            if d > 6 or not isinstance(v, dict):
                return
            en = v.get("enum")
            if isinstance(en, list):
                for o in en:
                    if isinstance(o, str):
                        out.add(o.strip().lower())
            for x in v.values():
                walk(x, d + 1)
        for s in self.schemas.values():
            walk(s, 0)
        return out

    def cat_key(self):
        stats = {}
        for row in self.rows.values():
            for k, v in row.items():
                if isinstance(v, str) and 1 <= len(v) <= 40 and not E2_THREE_DIGITS.search(v):
                    st = stats.setdefault(k, [[], 0, 0])
                    if v not in st[0]:
                        st[0].append(v)
                    st[1] += 1
                    st[2] += len(v)
        total = max(1, len(self.rows))
        labels = self.enum_labels()
        best, score = None, None
        for k, (vals, n, ln) in stats.items():
            if n < 0.8 * total or len(vals) > 40:
                continue
            share = sum(1 for v in vals if v.strip().lower() in labels) / float(len(vals))
            sc = (share, -len(vals), n, ln / float(n))
            if score is None or sc > score:
                best, score = k, sc
        return best

    def label_matches(self, label):
        lab = str(label)
        wf = word_form(lab)
        ls = set(s for s in stems(lab) if len(s) >= 4)
        for it in self.items:
            if wf == word_form(it["shelf"]) or wf == word_form(it["name"]):
                return True
            ist = set(s for s in (stems(it["shelf"]) | stems(it["name"])) if len(s) >= 4)
            if ls & ist:
                return True
        return bool(self.owner(lab))

    def key_slots(self):
        ck = self.cat_key()
        groups = {}
        for lid in self.seen:
            rec = self.full.get(lid)
            if rec is None:
                continue
            ab = attr_block(rec) or {}
            lv = rec.get(ck) if ck is not None else None
            label = str(lv) if lv is not None else "listings"
            g = groups.setdefault(label, {"n": 0, "keys": {}})
            g["n"] += 1
            used = set()
            for k, v in ab.items():
                wf = word_form(k)
                if not wf or wf in used:
                    continue
                used.add(wf)
                slot = g["keys"].setdefault(wf, {"wf": wf, "count": 0, "shown": {}, "meas": {}, "distinct": []})
                slot["count"] += 1
                shown = str(v).strip().lower()
                meas = str(FIRST_SPELLING.get(str(v), v)).strip().lower()
                slot["shown"][lid] = shown
                slot["meas"][lid] = meas
                if shown not in slot["distinct"]:
                    slot["distinct"].append(shown)
        labels = list(groups)
        match = [l for l in labels if self.label_matches(l)]
        if match:
            labels = match
        labels.sort(key=lambda l: -groups[l]["n"])
        out = []
        for l in labels:
            g = groups[l]
            order = sorted(g["keys"].values(), key=lambda s: (-s["count"], len(s["wf"])))
            kept = []
            for s in order:
                ms = list(s["meas"].values())
                if not ms:
                    continue
                mean = sum(len(m) for m in ms) / float(len(ms))
                if mean > 40 or sum(1 for m in ms if E2_SLASH.search(m)) * 2 >= len(ms):
                    continue
                if len(ms) >= 4 and len(set(ms)) == len(ms) and mean > 12:
                    runs = [set(E2_LETTERS3.findall(m)) for m in ms]
                    sharing = 0
                    for i, r in enumerate(runs):
                        others = set()
                        for j, o in enumerate(runs):
                            if j != i:
                                others |= o
                        if r & others:
                            sharing += 1
                    if sharing * 2 < len(ms):
                        continue
                twin = None
                for kk in kept:
                    if self.twin_keys(kk, s):
                        twin = kk
                        break
                if twin is not None:
                    for lid, mv in s["meas"].items():
                        if lid not in twin["meas"]:
                            twin["meas"][lid] = mv
                            twin["shown"][lid] = s["shown"][lid]
                    for d in s["distinct"]:
                        if d not in twin["distinct"]:
                            twin["distinct"].append(d)
                    twin["count"] = len(twin["meas"])
                    continue
                kept.append(s)
            kept.sort(key=lambda s: -s["count"])
            out.append((l, kept[:KEY_TABLE_SIZE], g["n"]))
        return out

    def twin_keys(self, a, b):
        wa, wb = set(a["wf"].split()), set(b["wf"].split())
        if not wa or not wb or not (wa & wb):
            return False

        def agree(x, y):
            if x == y:
                return True
            xw, yw = E2_ALNUM.findall(x), E2_ALNUM.findall(y)
            sh, lo = (xw, yw) if len(xw) <= len(yw) else (yw, xw)
            return bool(sh) and all(w in lo for w in sh) and len(lo) <= 2 * len(sh) + 2
        shared = [l for l in a["meas"] if l in b["meas"] and a["meas"][l].strip() and b["meas"][l].strip()]
        n = sum(1 for l in shared if agree(a["meas"][l], b["meas"][l]))
        if not (wa <= wb or wb <= wa):
            return len(shared) >= 2 and n == len(shared)
        if shared:
            return n * 2 > len(shared)
        return bool(set(x for x in a["distinct"] if x) & set(x for x in b["distinct"] if x))

    def not_stated(self, label, wf, shown):
        own = self.owner(label)
        if not own:
            return ""
        sw = set(E2_ALNUM3.findall(" ".join(shown).lower()))
        for r in self.live_reqs(own):
            prop, _, val = r.partition(":")
            if E2_COMPARE_WORDS.search(val):
                continue
            vw = E2_ALNUM3.findall(val.lower())
            if prop.strip() and word_form(prop) != wf and (stems(prop) & stems(wf)) and vw \
                    and not all(w in sw for w in vw):
                return " [not the stated " + prop.strip() + ": another property]"
        return ""

    def item_groups(self):
        ck = self.cat_key()
        groups = {}
        for lid in self.seen:
            row = self.rows.get(lid) or {}
            v = row.get(ck) if ck is not None else None
            groups.setdefault(str(v) if v is not None else "", []).append(lid)
        match = [l for l in groups if self.label_matches(l)]
        pool = match or list(groups)
        pool.sort(key=lambda l: -len(groups[l]))
        return [groups[l] for l in pool[:max(1, len(self.items))]]

    def single_target(self, tool):
        sc = self.schemas.get(tool)
        if not isinstance(sc, dict):
            return None
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")]
        tgt = next((k for k in req if k in texts), None)
        if tgt is None and len(texts) == 1:
            tgt = texts[0]
        if tgt is None:
            return None
        if any(k != tgt for k in req):
            return None
        return tgt

    def price(self, lid):
        if not self.price_key:
            return None
        v = (self.rows.get(lid) or {}).get(self.price_key)
        if isinstance(v, dict):
            for x in v.values():
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    return x
            return None
        if isinstance(v, str):
            m = E2_PRICE_TEXT.match(v)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except Exception:
                    return None
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
        return None

    def sold_out(self, lid):
        return lid in self.unavailable or lid in self.gone

    def buyable(self, lid):
        if self.sold_out(lid):
            return False
        if not self.stock_key:
            return True
        v = (self.rows.get(lid) or {}).get(self.stock_key)
        if isinstance(v, str):
            t = v.strip()
            if V4_SOLD_WHOLE.match(t) or V4_SOLD_PART.search(t):
                return False
        if v is False:
            return False
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0:
            return False
        return True

    def listing_records(self, obs):
        out = []
        if isinstance(obs, dict):
            lid = self.listing_id(obs, False, 0)
            if lid is not None:
                return [(lid, obs)]
            for v in obs.values():
                if isinstance(v, list):
                    for e in v:
                        i = self.listing_id(e, True, 1) if isinstance(e, dict) else None
                        if i is not None:
                            out.append((i, e))
                elif isinstance(v, dict):
                    i = self.listing_id(v, False, 1)
                    if i is not None:
                        out.append((i, v))
                        continue
                    for x in v.values():
                        if isinstance(x, list):
                            for e in x:
                                j = self.listing_id(e, True, 2) if isinstance(e, dict) else None
                                if j is not None:
                                    out.append((j, e))
        elif isinstance(obs, list):
            for e in obs:
                i = self.listing_id(e, True, 1) if isinstance(e, dict) else None
                if i is not None:
                    out.append((i, e))
        return out

    def ids_in(self, obs):
        return [i for i, _ in self.listing_records(obs)]

    def record_view_keys(self, tool, obs):
        for lid, rec in self.listing_records(obs):
            ks = set(rec.keys())
            self.tool_keys.setdefault(tool, set()).update(ks)
            cur = self.view_keys.get(lid)
            if cur and cur[0] == self.version:
                cur[1].update(ks)
            else:
                self.view_keys[lid] = (self.version, set(ks))

    def shown_in_full(self, tool, lid):
        tk = self.tool_keys.get(tool)
        if not tk:
            return False
        cur = self.view_keys.get(lid)
        if not cur or cur[0] != self.version:
            return False
        return tk <= cur[1] and lid not in self.gone

    def shape_of(self, tool, obs):
        if tool in self.shapes:
            return self.shapes[tool]
        if not isinstance(obs, dict):
            return None
        cands = [k for k, v in obs.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(e, dict) for e in v)]
        if not cands:
            return None
        self.shapes[tool] = None
        sample = {k: (v[:3] if k in cands else v) for k, v in obs.items()}
        prompt = (SHAPE_PROMPT + "\n\nTOOL: " + tool + " | " + cjson(self.schemas.get(tool) or {})
                  + "\n\nRESULT (only its first rows are shown):\n" + cjson(clip_deep(sample, 120))[:4000])
        ans = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rk = ans.get("rows")
        if not isinstance(rk, str) or rk not in cands:
            return None
        first = obs[rk][0]
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        shp = {"rows": rk, "count": None, "next": None, "next_arg": None, "price": None, "stock": None}
        for f, where in (("count", obs), ("next", obs), ("next_arg", props), ("price", first), ("stock", first)):
            v = ans.get(f)
            if isinstance(v, str) and isinstance(where, dict) and v in where:
                shp[f] = v
        if shp["count"] is not None:
            cv = obs.get(shp["count"])
            if not (isinstance(cv, int) and not isinstance(cv, bool)):
                shp["count"] = None
        self.shapes[tool] = shp
        if self.price_key is None and shp["price"]:
            self.price_key = shp["price"]
        if self.stock_key is None and shp["stock"]:
            self.stock_key = shp["stock"]
        self.trace.add({"kind": "shape", "tool": tool, "shape": shp})
        return shp

    def paging_plan(self, tool, obs):
        shp = self.shape_of(tool, obs)
        if not shp or not shp.get("count") or not shp.get("next") or not shp.get("next_arg"):
            return None
        if not isinstance(obs, dict):
            return None
        rows = obs.get(shp["rows"])
        if not isinstance(rows, list) or not rows:
            return None
        total = obs.get(shp["count"])
        if not isinstance(total, int) or isinstance(total, bool):
            return None
        nx = obs.get(shp["next"])
        if nx is not None and (isinstance(nx, bool) or not isinstance(nx, (str, int))):
            return None
        return {"total": total, "shown": len(rows), "arg": shp["next_arg"], "next": nx}

    def price_sorted(self, obs):
        ps = [self.price(i) for i in self.ids_in(obs)]
        if len(ps) < 3 or any(p is None for p in ps):
            return False
        return all(ps[i] <= ps[i + 1] for i in range(len(ps) - 1))

    def auto_page(self, tool, args, obs):
        self.paged = []
        if self.brief.get("pick") not in ("most", "least", "cheapest"):
            return ""
        plan = self.paging_plan(tool, obs)
        if not plan:
            return ""
        sent = args.get(plan["arg"])
        later = False
        if isinstance(sent, int) and not isinstance(sent, bool):
            later = sent not in (0, 1)
        elif sent not in (None, "", False):
            later = True
        if plan["next"] is None or later or plan["total"] <= plan["shown"]:
            return ""
        if self.brief.get("pick") == "cheapest" and self.price_sorted(obs):
            return ""
        shown = plan["shown"]
        need = -(-(plan["total"] - shown) // shown)
        res = self.reserve()
        if need > 2 and self.pages_read + 2 <= AUTO_PAGES and self.left() - res - 2 >= 12:
            need = 2
        if need > 2 or self.pages_read + need > AUTO_PAGES or self.left() - res - need < 12:
            return ("\n[Cairn] The search matched %d listings but displayed only %d, not sorted by price, and the pick"
                    " weighs every listing that fits. Search again with tighter words, those words being the value the"
                    " requirement names, or request the next page by sending %s set to %s."
                    % (plan["total"], shown, plan["arg"], cjson(plan["next"])))
        lines = []
        nxt = plan["next"]
        read = 0
        for _ in range(need):
            a = dict(args)
            a[plan["arg"]] = nxt
            a, f = shape_args(self.schemas.get(tool) or {}, a)
            if f:
                break
            r = self.env_call(tool, a)
            if r is None or self.finished:
                break
            self.pages_read += 1
            read += 1
            o = r.get("observation")
            self.observe(tool, a, o)
            self.paged.append(o)
            rend = self.display(o)
            self.read_cache[tool + cjson(a)] = (self.version, self.steps, rend)
            lines.append("%s=%s: %s" % (plan["arg"], cjson(nxt), rend))
            p2 = self.paging_plan(tool, o)
            nxt = p2["next"] if p2 else None
            if nxt is None:
                break
        self.trace.add({"kind": "auto_pages", "tool": tool, "read": read})
        if not read:
            return ""
        t = ("\n[Cairn fetched the following pages of this search by itself, one step per page]\n" + "\n".join(lines))
        if nxt is not None:
            t += ("\n[Cairn] The search matched %d listings; Cairn has now read %d rows. Make the search tighter, using"
                  " the value the requirement names, or keep requesting pages, so that the remaining listings come into"
                  " view." % (plan["total"], shown * (read + 1)))
        return t

    def cap_ignored(self, tool, args, obs):
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        rank = self.cap_args(tool)

        def priced(n):
            p = props.get(n) if isinstance(props.get(n), dict) else {}
            return bool(E2_PRICED.search(n + " " + str(p.get("description") or "")))
        caps = []
        for n, v in args.items():
            if n not in props or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if (E2_CAPWORD.search(n) and priced(n)) or (rank and rank[0] == n and len(rank) == 1):
                caps.append((v, n))
        if not caps or not self.price_key:
            return ""
        cap, cname = min(caps)
        over = [i for i in self.ids_in(obs) if (self.price(i) or 0) > cap]
        if not over:
            return ""
        hint = []
        for n, v in args.items():
            p = props.get(n)
            if isinstance(v, str) and isinstance(p, dict) and isinstance(p.get("enum"), list):
                others = [o for o in p["enum"] if o != v]
                if others:
                    hint.append("%s=%s" % (n, cjson(others[0])))
        t = ("\n[Cairn] The search did not honour %s at %s: %d of its rows are priced above it."
             % (cname, gnum(cap), len(over)))
        if hint:
            t += (" Browse the category setting %s while keeping the same %s, so that only the cheaper listings show."
                  % (", ".join(hint), cname))
        return t

    def cheapest_here(self, obs):
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        ids = []
        for o in [obs] + list(self.paged):
            for i in self.ids_in(o):
                if i not in ids:
                    ids.append(i)
        ok = [i for i in ids if self.buyable(i) and self.price(i) is not None]
        if len(ok) < 2:
            return ""
        ok.sort(key=lambda i: (self.price(i), i))
        return ("\n[Cairn] Buyable rows here from the lowest price up: "
                + ", ".join("%s (%s)" % (i, gnum(self.price(i))) for i in ok[:6]))

    def wider_hold(self):
        if self.wider_held or self.brief.get("pick") != "cheapest":
            return ""
        if self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = []
        for i in self.cart_ids():
            p = self.price(i)
            if p is not None:
                lines.append("%s at %s" % (i, gnum(p)))
        t = (HOLD + "The order call was held back. %d steps remain and the pick is cheapest. A search may match just a"
             " portion of a listing's text, so searching by a requirement's value surfaces only part of the listings"
             " that fit, and rows arrive unordered by price. Ahead of any checkout, for each item run a few extra"
             " searches under different product names: the product's bare one-word name, its synonyms, and words"
             " lifted from titles of fitting listings already seen. Consider only rows that can be bought and cost less"
             " than that item's current cart line" % self.left())
        if lines:
            t += " (" + "; ".join(lines) + ")"
        t += (". Then check the attribute records of the lowest-priced of those rows, replace the cart line with any"
              " one that fits, and then check out.")
        return t

    def note_facts(self, tool, rec):
        if attr_block(rec):
            self.facts[tool] = True
        elif self.facts.get(tool) is not True:
            self.bare[tool] = self.bare.get(tool, 0) + 1
            if self.bare[tool] >= 2:
                self.facts[tool] = False
        if tool in self.facts and tool not in self.facts_order:
            self.facts_order.append(tool)

    def lists_things(self, tool):
        st = self.single_target(tool)
        if st:
            u = under_form(st)
            if V10_SHELF.search(u) or V14_QUERY.search(u):
                return True
        return tool in self.shapes


E3_SNAKE = re.compile(r"[\s_.\-]+")
E3_RUN4 = re.compile(r"[a-z]{4,}")
E3_NAME_WORD = re.compile(r"[a-z][a-z'\-]{2,}")


def snake(n):
    return E3_SNAKE.sub("_", str(n)).lower()


def whole_down(p):
    return int(p) - 1 if p == int(p) else int(p)


class EpisodeHunts:
    def fetch_facts(self, ids, reserve):
        ft = self.facts_tool()
        tgt = self.single_target(ft) if ft else None
        if (not ft or not tgt) and self.compare_facts() is not True:
            return []
        wanted = [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]
        texts = []
        unread = []
        cmp = self.roles.get("compare")
        la = self.compare_target(cmp) if cmp else None
        while (cmp and la and self.facts.get(cmp) is True and len(wanted) >= 2 and self.left() > reserve
               and not self.finished and not self.rushed()):
            room = self.compare_room()
            batch, wanted = wanted[:room], wanted[room:]
            a, f = shape_args(self.schemas.get(cmp) or {}, {la: batch})
            if f:
                wanted = batch + wanted
                break
            r = self.env_call(cmp, a)
            if r is None:
                wanted = batch + wanted
                break
            obs = r.get("observation")
            self.observe(cmp, a, obs)
            self.record_view_keys(cmp, obs)
            recs = self.listing_records(obs)
            for lid, rec in recs:
                texts.append(lid + ": " + self.display(rec)[:1500])
            if recs:
                self.cmp_ok = max(self.cmp_ok, len(recs))
                unread += [i for i in batch if i not in self.full]
                continue
            self.cmp_fail = min(self.cmp_fail or 999, len(batch))
            wanted = batch + wanted
            if self.compare_room() >= len(batch):
                break
        if ft and tgt:
            done = set()
            for lid in unread + wanted:
                if lid in done:
                    continue
                done.add(lid)
                if self.left() <= reserve or self.finished or self.facts.get(ft) is False or self.rushed():
                    break
                if (ft, lid) in self.factless:
                    continue
                a, f = shape_args(self.schemas.get(ft) or {}, {tgt: lid})
                if f:
                    break
                r = self.env_call(ft, a)
                if r is None:
                    break
                obs = r.get("observation")
                self.observe(ft, a, obs)
                self.record_view_keys(ft, obs)
                if self.listing_id(obs, False, 0) is not None:
                    self.note_facts(ft, obs)
                texts.append(lid + ": " + self.display(obs)[:1500])
        if texts:
            self.trace.add({"kind": "facts", "n": len(texts)})
        return texts

    def follow_links(self, ids, reserve):
        tpl = self.value_template()
        if not tpl or (len(self.items) < 2 and len(ids) < 2):
            return
        tool, fixed, field = tpl
        for lid in ids:
            idx = self.fit_index()
            ptr = (idx.get(lid) or {}).get("pointer")
            if not ptr or ptr in self.link_tried or any(m.get("own") == ptr for m in idx.values()):
                continue
            if self.left() <= reserve or self.finished or self.rushed():
                return
            self.link_tried.add(ptr)
            a = dict(fixed)
            a[field] = ptr
            a, f = shape_args(self.schemas.get(tool) or {}, a)
            if f:
                return
            r = self.env_call(tool, a)
            if r is None:
                return
            self.observe(tool, a, r.get("observation"))
            self.trace.add({"kind": "link", "code": ptr})
            idx = self.fit_index()
            hits = [i for i, m in idx.items() if m.get("own") == ptr]
            self.fetch_facts(hits, max(reserve, self.reserve() + 6))

    def cap_args(self, tool):
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        cands = []
        for n, p in props.items():
            if not isinstance(p, dict) or not (set(kinds(p)) & set(("integer", "number"))):
                continue
            if V7_NONPRICE.search(n) or V9_LOWER.search(snake(n)):
                continue
            cands.append(n)
        ev = self.cap_evidence(tool)
        proved = [n for n in cands if any(k[0] == n and v == "holds" for k, v in ev.items())]
        named_max = [n for n in cands if V6_CAP.search(snake(n))]
        money = [n for n in cands if V8_PRICE.search(snake(n))]
        for grp in (proved, [n for n in named_max if n in money], money, named_max):
            if grp:
                return grp
        return []

    def cap_evidence(self, tool):
        ck = self.cat_key()
        pools = {}
        for lid, row in self.rows.items():
            p = self.price(lid)
            if p is not None:
                lab = str(row.get(ck)) if ck is not None else ""
                pools.setdefault(lab, []).append(p)
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        ev = {}
        for mk in list(self.mem_results):
            t, args = self.mem_args[mk]
            if t != tool:
                continue
            ids = self.ids_in(self.mem_results[mk])
            if not ids or any(self.price(i) is None for i in ids):
                continue
            mode = tuple(sorted((k, cjson(v)) for k, v in args.items()
                                if isinstance(props.get(k), dict) and isinstance(props[k].get("enum"), list)))
            pool = []
            if ck is not None:
                for lab in set(str((self.rows.get(i) or {}).get(ck)) for i in ids):
                    pool += pools.get(lab, [])
            dear = max(self.price(i) for i in ids)
            for n, v in args.items():
                if n not in props or not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                key = (n, mode)
                if dear > v:
                    ev[key] = "fails"
                elif ev.get(key) != "fails" and pool:
                    share = sum(1 for x in pool if x <= v) / float(len(pool))
                    if share ** len(ids) < LIMIT_CHANCE:
                        ev[key] = "holds"
        return ev

    def shelf_template(self):
        for t in self.tools:
            name = t["function"]["name"]
            sc = self.schemas.get(name) or {}
            props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
            req = sc.get("required") if isinstance(sc.get("required"), list) else []
            caps = self.cap_args(name)
            shelves = [n for n, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")
                       and V10_SHELF.search(snake(n))]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            cap, shelf = caps[0], shelves[0]
            learned = [(dict(k[1]), v) for k, v in self.cap_evidence(name).items() if k[0] == cap]
            enum_args = [n for n, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list)
                         and len(p["enum"]) >= 2]
            single = len(enum_args) == 1 and enum_args[0] in req
            opts = {}
            for n in enum_args:
                en = props[n]["enum"]
                val = None
                for mode, verdict in learned:
                    if verdict == "holds" and n in mode:
                        val = json.loads(mode[n])
                        break
                if val is None:
                    for s in reversed(self.searched):
                        if shelf in s and n in s and not any(
                                V14_QUERY.search(snake(k)) and isinstance(x, str) and x.strip()
                                for k, x in s.items() if k != shelf and k not in enum_args):
                            val = s[n]
                            break
                if val is None:
                    lm = [o for o in en if isinstance(o, str) and V11_LISTMODE.search(o)]
                    if len(lm) == 1:
                        val = lm[0]
                if val is None and single:
                    worded = set()
                    for s in self.searched:
                        if n in s and any(V14_QUERY.search(snake(k)) and isinstance(x, str) and x.strip()
                                          for k, x in s.items() if k != n):
                            worded.add(cjson(s[n]))
                    unused = [o for o in en if cjson(o) not in worded]
                    if worded and len(unused) == 1:
                        val = unused[0]
                if single:
                    spoiled = [json.loads(m[n]) for m, v in learned if v == "fails" and n in m]
                    if spoiled and (val is None or val in spoiled):
                        rest = [o for o in en if o not in spoiled]
                        val = rest[0] if len(rest) == 1 else None
                if val is None:
                    return None
                opts[n] = val
            return name, opts, shelf, cap
        return None

    def walk_below(self, lines):
        tpl = self.shelf_template()
        ck = self.cat_key()
        if not tpl or ck is None:
            return
        tool, opts, shelf, cap = tpl
        for lid in lines:
            p = self.price(lid)
            lab = (self.rows.get(lid) or {}).get(ck)
            if p is None or p < 2 or lab in (None, "") or (lid, p) in self.capped:
                continue
            self.capped.add((lid, p))
            ceil = whole_down(p)
            used = set()
            for n in range(WALK_PER_LINE):
                need = self.reserve() + (7 if n == 0 else WALK_RESERVE)
                if self.left() <= need or self.finished or self.rushed():
                    return
                if ceil < 1 or ceil in used:
                    break
                used.add(ceil)
                a = dict(opts)
                a[shelf] = lab
                a[cap] = ceil
                a, f = shape_args(self.schemas.get(tool) or {}, a)
                if f:
                    return
                r = self.env_call(tool, a)
                if r is None:
                    return
                obs = r.get("observation")
                self.observe(tool, a, obs)
                ids = self.ids_in(obs)
                self.trace.add({"kind": "walk", "line": lid, "price": p, "ceiling": ceil, "rows": len(ids)})
                if not ids or any((self.price(i) or 0) > ceil for i in ids):
                    break
                plan = self.paging_plan(tool, obs)
                if not plan or plan["total"] <= plan["shown"] or self.price_sorted(obs):
                    break
                if plan["total"] <= 2 * plan["shown"]:
                    if plan["next"] not in (None, "") and self.left() > self.reserve() + 7 and not self.rushed():
                        a2 = dict(a)
                        a2[plan["arg"]] = plan["next"]
                        a2, f2 = shape_args(self.schemas.get(tool) or {}, a2)
                        if not f2:
                            r2 = self.env_call(tool, a2)
                            if r2 is not None:
                                self.observe(tool, a2, r2.get("observation"))
                    break
                known = [self.price(i) for i in ids if self.price(i) is not None]
                if not known:
                    break
                ceil = whole_down(min(known))

    def cheaper_pool(self, lid, p, lab, ck):
        inc = set(self.cart_ids())
        lab = str(lab).strip().lower()
        pool = [i for i in self.seen if i not in inc and i not in self.full and self.buyable(i)
                and str((self.rows.get(i) or {}).get(ck)).strip().lower() == lab
                and self.price(i) is not None and self.price(i) < p]
        pool.sort(key=lambda i: self.price(i))
        return pool

    def cheaper_rows(self):
        if self.brief.get("pick") != "cheapest":
            return []
        mark = self.cart_mark() + "#" + str(self.version)
        ck = self.cat_key()
        if ck is None or mark in self.cheaper_marks:
            return []
        self.cheaper_marks.add(mark)
        lines = self.cart_ids()
        self.walk_below(lines)
        room = self.compare_room() if self.compare_facts() is True else 0
        picks = []
        for lid in lines:
            p = self.price(lid)
            lab = (self.rows.get(lid) or {}).get(ck)
            if p is None or lab in (None, ""):
                continue
            pool = self.cheaper_pool(lid, p, lab, ck)
            vr = next((i for i in pool if i in self.value_rows), None)
            if vr is not None:
                pool.remove(vr)
                pool.insert(0, vr)
            got = 0
            for i in pool:
                if got >= (room or 2):
                    break
                if i not in picks:
                    picks.append(i)
                    got += 1
        n = room * len(lines) if room else min(6, max(1, 3 * len(lines)))
        return self.fetch_facts(picks[:n], self.reserve() + 6)

    @soft("")
    def below_cart(self):
        ck = self.cat_key()
        if ck is None:
            return ""
        ent = []
        for lid in self.cart_ids():
            p = self.price(lid)
            lab = (self.rows.get(lid) or {}).get(ck)
            if p is None or lab in (None, ""):
                continue
            for i in self.cheaper_pool(lid, p, lab, ck)[:4]:
                ent.append("%s at %s" % (i, gnum(self.price(i))))
        if not ent:
            return ""
        return "Not read yet and priced below a line of the cart: " + ", ".join(ent) + "."

    def title_key(self):
        ck = self.cat_key()
        st = {}
        for row in self.rows.values():
            for k, v in row.items():
                if k == ck or not isinstance(v, str) or len(v) > TITLE_LIMIT:
                    continue
                s = st.setdefault(k, [0, 0, 0, set()])
                s[0] += len(v.split())
                s[1] += 1
                s[2] += len(v)
                s[3].add(v)
        total = len(self.rows)
        kept = {k: s for k, s in st.items() if s[1] * 5 >= total * 4 and s[0] >= 2 * s[1] and len(s[3]) * 2 >= s[1]}
        if not kept:
            return None
        most = max(len(s[3]) for s in kept.values())
        kept = {k: s for k, s in kept.items() if len(s[3]) * 10 >= most * 9}
        return min(kept, key=lambda k: (kept[k][2] / float(kept[k][1]), k))

    def hunt_words(self, ix):
        ck = self.cat_key()
        tk = self.title_key()
        it = self.items[ix - 1]
        labels, names = [], []
        if ck is not None:
            for i in self.seen:
                row = self.rows.get(i) or {}
                lab = row.get(ck)
                if lab is not None and self.owns_label(ix, lab):
                    labels.append(str(lab))
                    if tk and isinstance(row.get(tk), str):
                        names.append(row[tk])
            for i, rec in self.full.items():
                if self.owns_label(ix, rec.get(ck)):
                    ab = attr_block(rec) or {}
                    names.append(" ".join(v for v in ab.values() if isinstance(v, str) and len(v) <= HUNT_VALUE_LEN))
        other = set()
        for j, o in enumerate(self.items, 1):
            if j != ix:
                other |= set(E3_RUN4.findall((o["shelf"] + " " + o["name"]).lower()))
        if labels:
            cnt = {}
            for l in labels:
                cnt[l] = cnt.get(l, 0) + 1
            best = max(cnt.values())
            home = next(l for l in labels if cnt[l] == best).lower()
        else:
            home = (it["shelf"] or it["name"]).lower()
        cands = [w for w in E3_RUN4.findall(home) if w not in other]
        noun = ""
        for w in cands:
            if len(w) > len(noun):
                noun = w
        if ix in self.hunt_nonoun:
            noun = ""
        if not names:
            return noun, []
        counts = {}
        for n in names:
            for w in set(E3_NAME_WORD.findall(n.lower())):
                counts[w] = counts.get(w, 0) + 1
        own = roots(it["shelf"]) | roots(it["name"])
        ceil = max(2, int(HUNT_COMMON * len(names)))
        words = [w for w, c in counts.items() if 2 <= c <= ceil and w != noun and roots(w) and not (roots(w) & own)]
        words.sort(key=lambda w: (-counts[w], w))
        return noun, words

    def item_rows(self, ix):
        ck = self.cat_key()
        if ck is None:
            return len(self.rows)
        return sum(1 for r in self.rows.values() if r.get(ck) is not None and self.owns_label(ix, r.get(ck)))

    @soft("")
    def cheaper_hunt(self):
        live = [self.live_reqs(ix) for ix in range(1, len(self.items) + 1)]
        mark = cjson([live, self.budget, sorted(self.gone)])
        if self.brief.get("pick") != "cheapest" or self.hunts >= HUNT_ROUNDS or mark == self.hunt_mark:
            return ""
        tpl = self.value_template()
        if not tpl or not self.price_key or self.clock() > HUNT_DEADLINE:
            return ""
        room = min(HUNT_CAP, self.left() - self.reserve() - HUNT_RESERVE - 2 * len(self.items))
        if room < 3:
            return ""
        self.hunts += 1
        self.hunt_mark = mark
        share = max(3, room // len(self.items))
        spent = gain = 0
        for ix in range(1, len(self.items) + 1):
            if spent >= room or self.finished:
                break
            s, g = self.item_hunt(ix, tpl, min(share, room - spent))
            spent += s
            gain += g
        self.trace.add({"kind": "hunt", "steps": spent, "new": gain, "cart": self.cart_mark()})
        if spent:
            self.wider_held = True
            return ("Cairn itself ran %d extra searches chasing a lower price, one step each, and they brought up %d"
                    " new rows." % (spent, gain))
        return ""

    def item_hunt(self, ix, tpl, room):
        tool, fixed, field = tpl
        noun, _ = self.hunt_words(ix)
        values = [v for v, _ in self.value_words(ix)]
        st = {"spent": 0, "gain": 0, "idle": 0, "refused": False}

        def run(q):
            self.hunt_tried.add((ix, q.lower()))
            a = dict(fixed)
            a[field] = q
            a, f = shape_args(self.schemas.get(tool) or {}, a)
            if f:
                st["refused"] = True
                return False, False
            if st["spent"] >= room or self.finished:
                return False, False
            holds = bool(values) and all(v.lower() in q.lower() for v in values)
            found = False
            visited = set()
            while not self.finished and self.clock() <= HUNT_DEADLINE + HUNT_GRACE:
                mk = self.canon_mark(tool, a)
                if mk in visited:
                    break
                visited.add(mk)
                if mk in self.mem_results:
                    obs = self.mem_results[mk]
                    new = HUNT_YIELD
                else:
                    if st["spent"] >= room:
                        break
                    before = self.item_rows(ix)
                    r = self.env_call(tool, a)
                    if r is None:
                        break
                    st["spent"] += 1
                    obs = r.get("observation")
                    self.observe(tool, a, obs)
                    new = self.item_rows(ix) - before
                    st["gain"] += new
                    st["idle"] = 0 if new else st["idle"] + 1
                ids = self.ids_in(obs)
                if ids:
                    found = True
                    if holds:
                        self.value_rows.update(ids)
                plan = self.paging_plan(tool, obs)
                if not plan or plan["next"] in (None, ""):
                    break
                if new < HUNT_YIELD and plan["total"] > HUNT_WHOLE:
                    break
                a2 = dict(a)
                a2[plan["arg"]] = plan["next"]
                a, f = shape_args(self.schemas.get(tool) or {}, a2)
                if f:
                    break
            return True, found
        key = (ix, tuple(values))
        if key not in self.hunt_stems:
            tries = []
            if values:
                tries += [" ".join(values + [noun]).strip(), " ".join(values).strip()]
            if len(values) >= 2:
                tries.append(" ".join(values[:-1] + [noun]).strip())
            if noun:
                tries.append(noun)
            seen = []
            for t in tries:
                if t and t not in seen:
                    seen.append(t)
            stem = ""
            for t in seen:
                issued, found = run(t)
                if not issued:
                    return st["spent"], st["gain"]
                if found:
                    stem = t
                    break
                if noun and t == noun:
                    self.hunt_nonoun.add(ix)
                    noun = ""
            self.hunt_stems[key] = stem
        stem = self.hunt_stems[key]
        while (st["spent"] < room and st["idle"] < HUNT_IDLE and not st["refused"] and not self.finished
               and self.clock() <= HUNT_DEADLINE + HUNT_GRACE):
            _, words = self.hunt_words(ix)
            sw = set(stem.lower().split())
            cand = [w for w in words if (ix, (w + " " + stem).strip().lower()) not in self.hunt_tried and w not in sw]
            if not cand:
                break
            run((cand[0] + " " + stem).strip())
        return st["spent"], st["gain"]

    @soft(list)
    def fit_hunt(self):
        if self.fit_hunted or self.clock() > HUNT_DEADLINE:
            return []
        tpl = self.value_template()
        if not tpl:
            return []
        res = self.reserve() + len(self.items) + 1
        spare = self.left() - res
        if spare < 3:
            return []
        self.fit_hunted = True
        tool, fixed, field = tpl
        ck = self.cat_key()
        held = self.held_units()
        covered = [str((self.rows.get(i) or {}).get(ck)) for i, u in held.items() if u >= 1 and ck is not None
                   and (self.rows.get(i) or {}).get(ck) is not None]
        hunted = [ix for ix in range(1, len(self.items) + 1) if not any(self.owns_label(ix, l) for l in covered)]
        if not hunted:
            hunted = list(range(1, len(self.items) + 1))
        wanted = {}
        for ix in hunted:
            wanted[ix] = [t for v, _ in self.value_words(ix) for t in value_tokens(v)]
        walked = self.unread_walk(tool, wanted, min(FIT_PAGES, spare // 2))
        sroom = min(HUNT_CAP, self.left() - self.reserve() - HUNT_RESERVE)
        total = 0
        if sroom >= 3:
            for ix in hunted:
                vals = [v for v, _ in self.value_words(ix)]
                noun, words = self.hunt_words(ix)
                qs = []
                for s in self.plan_series(ix)[:SERIES_HUNTED]:
                    qs.append((s + " " + (vals[0] if vals else "") + " " + noun).strip())
                for v in vals:
                    qs.append((v + " " + noun).strip())
                if noun:
                    qs += [v.strip() for v in vals]
                for w in words[:FIT_WORDS]:
                    qs.append((w + " " + noun).strip())
                if noun:
                    qs.append(noun)
                order = []
                for q in qs:
                    if q and q not in order:
                        order.append(q)
                share = max(2, sroom // len(hunted))
                used = 0
                for q in order:
                    if total >= sroom or used >= share or self.finished or self.clock() > HUNT_DEADLINE + HUNT_GRACE:
                        break
                    a = dict(fixed)
                    a[field] = q
                    a, f = shape_args(self.schemas.get(tool) or {}, a)
                    if f:
                        continue
                    r = self.env_call(tool, a)
                    if r is None:
                        return []
                    total += 1
                    used += 1
                    self.observe(tool, a, r.get("observation"))
        texts = []
        opened = 0
        for ix in hunted:
            toks = set(wanted.get(ix) or [])
            picks = []
            for i in self.seen:
                if i in self.full or not self.buyable(i):
                    continue
                if ck is not None and not self.owns_label(ix, (self.rows.get(i) or {}).get(ck)):
                    continue
                if isinstance(self.budget, (int, float)) and (self.price(i) or 0) > self.budget:
                    continue
                picks.append(i)
            picks.sort(key=lambda i: (-len(toks & text_words(cjson(self.rows.get(i)))),
                                      self.price(i) if self.price(i) is not None else float("inf")))
            picks = picks[:FIT_OPENS]
            for k in range(0, len(picks), 4):
                if self.clock() > HUNT_DEADLINE + HUNT_GRACE:
                    break
                got = self.fetch_facts(picks[k:k + 4], res)
                opened += len(got)
                texts += got
        self.trace.add({"kind": "fit_hunt", "pages": walked, "searches": total, "opened": opened})
        return texts

    def unread_walk(self, tool, wanted, room):
        share = max(1, room // max(1, len(self.items)))
        spent = 0
        ck = self.cat_key()
        for limit in (share, room):
            for ix in wanted:
                stop = min(room, spent + limit)
                toks = set(wanted[ix])
                for mk in list(self.mem_results):
                    t, a = self.mem_args[mk]
                    if t != tool:
                        continue
                    txt = " ".join(str(v).lower() for v in a.values() if isinstance(v, str))
                    if not (text_words(txt) & toks):
                        continue
                    obs = self.mem_results[mk]
                    owners = []
                    if ck is not None:
                        for i in self.ids_in(obs):
                            lab = (self.rows.get(i) or {}).get(ck)
                            owners.append(self.owner(str(lab)) if lab is not None else None)
                    if owners and None not in owners and 0 not in owners and ix not in owners:
                        continue
                    visited = set()
                    while spent < stop and not self.finished:
                        plan = self.paging_plan(t, obs)
                        if not plan or plan["next"] in (None, ""):
                            break
                        a2 = dict(a)
                        a2[plan["arg"]] = plan["next"]
                        a2, f = shape_args(self.schemas.get(t) or {}, a2)
                        if f:
                            break
                        m2 = self.canon_mark(t, a2)
                        if m2 in visited:
                            break
                        visited.add(m2)
                        if m2 in self.mem_results:
                            obs, a = self.mem_results[m2], a2
                            continue
                        r = self.env_call(t, a2)
                        if r is None:
                            return spent
                        spent += 1
                        obs, a = r.get("observation"), a2
                        self.observe(t, a2, obs)
        return spent


E4_SPACES = re.compile(r"\s+")
E4_RUN3 = re.compile(r"[a-z0-9]{3,}")


def is_whole(v):
    return isinstance(v, int) and not isinstance(v, bool)


class EpisodeOpening:
    def canon_mark(self, tool, args):
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        kept = {}
        for k in sorted(args):
            v = args[k]
            p = props.get(k) if isinstance(props.get(k), dict) else {}
            if is_whole(v):
                mn, df = p.get("minimum"), p.get("default")
                if (is_whole(mn) or isinstance(mn, float)) and v == mn:
                    continue
                if (is_whole(df) or isinstance(df, float)) and v == df:
                    continue
                if v == 1 and self.page_arg.get(tool) == k and "minimum" not in p and "default" not in p:
                    continue
            if isinstance(v, str):
                v = E4_SPACES.sub(" ", v.lower()).strip()
            kept[k] = v
        return tool + cjson(kept)

    def note_read(self, tool, args, obs):
        skip = [self.roles.get("offer") or self.roles.get("open")]
        skip += [self.roles.get(r) for r in ("compare", "ask", "cart", "cart_view", "buy", "refuse")]
        if tool in [x for x in skip if x]:
            return
        self.query_fresh(tool, args)
        if isinstance(obs, (dict, list)) and not fail_only(obs):
            mk = self.canon_mark(tool, args)
            self.mem_results[mk] = obs
            self.mem_args[mk] = (tool, dict(args))

    def query_fresh(self, tool, args):
        kind = cjson({k: v for k, v in args.items() if not is_whole(v)})
        mem = self.page_queries.setdefault(tool, set())
        new = kind not in mem
        mem.add(kind)
        return new

    def low_value(self, tool, arg):
        p = ((self.schemas.get(tool) or {}).get("properties") or {}).get(arg)
        mn = p.get("minimum") if isinstance(p, dict) else None
        return mn if is_whole(mn) else 1

    @soft("")
    def last_page_first(self, tool, args, obs):
        if tool in self.page_arg or not isinstance(obs, dict):
            return ""
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        cands = []
        for k, v in args.items():
            if not is_whole(v) or k not in props:
                continue
            mn = props[k].get("minimum") if isinstance(props[k], dict) else None
            if v > (mn if is_whole(mn) else 1):
                cands.append(k)
        if len(cands) != 1 or self.left() <= self.reserve() + 4:
            return ""
        c = cands[0]
        shp = self.shape_of(tool, obs)
        if not shp or not shp.get("count") or not shp.get("next") or obs.get(shp["next"]) is not None:
            return ""
        rows = obs.get(shp["rows"])
        total = obs.get(shp["count"])
        if not isinstance(rows, list) or not is_whole(total) or total <= len(rows):
            return ""
        low = self.low_value(tool, c)
        a = dict(args)
        a[c] = low
        a, f = shape_args(self.schemas.get(tool) or {}, a)
        if f:
            return ""
        r = self.env_call(tool, a)
        if r is None:
            return ""
        o2 = r.get("observation")
        self.observe(tool, a, o2)
        rows2 = o2.get(shp["rows"]) if isinstance(o2, dict) else None
        if not isinstance(rows2, list) or len(rows2) < len(rows):
            return ""
        first = set(self.ids_in(obs))
        if any(i in first for i in self.ids_in(o2)):
            return ""
        self.page_arg[tool] = c
        return ("\n[Cairn] %s is this search's page number, and the call just made displayed the final page. Page %s"
                " of the same search:\n%s" % (c, low, self.display(o2)))

    @soft(False)
    def names_code(self, args):
        said = set()
        for s in strings_of(args):
            said |= set(find_codes(s))
        if not said:
            return False
        known = set()
        for m in self.fit_index().values():
            known |= set(m["models"]) | set(m["works"])
        return bool(said & known)

    def first_page(self, tool, args):
        pa = self.page_arg.get(tool)
        if pa and is_whole(args.get(pa)):
            a = dict(args)
            a[pa] = self.low_value(tool, pa)
            return a
        return args

    @soft("")
    def shorter_search(self, tool, args, obs):
        tpl = self.value_template()
        if not tpl or tpl[0] != tool:
            return ""
        if self.left() <= self.reserve() + SHORTER_RESERVE or self.finished or not isinstance(obs, dict):
            return ""
        field = tpl[2]
        q = args.get(field)
        if not isinstance(q, str) or len(q.split()) < 3 or self.listing_records(obs):
            return ""
        for v in obs.values():
            if isinstance(v, (list, str)) and v:
                return ""
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return ""
        words = [w for w in q.split() if not find_codes(w)][:2]
        short = " ".join(words)
        if not short:
            return ""
        a = {k: v for k, v in args.items() if not is_whole(v)}
        a[field] = short
        a, f = shape_args(self.schemas.get(tool) or {}, a)
        if f:
            return ""
        r = self.env_call(tool, a)
        if r is None:
            return ""
        o2 = r.get("observation")
        self.observe(tool, a, o2)
        return ("\n[Cairn] No listing carries every one of those words; the identical search with the words '%s'"
                " gives:\n%s" % (short, self.display(o2)))

    @soft("")
    def opening_moves(self):
        tpl = self.value_template()
        if not tpl or self.max_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_MIN_ITEMS:
            return ""
        tool, fixed, field = tpl
        self.gap = OPENING_PAUSE
        stop = self.max_steps - self.reserve() - self.msgs_free() - OPEN_LATER_STEPS * len(self.items) - HUNT_RESERVE // 2
        lines = []
        ref_texts = set(t for _, t in self.refs)
        for ix, it in enumerate(self.items, 1):
            vals = []
            for m in it["musts"]:
                if ":" not in m or m in ref_texts:
                    continue
                v = m.split(":", 1)[1].strip()
                if v and len(v.split()) <= 3 and not V3_OPEN_EXCL.search(v):
                    vals.append(v)
            nw = it["name"].split()
            noun = it["name"] if len(nw) <= 3 else " ".join(nw[-2:])
            tries = []
            if vals:
                tries.append(" ".join(vals[:2] + [noun]))
            if noun not in tries:
                tries.append(noun)
            for t in tries:
                a = dict(fixed)
                a[field] = t
                a, f = shape_args(self.schemas.get(tool) or {}, a)
                if f or self.finished or self.steps >= stop:
                    break
                r = self.env_call(tool, a)
                if r is None:
                    self.gap = STORE_PAUSE
                    return "\n".join(lines)
                obs = r.get("observation")
                self.observe(tool, a, obs)
                if self.listing_records(obs):
                    self.searched = (self.searched + [dict(a)])[-8:]
                    lines.append(t + ": " + self.display(obs))
                    plan = self.paging_plan(tool, obs)
                    if (plan and plan["next"] not in (None, "") and self.item_rows(ix) < OPEN_ROWS_PER_ITEM
                            and not self.finished and self.steps < stop):
                        a2 = dict(a)
                        a2[plan["arg"]] = plan["next"]
                        a2, f2 = shape_args(self.schemas.get(tool) or {}, a2)
                        if not f2:
                            r2 = self.env_call(tool, a2)
                            if r2 is not None:
                                o2 = r2.get("observation")
                                self.observe(tool, a2, o2)
                                lines.append("next page: " + self.display(o2))
                    if self.item_rows(ix) >= OPEN_ROWS_PER_ITEM:
                        break
                else:
                    lines.append(t + ": " + (("the store rejected the call: " + self.display(obs)) if fail_only(obs)
                                             else "no rows"))
        if self.rows and self.roles.get("ask") and self.msgs_free() > 0:
            found = [ix for ix in range(1, len(self.items) + 1) if self.item_rows(ix) >= 1]
            start = self.steps
            while (self.steps - start < self.max_steps // OPEN_WIDEN_SHARE and self.steps < stop
                   and not self.finished):
                if all(self.item_full(ix) >= OPEN_ROWS_PER_ITEM for ix in found):
                    break
                before = len(self.full)
                self.widen_compare()
                if len(self.full) <= before:
                    break
        self.gap = STORE_PAUSE
        for lid in self.full:
            lines.append("read in full: " + self.page_text(lid))
        ev = self.evidence(ids=list(self.full))
        if ev:
            lines.append(ev)
        return "\n".join(lines)

    def page_text(self, lid):
        rec = self.full.get(lid) or self.rows.get(lid) or {}
        ab = attr_block(rec) or {}
        a = cjson(clip_deep(ab, 120))[:PAGE_ATTR_CLIP]
        rest = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
        return lid + ": attributes " + a + " | " + cjson(clip_deep(rest, 200))[:PAGE_REST_CLIP]

    def observe(self, tool, args, obs):
        self.absorb(obs)
        lines = self.loose_lines(obs)
        if not lines:
            return ""
        meaning = self.read_note(lines)
        if meaning:
            self.queued.append("(seen on %s) %s" % (tool, meaning))
        return meaning or ""

    def widen_compare(self):
        cmp = self.roles.get("compare")
        sc = self.schemas.get(cmp) or {} if cmp else {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        tgt = next((k for k, p in props.items() if isinstance(p, dict) and "array" in kinds(p)), None)
        if not cmp or not tgt or any(r != tgt for r in req) or self.compare_bare():
            return self.widen_single()
        if self.left() <= 12:
            return
        decl = props[tgt].get("maxItems")
        cap = min(decl if is_whole(decl) else 4, self.compare_room())
        groups = self.item_groups()
        groups = sorted(groups, key=lambda g: sum(1 for i in g if i in self.full))
        lanes = [[i for i in g if i not in self.full] for g in groups]
        picks = []
        for rnk in range(cap):
            for lane in lanes:
                if rnk < len(lane):
                    picks.append(lane[rnk])
        picks = picks[:cap]
        if len(picks) < 2:
            self.fetch_facts(picks, 12)
            return
        a, f = shape_args(sc, {tgt: picks})
        if f:
            return
        before = len(self.full)
        r = self.env_call(cmp, a)
        recs = []
        if r is not None:
            obs = r.get("observation")
            self.observe(cmp, a, obs)
            self.record_view_keys(cmp, obs)
            recs = self.listing_records(obs)
            if recs:
                self.facts[cmp] = bool(self.facts.get(cmp) is True or any(attr_block(x) for _, x in recs))
                if cmp not in self.facts_order:
                    self.facts_order.append(cmp)
                self.cmp_ok = max(self.cmp_ok, len(recs))
            else:
                self.cmp_fail = min(self.cmp_fail or 999, len(picks))
        if recs and self.facts.get(cmp) is False:
            return self.widen_single()
        if len(self.full) == before and len(picks) > 2 and self.left() > 12:
            a2, f2 = shape_args(sc, {tgt: picks[:2]})
            if not f2:
                r2 = self.env_call(cmp, a2)
                if r2 is not None:
                    o2 = r2.get("observation")
                    self.observe(cmp, a2, o2)
                    self.cmp_ok = max(self.cmp_ok, len(self.listing_records(o2)))

    def widen_single(self):
        ft = self.facts_tool()
        tgt = self.single_target(ft) if ft else None
        if not ft or not tgt:
            return
        wanted = set()
        for ix in range(1, len(self.items) + 1):
            for r in self.live_reqs(ix):
                val = r.split(":", 1)[1] if ":" in r else r
                wanted |= set(E4_RUN3.findall(val.lower()))
        size = WIDEN_PER_ITEM if len(self.items) <= 2 else 4
        lanes, heads = [], []
        for g in self.item_groups():
            opened = sum(1 for i in g if i in self.full)
            pool = [i for i in g if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price(i) if self.price(i) is not None else float("inf"))
            pool.sort(key=lambda i: -len(wanted & set(E4_RUN3.findall(cjson(self.rows.get(i)).lower()))))
            lanes.append(pool[:max(0, size - opened)])
            heads.append(max(0, 4 - opened))
        first, more = [], []
        for rnk in range(4):
            for lane, h in zip(lanes, heads):
                if rnk < h and rnk < len(lane):
                    first.append(lane[rnk])
        for rnk in range(size):
            for lane, h in zip(lanes, heads):
                if rnk >= h and rnk < len(lane):
                    more.append(lane[rnk])
        cut = max(5, 2 * len(lanes) + 2)
        got = self.fetch_facts(first[:min(cut, 8)], 22)
        rest = (first[cut:] + more)[:max(0, 10 - len(got))]
        if rest:
            self.fetch_facts(rest, 30)

    def ask_ready(self):
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or self.clock() > ASK_DEADLINE:
            return False
        if self.turns < TURNS_BEFORE_ASK and any(self.item_full(ix) == 0 for ix in range(1, len(self.items) + 1)):
            return False
        if len(self.full) >= 3 * len(self.items) + 1:
            return True
        return self.steps >= max(8, self.max_steps // 4)


E5_EXCL = re.compile(r"\b(?:not|no|other\s+than|unlike|alongside|listing|page|shown|at\s+most|at\s+least|under|over|below|above|less|more|between|within|up\s+to|or)\b", re.I)
E5_LETTER = re.compile(r"[a-zA-Z]")
E6_MEASURED = re.compile(r"^\s*[\d.,/x \-]+[a-z]{0,6}\s*$")
E6_MODELS = re.compile(r"[a-z0-9][ _]models?\b")
E6_CUT = re.compile(r"\s(?:and|or)\s|/|,|;|\(")
E6_PLAIN_S = re.compile(r"(?<=[a-z]{3})s$")
E6_AND = re.compile(r"\band\b", re.I)
E6_PART_CUT = re.compile(r"\s*(?:,|\band\b)\s*", re.I)
E6_PLURAL = re.compile(r"[a-z]{2}[a-rtv-z]s$", re.I)
E6_FILLER = re.compile(r"\b(?:and|for|like|me|or|please|tell|the|what|which|would|you)\b", re.I)
E6_DIGIT = re.compile(r"\d")
E6_RUNS = re.compile(r"[a-z0-9]+")


def plain_spell(s):
    return " ".join(H_OUR.sub("or", E6_PLAIN_S.sub("", w)) for w in word_form(s).split())


def is_catch(k):
    return not V16_NOTCATCH.search(k) and bool(V16_CATCH.search(k))


class EpisodeInterview:
    def msg_args(self, text):
        tool = self.roles.get("ask")
        sc = self.schemas.get(tool) if tool else None
        sc = sc if isinstance(sc, dict) else {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p)
                 and "enum" not in p and "const" not in p]
        pool = [k for k in req if k in texts] or texts
        args = {}
        if pool:
            def score(ix_k):
                ix, k = ix_k
                u = under_form(k)
                ml = props[k].get("maxLength")
                ml = ml if isinstance(ml, (int, float)) and not isinstance(ml, bool) else 1000000
                return (not V2_NONMSG.search(u), bool(V1_MSG.search(u)), ml, -ix)
            best = max(enumerate(pool), key=score)[1]
            args[best] = text
        else:
            for k, p in props.items():
                if not isinstance(p, dict):
                    continue
                if "array" in kinds(p):
                    args[k] = [text]
                    break
                ip = p.get("properties")
                if isinstance(ip, dict):
                    sk = next((n for n, q in ip.items() if isinstance(q, dict) and "string" in kinds(q)), None)
                    if sk:
                        args[k] = {sk: text}
                        break
        if not args:
            return None
        for k in req:
            if k not in args:
                f = filler(props.get(k) or {})
                if f is None:
                    return None
                args[k] = f
        args, faults = shape_args(sc, args)
        if faults:
            return None
        return args

    def reset_lanes(self, keep_follow=False):
        for ix in range(1, len(self.items) + 1):
            self.lane_found[ix] = 0
            self.lane_spent[ix] = 0
            self.lane_used[ix] = set()
        if not keep_follow:
            self.follow_ups = []

    def interview(self):
        self.asked = True
        if self.msgs_free() <= 0:
            return ""
        self.widen_compare()
        count = self.msg_limit - self.msgs_used
        if self.listing and self.list_now() > 1 and (len(self.items) > 1 or self.msg_limit > SCARCE_MESSAGES or True):
            count = max(count, 8)
        self.plan_lanes(count)
        self.reset_lanes()
        sent_all, told = [], []
        groups = []
        while self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.clock() < QUESTION_DEADLINE:
            sent = []
            while groups and self.listing and not sent:
                sent = self.group_ask(groups.pop(0), told)
            if not sent:
                sent = self.ask_next(told)
            if not sent:
                break
            sent_all += sent
        found = [ix for ix in range(1, len(self.items) + 1) if self.lane_found.get(ix, 0) > 0]
        early = self.early_trail()
        vs = self.value_search(found) if found else ""
        if told:
            self.version += 1
        lines = ["[Cairn] Cairn's interviewer questioned the shopper as shown below, one step per question."]
        for _, q, a in sent_all:
            lines += ["Q: " + q, "A: " + a]
        lines.append("New firm requirements from the answers: " + ("; ".join(told) or "none")
                     + ". You must apply them to every pick, with a fresh search on the new values wherever one is"
                       " needed.")
        rt = [t for _, t in self.refs if any(t in l for l in told)]
        if rt:
            lines.append(self.ref_sentence(rt, True))
        if early:
            lines.append(early)
        if vs:
            lines.append(vs)
        if self.queued:
            lines.append("[notes seen in the meantime] " + " | ".join(self.queued))
            self.queued = []
        return "\n".join(lines)

    def ref_sentence(self, needs, full):
        t = ("These needs get their value from a different listing: " + " | ".join(needs) + ". Locate that listing,"
             " read its whole page, and then open every further listing to which that page directs onward (only the"
             " listings the page points onward to, not every listing it happens to mention); the final page along"
             " this chain supplies the value")
        if full:
            t += ", and an order is valid only once every one of them has been opened"
        return t + "."

    def late_question(self):
        if self.late_asked or not self.roles.get("ask") or self.msgs_used >= self.msg_limit:
            return ""
        if any(d["stage"] == "order" and d["messages"] >= 1 for d in self.duties):
            return ""
        if self.left() < 8 or self.clock() > LATE_ASK_DEADLINE or not self.judges():
            return ""
        self.late_asked = True
        if not self.lanes:
            self.plan_lanes(self.list_cap() if self.listing else 1)
            self.reset_lanes(keep_follow=True)
        old = self.budget
        told = []
        sent = self.ask_next(told, 6)
        if not sent:
            self.trace.add({"kind": "late_question", "sent": False})
            return ""
        if self.budget != old:
            told.append("budget now %s" % self.budget)
        if not told:
            return ""
        self.version += 1
        ix, q, a = sent[0]
        t = ("[Cairn] Cairn put a further question to the shopper ahead of the checkout, at a cost of one step.\nQ: "
             + q + "\nA: " + a + "\nNew firm requirements: " + "; ".join(told)
             + "\nThe checkout did not happen. Test every cart line against these requirements (search again on the"
               " new values where necessary), repair the cart, and after that check out.")
        rt = [x for _, x in self.refs if any(x in l for l in told)]
        if rt:
            t += "\n" + self.ref_sentence(rt, False)
        vs = self.value_search([ix])
        if vs:
            t += "\n" + vs
        return t

    def labels_now(self):
        ck = self.cat_key()
        if ck is None:
            return set()
        return set(str(r.get(ck)).strip().lower() for r in self.rows.values() if r.get(ck) is not None)

    def shelf_like(self, v):
        labs = self.labels_now()
        targets = [it["shelf"] for it in self.items] + [it["name"] for it in self.items] + list(labs)
        wv = word_form(v)

        def w4(s):
            return set(w[:-1] if w.endswith("s") else w for w in word_form(s).split() if len(w) >= 4)
        sv = w4(v)
        for t in targets:
            if not t:
                continue
            if wv == word_form(t) or (sv & w4(t)):
                return True
        return False

    def search_template(self):
        tool = self.roles.get("search")
        if not tool:
            return None
        sc = self.schemas.get(tool) or {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")]
        named = [k for k in texts if V14_QUERY.search(under_form(k))]
        labs = self.labels_now()
        for s in reversed(self.searched):
            cands = [k for k in (named or texts) if isinstance(s.get(k), str) and E5_LETTER.search(s[k])]
            if not named:
                ok = []
                for k in cands:
                    vals = [x.get(k) for x in self.searched if k in x]
                    if len(set(cjson(v) for v in vals)) < 2:
                        continue
                    if not any(isinstance(v, str) and len(v.split()) >= 2 for v in vals):
                        continue
                    if any(isinstance(v, str) and v.strip().lower() in labs for v in vals):
                        continue
                    ok.append(k)
                cands = ok
            if not cands:
                continue
            field = cands[0]
            kept = {}
            for k, v in s.items():
                if k == field:
                    continue
                p = props.get(k)
                if (isinstance(p, dict) and p.get("enum")) or k in req:
                    kept[k] = v
            for k in list(kept):
                if isinstance(kept[k], str) and k not in req and self.shelf_like(kept[k]):
                    del kept[k]
            for k in list(kept):
                p = props.get(k)
                if not (isinstance(p, dict) and p.get("enum")):
                    continue
                cnt, order = {}, []
                for x in self.searched:
                    if k in x and isinstance(x.get(field), str) and E5_LETTER.search(x[field]):
                        c = cjson(x[k])
                        if c not in cnt:
                            order.append(c)
                        cnt[c] = cnt.get(c, 0) + 1
                if order:
                    best = max(order, key=lambda c: (cnt[c], -order.index(c)))
                    kept[k] = json.loads(best)
            return tool, kept, field
        return None

    def value_template(self):
        st = self.search_template()
        if st:
            return st
        tool0 = self.roles.get("search")
        if not tool0:
            return None
        sw = self.brief.get("search_words")
        if isinstance(sw, dict):
            t = sw.get("tool") if sw.get("tool") in self.schemas else tool0
            f = sw.get("field")
            pt = (self.schemas.get(t) or {}).get("properties") or {}
            if isinstance(f, str) and isinstance(pt.get(f), dict) and "string" in kinds(pt[f]) and not pt[f].get("enum"):
                fixed = sw.get("args") if isinstance(sw.get("args"), dict) else {}
                a = dict(fixed)
                a[f] = "x"
                a2, faults = shape_args(self.schemas.get(t) or {}, a)
                if not faults:
                    return t, {k: v for k, v in a2.items() if k != f}, f
        sc = self.schemas.get(tool0) or {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        words = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")
                 and V14_QUERY.search(under_form(k))]
        last = self.searched[-1] if self.searched else None
        optsets = [k for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list)
                   and len(p["enum"]) >= 2]
        modes = {}
        for k in optsets:
            en = props[k]["enum"]
            q = [o for o in en if isinstance(o, str) and V14_QUERY.search(o)]
            if len(q) == 1:
                modes[k] = q[0]
            elif k in req and len(en) == 2 and last is not None and k in last:
                diff = [o for o in en if o != last[k]]
                if len(diff) == 1:
                    modes[k] = diff[0]
        if len(words) != 1 or len(modes) > 1 or (optsets and not modes):
            return None
        base = {k: last[k] for k in req if last and k in last and k != words[0]}
        base.update(modes)
        return tool0, base, words[0]

    def value_words(self, ix):
        out = []
        refs = set(t for _, t in self.refs)
        it = self.items[ix - 1]
        for r in self.live_reqs(ix):
            if r in refs or ":" not in r:
                continue
            v = r.split(":", 1)[1].strip()
            fresh = r in it["told"]
            m = V13_FLOOR.search(v)
            if m:
                if fresh:
                    out.append((m.group(1).strip(), True))
                continue
            if not v or E5_EXCL.search(v):
                continue
            out.append((v, fresh))
        return out

    def value_search(self, ixs):
        tpl = self.value_template()
        if not tpl or self.left() < self.reserve() + VALUE_ROOM:
            return ""
        tool, base, field = tpl
        allow = min(VALUE_RESERVE, self.left() - self.reserve() - VALUE_RESERVE)
        ck = self.cat_key()
        lines = []
        for ix in ixs:
            ph = self.value_words(ix)
            if not any(f for _, f in ph) or allow <= 2:
                continue
            lead = [p for p in ph if p[1] and (len(p[0].split()) == 1 or V12_AMOUNT_RE.match(p[0]))]
            order = lead + [p for p in ph if p not in lead]
            reqpos = [i for i, p in enumerate(order) if not p[1]]
            drop = reqpos[-1] if reqpos else 0
            vals = [p[0] for p in order]
            if len(vals) >= 2:
                tries = [vals, [v for i, v in enumerate(vals) if i != drop]]
            elif lead:
                tries = [[vals[0], self.items[ix - 1]["name"]], [vals[0]]]
            else:
                tries = [vals]
            for t in tries[:2]:
                a = dict(base)
                a[field] = " ".join(" ".join(t).split())
                a, f = shape_args(self.schemas.get(tool) or {}, a)
                if f or self.finished:
                    break
                r = self.env_call(tool, a)
                allow -= 1
                if r is None:
                    break
                obs = r.get("observation")
                self.observe(tool, a, obs)
                ids = self.ids_in(obs)
                kept = [i for i in ids if self.buyable(i)
                        and (ck is None or self.owns_label(ix, (self.rows.get(i) or {}).get(ck)))]
                if isinstance(self.budget, (int, float)):
                    ps = [self.price(i) for i in ids]
                    ignore = bool(ps) and all(p is not None for p in ps) and all(p > 10 * self.budget for p in ps)
                    if not ignore:
                        kept = [i for i in kept if self.price(i) is None or self.price(i) <= self.budget]
                home = self.home_label(ix)
                if ck is not None and home:
                    kept.sort(key=lambda i: 0 if word_form((self.rows.get(i) or {}).get(ck) or "") == home else 1)
                lines.append(json.dumps(a.get(field)) + ": " + (", ".join(kept[:6]) if kept
                                                                 else "no row of this item's kind came back"))
                if kept:
                    picks = [i for i in kept if i not in self.full][:max(0, min(3, allow))]
                    opened = self.fetch_facts(picks, self.reserve() + VALUE_RESERVE) if picks else []
                    allow -= len(opened)
                    if opened:
                        lines.append(("opened: " + " | ".join(opened))[:2400])
                    break
        if not lines:
            return ""
        self.trace.add({"kind": "value_search", "lines": len(lines)})
        return ("[Cairn searched all the requirement values of each such item at once and read the leading rows of"
                " the right kind]\n" + "\n".join(lines))

    def home_label(self, ix):
        ck = self.cat_key()
        if ck is None:
            return ""
        it = self.items[ix - 1]
        target = word_form(it["shelf"]) if it["shelf"] else word_form(it["name"])
        for pool in (list(self.full.values()), list(self.rows.values())):
            cnt, order = {}, []
            for rec in pool:
                lab = rec.get(ck)
                if lab is None or not self.owns_label(ix, lab):
                    continue
                w = word_form(str(lab))
                if w not in cnt:
                    order.append(w)
                cnt[w] = cnt.get(w, 0) + 1
            if target in cnt:
                return target
            if cnt:
                return max(order, key=lambda w: (cnt[w], -order.index(w)))
        return ""

    def ask_groups(self):
        n = len(self.items)
        order = [ix for ix in self.plan_order if self.lanes.get(ix)]
        order += [ix for ix in range(1, n + 1) if ix not in order and self.lanes.get(ix)]
        free = self.msgs_free()
        if free <= 0 or len(order) <= free:
            return []
        groups = []
        left_items = list(order)
        for g in range(free, 0, -1):
            size = -(-len(left_items) // g)
            groups.append(left_items[:size])
            left_items = left_items[size:]
        return [g for g in groups if g]

    def list_cap(self):
        return 10 if len(self.items) > 1 else 8

    def list_now(self):
        one = len(self.items) == 1
        if self.msg_limit <= SCARCE_MESSAGES:
            return 1 if one else 8
        if one:
            return 5 if self.msg_limit == 3 else 3
        return 8 if self.list_asks == 0 else 10

    def remaining(self, ix):
        used = self.lane_used.setdefault(ix, set())
        return [l for l in self.lanes.get(ix, []) if word_form(l) not in used and not self.told_about(ix, l)]

    def group_ask(self, group, told, floor=6):
        now, cap = self.list_now(), self.list_cap()
        share = max(1, now // len(group)) if now < cap else max(MIXED_LABELS, cap // len(group))
        parts = [(ix, self.remaining(ix)[:share]) for ix in group if self.remaining(ix)]
        if not parts:
            return []
        if len(parts) == 1:
            ix, labs = parts[0]
            fl, text = self.fit_ask(ix, labs, False)
            if not text:
                return []
            sent = self.send([(ix, fl[0], text)], floor)
            if sent:
                self.finish_ask(ix, fl, False, sent, told)
            return sent
        found = None
        while parts:
            for c in range(share, 0, -1):
                cut = [(ix, l[:c]) for ix, l in parts]
                text = self.group_text(cut)
                a = self.msg_args(text)
                if a is not None and says(a, text):
                    found = (cut, text)
                    break
            if found:
                break
            parts = parts[:-1]
        if not found:
            return []
        parts, text = found
        sent = self.send([(parts[0][0], parts[0][1][0], text)], floor)
        if not sent:
            return []
        for ix, _ in parts:
            self.lane_spent[ix] = self.lane_spent.get(ix, 0) + 1
        self.list_asks += 1
        before = dict((j, len(it["told"])) for j, it in enumerate(self.items, 1))
        reading = self.read_answers([(0, sent[0][1], sent[0][2])], [LIGHT, HEAVY])
        reading.pop("free", None)
        self.fold(reading, told)
        crowded = 1 in [count_int(x) for x in items_of(reading.get("crowded"))]
        answered = False
        for ix, labs in parts:
            new = self.items[ix - 1]["told"][before[ix]:]
            if new:
                answered = True
                self.lane_found[ix] = self.lane_found.get(ix, 0) + 1
                self.settle(ix, labs, False, new, False)
        if crowded and not answered:
            self.listing = False
            self.trace.add({"kind": "lists_off", "by": "group"})
            return sent
        if not answered:
            self.lane_used.setdefault(parts[0][0], set()).add(word_form(parts[0][1][0]))
        self.trace.add({"kind": "group_ask", "parts": parts, "answered": answered})
        return sent

    def finish_ask(self, ix, labels, shown, sent, told):
        self.lane_spent[ix] = self.lane_spent.get(ix, 0) + 1
        self.list_asks += 1
        before = dict((j, len(it["told"])) for j, it in enumerate(self.items, 1))
        reading = self.read_answers([(ix, sent[0][1], sent[0][2])], [LIGHT, HEAVY])
        if len(labels) > 1:
            reading.pop("free", None)
        self.fold(reading, told)
        new = []
        for j, it in enumerate(self.items, 1):
            nj = it["told"][before[j]:]
            if nj:
                self.lane_found[j] = self.lane_found.get(j, 0) + 1
                new += nj
        crowded = 1 in [count_int(x) for x in items_of(reading.get("crowded"))]
        self.settle(ix, labels, shown, new, crowded)

    def next_ask(self):
        scarce = self.msg_limit <= SCARCE_MESSAGES and not self.listing
        n = len(self.items)

        def key(ix):
            po = self.plan_order.index(ix) if ix in self.plan_order else len(self.plan_order) + ix
            return (self.lane_found.get(ix, 0) > 0, 0 if scarce else self.lane_spent.get(ix, 0), po)
        for ix in sorted(range(1, n + 1), key=key):
            rem = self.remaining(ix)
            if rem:
                return ix, (rem[:self.list_now()] if self.listing else rem[:1])
        return None

    def told_about(self, ix, label):
        ls = spelled(label)
        for t in self.items[ix - 1]["told"]:
            if names_test(ls, spelled(t.split(":", 1)[0])):
                return True
        return False

    def lookalikes(self, ix, label):
        ck = self.cat_key()
        kv, order = {}, []
        for i in self.seen:
            if ck is not None and not self.owns_label(ix, (self.rows.get(i) or {}).get(ck)):
                continue
            if i in self.full:
                for k, v in (attr_block(self.full[i]) or {}).items():
                    w = word_form(k)
                    if w not in kv:
                        kv[w] = set()
                        order.append(w)
                    kv[w] |= roots(str(v))
        wc = {}
        for k in order:
            for w in set(k.split()):
                wc[w] = wc.get(w, 0) + 1
        lw = word_form(label)
        plainw = [w for w in lw.split() if wc.get(w, 0) <= 2]
        own = kv.get(lw, set())
        sc = []
        for k in order:
            if not k or k == lw:
                continue
            shared = [w for w in plainw if w in k.split()]
            common = own & kv[k]
            if shared or common:
                sc.append((3 * len(shared) + min(3, len(common)), k))
        sc.sort(key=lambda x: -x[0])
        return [k for _, k in sc][:LOOKALIKE_KEYS]

    def q_text(self, ix, labels, shown):
        name = self.asked_name(ix)
        if len(labels) >= 2:
            return "What %s or %s would you like for the %s?" % (", ".join(labels[:-1]), labels[-1], name)
        lab = labels[0]
        ap = (self.apart.get(lab) or {}).get(ix)
        if ap:
            return "Apart from the %s%s, which %s would you like for the %s?" % (
                ap[0], (" and the " + ap[1]) if len(ap) > 1 else "", lab, name)
        if shown:
            ex = self.examples(ix, lab)
            return "What %s would you like for the %s, for example %s or %s?" % (lab, name, ex[0], ex[1])
        return "What %s would you like for the %s?" % (lab, name)

    def group_text(self, parts):
        cl = []
        for ix, labs in parts:
            piece = ("%s or %s" % (", ".join(labs[:-1]), labs[-1])) if len(labs) >= 2 else labs[0]
            cl.append("what %s for the %s" % (piece, self.asked_name(ix)))
        t = ", and ".join(cl)
        t = "W" + t[1:]
        return t.replace(" for the ", " would you like for the ", 1) + "?"

    def fit_ask(self, ix, labels, shown):
        for c in range(len(labels), 0, -1):
            labs = labels[:c]
            t = self.q_text(ix, labs, shown)
            a = self.msg_args(t)
            if a is not None and says(a, t):
                return labs, t
        return [], ""

    def ask_next(self, told, floor=6):
        got = None
        for _ in range(LIST_ONE):
            self.follow_ups = [f for f in self.follow_ups if not self.told_about(f[0], f[1][0])]
            if self.follow_ups:
                ix, labs, shown = self.follow_ups.pop(0)
            else:
                na = self.next_ask()
                if not na:
                    return []
                ix, labs = na
                shown = False
            if not labs:
                return []
            fl, text = self.fit_ask(ix, labs, shown)
            if not text and shown:
                fl, text = self.fit_ask(ix, labs, False)
            if text:
                got = (ix, fl, shown, text)
                break
            self.lane_used.setdefault(ix, set()).add(word_form(labs[0]))
            self.trace.add({"kind": "too_long", "item": ix, "label": labs[0]})
        if not got:
            return []
        ix, fl, shown, text = got
        sent = self.send([(ix, fl[0], text)], floor)
        if not sent:
            return []
        self.finish_ask(ix, fl, shown, sent, told)
        return sent

    def settle(self, ix, labels, shown, new, crowded):
        used = self.lane_used.setdefault(ix, set())
        if len(labels) == 1:
            used.add(word_form(labels[0]))
            if crowded and self.listing:
                self.listing = False
                self.trace.add({"kind": "lists_off", "item": ix})
            if new or not crowded:
                return
            if ix not in self.one_product and self.name_parts(ix):
                self.one_product.add(ix)
                self.follow_ups.insert(0, (ix, [labels[0]], shown))
                self.trace.add({"kind": "one_product", "item": ix})
                return
            if not self.apart:
                la = self.lookalikes(ix, labels[0])
                if la:
                    self.apart.setdefault(labels[0], {})[ix] = la
                    self.follow_ups.insert(0, (ix, [labels[0]], False))
                    self.trace.add({"kind": "set_apart", "item": ix, "keys": la})
            return
        if crowded:
            self.listing = False
        if crowded and not new:
            self.listing = False
            self.follow_ups.insert(0, (ix, [labels[0]], False))
            self.trace.add({"kind": "lists_off", "item": ix})
            return
        hit = self.hit_pos(labels, new) if new else None
        if hit is not None:
            ahead, done = labels[hit + 1:], labels[:hit + 1]
        else:
            ahead, done = labels[1:][::-1], labels[:1]
        for l in done:
            used.add(word_form(l))
        rest = [l for l in self.lanes.get(ix, []) if l not in labels]
        self.lane_misses[ix] = 0 if hit is not None else self.lane_misses.get(ix, 0) + 1
        m = self.lane_misses[ix]
        if m and m % 2 == 0:
            mid = (len(ahead) - 1) / 2.0
            order = sorted(range(len(ahead)), key=lambda i: (abs(i - mid), i))
            self.lanes[ix] = rest + [ahead[i] for i in order]
        else:
            self.lanes[ix] = ahead + rest
        self.trace.add({"kind": "settled", "item": ix, "hit": hit})

    def key_rows(self):
        out = []
        for label, kept, _ in self.key_slots():
            for s in kept:
                out.append({"label": label, "key": s["wf"], "count": s["count"],
                            "vals": [d[:40] for d in s["distinct"][:4]]})
        return out

    def hit_pos(self, labels, needs):
        pooled = {}
        for r in self.key_rows():
            pooled.setdefault(r["key"], set()).update(roots(" ".join(r["vals"])))
        pos = []
        for n in needs:
            prop, _, val = n.partition(":")
            ps = spelled(prop)
            sc = [len(spelled(l) & ps) for l in labels]
            if not sc or max(sc) == 0:
                vr = roots(val)
                sc = [len(pooled.get(word_form(l), set()) & vr) for l in labels]
            if sc and max(sc) > 0:
                pos.append(sc.index(max(sc)))
        return min(pos) if pos else None

    def ranked_keys(self, ix, rows):
        settled = [stems(m.split(":", 1)[0]) for m in self.items[ix - 1]["musts"]]
        own = [r for r in rows if self.owner(r["label"]) == ix]
        if len(self.items) == 1 and not own:
            own = list(rows)
        measured = {}
        for r in rows:
            vals = r["vals"]
            measured[r["key"]] = sum(1 for v in vals if E6_MEASURED.match(v)) * 2 > len(vals)
        own = sorted(own, key=lambda r: (-r["count"], measured.get(r["key"], False), -len(r["vals"])))
        keys = [r["key"] for r in own]
        keys = [k for k in keys if not any(names_test(s, stems(k)) for s in settled)]
        prop = [k for k in keys if not V15_NONPROP.search(k) or E6_MODELS.search(k)]
        dem = [k for k in keys if k not in prop]
        return [k for k in prop if not is_catch(k)] + [k for k in prop if is_catch(k)] + dem

    def adopt_strays(self):
        ck = self.cat_key()
        if len(self.items) < 2 or ck is None:
            return
        self.part_shelves()
        labels = []
        for rec in self.full.values():
            v = rec.get(ck)
            if v is not None and str(v) and str(v) not in labels:
                labels.append(str(v))
        strays = [l for l in labels if not self.owner(l)]
        orphans = [ix for ix in range(1, len(self.items) + 1)
                   if not any(self.owns_label(ix, l, False) for l in labels)]
        if strays and len(orphans) == 1:
            for l in strays:
                self.adopted[l] = orphans[0]
            self.trace.add({"kind": "adopt", "item": orphans[0], "labels": strays})

    def owner_raw(self, label):
        wl = word_form(label)
        lr = roots(label)
        best, bs = 0, 0
        for ix, it in enumerate(self.items, 1):
            if wl == word_form(it["shelf"]):
                s = 3
            elif wl == word_form(it["name"]):
                s = 2
            else:
                ir = roots(it["shelf"]) | roots(it["name"])
                s = len(lr & ir) / float(max(1, len(lr | ir)))
            if s > bs:
                best, bs = ix, s
        return best if bs > 0 else 0

    def owner(self, label, adopted=True):
        if label is None or isinstance(label, (dict, list)):
            return 0
        label = str(label)
        if not label:
            return 0
        if not adopted:
            return self.owner_raw(label)
        if label in self.adopted:
            return self.adopted[label]
        key = (label, len(self.items))
        if key not in self.owners_seen:
            self.owners_seen[key] = self.owner_raw(label)
        return self.owners_seen[key]

    def owns_label(self, ix, label, adopted=True):
        if label is None or isinstance(label, (dict, list)):
            return False
        label = str(label)
        if not label or not (1 <= ix <= len(self.items)):
            return False
        if self.owner(label, adopted) == ix:
            return True
        return word_form(label) == word_form(self.items[ix - 1]["shelf"])

    def item_full(self, ix):
        ck = self.cat_key()
        if ck is None:
            return 0
        return sum(1 for rec in self.full.values() if attr_block(rec) and self.owns_label(ix, rec.get(ck)))

    def plan_lanes(self, count):
        self.adopt_strays()
        rows = self.key_rows()
        n = len(self.items)
        parts = [PLAN_PROMPT.replace("COUNT", str(max(1, count))), "REQUEST:\n" + self.request]
        il = []
        for ix, it in enumerate(self.items, 1):
            il.append("item %d: %s - the request already settles: %s" % (ix, it["name"], "; ".join(it["musts"]) or "nothing"))
        parts.append("ITEMS:\n" + "\n".join(il))
        af = [str(x).strip() for x in items_of(self.brief.get("ask_first")) if str(x).strip()]
        if af:
            parts.append("ASK FIRST (the request wants the shopper asked about these):\n" + "\n".join(af))
        if rows:
            kl, prev = [], None
            for num, r in enumerate(rows, 1):
                if r["label"] != prev:
                    kl.append("category %s:" % repr(str(r["label"])))
                    prev = r["label"]
                line = "  %d. %s (on %d): %s" % (num, r["key"], r["count"], " | ".join(r["vals"]))
                line += self.not_stated(r["label"], r["key"], r["vals"])
                if is_catch(r["key"]):
                    line += " [lumps several features together; ask last]"
                kl.append(line)
            parts.append("ATTRIBUTE KEYS SEEN:\n" + "\n".join(kl))
        else:
            parts.append("ATTRIBUTE KEYS SEEN:\n(no listing has been read in full yet)")
        ans = self.ask_json("\n\n".join(parts), [HEAVY, LIGHT], 3500) or {}
        picks = ans.get("ask")
        picks = [picks] if isinstance(picks, dict) else (picks if isinstance(picks, list) else [])
        raw = dict((ix, []) for ix in range(1, n + 1))
        hinted = dict((ix, set()) for ix in range(1, n + 1))
        for p in picks:
            if not isinstance(p, dict):
                continue
            k = count_int(p.get("item"))
            k = k if k else 1
            k = min(max(k, 1), n)
            lab = E6_CUT.split(str(p.get("label") or ""), 1)[0].strip().rstrip("?.").strip()
            if not lab or len(lab.split()) > 5:
                continue
            if lab not in raw[k]:
                raw[k].append(lab)
            h = p.get("hint")
            if isinstance(h, str) and h.strip():
                hinted[k].add(word_form(lab))
                if k not in self.plan_order:
                    self.plan_order.append(k)
        for ix in range(1, n + 1):
            keys = self.ranked_keys(ix, rows)
            reads = max(1, self.item_full(ix))
            shares = {}
            for r in rows:
                if self.owner(r["label"]) == ix:
                    shares[r["key"]] = max(shares.get(r["key"], 0), r["count"] / float(reads))

            def sp(l):
                pl = plain_spell(l)
                for k in shares:
                    if plain_spell(k) == pl:
                        return k
                return l

            def share(l):
                w = word_form(l)
                if w in shares:
                    return shares[w]
                c = [v for k, v in shares.items() if names_test(stems(k), stems(l)) or names_test(stems(l), stems(k))]
                return max(c) if c else 0
            hs = set(word_form(sp(h)) for h in hinted[ix])
            lane, seen = [], set()
            for l in raw[ix]:
                s = sp(l)
                if word_form(s) not in seen:
                    seen.add(word_form(s))
                    lane.append(s)
            guard = [k for k in keys if not is_catch(k) and shares.get(k, 0) >= GUARD_SHARE][:GUARD_KEYS]
            for k in guard:
                top = [word_form(x) for x in lane[:3]]
                if word_form(k) in top:
                    continue
                ks = shares.get(k, 0)
                weak = next((x for x in lane[:3] if word_form(x) not in hs and share(x) + GUARD_GAP <= ks), None)
                if weak is None and len(lane) >= 3:
                    continue
                new = [x for x in lane if word_form(x) != word_form(k)]
                if weak is not None:
                    wi = new.index(weak)
                    if n > 1:
                        new[wi] = k
                        if len(new) >= 3:
                            new.insert(3, weak)
                        else:
                            new.append(weak)
                    else:
                        new.insert(wi, k)
                else:
                    new.append(k)
                tops = [word_form(y) for y in new[:3]]
                if any(word_form(x) in hs and word_form(x) not in tops for x in lane[:3]):
                    continue
                lane = new
            have = set(word_form(x) for x in lane)
            for k in keys:
                if word_form(k) not in have:
                    have.add(word_form(k))
                    lane.append(k)
            lane = [x for x in lane if not is_catch(word_form(x))] + [x for x in lane if is_catch(word_form(x))]
            self.lanes[ix] = lane
        self.trace.add({"kind": "plan", "lanes": self.lanes, "order": self.plan_order})

    def name_parts(self, ix):
        name = self.items[ix - 1]["name"]
        if not E6_AND.search(name):
            return []
        pieces = [p.strip() for p in E6_PART_CUT.split(name) if p.strip()]
        if len(pieces) >= 2 and all(E6_PLURAL.search(p) for p in pieces):
            return pieces
        return []

    def asked_name(self, ix):
        name = self.items[ix - 1]["name"]
        parts = self.name_parts(ix)
        if not parts:
            return name
        if ix not in self.one_product:
            return E6_AND.sub("or", name)
        ck = self.cat_key()
        rr = []
        for r in self.rows.values():
            if ck is not None and not self.owns_label(ix, r.get(ck)):
                continue
            rr.append(roots(" ".join(str(v) for k, v in r.items() if k != ck and isinstance(v, str))))
        best, bs = parts[0], -1
        for p in parts:
            hr = roots(p.split()[-1])
            s = sum(1 for x in rr if hr & x)
            if s > bs:
                best, bs = p, s
        return best

    def examples(self, ix, label):
        w = word_form(label)
        musts = " ".join(self.items[ix - 1]["musts"]).lower()
        for r in self.key_rows():
            if r["key"] == w:
                return [v for v in r["vals"] if v and len(v) <= 30 and v not in musts][:2]
        return []

    def send(self, qs, floor=6):
        sent = []
        tool = self.roles.get("ask")
        for ix, label, text in qs:
            if self.finished or self.left() <= floor or self.msgs_used >= self.msg_limit:
                break
            args = self.msg_args(text)
            if args is None or not tool:
                break
            r = self.env_call(tool, args)
            if r is None:
                break
            self.msgs_used += 1
            obs = r.get("observation")
            ans = self.spoken(obs)
            sent.append((ix, text, ans))
            self.done_log.append("step %d: shopper message sent %s -> reply %s (cart then: %s)"
                                 % (self.steps, repr(text), repr(ans[:300]), self.cart_mark()))
            self.done_steps.append(self.steps)
            self.trace.add({"kind": "sent", "item": ix, "q": text, "a": ans[:300]})
            if isinstance(obs, dict) and "error" in obs:
                break
        return sent

    def fold(self, reading, told):
        n = len(self.items)

        def item_no(e):
            k = count_int(e.get("item"))
            return min(max(k if k else 1, 1), n)
        for e in items_of(reading.get("needs")):
            if not isinstance(e, dict) or not str(e.get("need") or "").strip():
                continue
            k = item_no(e)
            need = str(e["need"]).strip()
            it = self.items[k - 1]
            if need not in it["told"]:
                it["told"].append(need)
                told.append(need if n == 1 else "item %d: %s" % (k, need))
            if e.get("points") is True and (k, need) not in self.refs:
                self.refs.append((k, need))
        for e in items_of(reading.get("drop")):
            if not isinstance(e, dict) or not str(e.get("property") or "").strip():
                continue
            k = item_no(e)
            it = self.items[k - 1]
            prop = word_form(str(e["property"]).split(":", 1)[0])
            for r in it["musts"] + it["told"]:
                if r not in it["dropped"] and word_form(r.split(":", 1)[0]) == prop:
                    it["dropped"].append(r)
                    told.append("dropped " + r)
        for e in items_of(reading.get("free")):
            if isinstance(e, dict) and e.get("property") is not None:
                self.items[item_no(e) - 1]["free"].append(str(e["property"]))
        b = reading.get("budget")
        if isinstance(b, (int, float)) and not isinstance(b, bool) and b > 0:
            self.budget = b

    def read_answers(self, sent, models=None):
        if self.quiet(sent):
            self.trace.add({"kind": "quiet"})
            return {}
        self.reply_reads += 1
        models = models or [HEAVY, LIGHT]
        il = "\n".join("item %d: %s" % (ix, it["name"]) for ix, it in enumerate(self.items, 1))
        qa = []
        for num, (ix, q, a) in enumerate(sent, 1):
            tag = "(about item %d)" % ix if ix else "(about several items)"
            qa.append("%d. %s Q: %s\nA: %s" % (num, tag, q, a))
        prompt = REPLY_PROMPT + "\n\nITEMS:\n" + il + "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(qa)
        return self.ask_json(prompt, models, 2500) or {}

    @soft(False)
    def quiet(self, sent):
        if self.reply_reads == 0:
            return False
        known = set()
        for it in self.items:
            known |= spelled(it["name"]) | spelled(it["shelf"])
            for r in it["musts"] + it["told"]:
                known |= spelled(" ".join(E6_RUNS.findall(r.split(":", 1)[0].lower())))
        for r in self.key_rows():
            known |= spelled(r["key"]) | spelled(r["label"])
        for rec in self.full.values():
            for v in (attr_block(rec) or {}).values():
                known |= spelled(" ".join(w for w in E6_RUNS.findall(str(v).lower()) if len(w) >= 4))
        for _, q, a in sent:
            qw = spelled(" ".join(E6_RUNS.findall(E6_FILLER.sub(" ", q).lower())))
            aw = spelled(" ".join(E6_RUNS.findall(a.lower())))
            if E6_DIGIT.search(a) or (aw & known) or (aw & qw) or V5_HOWTO.search(a):
                return False
        return True


E7_COUNT = re.compile(r"\b(?:qty|quantity|count|units?)\b")
E7_LIMIT = re.compile(r"\b(?:not|no|least|most|under|over|min|max|minimum|maximum|unlike|different)\b")
E7_REMOVE = re.compile(r"remov|delet|drop|clear|take", re.I)
E7_SPLIT = re.compile(r"[,/;|]")
GATE_LEAD = "Cairn carried out the request's own steps before continuing:\n"


class EpisodeChecks:
    def live_reqs(self, ix):
        it = self.items[ix - 1]
        return [r for r in it["musts"] + it["told"] if r not in it["dropped"]]

    def live_all(self):
        out = []
        for ix in range(1, len(self.items) + 1):
            out += [(ix, r) for r in self.live_reqs(ix)]
        return out

    def gate(self, stage):
        duties = [d for d in self.duties if d["stage"] == stage]
        if not duties or self.finished or not self.gate_due(stage):
            return False, ""
        excl = (self.roles.get("buy"), self.roles.get("refuse"))
        tl = []
        for t in self.tools:
            fn = t["function"]
            if fn["name"] in excl:
                continue
            tl.append("- " + fn["name"] + ": " + str(fn.get("description") or "") + " | parameters: "
                      + cjson(fn.get("parameters") if fn.get("parameters") is not None else {}))
        phrase = "place the order" if stage == "order" else "end the episode without buying anything"
        talk = stage == "order" and any(d["messages"] >= 1 for d in duties)
        said = []
        nudged = False
        ask = self.roles.get("ask")
        for _ in range(GATE_ROUNDS):
            rec = "\n".join("%d. %s" % (i, l) for i, l in enumerate(self.done_log, 1)) or "(nothing recorded yet)"
            parts = [GATE_PROMPT.replace("STAGE", phrase), "REQUEST:\n" + self.request,
                     "STEPS DUE NOW:\n" + "\n".join("- " + d["text"] for d in duties),
                     "LEDGER:\n" + self.ledger(), "CART: " + self.cart_text(),
                     "RECORD (shopper messages and store actions so far):\n" + rec]
            if self.changes:
                parts.append("STORE NOTES SEEN:\n" + "\n".join(self.changes[-4:]))
            parts.append("messages left: %d   steps left: %d (the %s itself uses one step)" % (
                max(0, self.msg_limit - self.msgs_used), self.left(), "order" if stage == "order" else "ending"))
            parts.append("TOOLS:\n" + "\n".join(tl))
            ans = self.ask_json("\n\n".join(parts), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if ans is None:
                break
            if said:
                told = []
                self.fold({"needs": ans.get("needs")}, told)
                hold = str(ans.get("hold") or "").strip()
                if told or hold:
                    self.version += 1
                    t = GATE_LEAD + "\n".join(said) + "\n"
                    t += ("The shopper's answer alters the task: " + "; ".join(told + ([hold] if hold else []))
                          + ". Repair the cart accordingly and check out once more.")
                    return True, t
            call = ans.get("call")
            has_call = isinstance(call, dict) and bool(call.get("tool"))
            if not has_call:
                last = self.cart_changed_at
                if talk:
                    proof = set(i for i, st in enumerate(self.done_steps, 1) if st > last)
                else:
                    proof = set(range(1, len(self.done_log) + 1))
                unproven = []
                for s in items_of(ans.get("steps")):
                    if not isinstance(s, dict):
                        continue
                    cited = [count_int(x) for x in items_of(s.get("lines"))]
                    if s.get("done") is not True or not any(c in proof for c in cited if c is not None):
                        unproven.append(str(s.get("step") or ""))
                if (unproven and not said and not nudged and ask and self.msgs_used < self.msg_limit
                        and self.left() >= 2):
                    nudged = True
                    self.done_log.append(
                        "(Cairn: none of the record lines written after the cart's latest change at step %d proves: %s."
                        " Messages left: %d, steps left: %d. Name the action that performs it.)"
                        % (last, "; ".join(unproven)[:200], self.msg_limit - self.msgs_used, self.left()))
                    self.done_steps.append(-1)
                    continue
                self.gate_passes.add((stage, self.cart_mark() + "#" + str(self.version)))
                break
            tool = call.get("tool")
            if tool not in self.schemas or tool in excl:
                break
            if self.left() < 2 or (tool == ask and self.msgs_used >= self.msg_limit):
                break
            args, f = shape_args(self.schemas.get(tool) or {}, call.get("args") if isinstance(call.get("args"), dict) else {})
            if f:
                break
            if tool == ask:
                ss = strings_of(args)
                text = max(ss, key=len) if ss else ""
                if not text:
                    break
                sent = self.send([(1, "", text)], 1)
                if not sent:
                    break
                self.gate_msgs += 1
                said.append("Q: " + text + "\nA: " + sent[0][2])
                continue
            r = self.env_call(tool, args)
            if r is None:
                break
            obs = r.get("observation")
            self.absorb(obs)
            shown = self.display(obs)
            self.done_log.append("step %d: %s %s -> %s (cart then: %s)" % (self.steps, tool, cjson(args), shown[:300],
                                                                           self.cart_mark()))
            self.done_steps.append(self.steps)
            said.append("%s %s -> %s" % (tool, cjson(args), shown[:400]))
        if said:
            return False, GATE_LEAD + "\n".join(said)
        return False, ""

    def spoken(self, obs):
        if isinstance(obs, str):
            return obs
        out = []

        def walk(v, d):
            if d > 4:
                return
            if isinstance(v, str):
                if len(v.split()) >= 2:
                    out.append(v)
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x, d + 1)
            elif isinstance(v, list):
                for x in v:
                    walk(x, d + 1)
        walk(obs, 0)
        if out:
            return " ".join(out)[:800]
        return cjson(obs)[:400]

    def loose_lines(self, obs):
        if not isinstance(obs, dict) or self.listing_id(obs, False, 0) is not None:
            return []
        out = []
        for v in obs.values():
            if isinstance(v, str):
                if len(v.split()) >= 5:
                    out.append(v)
            elif isinstance(v, list):
                if v and all(isinstance(x, str) for x in v):
                    out += [x for x in v if len(x.split()) >= 5]
            elif isinstance(v, dict) and self.listing_id(v, False, 1) is None:
                out += [x for x in v.values() if isinstance(x, str) and len(x.split()) >= 6]
        res = []
        for l in out:
            if l not in self.heard and l not in res:
                res.append(l)
        return res

    def read_note(self, lines):
        self.heard.update(lines)
        prompt = (NOTE_PROMPT + "\n\nCURRENT ITEMS AND REQUIREMENTS:\n" + self.ledger() + "\n\nNOTICE:\n"
                  + "\n".join(lines))
        ans = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(x) for x in items_of(ans.get("gone")) if isinstance(x, str) or is_whole(x)]
        for g in gone:
            if g in self.rows:
                self.unavailable.add(g)
        self.note_sold = bool(gone) or ans.get("sold_out") is True
        self.note_named = any(g in self.rows for g in gone)
        if self.note_sold and len(lines) == 1:
            self.sold_lines.add(lines[0])
        kind = ans.get("kind")
        if kind is not None and kind != "change":
            return ""
        n = len(self.items)
        changes = []
        bchanged = dropped = False
        b = ans.get("budget")
        if isinstance(b, (int, float)) and not isinstance(b, bool) and b > 0 and b != self.budget:
            self.budget = b
            bchanged = True
            changes.append("budget now %s" % gnum(b))
        for e in items_of(ans.get("drop")):
            if not isinstance(e, dict) or not str(e.get("need") or "").strip():
                continue
            k = min(max(count_int(e.get("item")) or 1, 1), n)
            need = str(e["need"]).strip()
            live = self.live_reqs(k)
            m = need if need in live else next(
                (r for r in live if word_form(r.split(":", 1)[0]) == word_form(need.split(":", 1)[0])), None)
            self.items[k - 1]["dropped"].append(m or need)
            dropped = True
            changes.append("dropped " + need)
        for e in items_of(ans.get("add")):
            if not isinstance(e, dict) or not str(e.get("need") or "").strip():
                continue
            k = min(max(count_int(e.get("item")) or 1, 1), n)
            need = str(e["need"]).strip()
            it = self.items[k - 1]
            if need in it["dropped"]:
                it["dropped"].remove(need)
            if need not in it["musts"] and need not in it["told"]:
                it["told"].append(need)
            changes.append("added " + need)
        if gone:
            changes.append("unavailable: " + ", ".join(gone))
        if ans.get("price_moved") is True:
            changes.append("a price moved")
        if self.brief.get("pick") == "cheapest" and (dropped or gone or self.note_sold or bchanged):
            self.wider_held = False
            self.cheaper_marks = set()
        summary = str(ans.get("summary") or "").strip()
        if changes or summary:
            self.version += 1
            self.recent = []
            self.changes.append(summary or "; ".join(changes))
        return (summary + ((" (" + "; ".join(changes) + ")") if changes else "")).strip()

    def evidence(self, obs=None, ids=None):
        if ids is None:
            ids = []

            def walk(v, d, inrows):
                if d > 6:
                    return
                if isinstance(v, dict):
                    lid = self.listing_id(v, inrows, d)
                    if lid is not None and attr_block(v):
                        if lid not in ids:
                            ids.append(lid)
                        return
                    for x in v.values():
                        walk(x, d + 1, False)
                elif isinstance(v, list):
                    for x in v:
                        walk(x, d + 1, True)
            walk(obs, 0, False)
        live = self.live_all()
        if not ids or not live:
            return ""
        reqs = live[:8]
        lines = []
        for lid in ids[:12]:
            ab = attr_block(self.full.get(lid)) or {}
            parts = []
            for ix, r in reqs:
                prop = r.split(":", 1)[0].strip()
                ps = stems(prop)
                hits = ["%s=%s" % (k, str(v)[:80]) for k, v in ab.items() if names_test(ps, stems(k))][:3]
                if hits:
                    tag = ("item %d " % ix) if len(self.items) >= 2 else ""
                    parts.append(tag + prop + " -> " + ", ".join(hits))
            if parts:
                lines.append(lid + ": " + "; ".join(parts))
        if not lines:
            return ""
        return ("[evidence: the keys listed carry a requirement's property inside their name; a key about another"
                " property (another component or aspect) settles nothing either way; a differently named key might"
                " still hold that same property] " + " | ".join(lines))

    def fresh_view(self):
        return self.cart_view is not None and self.cart_seen_at >= self.cart_changed_at

    def cart_text(self):
        if self.fresh_view():
            return cjson(clip_deep(self.cart_view, 300))[:3000]
        return "(from Cairn's cart record, listing: units) " + cjson(self.cart)

    def units_held(self, lid):
        if self.fresh_view():
            stack = [self.cart_view]
            while stack:
                o = stack.pop()
                if isinstance(o, dict):
                    if lid in [v.strip() for v in o.values() if isinstance(v, str)]:
                        vals = []
                        for k, v in o.items():
                            if isinstance(v, str) or not E7_COUNT.search(word_form(k)):
                                continue
                            c = count_int(v)
                            if c is not None:
                                vals.append(c)
                        if len(vals) == 1:
                            return vals[0]
                    for v in o.values():
                        if isinstance(v, (dict, list)):
                            stack.append(v)
                elif isinstance(o, list):
                    stack.extend(o)
        return self.cart.get(lid, 1)

    def cart_ids(self):
        if self.fresh_view():
            vt = cjson(self.cart_view)
            return [i for i in self.seen if i in vt]
        return list(self.cart)

    def held_units(self):
        return dict((i, self.units_held(i)) for i in self.cart_ids())

    def record_text(self, lid):
        rec = self.full.get(lid) or self.rows.get(lid)
        if rec is None:
            return "(no record seen)"
        return cjson(clip_deep(rec, 400))[:3000]

    def other_listings(self, skip, limit=30):
        ids = list(reversed(self.seen))
        if self.brief.get("pick") == "cheapest":
            ids.sort(key=lambda i: (i not in self.full, self.price(i) is None,
                                    self.price(i) if self.price(i) is not None else 0))
        out = []
        for i in ids:
            if i in skip or self.sold_out(i):
                continue
            out.append(cjson(clip_deep(self.full.get(i) or self.rows.get(i), 160))[:900])
            if len(out) >= limit:
                break
        return out

    def fit_index(self):
        key = (len(self.seen), len(self.full), sum(len(v) for v in self.views.values()))
        if self.fit_cache and self.fit_cache[0] == key:
            return self.fit_cache[1]
        idx, notes = {}, {}
        for lid in self.seen:
            rec = self.full.get(lid) or self.rows.get(lid) or {}
            plain_t = " ".join(v for v in rec.values() if isinstance(v, str))
            nested = []
            for v in rec.values():
                if isinstance(v, dict):
                    nested += ["%s %s" % (k, x) for k, x in v.items() if isinstance(x, str)]
            marks = fit_marks(plain_t + " " + ". ".join(nested))
            idx[lid] = marks
            for part, codes in marks["notes"]:
                notes.setdefault(part, []).extend(codes)
        self.fit_cache = (key, idx, notes)
        return idx

    def fit_set(self, lid):
        idx = self.fit_index()
        m = idx.get(lid)
        if not m:
            return set()
        s = set(m["works"])
        if m["own"]:
            s |= set(self.fit_cache[2].get(m["own"], []))
        return s

    def item_listings(self, ix):
        ck = self.cat_key()
        if ck is None:
            return []
        return [i for i in self.seen if self.owns_label(ix, (self.rows.get(i) or {}).get(ck))]

    def find_system(self, part, main):
        idx = self.fit_index()
        for carrier, namer in ((part, main), (main, part)):
            ls = self.item_listings(carrier)
            tagged = [i for i in ls if idx.get(i, {}).get("own_series")]
            fams = set()
            for i in tagged:
                fams |= set(idx[i]["own_series"])
            if len(tagged) < SERIES_ROWS or len(tagged) * 2 < len(ls) or len(fams) > SERIES_BREADTH:
                continue
            if any(set(idx.get(i, {}).get("named_series") or []) & fams for i in self.item_listings(namer)):
                return carrier, namer, fams
        return None

    def link_holds(self, carried, naming, fams):
        idx = self.fit_index()
        a = set(idx.get(carried, {}).get("own_series") or []) & fams
        b = set(idx.get(naming, {}).get("named_series") or []) & fams
        if a and b:
            return bool(a & b)
        return None

    def link_faults(self, part, main, device, plines):
        sysm = self.find_system(part, main)
        if not sysm:
            return []
        carrier, namer, fams = sysm
        idx = self.fit_index()
        out = []
        for pl in plines:
            carried, naming = (pl, device) if carrier == part else (device, pl)
            if self.link_holds(carried, naming, fams) is not False:
                continue
            wanted = set(idx[naming]["named_series"]) & fams
            alts = [i for i in self.item_listings(carrier) if set(idx[i]["own_series"]) & wanted
                    and not self.sold_out(i) and i != carried][:4]
            t = "%s (series %s) against %s, which takes %s" % (
                carried, ", ".join(sorted(set(idx[carried]["own_series"]) & fams)), naming, ", ".join(sorted(wanted)))
            if alts:
                t += "; seen listings carrying that series: " + ", ".join(alts)
            else:
                t += ("; search for that series together with the item's own words, or take a different %s, one with a"
                      " series carried by the wanted lines" % self.items[main - 1]["name"])
            out.append(t)
        return out

    def plan_series(self, ix):
        idx = self.fit_index()
        counts, held, order = {}, [], []
        cart = set(self.cart_ids())
        for p, m in self.fit_pairs:
            if ix not in (p, m):
                continue
            sysm = self.find_system(p, m)
            if not sysm:
                continue
            fams = sysm[2]
            for i in self.item_listings(p) + self.item_listings(m):
                for f in sorted((set(idx[i]["own_series"]) | set(idx[i]["named_series"])) & fams):
                    if f not in counts:
                        order.append(f)
                    counts[f] = counts.get(f, 0) + 1
                    if i in cart and f not in held:
                        held.append(f)
        rest = sorted([f for f in order if f not in held], key=lambda f: (-counts[f], f))
        return (held + rest)[:SERIES_HUNTED]

    def covers_dev(self, part, dev):
        idx = self.fit_index()
        dcodes = idx.get(dev, {}).get("models") or []
        fs = self.fit_set(part)
        barred = idx.get(part, {}).get("barred") or []
        return (any(code_covers(c, d) for c in fs for d in dcodes)
                and not any(code_covers(b, d) for b in barred for d in dcodes))

    def fit_faults(self):
        if not self.fit_pairs:
            return []
        ck = self.cat_key()
        cart = self.cart_ids()
        idx = self.fit_index()
        out = []
        for p, m in self.fit_pairs:
            if ck is None:
                continue
            dev = [i for i in cart if self.owns_label(m, (self.rows.get(i) or {}).get(ck))]
            pls = [i for i in cart if self.owns_label(p, (self.rows.get(i) or {}).get(ck))]
            if len(dev) != 1 or not pls:
                continue
            lf = self.link_faults(p, m, dev[0], pls)
            if lf:
                out += lf
                continue
            dcodes = idx.get(dev[0], {}).get("models") or []
            if not dcodes:
                continue
            for pl in pls:
                fs = self.fit_set(pl)
                if not fs or self.covers_dev(pl, dev[0]):
                    continue
                alts = [i for i in self.item_listings(p) if self.covers_dev(i, dev[0]) and not self.sold_out(i)
                        and i not in cart][:4]
                t = "%s (%s) is meant for %s, against %s (%s)" % (pl, self.items[p - 1]["name"],
                                                                   ", ".join(sorted(fs)[:4]), dev[0],
                                                                   ", ".join(sorted(dcodes)))
                if alts:
                    t += "; compatible parts of the same sort: " + ", ".join(alts)
                else:
                    t += ("; none of the seen parts of this sort suits the device. Hunt for a part that suits it or"
                          " change the device: " + self.fit_plan(True))
                out.append(t)
        return out

    def fit_plan(self, short=False):
        if not self.fit_pairs:
            return ""
        idx = self.fit_index()
        lines = []
        for m in sorted(set(mm for _, mm in self.fit_pairs)):
            parts = [p for p, mm in self.fit_pairs if mm == m]
            tally, order = {}, []
            for p in parts:
                for i in self.item_listings(p):
                    if self.sold_out(i):
                        continue
                    for c in self.fit_set(i):
                        if c not in tally:
                            tally[c] = {}
                            order.append(c)
                        tally[c][p] = tally[c].get(p, 0) + 1
                sysm = self.find_system(p, m)
                if sysm and sysm[0] == p:
                    for i in self.item_listings(p):
                        if self.sold_out(i):
                            continue
                        for f in set(idx[i]["own_series"]) & sysm[2]:
                            if f not in tally:
                                tally[f] = {}
                                order.append(f)
                            tally[f][p] = tally[f].get(p, 0) + 1
            ranked = sorted(order, key=lambda c: (-len(tally[c]), -sum(tally[c].values())))[:6]
            segs = []
            for c in ranked:
                devs = [i for i in self.item_listings(m)
                        if c.upper() in [s.upper() for s in idx[i]["named_series"]]
                        or any(code_covers(c, d) for d in idx[i]["models"])][:2]
                seg = c + ": " + ", ".join("%d %s" % (tally[c][p], self.items[p - 1]["name"]) for p in parts if p in tally[c])
                seg += (" (devices seen: %s)" % ", ".join(devs)) if devs else " (no device has turned up yet)"
                segs.append(seg)
            if segs:
                lines.append("codes the parts name for %s candidates -> " % self.items[m - 1]["name"] + "; ".join(segs))
            if self.brief.get("pick") == "cheapest":
                cs = self.cheapest_sets(m, parts)
                if cs:
                    lines.append(cs)
        if not lines:
            return ""
        if short:
            return " | ".join(lines)
        return ("[fit plan: a part suits a device if the part's wording mentions the device's model number or an"
                " alternative trade name of it, or if the device's wording mentions the series that the part's code"
                " belongs to. Try a single query combining one code (or one series) plus the item's own terms; should"
                " that return nothing, browse the category the device belongs to and read the codes off its rows] "
                + " | ".join(lines))

    @soft("")
    def cheapest_sets(self, m, parts):
        idx = self.fit_index()
        shelves = {}
        for p in parts:
            lst = [i for i in self.item_listings(p) if self.price(i) is not None and self.buyable(i)]
            lst.sort(key=lambda i: (self.price(i), i))
            shelves[p] = (self.find_system(p, m), lst)
        sets = []
        for d in self.item_listings(m):
            dm = idx.get(d) or {}
            if not (dm.get("models") or dm.get("named_series")) or self.price(d) is None or not self.buyable(d):
                continue
            total = self.price(d) * self.items[m - 1]["qty"]
            ids = [d]
            ok = True
            for p in parts:
                sysm, lst = shelves[p]
                fit = None
                for i in lst:
                    if self.covers_dev(i, d) or (sysm and sysm[0] == p and self.link_holds(i, d, sysm[2]) is True):
                        fit = i
                        break
                if fit is None:
                    ok = False
                    break
                total += self.price(fit) * self.items[p - 1]["qty"]
                ids.append(fit)
            if ok:
                sets.append((total, ids))
        sets.sort(key=lambda s: s[0])
        if not sets:
            return ""
        return ("cheapest complete sets seen (requirements still unverified): "
                + "; ".join("+".join(ids) + "=" + gnum(t) for t, ids in sets[:CHEAP_SETS]))

    def missing_items(self):
        ck = self.cat_key()
        if ck is None:
            return []
        lines = [i for i in self.cart_ids() if self.units_held(i) > 0]
        if len(lines) >= len(self.items):
            return []
        out = []
        for ix in range(1, len(self.items) + 1):
            if not any(self.owns_label(ix, r.get(ck)) for r in self.rows.values()):
                continue
            if any(self.owns_label(ix, (self.rows.get(l) or {}).get(ck)) for l in lines):
                continue
            out.append(ix)
        return out

    def states_all(self, ix, lid):
        ab = attr_block(self.full.get(lid)) or {}
        for r in self.live_reqs(ix):
            prop, _, val = r.partition(":")
            vw = set(word_form(E7_SPLIT.sub(" ", val)).split())
            if not vw or E7_LIMIT.search(" ".join(vw)):
                return False
            ok = False
            for k, v in ab.items():
                if names_test(spelled(prop), spelled(k)) and vw <= set(word_form(E7_SPLIT.sub(" ", str(v))).split()):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def fill_short_cart(self):
        bound = set()
        for p, m in self.fit_pairs:
            bound.update((p, m))
        picks = []
        each = self.brief.get("budget_scope") == "each"
        held = sum((self.price(i) or 0) * self.units_held(i) for i in self.cart_ids())
        for ix in self.missing_items():
            cands = [i for i in self.item_listings(ix) if i in self.full and i not in self.cart and self.buyable(i)
                     and self.price(i) is not None and self.states_all(ix, i)]
            cands.sort(key=lambda i: self.price(i))
            pick = cands[0] if cands and ix not in bound else None
            qty = self.items[ix - 1]["qty"]
            if pick and isinstance(self.budget, (int, float)):
                cost = self.price(pick) * (1 if each else qty)
                if cost + (0 if each else held) > self.budget + 1e-9:
                    pick = None
            if not pick:
                self.trace.add({"kind": "short_fill_fail", "item": ix})
                return False
            picks.append((pick, qty))
        if not picks:
            return True
        target = dict((i, u) for i, u in self.cart.items())
        for p, q in picks:
            target[p] = q
        calls = self.cart_calls(target)
        if not calls or len(calls) > self.left() - 1:
            return False
        for tool, a in calls:
            a2, f = shape_args(self.schemas.get(tool) or {}, a)
            if f:
                return False
            r = self.env_call(tool, a2)
            if r is None:
                return False
            self.note_cart(tool, a2, r.get("observation"))
        self.trace.add({"kind": "short_fill", "picks": picks})
        return True

    def part_shelves(self):
        ck = self.cat_key()
        if ck is None:
            return
        labels = sorted(set(str(r.get(ck)).strip() for r in self.rows.values()
                            if r.get(ck) is not None and str(r.get(ck)).strip()))
        moved = False
        for ix, it in enumerate(self.items, 1):
            nr = roots(it["name"])
            if not nr or not it["shelf"]:
                continue
            if not any(j != ix and word_form(o["shelf"]) == word_form(it["shelf"]) for j, o in enumerate(self.items, 1)):
                continue
            if any(j != ix and roots(o["name"]) == nr for j, o in enumerate(self.items, 1)):
                continue
            own = [l for l in labels if roots(l) == nr and word_form(l) != word_form(it["shelf"])]
            if len(own) == 1 and not any(word_form(o["shelf"]) == word_form(own[0]) for o in self.items):
                it["shelf"] = own[0]
                moved = True
                self.trace.add({"kind": "shelf_parted", "item": ix, "shelf": own[0]})
        if moved:
            self.owners_seen = {}
            self.fit_cache = None

    def fuller_label(self, ix, lab):
        it = self.items[ix - 1]
        nr = roots(it["name"])
        ck = self.cat_key()
        lr = roots(str(lab))
        if len(nr) < 2 or nr <= lr or ck is None or (it["shelf"] and lr == roots(it["shelf"])):
            return None
        labels = sorted(set(str(r.get(ck)) for r in self.rows.values() if r.get(ck) is not None and str(r.get(ck))))
        return next((l for l in labels if nr <= roots(l)), None)

    def named_label(self, ix, lab):
        it = self.items[ix - 1]
        ck = self.cat_key()
        if not it["shelf"] or ck is None or word_form(str(lab)) == word_form(it["shelf"]):
            return None
        for i in self.seen:
            v = (self.rows.get(i) or {}).get(ck)
            if v is not None and word_form(str(v)) == word_form(it["shelf"]):
                return str(v)
        return None

    def line_faults(self):
        self.adopt_strays()
        ck = self.cat_key()
        lines = [i for i in self.cart_ids() if self.units_held(i) > 0]
        if not lines or ck is None:
            return []
        n = len(self.items)
        owners = dict((l, [ix for ix in range(1, n + 1) if self.owns_label(ix, (self.rows.get(l) or {}).get(ck))])
                      for l in lines)
        known = [ix for ix in range(1, n + 1) if any(self.owns_label(ix, r.get(ck)) for r in self.rows.values())]
        out = []
        for ix in known:
            it = self.items[ix - 1]
            mine = [l for l in lines if ix in owners[l]]
            if not mine:
                out.append(("item", "item %d (%s) lacks a cart line" % (ix, it["name"])))
                continue
            alone = [l for l in mine if owners[l] == [ix]]
            if len(alone) > 1:
                out.append(("several", "item %d (%s) has %d lines (%s); keep one" % (ix, it["name"], len(mine),
                                                                                    ", ".join(mine))))
            for l in mine:
                if owners[l] == [ix]:
                    u = self.units_held(l)
                    if u != it["qty"]:
                        out.append(("count", "%s is held at %d %s; item %d (%s) wants %d" % (
                            l, u, "unit" if u == 1 else "units", ix, it["name"], it["qty"])))
                lab = (self.rows.get(l) or {}).get(ck)
                alt = self.fuller_label(ix, lab) or self.named_label(ix, lab)
                if alt:
                    out.append(("label", "%s comes from the label %s while the store also has the label %s, which names"
                                         " item %d (%s) completely; source this line from %s instead"
                                % (l, lab, alt, ix, it["name"], alt)))
        if len(known) == n:
            for l in lines:
                if not owners[l]:
                    lab = (self.rows.get(l) or {}).get(ck)
                    out.append(("kind", "%s (%s) is a kind nobody requested; remove it"
                                % (l, lab if lab is not None else "unknown kind")))
        for l in lines:
            if self.sold_out(l):
                out.append(("sold", "%s is sold out" % l))
        b = self.budget
        if isinstance(b, (int, float)) and all(self.price(l) is not None for l in lines):
            if self.brief.get("budget_scope") == "each":
                for l in lines:
                    p = self.price(l)
                    if b < p <= 10 * b:
                        out.append(("budget", "%s costs %s, above the per-item limit %s" % (l, gnum(p), gnum(b))))
            else:
                tot = sum(self.price(l) * self.units_held(l) for l in lines)
                if tot > b + 1e-9 and tot <= 10 * b:
                    out.append(("budget", "the cart totals %s, above the budget %s" % (gnum(tot), gnum(b))))
        return out

    def count_fixes(self, faults=None):
        ck = self.cat_key()
        if ck is None:
            return []
        n = len(self.items)
        fixes = []
        for i in self.cart_ids():
            u = self.units_held(i) or 0
            own = [ix for ix in range(1, n + 1) if self.owns_label(ix, (self.rows.get(i) or {}).get(ck))]
            if u <= 0 or len(own) != 1:
                continue
            o = own[0]
            nl = sum(1 for j in self.cart_ids() if self.cart.get(j, 1) > 0
                     and self.owns_label(o, (self.rows.get(j) or {}).get(ck)))
            if nl == 1 and u != self.items[o - 1]["qty"] and not self.sold_out(i):
                fixes.append((i, u, self.items[o - 1]["qty"]))
        ct = self.roles.get("cart")
        if not ct or not fixes:
            return []
        sc = self.schemas.get(ct) or {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")]
        idp = next((a for (t, a) in self.id_args if t == ct and a in texts), None)
        if idp is None:
            idp = next((k for k in req if k in texts), None)
        if idp is None and len(texts) == 1:
            idp = texts[0]
        cnt = next((k for k, p in props.items() if isinstance(p, dict) and "integer" in kinds(p)), None)
        mode = next((k for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list)
                     and len(p["enum"]) >= 2), None)
        if not idp or not cnt:
            return []
        calls = []
        for i, u, w in fixes:
            if self.sets_counts():
                calls.append({idp: i, cnt: w})
            elif mode:
                opts = props[mode]["enum"]
                rem = next((o for o in opts if E7_REMOVE.search(str(o))), opts[1])
                add = next((o for o in opts if o != rem), opts[0])
                calls += [{idp: i, mode: rem, cnt: u}, {idp: i, mode: add, cnt: w}]
            else:
                return []
        if len(calls) + 3 > self.left():
            return []
        done = []
        for a in calls:
            a2, f = shape_args(sc, a)
            if f:
                break
            r = self.env_call(ct, a2)
            if r is None:
                break
            obs = r.get("observation")
            self.note_cart(ct, a2, obs)
            self.cart_said = cjson(obs)
            self.observe(ct, a2, obs)
            done.append("%s -> %s" % (cjson(a2), self.display(obs)[:300]))
            if self.finished:
                break
        self.trace.add({"kind": "count_fixes", "done": len(done)})
        return done

    def check_key(self, ids):
        return "|".join(sorted(ids)) + "#" + str(self.version)

    def cart_check(self, planned=None, limit=SOFT_SPEND):
        ids = list(planned) if planned else self.cart_ids()
        key = self.check_key(ids)
        judges = self.judges(limit)
        if key in self.checked or not judges or not ids:
            return ""
        self.checked.add(key)
        pick = self.brief.get("pick")
        comp = pick in ("cheapest", "most", "least")
        parts = [CHECK_PROMPT, "REQUEST:\n" + self.request, "BINDING REQUIREMENTS:\n" + self.ledger()]
        if planned:
            parts.append("CART (a plan only: judge it as though these listings sat in the cart, each at its asked"
                         " quantity): " + "; ".join("item %d: quantity %d" % (ix, it["qty"])
                                                    for ix, it in enumerate(self.items, 1))
                         + "\n" + ", ".join(ids))
        else:
            parts.append("CART: " + self.cart_text())
        recs = []
        for i in ids:
            so = "SOLD OUT (the store reports it cannot be purchased any more) " if self.sold_out(i) else ""
            recs.append("%s: %s%s" % (i, so, self.record_text(i)))
        parts.append("RECORDS OF THE CARTED LISTINGS:\n" + "\n".join(recs))
        if comp:
            parts.append("OTHER LISTINGS SEEN (newest first):\n" + "\n".join(self.other_listings(set(ids), 15)))
        prompt = "\n\n".join(parts)
        ans = None
        order = list(judges)
        if len(order) == 2 and not planned and not self.audited and not comp:
            ans = self.ask_json(prompt, [LIGHT], 3000)
            if ans is not None:
                order = [HEAVY] if (ans.get("ok") is False and HEAVY in self.judges(limit)) else []
        if order and not self.no_credit:
            a2 = self.ask_json(prompt, order, 3000)
            if a2 is not None:
                ans = a2
        self.trace.add({"kind": "cart_check", "ids": ids, "ok": None if ans is None else ans.get("ok")})
        self.verdicts[key] = ans is not None and ans.get("ok") is not False
        if ans is None or ans.get("ok") is not False:
            return ""
        self.leads = [l for l in self.leads if l not in ids]
        probs = [str(p) for p in items_of(ans.get("problems"))][:5]
        better = [str(b) for b in items_of(ans.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        t = "A careful review finds the cart still wrong: " + ("; ".join(probs) if probs else
                                                               "check it against the requirements in the ledger") + "."
        if better:
            t += " Candidates that seem preferable: " + ", ".join(better) + "."
        t += (" Repair the cart, or else demonstrate that the reviewer erred by reading the records afresh, and then"
              " check out.")
        return t


E8_STOP1 = re.compile(r"^(?:the|and|with|from|that|comes|listing|page|its|for|of)$")
E8_STOP2 = re.compile(r"^(?:the|and|with|from|that|comes|listing|page|its|for)$")
E8_RUN3 = re.compile(r"[a-z0-9]{3,}")
E8_RUN = re.compile(r"[a-z0-9]+")
E8_DIGITS = re.compile(r"\d+")
E8_SPACES = re.compile(r"\s+")
DECLINE_TEXT = ("I am sorry: no product in this shop satisfies all of your requirements, so I am not making a"
                " purchase.")


def tok_in(name, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])", text) is not None


def leaves(v):
    if isinstance(v, dict):
        out = []
        for x in v.values():
            out += leaves(x)
        return out
    if isinstance(v, list):
        out = []
        for x in v:
            out += leaves(x)
        return out
    return [v]


class EpisodeTrails:
    def page_read(self, lid):
        if lid not in self.opened:
            return False
        off = self.roles.get("offer")
        if off and self.single_target(off):
            return off in self.views.get(lid, set())
        return True

    def pending_refs(self):
        live = set(r for _, r in self.live_all())
        out = []
        for _, t in self.refs:
            if t in live and t not in self.ref_values and t not in out:
                out.append(t)
        return out

    def trail(self, limit=None):
        pending = self.pending_refs()
        op = self.roles.get("offer") or self.roles.get("open")
        tgt = self.single_target(op) if op else None
        seen_lines, lost = [], []
        while pending:
            cap = 6 if self.left() > 15 else 4
            room = self.chain_room if self.chain_room is not None else 6
            if self.chain_rounds >= min(cap, room) or self.spend >= HARD_SPEND:
                break
            if limit is not None and self.clock() > limit:
                break
            self.chain_rounds += 1
            lines, n_open = [], 0
            for lid in reversed(self.seen):
                rec = self.full.get(lid) or self.rows.get(lid)
                if self.page_read(lid) and rec is not None and n_open < 45:
                    lines.append("OPENED " + cjson(clip_deep(rec, 300))[:1400])
                    n_open += 1
            for lid in reversed(self.seen):
                if len(lines) >= 60:
                    break
                rec = self.full.get(lid) or self.rows.get(lid)
                if not self.page_read(lid) or rec is None:
                    if lid in self.full:
                        lines.append("ROW " + cjson(clip_deep(self.full[lid], 120))[:700])
                    else:
                        lines.append("ROW " + cjson(clip_deep(self.rows.get(lid), 80))[:300])
            prompt = (CHAIN_PROMPT + "\n\nREQUIREMENTS WHOSE VALUE SITS ON ANOTHER LISTING:\n" + "\n".join(pending)
                      + "\n\nRECORDS (newest first):\n" + "\n".join(lines))
            models = [LIGHT] + ([HEAVY] if self.spend + HEAVY_ROOM <= HARD_SPEND else [])
            ans = self.ask_json(prompt, models, 4500) or {}
            reach, lost, gaps = [], [], []
            for ch in items_of(ans.get("chains")):
                if not isinstance(ch, dict):
                    continue
                nt = str(ch.get("need") or "").strip()
                req = nt if nt in pending else (pending[0] if len(pending) == 1 else None)
                if not req:
                    continue
                links = [l for l in items_of(ch.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                val = str(ch.get("value") or "").strip()
                allread = bool(links) and all(self.page_read(i) for i in ids)
                if ch.get("done") is True and allread and val:
                    self.ref_values[req] = val
                    self.version += 1
                    continue
                if allread and not str(links[-1].get("find") or "").strip():
                    self.ref_values[req] = val or ("whatever listing %s shows" % ids[-1])
                    self.version += 1
                    continue
                unread = [i for i in ids if i in self.rows and not self.page_read(i)]
                if unread:
                    for i in unread:
                        if i not in reach:
                            reach.append(i)
                    continue
                ll = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                desc = str(ll.get("about") or "").strip() or "the listing this requirement leads to"
                words = str(ll.get("find") or "").strip()
                lost.append(req + " -> find " + desc + ((" (search: %s)" % words) if words else ""))
                gaps.append((desc, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and op and tgt and self.left() > 12 and self.link_searches < 3:
                reach, known = self.link_finder(gaps[0][0], gaps[0][1])
                if known:
                    seen_lines.append("%s: a page, already open, of the shop the chain names; it shows the values"
                                      " listed in the description; take the value off that page" % known)
                    if lost:
                        lost.pop(0)
            if not reach or not op or not tgt or self.left() < len(reach) + 4 or self.finished:
                break
            for lid in reach[:max(0, min(3, 6 - len(seen_lines)))]:
                a, f = shape_args(self.schemas.get(op) or {}, {tgt: lid})
                if f or self.rushed():
                    break
                r = self.env_call(op, a)
                if r is None or self.finished:
                    break
                obs = r.get("observation")
                self.observe(op, a, obs)
                seen_lines.append(lid + ": " + self.display(obs)[:1500])
        text = ""
        if seen_lines:
            text = "[Cairn opened these pages, which the requirements lead to]\n" + "\n".join(seen_lines)
        if self.ref_values:
            text += ("\n" if text else "") + "[values taken from those pages] " + "; ".join(
                "%s -> %s" % (k, v) for k, v in self.ref_values.items())
        hold = False
        cap_h = 3 if self.left() > 15 else 2
        if lost and self.chain_holds < cap_h and self.left() > 5:
            hold = True
            self.chain_holds += 1
            text += (("\n" if text else "") + "The purchase is only valid once you have read completely every page a"
                     " requirement leads to (a result row will not do), including each page that leads on to yet"
                     " another listing. Not yet located or read: " + " | ".join(lost[:3]) + ". Search using the"
                     " shop's name plus the values in the description (another product of that kind sold by the same"
                     " shop is the wrong one), open whichever row from that shop has a page showing those values, use"
                     " the value printed on that page, and then order.")
        return text, hold

    def early_trail(self):
        if self.left() < TRAIL_ROOM or self.clock() > HUNT_DEADLINE or not self.pending_refs():
            return ""
        saved = (self.chain_rounds, self.chain_holds)
        self.chain_rounds = 0
        self.chain_room = EARLY_TRAIL_ROUNDS
        try:
            text, _ = self.trail(HUNT_DEADLINE)
        finally:
            self.chain_rounds, self.chain_holds = saved
            self.chain_room = None
        return text

    def decline_trail(self):
        if self.decline_chain_done:
            return ""
        if self.left() < TRAIL_ROOM or self.clock() > HUNT_DEADLINE or not self.pending_refs():
            return ""
        self.decline_chain_done = True
        text, _ = self.trail(HUNT_DEADLINE)
        if text:
            return text + ("\nRun searches for those values at once and purchase whatever matches; decline only if"
                           " nothing matches.")
        return ""

    def name_keys(self):
        return set(x for x in (self.id_key, self.cat_key(), self.title_key()) if x is not None)

    def name_cands(self, row, excl):
        return [v.strip() for k, v in (row or {}).items() if k not in excl and isinstance(v, str)
                and 2 < len(v.strip()) <= 40 and not V24_PLACEHOLDER.match(v)]

    def shop_rows(self, desc, ids, excl):
        d = desc.lower()
        return [i for i in ids if i not in self.opened and any(tok_in(n, d) for n in self.name_cands(self.rows.get(i), excl))]

    def opened_shops(self, desc, excl):
        d = desc.lower()
        return [i for i in self.opened if i in self.rows and any(tok_in(n, d) for n in self.name_cands(self.rows[i], excl))]

    def link_finder(self, desc, words):
        excl = self.name_keys()
        shopl = self.shop_rows(desc, list(self.rows), excl) + self.opened_shops(desc, excl)
        namew = set()
        for i in shopl:
            for k, v in (self.rows.get(i) or {}).items():
                if k not in excl and isinstance(v, str) and len(v.strip()) <= 40 and not V24_PLACEHOLDER.match(v):
                    namew |= set(E8_RUN3.findall(v.lower()))
        src = words or desc
        valw = [w for w in E8_RUN.findall(src.lower()) if (len(w) >= 3 or w.isdigit()) and w not in namew
                and not E8_STOP1.match(w)]
        for i in self.opened_shops(desc, excl):
            lv = " ".join(str(x) for x in leaves(self.full.get(i) or self.rows.get(i))).lower()
            if valw and all(tok_in(w, lv) for w in valw):
                return [], i
        tpl = self.value_template()
        key = E8_SPACES.sub(" ", words.lower()).strip()
        cands = []
        if key in self.link_seen:
            cands = self.shop_rows(desc, self.link_seen[key], excl)
        elif tpl and words:
            tool, fixed, field = tpl
            qs = [words]
            w2 = " ".join(words.split()[:2])
            if w2 != words:
                qs.append(w2)
            for q in qs:
                a = dict(fixed)
                a[field] = q
                a, f = shape_args(self.schemas.get(tool) or {}, a)
                if f:
                    break
                self.link_searches += 1
                r = self.env_call(tool, a)
                if r is None:
                    break
                obs = r.get("observation")
                self.observe(tool, a, obs)
                ids = self.ids_in(obs)
                self.link_seen[key] = ids
                cands = self.shop_rows(desc, ids, excl)
                if ids:
                    break
        kindw = set(w for w in E8_RUN3.findall(src.lower()) if w not in namew and not E8_STOP2.match(w))

        def of_kind(i):
            return not kindw or bool(kindw & set(E8_RUN3.findall(cjson(self.rows.get(i)).lower())))
        res = [i for i in cands if of_kind(i)][:2]
        if not res:
            res = [i for i in self.shop_rows(desc, list(reversed(self.seen)), excl) if of_kind(i)][:2]
        return res, None

    def reread_offers(self):
        tool = self.roles.get("offer") or self.roles.get("open")
        tgt = self.single_target(tool) if tool else None
        if not tool or not tgt:
            return ""
        cut = max(self.steps - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if i not in self.gone and self.offer_seen.get(i, -99) < cut]
        if not stale or self.left() < len(stale) + 5:
            return ""
        lines = []
        for lid in stale:
            a, f = shape_args(self.schemas.get(tool) or {}, {tgt: lid})
            if f or self.rushed():
                break
            before = dict(self.rows.get(lid) or {})
            r = self.env_call(tool, a)
            if r is None or self.finished:
                break
            self.offer_seen[lid] = self.steps
            obs = r.get("observation")
            self.observe(tool, a, obs)
            after = self.rows.get(lid) or {}
            ch = [k for k in before if k in after and after[k] != before[k] and not isinstance(after[k], (dict, list))]
            if ch:
                lines.append("%s changed (%s): %s" % (lid, ", ".join(ch[:4]), cjson(clip_deep(obs, 300))[:900]))
        if lines:
            self.version += 1
        return "\n".join(lines)

    def swap_hint(self):
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for i in lost:
            self.gone[i] = self.steps
        self.leads = [l for l in self.leads if l not in lost]
        self.version += 1
        reps = self.replacements(lost)
        t = ("[Cairn] %s sold out from now on (the store shows they cannot be purchased any more); they are gone for"
             " good and must not be opened again. " % ", ".join(lost))
        if reps and self.left() <= 6:
            t += ("Steps are scarce: within this very turn put %s in the cart in the quantity asked and remove the line"
                  " that sold out (that is two calls), and then order. Other already-read listings that seem to"
                  " match: %s." % (reps[0], ", ".join(reps[1:]) or "none"))
        elif reps and self.brief.get("pick") == "cheapest":
            t += ("Already-read listings that seem to match that item, lowest price first: %s. Since the request wants"
                  " the lowest price, unless you have already seen every matching listing, make one search capped"
                  " below the price of %s (through the search's price ceiling argument, if the search has one) ahead"
                  " of replacing the line that sold out with the lowest-priced matching listing." % (", ".join(reps),
                                                                                                    reps[0]))
        elif reps:
            t += ("Already-read listings that seem to match that item: %s. Within this turn remove the line that sold"
                  " out, put %s in the cart in the quantity asked, and order after that." % (", ".join(reps), reps[0]))
        else:
            t += ("Remove that line and put in the best other matching listing, searching when none of those already"
                  " read matches.")
        return t

    def sold_out_finder(self):
        lines = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in lines):
            return ""
        if len(lines) == 1:
            self.unavailable.add(lines[0])
            return ""
        if len(lines) >= 2:
            return self.reread_offers()
        return ""

    def replacements(self, lost):
        j = self.judges()
        if not j:
            return []
        recs = []
        for i in reversed(self.seen):
            if i in lost or i in self.gone or self.sold_out(i):
                continue
            recs.append(cjson(clip_deep(self.full.get(i) or self.rows.get(i), 200))[:1200])
            if len(recs) >= 25:
                break
        if not recs:
            return []
        prompt = (SWAP_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nLIVE REQUIREMENTS:\n" + self.ledger()
                  + "\n\nTHE LINE THAT SOLD OUT:\n" + "\n".join("%s: %s" % (i, self.record_text(i)) for i in lost)
                  + "\n\nCURRENT CART: " + self.cart_text() + "\n\nREMAINING RECORDS SEEN (latest first):\n"
                  + "\n".join(recs))
        ans = self.ask_json(prompt, list(reversed(j)), 3500) or {}
        cart = set(self.cart_ids())
        kept = []
        for e in items_of(ans.get("fits")):
            i = str(e.get("id")) if isinstance(e, dict) else str(e)
            p = e.get("price") if isinstance(e, dict) else None
            p = p if isinstance(p, (int, float)) and not isinstance(p, bool) else float("inf")
            if i in self.rows and i not in lost and i not in self.gone and i not in cart and i not in [k for k, _ in kept]:
                kept.append((i, p))
        if self.brief.get("pick") == "cheapest":
            kept.sort(key=lambda x: x[1])
        return [i for i, _ in kept][:4]

    def audit(self):
        if self.audited:
            return ""
        if not self.judges():
            return ""
        self.audited = True
        ans = self.audit_call()
        fits = self.audit_fits(ans)
        if not fits:
            ops = []
            for x in items_of((ans or {}).get("open")):
                x = str(x)
                if x in self.rows and x not in self.opened and not self.sold_out(x) and x not in ops:
                    ops.append(x)
            if ops and self.audit_open(ops):
                ans = self.audit_call()
                fits = self.audit_fits(ans)
        if not fits:
            return ""
        self.lead_items = fits
        self.leads = []
        for v in fits.values():
            for i in v:
                if i not in self.leads:
                    self.leads.append(i)
        reason = str((ans or {}).get("reason") or "").strip()
        t = "The review believes the following listings could match: " + "; ".join(
            "item %s: %s" % (k, ", ".join(v)) for k, v in fits.items())
        if reason:
            t += " (%s)" % reason
        return t + ". Open these listings and purchase what really matches; decline only when every one of them fails."

    def audit_call(self):
        j = self.judges()
        ids = sorted(reversed(self.seen), key=lambda i: i not in self.full)
        lines = []
        for i in ids:
            if self.sold_out(i):
                continue
            if i in self.full:
                lines.append("OPENED " + cjson(clip_deep(self.full[i], 200))[:1200])
            else:
                lines.append("ROW " + cjson(clip_deep(self.rows.get(i), 120))[:500])
            if len(lines) >= 30:
                break
        if not lines or not j:
            return None
        prompt = (AUDIT_PROMPT + "\n\nREQUEST:\n" + self.request + "\n\nLIVE REQUIREMENTS:\n" + self.ledger()
                  + "\n\nRECORDS SEEN (latest first):\n" + "\n".join(lines))
        ans = self.ask_json(prompt, [LIGHT], 4000)
        if HEAVY in j and not self.no_credit and not self.audit_fits(ans):
            a2 = self.ask_json(prompt, [HEAVY], 4000)
            if a2 is not None:
                ans = a2
        return ans

    def audit_fits(self, ans):
        fits = (ans or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        out = {}
        for k, v in fits.items():
            vals = v if isinstance(v, list) else [v]
            ids = [str(x) for x in vals if x is not None and str(x) in self.rows and not self.sold_out(str(x))][:4]
            if ids:
                out[str(k)] = ids
        return out

    def audit_open(self, ids):
        cmp = self.roles.get("compare")
        tgt = self.compare_target(cmp) if cmp else None
        if cmp and tgt and self.compare_facts() is not False:
            plan = [(cmp, {tgt: ids[:self.compare_room()]})]
        else:
            ft = self.facts_tool()
            ftt = self.single_target(ft) if ft else None
            if not (ft and ftt):
                return False
            plan = [(ft, {ftt: i}) for i in ids[:2]]
        calls = self.cart_calls(dict(("#%d" % k, 1) for k in range(len(self.items))))
        cs = len(calls) if calls is not None else len(self.items)
        if self.left() < len(plan) + max(5, self.reserve() + cs):
            return False
        before = len(self.full)
        for t, a in plan:
            a2, f = shape_args(self.schemas.get(t) or {}, a)
            if f:
                return False
            r = self.env_call(t, a2)
            if r is None or self.finished:
                break
            self.observe(t, a2, r.get("observation"))
        return len(self.full) > before

    def cart_props(self):
        ct = self.roles.get("cart")
        sc = (self.schemas.get(ct) or {}) if ct else {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        return ct, props, req

    def cart_calls(self, wanted, drops=()):
        ct, props, req = self.cart_props()
        if not ct:
            return None
        for k, p in props.items():
            it = p.get("items") if isinstance(p, dict) else None
            if isinstance(p, dict) and "array" in kinds(p) and isinstance(it, dict) and isinstance(it.get("properties"), dict):
                ip = it["properties"]
                idk = next((n for n, q in ip.items() if isinstance(q, dict) and "string" in kinds(q) and not q.get("enum")), None)
                ck = next((n for n, q in ip.items() if isinstance(q, dict) and "integer" in kinds(q)), None)
                if not idk:
                    return None
                els = []
                for i, u in wanted.items():
                    e = {idk: i}
                    if ck:
                        e[ck] = u
                    els.append(e)
                return [(ct, {k: els})]
        texts = [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p) and not p.get("enum")]
        target = next((a for (t, a) in self.id_args if t == ct and a in texts), None)
        if target is None:
            target = next((k for k in req if k in texts), None)
        if target is None and len(texts) == 1:
            target = texts[0]
        if not target:
            return None
        cnt = next((k for k, p in props.items() if isinstance(p, dict) and "integer" in kinds(p)), None)
        mode = next((k for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list)
                     and len(p["enum"]) >= 2), None)
        setsc = self.sets_counts()
        calls = []
        for i in drops:
            held = self.cart.get(i, 1)
            if mode:
                a = {target: i, mode: props[mode]["enum"][1]}
                if cnt:
                    a[cnt] = held
            elif cnt and setsc:
                a = {target: i, cnt: 0}
            elif cnt and is_whole(props[cnt].get("minimum")) and props[cnt]["minimum"] <= -held:
                a = {target: i, cnt: -held}
            else:
                return None
            calls.append((ct, a))
        cids = set(self.cart_ids())
        for i, w in wanted.items():
            held = self.cart.get(i, 0) if i in cids else 0
            if held == w:
                continue
            if held > 0 and not setsc:
                return None
            a = {target: i}
            if mode:
                a[mode] = props[mode]["enum"][0]
            if cnt:
                a[cnt] = w
            calls.append((ct, a))
        return calls

    def sets_counts(self):
        _, props, _ = self.cart_props()
        if any(isinstance(p, dict) and isinstance(p.get("enum"), list) and len(p["enum"]) >= 2 for p in props.values()):
            return False
        for p in props.values():
            if isinstance(p, dict) and "integer" in kinds(p):
                mn = p.get("minimum")
                if isinstance(mn, (int, float)) and not isinstance(mn, bool) and mn == 0:
                    return True
        return False

    def whole_removal(self, args):
        _, props, _ = self.cart_props()
        cnt = next((k for k, p in props.items() if isinstance(p, dict) and "integer" in kinds(p)), None)
        if not cnt:
            return args
        removal = any(isinstance(props.get(k), dict) and isinstance(props[k].get("enum"), list)
                      and len(props[k]["enum"]) >= 2 and v in props[k]["enum"][1:] for k, v in args.items())
        line = next((v.strip() for v in args.values() if isinstance(v, str) and v.strip() in self.cart), None)
        if removal and line and cnt not in args:
            held = self.cart.get(line, 1)
            mx = props[cnt].get("maximum")
            if isinstance(mx, (int, float)) and not isinstance(mx, bool):
                held = min(held, int(mx))
            a = dict(args)
            a[cnt] = held
            return a
        return args

    def note_cart(self, tool, args, obs):
        self.cart_touched = True
        self.cart_changed_at = self.steps
        sc = self.schemas.get(tool) or {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        removal = any(isinstance(props.get(k), dict) and isinstance(props[k].get("enum"), list)
                      and len(props[k]["enum"]) >= 2 and v in props[k]["enum"][1:] for k, v in args.items())
        loose = None
        for v in args.values():
            if is_whole(v) or (isinstance(v, float) and v == int(v)):
                loose = int(v)
                break
        lines, whole = [], False
        for k, v in args.items():
            if isinstance(v, list):
                p = props.get(k) if isinstance(props.get(k), dict) else {}
                it = p.get("items")
                objel = isinstance(it, dict) and ("object" in kinds(it) or isinstance(it.get("properties"), dict))
                if not v and objel:
                    whole = True
                for e in v:
                    if isinstance(e, dict):
                        whole = True
                        lid = next((x for x in e.values() if isinstance(x, str) and x in self.rows), None)
                        c = next((count_int(x) for x in e.values() if not isinstance(x, str) and count_int(x) is not None), None)
                        if lid:
                            lines.append((lid, c if c is not None else 1))
                    elif isinstance(e, str) and e in self.rows:
                        lines.append((e, loose))
            elif isinstance(v, str) and v.strip() in self.rows:
                lines.append((v.strip(), loose))
        if whole:
            self.cart = dict((i, c) for i, c in lines if c is not None and c > 0)
            return
        has_int = any(isinstance(p, dict) and "integer" in kinds(p) for p in props.values())
        setsc = self.sets_counts() if tool == self.roles.get("cart") else False
        for lid, c in lines:
            held = self.cart.get(lid, 0)
            if removal:
                if has_int:
                    held -= c if c else 1
                else:
                    held = held - c if c is not None else 0
            elif setsc:
                held = c if c is not None else 1
            else:
                held += c if c is not None else 1
            if held > 0:
                self.cart[lid] = held
            else:
                self.cart.pop(lid, None)

    def complete_cart(self):
        if self.completed or self.confirmed is None or self.sold_out(self.confirmed):
            return ""
        if not self.roles.get("cart") or not self.roles.get("buy"):
            return ""
        self.completed = True
        ck = self.cat_key()

        def kind(i):
            return str((self.rows.get(i) or {}).get(ck) or "").strip().lower() if ck is not None else i
        plan, pk = {}, set()
        for key, ids in self.lead_items.items():
            m = E8_DIGITS.search(str(key))
            n = int(m.group()) if m else (1 if len(self.items) == 1 else None)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and n and 1 <= n <= len(self.items) and kind(pick) not in pk:
                plan[pick] = self.items[n - 1]["qty"]
                pk.add(kind(pick))
        if self.confirmed not in plan:
            return ""
        keep, drops = {}, []
        for i in self.cart_ids():
            if i in plan:
                continue
            if not self.sold_out(i) and kind(i) not in pk:
                keep[i] = self.cart.get(i, 1)
            else:
                drops.append(i)
        if len(plan) + len(set(kind(i) for i in keep)) < len(self.items):
            return ""
        wanted = dict(keep)
        wanted.update(plan)
        calls = self.cart_calls(wanted, drops)
        if not calls or self.left() < len(calls) + 1:
            return ""
        ids = list(wanted)
        self.fetch_facts(ids, len(calls) + 3)
        self.follow_links(ids, len(calls) + 3)
        self.cart_check(planned=ids, limit=HARD_SPEND)
        if self.verdicts.get(self.check_key(ids)) is not True:
            return ""
        done = []
        for t, a in calls:
            a2, f = shape_args(self.schemas.get(t) or {}, a)
            if f:
                return ""
            r = self.env_call(t, a2)
            if r is None or self.finished:
                return ""
            obs = r.get("observation")
            self.note_cart(t, a2, obs)
            self.observe(t, a2, obs)
            done.append(cjson(a2))
        buy = self.roles["buy"]
        ba, f = shape_args(self.schemas.get(buy) or {}, dict(self.brief.get("buy_args") or {}))
        if f:
            return ""
        kept = []
        for attempt in range(2):
            if attempt == 1:
                self.reread_offers()
                self.cart_check(planned=ids, limit=HARD_SPEND)
                if (self.verdicts.get(self.check_key(ids)) is not True or self.left() <= 0 or self.finished
                        or any(self.sold_out(i) for i in ids)):
                    break
            stop, gt = self.gate("order")
            if gt:
                kept.append(gt)
            if stop:
                break
            r = self.env_call(buy, ba)
            if r is None:
                break
            self.checkouts += 1
            if not self.finished:
                self.refused_at = self.steps
            obs = r.get("observation")
            self.observe(buy, ba, obs)
            kept.append(self.display(obs))
            if self.finished:
                break
        if not kept:
            return ""
        return ("[Cairn] The decline was not sent, because Cairn acted in its place: the independent check passed %s,"
                " so Cairn changed the cart (%s) and placed the order. %s" % (self.confirmed, "; ".join(done),
                                                                            " then ".join(kept)))

    def is_order(self, tool, args):
        if not tool or tool != self.roles.get("buy"):
            return False
        ba = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(k in args and args[k] == v for k, v in ba.items())

    def is_decline(self, tool, args):
        if not tool or tool != self.roles.get("refuse"):
            return False
        ra = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(k in args and args[k] == v for k, v in ra.items())

    def offered_tools(self):
        names = [t["function"]["name"] for t in self.tools]
        ending = [x for x in (self.roles.get("buy"), self.roles.get("refuse")) if x]
        sel = names
        if ending:
            res, l = self.reserve(), self.left()
            pick = None
            if l <= res:
                pick = set(ending)
            elif l <= res + 3:
                pick = set(ending + [x for x in (self.roles.get("cart"), self.roles.get("cart_view")) if x])
            elif l <= len(self.items) + 5 and not self.checkouts:
                pick = set(ending + [x for x in (self.roles.get("cart"), self.roles.get("cart_view"),
                                                 self.roles.get("offer")) if x])
            if pick is not None:
                sel = [n for n in names if n in pick] or names
        ask = self.roles.get("ask")
        if ask and self.msg_args("x" * 200) is not None:
            sel = [n for n in sel if n != ask]
        return sel

    def quiet_fill(self, args):
        dt = self.roles.get("refuse")
        sc = (self.schemas.get(dt) or {}) if dt else {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        qd = self.brief.get("quiet_decline")
        args = dict(args)
        for k, p in props.items():
            if not isinstance(p, dict) or "string" not in kinds(p) or p.get("enum"):
                continue
            if k not in args and k not in req:
                continue
            v = args.get(k)
            vs = v if isinstance(v, str) else ""
            if not vs or any(i in vs for i in self.rows) or qd is not False:
                ml = p.get("maxLength")
                ml = ml if is_whole(ml) else 200
                args[k] = DECLINE_TEXT[:ml]
        return args

    def resolve_ids(self, tool, args):
        roles = [self.roles.get(r) for r in ("cart", "open", "offer", "compare")]
        if tool not in [r for r in roles if r] or self.id_key is None:
            return args, None
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        free = [k for k, p in props.items() if isinstance(p, dict) and ("string" in kinds(p) or "array" in kinds(p))
                and not p.get("enum")]
        if len(free) != 1:
            return args, None
        k = free[0]
        if tool != self.roles.get("cart") and (tool, k) not in self.id_args:
            return args, None

        def map1(x):
            if not isinstance(x, str):
                return x, None
            s = x.strip()
            if s in self.rows:
                return s, None
            if id_like(s, True) or len(s) < 8:
                return x, None
            low = s.lower()
            hits = [i for i, r in self.rows.items()
                    if any(isinstance(fv, str) and fv.lower().startswith(low) for fk, fv in r.items() if fk != self.id_key)]
            if len(hits) == 1:
                return hits[0], None
            if hits:
                return x, "more than one listing holds the text '%s' (%s); give a single listing id" % (s, ", ".join(hits[:6]))
            return x, ("'%s' is no listing id; give the id as the result rows show it, in the field %s" % (s, self.id_key))
        v = args.get(k)
        out = dict(args)
        if isinstance(v, list):
            new = []
            for x in v:
                m, p = map1(x)
                if p:
                    return args, p
                new.append(m)
            out[k] = new
        elif v is not None:
            m, p = map1(v)
            if p:
                return args, p
            out[k] = m
        return out, None

    def display(self, obs):
        has = bool(self.listing_records(obs))
        t = cjson(clip_deep(obs, ROW_CLIP if has else TEXT_CLIP))
        if len(t) > OBS_CAP:
            t = t[:OBS_CAP] + " ...(cut)"
        return t

    def compare_target(self, tool):
        if not tool:
            return None
        sc = self.schemas.get(tool) or {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        k = next((k for k, p in props.items() if isinstance(p, dict) and "array" in kinds(p)), None)
        if not k or any(r != k for r in req):
            return None
        return k

    def compare_room(self):
        cmp = self.roles.get("compare")
        tgt = self.compare_target(cmp) if cmp else None
        if tgt:
            mx = ((self.schemas.get(cmp) or {}).get("properties") or {}).get(tgt, {}).get("maxItems")
            if is_whole(mx):
                return mx
        if self.cmp_fail:
            return max(2, min(self.cmp_fail - 1, max(self.cmp_ok, 2)))
        return max(4, self.cmp_ok)

    def join_opens(self, calls):
        op, cmp = self.roles.get("open"), self.roles.get("compare")
        ot = self.single_target(op) if op else None
        ct = self.compare_target(cmp) if cmp else None
        if not (op and ot and cmp and ct and op != cmp and self.compare_facts() is not False):
            return calls
        picks = []
        for ix, c in enumerate(calls):
            if c["function"].get("name") != op:
                continue
            raw = c["function"].get("arguments")
            if not isinstance(raw, str):
                continue
            a = json_obj(raw)
            if not isinstance(a, dict) or list(a.keys()) != [ot]:
                continue
            v = a[ot]
            if isinstance(v, str) and v.strip() in self.rows and v.strip() not in self.gone:
                picks.append((ix, v.strip()))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        out = list(calls)
        first = picks[0][0]
        out[first] = {"id": calls[first]["id"], "type": "function",
                      "function": {"name": cmp, "arguments": cjson({ct: [p[1] for p in picks]})}}
        for ix, _ in picks[1:]:
            d = dict(calls[ix])
            d["joined"] = True
            out[ix] = d
        return out


E9_ID_TOKEN = re.compile(r"\"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})\"")

SKIP = "[Cairn] Not executed; no step was spent. "
HOLD = "[Cairn] Not executed yet; no step was spent. "
BARE_HOLD = "[Cairn] Not executed yet. "


class EpisodeLoop:
    def footer(self):
        return "\nCart now: " + self.cart_text()

    def tool_defs(self, names):
        out = []
        for t in self.tools:
            if t["function"]["name"] in names:
                out.append(t)
        return out

    def run(self):
        if self.finished:
            return self.trace.items
        self.read_brief()
        self.trace.add({"kind": "brief_raw", "brief": self.brief})
        if self.policy.get("opening") is not None:
            self.absorb(restore(self.policy.get("opening"), True))
        text = ""
        if not self.finished:
            text = self.opening_moves()
        self.gap = STORE_PAUSE
        msg = "SHOPPER'S REQUEST:\n" + self.request
        if self.policy.get("opening") is not None:
            msg += "\n\nSTORE'S STARTING DISPLAY:\n" + self.display(self.policy.get("opening"))
        if text:
            msg += ("\n\nCairn has already searched for every item, and part of each item's listings were read in full,"
                    " one step per read:\n" + text + "\nJudge these listings by the requirements, and search more only"
                    " where something is still lacking.\n\n" + self.status())
        else:
            msg += ("\n\nBegin with a search for each item requested, sought within the category it belongs to, and"
                    " read several listings completely, so that the interviewer can then tell which properties such"
                    " listings show.\n\n" + self.status())
        self.convo = [{"role": "system", "content": GUIDE}, {"role": "user", "content": msg}]
        self.work(TURN_CAP, HARD_SPEND)
        if not self.finished:
            t = self.finish_by_code(again=True)
            if t and not self.finished and not self.store_broke and not self.no_credit and self.left() > 1:
                self.convo.append({"role": "user", "content": "[Cairn] Cairn attempted to finish and " + t
                                   + "\nFollow the store's answer, then finish.\n" + self.status()})
                self.work(self.turns + 10, HARD_SPEND + 0.01)
        if not self.finished:
            self.finish_by_code(again=False)
        self.trace.add({"kind": "end", "finished": self.finished, "steps": self.steps,
                        "spend": round(self.spend, 5), "calls": self.calls})
        return self.trace.items

    def work(self, cap, spend_cap):
        for _ in range(CRASH_RETRIES):
            try:
                self.turn_loop(cap, spend_cap)
                return
            except Exception as e:
                self.trace.add({"kind": "crash", "error": clip(type(e).__name__ + ": " + str(e), 400)})
                self.answer_open()

    def answer_open(self):
        last = None
        for ix in range(len(self.convo) - 1, -1, -1):
            if self.convo[ix].get("tool_calls"):
                last = ix
                break
        if last is None:
            return
        answered = set(m.get("tool_call_id") for m in self.convo[last + 1:] if m.get("role") == "tool")
        for c in self.convo[last]["tool_calls"]:
            if c.get("id") not in answered:
                self.convo.append({"role": "tool", "tool_call_id": c.get("id"),
                                   "content": "[Cairn] This call possibly did not execute, because something failed"
                                              " along the way. Inspect the present state before repeating it."})
        self.convo.append({"role": "user", "content": self.status()})

    def loop_ok(self, cap, spend_cap):
        if self.finished or self.store_broke:
            return False
        if self.left() < 1 or self.turns >= cap or self.spend >= spend_cap:
            return False
        if self.clock() < LAST_TURN:
            return True
        return not self.cart_ids() and self.room() > EMPTY_CART_ROOM

    def turn_loop(self, cap, spend_cap):
        idle = 0
        while self.loop_ok(cap, spend_cap):
            if self.ask_ready():
                s = self.interview()
                if s:
                    self.convo.append({"role": "user", "content": s + "\n" + self.status()})
                continue
            self.trim()
            self.turns += 1
            names = self.offered_tools()
            msg = self.think(self.convo, [LIGHT, SPARE], 3000, self.tool_defs(names))
            if msg is None:
                self.misses += 1
                if self.no_credit or self.calls >= CALL_CAP:
                    return
                if self.misses > 2:
                    if self.clock() >= OUTAGE_CUTOFF:
                        return
                    time.sleep(OUTAGE_PAUSE)
                continue
            calls = []
            for pos, tc in enumerate(msg.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                a = fn.get("arguments")
                if isinstance(a, dict):
                    args = a
                elif isinstance(a, str) and a.strip():
                    args = json_obj(a) or {}
                else:
                    args = {}
                calls.append({"id": tc.get("id") or ("k%d_%d" % (self.turns, pos)), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": cjson(args)}})
            c = msg.get("content")
            text = c if isinstance(c, str) else (cjson(c) if c is not None else "")
            if not calls:
                idle += 1
                self.convo.append({"role": "assistant", "content": text or "(no call)"})
                self.convo.append({"role": "user", "content": "[Cairn] Use one of the tools so the work goes on.\n"
                                   + self.status()})
                if idle >= 3:
                    return
                continue
            idle = 0
            commit_roles = [self.roles.get(r) for r in ("cart", "buy", "refuse") if self.roles.get(r)]
            if (not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and self.left() > 8
                    and self.clock() <= ASK_DEADLINE
                    and any(cc["function"]["name"] in commit_roles for cc in calls)):
                self.convo.append({"role": "assistant", "content": text or None, "tool_calls": calls})
                for cc in calls:
                    self.convo.append({"role": "tool", "tool_call_id": cc["id"],
                                       "content": SKIP + "The shopper is questioned first."})
                s = self.interview()
                if not s:
                    s = "[Cairn] The interviewer had no question to put to the shopper."
                self.convo.append({"role": "user", "content": s + "\n" + self.status()})
                continue
            self.convo.append({"role": "assistant", "content": text or None, "tool_calls": calls})
            self.trace.add({"kind": "turn", "text": text[:300], "calls": [cc["function"] for cc in calls]})
            replies = self.dispatch(calls)
            for ix, (cc, rep) in enumerate(zip(calls, replies)):
                if ix == len(calls) - 1:
                    rep = rep + "\n" + self.status()
                self.convo.append({"role": "tool", "tool_call_id": cc["id"], "content": rep})

    def trim(self):
        total = sum(len(cjson(m.get("content"))) for m in self.convo)
        if total <= HISTORY_CAP:
            return
        tms = [m for m in self.convo if m.get("role") == "tool"]
        n = max(1, len(tms) // 2)
        for m in tms[:n]:
            c = m.get("content")
            c = c if isinstance(c, str) else cjson(c)
            if len(c) > 600:
                ids = []
                for x in E9_ID_TOKEN.findall(c):
                    if x in self.rows and x not in ids:
                        ids.append(x)
                m["content"] = c[:400] + "... (older result shortened; listing ids in it: " + ", ".join(ids[:20]) + ")"

    def parse_args(self, raw):
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            if not raw.strip():
                return {}
            return json_obj(raw)
        return {}

    def dispatch(self, calls):
        calls = self.join_opens(calls)
        out = []
        n = len(calls)
        for cc in calls:
            try:
                out.append(self.dispatch_one(cc, n))
            except Exception:
                raise
        return out

    def dispatch_one(self, cc, n):
        if cc.get("joined"):
            return ("[Cairn] This listing is in the comparison result given earlier in this turn: Cairn read them"
                    " jointly in a single step.")
        allowed = self.offered_tools()
        tool = cc["function"]["name"]
        if self.finished:
            return "[Cairn] Not executed: the episode has ended."
        if tool and tool == self.roles.get("ask") and self.msg_args("x" * 200) is not None:
            return SKIP + "Cairn's interviewer sends all shopper messages."
        if tool not in self.schemas:
            return SKIP + "unknown tool; the tools are: " + ", ".join(sorted(allowed))
        if tool not in allowed:
            return SKIP + "too few steps remain for this tool; use one of: " + ", ".join(sorted(allowed))
        args = self.parse_args(cc["function"].get("arguments"))
        if args is None:
            return SKIP + "the argument text did not form exactly one JSON object."
        args, faults = shape_args(self.schemas.get(tool) or {}, args)
        if faults:
            return SKIP + "; ".join(faults[:6])
        args, problem = self.resolve_ids(tool, args)
        if problem:
            return SKIP + problem + "."
        is_cart = tool == self.roles.get("cart")
        if is_cart:
            args = self.whole_removal(args)
        tail = ""
        cmp_tool = self.roles.get("compare")
        if tool == cmp_tool:
            tgt = self.compare_target(tool)
            room = self.compare_room()
            if tgt and isinstance(args.get(tgt), list) and self.cmp_fail and len(args[tgt]) > room:
                rest = args[tgt][room:]
                args = dict(args)
                args[tgt] = args[tgt][:room]
                tail += ("\n[Cairn] Only the first %d listings went out, as this store's comparison takes no more than"
                         " %d; not yet sent: %s" % (room, room, ", ".join(str(x) for x in rest)))
        st = self.single_target(tool)
        if st and tool in (self.roles.get("open"), self.roles.get("offer")):
            v = args.get(st)
            if isinstance(v, str) and v.strip() in self.gone:
                return (SKIP + "this listing's sell-out was recorded at step %d and it will not return. Put a different"
                        " listing that fits in place of its cart line." % self.gone[v.strip()])
        if is_cart and any(isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0 for v in args.values()):
            vt = cjson(self.cart_view) if self.fresh_view() else ""
            for v in args.values():
                if isinstance(v, str) and v.strip() in self.rows:
                    s = v.strip()
                    if s not in self.cart and s not in vt and s not in (self.cart_said or ""):
                        return SKIP + "listing %s is not held, so no units can come off it." % s + self.footer()
        is_order = self.is_order(tool, args)
        is_decline = self.is_decline(tool, args)
        fin = is_order or is_decline
        if fin and n > 1:
            return SKIP + ("a finishing call must go alone, once the results of the other calls have been read.")
        if not fin and self.left() <= self.reserve() and (self.roles.get("buy") or self.roles.get("refuse")):
            extra = " (with the shopper approving the basket first)" if self.gate_due("order") and self.duties else ""
            return SKIP + "the last few steps are reserved for the order%s or for a decline." % extra
        role_tools = set(v for k, v in self.roles.items() if k != "search")
        search_like = tool not in role_tools
        fresh = False
        if search_like:
            fresh = self.query_fresh(tool, args)
            if fresh:
                na = self.first_page(tool, args)
                if na != args:
                    tail += ("\n[Cairn] This being a new search, Cairn sent it for page one; you had sent "
                             + cjson(args) + ".")
                    args = na
        args = self.map_row_ids(tool, args)
        if tool == self.roles.get("search") and self.codes_dead and self.names_code(args):
            return (SKIP + "code searches in this store returned nothing. Search with words and attribute values"
                    " instead, browse the category listing, and take the codes from the rows shown.")
        mark = tool + cjson(args)
        readers = [self.roles.get(r) for r in ("search", "open", "offer", "compare") if self.roles.get(r)]
        if not fin and not is_cart and mark in self.recent:
            return SKIP + "this exact call was just made, with no change since; the earlier result stands."
        if tool in readers:
            rc = self.read_cache.get(mark)
            if rc and rc[0] == self.version:
                return (SKIP + "the identical call at step %d gave the result below, with no change since then.\n"
                        % rc[1] + rc[2][:3000])
        if tool in readers and tool != cmp_tool and st:
            v = args.get(st)
            if isinstance(v, str):
                lid = v.strip()
                if self.shown_in_full(tool, lid) or (tool, lid) in self.factless:
                    return (SKIP + "this view has nothing new to show for that listing; its record: "
                            + self.record_text(lid))
        if tool == cmp_tool:
            tgt = self.compare_target(tool)
            if tgt and isinstance(args.get(tgt), list) and args[tgt]:
                keep = [x for x in args[tgt] if not (isinstance(x, str) and self.shown_in_full(tool, x.strip()))]
                if not keep:
                    lines = "\n".join("%s: %s" % (x, self.record_text(str(x).strip())) for x in args[tgt])
                    return SKIP + "this view has nothing new to show for these listings; their records:\n" + lines[:4000]
                if len(keep) < len(args[tgt]):
                    left_out = [x for x in args[tgt] if x not in keep]
                    tail += ("\n[Cairn] Left out as already shown in full with no change since: "
                             + ", ".join(str(x) for x in left_out))
                    args = dict(args)
                    args[tgt] = keep
                    mark = tool + cjson(args)
        if is_order and self.left() > 2:
            r = self.order_holds(tool, args)
            if r is not None:
                return r
        if is_decline and self.left() > 5:
            r = self.decline_holds()
            if r is not None:
                return r
        if is_decline and self.left() <= 5 and not self.late_held:
            built = [i for i in self.built if not self.sold_out(i)]
            ids = self.cart_ids()
            missing = [i for i in built if i not in ids]
            if not self.fresh_view():
                missing += [i for i in built if i in self.dropped and i not in missing]
            if len(built) >= len(self.items) and len(missing) + 1 <= self.left():
                self.late_held = True
                self.trace.add({"kind": "late_decline_hold", "missing": missing})
                t = (HOLD + "A decline scores zero whenever a fitting listing exists, and running low on steps is no"
                     " evidence about whether one exists. ")
                if missing:
                    t += ("Cart these lines again in the quantities asked: %s, and order. " % ", ".join(missing))
                else:
                    t += "Order the cart as it stands. "
                t += ("Decline only if the searching proved that, for one of the requested items, no listing satisfies"
                      " its requirements, and in that case send the decline once more.")
                return t
        if is_decline and not self.leads and self.left() > self.max_steps // 3 and not self.pushed and self.left() > 5:
            self.pushed = True
            return (HOLD + "%d steps remain. Try untried kinds of search before giving up: list every requested item's"
                    " category and page onward through that listing using the page token, or offset value, returned"
                    " by the store; try one-word queries of values and alternative names of the property; open any"
                    " listing that could fit. Decline only after all that." % self.left())
        if is_decline:
            if self.confirmed is not None:
                t = self.complete_cart()
                if t:
                    return t + (" [episode finished]" if self.finished else "")
            args = self.quiet_fill(args)
        if is_order:
            s = self.late_question()
            if s:
                return BARE_HOLD + s + self.footer()
        if fin and self.left() >= 2:
            stop, gt = self.gate("order" if is_order else "end")
            if stop:
                return BARE_HOLD + (gt or "") + self.footer()
            if gt:
                tail = "\n[Cairn] " + gt + tail
            if self.finished:
                return "[Cairn] The episode has ended."
        return self.execute(tool, args, mark, tail, fin, is_order, is_cart, search_like, fresh, st)

    def order_holds(self, tool, args):
        hunt_text = self.cheaper_hunt()
        hunt_ran = bool(hunt_text)
        w = self.wider_hold()
        if w:
            return w + self.footer()
        change = self.reread_offers()
        if self.note_sold:
            self.sold_out_finder()
        if (self.cart_touched and not self.fresh_view() and self.roles.get("cart_view") and self.left() > 5
                and not self.rushed()):
            cv = self.roles["cart_view"]
            a, f = shape_args(self.schemas.get(cv) or {}, {})
            if not f:
                res = self.env_call(cv, a)
                if res is not None:
                    self.cart_view = res.get("observation")
                    self.cart_seen_at = self.steps
                    self.cart_seen_version = self.cart_changed_at
                    ch = self.observe(cv, a, res.get("observation"))
                    if ch:
                        self.queued = []
                        return (HOLD + "viewing the cart revealed a change: " + ch + self.footer()
                                + "\nApply it, repair the cart where necessary, then order.")
        hint = self.swap_hint()
        gone_lines = []
        if self.sold_holds < 2:
            gone_lines = [i for i in self.cart_ids() if i in self.gone]
        if hint or gone_lines:
            if not hint:
                self.sold_holds += 1
                return (HOLD + "these carted lines are sold out: %s. Replace that carted line first." %
                        ", ".join(gone_lines) + self.footer())
            return HOLD + hint + self.footer()
        trail_text, trail_hold = self.trail()
        if trail_hold:
            return HOLD + trail_text + self.footer()
        if self.fit_holds < 2 and self.left() > self.reserve() + 3:
            ff = self.fit_faults()
            if ff:
                self.fit_holds += 1
                self.trace.add({"kind": "fit_hold", "faults": ff[:4]})
                return (HOLD + "a carted part is incompatible with the carted device: " + "; ".join(ff[:3])
                        + ". Replace the part, or else the device, and then order." + self.footer())
        if self.line_holds < 2 and self.left() > self.reserve() + 2:
            lf = self.line_faults()
            if lf and all(k == "count" for k, _ in lf):
                fixes = self.count_fixes(lf)
                if fixes:
                    lf = self.line_faults()
                    if not lf:
                        self.line_holds += 1
                        return (HOLD + "Cairn adjusted the unit counts itself: " + "; ".join(fixes)
                                + ". If the store rejected any of those fixes, correct those lines. Then order."
                                + self.footer())
            if lf:
                self.line_holds += 1
                self.trace.add({"kind": "line_hold", "faults": [t for _, t in lf[:6]]})
                return (HOLD + "the cart and the request do not match yet: " + "; ".join(t for _, t in lf[:6])
                        + ". Correct those lines with the means the cart tool offers for adding and removing units,"
                          " then order." + self.footer())
        ids = self.cart_ids()
        res = self.reserve() + 1 if len(ids) >= 2 else self.reserve() + 3
        fetched = self.fetch_facts(ids, res)
        self.follow_links(ids, self.reserve() + 1)
        fetched += self.cheaper_rows()
        verdict = self.cart_check()
        if verdict:
            t = BARE_HOLD + (trail_text or "")
            if hunt_ran:
                t += hunt_text + " " + (self.below_cart() or "") + "\n"
            if fetched:
                t += ("Cairn fetched the attribute records of the listings below for this review, at one step"
                      " apiece:\n" + "\n".join(fetched) + "\n")
            return t + verdict + self.footer()
        if change and self.spend > SOFT_SPEND:
            return (HOLD + "Cairn has just fetched the cart's listings again: " + change
                    + "\nOrder again once all lines are still in stock and priced within budget; otherwise swap that"
                      " line before ordering.")
        return None

    def decline_holds(self):
        t = self.decline_trail()
        if t:
            return HOLD + t
        self.fit_hunt()
        t = self.audit()
        if t:
            return HOLD + t
        if self.leads and self.left() > 4 and self.decline_blocks == 0:
            self.decline_blocks = 1
            self.trace.add({"kind": "lead_hold", "leads": self.leads})
            return (HOLD + "The review named listings that seem to fit: %s. Open them where necessary, cart whichever"
                    " fits best while obeying the pick rule, then order. Decline only if all of them plainly fail"
                    " some requirement, and in that case state in your own reply text (not a shopper message) which"
                    " one fails and send the decline once more." % ", ".join(str(x) for x in self.leads))
        if self.leads and self.left() > 4 and self.decline_blocks == 1:
            self.decline_blocks = 2
            first = self.leads[0]
            self.cart_check(planned=[first], limit=HARD_SPEND)
            ok = self.verdicts.get(self.check_key([first])) is True
            if ok and first in self.leads:
                self.confirmed = first
                self.trace.add({"kind": "confirmed", "id": first})
                return (HOLD + "A second, separate check of the records of %s finds that it satisfies all"
                        " requirements. Cart it in the quantity asked and order. Decline only when its record plainly"
                        " breaks a requirement." % first)
        return None

    def execute(self, tool, args, mark, tail, fin, is_order, is_cart, search_like, fresh, st):
        res = self.env_call(tool, args)
        if res is None:
            return ("[Cairn] The call broke down through a network or server problem; you may retry it.")
        obs = res.get("observation")
        cmp_tool = self.roles.get("compare")
        if tool == cmp_tool:
            tgt = self.compare_target(tool)
            lst = args.get(tgt) if tgt else None
            if isinstance(lst, list) and len(lst) > 1:
                sj = cjson(obs)
                if not any(str(x) in sj for x in lst):
                    self.cmp_fail = min(self.cmp_fail or 999, len(lst))
                    tail += ("\n[Cairn] The comparison came back empty for %d listings. Send no more than %d per call."
                             % (len(lst), self.compare_room()))
                else:
                    self.cmp_ok = max(self.cmp_ok, len(lst))
        self.recent = (self.recent + [mark])[-3:]
        for k, v in args.items():
            vals = v if isinstance(v, list) and v else [v]
            if any(isinstance(x, str) and x in self.rows for x in vals):
                self.id_args.add((tool, k))
        self.absorb(obs)
        if tool == cmp_tool:
            recs = [r for _, r in self.listing_records(obs)]
            if recs:
                self.facts[tool] = bool(self.facts.get(tool) is True or any(attr_block(r) for r in recs))
                if tool not in self.facts_order:
                    self.facts_order.append(tool)
        readers = [self.roles.get(r) for r in ("open", "offer", "compare") if self.roles.get(r)]
        if tool in readers:
            self.record_view_keys(tool, obs)
        notes = ""
        if search_like and not fin:
            notes += self.auto_page(tool, args, obs) or ""
            if fresh:
                notes += self.last_page_first(tool, args, obs) or ""
            notes += self.shorter_search(tool, args, obs) or ""
        excl = [self.roles.get(r) for r in ("cart", "buy", "refuse", "ask") if self.roles.get(r)]
        if st and tool not in excl:
            self.single_view(tool, args, obs, st)
        if is_cart:
            self.note_cart(tool, args, obs)
            self.cart_said = obs if isinstance(obs, str) else cjson(obs)
            self.recent = []
            counts = [v for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if counts:
                for v in args.values():
                    if isinstance(v, str) and v.strip() in self.rows:
                        s = v.strip()
                        if counts[0] <= 0 and not row_has(s, obs):
                            self.dropped.add(s)
                        elif counts[0] > 0:
                            self.dropped.discard(s)
            live = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
            if len(live) >= len(self.built):
                self.built = live
        if tool == self.roles.get("search") and self.names_code(args):
            self.code_tries.append(bool(self.ids_in(obs)))
            if len(self.code_tries) >= 2 and not any(self.code_tries):
                self.codes_dead = True
        if tool == self.roles.get("search") and self.ids_in(obs):
            self.searched = (self.searched + [dict(args)])[-8:]
        if tool == self.roles.get("cart_view"):
            self.cart_view = obs
            self.cart_seen_at = self.steps
            self.cart_seen_version = self.cart_changed_at
        if is_order:
            self.checkouts += 1
            if not self.finished:
                self.refused_at = self.steps
        text = self.display(obs) + notes + tail
        if search_like and not fin:
            ch = self.cheapest_here(obs)
            if ch:
                text += ch
            ci = self.cap_ignored(tool, args, obs)
            if ci:
                text += ci
        cacheable = text
        ev = self.evidence(obs)
        if ev:
            text += "\n" + ev
        if is_cart:
            zero = [v for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if zero and zero[0] == 0:
                if any(isinstance(v, str) and v.strip() in self.rows and row_has(v.strip(), obs) for v in args.values()):
                    text += ("\n[Cairn] A zero count did not remove the line: this tool adds the number given to the"
                             " line, and removing units takes a negative number.")
        self.note_sold = False
        lines = self.loose_lines(obs)
        flagged = False
        if lines:
            meaning = self.read_note(lines)
            if meaning:
                text += "\n[note read] " + meaning + " Carry the change into the picks before ordering."
            flagged = self.note_sold
        refused = is_order and not self.finished
        sp = self.spoken(obs)
        again = refused and any(l in sp for l in self.sold_lines)
        loose = flagged and (refused or (not self.note_named and not is_order))
        if (loose or again) and self.cart_ids() and not self.finished:
            t = self.sold_out_finder()
            if t:
                text += "\n[Cairn fetched the cart lines again following the rejection]\n" + t
        self.note_sold = False
        hint = self.swap_hint()
        if hint:
            text += "\n" + hint
        if is_cart and self.fit_pairs and not self.finished:
            ff = self.fit_faults()
            if ff:
                text += "\n[Cairn] These lines are not yet compatible with each other: " + "; ".join(ff)
        if self.queued:
            text += "\n[store notes seen during Cairn's own calls] " + " | ".join(self.queued)
            self.queued = []
        if tool in readers or tool == self.roles.get("search"):
            self.read_cache[mark] = (self.version, self.steps, cacheable)
        if self.finished:
            text += " [episode finished]"
        return text

    def single_view(self, tool, args, obs, st):
        if self.listing_id(obs, False, 0) is None:
            return
        recs = [obs]
        self.note_facts(tool, recs[0])
        bare = self.facts.get(tool) is False
        if tool == self.roles.get("open") and bare:
            excl = set(self.roles.get(r) for r in ("cart", "buy", "refuse", "ask", "search") if self.roles.get(r))
            for t in self.tools:
                nm = t["function"]["name"]
                if nm == tool or nm in excl or self.facts.get(nm) is False:
                    continue
                if not self.single_target(nm):
                    continue
                if self.lists_things(nm):
                    continue
                self.roles["open"] = nm
                break
        if bare and tool != self.roles.get("open") and not self.roles.get("offer"):
            self.roles["offer"] = tool
        if tool == self.roles.get("offer") or (not self.roles.get("offer") and tool == self.roles.get("open")):
            lid = self.listing_id(recs[0])
            if lid is not None:
                self.offer_seen[lid] = self.steps

    def refusal_possible(self):
        dt = self.roles.get("refuse")
        if not dt:
            return False
        a = self.quiet_fill(dict(self.brief.get("refuse_args") or {}))
        _, f = shape_args(self.schemas.get(dt) or {}, a)
        return not f

    def finish_by_code(self, again=False):
        if self.finished or self.store_broke or self.left() <= 0:
            return ""
        ids = self.cart_ids()
        holding = bool(ids)
        whole = True
        if holding and len(self.items) >= 2:
            covered = self.fill_short_cart()
            whole = covered or not self.refusal_possible()
            ids = self.cart_ids()
        if whole and len(ids) >= 2:
            self.fetch_facts(ids, self.reserve())
            if self.left() >= 3 and self.room() > 5:
                self.follow_links(ids, self.reserve())
        if not whole:
            self.trace.add({"kind": "short_cart_decline"})
        if self.finished or self.left() <= 0:
            return ""
        buy = self.roles.get("buy")
        holding = bool(self.cart_ids())
        if whole and buy and (holding or (self.cart_touched and not self.fresh_view() and self.left() >= 2)):
            a, f = shape_args(self.schemas.get(buy) or {}, dict(self.brief.get("buy_args") or {}))
            if not f:
                if self.left() >= 2:
                    self.gate("order")
                    if self.finished or self.left() <= 0:
                        return ""
                res = self.env_call(buy, a)
                self.trace.add({"kind": "code_order", "ok": res is not None})
                if self.finished or self.left() <= 0:
                    return ""
                if again and res is not None:
                    self.checkouts += 1
                    self.refused_at = self.steps
                    self.observe(buy, a, res.get("observation"))
                    return "the order was rejected: " + self.display(res.get("observation"))
        dt = self.roles.get("refuse")
        if dt:
            a, f = shape_args(self.schemas.get(dt) or {}, self.quiet_fill(dict(self.brief.get("refuse_args") or {})))
            if not f:
                if self.left() >= 2:
                    self.gate("end")
                    if self.finished or self.left() <= 0:
                        return ""
                res = self.env_call(dt, a)
                self.trace.add({"kind": "code_decline", "ok": res is not None})
                if res is not None and not self.finished:
                    self.observe(dt, a, res.get("observation"))
                    return "the decline was rejected: " + self.display(res.get("observation"))
        return ""


class Episode(EpisodeCore, EpisodeRows, EpisodeHunts, EpisodeOpening, EpisodeInterview, EpisodeChecks, EpisodeTrails,
              EpisodeLoop):
    pass


def agent_main(problem_data):
    try:
        ep = Episode(problem_data)
    except Exception as e:
        return [{"kind": "crash", "error": clip(type(e).__name__ + ": " + str(e), 400)}]
    try:
        return list(ep.run())
    except Exception as e:
        ep.trace.add({"kind": "crash", "error": clip(type(e).__name__ + ": " + str(e), 400)})
        if not ep.finished:
            try:
                ep.finish_by_code()
            except Exception as e2:
                ep.trace.add({"kind": "crash", "error": clip(type(e2).__name__ + ": " + str(e2), 400)})
        return list(ep.trace.items)


__all__ = ["agent_main"]
