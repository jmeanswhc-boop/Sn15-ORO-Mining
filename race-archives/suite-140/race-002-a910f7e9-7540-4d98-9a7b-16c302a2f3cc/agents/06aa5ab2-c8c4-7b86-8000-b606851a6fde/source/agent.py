import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_MODELS = {
    "chutes": (
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "Qwen/Qwen3-32B-TEE",
        "zai-org/GLM-5.2-TEE",
        "moonshotai/Kimi-K2.6-TEE",
    ),
    "openrouter": (
        "deepseek/deepseek-v3.2",
        "qwen/qwen3-32b",
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
    ),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_MONEY = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_KEY = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_INCH = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:inch(?:es)?|in)\b", re.I)
_HZ = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_RAM = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*ram", re.I)
_PAIR = re.compile(r"(\d+(?:\.\d+)?)\s*g(?:b)?\s*\+\s*(\d+(?:\.\d+)?)\s*g", re.I)
_PIX = re.compile(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})")
_IP = re.compile(r"\bipx?[0-9]{1,2}\b", re.I)
_HOURS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?\b", re.I)
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)", re.I)
_RES_FLOOR = re.compile(
    r"(?:^|[,;:\s]|and\s+)([A-Za-z0-9][A-Za-z0-9]*(?:\s+HD)?(?:/[A-Za-z0-9]+)*)"
    r"\s+(?:starts\s+)?at\s+(\d{3,5})\s+(?:by|x|×)\s+(\d{3,5})",
    re.I,
)
_RES_P = re.compile(r"\b(\d{3,5}p)\b", re.I)


def _stop_budget() -> float:
    raw = getenv("SANDBOX_PROBLEM_TIMEOUT")
    try:
        total = float(raw) if raw else 300.0
    except (TypeError, ValueError):
        total = 300.0
    return max(60.0, total - 30.0)


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]{2,}", (text or "").lower())


def _rules_core(rules: str) -> str:
    return re.sub(r"(?i)\breject(?:ing)?\b.{0,200}?(?:\.|$)", " ", rules or "")


def _lead_chrome(head: str, rules: str) -> set:
    words = (head or "").split()
    blob = (rules or "").lower()
    best = 0
    for n in range(min(14, len(words)), 3, -1):
        span = " ".join(words[:n]).lower().strip(" .,;:")
        if len(span) >= 16 and span in blob:
            best = n
            break
    out = set(_words(" ".join(words[:best]))) if best else set()
    opener = set(_words(" ".join((rules or "").split()[:40])))
    for tok in _words(head):
        if tok in opener:
            out.add(tok)
            continue
        break
    return out


def _chrome(head: str, rules: str = "") -> set:
    core = _rules_core(rules)
    head_set = set(_words(head))
    rule_set = set(_words(core))
    opener = set(_words(" ".join((rules or "").split()[:40])))
    out = set(_lead_chrome(head, rules))
    out |= {t for t in opener if len(t) >= 3 and t not in head_set}
    for t in head_set | rule_set:
        if len(t) < 3:
            out.add(t)
        elif t in rule_set and t not in head_set:
            out.add(t)
    return out


def _live_caps(head: str, chrome: set, kinds: list) -> list:
    skip = set(chrome) | {k.lower() for k in kinds}
    out = []
    for m in re.finditer(r"\b([A-Z][\w&.'\-]+(?:\s+[A-Z][\w&.'\-]+)?)", head or ""):
        tok = m.group(1).strip()
        if "." in tok or tok.lower() in skip:
            continue
        if tok.isupper() and 2 <= len(tok) <= 4:
            continue
        right = (head or "")[m.end() : m.end() + 16].lower()
        if re.match(r"\s+category\b", right):
            continue
        left = (head or "")[: m.start()].rstrip()
        if left.endswith((".", "!", "?")) or not left:
            continue
        if tok.lower() not in {x.lower() for x in out}:
            out.append(tok)
    return out


def _fnum(s: Any) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _pause(step: int, cap: float, left: float | None = None) -> None:
    delay = min(2.0 * step, cap)
    if left is not None and left < delay + 8:
        return
    try:
        time.sleep(delay)
    except Exception:
        return


class _Limit:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def left(self) -> float:
        return self.budget - (time.monotonic() - self.start)


def _money(text: str, loose: bool = False) -> float | None:
    best = None
    src = text or ""
    for m in _MONEY.finditer(src):
        v = _fnum(m.group(1) + "." + (m.group(2) or "0"))
        if v is None or v < 1 or v >= 1e6:
            continue
        after = src[m.end() : m.end() + 16]
        if re.match(r"\s*[-–]?\s*in(?:ch(?:es)?|$)", after, re.I):
            continue
        if re.match(r"\s*(?:hz|gb|h(?:ours?|rs?)?|p(?:x)?)\b", after, re.I):
            continue
        dec = m.group(2) or ""
        letter_after = re.match(r"\s*([A-Za-z]{2,12})", after)
        letter_before = re.search(
            r"([A-Za-z]{2,12})\s*$", src[max(0, m.start() - 16) : m.start()]
        )
        dollar = bool(re.search(r"\$\s*$", src[: m.start()]))
        tagged = len(dec) == 2 or bool(letter_after) or bool(letter_before) or dollar
        if not tagged and not (loose and v >= 100):
            continue
        best = v if best is None else min(best, v)
    return best


def _money_src(text: str, phrase: str = "") -> str:
    src = text or ""
    if phrase:
        src = re.sub(re.escape(phrase), " ", src, flags=re.I)
    return src


def _res_chart(text: str) -> list:
    chart = []
    seen = set()
    for m in _RES_FLOOR.finditer(text or ""):
        raw = (m.group(1) or "").strip(" ,.;:")
        if not raw:
            continue
        lo, hi = sorted((int(m.group(2)), int(m.group(3))))
        aliases = []
        for part in raw.split("/"):
            name = " ".join(part.lower().split())
            if not name or name in aliases:
                continue
            aliases.append(name)
            packed = name.replace(" ", "")
            if packed not in aliases:
                aliases.append(packed)
        aliases.append("%dp" % lo)
        canon = aliases[0]
        if canon in seen:
            continue
        seen.add(canon)
        chart.append((canon, aliases, lo, hi))
    chart.sort(key=lambda row: row[2] * row[3], reverse=True)
    return chart


def _alias_hit(text: str, aliases: list) -> bool:
    t = (text or "").lower()
    for a in aliases:
        if " " in a:
            if a in t:
                return True
        elif re.search(r"\b" + re.escape(a) + r"\b", t):
            return True
    return False


def _wanted_res(text: str, chart: list) -> str | None:
    t = text or ""
    hit = None
    for canon, aliases, _lo, _hi in sorted(chart, key=lambda row: row[2] * row[3]):
        if _alias_hit(t, aliases):
            hit = canon
    if hit:
        return hit
    m = _RES_P.search(t)
    return m.group(1).lower() if m else None


def _res_labels(text: str, chart: list) -> set:
    t = (text or "").replace("*", " x ").replace("×", " x ")
    out = set()
    pairs = [tuple(sorted((int(x), int(y)))) for x, y in _PIX.findall(t)]
    for canon, aliases, lo, hi in chart:
        if _alias_hit(t, aliases):
            out.add(canon)
        elif any(a >= lo and b >= hi for a, b in pairs):
            out.add(canon)
    for m in _RES_P.finditer(t):
        out.add(m.group(1).lower())
    for lo, hi in pairs:
        out.add("%dp" % lo)
        out.add("%dp" % hi)
    return out


def _inch(text: str) -> float | None:
    m = _INCH.search(text or "")
    return float(m.group(1)) if m else None


def _size_fact(raw: str) -> tuple[float | None, bool]:
    t = (raw or "").strip()
    if not t:
        return None, False
    if _RANGE.search(t):
        return None, True
    m = _INCH.search(t) or re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        return None, False
    return float(m.group(1)), False


def _batt_hours(text: str) -> float:
    t = (text or "").lower()
    t = re.sub(r"\d+(?:\.\d+)?\s*m(?:ah|wh|hz)\b", " ", t)
    m = _HOURS.search(t)
    return float(m.group(1)) if m else 0.0


def _hours_tok(text: str) -> str:
    t = re.sub(r"\d+(?:\.\d+)?\s*m(?:ah|wh|hz)\b", " ", (text or ""), flags=re.I)
    compact = []
    spaced = []
    for m in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(h(?:ours?|rs?)?)\b", t, re.I):
        raw = m.group(0).strip()
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if val <= 0 or val > 200:
            continue
        if re.search(r"\s", raw):
            spaced.append((val, raw.lower()))
        else:
            compact.append((val, raw.lower()))
    if compact:
        return min(compact)[1]
    if spaced:
        return min(spaced)[1]
    return ""


def _cue_hit(text: str, cues) -> bool:
    t = (text or "").lower()
    for c in cues or ():
        if not c or len(c) < 3:
            continue
        stem = c.rstrip("s")
        if re.search(r"\b(?:not|non|no)\s+" + re.escape(stem), t):
            continue
        if re.search(r"\b" + re.escape(stem) + r"[a-z]{0,10}", t):
            return True
    return False


def _dead_keys(title: str) -> set:
    out = set()
    for tok in re.findall(r"[a-z0-9]{2,}", (title or "").lower()):
        if len(tok) < 3 or tok.isdigit():
            continue
        out.add(tok)
    return out


def _hz(text: str) -> float | None:
    vals = [float(x) for x in _HZ.findall(text or "")]
    return max(vals) if vals else None


def _ram_gb(text: str) -> float | None:
    m = _RAM.search(text or "")
    if m:
        return float(m.group(1))
    m = _PAIR.search(text or "")
    if m:
        return float(m.group(1))
    return None


def _wet(text: str, cues=()) -> int:
    t = (text or "").lower()
    n = 0
    if _IP.search(t):
        n += 4
    if _cue_hit(t, cues):
        n += 3
    return n


def _blob(parts: list) -> str:
    return " ".join(str(p) for p in parts if p)


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _inner(obs: Any) -> dict:
    if not isinstance(obs, dict):
        return {}
    cur = obs
    for _ in range(3):
        nxt = cur.get("observation")
        if isinstance(nxt, dict) and (
            "results" in nxt
            or "variants" in nxt
            or "product_id" in nxt
            or "cart" in nxt
            or "compared" in nxt
            or "added" in nxt
        ):
            return nxt
        if isinstance(nxt, dict):
            cur = nxt
            continue
        break
    return cur


def _os_chart(text: str, chrome: set | None = None) -> list:
    blob = _rules_core(text)
    cands = []
    for m in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9.+]*(?:\s+[A-Za-z][A-Za-z0-9.+]*)?"
        r"(?:\s*,\s*[A-Za-z][A-Za-z0-9.+]*(?:\s+[A-Za-z][A-Za-z0-9.+]*)?){2,12})",
        blob,
    ):
        raw = m.group(1)
        if raw.count(",") >= 5:
            cands.append(raw)
    out = []
    seen = set()
    for raw in sorted(cands, key=lambda s: s.count(","), reverse=True)[:1]:
        parts = re.split(r",|\bor\b", raw)
        for i, part in enumerate(parts):
            name = " ".join(part.lower().split()).strip(" .,;:")
            bits = name.split()
            if i == 0 and len(bits) >= 2:
                bits = bits[1:]
            name = " ".join(bits).strip(" .,;:")
            if not name or name in seen:
                continue
            if 2 <= len(name) <= 40 and re.search(r"[a-z]", name):
                seen.add(name)
                out.append(name)
    return out


def _reject_phrases(head: str, rules: str = "", chrome: set | None = None) -> list:
    skip = chrome or set()
    out = []
    seen = set()

    def add(raw: str) -> None:
        name = " ".join((raw or "").lower().replace("/", " ").replace("-", " ").split())
        name = " ".join(t for t in name.split() if t not in skip)
        if not name or name in seen:
            return
        if len(name.split()) > 3:
            name = name.split()[-1]
        if not name or name in seen:
            return
        seen.add(name)
        out.append(name)

    for m in re.finditer(r"\breject(?:ing)?\s+(.{4,160}?)(?:\.|$)", rules or "", re.I):
        for part in re.split(r",|\bor\b|\band\b", m.group(1)):
            add(part)
    for m in re.finditer(r"rather than (?:an? )?(.{3,60}?)(?:\.|$)", rules or "", re.I):
        for part in re.split(r",|\bor\b|\band\b", m.group(1)):
            add(part)
    low_h = (head or "").lower()
    if "not" in _words(low_h) and "listing" in _words(low_h):
        m = re.search(r"\bnot\s+\w+\s+(.{3,40}?)\s+listing", head or "", re.I)
        if m:
            for part in re.split(r",|\bor\b", m.group(1)):
                add(part)
    return out


def _junk_hit(text: str, phrases) -> bool:
    t = (text or "").lower()
    for p in phrases or ():
        if not p:
            continue
        if " " in p:
            if p in t:
                return True
        elif re.search(r"\b" + re.escape(p) + r"\b", t):
            return True
    return False


_CLAUSE = re.compile(r"(?<!\d)[.!?](?!\d)|\s+[\u2014\-]\s+|\bit['’]s\b", re.I)


def _use_span(head: str) -> str:
    h = head or ""
    stop = r"(?:[.!?]|\s+[\u2014\-]\s+|$)"
    m = re.search(r"\bit['’]s\s+for\s+(.+?)" + stop, h, re.I)
    if m:
        return m.group(1)
    m = re.search(r"\b(?:php|under|below)\b[^.]{0,48}?\bfor\s+(.+?)" + stop, h, re.I)
    if m:
        return m.group(1)
    return ""


def _facts_axis(rules: str) -> list:
    m = re.search(r"facts such as ([^.]+)", rules or "", re.I)
    if not m:
        return []
    raw = re.split(r",\s*then\b", m.group(1) or "", maxsplit=1, flags=re.I)[0]
    out = []
    for part in re.split(r",|\band\b", raw):
        bits = _words((part or "").replace("/", " ").replace("-", " "))
        if not bits or len(bits) > 4:
            continue
        for tok in bits:
            if len(tok) >= 4 and tok not in out:
                out.append(tok)
    return out


def _first_clause(head: str) -> str:
    return _CLAUSE.split(head or "", maxsplit=1)[0]


def _rest_clause(head: str) -> str:
    parts = _CLAUSE.split(head or "", maxsplit=1)
    return parts[1] if len(parts) > 1 else ""


def _live_kinds(head: str, rules: str, chrome: set) -> list:
    out = []
    seen = set()

    def add(tok: str) -> None:
        t = (tok or "").lower().strip(" .,")
        if len(t) < 3 or t in seen or t in chrome or t.isdigit():
            return
        seen.add(t)
        out.append(t)

    if "category" in (head or "").lower():
        cm = re.search(r"([A-Za-z][\w&'\- ]{1,60}?)\s+category", head or "", re.I)
        if cm:
            for t in cm.group(1).split():
                add(t)
    if re.search(r"inch|\bhd\b", head or "", re.I):
        for m in re.finditer(r"(?:inch(?:es)?|hd)\s+([a-z]+(?:\s+[a-z]+)?)", head or "", re.I):
            for t in m.group(1).lower().split():
                add(t)
    clause = _first_clause(head)
    for m in re.finditer(r"\b(?:a|an|the)\s+((?:[\w\-]+\s+){0,1}[A-Za-z]{3,})", clause, re.I):
        for t in _words(m.group(1)):
            add(t)
    if not out:
        for tok in _words(clause):
            if tok not in chrome:
                add(tok)
            if len(out) >= 4:
                break
    return out


def _has_os(text: str, names) -> bool:
    t = (text or "").lower()
    return any(tok and tok in t for tok in names or ())


def _os_asked(text: str, names) -> list:
    blob = (text or "").lower()
    out = []
    for name in names or ():
        if not name:
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", blob) and name not in out:
            out.append(name)
    return out


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "").strip().lower() or "chutes"


def _sku_stem(sku: str) -> str:
    if not sku:
        return ""
    return sku.rsplit("-", 1)[0]


class Wish:
    def __init__(self, query: str) -> None:
        raw = query or ""
        head, _, rules = raw.partition("\n\nTask rules:")
        if not rules:
            head, _, rules = raw.partition("Task rules:")
        self.head = head.strip()
        self.rules = rules
        self.chrome = _chrome(self.head, self.rules)
        self.budget_cut = False
        self.late_firm = False
        self.need_page = False
        self.phrase = ""
        qm = re.search(r'"([^"]{2,80})"', self.head)
        if qm:
            self.phrase = qm.group(1).strip()
        else:
            pm = re.search(r"matches(?: the phrase)?\s+([A-Za-z0-9][\w\-/]*(?:\s+[A-Za-z0-9][\w\-/]*)?)", self.head, re.I)
            if pm:
                self.phrase = pm.group(1).strip(" .,")
        src = _money_src(self.head, self.phrase)
        self.budget = _money(src) or _money(_money_src(raw, self.phrase)) or _money(src, loose=True)
        self.aisle = None
        self.aisle_live = False
        if "category" in self.head.lower():
            cm = re.search(r"in the ([A-Za-z][\w&'\-][\w&'\- ]{0,40}?) category", self.head, re.I)
            if cm:
                self.aisle = cm.group(1).strip()
                self.aisle_live = True
        self.os_names = _os_chart(raw, self.chrome)
        self.os_need = _os_asked(self.head, self.os_names)
        self.reject_words = _reject_phrases(self.head, self.rules, self.chrome)
        self.seeds = [(a, b) for a, b in _KEY.findall(self.head)]
        self.rank = bool(self.seeds and "refresh" in raw.lower())
        self.inch = _inch(self.head)
        self.res_chart = _res_chart(raw)
        self.res = _wanted_res(self.head, self.res_chart)
        self.res_marks = []
        if self.res:
            self.res_marks.append(self.res)
        for m in _RES_P.finditer(self.head):
            tok = m.group(1).lower()
            if tok not in self.res_marks:
                self.res_marks.append(tok)
        low = self.head.lower()
        for phrase in re.findall(r"\b([a-z]+\s+hd)\b", low):
            if phrase not in self.res_marks:
                self.res_marks.append(phrase)
        if re.search(r"\bhd\b", low) and "hd" not in self.res_marks:
            self.res_marks.append("hd")
        self.hz_need = _hz(self.head)
        self.ram = _ram_gb(self.head)
        low = self.head.lower()
        self.kinds = _live_kinds(self.head, self.rules, self.chrome)
        self.phone = bool(self.os_names)
        self.use_cues = []
        use = _use_span(self.head)
        if use:
            for tok in _words(use):
                if tok in self.chrome or tok in self.kinds or tok.isdigit():
                    continue
                if tok not in self.use_cues:
                    self.use_cues.append(tok)
        head_set = set(_words(self.head))
        self.axis_cues = [t for t in _facts_axis(self.rules) if t in head_set and t not in self.kinds]
        self.rest_cues = list(self.use_cues)
        for tok in self.axis_cues:
            if tok not in self.rest_cues:
                self.rest_cues.append(tok)
        if not self.rest_cues:
            rest = _rest_clause(self.head)
            if rest:
                for tok in _words(rest):
                    if tok in self.chrome or tok in self.kinds or tok.isdigit():
                        continue
                    if tok not in self.rest_cues:
                        self.rest_cues.append(tok)
        if not self.rest_cues and self.kinds:
            kind = max(self.kinds, key=len)
            low_h = self.head.lower()
            idx = low_h.find(kind)
            tail = self.head[idx + len(kind) :] if idx >= 0 else ""
            for tok in _words(tail):
                if tok in self.chrome or tok in self.kinds or tok.isdigit():
                    continue
                if tok not in self.rest_cues:
                    self.rest_cues.append(tok)
        self.wet_cues = list(self.rest_cues)
        self.wire_cues = list(self.axis_cues)
        self.soft_brand = None
        self.hard_brand = None
        caps = _live_caps(self.head, self.chrome, [])
        if any(t.startswith("prefer") for t in _words(self.head)):
            if caps:
                self.soft_brand = caps[0]
        elif caps:
            self.hard_brand = caps[0]
        if self.aisle and self.hard_brand and self.hard_brand.lower() in self.aisle.lower():
            self.hard_brand = None
        if self.aisle and self.soft_brand and self.soft_brand.lower() in self.aisle.lower():
            self.soft_brand = None
        self.color = None
        self.model = None
        if any(t.startswith("prefer") for t in _words(self.head)):
            rm = re.search(
                r"prefer\w*\s+(?:the\s+)?([A-Za-z]{3,16})\s+(\d{3,12})\b",
                self.head,
                re.I,
            )
            if rm:
                self.color = rm.group(1).lower()
                self.model = rm.group(2)
        self.ask = bool(re.search(r"\bask\b", low))
        self.why = bool(re.search(r"\bwhy\b", low))
        firm = (
            self.inch is not None
            or bool(self.phrase)
            or bool(self.seeds)
            or bool(self.res)
            or self.hz_need is not None
            or self.ram is not None
            or bool(self.hard_brand)
            or bool(self.model)
        )
        self.wet = (not firm) and (len(self.use_cues) >= 2 or len(self.axis_cues) >= 2 or len(self.rest_cues) >= 3)
        self.reject_words = [
            p
            for p in self.reject_words
            if p and p not in self.kinds and p not in {k.rstrip("s") for k in self.kinds}
        ]
        self.pref = self.wet
        cm2 = re.search(r"surface at least (\d+) distinct", self.rules, re.I)
        self.cover = int(cm2.group(1)) if cm2 else 0

    def _mark_firm(self) -> None:
        self.late_firm = True
        self.need_page = True

    def absorb(self, text: str) -> None:
        t = text or ""
        found = ""
        if not self.phrase:
            qm = re.search(r'"([^"]{2,80})"', t)
            if qm:
                found = qm.group(1).strip()
            else:
                pm = re.search(r"matches(?: the phrase)?\s+([A-Za-z0-9][\w\-/]*(?:\s+[A-Za-z0-9][\w\-/]*)?)", t, re.I)
                if pm:
                    found = pm.group(1).strip(" .,")
        src = _money_src(t, found or self.phrase)
        b = _money(src) or _money(src, loose=True)
        if b is not None:
            if self.budget is None:
                self._mark_firm()
            elif b + 1e-6 < self.budget:
                self.budget_cut = True
                self.need_page = True
            self.budget = b
        if re.search(r"\bwhy\b", t, re.I):
            self.why = True
        if "category" in t.lower():
            am = re.search(r"in the ([A-Za-z][\w&'\-][\w&'\- ]{0,40}?) category", t, re.I)
            if not am:
                am = re.search(r"category(?:\s+is)?\s+([A-Za-z][\w&'\-]{2,40})", t, re.I)
            if am:
                if not self.aisle:
                    self._mark_firm()
                self.aisle = am.group(1).strip()
                self.aisle_live = True
        extra_os = _os_chart(t, self.chrome)
        for name in extra_os:
            if name not in self.os_names:
                self.os_names.append(name)
        for name in _os_asked(t, self.os_names):
            if name not in self.os_need:
                if not self.os_need:
                    self._mark_firm()
                self.os_need.append(name)
        for phrase in _reject_phrases(t, t, self.chrome):
            if phrase not in self.reject_words:
                self.reject_words.append(phrase)
        low = t.lower()
        extra_bits = []
        extra_use = _use_span(t)
        if extra_use:
            extra_bits.extend(_words(extra_use))
        live = set(_words(t))
        for tok in _facts_axis(self.rules):
            if tok in live and tok not in extra_bits:
                extra_bits.append(tok)
        if extra_bits:
            for tok in extra_bits:
                if tok in self.chrome or tok.isdigit():
                    continue
                if tok not in self.rest_cues:
                    self.rest_cues.append(tok)
                if tok not in self.wet_cues:
                    self.wet_cues.append(tok)
                if tok not in self.use_cues and extra_use and tok in set(_words(extra_use)):
                    self.use_cues.append(tok)
                if tok not in self.axis_cues and tok in set(_facts_axis(self.rules)):
                    self.axis_cues.append(tok)
                if tok not in self.wire_cues:
                    self.wire_cues.append(tok)
        extra_kinds = _live_kinds(t, self.rules, self.chrome)
        if extra_kinds and (
            "category" in t.lower()
            or re.search(r"inch|\bhd\b|\b(?:a|an|the)\s+", t, re.I)
        ):
            for tok in extra_kinds:
                if tok not in self.kinds:
                    self.kinds.append(tok)
        if found:
            self.phrase = found
            self._mark_firm()
            self.ask = False
        if any(w.startswith("prefer") for w in _words(t)):
            rm = re.search(
                r"prefer\w*\s+(?:the\s+)?([A-Za-z]{3,16})\s+(\d{3,12})\b",
                t,
                re.I,
            )
            if rm:
                if not self.color:
                    self.color = rm.group(1).lower()
                if not self.model:
                    self.model = rm.group(2)
                    self._mark_firm()
                    self.ask = False
        inch = _inch(t)
        if inch is not None:
            if self.inch is None:
                self._mark_firm()
            self.inch = inch
            self.ask = False
        extra = _res_chart(t)
        if extra:
            have = {row[0] for row in self.res_chart}
            for row in extra:
                if row[0] not in have:
                    self.res_chart.append(row)
                    have.add(row[0])
            self.res_chart.sort(key=lambda row: row[2] * row[3], reverse=True)
        res = _wanted_res(t, self.res_chart)
        if res:
            if not self.res:
                self._mark_firm()
            self.res = res
            if res not in self.res_marks:
                self.res_marks.append(res)
            self.ask = False
        for m in _RES_P.finditer(t):
            tok = m.group(1).lower()
            if tok not in self.res_marks:
                self.res_marks.append(tok)
        hz = _hz(t)
        if hz is not None:
            if self.hz_need is None:
                self._mark_firm()
            self.hz_need = hz
            self.ask = False
        ram = _ram_gb(t)
        if ram is not None:
            if self.ram is None:
                self._mark_firm()
            self.ram = ram
            self.ask = False
        if inch or res or hz or ram:
            self.ask = False
        self._take_named(t)
        still_open = not (
            self.inch is not None
            or self.phrase
            or self.seeds
            or self.res
            or self.hz_need is not None
            or self.ram is not None
            or self.os_names
            or self.hard_brand
            or self.model
        )
        if still_open and len(self.rest_cues) >= 3:
            self.wet = True
            self.pref = True
        elif self.late_firm and (
            self.hard_brand
            or self.model
            or self.inch is not None
            or self.res
            or self.hz_need is not None
            or self.ram is not None
        ):
            self.wet = False
            self.pref = False

    def _take_named(self, text: str) -> None:
        t = text or ""
        skip = set(self.chrome) | {k.lower() for k in self.kinds}
        framed = None
        fm = re.search(
            r"\b(?:required\s+)?brand(?:\s+is)?\s+([A-Za-z][\w&'\-]{1,24})\b",
            t,
            re.I,
        )
        if fm:
            cand = fm.group(1).strip()
            if cand.lower() not in skip:
                framed = cand
        caps = _live_caps(t, self.chrome, self.kinds)
        prefer = any(w.startswith("prefer") for w in _words(t))
        pick = framed or (caps[0] if caps else None)
        if (not pick) and self.ask and not (self.inch or self.res or self.hz_need is not None or self.ram is not None):
            bare = re.match(
                r"^\s*([A-Z][\w&'\-]+(?:\s+[A-Z][\w&'\-]+)?)\s*[.!]?\s*$",
                t,
            )
            if bare:
                tok = bare.group(1).strip()
                if tok.lower() not in skip and not tok.isdigit():
                    pick = tok
        if not pick:
            return
        if self.aisle and pick.lower() in self.aisle.lower():
            return
        if prefer:
            if not self.soft_brand and not self.hard_brand:
                self.soft_brand = pick
                self._mark_firm()
            return
        if not self.hard_brand:
            self.hard_brand = pick
            self._mark_firm()
            self.ask = False

    def take_aisle(self, item: dict) -> None:
        if self.aisle:
            return
        path = (item or {}).get("category_path") or []
        if isinstance(path, list):
            parts = [str(x).strip() for x in path if str(x).strip()]
        else:
            parts = [str(path).strip()] if path else []
        if not parts:
            return
        stems = []
        for tok in self.kinds:
            stem = tok.rstrip("s")
            if stem not in stems:
                stems.append(stem)
        if not stems:
            return
        for part in reversed(parts):
            low = part.lower()
            if any(s in low for s in stems):
                self.aisle = part
                return


class Shelf:
    def __init__(self) -> None:
        self.row: dict = {}
        self.page: dict = {}
        self.tick: dict = {}
        self.cart: list = []
        self.ban: set = set()
        self.seen: set = set()
        self.viewed: set = set()
        self.hits = 0
        self.search_n = 0
        self.sent_q: list = []
        self.found: list = []
        self.asked = False
        self.inbox: list = []
        self.skip_order = False
        self.notes: list = []
        self.confirm_n = 0
        self.final_look = False
        self.event = False
        self.first_add = False
        self.rechecked = False
        self.dead_stem = ""
        self.dead_title = ""
        self.dead_keys: set = set()
        self.need_rescan = False
        self.compares = 0
        self.trusted: set = set()
        self.named = None

    def put_row(self, d: dict, trusted: bool = False) -> None:
        pid = str(d.get("product_id") or "")
        sku = str(d.get("sku") or "")
        if not pid:
            return
        key = (pid, sku)
        cur = self.row.get(key) or {"product_id": pid, "sku": sku}
        for k in ("title", "brand", "price", "in_stock", "currency", "options"):
            if d.get(k) is not None:
                cur[k] = d[k]
        self.row[key] = cur
        self.seen.add(pid)
        if trusted:
            self.trusted.add(key)

    def put_page(self, d: dict) -> None:
        pid = str(d.get("product_id") or "")
        if not pid:
            return
        self.page[pid] = d
        self.viewed.add(pid)
        title = d.get("title")
        brand = d.get("brand")
        for var in d.get("variants") or []:
            if not isinstance(var, dict):
                continue
            sku = str(var.get("sku") or "")
            blob = {
                "product_id": pid,
                "sku": sku,
                "title": title,
                "brand": brand,
                "price": var.get("price"),
                "in_stock": var.get("in_stock"),
                "options": var.get("options"),
            }
            self.put_row(blob, trusted=True)

    def live(self, pid: str, sku: str) -> dict:
        key = (str(pid), str(sku))
        out = dict(self.row.get(key) or {"product_id": pid, "sku": sku})
        tick = self.tick.get(key)
        if tick:
            out["price"] = tick.get("price", out.get("price"))
            out["in_stock"] = tick.get("in_stock", out.get("in_stock"))
        page = self.page.get(str(pid)) or {}
        out["title"] = out.get("title") or page.get("title")
        out["brand"] = out.get("brand") or page.get("brand")
        out["facts"] = page.get("facts") if isinstance(page.get("facts"), dict) else {}
        out["category_path"] = page.get("category_path") or []
        return out


class Loop:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.bind = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = {(t.get("function") or {}).get("name") for t in (self.policy.get("tools") or []) if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.pid = str(self.bind.get("session_id") or problem.get("problem_id") or problem.get("id") or "p")
        self.wish = Wish(str(self.policy.get("query") or ""))
        self.shelf = Shelf()
        self.turn = 0
        self.wave = 0
        self.hold: dict | None = None
        self.hold_n = 0
        self.limit = _Limit(_stop_budget())
        self.talk: list = []
        self.models = list(_MODELS.get(_provider(), _MODELS["chutes"]))
        pin = getenv("SANDBOX_MODEL")
        if pin:
            self.models = [pin] + [m for m in self.models if m != pin]
        for a, b in self.wish.seeds:
            self.shelf.put_row({"product_id": a, "sku": b, "in_stock": True}, trusted=True)

    def ok(self, name: str) -> bool:
        return name in self.tools

    def _post(self, path: str, payload: dict) -> dict | None:
        try:
            got = _PROXY.post(path, json_data=payload)
        except Exception:
            return None
        return got if isinstance(got, dict) else None

    def infer(self, messages: list, tools: list | None = None) -> dict | None:
        body = {"messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "required"
        for i, model in enumerate(self.models):
            tries = 4 if i == 0 else 2
            for attempt in range(tries):
                left = self.limit.left()
                if left < 25:
                    return None
                payload = dict(body)
                payload["model"] = model
                got = self._post("/inference/chat/completions", payload)
                if got and got.get("choices"):
                    return got["choices"][0].get("message") or {}
                if left > 30:
                    _pause(attempt + 1, 5.0, left)
        return None

    def _envelope(self, batch: list, turn: int, wave: int) -> dict:
        group = "%s-t%s" % (self.pid, turn)
        if wave:
            group = "%s-w%s" % (group, wave)
        return {
            **self.bind,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [
                {"call_id": "%s-%s" % (group, i), "action": {"name": n, "args": a}}
                for i, (n, a) in enumerate(batch, start=1)
            ],
        }

    def _has_write(self, batch: list) -> bool:
        return any(re.search(r"cart|order", n or "") for n, _a in batch)

    def _env_send(self, envelope: dict, must: bool = False) -> tuple:
        if self.limit.left() < 8 and not must:
            return None, None
        try:
            send = getattr(_PROXY, "post_verbose", None)
            if callable(send):
                got = send("/environment/call", json_data=envelope)
                data = getattr(got, "data", None) if got is not None else None
                err = getattr(got, "error", None) if got is not None else None
                if isinstance(data, dict) and isinstance(data.get("calls"), list) and data.get("calls"):
                    return data, 200
                status = err.get("status") if isinstance(err, dict) else None
                return None, status
            data = self._post("/environment/call", envelope)
            if isinstance(data, dict) and isinstance(data.get("calls"), list) and data.get("calls"):
                return data, 200
            return None, None
        except Exception:
            return None, None

    def _accept(self, batch: list, result: dict) -> tuple:
        self.hold = None
        self.hold_n = 0
        self.talk.append({"role": "assistant", "batch": batch, "environment_result": result})
        um = result.get("user_message")
        if isinstance(um, dict) and um.get("content"):
            try:
                self.wish.absorb(str(um["content"]))
                self.shelf.inbox.append(str(um["content"]))
                note = str(um["content"]).strip()
                if note:
                    self.shelf.notes.append(note)
            except Exception:
                pass
        done = False
        outs = []
        for call, rec in zip(batch, result.get("calls") or []):
            obs = rec.get("observation") if isinstance(rec, dict) else {}
            if isinstance(obs, dict) and obs.get("done"):
                done = True
            try:
                self.ingest(call[0], call[1], obs)
            except Exception as exc:
                self.talk.append(
                    {"role": "assistant", "content": "ingest: %s: %s" % (type(exc).__name__, str(exc)[:120])}
                )
            outs.append(obs)
        return outs, done

    def _assume_commit(self, batch: list) -> bool:
        done = False
        for name, args in batch:
            if name == "place_test_order":
                done = True
            elif name == "search":
                self.shelf.search_n += 1
                self.shelf.hits += 1
                q = str((args or {}).get("query") or "").strip()
                if q:
                    self.shelf.sent_q.append(q)
            elif name == "filter":
                self.shelf.hits += 1
            elif name == "add_to_cart":
                pid = str((args or {}).get("product_id") or "")
                sku = str((args or {}).get("sku") or "")
                if pid and sku:
                    if not self.shelf.cart:
                        self.shelf.cart = [{"product_id": pid, "sku": sku}]
                    self.shelf.first_add = True
            elif name == "remove_from_cart":
                pid = str((args or {}).get("product_id") or "")
                sku = str((args or {}).get("sku") or "")
                if pid and sku:
                    self.shelf.ban.add((pid, sku))
                    self.shelf.cart = [
                        c
                        for c in self.shelf.cart
                        if not (str(c.get("product_id")) == pid and str(c.get("sku")) == sku)
                    ]
            elif name == "compare":
                self.shelf.compares += 1
            elif name == "message":
                self.shelf.asked = True
            elif name == "view":
                pid = str((args or {}).get("product_id") or "")
                if pid:
                    self.shelf.viewed.add(pid)
            elif name == "inspect_stock":
                pid = str((args or {}).get("product_id") or "")
                sku = str((args or {}).get("sku") or "")
                if pid and sku and (pid, sku) not in self.shelf.tick:
                    self.shelf.tick[(pid, sku)] = {}
            elif name == "inspect_cart":
                self.shelf.rechecked = True
        self.hold = None
        self.hold_n = 0
        self.talk.append({"role": "assistant", "content": "env committed", "batch": batch})
        return done

    def fire(self, batch: list) -> tuple:
        from_hold = bool(self.hold)
        if from_hold:
            batch = list(self.hold.get("batch") or [])
            if not batch:
                self.hold = None
                return [], False
            if self.hold.get("same") and isinstance(self.hold.get("envelope"), dict):
                envelope = self.hold["envelope"]
            else:
                self.wave += 1
                envelope = self._envelope(batch, self.turn, self.wave)
        else:
            if not batch:
                return [], False
            self.turn += 1
            self.wave = 0
            envelope = self._envelope(batch, self.turn, 0)
        must = any(n == "place_test_order" for n, _a in batch)
        result, status = self._env_send(envelope, must=must)
        if isinstance(result, dict):
            return self._accept(batch, result)
        if status == 409:
            return [], self._assume_commit(batch)
        if (not from_hold) and (not self._has_write(batch)) and self.limit.left() >= 20:
            self.wave += 1
            envelope = self._envelope(batch, self.turn, self.wave)
            result, status = self._env_send(envelope, must=must)
            if isinstance(result, dict):
                return self._accept(batch, result)
            if status == 409:
                return [], self._assume_commit(batch)
            self.hold = {"batch": batch, "envelope": envelope, "same": True}
            self.talk.append({"role": "assistant", "content": "env miss", "batch": batch})
            return [], False
        self.hold = {"batch": batch, "envelope": envelope, "same": True}
        self.talk.append({"role": "assistant", "content": "env miss", "batch": batch})
        return [], False

    def ingest(self, name: str, args: dict, obs: Any) -> None:
        if isinstance(obs, dict) and obs.get("error"):
            return
        inner = _inner(obs)
        if name == "search":
            self.shelf.hits += 1
            self.shelf.search_n += 1
            q = str(args.get("query") or "").strip()
            if q:
                self.shelf.sent_q.append(q)
            page = []
            for row in inner.get("results") or []:
                if isinstance(row, dict):
                    self.shelf.put_row(row, trusted=True)
                    page.append(
                        (
                            str(row.get("product_id") or ""),
                            str(row.get("sku") or ""),
                            str(row.get("title") or ""),
                            str(row.get("brand") or ""),
                        )
                    )
            if self.shelf.search_n == 1:
                self.shelf.found = page
        elif name == "filter":
            self.shelf.hits += 1
            trust = bool(
                args.get("brand")
                or args.get("category")
                or args.get("max_price") is not None
                or self.wish.phrase
                or self.wish.model
                or self.wish.wet
            )
            for row in inner.get("results") or []:
                if not isinstance(row, dict):
                    continue
                self.shelf.put_row(row, trusted=trust)
        elif name == "view":
            self.shelf.put_page(inner)
            self.wish.take_aisle(inner)
        elif name == "compare":
            self.shelf.compares += 1
            for block in inner.get("compared") or []:
                if isinstance(block, dict):
                    self.shelf.put_page(block)
                    self.wish.take_aisle(block)
        elif name == "inspect_stock":
            pid = str(args.get("product_id") or inner.get("product_id") or "")
            sku = str(args.get("sku") or inner.get("sku") or "")
            if pid and sku:
                self.shelf.tick[(pid, sku)] = {
                    "price": inner.get("price"),
                    "in_stock": inner.get("in_stock"),
                }
        elif name == "add_to_cart":
            added = inner.get("added") if isinstance(inner.get("added"), dict) else {}
            cart = inner.get("cart") if isinstance(inner.get("cart"), list) else []
            self.shelf.cart = cart
            self.shelf.first_add = True
            self.shelf.rechecked = False
            ev = inner.get("market_event")
            if isinstance(ev, dict):
                self.shelf.event = True
                self.shelf.need_rescan = True
                tgt = ev.get("target") if isinstance(ev.get("target"), dict) else {}
                pid = str(tgt.get("product_id") or added.get("product_id") or args.get("product_id") or "")
                sku = str(tgt.get("sku") or added.get("sku") or args.get("sku") or "")
                if pid and sku:
                    self.shelf.dead_stem = _sku_stem(sku)
                    title = str((self.shelf.live(pid, sku) or {}).get("title") or "")
                    self.shelf.dead_title = " ".join(title.split()[:8])
                    self.shelf.dead_keys = _dead_keys(title)
                    self.shelf.tick[(pid, sku)] = {
                        "price": ev.get("new_price", added.get("price")),
                        "in_stock": ev.get("in_stock_after", added.get("in_stock")),
                    }
            elif added:
                pid = str(added.get("product_id") or args.get("product_id") or "")
                sku = str(added.get("sku") or args.get("sku") or "")
                if pid and sku:
                    self.shelf.tick[(pid, sku)] = {
                        "price": added.get("price"),
                        "in_stock": added.get("in_stock"),
                    }
        elif name == "inspect_cart":
            cart = inner.get("cart")
            if isinstance(cart, list):
                self.shelf.cart = cart
            elif isinstance(inner.get("items"), list):
                self.shelf.cart = inner["items"]
            self.shelf.rechecked = True
        elif name == "remove_from_cart":
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and sku:
                self.shelf.ban.add((pid, sku))
                if self.shelf.named == (pid, sku):
                    self.shelf.named = None
                self.shelf.cart = [
                    c
                    for c in self.shelf.cart
                    if not (str(c.get("product_id")) == pid and str(c.get("sku")) == sku)
                ]
        elif name == "message":
            self.shelf.asked = True
        for d in _walk(inner):
            if d.get("product_id") and (d.get("sku") or d.get("title") or d.get("variants")):
                if "variants" in d:
                    self.shelf.put_page(d)
                elif d.get("sku") and name in ("search", "view", "compare", "inspect_stock"):
                    self.shelf.put_row(d, trusted=True)

    def text_of(self, pid: str, sku: str) -> str:
        item = self.shelf.live(pid, sku)
        facts = item.get("facts") or {}
        return _blob(
            [
                item.get("title"),
                item.get("brand"),
                item.get("options"),
                " ".join("%s %s" % (k, v) for k, v in facts.items()),
                " ".join(map(str, item.get("category_path") or [])),
            ]
        )

    def opt_blob(self, pid: str, sku: str) -> str:
        item = self.shelf.live(pid, sku)
        return _blob([item.get("options"), item.get("title")])

    def res_classes(self, pid: str, sku: str) -> set:
        item = self.shelf.live(pid, sku)
        facts = item.get("facts") or {}
        fact_disp = " ".join(str(facts.get(k) or "") for k in ("display", "resolution"))
        fact_disp = fact_disp.replace("*", " x ").replace("×", " x ")
        opts = str(item.get("options") or "")
        chart = self.wish.res_chart
        if fact_disp.strip() and pid in self.shelf.viewed:
            named = _res_labels(fact_disp, chart)
            if named:
                return named
            return _res_labels(fact_disp + " " + opts, chart)
        return _res_labels(opts + " " + str(item.get("title") or "") + " " + self.text_of(pid, sku), chart)

    def inch_of(self, pid: str, sku: str) -> float | None:
        item = self.shelf.live(pid, sku)
        facts = item.get("facts") or {}
        from_opt = _inch(str(item.get("options") or "")) or _inch(str(item.get("title") or ""))
        if self.wish.rank:
            return from_opt or _inch(str(facts.get("size") or ""))
        val, is_range = _size_fact(str(facts.get("size") or ""))
        if is_range:
            return None
        if val is not None:
            return val
        return from_opt

    def hz_of(self, pid: str, sku: str) -> float | None:
        item = self.shelf.live(pid, sku)
        facts = item.get("facts") or {}
        return _hz(self.opt_blob(pid, sku)) or _hz(str(facts.get("refresh_rate") or facts.get("display") or ""))

    def identity_ok(self, pid: str, sku: str) -> bool:
        item = self.shelf.live(pid, sku)
        text = self.text_of(pid, sku).lower()
        title = str(item.get("title") or "").lower()
        facts = item.get("facts") or {}
        if self.wish.phrase:
            return True
        if not self.wish.kinds:
            if self.wish.model and self.wish.model.lower() in text:
                return True
            return False
        if _junk_hit(text, self.wish.reject_words) or _junk_hit(str(item.get("options") or ""), self.wish.reject_words):
            return False
        named = (self.wish.hard_brand or "").lower()
        if named:
            brand = str(item.get("brand") or "").lower()
            if named not in brand and named not in title and named not in text:
                return False
        stems = []
        kind = (self.shop_kind() or "").rstrip("s")
        if kind:
            stems.append(kind)
        if self.wish.aisle:
            for tok in _words(self.wish.aisle):
                stem = tok.rstrip("s")
                if stem and stem not in stems:
                    stems.append(stem)
        hit = any(s and s in text for s in stems)
        if self.wish.os_names:
            return (
                hit
                or _has_os(text, self.wish.os_names)
                or any(s and s in str(facts.get("phone_type") or "").lower() for s in stems)
            )
        if self.wish.seeds:
            return True
        if hit:
            return True
        return False

    def spec_ok(self, pid: str, sku: str) -> bool:
        item = self.shelf.live(pid, sku)
        text = self.text_of(pid, sku)
        if self.wish.inch is not None:
            facts = item.get("facts") or {}
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if pid in self.shelf.viewed and str(facts.get("size") or "").strip():
                if is_range or val is None or round(val, 2) != round(self.wish.inch, 2):
                    return False
            else:
                sizes = [
                    _inch(str(item.get("options") or "")),
                    val,
                    _inch(str(item.get("title") or "")),
                ]
                if not any(s is not None and round(s, 2) == round(self.wish.inch, 2) for s in sizes):
                    return False
        if self.wish.res:
            labels = self.res_classes(pid, sku)
            marks = self.wish.res_marks or [self.wish.res]
            if not any(m in labels or _alias_hit(text, [m]) for m in marks):
                return False
        if self.wish.hz_need is not None:
            got = self.hz_of(pid, sku)
            if got is None or got + 1e-6 < self.wish.hz_need:
                return False
        if self.wish.ram is not None:
            got = _ram_gb(str(item.get("options") or "") + " " + text)
            if got is None or abs(got - self.wish.ram) > 0.2:
                return False
        if self.wish.hard_brand and not self.shows_brand(pid, sku):
            return False
        return True

    def aisle_ok(self, pid: str, sku: str) -> bool:
        if not self.wish.aisle:
            return True
        item = self.shelf.live(pid, sku)
        path = item.get("category_path") or []
        if isinstance(path, list):
            parts = [str(x) for x in path]
        else:
            parts = [str(path)] if path else []
        blob = " ".join(parts + [str(item.get("title") or ""), self.text_of(pid, sku)]).lower()
        toks = [t for t in _words(self.wish.aisle) if len(t) >= 3 and t not in self.wish.chrome]
        if not toks:
            return True
        for t in toks:
            stem = t.rstrip("s")
            if t in blob or (stem and stem in blob):
                return True
        return False

    def budget_ok(self, pid: str, sku: str) -> bool:
        item = self.shelf.live(pid, sku)
        price = _fnum(item.get("price"))
        if self.wish.budget is None or price is None:
            return price is not None
        return price <= self.wish.budget + 1e-6

    def stock_ok(self, pid: str, sku: str) -> bool:
        item = self.shelf.live(pid, sku)
        return bool(item.get("in_stock"))

    def hard_ok(self, pid: str, sku: str) -> bool:
        if (pid, sku) in self.shelf.ban:
            return False
        if not sku:
            return False
        if self.wish.rank:
            if (pid, sku) not in self.wish.seeds:
                return False
            if self.hz_of(pid, sku) is None or self.inch_of(pid, sku) is None:
                return False
            return self.stock_ok(pid, sku) and self.budget_ok(pid, sku)
        if self.wish.model and self.shelf.dead_stem and _sku_stem(sku) == self.shelf.dead_stem:
            return self.stock_ok(pid, sku) and self.budget_ok(pid, sku)
        if (
            self.wish.hard_brand
            and self.shelf.dead_stem
            and _sku_stem(sku) == self.shelf.dead_stem
            and pid in self.shelf.viewed
        ):
            return self.stock_ok(pid, sku) and self.budget_ok(pid, sku) and self.identity_ok(pid, sku)
        if pid not in self.shelf.viewed and not self.wish.phrase:
            if self.wish.kinds or self.wish.why or self.wish.model or self.wish.wet:
                return False
        if self.wish.phrase and not self.phrase_hit(pid, sku):
            return False
        return self.stock_ok(pid, sku) and self.budget_ok(pid, sku) and self.spec_ok(pid, sku) and self.identity_ok(pid, sku)

    def phrase_hit(self, pid: str, sku: str) -> bool:
        if not self.wish.phrase:
            return True
        title = (self.shelf.live(pid, sku).get("title") or "").lower()
        have = set(re.findall(r"[a-z0-9]+", title))
        toks = [t for t in re.findall(r"[a-z0-9]+", self.wish.phrase.lower()) if t]
        return all(t in have for t in toks) if toks else True

    def phrase_toks(self) -> list:
        return [t for t in re.findall(r"[a-z0-9]+", (self.wish.phrase or "").lower()) if t]

    def phrase_in_query(self, q: str) -> bool:
        want = self.phrase_toks()
        if not want:
            return False
        have = set(re.findall(r"[a-z0-9]+", str(q).lower()))
        return all(t in have for t in want)

    def phrase_reword(self) -> str:
        want = self.phrase_toks()
        if len(want) < 2:
            return ""
        if not any(re.search(r"\d", t) for t in want):
            return ""
        alt = " ".join(reversed(want))
        if alt.lower() == (self.wish.phrase or "").strip().lower():
            return ""
        return alt

    def cover_count(self) -> int:
        ids = set()
        for pid, sku in self.shelf.row:
            if self.phrase_hit(pid, sku) and self.stock_ok(pid, sku) and self.budget_ok(pid, sku):
                ids.add(pid)
        return len(ids)

    def relax(self, pid: str, sku: str) -> int:
        if not self.wish.model:
            return 0
        text = self.text_of(pid, sku).lower()
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        same_model = bool(
            re.search(r"(?<!\d)" + re.escape(self.wish.model) + r"(?!\d)", text)
            or squash(self.wish.model) in squash(text)
        )
        if self.shelf.dead_stem and _sku_stem(sku) == self.shelf.dead_stem:
            same_model = True
        shade = self.opt_blob(pid, sku).lower() + " " + str((self.shelf.live(pid, sku).get("facts") or {}).get("color") or "")
        same_color = bool(self.wish.color) and re.search(
            r"(?<![a-z])" + re.escape(self.wish.color) + r"(?![a-z])", shade + " " + text
        )
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def candidates(self) -> list:
        out = []
        keys = list(self.shelf.row)
        if self.wish.seeds:
            keys = list(self.wish.seeds) + [k for k in keys if k not in self.wish.seeds]
        for pid, sku in keys:
            if not self.hard_ok(pid, sku):
                continue
            if self.wish.phrase and not self.phrase_hit(pid, sku):
                continue
            if not self.wish.seeds and (pid, sku) not in self.shelf.trusted and not self.wish.phrase:
                if self.wish.kinds:
                    continue
            out.append((pid, sku))
        return self.honor_brand(out)

    def shows_brand(self, pid: str, sku: str, named: str = "") -> bool:
        token = (named or self.wish.hard_brand or "").lower().strip()
        if not token:
            return True
        blob = self.text_of(pid, sku).lower()
        toks = token.split()
        return all(t in blob for t in toks) if toks else True

    def labels_deny(self, pid: str, sku: str, tokens=None) -> bool:
        facts = (self.shelf.live(pid, sku).get("facts") or {})
        if not isinstance(facts, dict):
            return False
        toks = [t for t in (tokens or []) if t and len(str(t).rstrip("s")) >= 3]
        if not toks:
            return False
        nos = {"no", "false", "none", "n/a", "na", "not"}
        for key, raw in facts.items():
            if raw is None or raw is True:
                continue
            val = "no" if raw is False else str(raw).lower().strip()
            if not val:
                continue
            key_blob = str(key).lower().replace("_", " ").replace("-", " ")
            for tok in toks:
                stem = str(tok).lower().rstrip("s")
                if re.search(r"\b(?:not|non|no)\s+" + re.escape(stem), val):
                    return True
                if re.search(r"\b(?:not|non|no)-" + re.escape(stem), val):
                    return True
                if re.search(r"\b" + re.escape(stem), key_blob) and val in nos:
                    return True
        return False

    def honor_brand(self, pool: list) -> list:
        named = (self.wish.hard_brand or "").lower()
        if (not named) and self.wish.soft_brand and (not self.shelf.first_add) and (not self.shelf.event):
            named = self.wish.soft_brand.lower()
        if not named or not pool:
            return pool
        hit = [k for k in pool if self.shows_brand(*k, named)]
        if hit:
            return hit
        bag = any(
            self.shows_brand(*k, named) and self.stock_ok(*k) and self.budget_ok(*k)
            for k in self.shelf.row
        )
        if bag:
            return []
        return pool

    def needs_observe(self, pid: str) -> bool:
        return bool(pid) and (not self.wish.phrase) and pid not in self.shelf.viewed

    def fact_fit(self, pid: str, sku: str) -> int:
        item = self.shelf.live(pid, sku)
        facts = item.get("facts") or {}
        n = 0
        if self.wish.inch is not None:
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if (not is_range) and val is not None and round(val, 2) == round(self.wish.inch, 2):
                n += 2
        if self.wish.res:
            fact_disp = " ".join(str(facts.get(k) or "") for k in ("display", "resolution"))
            named = _res_labels(fact_disp, self.wish.res_chart)
            if self.wish.res in named and len(named) == 1:
                n += 2
            elif self.wish.res in named:
                n += 1
        return n

    def dead_kin(self, pid: str, sku: str) -> int:
        if not self.shelf.dead_keys:
            return 0
        text = self.text_of(pid, sku).lower()
        return sum(1 for t in self.shelf.dead_keys if t in text)

    def pref_axes(self, pid: str, sku: str) -> dict:
        text = self.text_of(pid, sku)
        cues = list(self.wish.rest_cues or []) + list(self.wish.axis_cues or []) + list(self.wish.use_cues or [])
        if self.labels_deny(pid, sku, cues):
            return {"pref": 0.0, "hours": _batt_hours(text), "wet": 0, "sport": 0.0}
        wet = _wet(text, cues)
        pref = 1.0 if _cue_hit(text, cues) else 0.0
        sport = 1.0 if _cue_hit(text, self.wish.use_cues) else 0.0
        return {
            "pref": 1.0 if wet >= 3 or pref >= 1.0 else (0.4 if wet > 0 else 0.0),
            "hours": _batt_hours(text),
            "wet": wet,
            "sport": sport,
        }

    def firm_pref(self, pid: str, sku: str) -> bool:
        ax = self.pref_axes(pid, sku)
        return ax["pref"] >= 1.0

    def four_axis(self, pid: str, sku: str) -> bool:
        text = self.text_of(pid, sku)
        ax = self.pref_axes(pid, sku)
        kind = (self.shop_kind() or "").rstrip("s")
        if kind and kind not in text.lower():
            return False
        if self.labels_deny(pid, sku, self.wish.axis_cues or self.wish.wet_cues):
            return False
        return _wet(text, self.wish.axis_cues) >= 3 and ax["hours"] > 0

    def paid_four(self) -> bool:
        for pid, sku in self.shelf.row:
            if pid not in self.shelf.viewed:
                continue
            if (
                self.budget_ok(pid, sku)
                and self.stock_ok(pid, sku)
                and self.identity_ok(pid, sku)
                and self.four_axis(pid, sku)
            ):
                return True
        return False

    def score(self, pid: str, sku: str) -> tuple:
        item = self.shelf.live(pid, sku)
        text = self.text_of(pid, sku)
        price = _fnum(item.get("price")) or 1e18
        wet = _wet(text, self.wish.wet_cues)
        if self.labels_deny(pid, sku, self.wish.wet_cues or self.wish.axis_cues):
            wet = 0
        brand = str(item.get("brand") or "")
        soft = 0
        if self.wish.soft_brand and brand.lower() == self.wish.soft_brand.lower():
            soft = 1
        if self.wish.rank:
            return (self.hz_of(pid, sku) or 0, self.inch_of(pid, sku) or 0, -price)
        if self.wish.model:
            return (-self.relax(pid, sku), soft, -price)
        cov = len(self.row_marks(pid, sku))
        if self.wish.wet:
            ax = self.pref_axes(pid, sku)
            four = 1 if self.four_axis(pid, sku) else 0
            return (
                cov,
                four,
                ax["wet"],
                ax["pref"],
                ax["hours"],
                ax.get("sport") or 0.0,
                -price if four or ax["pref"] >= 1.0 else -price * 0.01,
            )
        if self.wish.inch or self.wish.res or self.wish.hard_brand or self.wish.why:
            if self.shelf.event:
                return (soft, cov, self.fact_fit(pid, sku), self.dead_kin(pid, sku), -price)
            return (soft, cov, -price)
        if not self.shelf.first_add and self.wish.soft_brand:
            return (soft, -price)
        return (soft, wet, -price)

    def cover_ratio(self, pid: str, sku: str) -> float:
        tags = self.needed()
        if not tags:
            return 1.0
        return float(len(self.row_marks(pid, sku))) / float(len(tags))

    def cover_keep(self, pool: list) -> list:
        if len(pool) < 3 or not self.needed() or self.wish.phrase:
            return pool
        marked = [k for k in pool if self.row_marks(*k)]
        if len(marked) >= 2:
            return marked
        return pool

    def has_identity(self) -> bool:
        for pid, sku in self.shelf.row:
            if (pid, sku) in self.shelf.ban:
                continue
            if self.identity_ok(pid, sku):
                return True
        return False

    def rescue_queries(self) -> list:
        kind = self.shop_kind()
        brand = self.wish.hard_brand or self.wish.soft_brand
        out = []

        def add(parts) -> None:
            s = " ".join(str(p) for p in parts if p)
            s = " ".join(s.split())
            if s and s.lower() not in {x.lower() for x in out}:
                out.append(s)

        if brand and kind:
            add([brand, kind])
        if self.wish.model and kind:
            add([kind, self.wish.model])
        if self.wish.color and kind:
            if self.wish.inch is not None:
                add([self.wish.color, kind, "%g inch" % self.wish.inch])
            add([self.wish.color, kind])
        if kind and self.wish.inch is not None:
            add(["%g inch" % self.wish.inch, kind])
        if kind and self.wish.res:
            add([kind, self.wish.res])
        if kind:
            add([kind])
        return out

    def pick(self, vote: bool = True) -> tuple | None:
        pool = self.candidates()
        if not pool:
            return None
        pool = self.cover_keep(pool)
        if self.wish.model:
            if self.shelf.dead_stem:
                sib = [k for k in pool if _sku_stem(k[1]) == self.shelf.dead_stem]
                if sib:
                    pool = sib
            close = [k for k in pool if self.relax(*k) <= (1 if self.shelf.first_add else 0)]
            if close:
                pool = close
            elif self.shelf.search_n < 3:
                return None
        if self.shelf.dead_stem and self.wish.hard_brand:
            sib = [k for k in pool if _sku_stem(k[1]) == self.shelf.dead_stem]
            if sib:
                pool = sib
        if self.wish.wet:
            wet_ok = [
                k
                for k in pool
                if _wet(self.text_of(*k), self.wish.wet_cues) > 0
                or _batt_hours(self.text_of(*k)) > 0
            ]
            if wet_ok:
                pool = wet_ok
            elif self.shelf.search_n < 3:
                return None
            four = [k for k in pool if self.four_axis(*k)]
            if four:
                pool = four
            else:
                return None
            hinted = None
            if vote and self._needs_vote(pool):
                hinted = self.consult_pick(pool)
            return self._review(pool, hinted)
        proven = max(pool, key=lambda ks: self.score(*ks))
        if vote and self._needs_vote(pool):
            return self._review(pool, self.consult_pick(pool))
        return proven

    def _needs_vote(self, pool: list) -> bool:
        if len(pool) < 2 or self.limit.left() < 40 or self.shelf.confirm_n >= 2:
            return False
        if self.wish.phrase:
            return False
        if (
            self.shelf.confirm_n < 1
            and not self.wish.wet
            and (
                self.wish.soft_brand
                or self.wish.hard_brand
                or self.wish.model
                or self.wish.rank
                or self.wish.why
                or self.wish.inch is not None
                or self.wish.res
            )
        ):
            return True
        if self.wish.wet and any(self.four_axis(*k) for k in pool):
            tops = sorted(pool, key=lambda ks: self.score(*ks), reverse=True)[:2]
            return self.score(*tops[0])[:4] == self.score(*tops[1])[:4]
        tops = sorted(pool, key=lambda ks: self.score(*ks), reverse=True)[:2]
        return self.score(*tops[0]) == self.score(*tops[1])

    def fallback_pick(self, vote: bool = False) -> tuple | None:
        choice = self.pick(vote=vote)
        if choice or not self.shelf.event:
            return choice
        pool = []
        for pid, sku in self.shelf.row:
            if (pid, sku) in self.shelf.ban or not sku:
                continue
            if not self.stock_ok(pid, sku) or not self.budget_ok(pid, sku):
                continue
            if not self.spec_ok(pid, sku) or not self.identity_ok(pid, sku):
                continue
            if pid not in self.shelf.viewed and not self.wish.phrase:
                if self.wish.wet or self.wish.kinds:
                    continue
            if self.wish.wet and _wet(self.text_of(pid, sku), self.wish.wet_cues) <= 0 and _batt_hours(self.text_of(pid, sku)) <= 0:
                continue
            pool.append((pid, sku))
        pool = self.honor_brand(pool)
        if not pool:
            return None
        if self.wish.wet:
            four = [k for k in pool if self.four_axis(*k)]
            if four:
                hinted = self.consult_pick(four) if vote and self._needs_vote(four) else None
                return self._review(four, hinted)
            return None
        if vote and self._needs_vote(pool):
            return self._review(pool, self.consult_pick(pool))
        return max(pool, key=lambda ks: self.score(*ks))

    def last_pick(self) -> tuple | None:
        pool = []
        for pid, sku in self.shelf.row:
            if (pid, sku) in self.shelf.ban or not sku:
                continue
            if not self.stock_ok(pid, sku) or not self.budget_ok(pid, sku):
                continue
            if not self.identity_ok(pid, sku):
                continue
            if self.wish.phrase and not self.phrase_hit(pid, sku):
                continue
            pool.append((pid, sku))
        if not pool:
            return None
        viewed = [k for k in pool if k[0] in self.shelf.viewed]
        use = viewed or pool
        use = self.cover_keep(use)
        if self.wish.wet:
            four = [k for k in use if self.four_axis(*k)]
            if four:
                use = four
            else:
                firm = [k for k in use if self.firm_pref(*k)]
                use = firm or use
        return max(use, key=lambda ks: self.score(*ks))

    def _review(self, pool: list, hinted: tuple | None) -> tuple | None:
        if not pool:
            return None
        use = list(pool)
        if self.wish.wet:
            four = [k for k in use if self.four_axis(*k)]
            use = four or [k for k in use if self.firm_pref(*k)] or use
            proven = max(use, key=lambda ks: self.score(*ks))
            if hinted in use:
                if four and not self.four_axis(*hinted):
                    return proven
                if len(self.row_marks(*hinted)) < len(self.row_marks(*proven)):
                    return proven
                if self.score(*hinted) >= self.score(*proven):
                    return hinted
            return proven
        if (not self.shelf.event) and self.wish.soft_brand:
            branded = [k for k in use if self.shows_brand(*k, self.wish.soft_brand)]
            if branded:
                use = branded
        proven = max(use, key=lambda ks: self.score(*ks))
        if hinted in use:
            if self.wish.phrase and not self.phrase_hit(*hinted):
                return proven
            if len(self.row_marks(*hinted)) < len(self.row_marks(*proven)):
                return proven
            if self.score(*hinted) >= self.score(*proven):
                return hinted
        return proven

    def better_cover(self, nxt: tuple, cur: tuple) -> bool:
        if not nxt or not cur:
            return False
        return len(self.row_marks(*nxt)) > len(self.row_marks(*cur))

    def _need_brief(self) -> str:
        bits = []
        if self.wish.phrase:
            bits.append("phrase %s" % self.wish.phrase)
        if self.wish.inch is not None:
            bits.append("size %g" % self.wish.inch)
        if self.wish.res:
            bits.append("resolution %s" % self.wish.res)
        if self.wish.model:
            bits.append("model %s %s" % (self.wish.color or "", self.wish.model))
        if self.wish.rank:
            bits.append("highest refresh among named variants")
        if self.wish.hard_brand:
            bits.append("brand %s" % self.wish.hard_brand)
        if self.wish.soft_brand and not self.shelf.event:
            bits.append("brand %s" % self.wish.soft_brand)
        if self.wish.wet:
            cues = " ".join((self.live_need_words() or [])[:6])
            bits.append("tokens %s plus hours" % (cues or "from the query"))
        bits.append("Use option text and labelled facts before the title.")
        bits.append("A number or model must appear as written.")
        bits.append("Prefer the row that proves more of these constraints; then the lower price.")
        bits.append('Reply with JSON {"product_id":"...","sku":"..."} only.')
        return " ".join(bits)

    def consult_pick(self, pool: list) -> tuple | None:
        lines = []
        for i, (pid, sku) in enumerate(pool[:8], start=1):
            item = self.shelf.live(pid, sku)
            want = (self.wish.soft_brand or self.wish.hard_brand or "").lower()
            brand = str(item.get("brand") or "")
            marks = sorted(self.row_marks(pid, sku))
            lines.append(
                "%d. %s::%s price=%s stock=%s phrase=%s inch=%s spec=%s brand=%s "
                "marks=%s cover=%s wet=%s pref=%s hours=%s hz=%s options=%s title=%s facts=%s"
                % (
                    i,
                    pid,
                    sku,
                    item.get("price"),
                    item.get("in_stock"),
                    self.phrase_hit(pid, sku),
                    self.inch_of(pid, sku),
                    self.spec_ok(pid, sku),
                    1 if want and want in (brand.lower() + " " + str(item.get("title") or "").lower()) else 0,
                    ",".join(marks) or "none",
                    len(marks),
                    _wet(self.text_of(pid, sku), self.wish.wet_cues),
                    _cue_hit(self.text_of(pid, sku), self.wish.wet_cues),
                    _batt_hours(self.text_of(pid, sku)),
                    self.hz_of(pid, sku),
                    self.opt_blob(pid, sku)[:160],
                    (item.get("title") or "")[:120],
                    item.get("facts"),
                )
            )
        messages = [
            {"role": "system", "content": self._need_brief()},
            {"role": "user", "content": "\n".join(lines)},
        ]
        votes = []
        rounds = 1 if self.limit.left() < 40 else 2
        for _ in range(rounds):
            if self.limit.left() < 25:
                break
            msg = self.infer(messages)
            raw = str((msg or {}).get("content") or "")
            m = re.search(r"\{[^{}]+\}", raw)
            if not m:
                continue
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            pid = str(data.get("product_id") or "")
            sku = str(data.get("sku") or "")
            for a, b in pool:
                if a == pid and (not sku or b == sku):
                    votes.append((a, b))
                    break
        if not votes:
            return None
        self.shelf.confirm_n = min(2, self.shelf.confirm_n + 1)
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        winners = [k for k, c in counts.items() if c == top]
        if top >= 2:
            return min(winners, key=lambda ks: _fnum(self.shelf.live(*ks).get("price")) or 1e18)
        return votes[0]

    def claims(self, pid: str, sku: str) -> list:
        item = self.shelf.live(pid, sku)
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        opts = str(item.get("options") or "")
        title = str(item.get("title") or "")
        fact_blob = " ".join(str(v) for v in facts.values())
        out = []
        if self.wish.inch is not None:
            val, is_range = _size_fact(str(facts.get("size") or ""))
            got = None if is_range else val
            if got is None:
                got = _inch(opts) or _inch(title)
            if got is not None and round(got, 2) == round(self.wish.inch, 2):
                out.append({"field": "screen_size", "value": "%gin" % got})
        if self.wish.res:
            observed = str(facts.get("resolution") or facts.get("display") or "").strip()
            if observed:
                out.append({"field": "resolution", "value": observed})
            else:
                blob = opts + " " + title
                marks = list(self.wish.res_marks or [])
                if self.wish.res not in marks:
                    marks.append(self.wish.res)
                shown = ""
                for m in marks:
                    if m and _alias_hit(blob, [m]):
                        shown = m
                        break
                if shown:
                    out.append({"field": "resolution", "value": shown})
        if self.wish.hz_need is not None:
            got = _hz(opts) or _hz(str(facts.get("refresh_rate") or facts.get("display") or ""))
            if got is not None and got + 1e-6 >= self.wish.hz_need:
                out.append({"field": "refresh_rate", "value": "%ghz" % got})
        if self.wish.ram is not None:
            got = _ram_gb(opts + " " + fact_blob)
            if got is None:
                got = _ram_gb(title)
            if got is not None and abs(got - self.wish.ram) <= 0.2:
                out.append({"field": "ram", "value": "%ggb" % got})
        price = _fnum(item.get("price"))
        if price is not None:
            cur = str(item.get("currency") or "").strip()
            out.append({"field": "price", "value": "%.2f%s" % (price, (" " + cur) if cur else "")})
        if item.get("in_stock") is not None:
            out.append({"field": "in_stock", "value": "true" if item.get("in_stock") else "false"})
        brand = str(item.get("brand") or "").strip()
        if brand:
            out.append({"field": "brand", "value": brand})
        elif self.shows_brand(pid, sku):
            named = self.wish.hard_brand or self.wish.soft_brand
            if named:
                out.append({"field": "brand", "value": named})
        shown = (opts + " " + title + " " + fact_blob).lower()
        if self.wish.model and self.wish.model.lower() in shown:
            out.append({"field": "model", "value": self.wish.model})
        if self.wish.color and re.search(
            r"(?<![a-z])" + re.escape(self.wish.color) + r"(?![a-z])", shown
        ):
            out.append({"field": "color", "value": self.wish.color})
        return out

    def live_theme(self) -> list:
        out = []
        for tok in _words(self.wish.head):
            if tok in self.wish.chrome or tok.isdigit() or len(tok) < 3:
                continue
            if tok not in out:
                out.append(tok)
        return out[:8]

    def kind_words(self) -> list:
        return list(self.wish.kinds)

    def shop_kind(self) -> str:
        kinds = self.kind_words()
        named = (self.wish.hard_brand or "").lower()
        cands = [k for k in kinds if k and not (named and k in named)]
        if cands:
            return max(cands, key=len)
        return kinds[0] if kinds else ""

    def live_need_words(self) -> list:
        kind = self.shop_kind()
        rest = []
        for tok in list(self.wish.axis_cues or []) + list(self.wish.use_cues or [])[:2]:
            if tok and tok != kind and tok not in rest:
                rest.append(tok)
        return ([kind] if kind else []) + rest

    def follow_from_hits(self) -> str:
        phrase = self.wish.phrase or ""
        want = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if t]
        if not want:
            return ""
        skip = set(self.wish.chrome)
        rows = list(self.shelf.found) if self.shelf.found else []
        if not rows:
            for pid, sku in self.shelf.row:
                item = self.shelf.live(pid, sku)
                rows.append((pid, sku, str(item.get("title") or ""), str(item.get("brand") or "")))
        seen_pid: set = set()
        uniq = []
        for pid, sku, title, brand in rows:
            if not pid or pid in seen_pid:
                continue
            seen_pid.add(pid)
            uniq.append((title, brand))
        brands: set = set()
        hits: list = []
        for title, brand in uniq:
            for b in re.findall(r"[a-z0-9]+", (brand or "").lower()):
                if len(b) >= 3:
                    brands.add(b)
            raw = (title or "").lower()
            toks = re.findall(r"[a-z0-9]+", raw)
            if not all(w in toks for w in want):
                continue
            hits.append((set(toks), raw))

        def _aisle() -> str:
            if (not any(re.search(r"\d", t) for t in want)) and self.wish.aisle:
                return "%s %s" % (phrase, self.wish.aisle.split("&")[0].strip())
            return ""

        if not hits:
            return _aisle() or self.phrase_reword()

        def glued(tok: str, raw: str) -> bool:
            for w in want:
                if re.search(r"\b%s\s*[-~/]\s*%s\b" % (re.escape(w), re.escape(tok)), raw):
                    return True
                if re.search(r"\b%s\s*[-~/]\s*%s\b" % (re.escape(tok), re.escape(w)), raw):
                    return True
            return False

        present: dict = {}
        for toks, raw in hits:
            seen = set()
            for tok in toks:
                if tok in skip or tok in seen or tok.isdigit() or len(tok) < 3:
                    continue
                if any(tok == w or (len(tok) >= 3 and (tok in w or w in tok)) for w in want):
                    continue
                if glued(tok, raw):
                    continue
                seen.add(tok)
                present[tok] = present.get(tok, 0) + 1
        n_hits = len(hits)

        def pool_of(allow_brand: bool) -> dict:
            out = {}
            for tok, n in present.items():
                if n < 2 or re.search(r"\d", tok):
                    continue
                if n >= n_hits:
                    continue
                if (not allow_brand) and tok in brands:
                    continue
                out[tok] = n
            return out

        pool = pool_of(False) or pool_of(True)
        if not pool:
            return _aisle() or self.phrase_reword()
        extra = max(pool, key=lambda t: (pool[t], len(t)))
        coded = any(re.search(r"\d", t) for t in want)
        if not coded:
            peers = [t for t in pool if t != extra and pool[t] == pool[extra] and len(t) == len(extra)]
            if peers:
                other = max(peers, key=lambda t: (pool[t], len(t)))
                have_a = sum(1 for toks, _raw in hits if extra in toks)
                have_both = sum(1 for toks, _raw in hits if extra in toks and other in toks)
                if have_a and have_both * 2 <= have_a:
                    head = hits[: max(1, (n_hits + 1) // 2)]
                    na = sum(1 for toks, _raw in head if extra in toks)
                    nb = sum(1 for toks, _raw in head if other in toks)
                    if na > nb:
                        extra = other
        q = "%s %s" % (phrase, extra)
        return q if q.lower() != phrase.lower() else ""

    def wet_title_extra(self) -> str:
        words = self.live_need_words() or self.live_theme()
        prefix = " ".join(words[:2] if self.wish.wet else words[:5])
        if not prefix:
            return ""
        want = set(re.findall(r"[a-z0-9]+", prefix))
        skip = set(self.wish.chrome)
        ranked = []
        for pid, sku in self.shelf.row:
            if not self.identity_ok(pid, sku) or not self.stock_ok(pid, sku):
                continue
            blob = (
                str(self.shelf.live(pid, sku).get("title") or "")
                + " "
                + self.opt_blob(pid, sku)
                + " "
                + self.text_of(pid, sku)
            ).lower()
            whole = (
                _batt_hours(blob),
                _wet(blob, self.wish.wet_cues),
                1.0 if _cue_hit(blob, self.wish.wet_cues) else 0.0,
            )
            if whole == (0.0, 0, 0.0):
                continue
            picked = _hours_tok(blob)
            compact = 1 if picked and not re.search(r"\s", picked) else 0
            if not picked:
                live = set(self.live_need_words())
                seen = set()
                for tok in re.findall(r"[a-z0-9]+", blob):
                    if tok in skip or tok in want or tok in seen or len(tok) < 3:
                        continue
                    if tok in live or _wet(tok, self.wish.wet_cues) > 0 or _cue_hit(tok, self.wish.wet_cues):
                        picked = tok
                        break
                    seen.add(tok)
            if not picked:
                continue
            price = _fnum(self.shelf.live(pid, sku).get("price")) or 1e18
            hours = whole[0] if whole[0] > 0 else 1e9
            ranked.append(((-compact, hours, price), picked))
        if not ranked:
            return ""
        extra = min(ranked)[1]
        q = "%s %s" % (prefix, extra)
        return q if q.lower() != prefix.lower() else ""

    def needed(self) -> list:
        tags = []
        if self.wish.hard_brand:
            tags.append("brand")
        if self.wish.inch is not None:
            tags.append("inch")
        if self.wish.res:
            tags.append("res")
        if self.wish.hz_need is not None:
            tags.append("hz")
        if self.wish.model:
            tags.append("model")
        if self.wish.color:
            tags.append("color")
        if self.wish.phrase:
            tags.append("phrase")
        if self.wish.aisle_live and self.wish.aisle:
            tags.append("aisle")
        if self.wish.ram is not None:
            tags.append("ram")
        if self.wish.os_need:
            tags.append("os")
        if self.wish.wet:
            tags.extend(["pref", "hours"])
        return tags

    def row_marks(self, pid: str, sku: str) -> set:
        got = set()
        text = self.text_of(pid, sku)
        if self.wish.hard_brand and self.shows_brand(pid, sku):
            got.add("brand")
        if self.wish.inch is not None:
            val = self.inch_of(pid, sku)
            if val is not None and round(val, 2) == round(self.wish.inch, 2):
                got.add("inch")
        if self.wish.res and self.wish.res in self.res_classes(pid, sku):
            got.add("res")
        if self.wish.hz_need is not None:
            hz = self.hz_of(pid, sku)
            if hz is not None and hz + 1e-6 >= self.wish.hz_need:
                got.add("hz")
        if self.wish.model and self.wish.model.lower() in text.lower():
            got.add("model")
        if self.wish.color and not self.labels_deny(pid, sku, [self.wish.color]):
            shade = self.opt_blob(pid, sku).lower() + " " + text.lower()
            if re.search(r"(?<![a-z])" + re.escape(self.wish.color) + r"(?![a-z])", shade):
                got.add("color")
        if self.wish.phrase and self.phrase_hit(pid, sku):
            got.add("phrase")
        if self.wish.aisle_live and self.aisle_ok(pid, sku):
            got.add("aisle")
        if self.wish.ram is not None:
            got_ram = _ram_gb(str(self.shelf.live(pid, sku).get("options") or "") + " " + text)
            if got_ram is not None and abs(got_ram - self.wish.ram) <= 0.2:
                got.add("ram")
        if self.wish.os_need and _has_os(text, self.wish.os_need) and not self.labels_deny(pid, sku, self.wish.os_need):
            got.add("os")
        if self.wish.wet:
            ax = self.pref_axes(pid, sku)
            if ax["pref"] >= 1.0:
                got.add("pref")
            if ax["hours"] > 0:
                got.add("hours")
        return got

    def paid_marks(self) -> set:
        acc = set()
        for pid, sku in self.shelf.row:
            if (pid, sku) in self.shelf.ban:
                continue
            if pid not in self.shelf.viewed:
                continue
            if not self.stock_ok(pid, sku) or not self.budget_ok(pid, sku):
                continue
            if not self.identity_ok(pid, sku):
                continue
            acc |= self.row_marks(pid, sku)
        return acc

    def gaps(self) -> list:
        have = self.paid_marks()
        return [t for t in self.needed() if t not in have]

    def _gap_span(self, tag: str) -> str:
        if tag == "brand" and self.wish.hard_brand:
            bits = [self.wish.hard_brand]
            if self.wish.inch is not None:
                bits.append("%g inch" % self.wish.inch)
            return " ".join(bits)
        if tag == "inch" and self.wish.inch is not None:
            kind = self.shop_kind()
            return " ".join(x for x in ("%g inch" % self.wish.inch, kind) if x)
        if tag == "res" and self.wish.res:
            return self.wish.res
        if tag == "hz" and self.wish.hz_need is not None:
            return "%g hz" % self.wish.hz_need
        if tag == "model" and self.wish.model:
            return ("%s %s %s" % (self.wish.color or "", self.wish.model, self.shop_kind())).strip()
        if tag == "color" and self.wish.color:
            return ("%s %s" % (self.wish.color, self.shop_kind())).strip()
        if tag == "aisle" and self.wish.aisle:
            return ("%s %s" % (self.wish.aisle.split("&")[0].strip(), self.shop_kind())).strip()
        if tag == "ram" and self.wish.ram is not None:
            return ("%g gb ram %s" % (self.wish.ram, self.shop_kind() or "")).strip()
        if tag == "os" and self.wish.os_need:
            return ("%s %s" % (self.wish.os_need[0], self.shop_kind() or "")).strip()
        if tag == "hours":
            for pid, sku in self.shelf.row:
                if not self.identity_ok(pid, sku):
                    continue
                tok = _hours_tok(self.text_of(pid, sku) + " " + self.opt_blob(pid, sku))
                if tok:
                    return tok
            return ""
        if tag == "pref":
            sent = " ".join(self.shelf.sent_q).lower()
            words = self.live_need_words() or []
            leftover = [t for t in words if t.lower() not in sent]
            if leftover:
                return ("%s %s" % (" ".join(words[:3]), leftover[0])).strip()
            return ""
        return ""

    def gap_query(self) -> str:
        if self.wish.phrase:
            return ""
        sent = {str(q).lower() for q in self.shelf.sent_q}
        for tag in self.gaps():
            cand = " ".join(self._gap_span(tag).split())
            if cand and cand.lower() not in sent:
                return cand
        return ""

    def gap_hint(self, pid: str) -> int:
        missing = self.gaps()
        if not missing:
            return 0
        best = 0
        for p, sku in self.shelf.row:
            if p != pid:
                continue
            marks = self.row_marks(p, sku)
            best = max(best, sum(1 for t in missing if t in marks))
        return best

    def page_wide(self) -> bool:
        return bool(self.shelf.event or self.wish.budget_cut or self.wish.late_firm)

    def next_query(self) -> str:
        sent = {str(q).lower() for q in self.shelf.sent_q}
        qs = self.searches()
        extra = ""
        if self.wish.wet:
            extra = self.wet_title_extra()
        elif self.wish.phrase:
            extra = self.follow_from_hits()
        ordered = []
        if qs:
            ordered.append(qs[0])
        if extra and extra.lower() not in {q.lower() for q in ordered}:
            ordered.append(extra)
            tok = extra.split()[-1] if extra.split() else ""
            if self.wish.wet and tok and tok.lower() not in {x.lower() for x in ordered}:
                ordered.append(tok)
        if (not self.wish.phrase) and self.shelf.search_n >= 1 and not self.has_identity():
            for q in self.rescue_queries():
                if q.lower() not in sent and q.lower() not in {x.lower() for x in ordered}:
                    ordered.append(q)
        for q in qs[1:]:
            if q.lower() not in {x.lower() for x in ordered}:
                ordered.append(q)
        gap = self.gap_query()
        live_need = min(2, len(qs)) if qs else 2
        if gap and self.shelf.search_n >= live_need:
            if gap.lower() not in {q.lower() for q in ordered} and gap.lower() not in sent:
                ordered.append(gap)
        if self.wish.wet:
            seed = list(self.wish.axis_cues or []) + list(self.wish.use_cues or [])[:2]
            leftover = [t for t in seed if t and t.lower() not in " ".join(sent)]
            if leftover:
                span = " ".join((self.live_need_words() or self.live_theme())[:4])
                more = "%s %s" % (span, leftover[0])
                if more.lower() not in {x.lower() for x in ordered}:
                    ordered.append(more)
        for q in ordered:
            if q and q.lower() not in sent:
                return q
        if not qs:
            return (self.wish.head or "")[:80]
        if self.wish.phrase:
            return ""
        return qs[min(self.shelf.search_n, len(qs) - 1)]

    def searches(self) -> list:
        q = []
        kind = self.shop_kind()
        if self.wish.model:
            if self.wish.color:
                q.append("%s %s %s" % (self.wish.color, self.wish.model, kind))
            q.append(("%s %s" % (self.wish.model, kind)).strip())
            q.append(self.wish.model)
        if self.wish.phrase:
            q.append(self.wish.phrase)
            extra = self.follow_from_hits()
            if extra and extra.lower() != self.wish.phrase.lower():
                q.append(extra)
            out = []
            seen = set()
            for s in q:
                s = " ".join(s.split())
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    out.append(s)
            return out or [self.wish.phrase]
        bits = []
        if (self.wish.hard_brand or self.wish.soft_brand) and not self.wish.wet:
            bits.append(self.wish.hard_brand or self.wish.soft_brand)
        if self.wish.wet:
            span = []
            if kind:
                span.append(kind)
            for tok in (self.wish.axis_cues or [])[:4]:
                if tok not in span:
                    span.append(tok)
            for tok in (self.wish.use_cues or [])[:2]:
                if tok not in span:
                    span.append(tok)
            if span:
                bits.append(" ".join(span))
            elif kind:
                bits.append(kind)
        elif kind:
            bits.append(kind)
        if self.wish.inch:
            bits.append("%g inch" % self.wish.inch)
        if self.wish.res:
            bits.append(self.wish.res)
        if self.wish.color:
            bits.append(self.wish.color)
        if bits:
            q.append(" ".join(str(b) for b in bits if b))
        if self.wish.wet:
            words = self.live_need_words()
            if self.wish.axis_cues and kind:
                q.append(" ".join([kind] + list(self.wish.axis_cues)[:4]))
            if words:
                short = " ".join(words[:4])
                if short:
                    q.append(short)
                if self.wish.aisle:
                    q.append("%s %s" % (self.wish.aisle.split("&")[0].strip(), " ".join(words[:3])))
            extra = self.wet_title_extra()
            if extra and extra.lower() not in {s.lower() for s in q}:
                q.insert(1 if q else 0, extra)
            if self.shelf.event and extra:
                q.insert(0, extra)
        if self.wish.soft_brand and kind:
            q.append("%s %s" % (self.wish.soft_brand, kind))
        if kind and self.wish.inch:
            q.append("%g inch %s" % (self.wish.inch, kind))
        if self.wish.hard_brand and self.wish.inch:
            q.append("%s %g inch %s" % (self.wish.hard_brand, self.wish.inch, kind))
        if self.wish.hard_brand and (self.wish.kinds or self.wish.why):
            q.insert(0, self.wish.hard_brand + (" %g inch" % self.wish.inch if self.wish.inch else "") + (" " + self.wish.res if self.wish.res else ""))
        if self.shelf.dead_title and not self.wish.wet:
            dead = self.shelf.dead_title.lower()
            keep = (kind and kind in dead) or (self.wish.model and self.wish.model.lower() in dead)
            if keep:
                q.insert(0, self.shelf.dead_title)
                if self.wish.inch:
                    q.insert(0, ("%s %g inches %s" % (self.shelf.dead_title, self.wish.inch, kind)).strip())
        out = []
        seen = set()
        for s in q:
            s = " ".join(s.split())
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out or [(self.wish.head or "")[:80]]

    def need_view(self) -> list:
        pids = []
        if self.wish.seeds:
            for pid, _sku in self.wish.seeds:
                if pid not in self.shelf.viewed:
                    pids.append(pid)
            return pids
        for pid, sku in self.shelf.row:
            if pid in self.shelf.viewed:
                continue
            if (pid, sku) not in self.shelf.trusted and not self.wish.phrase:
                continue
            if self.wish.phrase and not self.phrase_hit(pid, sku):
                continue
            title = str(self.shelf.live(pid, sku).get("title") or "")
            if self.wish.budget is not None and not self.wish.wet:
                price = _fnum(self.shelf.live(pid, sku).get("price"))
                if price is not None and price > self.wish.budget * 1.05:
                    continue
            if self.wish.reject_words and _junk_hit(title, self.wish.reject_words):
                continue
            if self.wish.kinds and not self.identity_ok(pid, sku):
                continue
            pids.append(pid)
        if self.shelf.dead_stem:
            for pid, sku in self.shelf.row:
                if pid in self.shelf.viewed:
                    continue
                if _sku_stem(sku) == self.shelf.dead_stem:
                    pids.insert(0, pid)
        uniq = []
        seen = set()
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                uniq.append(pid)
        if self.gaps() and not (self.wish.seeds or self.wish.model or self.shelf.dead_stem):
            uniq.sort(key=lambda p: self.gap_hint(p), reverse=True)
        if self.wish.wet:

            def _wet_view_key(pid: str) -> tuple:
                sku = ""
                for p, s in self.shelf.row:
                    if p == pid:
                        sku = s
                        break
                title = str(self.shelf.live(pid, sku).get("title") or "")
                blob = title + " " + self.text_of(pid, sku)
                return (
                    1 if _batt_hours(blob) > 0 else 0,
                    1 if _cue_hit(blob, self.wish.rest_cues or self.wish.wet_cues) else 0,
                    self.gap_hint(pid),
                )

            uniq.sort(key=_wet_view_key, reverse=True)
        return uniq[:8]

    def cart_pair(self) -> tuple | None:
        if not self.shelf.cart:
            return None
        row = self.shelf.cart[0]
        return (str(row.get("product_id") or ""), str(row.get("sku") or ""))

    def cart_valid(self) -> bool:
        pair = self.cart_pair()
        if not pair or not pair[0] or not pair[1]:
            return False
        return self.hard_ok(*pair)

    def commit_ready(self) -> bool:
        if self.shelf.skip_order:
            return False
        if not self.cart_valid():
            return False
        if self.wish.phrase:
            return True
        if self.gaps():
            return False
        return True

    def receipt_ok(self, pid: str, sku: str) -> bool:
        if not pid or not sku or (pid, sku) in self.shelf.ban:
            return False
        return self.stock_ok(pid, sku) and self.budget_ok(pid, sku) and self.identity_ok(pid, sku)

    def reply(self, note: str, pair: tuple | None) -> str:
        low = (note or "").lower()
        if not pair:
            pair = self.last_pick()
        if pair:
            item = self.shelf.live(*pair)
            title = (item.get("title") or pair[0])[:90]
            price = _fnum(item.get("price"))
            cur = str(item.get("currency") or "").strip()
            if "expensive" in low:
                if price is not None and self.wish.budget is not None and price > self.wish.budget + 1e-6:
                    nxt = self.last_pick()
                    extra = ""
                    if nxt and nxt != pair:
                        ntitle = (self.shelf.live(*nxt).get("title") or nxt[0])[:90]
                        extra = " I am going with " + ntitle
                        self.shelf.named = nxt
                    return (
                        "Yes, %s is now %.2f%s, above the budget.%s"
                        % (title[:60], price, (" " + cur) if cur else "", extra)
                    )
            parts = ["I am going with " + title]
            if self.wish.inch is not None:
                parts.append("%gin" % self.wish.inch)
            elif self.wish.res:
                parts.append(self.wish.res)
            if price is not None:
                parts.append("%.2f%s" % (price, (" " + cur) if cur else ""))
            self.shelf.named = pair
            return " ".join(parts)
        need = " ".join((self.live_need_words() or [])[:4])
        return need or (self.wish.head or "")[:80]

    def order_args(self, pid: str, sku: str) -> dict:
        args = {"product_id": pid, "sku": sku}
        if self.wish.why:
            args["justification"] = {"claims": self.claims(pid, sku)}
        return args

    def filt_args(self) -> dict:
        filt: dict = {"in_stock": True, "k": 10}
        if self.wish.aisle:
            filt["category"] = self.wish.aisle
        if self.wish.hard_brand:
            filt["brand"] = self.wish.hard_brand
        elif self.wish.soft_brand:
            keep = True
            if self.shelf.event or self.wish.budget_cut:
                keep = any(
                    str(self.shelf.live(pid, sku).get("brand") or "").lower() == self.wish.soft_brand.lower()
                    and self.budget_ok(pid, sku)
                    and self.stock_ok(pid, sku)
                    and self.spec_ok(pid, sku)
                    for pid, sku in self.shelf.row
                )
            if keep:
                filt["brand"] = self.wish.soft_brand
        if self.wish.budget is not None and not self.wish.wet:
            if not (self.shelf.event or self.wish.budget_cut):
                filt["max_price"] = self.wish.budget
        return filt

    def plan(self) -> list:
        acts: list = []
        self.shelf.skip_order = bool(self.wish.need_page)
        self.wish.need_page = False

        def add(name: str, args: dict) -> None:
            if self.ok(name) and len(acts) < self.max_calls:
                acts.append((name, args))

        def observe(pid: str) -> bool:
            if not self.needs_observe(pid):
                return False
            extra = [p for p in self.need_view() if p and p != pid]
            ids = [pid] + extra
            if self.ok("compare") and len(ids) >= 2 and self.shelf.compares < 8:
                add("compare", {"product_ids": ids[:5]})
                return True
            if self.ok("view"):
                add("view", {"product_id": pid})
                return True
            return False

        if self.wish.ask and not self.shelf.asked:
            need = " ".join((self.live_need_words() or [])[:4])
            add(
                "message",
                {
                    "content": "What value should I use for %s?" % (need or "this"),
                },
            )
            return acts
        if self.wish.pref and not self.shelf.asked and len(self.wish.rest_cues or []) < 3:
            need = " ".join((self.wish.wet_cues or [])[:4])
            add(
                "message",
                {
                    "content": "Which of %s should I use?" % (need or "this"),
                },
            )
            return acts

        if self.shelf.inbox:
            note = self.shelf.inbox[-1]
            self.shelf.inbox.clear()
            note_toks = set(_words(note))
            nudge = len(note_toks & self.wish.chrome) >= max(3, len(note_toks) // 2)
            ask = "?" in note or self.wish.why or bool(re.search(r"\bwhy\b", note, re.I))
            if ask and not nudge:
                if self.shelf.event and (self.wish.rank or self.wish.model or not self.cart_valid()):
                    pair = self.fallback_pick() or self.cart_pair() or self.last_pick()
                else:
                    pair = self.cart_pair() or self.fallback_pick() or self.last_pick()
                add("message", {"content": self.reply(note, pair)})
                if pair and self.ok("inspect_stock"):
                    add("inspect_stock", {"product_id": pair[0], "sku": pair[1]})
                if self.cart_pair() and self.ok("inspect_cart"):
                    add("inspect_cart", {})
                if self.commit_ready() and self.shelf.rechecked:
                    return acts

        pair = self.cart_pair()
        if pair and observe(pair[0]):
            return acts
        if pair and not self.shelf.rechecked:
            add("inspect_stock", {"product_id": pair[0], "sku": pair[1]})
            add("inspect_cart", {})
            return acts

        if self.wish.rank and self.shelf.event and self.wish.seeds:
            missing = [ks for ks in self.wish.seeds if ks not in self.shelf.tick or self.shelf.need_rescan]
            if self.shelf.need_rescan:
                for pid, sku in self.wish.seeds:
                    add("inspect_stock", {"product_id": pid, "sku": sku})
                self.shelf.need_rescan = False
                if acts:
                    return acts
            top = self.pick()
            if pair and top and top != pair:
                if observe(top[0]):
                    return acts
                add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                return acts

        if pair and not self.cart_valid():
            add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
            if pair[0] not in self.shelf.viewed and self.ok("view"):
                add("view", {"product_id": pair[0]})
            nxt = self.fallback_pick(vote=self.limit.left() > 40)
            if (not nxt) and self.shelf.named and self.shelf.named not in self.shelf.ban and self.shelf.named != pair:
                if self.hard_ok(*self.shelf.named) or (
                    self.identity_ok(*self.shelf.named) and self.spec_ok(*self.shelf.named)
                ):
                    nxt = self.shelf.named
            weak = bool(self.wish.wet and nxt and not self.four_axis(*nxt))
            extra = [p for p in self.need_view() if p]
            if extra and (weak or self.page_wide()):
                if self.ok("compare") and len(extra) >= 2 and self.shelf.compares < 8:
                    add("compare", {"product_ids": extra[:5]})
                    return acts
                if self.ok("view"):
                    for pid in extra[: min(5, self.max_calls)]:
                        add("view", {"product_id": pid})
                    if acts:
                        return acts
            if nxt and nxt != pair and nxt not in self.shelf.ban and not weak:
                if observe(nxt[0]):
                    return acts
                add("add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                add("inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
                return acts
            if (not self.wish.seeds) and (not self.wish.phrase) and self.shelf.search_n < (12 if self.shelf.first_add else 8):
                q = self.next_query()
                if q:
                    add("search", {"query": q, "in_stock": True, "k": 10})
                    add("filter", self.filt_args())
                    return acts
            if nxt and nxt != pair and nxt not in self.shelf.ban and not (self.wish.wet and not self.four_axis(*nxt)):
                if observe(nxt[0]):
                    return acts
                add("add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                add("inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
            return acts

        if pair and self.cart_valid():
            top = self.pick(vote=False)
            if (
                top
                and top != pair
                and self.better_cover(top, pair)
                and (not self.wish.wet or self.four_axis(*top) or not self.four_axis(*pair))
            ):
                if observe(top[0]):
                    return acts
                add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                return acts
            if self.wish.rank and self.shelf.event:
                top = self.pick(vote=self.limit.left() > 40)
                if top and top != pair:
                    if observe(top[0]):
                        return acts
                    add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                    add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                    add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                    return acts
            if (self.wish.pref or self.wish.wet) and self.shelf.event:
                if not (self.wish.wet and self.four_axis(*pair)):
                    top = self.fallback_pick(vote=self.limit.left() > 40)
                    if top and top != pair:
                        better = self.score(*top) > self.score(*pair)
                        if self.wish.wet:
                            better = self.score(*top)[:4] > self.score(*pair)[:4]
                        if better:
                            if observe(top[0]):
                                return acts
                            add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                            add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                            add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                            return acts
                extra = [p for p in self.need_view() if p]
                four = self.paid_four()
                if (not four) and extra and self.ok("compare") and self.shelf.compares < 8:
                    add("compare", {"product_ids": extra[:5]})
                    return acts
                if (not four) and (not self.wish.phrase) and self.shelf.search_n < (12 if self.shelf.first_add else 8):
                    q = self.next_query()
                    if q:
                        add("search", {"query": q, "in_stock": True, "k": 10})
                        add("filter", self.filt_args())
                    return acts
            if self.limit.left() > 40 and not self.shelf.final_look:
                self.shelf.final_look = True
                add("inspect_stock", {"product_id": pair[0], "sku": pair[1]})
                add("inspect_cart", {})
                return acts
            if self.commit_ready() and self.ok("place_test_order"):
                add("place_test_order", self.order_args(*pair))
                return acts

        unseen = self.need_view()
        view_cap = 8 if (self.wish.wet or self.page_wide()) else 6
        if unseen and self.ok("compare") and len(unseen) >= 2 and self.shelf.compares < view_cap:
            add("compare", {"product_ids": unseen[:5]})
            return acts
        if unseen and self.ok("view"):
            for pid in unseen[: min(5, self.max_calls)]:
                add("view", {"product_id": pid})
            if acts:
                return acts

        if (not pair) and self.wish.wet and self.shelf.search_n >= 3:
            nxt = self.fallback_pick(vote=False) or self.last_pick()
            if nxt and nxt not in self.shelf.ban and (self.four_axis(*nxt) or not self.next_query()):
                if observe(nxt[0]):
                    return acts
                add("add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                add("inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
                return acts

        if self.wish.seeds and self.wish.rank:
            missing = [ks for ks in self.wish.seeds if ks not in self.shelf.tick]
            if missing:
                for pid, sku in missing[: self.max_calls]:
                    add("inspect_stock", {"product_id": pid, "sku": sku})
                return acts

        sent_q = {str(q).lower() for q in self.shelf.sent_q}
        live_two = []
        for cand in self.searches():
            s = " ".join(str(cand).split())
            if s and s.lower() not in {x.lower() for x in live_two}:
                live_two.append(s)
            if len(live_two) >= 2:
                break
        unsent_live = [q for q in live_two if q.lower() not in sent_q]
        if self.wish.phrase:
            sent_p = sum(1 for q in self.shelf.sent_q if self.phrase_in_query(q))
            need_more = sent_p < 2
        elif self.wish.cover and self.cover_count() < self.wish.cover:
            need_more = True
        elif unsent_live and self.shelf.search_n < 2 and not self.wish.seeds:
            need_more = True
        elif self.wish.seeds:
            need_more = False
        else:
            open_gap = bool(self.gaps())
            no_pick = self.pick(vote=False) is None
            no_hard = not any(self.hard_ok(pid, sku) for pid, sku in self.shelf.row)
            if open_gap or no_pick or (self.page_wide() and no_hard):
                if self.shelf.first_add and self.page_wide():
                    need_more = self.shelf.search_n < 12
                elif self.page_wide():
                    need_more = self.shelf.search_n < 8
                elif open_gap:
                    need_more = self.shelf.search_n < 3
                else:
                    need_more = self.shelf.search_n < 2
            else:
                need_more = False
        if not self.wish.seeds and need_more:
            q = self.next_query()
            if q:
                add("search", {"query": q, "in_stock": True, "k": 10})
                add("filter", self.filt_args())
            return acts

        choice = self.fallback_pick(vote=True)
        if choice:
            pid, sku = choice
            unread = [x for x in self.need_view() if x]
            if (
                self.wish.wet
                and not self.shelf.cart
                and not self.four_axis(pid, sku)
                and unread
                and self.ok("compare")
                and self.shelf.compares < 8
            ):
                add("compare", {"product_ids": unread[:5]})
                return acts
            if self.wish.wet and not self.shelf.cart and not self.paid_four() and self.shelf.search_n < 3:
                q = self.next_query()
                if q:
                    add("search", {"query": q, "in_stock": True, "k": 10})
                    add("filter", self.filt_args())
                return acts
            if (pid, sku) not in self.shelf.tick and self.ok("inspect_stock"):
                add("inspect_stock", {"product_id": pid, "sku": sku})
                pool = self.candidates()
                if len(pool) >= 2:
                    alt = next((k for k in pool if k != (pid, sku) and k not in self.shelf.tick), None)
                    if alt:
                        add("inspect_stock", {"product_id": alt[0], "sku": alt[1]})
                return acts
            if not self.shelf.cart and self.ok("add_to_cart"):
                if observe(pid):
                    return acts
                add("add_to_cart", {"product_id": pid, "sku": sku})
                add("inspect_stock", {"product_id": pid, "sku": sku})
                add("inspect_cart", {})
                return acts

        if (
            not self.wish.seeds
            and not self.wish.phrase
            and self.shelf.search_n < (12 if self.shelf.first_add and self.page_wide() else 8)
            and self.fallback_pick() is None
        ):
            q = self.next_query()
            late = self.last_pick()
            if late and (not q or self.shelf.search_n >= 6):
                pid, sku = late
                if observe(pid):
                    return acts
                add("add_to_cart", {"product_id": pid, "sku": sku})
                add("inspect_stock", {"product_id": pid, "sku": sku})
                return acts
            if q:
                add("search", {"query": q, "in_stock": True, "k": 10})
                add("filter", self.filt_args())
            return acts
        if self.shelf.named and self.shelf.named not in self.shelf.ban and not self.shelf.cart:
            pid, sku = self.shelf.named
            if self.hard_ok(pid, sku) or (self.identity_ok(pid, sku) and self.spec_ok(pid, sku) and self.shows_brand(pid, sku)):
                if observe(pid):
                    return acts
                add("add_to_cart", {"product_id": pid, "sku": sku})
                add("inspect_stock", {"product_id": pid, "sku": sku})
        return acts

    def close(self) -> bool:
        held = list((self.hold or {}).get("batch") or [])
        if held and held[0][0] == "place_test_order":
            try:
                _outs, done = self.fire([])
                if done:
                    return True
            except Exception:
                pass
            try:
                return self._assume_commit(held)
            except Exception:
                self.hold = None
                self.hold_n = 0
                return False
        self.hold = None
        self.hold_n = 0

        def _order_if_ready() -> bool:
            pair = self.cart_pair()
            if not pair or not self.ok("place_test_order"):
                return False
            if self.cart_valid() or self.receipt_ok(*pair):
                _outs, done = self.fire([("place_test_order", self.order_args(*pair))])
                return done
            return False

        if _order_if_ready():
            return True
        choice = self.fallback_pick(vote=False) or self.last_pick()
        if choice and self.ok("add_to_cart") and not self.shelf.cart:
            if self.needs_observe(choice[0]) and self.ok("view"):
                self.fire([("view", {"product_id": choice[0]})])
            self.fire([("add_to_cart", {"product_id": choice[0], "sku": choice[1]})])
            if self.ok("inspect_stock"):
                self.fire([("inspect_stock", {"product_id": choice[0], "sku": choice[1]})])
        if _order_if_ready():
            return True
        return False

    def run(self) -> list:
        while self.turn < self.max_steps:
            if self.limit.left() < 25 or self.turn >= self.max_steps - 3:
                try:
                    if self.close():
                        break
                except Exception:
                    pass
                if self.limit.left() < 8 and self.turn >= self.max_steps - 1:
                    break
            if self.hold:
                held = list(self.hold.get("batch") or [])
                try:
                    _outs, done = self.fire([])
                except Exception as exc:
                    self.talk.append(
                        {"role": "assistant", "content": "fire: %s: %s" % (type(exc).__name__, str(exc)[:120])}
                    )
                    done = False
                if done:
                    break
                if self.hold:
                    self.hold_n += 1
                    if self.hold_n >= 2:
                        if held and held[0][0] == "place_test_order":
                            try:
                                self._assume_commit(held)
                            except Exception:
                                self.hold = None
                                self.hold_n = 0
                            break
                        if self._has_write(held):
                            try:
                                if self._assume_commit(held):
                                    break
                            except Exception:
                                self.hold = None
                                self.hold_n = 0
                        else:
                            self.hold = None
                            self.hold_n = 0
                    continue
                if held and held[0][0] == "place_test_order":
                    break
                continue
            try:
                batch = self.plan()
            except Exception as exc:
                self.talk.append(
                    {"role": "assistant", "content": "plan: %s: %s" % (type(exc).__name__, str(exc)[:120])}
                )
                batch = []
            if not batch:
                if self.shelf.cart and self.ok("place_test_order") and self.cart_valid():
                    pair = self.cart_pair()
                    batch = [("place_test_order", self.order_args(*pair))]
                elif (
                    self.ok("search")
                    and not self.wish.phrase
                    and self.shelf.search_n
                    < (
                        12
                        if self.shelf.first_add and self.page_wide()
                        else (8 if self.page_wide() else 3)
                    )
                    and self.fallback_pick() is None
                ):
                    q = self.next_query()
                    batch = [("search", {"query": q, "in_stock": True, "k": 10})] if q else []
                else:
                    choice = self.fallback_pick()
                    if (not choice) and self.shelf.named and self.shelf.named not in self.shelf.ban:
                        if self.hard_ok(*self.shelf.named) or (
                            self.identity_ok(*self.shelf.named) and self.spec_ok(*self.shelf.named)
                        ):
                            choice = self.shelf.named
                    if choice and choice in self.shelf.ban:
                        choice = None
                    if (not choice):
                        pool = [
                            k
                            for k in self.shelf.row
                            if self.hard_ok(*k) and k not in self.shelf.ban and k[1]
                        ]
                        pool = self.honor_brand(pool)
                        if pool:
                            choice = max(pool, key=lambda ks: self.score(*ks))
                    if choice and self.ok("add_to_cart") and not self.shelf.cart:
                        if self.needs_observe(choice[0]) and self.ok("view"):
                            batch = [("view", {"product_id": choice[0]})]
                        else:
                            batch = [("add_to_cart", {"product_id": choice[0], "sku": choice[1]})]
                    elif self.ok("place_test_order") and (
                        self.cart_valid() or (self.cart_pair() and self.receipt_ok(*self.cart_pair()))
                    ):
                        pair = self.cart_pair()
                        batch = [("place_test_order", self.order_args(*pair))]
                    else:
                        late = self.last_pick()
                        if late and self.ok("add_to_cart") and not self.shelf.cart:
                            if self.needs_observe(late[0]) and self.ok("view"):
                                batch = [("view", {"product_id": late[0]})]
                            else:
                                batch = [("add_to_cart", {"product_id": late[0], "sku": late[1]})]
                        elif self.ok("search") and not self.wish.phrase:
                            q = self.next_query()
                            batch = [("search", {"query": q, "in_stock": True, "k": 10})] if q else []
                        if not batch:
                            try:
                                if self.close():
                                    break
                            except Exception:
                                pass
                            if self.turn >= self.max_steps - 1 or self.limit.left() < 8:
                                break
                            continue
            if not batch:
                continue
            try:
                _outs, done = self.fire(batch)
            except Exception as exc:
                self.talk.append(
                    {"role": "assistant", "content": "fire: %s: %s" % (type(exc).__name__, str(exc)[:120])}
                )
                done = False
            if done:
                break
            if batch and batch[0][0] == "place_test_order":
                break
            if self.cart_valid() and (self.limit.left() < 25 or self.turn >= self.max_steps - 3):
                try:
                    if self.close():
                        break
                except Exception:
                    pass
        return self.talk


def agent_main(problem_data: dict) -> list:
    try:
        if not isinstance(problem_data, dict) or "environment" not in problem_data:
            return []
        return Loop(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "halt: %s: %s" % (type(exc).__name__, str(exc)[:180])}]
