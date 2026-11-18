import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_MODELS = {
    "chutes": (
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "google/gemma-4-31B-turbo-TEE",
        "Qwen/Qwen3.6-27B-TEE",
        "Qwen/Qwen3-32B-TEE",
        "zai-org/GLM-5.2-TEE",
        "unsloth/Mistral-Nemo-Instruct-2407-TEE",
        "moonshotai/Kimi-K2.6-TEE",
        "openai/gpt-oss-120b-TEE",
        "zai-org/GLM-5.1-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "deepseek-ai/DeepSeek-V3.1-TEE",
        "deepseek-ai/DeepSeek-V3-0324-TEE",
        "XiaomiMiMo/MiMo-V2-Flash-TEE",
        "deepseek-ai/DeepSeek-R1-0528-TEE",
    ),
    "openrouter": (
        "deepseek/deepseek-v3.2",
        "google/gemma-4-31b-it",
        "qwen/qwen3.6-27b",
        "qwen/qwen3-32b",
        "z-ai/glm-5.2",
        "xiaomi/mimo-v2-flash",
        "mistralai/mistral-small-2603",
        "moonshotai/kimi-k2.6",
        "openai/gpt-oss-120b",
        "z-ai/glm-5.1",
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-chat-v3-0324",
        "qwen/qwen3.5-397b-a17b",
        "deepseek/deepseek-r1-0528",
    ),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_LIMIT_S = 270.0
_MONEY = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
    r"at most|within|budget (?:is|of)|only go up to|can spend up to)\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?",
    re.I,
)
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
_CLAUSE = re.compile(r"(?<!\d)[.!?](?!\d)|\s+[\u2014\-]\s+|\bit['’]s\b", re.I)


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]{2,}", (text or "").lower())


def _fnum(s: Any) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


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


def _money(text: str) -> float | None:
    best = None
    for m in _MONEY.finditer(text or ""):
        v = _fnum(m.group(1) + "." + (m.group(2) or "0"))
        if v is None:
            continue
        best = v if best is None else min(best, v)
    return best


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
        unit = (m.group(2) or "").lower()
        if unit == "h" and val < 16:
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


def _dead_keys(title: str, chrome: set | None = None) -> set:
    skip = chrome or set()
    out = set()
    for tok in re.findall(r"[a-z0-9]{2,}", (title or "").lower()):
        if tok in skip or tok.isdigit() or len(tok) < 3:
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


def _use_span(head: str) -> str:
    h = head or ""
    stop = r"(?:[.!?]|\s+[\u2014\-]\s+|$)"
    m = re.search(r"\bit['’]s\s+for\s+(.+?)" + stop, h, re.I)
    if m:
        return m.group(1)
    return ""


def _facts_axis(rules: str) -> list:
    m = re.search(r"such as ([^.]+)", rules or "", re.I)
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


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "").strip().lower() or "chutes"


def _sku_stem(sku: str) -> str:
    if not sku:
        return ""
    return sku.rsplit("-", 1)[0]


class Need:
    def __init__(self, query: str) -> None:
        raw = query or ""
        head, _, rules = raw.partition("\n\nTask rules:")
        if not rules:
            head, _, rules = raw.partition("Task rules:")
        self.head = head.strip()
        self.rules = rules
        self.chrome = _chrome(self.head, self.rules)
        self.budget = _money(self.head) or _money(raw)
        self.phrase = ""
        qm = re.search(r'"([^"]{2,80})"', self.head)
        if qm:
            self.phrase = qm.group(1).strip()
        else:
            pm = re.search(r"matches(?: the phrase)?\s+([A-Za-z0-9][A-Za-z0-9 \-_/]{1,60})", self.head, re.I)
            if pm:
                self.phrase = pm.group(1).strip(" .,")
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
        self.kinds = _live_kinds(self.head, self.rules, self.chrome)
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
            idx = low.find(kind)
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
        caps = []
        skip = set(self.chrome) | {k.lower() for k in self.kinds}
        for m in re.finditer(r"\b([A-Z][\w&.'\-]+(?:\s+[A-Z][\w&.'\-]+)?)", self.head):
            tok = m.group(1).strip()
            if "." in tok or tok.lower() in skip:
                continue
            if tok.isupper() and 2 <= len(tok) <= 4:
                continue
            right = self.head[m.end() : m.end() + 16].lower()
            if re.match(r"\s+category\b", right):
                continue
            left = self.head[: m.start()].rstrip()
            if left.endswith((".", "!", "?")) or not left:
                continue
            if tok.lower() not in {x.lower() for x in caps}:
                caps.append(tok)
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
        self.explain = self.why or bool(re.search(r"\breason\b", low))
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
        cm2 = re.search(r"(\d+)\s+distinct", self.rules, re.I)
        self.cover = int(cm2.group(1)) if cm2 else 0

    def take_aisle(self, item: dict) -> None:
        if self.aisle:
            return
        path = (item or {}).get("category_path") or []
        if not isinstance(path, list):
            return
        stems = [k.rstrip("s") for k in self.kinds if k]
        for part in path:
            raw = str(part or "").strip()
            if not raw:
                continue
            low = raw.lower()
            if any(s and s in low for s in stems):
                self.aisle = raw
                return

    def absorb(self, text: str) -> None:
        t = text or ""
        b = _money(t)
        if b is not None and any(w in t.lower() for w in ("budget", "spend", "only go", "balance", "up to", "at most")):
            self.budget = b
        if re.search(r"\bwhy\b", t, re.I):
            self.why = True
            self.explain = True
        if re.search(r"\breason\b", t, re.I):
            self.explain = True
        if "category" in t.lower():
            am = re.search(r"in the ([A-Za-z][\w&'\-][\w&'\- ]{0,40}?) category", t, re.I)
            if not am:
                am = re.search(r"category(?:\s+is)?\s+([A-Za-z][\w&'\-]{2,40})", t, re.I)
            if am:
                self.aisle = am.group(1).strip()
                self.aisle_live = True
        extra_os = _os_chart(t, self.chrome)
        for name in extra_os:
            if name not in self.os_names:
                self.os_names.append(name)
        for name in _os_asked(t, self.os_names):
            if name not in self.os_need:
                self.os_need.append(name)
        for phrase in _reject_phrases(t, t, self.chrome):
            if phrase not in self.reject_words:
                self.reject_words.append(phrase)
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
        inch = _inch(t)
        if inch is not None:
            self.inch = inch
            self.ask = False
        res = _wanted_res(t, self.res_chart or _res_chart(t))
        if res:
            self.res = res
            if res not in self.res_marks:
                self.res_marks.append(res)
            self.ask = False
        hz = _hz(t)
        if hz is not None:
            self.hz_need = hz
            self.ask = False
        ram = _ram_gb(t)
        if ram is not None:
            self.ram = ram
            self.ask = False
class Store:
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
        self.asked = False
        self.inbox: list = []
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
        self.explained = False
        self.confirm_n = 0

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
        if not out.get("currency") and page.get("currency"):
            out["currency"] = page.get("currency")
        out["facts"] = page.get("facts") if isinstance(page.get("facts"), dict) else {}
        out["category_path"] = page.get("category_path") or []
        return out


class Shop:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.bind = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = {(t.get("function") or {}).get("name") for t in (self.policy.get("tools") or []) if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.pid = str(self.bind.get("session_id") or problem.get("problem_id") or problem.get("id") or "p")
        self.need = Need(str(self.policy.get("query") or ""))
        self.bag = Store()
        self.turn = 0
        self.wave = 0
        self.hold: dict | None = None
        self.hold_n = 0
        self.receipt = False
        self.limit = _Limit(_LIMIT_S)
        self.talk: list = []
        self.models = list(_MODELS.get(_provider(), _MODELS["chutes"]))
        pin = getenv("SANDBOX_MODEL")
        if pin:
            self.models = [pin] + [m for m in self.models if m != pin]
        for a, b in self.need.seeds:
            self.bag.put_row({"product_id": a, "sku": b, "in_stock": True}, trusted=True)

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

    def _env_post(self, envelope: dict) -> tuple:
        try:
            got = _PROXY.post_verbose("/environment/call", json_data=envelope)
        except Exception:
            return None, None
        data = got.data if got is not None else None
        if isinstance(data, dict) and isinstance(data.get("calls"), list) and data.get("calls"):
            return data, 200
        err = (got.error if got is not None else None) or {}
        return None, err.get("status")

    def _assume(self, batch: list) -> None:
        for name, args in batch:
            if name in ("search", "filter"):
                self.bag.search_n += 1
                self.bag.hits += 1
            elif name == "compare":
                self.bag.compares += 1
            elif name == "view":
                pid = str((args or {}).get("product_id") or "")
                if pid:
                    self.bag.viewed.add(pid)
            elif name == "add_to_cart":
                pid = str((args or {}).get("product_id") or "")
                sku = str((args or {}).get("sku") or "")
                if pid and sku:
                    self.bag.cart = [{"product_id": pid, "sku": sku}]
                    self.bag.first_add = True
                    self.bag.rechecked = False
            elif name == "remove_from_cart":
                pid = str((args or {}).get("product_id") or "")
                sku = str((args or {}).get("sku") or "")
                if pid and sku:
                    self.bag.ban.add((pid, sku))
                    self.bag.cart = [
                        c
                        for c in self.bag.cart
                        if not (str(c.get("product_id")) == pid and str(c.get("sku")) == sku)
                    ]
            elif name == "message":
                self.bag.asked = True

    def _send_env(self, envelope: dict, replay: bool = False) -> tuple:
        data, status = self._env_post(envelope)
        if data:
            return data, status
        if status == 409:
            return None, 409
        if replay and self.limit.left() >= 20:
            _pause(1, 6.0, self.limit.left())
            data, status = self._env_post(envelope)
            if data:
                return data, status
            if status == 409:
                return None, 409
        return None, status

    def fire(self, batch: list) -> tuple[list, bool]:
        status = None
        if self.hold:
            batch = self.hold["batch"]
            mutating = any(
                n in ("add_to_cart", "remove_from_cart", "place_test_order") for n, _a in batch
            )
            if not mutating:
                self.wave += 1
            envelope = self._envelope(batch, self.turn, self.wave)
            result, status = self._send_env(envelope, replay=False)
        else:
            if not batch:
                return [], False
            self.turn += 1
            self.wave = 0
            envelope = self._envelope(batch, self.turn, 0)
            result, status = self._send_env(envelope, replay=True)
        if not isinstance(result, dict) and status == 409:
            self._assume(batch)
            self.hold = None
            self.hold_n = 0
            self.talk.append({"role": "assistant", "content": "env commit", "batch": batch})
            done = any(n == "place_test_order" for n, _a in batch)
            if done:
                self.receipt = True
            return [], done
        if not isinstance(result, dict):
            mutating = any(n in ("add_to_cart", "remove_from_cart", "place_test_order") for n, _a in batch)
            if (not mutating) and self.limit.left() >= 20:
                self.wave += 1
                envelope = self._envelope(batch, self.turn, self.wave)
                result, status = self._send_env(envelope, replay=False)
                if not isinstance(result, dict) and status == 409:
                    self._assume(batch)
                    self.hold = None
                    self.hold_n = 0
                    self.talk.append({"role": "assistant", "content": "env commit", "batch": batch})
                    done = any(n == "place_test_order" for n, _a in batch)
                    if done:
                        self.receipt = True
                    return [], done
        if not isinstance(result, dict):
            self.hold = {"batch": batch}
            self.talk.append({"role": "assistant", "content": "env miss", "batch": batch})
            return [], False
        self.hold = None
        self.hold_n = 0
        self.talk.append({"role": "assistant", "batch": batch, "environment_result": result})
        um = result.get("user_message")
        if isinstance(um, dict) and um.get("content"):
            try:
                self.need.absorb(str(um["content"]))
                self.bag.inbox.append(str(um["content"]))
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
        if done:
            self.receipt = True
        return outs, done

    def ingest(self, name: str, args: dict, obs: Any) -> None:
        if isinstance(obs, dict) and obs.get("error"):
            return
        inner = _inner(obs)
        if name == "search":
            self.bag.hits += 1
            self.bag.search_n += 1
            for row in inner.get("results") or []:
                if isinstance(row, dict):
                    self.bag.put_row(row, trusted=True)
        elif name == "filter":
            self.bag.hits += 1
            trust = bool(args.get("brand") or self.need.phrase or self.need.model)
            for row in inner.get("results") or []:
                if not isinstance(row, dict):
                    continue
                self.bag.put_row(row, trusted=trust)
        elif name == "view":
            self.bag.put_page(inner)
            self.need.take_aisle(inner)
        elif name == "compare":
            self.bag.compares += 1
            for block in inner.get("compared") or []:
                if isinstance(block, dict):
                    self.bag.put_page(block)
                    self.need.take_aisle(block)
        elif name == "inspect_stock":
            pid = str(args.get("product_id") or inner.get("product_id") or "")
            sku = str(args.get("sku") or inner.get("sku") or "")
            if pid and sku:
                self.bag.tick[(pid, sku)] = {
                    "price": inner.get("price"),
                    "in_stock": inner.get("in_stock"),
                }
        elif name == "add_to_cart":
            added = inner.get("added") if isinstance(inner.get("added"), dict) else {}
            cart = inner.get("cart") if isinstance(inner.get("cart"), list) else []
            self.bag.cart = cart
            self.bag.first_add = True
            self.bag.rechecked = False
            ev = inner.get("market_event")
            if isinstance(ev, dict):
                self.bag.event = True
                self.bag.need_rescan = True
                tgt = ev.get("target") if isinstance(ev.get("target"), dict) else {}
                pid = str(tgt.get("product_id") or added.get("product_id") or args.get("product_id") or "")
                sku = str(tgt.get("sku") or added.get("sku") or args.get("sku") or "")
                if pid and sku:
                    self.bag.dead_stem = _sku_stem(sku)
                    title = str((self.bag.live(pid, sku) or {}).get("title") or "")
                    self.bag.dead_title = " ".join(title.split()[:8])
                    self.bag.dead_keys = _dead_keys(title, self.need.chrome)
                    self.bag.tick[(pid, sku)] = {
                        "price": ev.get("new_price", added.get("price")),
                        "in_stock": ev.get("in_stock_after", added.get("in_stock")),
                    }
            elif added:
                pid = str(added.get("product_id") or args.get("product_id") or "")
                sku = str(added.get("sku") or args.get("sku") or "")
                if pid and sku:
                    self.bag.tick[(pid, sku)] = {
                        "price": added.get("price"),
                        "in_stock": added.get("in_stock"),
                    }
        elif name == "inspect_cart":
            cart = inner.get("cart")
            if isinstance(cart, list):
                self.bag.cart = cart
            elif isinstance(inner.get("items"), list):
                self.bag.cart = inner["items"]
            self.bag.rechecked = True
        elif name == "remove_from_cart":
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and sku:
                self.bag.ban.add((pid, sku))
                if self.bag.named == (pid, sku):
                    self.bag.named = None
                self.bag.cart = [
                    c
                    for c in self.bag.cart
                    if not (str(c.get("product_id")) == pid and str(c.get("sku")) == sku)
                ]
        elif name == "message":
            self.bag.asked = True
        for d in _walk(inner):
            if d.get("product_id") and (d.get("sku") or d.get("title") or d.get("variants")):
                if "variants" in d:
                    self.bag.put_page(d)
                elif d.get("sku") and name in ("search", "view", "compare", "inspect_stock"):
                    self.bag.put_row(d, trusted=True)

    def text_of(self, pid: str, sku: str) -> str:
        item = self.bag.live(pid, sku)
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

    def evidence_of(self, pid: str, sku: str) -> str:
        item = self.bag.live(pid, sku)
        facts = item.get("facts") or {}
        return _blob(
            [
                item.get("options"),
                " ".join("%s %s" % (k, v) for k, v in facts.items()),
            ]
        )

    def opt_blob(self, pid: str, sku: str) -> str:
        item = self.bag.live(pid, sku)
        return _blob([item.get("options"), item.get("title")])

    def res_classes(self, pid: str, sku: str) -> set:
        item = self.bag.live(pid, sku)
        facts = item.get("facts") or {}
        fact_disp = " ".join(str(facts.get(k) or "") for k in ("display", "resolution"))
        fact_disp = fact_disp.replace("*", " x ").replace("×", " x ")
        opts = str(item.get("options") or "")
        chart = self.need.res_chart
        if fact_disp.strip() and pid in self.bag.viewed:
            named = _res_labels(fact_disp, chart)
            if named:
                return named
            return _res_labels(fact_disp + " " + opts, chart)
        return _res_labels(opts + " " + str(item.get("title") or "") + " " + self.text_of(pid, sku), chart)

    def inch_of(self, pid: str, sku: str) -> float | None:
        item = self.bag.live(pid, sku)
        facts = item.get("facts") or {}
        from_opt = _inch(str(item.get("options") or "")) or _inch(str(item.get("title") or ""))
        if self.need.rank:
            return from_opt or _inch(str(facts.get("size") or ""))
        val, is_range = _size_fact(str(facts.get("size") or ""))
        if is_range:
            return None
        if val is not None:
            return val
        return from_opt

    def hz_of(self, pid: str, sku: str) -> float | None:
        item = self.bag.live(pid, sku)
        facts = item.get("facts") or {}
        return _hz(self.opt_blob(pid, sku)) or _hz(str(facts.get("refresh_rate") or facts.get("display") or ""))

    def shop_kind(self) -> str:
        kinds = list(self.need.kinds or [])
        named = (self.need.hard_brand or "").lower()
        cands = [k for k in kinds if k and not (named and k in named)]
        if cands:
            return max(cands, key=len)
        return kinds[0] if kinds else ""

    def identity_ok(self, pid: str, sku: str) -> bool:
        item = self.bag.live(pid, sku)
        text = self.text_of(pid, sku).lower()
        title = str(item.get("title") or "").lower()
        facts = item.get("facts") or {}
        if self.need.phrase:
            return True
        if not self.need.kinds:
            if self.need.model and self.need.model.lower() in text:
                return True
            return False
        if _junk_hit(text, self.need.reject_words) or _junk_hit(str(item.get("options") or ""), self.need.reject_words):
            return False
        named = (self.need.hard_brand or "").lower()
        if named:
            brand = str(item.get("brand") or "").lower()
            if named not in brand and named not in title and named not in text:
                return False
        stems = []
        kind = (self.shop_kind() or "").rstrip("s")
        if kind:
            stems.append(kind)
        if self.need.aisle:
            for tok in _words(self.need.aisle):
                stem = tok.rstrip("s")
                if stem and stem not in stems:
                    stems.append(stem)
        hit = any(s and s in text for s in stems)
        if self.need.os_names:
            return (
                hit
                or _has_os(text, self.need.os_names)
                or any(s and s in str(facts.get("phone_type") or "").lower() for s in stems)
            )
        if self.need.seeds:
            return True
        return hit

    def spec_ok(self, pid: str, sku: str) -> bool:
        item = self.bag.live(pid, sku)
        text = self.text_of(pid, sku)
        if self.need.inch is not None:
            facts = item.get("facts") or {}
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if pid in self.bag.viewed and str(facts.get("size") or "").strip():
                if is_range or val is None or round(val, 2) != round(self.need.inch, 2):
                    return False
            else:
                sizes = [
                    _inch(str(item.get("options") or "")),
                    val,
                    _inch(str(item.get("title") or "")),
                ]
                if not any(s is not None and round(s, 2) == round(self.need.inch, 2) for s in sizes):
                    return False
        if self.need.res:
            labels = self.res_classes(pid, sku)
            marks = self.need.res_marks or [self.need.res]
            if not any(m in labels or _alias_hit(text, [m]) for m in marks):
                return False
        if self.need.hz_need is not None:
            got = self.hz_of(pid, sku)
            if got is None or got + 1e-6 < self.need.hz_need:
                return False
        if self.need.ram is not None:
            got = _ram_gb(str(item.get("options") or "") + " " + text)
            if got is None or abs(got - self.need.ram) > 0.2:
                return False
        if self.need.hard_brand:
            title = str(item.get("title") or "").lower()
            brand = str(item.get("brand") or "").lower()
            toks = self.need.hard_brand.lower().split()
            if not (all(t in brand for t in toks) or all(t in title for t in toks)):
                return False
        return True

    def budget_ok(self, pid: str, sku: str) -> bool:
        item = self.bag.live(pid, sku)
        price = _fnum(item.get("price"))
        if self.need.budget is None or price is None:
            return price is not None
        return price <= self.need.budget + 1e-6

    def stock_ok(self, pid: str, sku: str) -> bool:
        item = self.bag.live(pid, sku)
        return bool(item.get("in_stock"))

    def fact_stated(self, pid: str, sku: str) -> bool:
        if not self.need.explain:
            return True
        want = []
        if self.need.inch is not None:
            want.append("screen_size")
        if self.need.res:
            want.append("resolution")
        if self.need.hz_need is not None:
            want.append("refresh_rate")
        if self.need.ram is not None:
            want.append("ram")
        if not want:
            return True
        if pid not in self.bag.viewed:
            return False
        have = {str(c.get("field") or "").lower() for c in self.claims(pid, sku)}
        return all(f in have for f in want)

    def hard_ok(self, pid: str, sku: str) -> bool:
        if (pid, sku) in self.bag.ban:
            return False
        if not sku:
            return False
        if self.need.rank:
            if (pid, sku) not in self.need.seeds:
                return False
            if self.hz_of(pid, sku) is None or self.inch_of(pid, sku) is None:
                return False
            return self.stock_ok(pid, sku) and self.budget_ok(pid, sku)
        if self.need.model and self.bag.dead_stem and _sku_stem(sku) == self.bag.dead_stem:
            return self.stock_ok(pid, sku) and self.budget_ok(pid, sku)
        if pid not in self.bag.viewed and not self.need.phrase:
            if not self.bag.event and (self.need.kinds or self.need.why or self.need.explain or self.need.model):
                return False
        if self.need.explain and not self.fact_stated(pid, sku):
            return False
        return self.stock_ok(pid, sku) and self.budget_ok(pid, sku) and self.spec_ok(pid, sku) and self.identity_ok(pid, sku)

    def phrase_hit(self, pid: str, sku: str) -> bool:
        if not self.need.phrase:
            return True
        title = (self.bag.live(pid, sku).get("title") or "").lower()
        have = set(re.findall(r"[a-z0-9]+", title))
        toks = [t for t in re.findall(r"[a-z0-9]+", self.need.phrase.lower()) if t]
        return all(t in have for t in toks) if toks else True

    def follow_from_hits(self) -> str:
        phrase = self.need.phrase or ""
        want = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if t]
        skip = set(self.need.chrome)
        hits = 0
        present: dict = {}
        brands = set()
        for pid, sku in self.bag.row:
            item = self.bag.live(pid, sku)
            for b in re.findall(r"[a-z0-9]+", str(item.get("brand") or "").lower()):
                if len(b) >= 3:
                    brands.add(b)
            title = str(item.get("title") or "").lower()
            toks = re.findall(r"[a-z0-9]+", title)
            if want and not all(w in toks for w in want):
                continue
            hits += 1
            seen = set()
            for tok in toks:
                if tok in skip or tok in seen or tok.isdigit() or len(tok) < 3:
                    continue
                if any(tok == w or (len(tok) >= 3 and (tok in w or w in tok)) for w in want):
                    continue
                seen.add(tok)
                present[tok] = present.get(tok, 0) + 1

        def pool_of(allow_brand: bool) -> dict:
            out = {}
            for tok, n in present.items():
                if n < 2 or re.search(r"\d", tok):
                    continue
                if (not allow_brand) and tok in brands:
                    continue
                if hits and n >= hits:
                    continue
                out[tok] = n
            return out

        pool = pool_of(False) or pool_of(True)
        if pool and hits:
            rank = lambda t: (pool[t] * (hits - pool[t]), -pool[t], len(t))
            top = max(pool, key=rank)
            coded = any(re.search(r"\d", t) for t in want)
            extra = top
            if not coded:
                brand_n: dict = {}
                for pid, sku in self.bag.row:
                    item = self.bag.live(pid, sku)
                    title = str(item.get("title") or "").lower()
                    toks = re.findall(r"[a-z0-9]+", title)
                    if want and not all(w in toks for w in want):
                        continue
                    b = str(item.get("brand") or "").strip().lower()
                    if len(b) >= 3 and b not in skip:
                        brand_n[b] = brand_n.get(b, 0) + 1
                rare = [b for b, n in brand_n.items() if n == 1]
                if rare:
                    pick = max(rare, key=len)
                    q = "%s %s" % (phrase, pick)
                    if q.lower() != phrase.lower():
                        return q
                other: dict = {}
                for pid, sku in self.bag.row:
                    title = str(self.bag.live(pid, sku).get("title") or "").lower()
                    toks = re.findall(r"[a-z0-9]+", title)
                    if want and not all(w in toks for w in want):
                        continue
                    if top in toks:
                        continue
                    seen = set()
                    for tok in toks:
                        if tok in skip or tok in seen or tok.isdigit() or len(tok) < 3:
                            continue
                        if any(tok == w or (len(tok) >= 3 and (tok in w or w in tok)) for w in want):
                            continue
                        if re.search(r"\d", tok):
                            continue
                        seen.add(tok)
                        other[tok] = other.get(tok, 0) + 1
                if other:
                    extra = max(other, key=lambda t: (other[t], len(t)))
            q = "%s %s" % (phrase, extra)
            if q.lower() != phrase.lower():
                return q
        if self.need.aisle:
            return "%s %s" % (phrase, self.need.aisle.split("&")[0].strip())
        return ""

    def cover_count(self) -> int:
        ids = set()
        for pid, sku in self.bag.row:
            if self.phrase_hit(pid, sku) and self.stock_ok(pid, sku) and self.budget_ok(pid, sku):
                ids.add(pid)
        return len(ids)

    def relax(self, pid: str, sku: str) -> int:
        if not self.need.model:
            return 0
        text = self.text_of(pid, sku).lower()
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        same_model = bool(
            re.search(r"(?<!\d)" + re.escape(self.need.model) + r"(?!\d)", text)
            or squash(self.need.model) in squash(text)
        )
        if self.bag.dead_stem and _sku_stem(sku) == self.bag.dead_stem:
            same_model = True
        shade = self.opt_blob(pid, sku).lower() + " " + str((self.bag.live(pid, sku).get("facts") or {}).get("color") or "")
        same_color = bool(self.need.color) and re.search(
            r"(?<![a-z])" + re.escape(self.need.color) + r"(?![a-z])", shade + " " + text
        )
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def candidates(self) -> list:
        out = []
        keys = list(self.bag.row)
        if self.need.seeds:
            keys = list(self.need.seeds) + [k for k in keys if k not in self.need.seeds]
        for pid, sku in keys:
            if not self.hard_ok(pid, sku):
                continue
            if self.need.phrase and not self.phrase_hit(pid, sku):
                continue
            if not self.need.seeds and (pid, sku) not in self.bag.trusted and not self.need.phrase:
                if self.need.kinds:
                    continue
            out.append((pid, sku))
        return self.honor_brand(out)

    def shows_brand(self, pid: str, sku: str, named: str = "") -> bool:
        token = (named or self.need.hard_brand or "").lower().strip()
        if not token:
            return True
        blob = self.text_of(pid, sku).lower()
        toks = token.split()
        return all(t in blob for t in toks) if toks else True

    def labels_deny(self, pid: str, sku: str, tokens=None) -> bool:
        facts = self.bag.live(pid, sku).get("facts") or {}
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
        named = (self.need.hard_brand or "").lower()
        if (not named) and self.need.soft_brand and (not self.bag.first_add) and (not self.bag.event):
            named = self.need.soft_brand.lower()
        if not named or not pool:
            return pool
        hit = [k for k in pool if self.shows_brand(*k, named)]
        if hit:
            return hit
        bag = any(
            self.shows_brand(*k, named) and self.stock_ok(*k) and self.budget_ok(*k)
            for k in self.bag.row
        )
        if bag:
            return []
        return pool

    def fact_fit(self, pid: str, sku: str) -> int:
        item = self.bag.live(pid, sku)
        facts = item.get("facts") or {}
        n = 0
        if self.need.inch is not None:
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if (not is_range) and val is not None and round(val, 2) == round(self.need.inch, 2):
                n += 2
        if self.need.res:
            fact_disp = " ".join(str(facts.get(k) or "") for k in ("display", "resolution"))
            named = _res_labels(fact_disp, self.need.res_chart)
            if self.need.res in named and len(named) == 1:
                n += 2
            elif self.need.res in named:
                n += 1
        return n

    def dead_kin(self, pid: str, sku: str) -> int:
        if not self.bag.dead_keys:
            return 0
        text = self.text_of(pid, sku).lower()
        return sum(1 for t in self.bag.dead_keys if t in text)

    def pref_axes(self, pid: str, sku: str) -> dict:
        evidence = self.evidence_of(pid, sku)
        whole = self.text_of(pid, sku)
        wet = _wet(evidence, self.need.wet_cues) or _wet(whole, self.need.wet_cues)
        sport = 1.0 if (_cue_hit(evidence, self.need.wet_cues) or _cue_hit(whole, self.need.wet_cues)) else 0.0
        hours = _batt_hours(evidence) or _batt_hours(whole)
        wire = 1.0 if (_cue_hit(evidence, self.need.wire_cues) or _cue_hit(whole, self.need.wire_cues)) else 0.0
        return {
            "sweat": 1.0 if wet >= 3 else (0.4 if wet > 0 else 0.0),
            "sport": sport,
            "hours": hours,
            "wire": wire,
            "wet": wet,
        }

    def firm_pref(self, pid: str, sku: str) -> bool:
        ax = self.pref_axes(pid, sku)
        if not self.need.wet_cues:
            return True
        return ax["wet"] > 0 or ax["sweat"] >= 1.0 or ax["sport"] >= 1.0

    def four_axis(self, pid: str, sku: str) -> bool:
        ax = self.pref_axes(pid, sku)
        if self.need.wet or self.need.wet_cues:
            if ax["wet"] <= 0 and ax["sweat"] < 1.0:
                return False
            if ax["hours"] <= 0:
                return False
        if self.need.wire_cues and ax["wire"] < 1.0:
            return False
        return True

    def score(self, pid: str, sku: str) -> tuple:
        item = self.bag.live(pid, sku)
        text = self.text_of(pid, sku)
        price = _fnum(item.get("price")) or 1e18
        wet = _wet(text, self.need.wet_cues)
        brand = str(item.get("brand") or "")
        soft = 0
        if self.need.soft_brand and brand.lower() == self.need.soft_brand.lower():
            soft = 1
        if self.need.rank:
            return (self.hz_of(pid, sku) or 0, self.inch_of(pid, sku) or 0, -price)
        if self.need.model:
            return (-self.relax(pid, sku), soft, -price)
        if self.need.wet:
            ax = self.pref_axes(pid, sku)
            return (
                ax["sweat"],
                ax["sport"],
                ax["hours"],
                ax["wire"],
                ax["wet"],
                -price if ax["sweat"] >= 1.0 else -price * 0.01,
            )
        if (self.need.inch and self.need.kinds) or self.need.hard_brand or self.need.why:
            if self.bag.event:
                return (soft, self.fact_fit(pid, sku), self.dead_kin(pid, sku), -price)
            return (soft, price)
        if not self.bag.first_add and self.need.soft_brand:
            return (soft, -price)
        return (soft, wet, -price)

    def pick(self) -> tuple | None:
        pool = self.honor_brand(self.candidates())
        if not pool:
            return None
        if self.need.model:
            if self.bag.dead_stem:
                sib = [k for k in pool if _sku_stem(k[1]) == self.bag.dead_stem]
                if sib:
                    pool = sib
            close = [k for k in pool if self.relax(*k) <= (1 if self.bag.first_add else 0)]
            if close:
                pool = close
            elif self.bag.search_n < 3:
                return None
        if self.need.wet:
            wet_ok = [k for k in pool if _wet(self.text_of(*k), self.need.wet_cues) > 0]
            if wet_ok:
                pool = wet_ok
            elif self.bag.search_n < 3:
                return None
            firm = [k for k in pool if self.firm_pref(*k)]
            if firm:
                pool = firm
        pool = self.honor_brand(pool)
        if not pool:
            return None
        voted = self.take_confirm(pool)
        if voted:
            return voted
        return max(pool, key=lambda ks: self.score(*ks))

    def fallback_pick(self) -> tuple | None:
        choice = self.pick()
        if choice or not self.bag.event:
            return choice
        pool = []
        for pid, sku in self.bag.row:
            if (pid, sku) in self.bag.ban or not sku:
                continue
            if not self.stock_ok(pid, sku) or not self.budget_ok(pid, sku):
                continue
            if not self.spec_ok(pid, sku) or not self.identity_ok(pid, sku):
                continue
            if self.need.wet and _wet(self.text_of(pid, sku), self.need.wet_cues) <= 0:
                continue
            pool.append((pid, sku))
        pool = self.honor_brand(pool)
        if not pool:
            return None
        return max(pool, key=lambda ks: self.score(*ks))

    def last_pick(self) -> tuple | None:
        pool = []
        for pid, sku in self.bag.row:
            if (pid, sku) in self.bag.ban or not sku:
                continue
            if not self.stock_ok(pid, sku) or not self.budget_ok(pid, sku):
                continue
            if not self.identity_ok(pid, sku):
                continue
            if self.need.phrase and not self.phrase_hit(pid, sku):
                continue
            pool.append((pid, sku))
        if not pool:
            return None
        viewed = [k for k in pool if k[0] in self.bag.viewed]
        use = viewed or pool
        spec = [k for k in use if self.spec_ok(*k)]
        if spec:
            use = spec
        return max(use, key=lambda ks: self.score(*ks))

    def stop_pick(self) -> tuple | None:
        choice = self.fallback_pick() or self.last_pick()
        if choice:
            return choice
        if self.bag.named and self.bag.named not in self.bag.ban:
            return self.bag.named
        pool = []
        for pid, sku in self.bag.row:
            if (pid, sku) in self.bag.ban or not sku:
                continue
            if not self.stock_ok(pid, sku) or not self.budget_ok(pid, sku):
                continue
            if self.need.phrase and not self.phrase_hit(pid, sku):
                continue
            if pid not in self.bag.viewed:
                continue
            pool.append((pid, sku))
        if not pool:
            return None
        spec = [k for k in pool if self.spec_ok(*k)]
        use = spec or pool
        return max(use, key=lambda ks: self.score(*ks))

    def wet_prefix(self, pid: str, sku: str) -> tuple:
        ax = self.pref_axes(pid, sku)
        return (ax["sweat"], ax["sport"], ax["hours"], ax["wire"], ax["wet"])

    def want_confirm(self, pool: list) -> bool:
        if self.need.phrase or len(pool) < 2:
            return False
        if self.limit.left() <= 40 or self.bag.confirm_n >= 2:
            return False
        ranked = sorted(pool, key=lambda ks: self.score(*ks), reverse=True)
        if self.bag.confirm_n >= 1:
            return self.score(*ranked[0]) == self.score(*ranked[1])
        if self.need.wet:
            return self.wet_prefix(*ranked[0]) == self.wet_prefix(*ranked[1])
        return bool(
            self.need.soft_brand or self.need.model or self.need.rank or self.need.why
        )

    def take_confirm(self, pool: list) -> tuple | None:
        if not self.want_confirm(pool):
            return None
        self.bag.confirm_n += 1
        choice = self.consult_pick(pool)
        if not choice or choice not in pool:
            return None
        if not self.honor_brand([choice]):
            return None
        if self.need.phrase and not self.phrase_hit(*choice):
            return None
        if self.need.wet:
            firm = [k for k in pool if self.firm_pref(*k)]
            if firm and not self.firm_pref(*choice):
                return None
        best = max(pool, key=lambda ks: self.score(*ks))
        if self.hard_ok(*best) and not self.hard_ok(*choice):
            return None
        if self.score(*choice) < self.score(*best):
            return None
        return choice

    def consult_pick(self, pool: list) -> tuple | None:
        lines = []
        for i, (pid, sku) in enumerate(pool[:8], start=1):
            item = self.bag.live(pid, sku)
            lines.append(
                "%d. %s::%s price=%s stock=%s wet=%s sport=%s hours=%s title=%s facts=%s"
                % (
                    i,
                    pid,
                    sku,
                    item.get("price"),
                    item.get("in_stock"),
                    _wet(self.text_of(pid, sku), self.need.wet_cues),
                    1.0 if _cue_hit(self.text_of(pid, sku), self.need.wet_cues) else 0.0,
                    _batt_hours(self.text_of(pid, sku)),
                    (item.get("title") or "")[:120],
                    item.get("facts"),
                )
            )
        cues = " ".join((self.live_need_words() or [])[:6])
        msg = self.infer(
            [
                {
                    "role": "system",
                    "content": (
                        "Confirm each listed variant against the shopper constraints already written (%s). "
                        "Use option text and labelled facts first; title tokens are weaker. "
                        "Prefer a listed battery life in hours when that figure is listed. "
                        "Reply with JSON product_id and sku only."
                    )
                    % (cues or "the leftover use tokens"),
                },
                {"role": "user", "content": "\n".join(lines)},
            ]
        )
        if not msg:
            return None
        raw = str(msg.get("content") or "")
        m = re.search(r"\{[^{}]+\}", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        pid = str(data.get("product_id") or "")
        sku = str(data.get("sku") or "")
        if pid and sku:
            return (pid, sku)
        return None

    def claims(self, pid: str, sku: str) -> list:
        item = self.bag.live(pid, sku)
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        opts = str(item.get("options") or "")
        title = str(item.get("title") or "")
        fact_blob = " ".join(str(v) for v in facts.values())
        out = []
        if self.need.explain:
            if self.need.inch is not None:
                raw = str(facts.get("size") or "").strip()
                val, is_range = _size_fact(raw)
                shown = ""
                if (not is_range) and val is not None and round(val, 2) == round(self.need.inch, 2):
                    shown = raw
                else:
                    got = _inch(opts)
                    if got is not None and round(got, 2) == round(self.need.inch, 2):
                        m = _INCH.search(opts)
                        shown = m.group(0).strip() if m else ""
                if shown:
                    out.append({"field": "screen_size", "value": shown})
            if self.need.res:
                observed = str(facts.get("resolution") or facts.get("display") or "").strip()
                if observed:
                    out.append({"field": "resolution", "value": observed})
                else:
                    marks = list(self.need.res_marks or [])
                    if self.need.res not in marks:
                        marks.append(self.need.res)
                    shown = ""
                    for m in marks:
                        if m and _alias_hit(opts, [m]):
                            hit = re.search(re.escape(m), opts, re.I)
                            shown = hit.group(0) if hit else m
                            break
                    if shown:
                        out.append({"field": "resolution", "value": shown})
            if self.need.hz_need is not None:
                raw_hz = str(facts.get("refresh_rate") or facts.get("display") or "").strip()
                shown = ""
                if _hz(raw_hz) is not None and _hz(raw_hz) + 1e-6 >= self.need.hz_need:
                    shown = raw_hz
                else:
                    got = _hz(opts)
                    if got is not None and got + 1e-6 >= self.need.hz_need:
                        m = _HZ.search(opts)
                        shown = m.group(0).strip() if m else ""
                if shown:
                    out.append({"field": "refresh_rate", "value": shown})
            if self.need.ram is not None:
                raw_ram = str(facts.get("ram") or facts.get("memory") or "").strip()
                shown = ""
                if _ram_gb(raw_ram) is not None and abs(_ram_gb(raw_ram) - self.need.ram) <= 0.2:
                    shown = raw_ram
                else:
                    blob = opts + " " + fact_blob
                    got = _ram_gb(blob)
                    if got is not None and abs(got - self.need.ram) <= 0.2:
                        m = _RAM.search(blob) or _PAIR.search(blob)
                        shown = m.group(0).strip() if m else ""
                if shown:
                    out.append({"field": "ram", "value": shown})
        else:
            if self.need.inch is not None:
                val, is_range = _size_fact(str(facts.get("size") or ""))
                got = None if is_range else val
                if got is None:
                    got = _inch(opts) or _inch(title)
                if got is not None and round(got, 2) == round(self.need.inch, 2):
                    out.append({"field": "screen_size", "value": "%gin" % got})
            if self.need.res:
                observed = str(facts.get("resolution") or facts.get("display") or "").strip()
                if observed:
                    out.append({"field": "resolution", "value": observed})
                else:
                    blob = opts + " " + title
                    marks = list(self.need.res_marks or [])
                    if self.need.res not in marks:
                        marks.append(self.need.res)
                    shown = ""
                    for m in marks:
                        if m and _alias_hit(blob, [m]):
                            shown = m
                            break
                    if shown:
                        out.append({"field": "resolution", "value": shown})
            if self.need.hz_need is not None:
                got = _hz(opts) or _hz(str(facts.get("refresh_rate") or facts.get("display") or ""))
                if got is not None and got + 1e-6 >= self.need.hz_need:
                    out.append({"field": "refresh_rate", "value": "%ghz" % got})
            if self.need.ram is not None:
                got = _ram_gb(opts + " " + fact_blob)
                if got is None:
                    got = _ram_gb(title)
                if got is not None and abs(got - self.need.ram) <= 0.2:
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
        elif (not self.need.explain) and self.shows_brand(pid, sku):
            named = self.need.hard_brand or self.need.soft_brand
            if named:
                out.append({"field": "brand", "value": named})
        shown = (
            (opts + " " + fact_blob).lower()
            if self.need.explain
            else (opts + " " + title + " " + fact_blob).lower()
        )
        if self.need.model and self.need.model.lower() in shown:
            out.append({"field": "model", "value": self.need.model})
        if self.need.color and re.search(
            r"(?<![a-z])" + re.escape(self.need.color) + r"(?![a-z])", shown
        ):
            out.append({"field": "color", "value": self.need.color})
        if self.need.os_need and not self.labels_deny(pid, sku, self.need.os_need):
            fact_os = str(
                facts.get("os") or facts.get("operating_system") or facts.get("phone_type") or ""
            ).strip()
            if fact_os and _has_os(fact_os, self.need.os_need):
                out.append({"field": "os", "value": fact_os})
            else:
                shown_os = ""
                for name in self.need.os_need:
                    if name and re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", shown):
                        shown_os = name
                        break
                if shown_os:
                    out.append({"field": "os", "value": shown_os})
        return out

    def live_need_words(self) -> list:
        kind = self.shop_kind()
        rest = []
        for tok in list(self.need.axis_cues or []) + list(self.need.use_cues or [])[:2]:
            if tok and tok != kind and tok not in rest:
                rest.append(tok)
        return ([kind] if kind else []) + rest

    def wet_title_extra(self) -> str:
        words = self.live_need_words()
        prefix = " ".join(words[:2] if self.need.wet else words[:5])
        if not prefix:
            return ""
        want = set(re.findall(r"[a-z0-9]+", prefix))
        skip = set(self.need.chrome)
        ranked = []
        for pid, sku in self.bag.row:
            if not self.identity_ok(pid, sku) or not self.stock_ok(pid, sku):
                continue
            blob = (
                str(self.bag.live(pid, sku).get("title") or "")
                + " "
                + self.opt_blob(pid, sku)
                + " "
                + self.text_of(pid, sku)
            ).lower()
            whole = (
                _batt_hours(blob),
                _wet(blob, self.need.wet_cues),
                1.0 if _cue_hit(blob, self.need.wet_cues) else 0.0,
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
                    if tok in live or _wet(tok, self.need.wet_cues) > 0 or _cue_hit(tok, self.need.wet_cues):
                        picked = tok
                        break
                    seen.add(tok)
            if not picked:
                continue
            price = _fnum(self.bag.live(pid, sku).get("price")) or 1e18
            hours = whole[0] if whole[0] > 0 else 1e9
            ranked.append(((-compact, hours, price), picked))
        if not ranked:
            return ""
        extra = min(ranked)[1]
        q = "%s %s" % (prefix, extra)
        return q if q.lower() != prefix.lower() else ""

    def searches(self) -> list:
        q = []
        kind = self.shop_kind()
        if self.need.model:
            if self.need.color:
                q.append("%s %s %s" % (self.need.color, self.need.model, kind))
            q.append(("%s %s" % (self.need.model, kind)).strip())
            q.append(self.need.model)
        if self.need.phrase:
            q.append(self.need.phrase)
            extra = self.follow_from_hits()
            if extra and extra.lower() != self.need.phrase.lower():
                q.append(extra)
            out = []
            seen = set()
            for s in q:
                s = " ".join(s.split())
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    out.append(s)
            return out or [self.need.phrase]
        bits = []
        if (self.need.hard_brand or self.need.soft_brand) and not self.need.wet:
            bits.append(self.need.hard_brand or self.need.soft_brand)
        if self.need.wet:
            span = []
            if kind:
                span.append(kind)
            for tok in (self.need.axis_cues or [])[:4]:
                if tok not in span:
                    span.append(tok)
            for tok in (self.need.use_cues or [])[:2]:
                if tok not in span:
                    span.append(tok)
            if span:
                bits.append(" ".join(span))
            elif kind:
                bits.append(kind)
        elif kind:
            bits.append(kind)
        if self.need.inch:
            bits.append("%g inch" % self.need.inch)
        if self.need.res:
            bits.append(self.need.res)
        if self.need.color:
            bits.append(self.need.color)
        if bits:
            q.append(" ".join(str(b) for b in bits if b))
        if self.need.wet:
            extra = self.wet_title_extra()
            if extra:
                q.append(extra)
                tok = extra.split()[-1] if extra.split() else ""
                if tok and tok.lower() not in {x.lower() for x in q}:
                    q.append(tok)
            words = self.live_need_words()
            if self.need.axis_cues and kind:
                q.append(" ".join([kind] + list(self.need.axis_cues)[:4]))
            if words:
                short = " ".join(words[:4])
                if short:
                    q.append(short)
                if self.need.aisle:
                    q.append("%s %s" % (self.need.aisle.split("&")[0].strip(), " ".join(words[:3])))
        if self.need.soft_brand and kind:
            q.append("%s %s" % (self.need.soft_brand, kind))
        if kind and self.need.inch:
            q.append("%g inch %s" % (self.need.inch, kind))
        if self.need.hard_brand and self.need.inch:
            q.append("%s %g inch %s" % (self.need.hard_brand, self.need.inch, kind))
        if self.need.hard_brand and (self.need.kinds or self.need.why):
            q.insert(0, self.need.hard_brand + (" %g inch" % self.need.inch if self.need.inch else "") + (" " + self.need.res if self.need.res else ""))
        if self.bag.dead_title and not self.need.wet:
            dead = self.bag.dead_title.lower()
            keep = (kind and kind in dead) or (self.need.model and self.need.model.lower() in dead)
            if keep:
                q.insert(0, self.bag.dead_title)
                if self.need.inch:
                    q.insert(0, ("%s %g inches %s" % (self.bag.dead_title, self.need.inch, kind)).strip())
        out = []
        seen = set()
        for s in q:
            s = " ".join(s.split())
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out or [(self.need.head or "")[:80]]

    def need_view(self) -> list:
        pids = []
        if self.need.seeds:
            for pid, _sku in self.need.seeds:
                if pid not in self.bag.viewed:
                    pids.append(pid)
            return pids
        for pid, sku in self.bag.row:
            if pid in self.bag.viewed:
                continue
            if (pid, sku) not in self.bag.trusted and not self.need.phrase:
                continue
            if self.need.phrase and not self.phrase_hit(pid, sku):
                continue
            if self.need.budget is not None and not self.need.wet:
                price = _fnum(self.bag.live(pid, sku).get("price"))
                if price is not None and price > self.need.budget * 1.05:
                    continue
            title = str(self.bag.live(pid, sku).get("title") or "")
            if self.need.reject_words and _junk_hit(title, self.need.reject_words):
                continue
            pids.append(pid)
        uniq = []
        seen = set()
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                uniq.append(pid)
        return uniq[:8]

    def cart_pair(self) -> tuple | None:
        if not self.bag.cart:
            return None
        row = self.bag.cart[0]
        return (str(row.get("product_id") or ""), str(row.get("sku") or ""))

    def cart_valid(self) -> bool:
        pair = self.cart_pair()
        if not pair or not pair[0] or not pair[1]:
            return False
        return self.hard_ok(*pair)

    def reply(self, note: str, pair: tuple | None) -> str:
        low = (note or "").lower()
        if pair:
            item = self.bag.live(*pair)
            title = (item.get("title") or pair[0])[:90]
            price = _fnum(item.get("price"))
            cur = str(item.get("currency") or "").strip()
            money = ""
            if price is not None:
                money = "%.2f%s" % (price, (" " + cur) if cur else "")
            if "expensive" in low or "more expensive" in low:
                if price is not None and self.need.budget is not None and price > self.need.budget + 1e-6:
                    return (
                        "Yes, %s is now %s, above the budget. I am going with another "
                        "in-stock option that still meets the firm limits." % (title[:60], money)
                    )
            parts = ["I am going with " + title]
            if self.need.inch is not None:
                parts.append("its listed screen size is %gin" % self.need.inch)
            elif self.need.res:
                parts.append("its listed resolution is " + self.need.res)
            if price is not None:
                parts.append("it is %s and in stock" % money)
            self.bag.named = pair
            return ", ".join(parts) + "."
        return "I am still checking the options against your requirements; nothing is decided yet."

    def order_args(self, pid: str, sku: str) -> dict:
        args = {"product_id": pid, "sku": sku}
        if self.need.explain:
            args["justification"] = {"claims": self.claims(pid, sku)}
        return args

    def filt_args(self) -> dict:
        filt: dict = {"in_stock": True, "k": 10}
        if self.need.aisle:
            filt["category"] = self.need.aisle
        if self.need.hard_brand:
            filt["brand"] = self.need.hard_brand
        elif self.need.soft_brand:
            filt["brand"] = self.need.soft_brand
        return filt

    def plan(self) -> list:
        acts: list = []

        def add(name: str, args: dict) -> None:
            if self.ok(name) and len(acts) < self.max_calls:
                acts.append((name, args))

        if self.need.ask and not self.bag.asked:
            add(
                "message",
                {"content": "Which constraint is firm, and what value should I use?"},
            )
            return acts

        if self.bag.inbox:
            note = self.bag.inbox[-1]
            self.bag.inbox.clear()
            talk = (
                "?" in (note or "")
                or self.need.explain
                or bool(re.search(r"\bwhy\b", note or "", re.I))
            )
            if talk:
                if self.bag.event and (self.need.rank or self.need.model or not self.cart_valid()):
                    pair = self.fallback_pick() or self.cart_pair()
                else:
                    pair = self.cart_pair() or self.fallback_pick()
                add("message", {"content": self.reply(note, pair)})
                if pair and self.ok("inspect_stock"):
                    add("inspect_stock", {"product_id": pair[0], "sku": pair[1]})
                if self.cart_pair() and self.ok("inspect_cart"):
                    add("inspect_cart", {})
                if self.cart_valid() and self.bag.rechecked:
                    return acts

        pair = self.cart_pair()
        if pair and not self.bag.rechecked:
            add("inspect_stock", {"product_id": pair[0], "sku": pair[1]})
            add("inspect_cart", {})
            return acts

        if self.need.rank and self.bag.event and self.need.seeds:
            missing = [ks for ks in self.need.seeds if ks not in self.bag.tick or self.bag.need_rescan]
            if self.bag.need_rescan:
                for pid, sku in self.need.seeds:
                    add("inspect_stock", {"product_id": pid, "sku": sku})
                self.bag.need_rescan = False
                if acts:
                    return acts
            top = self.pick()
            if pair and top and top != pair:
                add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                return acts

        if pair and not self.cart_valid():
            add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
            if pair[0] not in self.bag.viewed and self.ok("view"):
                add("view", {"product_id": pair[0]})
            nxt = self.fallback_pick()
            if (not nxt) and self.bag.named and self.bag.named not in self.bag.ban and self.bag.named != pair:
                nxt = self.bag.named
            if nxt and nxt != pair and nxt not in self.bag.ban:
                add("add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                add("inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
                return acts
            if not self.need.seeds and self.bag.search_n < 8:
                qs = self.searches()
                add("search", {"query": qs[0], "in_stock": True, "k": 10})
                add("filter", self.filt_args())
            return acts

        if pair and self.cart_valid():
            if self.need.rank and self.bag.event:
                top = self.pick()
                if top and top != pair:
                    add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                    add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                    add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                    return acts
            if (self.need.pref or self.need.wet) and self.bag.event:
                top = self.fallback_pick()
                if top and top != pair and self.score(*top) > self.score(*pair):
                    add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                    add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                    add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                    return acts
                extra = [p for p in self.need_view() if p]
                four = any(self.four_axis(*k) for k in self.candidates())
                if (not four) and extra and self.ok("compare") and self.bag.compares < 6:
                    add("compare", {"product_ids": extra[:5]})
                    return acts
                if (not four) and self.bag.search_n < 4:
                    qs = self.searches()
                    idx = min(self.bag.search_n, len(qs) - 1)
                    add("search", {"query": qs[idx], "in_stock": True, "k": 10})
                    add("filter", self.filt_args())
                    return acts
            if self.ok("place_test_order"):
                if self.need.explain and not self.bag.explained and self.ok("message"):
                    self.bag.explained = True
                    add("message", {"content": self.reply("", pair)})
                    return acts
                add("place_test_order", self.order_args(*pair))
                return acts

        if self.bag.event and not self.cart_pair() and self.fallback_pick() is None and self.ok("search") and self.bag.search_n < 8:
            qs = self.searches()
            add("search", {"query": qs[0], "in_stock": True, "k": 10})
            add("filter", self.filt_args())
            return acts

        unseen = self.need_view()
        if unseen and self.ok("compare") and len(unseen) >= 2 and self.bag.compares < 6:
            add("compare", {"product_ids": unseen[:5]})
            return acts
        if unseen and self.ok("view"):
            for pid in unseen[: min(5, self.max_calls)]:
                add("view", {"product_id": pid})
            if acts:
                return acts

        if self.need.seeds and self.need.rank:
            missing = [ks for ks in self.need.seeds if ks not in self.bag.tick]
            if missing:
                for pid, sku in missing[: self.max_calls]:
                    add("inspect_stock", {"product_id": pid, "sku": sku})
                return acts

        need_more = self.bag.search_n < (2 if self.need.phrase or self.need.cover or self.need.pref or self.need.model else 1)
        if self.need.hard_brand and self.pick() is None:
            need_more = self.bag.search_n < 5
        if self.need.cover and self.cover_count() < self.need.cover:
            need_more = True
        if self.need.wet and self.bag.event and self.bag.search_n < 4:
            need_more = True
        if not self.need.seeds and need_more:
            qs = self.searches()
            idx = min(self.bag.search_n, len(qs) - 1)
            add("search", {"query": qs[idx], "in_stock": True, "k": 10})
            add("filter", self.filt_args())
            return acts

        choice = self.fallback_pick()
        if choice:
            pid, sku = choice
            if (
                self.need.explain
                and not self.bag.cart
                and self.ok("compare")
                and self.bag.compares < 1
            ):
                unread = [p for p in self.need_view() if p]
                ids = list(unread)
                if pid not in ids:
                    ids.insert(0, pid)
                for p, _s in self.bag.row:
                    if p not in ids and p in self.bag.viewed:
                        ids.append(p)
                    if len(ids) >= 2:
                        break
                if len(ids) >= 2:
                    add("compare", {"product_ids": ids[: self.max_calls]})
                    return acts
            if (pid, sku) not in self.bag.tick and self.ok("inspect_stock"):
                add("inspect_stock", {"product_id": pid, "sku": sku})
                return acts
            if not self.bag.cart and self.ok("add_to_cart"):
                add("add_to_cart", {"product_id": pid, "sku": sku})
                add("inspect_stock", {"product_id": pid, "sku": sku})
                add("inspect_cart", {})
                return acts

        if not self.need.seeds and self.bag.search_n < 4 and self.fallback_pick() is None:
            qs = self.searches()
            add("search", {"query": qs[self.bag.search_n % len(qs)], "in_stock": True, "k": 10})
            add("filter", self.filt_args())
            return acts
        if self.bag.named and self.bag.named not in self.bag.ban and not self.bag.cart:
            pid, sku = self.bag.named
            add("add_to_cart", {"product_id": pid, "sku": sku})
            add("inspect_stock", {"product_id": pid, "sku": sku})
        return acts

    def close(self) -> bool:
        if self.receipt:
            return True
        late = self.limit.left() < 25 or self.turn >= self.max_steps - 3

        def _order_if_ready() -> bool:
            pair = self.cart_pair()
            if not pair or not pair[0] or not pair[1] or not self.ok("place_test_order"):
                return False
            if self.cart_valid() or late:
                _outs, done = self.fire([("place_test_order", self.order_args(*pair))])
                return done
            return False

        if _order_if_ready():
            return True
        choice = self.stop_pick()
        pair = self.cart_pair()
        if pair and choice and choice != pair and (not self.cart_valid()) and self.ok("remove_from_cart"):
            self.fire([("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})])
        if choice and self.ok("add_to_cart") and not self.bag.cart:
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
                        self.hold = None
                        self.hold_n = 0
                    continue
                continue
            try:
                batch = self.plan()
            except Exception as exc:
                self.talk.append(
                    {"role": "assistant", "content": "plan: %s: %s" % (type(exc).__name__, str(exc)[:120])}
                )
                batch = []
            if not batch:
                if self.bag.cart and self.ok("place_test_order") and self.cart_valid():
                    pair = self.cart_pair()
                    batch = [("place_test_order", self.order_args(*pair))]
                elif (
                    self.ok("search")
                    and self.bag.search_n < 6
                    and self.fallback_pick() is None
                    and self.last_pick() is None
                ):
                    qs = self.searches()
                    batch = [("search", {"query": qs[self.bag.search_n % len(qs)], "in_stock": True, "k": 10})]
                else:
                    choice = self.stop_pick()
                    if choice and choice in self.bag.ban:
                        choice = None
                    if choice and self.ok("add_to_cart") and not self.bag.cart:
                        batch = [("add_to_cart", {"product_id": choice[0], "sku": choice[1]})]
                    elif self.ok("place_test_order") and self.cart_valid():
                        pair = self.cart_pair()
                        batch = [("place_test_order", self.order_args(*pair))]
                    else:
                        late = self.stop_pick()
                        if late and self.ok("add_to_cart") and not self.bag.cart:
                            batch = [("add_to_cart", {"product_id": late[0], "sku": late[1]})]
                        if not batch:
                            before = self.turn
                            try:
                                if self.close():
                                    break
                            except Exception:
                                pass
                            if self.turn >= self.max_steps - 1 or self.turn == before:
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
            if self.cart_valid() and (self.limit.left() < 25 or self.turn >= self.max_steps - 3):
                try:
                    if self.close():
                        break
                except Exception:
                    pass
        if not self.receipt:
            try:
                self.close()
            except Exception:
                pass
        return self.talk


def agent_main(problem_data: dict) -> list:
    try:
        if not isinstance(problem_data, dict) or "environment" not in problem_data:
            return []
        return Shop(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "halt: %s: %s" % (type(exc).__name__, str(exc)[:180])}]
