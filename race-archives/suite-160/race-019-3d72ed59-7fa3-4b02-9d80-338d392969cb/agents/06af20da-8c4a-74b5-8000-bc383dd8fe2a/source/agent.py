from __future__ import annotations

import collections
import csv
import io
import itertools
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient

_local = threading.local()


def proxy() -> Any:
    p = getattr(_local, "p", None)
    if p is None:
        p = _local.p = ProxyClient(timeout=60, max_retries=1)
    return p


class Bucket:

    def __init__(self, rate: float, cap: float) -> None:
        self.rate, self.cap = rate, cap
        self.tokens = 1.0
        self.at = monotonic()
        self.lock = threading.Lock()

    def take(self) -> None:
        while True:
            with self.lock:
                now = monotonic()
                self.tokens = min(self.cap, self.tokens + (now - self.at) * self.rate)
                self.at = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            sleep(min(2.0, wait))


BUCKET = [Bucket(0.9, 5.0)]
POOL_SIZE = 4

TASK_SECONDS = 300.0
GUARD_AT = 238.0
LAST_ENV_AT = 280.0
SETUP_BY = 30.0
ASK_BY = 205.0
ENV_TRIES = 5
ENV_PACE = 0.35
ERR_STREAK = 4
BROWSE_FEW = 4
CALL_TIMEOUT = 25.0

DOWNGRADE_AT = 0.035
CHEAP_ONLY_AT = 0.045
STOP_AT_USD = 0.053

SONNET = "anthropic/claude-sonnet-5.5"
SONNET_OLD = "anthropic/claude-sonnet-4.6"
HAIKU = "anthropic/claude-haiku-5.5"
HAIKU_OLD = "anthropic/claude-haiku-4.5"
MISTRAL = "mistralai/mistral-small-2603"
GEMMA = "google/gemma-4-31b-it"

STRONG = [(SONNET, "low"), (HAIKU, "none"), (MISTRAL, None)]
CHEAP = [(HAIKU, "none"), (MISTRAL, None)]
SENSE_CHAIN = CHEAP
SENSE_SECOND = [(SONNET, "low"), (MISTRAL, None)]
DRIVER = [(HAIKU, "none"), (MISTRAL, None)]
JUDGE_CHAIN = [(HAIKU, "none"), (MISTRAL, None)]
JUDGE_LINE_CAP = 8
JUDGE_TASK_CAP = 25
JUDGE_BATCH = 4
JUDGE_TOKENS = 1500
JUDGE_UNIT_USD = 0.0004
JUDGE_RESERVE_USD = 0.004
JUDGE_LEVELS = ("exact", "close", "no")
AUDIT_CHAIN = [(HAIKU, "low"), (SONNET, "low")]
METRIC_ROWS = 12
AUDIT_ALTS = 5
DECIDE_TOKENS_BASE = 400
DECIDE_TOKENS_LINE = 300
DECIDE_TOKENS_MAX = 2400
SOLVE_ALTS = 5
FIT_ALTS = 5
FIT_POOL = 40
FIT_ROUNDS = 16
FIT_VERIFY_CAP = 10
FIT_VOI = 2
FIT_WIDEN = 2
FIT_CLAIMS_PRICED = False
FIT_EARLY = False
FIT_READ_LEAD = False
FIT_WIDEN_STEPS = 4
FIT_FRONT = 4
FIT_HUB_COMBOS = 400
FIT_SCAN_PAGES = 6
FIT_SCAN_KEEP = 4
FIT_SEEK = 2
FIT_SEEK_STEPS = 8
FIT_HOLD = 5
FIT_HOLD_CAP = 15
FIT_FEEDS = 6
FIT_FRUITLESS = 2
SIEVE_QUERIES = 3
SIEVE_STOP = 3
HOP_TOKENS = 1000
LEAD_ROWS = 1
HOP_FIELDS = ("denies", "pointer_quote", "seller", "value", "mapping_quote")
VERIFY_CAP = 12
CHEAP_POOL = 160
VERIFY_EMPTY = 4.0
WEAK_STEPS = 6
WEAK_JUDGE_EXTRA = 34
WEAK_DIP = 3
LOOK_ALTS = 5
LOOK_TOKENS = 2000
JUDGED = {"exact": "yes", "close": "close", "no": "no"}
OPEN_CHAIN = [(HAIKU, "low"), (GEMMA, None)]
OPEN_TASK_CAP = 20
OPEN_ROWS = 24
OPEN_TEXT = 1200
OPEN_REQUEST = 2500
OPEN_TOKENS = 2400
OPEN_WAIT = 30.0
OPEN_UNIT_USD = 0.0012
OPEN_STOP_USD = 0.032
OPEN_LEVELS = ("likely", "maybe", "unlikely")

PRICES = {
    SONNET: (2.0, 10.0, 0.10, 2.5),
    SONNET_OLD: (3.0, 15.0, 0.30, 3.75),
    HAIKU: (0.10, 0.50, 0.01, 0.125),
    HAIKU_OLD: (1.0, 5.0, 0.10, 1.25),
    MISTRAL: (0.15, 0.60, 0.015, 0.15),
    GEMMA: (0.20, 0.40, 0.02, 0.20),
}
UNKNOWN_PRICE = (2.0, 10.0, 0.2, 2.5)

ID_RX = re.compile(r"^(?=.*\d)[A-Za-z0-9][A-Za-z0-9_\-.:]{3,47}$")


class Text:
    WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
    AMOUNT = re.compile(r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?)\s*([a-zA-Zµ°%\"']{0,12})")
    GROUPED = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
    CODE = re.compile(r"\b([A-Za-z]{2,5})[\s-]?(\d[\dXx]{1,5})\b")
    CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
    KEY_GAP = re.compile(r"[_\-./]+")
    PLURAL_KEEP = ("ss", "us", "is")

    __slots__ = ("raw",)

    def __init__(self, source: Any = "") -> None:
        self.raw = self.of(source)

    @staticmethod
    def of(x: Any) -> str:
        if isinstance(x, str):
            return x
        if x is None:
            return ""
        if isinstance(x, list):
            parts = (p.get("text") if isinstance(p, dict) else p for p in x)
            return " ".join(Text.of(p) for p in parts)
        return str(x)

    @staticmethod
    def json(x: Any, limit: int | None = None) -> str:
        s = json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=str)
        if limit is not None and len(s) > limit:
            return s[:limit] + "…"
        return s

    @staticmethod
    def unfence(s: str) -> str:
        if not s.startswith("```"):
            return s
        s = s.strip("`")
        if "\n" in s:
            s = s[s.find("\n") + 1:]
        return s.rsplit("```", 1)[0]

    @staticmethod
    def bracketed(s: str) -> Any:
        best: tuple[int, Any] | None = None
        for open_, close in (("{", "}"), ("[", "]")):
            i, j = s.find(open_), s.rfind(close)
            if not 0 <= i < j:
                continue
            piece = s[i:j + 1]
            try:
                parsed = json.loads(piece)
            except ValueError:
                continue
            if best is None or len(piece) > best[0]:
                best = (len(piece), parsed)
        return Text.last_json(s) if best is None else best[1]

    @staticmethod
    def last_json(s: str) -> Any:
        dec = json.JSONDecoder()
        found: list[Any] = []
        i, depth, quoted, n = 0, 0, False, len(s)
        while i < n:
            ch = s[i]
            if depth:
                if quoted:
                    if ch == "\\":
                        i += 2
                        continue
                    if ch == '"':
                        quoted = False
                elif ch == '"':
                    quoted = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                i += 1
                continue
            if ch in "{[":
                try:
                    got, j = dec.raw_decode(s, i)
                except ValueError:
                    depth, quoted = 1, False
                    i += 1
                    continue
                if isinstance(got, (dict, list)) and got:
                    found.append(got)
                i = max(j, i + 1)
                continue
            i += 1
        same = [x for x in found if found and type(x) is type(found[0])]
        return same[-1] if same else None

    @staticmethod
    def loads(raw: Any) -> Any:
        depth = 0
        while isinstance(raw, str) and depth < 3:
            depth += 1
            s = Text.unfence(raw.strip())
            try:
                raw = json.loads(s)
            except ValueError:
                return Text.bracketed(s)
        return raw

    @property
    def words(self) -> list[str]:
        return self.WORD.findall(self.raw.lower())

    @staticmethod
    def stem(w: str) -> str:
        n = len(w)
        if n > 4 and w[-3:] == "ies":
            return w[:-3] + "y"
        if n > 4 and w[-2:] == "es" and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh")):
            return w[:-2]
        if n > 3 and w[-1] == "s" and w[-2] != "s":
            return w[:-1]
        return w

    @property
    def stems(self) -> list[str]:
        return [self.stem(w) for w in self.words]

    @property
    def singular(self) -> str:
        out = []
        for w in self.words:
            if len(w) >= 7 and w.endswith("ies"):
                out.append(w[:-3] + "y")
                continue
            if len(w) > 4 and w[-2:] == "es" and (w[-4:-2] in ("ss", "ch", "sh") or w[-3] in "xz"):
                out.append(w[:-2])
                continue
            plural = len(w) > 3 and w[-1] == "s" and not w.endswith(self.PLURAL_KEEP)
            out.append(w[:-1] if plural else w)
        return " ".join(out)

    @staticmethod
    def key(k: Any) -> str:
        spaced = Text.CAMEL.sub(" ", Text.of(k))
        return " ".join(Text.KEY_GAP.sub(" ", spaced).lower().split())

    @staticmethod
    def key_stems(k: Any) -> list[str]:
        return Text(Text.key(k)).stems

    @staticmethod
    def alike(a: str, b: str) -> bool:
        if a == b:
            return True
        if min(len(a), len(b)) < 5:
            return False
        k = max(5, len(b) - 2)
        return a[:k] == b[:k]

    @staticmethod
    def run_in(needle: list[str], hay: list[str]) -> bool:
        n = len(needle)
        if n == 0:
            return False
        for i in range(len(hay) - n + 1):
            if all(Text.alike(hay[i + j], needle[j]) for j in range(n)):
                return True
        return False

    def has(self, needle: Any) -> bool:
        return self.run_in(Text(needle).stems, self.stems)

    @property
    def numbers(self) -> list[tuple[float, str]]:
        out = []
        for m in self.AMOUNT.finditer(self.raw):
            digits = m.group(1)
            digits = digits.replace(",", "") if self.GROUPED.fullmatch(digits) else digits.replace(",", ".")
            try:
                out.append((float(digits), m.group(2)))
            except ValueError:
                continue
        return out

    @staticmethod
    def unit_scale(u1: str, u2: str) -> float | None:
        a, b = u1.strip().strip(".\"'"), u2.strip().strip(".\"'")
        if not a or not b:
            return 1.0
        la, lb = a.lower(), b.lower()
        if la == lb:
            return 1.0
        la, lb = la.rstrip("s"), lb.rstrip("s")
        if la == lb:
            return 1.0
        if min(len(la), len(lb)) >= 2 and (la.startswith(lb) or lb.startswith(la)):
            return 1.0
        short, long_ = sorted((la, lb), key=len)
        if short.isalpha() and long_.isalpha() and len(short) >= 2 and len(long_) >= len(short) + 3 and short[0] == long_[0]:
            rest = iter(long_[1:])
            if all(ch in rest for ch in short[1:]):
                return 1.0
        return None

    @property
    def codes(self) -> list[tuple[str, str]]:
        return [(m.group(1).lower(), m.group(2).lower()) for m in self.CODE.finditer(self.raw)]

    @staticmethod
    def code_like(pattern: tuple[str, str], code: tuple[str, str]) -> bool:
        if pattern[0] != code[0] or len(pattern[1]) != len(code[1]):
            return False
        return all(p in ("x", c) for p, c in zip(pattern[1], code[1]))

    def code_fits(self, wanted: str) -> str:
        want = Text(wanted).codes
        have = self.codes
        if not want or not have:
            return "unknown"
        banned = [c for x in sorted(fit_info(self.raw, "dep").exc) for c in Text(x).codes]
        for w in want:
            if any(self.code_like(e, w) for e in banned):
                return "no"
            if any(self.code_like(h, w) for h in have):
                return "yes"
        return "unknown"


class Reply:
    __slots__ = ("text", "calls", "ok")

    def __init__(self, text: str = "", calls: list | None = None, ok: bool = False) -> None:
        self.text = text
        self.calls = calls or []
        self.ok = ok


class Wire:

    @staticmethod
    def post(path: str, body: dict, timeout: float) -> tuple[Any, int | None, str]:
        BUCKET[0].take()
        px = proxy()
        px.timeout = max(3.0, timeout)
        verbose = getattr(px, "post_verbose", None)
        if verbose is None:
            return px.post(path, json_data=body), None, ""
        res = verbose(path, json_data=body)
        err = getattr(res, "error", None)
        if not isinstance(err, dict):
            return getattr(res, "data", None), None, ""
        return getattr(res, "data", None), err.get("status"), Text.of(err.get("body"))[:300]

    @staticmethod
    def cost(model: str, usage: Any, prompt_chars: int) -> float:
        rate = PRICES.get(model, UNKNOWN_PRICE)
        if isinstance(usage, dict):
            billed = usage.get("cost")
            if isinstance(billed, (int, float)) and not isinstance(billed, bool) and billed > 0:
                up = (usage.get("cost_details") or {}).get("upstream_inference_cost") if isinstance(usage.get("cost_details"), dict) else None
                return float(billed) + (float(up) if isinstance(up, (int, float)) and usage.get("is_byok") else 0.0)
        if not isinstance(usage, dict):
            return (prompt_chars / 3.5 * rate[0] + 600 * rate[1]) / 1e6
        detail = usage.get("prompt_tokens_details")
        detail = detail if isinstance(detail, dict) else {}
        cached = float(detail.get("cached_tokens") or 0)
        written = float(detail.get("cache_write_tokens") or 0)
        fresh = max(0.0, float(usage.get("prompt_tokens") or 0) - cached - written)
        parts = (fresh, float(usage.get("completion_tokens") or 0), cached, written)
        return sum(n * p for n, p in zip(parts, rate)) / 1e6


class NoTime(Exception):
    pass


class StoreBroken(Exception):
    pass


def crowd_wait(sid: str) -> float:
    tail = "".join(ch for ch in str(sid).lower() if ch.isalnum())[-2:] or "0"
    try:
        n = int(tail, 36)
    except ValueError:
        n = sum(ord(ch) for ch in tail)
    return 1.0 + (n % 100) / 100.0


class Call:
    STRONG_ROLES = ("setup", "rank", "decide")
    CHEAP_OK = ("read", "label", "drive", "decide")

    def __init__(self, run: Any, role: str, system: str, user: str, *, max_tokens: int = 1200, deadline: float | None = None,
                 history: list | None = None, tools: list | None = None, chain: list | None = None) -> None:
        self.r = run
        self.role = role
        self.system = system
        self.user = user
        self.max_tokens = max_tokens
        self.history = list(history or [])
        self.tools = tools
        self.chain = chain
        limit = TASK_SECONDS - 12
        self.end = run.began + min(limit, limit if deadline is None else deadline)

    @property
    def strong(self) -> bool:
        return self.role in self.STRONG_ROLES

    def ladder(self) -> list:
        r = self.r
        if r.exhausted or r.spent >= STOP_AT_USD:
            return []
        if r.spent >= CHEAP_ONLY_AT and self.role not in self.CHEAP_OK:
            return []
        if self.strong and r.spent >= DOWNGRADE_AT:
            return CHEAP
        if self.chain:
            return self.chain
        return DRIVER if self.role == "drive" else (STRONG if self.strong else CHEAP)

    def messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}] + self.history
        if self.user:
            msgs.append({"role": "user", "content": self.user})
        return msgs

    def body(self, model: str, effort: str | None, msgs: list[dict]) -> dict:
        body: dict[str, Any] = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": self.max_tokens,
                                "session_id": self.r.sid, "usage": {"include": True}}
        if effort:
            body["reasoning"] = {"effort": effort}
        head = msgs[0].get("content")
        if model.startswith("anthropic/") and isinstance(head, str):
            cached = {"type": "text", "text": head, "cache_control": {"type": "ephemeral"}}
            body["messages"] = [{"role": "system", "content": [cached]}] + msgs[1:]
        if self.tools:
            body["tools"] = self.tools
            body["tool_choice"] = "auto"
        return body

    def left(self) -> float:
        left = self.end - monotonic()
        if left < 4:
            raise NoTime()
        return left

    def transmit(self, body: dict) -> tuple[Any, int | None, str]:
        cap = 30.0 if self.role == "drive" else CALL_TIMEOUT
        for attempt in range(3):
            left = self.left()
            data, status, err = Wire.post("/inference/chat/completions", body, min(cap, left - 2))
            if status != 429:
                return data, status, err
            sleep(min(crowd_wait(self.r.sid + str(attempt)) * (attempt + 1), max(0.3, left - 4)))
        return None, 429, "rate limited"

    def accept(self, model: str, data: dict, chars: int) -> Reply:
        r = self.r
        with r.lock:
            r.calls += 1
            r.spent += Wire.cost(model, data.get("usage"), chars)
        msg = (data["choices"][0] or {}).get("message") or {}
        calls = msg.get("tool_calls")
        return Reply(Text.of(msg.get("content")), calls if isinstance(calls, list) else [], True)

    def send(self) -> Reply:
        ladder = self.ladder()
        if not ladder:
            return Reply()
        msgs = self.messages()
        chars = sum(len(Text.of(m.get("content"))) for m in msgs)
        try:
            for round_ in range(2):
                for model, effort in ladder:
                    self.left()
                    data, status, err = self.transmit(self.body(model, effort, msgs))
                    if isinstance(data, dict) and data.get("choices"):
                        return self.accept(model, data, chars)
                    if status in (402, 403) and re.search(r"limit|credit", err or "", re.I):
                        self.r.exhausted = True
                        return Reply()
                    if status is None or (isinstance(status, int) and status >= 500):
                        with self.r.lock:
                            self.r.spent += Wire.cost(model, None, chars) * 0.2
                sleep(min(2.0, 0.5 * (round_ + 1)))
        except NoTime:
            pass
        return Reply()

    def json(self) -> Any:
        reply = self.send()
        got = Text.loads(reply.text) if reply.ok else None
        if got is None and reply.ok and not self.chain:
            self.chain = (STRONG if self.strong else CHEAP)[1:]
            reply = self.send()
            got = Text.loads(reply.text) if reply.ok else None
        return got


REC_KEY_RX2 = re.compile(r"(?:^|(?<=[^A-Za-z0-9]))([A-Za-z][A-Za-z0-9_\- ]{0,38}[A-Za-z0-9_]|[A-Za-z])[ \t]?:[ \t]?(?=\S|$)")
REC_PAIR_RX2 = re.compile(r"^([^=;:]{1,60}?)\s*[=:]\s*(.*)$", re.S)


def rec_scalar(v: str) -> Any:
    t = v.strip()
    if t in ("True", "False", "true", "false"):
        return t.lower() == "true"
    if t in ("None", "null"):
        return None
    if re.fullmatch(r"-?\d+", t) and len(t) < 18:
        return int(t)
    if re.fullmatch(r"-?\d+\.\d+", t):
        return float(t)
    m = rec_money(t)
    return t if m is None else m


REC_MONEY_RX = re.compile(r"^\s*(?:[^\w\s.,\-]{1,3}\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?[^\w\s.,\-]{1,3}|(-?\d+\.\d+))\s*$")


def rec_money(v: str) -> float | None:
    m = REC_MONEY_RX.match(v) if len(v) <= 24 else None
    if not m:
        return None
    t = (m.group(1) or m.group(2) or m.group(3)).replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def rec_as_json(s: str) -> Any:
    t = s.strip()
    if len(t) < 2 or t[0] not in "[{" or t[-1] not in "]}":
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def rec_keyish(k: str) -> bool:
    k = k.strip()
    return 0 < len(k) <= 40 and len(k.split()) <= 4 and not re.fullmatch(r"[\d\W_]+", k)


def rec_md_table(s: str) -> list | None:
    lines = [l.strip() for l in s.strip().split("\n") if l.strip()]
    if len(lines) < 2 or not lines[0].startswith("|") or not re.fullmatch(r"\|?[\s:\-|]+\|?", lines[1]) or "-" not in lines[1]:
        return None
    cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
    head = cells(lines[0])
    if len(head) < 2 or not all(rec_keyish(h) for h in head):
        return None
    out = []
    for l in lines[2:]:
        c = cells(l)
        if len(c) != len(head):
            return None
        out.append({h: rec_nested(x) for h, x in zip(head, c)})
    return out


def rec_delim_table(s: str) -> list | None:
    t = s.strip("\n")
    if "\n" not in t:
        return None
    first = t.split("\n", 1)[0]
    marks = collections.Counter(ch for ch in first if not ch.isalnum() and ch not in " _-'\"")
    for d, _ in marks.most_common(3):
        try:
            rows = list(csv.reader(io.StringIO(t), delimiter=d))
        except Exception:
            continue
        rows = [r for r in rows if r]
        if len(rows) < 2 or len(rows[0]) < 3 or not all(rec_keyish(h) for h in rows[0]):
            continue
        if any(len(r) != len(rows[0]) for r in rows[1:]):
            continue
        return [{h.strip(): rec_nested(x) for h, x in zip(rows[0], r)} for r in rows[1:]]
    return None


def rec_kv_record(s: str, allowed: set | None = None) -> dict | None:
    hits = list(REC_KEY_RX2.finditer(s))
    if len(hits) < 3 or hits[0].start() != 0:
        return None
    lead = hits[0].group(1)[:1].isupper()
    seps = collections.Counter()
    for h in hits[1:]:
        j = h.start()
        i = j
        while i > 0 and not s[i - 1].isalnum() and s[i - 1] not in ".)]\"'":
            i -= 1
        if i < j:
            seps[s[i:j]] += 1
    if not seps:
        return None
    best = None
    for sep, n in seps.most_common(4):
        if not sep.strip() and "\n" not in sep and "\t" not in sep:
            continue
        rec, last = {}, None
        for part in s.split(sep):
            m = REC_KEY_RX2.match(part)
            if (m and m.start() == 0 and m.group(1) not in rec and rec_keyish(m.group(1)) and m.group(1)[:1].isupper() == lead
                    and (allowed is None or m.group(1) in allowed)):
                last = m.group(1)
                rec[last] = part[m.end():]
            elif last is not None:
                rec[last] += sep + part
            else:
                rec = {}
                break
        if len(rec) >= 3 and (best is None or len(rec) > len(best)):
            best = rec
    if best is None:
        return None
    return {k: rec_nested(v) for k, v in best.items()}


def rec_nested(v: str) -> Any:
    t = v.strip()
    j = rec_as_json(t)
    if j is not None:
        return rec_pairs_from(j)
    for fs in (";", "\n"):
        if fs in t and ("=" in t or ":" in t):
            parts = [p.strip() for p in t.split(fs) if p.strip()]
            d, last, n = {}, None, 0
            for p in parts:
                m = REC_PAIR_RX2.match(p)
                if m and m.group(1).strip() not in d and rec_keyish(m.group(1)):
                    last = m.group(1).strip()
                    d[last] = m.group(2).strip()
                    n += 1
                elif last is not None:
                    d[last] += fs + " " + p
                else:
                    d = None
                    break
            if d and n >= 2 and n * 2 >= len(parts):
                return {k: rec_scalar(x) for k, x in d.items()}
    if t.count("=") == 1 and ";" not in t:
        m = REC_PAIR_RX2.match(t)
        if m and rec_keyish(m.group(1)) and m.group(2).strip():
            return {m.group(1).strip(): rec_scalar(m.group(2))}
    return rec_scalar(t)


def rec_pairs_from(j: Any) -> Any:
    if isinstance(j, list) and j and all(isinstance(x, str) for x in j):
        d = {}
        for x in j:
            m = REC_PAIR_RX2.match(x)
            if not m or not rec_keyish(m.group(1)):
                return j
            d[m.group(1).strip()] = rec_scalar(m.group(2))
        return d
    if isinstance(j, list) and j and all(isinstance(x, dict) and len(x) == 2 for x in j):
        ks = [list(x) for x in j]
        if all(k == ks[0] for k in ks):
            a, b = ks[0]
            if all(isinstance(x[a], str) for x in j):
                return {x[a]: x[b] for x in j}
    return j


def rec_text_value(s: str) -> Any:
    if len(s) > 200000:
        return None
    j = rec_as_json(s)
    if j is not None:
        return rec_decode(j, True)
    t = rec_md_table(s)
    if t is not None:
        return t
    t = rec_delim_table(s)
    if t is not None:
        return t
    return rec_kv_record(s)


def rec_table_like(s: str) -> list | None:
    j = rec_as_json(s)
    if isinstance(j, list):
        return rec_decode(j, True)
    t = rec_md_table(s)
    if t is None:
        t = rec_delim_table(s)
    if t is not None and len(t) >= 1 and len(t[0]) >= 3:
        return t
    return None


def rec_page_facts(rec: dict, row: bool = False) -> dict:
    if any(isinstance(v, dict) for v in rec.values()):
        return rec
    out = dict(rec)
    longest = max((len(v) for v in rec.values() if isinstance(v, str)), default=0)
    for k, v in rec.items():
        if isinstance(v, str) and "\n" in v.strip() and not (row and len(v) >= longest):
            lines = [l for l in v.strip().split("\n") if l.strip()]
            if len(lines) >= 2 and all(REC_PAIR_RX2.match(l) and rec_keyish(REC_PAIR_RX2.match(l).group(1)) for l in lines):
                d = rec_nested(v.replace("\n", ";")) if ";" not in v else None
                if isinstance(d, dict) and len(d) == len(lines):
                    out[k] = d
    return out


def rec_idlike(k: Any) -> bool:
    return isinstance(k, str) and 0 < len(k) <= 64 and not any(c.isspace() for c in k) and any(c.isdigit() for c in k)


def rec_keyed(value: dict) -> list | None:
    vals = list(value.values())
    if not vals or not all(isinstance(v, dict) and len(v) >= 2 for v in vals) or not all(rec_idlike(k) for k in value):
        return None
    common = set(vals[0])
    for v in vals[1:]:
        common &= set(v)
    if len(common) < 2 and len(vals) > 1:
        return None
    name = next(n for n in ("id", "_id", "__id") if all(n not in v for v in vals))
    return [{name: k, **v} for k, v in value.items()]


def rec_columns(value: dict) -> dict | None:
    for ck, cols in value.items():
        if not (isinstance(cols, list) and len(cols) >= 2 and all(isinstance(c, str) and rec_keyish(c) for c in cols)):
            continue
        for rk, rows in value.items():
            if rk != ck and isinstance(rows, list) and all(isinstance(r, list) and len(r) == len(cols) for r in rows):
                out = {k: v for k, v in value.items() if k != ck}
                out[rk] = [dict(zip(cols, r)) for r in rows]
                return out
    return None


def rec_decode(value: Any, top: bool = True, root: bool = True) -> Any:
    if root and isinstance(value, dict) and len(value) == 1:
        inner = next(iter(value.values()))
        while isinstance(inner, dict) and len(inner) == 1 and isinstance(next(iter(inner.values())), dict):
            inner = next(iter(inner.values()))
        if isinstance(inner, dict) and len(inner) >= 3 and inner is not next(iter(value.values())):
            return rec_decode(inner, True, True)
    if root and isinstance(value, dict) and len(value) in (2, 3):
        boxes = [k for k, v in value.items() if isinstance(v, dict) and len(v) >= 2 and rec_keyed(v) is None and rec_columns(v) is None]
        rest = [v for k, v in value.items() if k not in boxes]
        if len(boxes) == 1 and all(isinstance(v, (str, int, float, bool)) and len(str(v)) <= 24 and not rec_idlike(v) for v in rest):
            lifted = {k: v for k, v in value.items() if k not in boxes}
            lifted.update(value[boxes[0]])
            return rec_decode(lifted, True, True)
    if isinstance(value, dict) and top:
        keyed = rec_keyed(value)
        if keyed is not None:
            return {"rows": keyed} if root else keyed
        cols = rec_columns(value)
        if cols is not None:
            value = cols
    if isinstance(value, str):
        r = rec_text_value(value) if top else None
        return value if r is None else r
    if isinstance(value, list):
        out = [rec_decode(x, top, False) for x in value]
        texts = [(i, x) for i, x in enumerate(value) if isinstance(x, str) and isinstance(out[i], dict)]
        if top and len(texts) >= 3:
            seen = collections.Counter(k for i, _ in texts for k in out[i])
            allowed = {k for k, n in seen.items() if n >= 0.3 * len(texts)}
            for i, x in texts:
                if set(out[i]) - allowed:
                    r = rec_kv_record(x, allowed)
                    if r is not None:
                        out[i] = r
        return out
    if isinstance(value, dict):
        if top and len(value) == 1:
            only = next(iter(value.values()))
            if isinstance(only, str):
                r = rec_text_value(only)
                if isinstance(r, dict):
                    return r
            if root and isinstance(only, dict) and len(only) >= 3:
                return {next(iter(value)): rec_page_facts(only)}
        wrapper = top and len(value) <= 2
        out = {}
        for k, v in value.items():
            if isinstance(v, str):
                if wrapper:
                    r = rec_text_value(v)
                elif top and "\n" in v:
                    r = rec_table_like(v)
                else:
                    r = None
                if r is None and root is not None:
                    r = rec_money(v)
                out[k] = v if r is None else r
            elif isinstance(v, dict) and not wrapper and not any(isinstance(x, (list, dict)) for x in v.values()):
                out[k] = v
            elif isinstance(v, dict) and top and (rec_keyed(v) is not None or rec_columns(v) is not None):
                out[k] = rec_decode(v, True, False)
            else:
                out[k] = rec_decode(v, wrapper or isinstance(v, list) and top, False)
        if root and len(out) >= 3 and not any(isinstance(v, list) for v in out.values()):
            out = rec_page_facts(out)
        return out
    return value


def money_of(v: Any) -> Any:
    if isinstance(v, dict) and 2 <= len(v) <= 3:
        nums = [x for x in v.values() if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if len(nums) == 1 and all(isinstance(x, str) and len(x) <= 12 and " " not in x for x in v.values() if x is not nums[0]):
            return nums[0]
    return v


def records_in(value: Any) -> Any:
    return rec_decode(value)


ROWS_SYS = """A store tool answered a shopping agent with listings written as free text that no parser could split into records. Copy every listing in TEXTS into one JSON record. Answer: a single JSON object, no prose around it:
{"rows": [{"id": "...", "seller": "...", "title": "...", "brand": "...", "price": 0, "stock": true, "category": "...", "about": "...", "notes": "...", "facts": {"property": "value"}}]}
- One record per listing, in the order the text gives them. A listing only mentioned in passing inside another listing's text is not a listing.
- id: the code the store uses to name that listing, copied character for character.
- price: the listing's price as the bare number written in the text, without currency signs, codes or units.
- stock: true when the text says it can be bought now, false when it says it cannot, null when unsaid.
- seller: who sells it. brand: who makes it. category: the shelf or section it sits in. about: its description text. notes: remarks from the shop. facts: every stated property name with its value.
- Copy every text value verbatim from TEXTS. Leave out any field the text does not give. Never invent a listing or a value.
- No listing in TEXTS: answer {"rows": []}."""
ROWS_ROLES = ("search", "filter", "detail", "specs", "compare")
ROW_FIELDS = ("id", "seller", "title", "brand", "price", "stock", "category", "about", "notes", "facts")
ROW_TEXT_FIELDS = ("id", "seller", "title", "brand", "category", "about", "notes")
ROWS_MIN_TEXT = 120
ROWS_TOKENS = 6000
ROWS_CHUNK = 3500
ROWS_WAIT = 30.0
ROWS_SPARE = 45.0
ROWS_CHARS = 16000
ROWS_SHAPES = 24
ROWS_LABEL_RX = re.compile(r"[.!?;]\s+[A-Z][A-Za-z]*(?: [A-Za-z]+){0,3}: ")
ROWS_EDGE = r"(?<![A-Za-z0-9])"
ROWS_EDGE_END = r"(?![A-Za-z0-9])"


class RowReader:
    SLOT_OF = {"id": "id", "seller": "seller", "title": "title", "brand": "maker", "price": "price", "stock": "stock",
               "category": "category", "about": "desc", "notes": "notes", "facts": "attrs"}

    def __init__(self, run: Any) -> None:
        self.r = run
        self.done: dict[str, list] = {}
        self.shapes: list[dict] = []
        self.texts: set[str] = set()
        self.asked = 0
        self.shaped = 0

    @staticmethod
    def word_set(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    @staticmethod
    def said(v: str, words: set[str]) -> bool:
        ws = re.findall(r"[a-z0-9]+", v.lower())
        return bool(ws) and sum(1 for w in ws if w in words) >= 0.8 * len(ws)

    @staticmethod
    def bare(tok: str) -> str:
        return tok.strip("()[]{}<>,.;:!?\"'`")

    @staticmethod
    def idlike(tok: str) -> bool:
        t = RowReader.bare(tok)
        if not ID_RX.match(t) or not any(c.isdigit() for c in t):
            return False
        if re.fullmatch(r"[\d.,\-]+", t):
            return len(t) >= 6 and t.isdigit()
        return any(c.isalpha() for c in t) and len(t) >= 5

    @staticmethod
    def found(needle: str, hay: str) -> bool:
        return bool(needle) and re.search(ROWS_EDGE + re.escape(needle) + ROWS_EDGE_END, hay) is not None

    @staticmethod
    def rid_of(d: dict, id_key: str | None, skip: set[str]) -> str | None:
        named = d.get(id_key) if id_key else None
        if isinstance(named, (str, int)) and not isinstance(named, bool) and str(named) not in skip:
            return str(named)
        for v in d.values():
            if isinstance(v, str) and v not in skip and " " not in v and not v.isdigit() and ID_RX.match(v):
                return v
            if isinstance(v, int) and not isinstance(v, bool) and v >= 10 ** 6 and str(v) not in skip:
                return str(v)
        return None

    @staticmethod
    def has_record(x: Any, id_key: str | None, skip: set[str], depth: int = 0) -> bool:
        if depth > 4:
            return False
        if isinstance(x, list):
            return any(RowReader.has_record(v, id_key, skip, depth + 1) for v in x[:80])
        if not isinstance(x, dict):
            return False
        flat = sum(1 for v in x.values() if v is None or isinstance(v, (str, int, float, bool)))
        if flat >= 4 or (len(x) >= 3 and RowReader.rid_of(x, id_key, skip)):
            return True
        return any(RowReader.has_record(v, id_key, skip, depth + 1) for v in x.values() if isinstance(v, (dict, list)))

    @staticmethod
    def pieces(x: Any, path: tuple = (), depth: int = 0) -> list[tuple[tuple, list[str]]]:
        out: list[tuple[tuple, list[str]]] = []
        if depth > 4:
            return out
        if isinstance(x, str):
            if len(x) >= 80:
                out.append((path, [x]))
        elif isinstance(x, list):
            if x and all(isinstance(s, str) for s in x):
                if any(len(s) >= 40 for s in x):
                    out.append((path, list(x)))
            else:
                for i, v in enumerate(x[:80]):
                    out += RowReader.pieces(v, path + (i,), depth + 1)
        elif isinstance(x, dict):
            for k, v in x.items():
                out += RowReader.pieces(v, path + (k,), depth + 1)
        return out

    @staticmethod
    def signal(role: str | None, args: dict, dec: Any, id_key: str | None, known: set[str], total_key: str | None = None) -> tuple[tuple, list[str]] | None:
        if role not in ROWS_ROLES:
            return None
        passed = {str(v) for val in (args or {}).values() for v in (val if isinstance(val, list) else [val])
                  if isinstance(v, (str, int)) and not isinstance(v, bool) and len(str(v)) >= 3}
        listing = role in ("search", "filter")
        if RowReader.has_record(dec, id_key, passed if listing else set()):
            return None
        if total_key and isinstance(dec, dict) and dec.get(total_key) == 0:
            return None
        names = known | (set() if listing else passed)
        best, top = None, None
        for path, texts in RowReader.pieces(dec):
            joined = "\n".join(texts)
            if len(joined) < ROWS_MIN_TEXT:
                continue
            toks = {RowReader.bare(w) for w in joined.split()}
            named = len(toks & names)
            spread = sum(1 for t in texts if any(RowReader.idlike(w) for w in t.split()))
            if not named and (not listing or not spread or spread * 2 < len(texts)):
                continue
            score = (named, spread, len(joined))
            if top is None or score > top:
                best, top = (path, texts), score
        return best

    @staticmethod
    def put(x: Any, path: tuple, new: Any) -> Any:
        if not path:
            return new
        k = path[0]
        if isinstance(x, dict) and k in x:
            y = dict(x)
            y[k] = RowReader.put(x[k], path[1:], new)
            return y
        if isinstance(x, list) and isinstance(k, int) and k < len(x):
            y = list(x)
            y[k] = RowReader.put(x[k], path[1:], new)
            return y
        return x

    @staticmethod
    def place(dec: Any, path: tuple, rows: list[dict], role: str | None) -> Any:
        one = len(rows) == 1 and role in ("detail", "specs")
        if not path:
            return rows[0] if one else {"rows": rows}
        if one and len(path) == 1 and isinstance(dec, dict) and len(dec) == 1:
            return rows[0]
        return RowReader.put(dec, path, rows[0] if one and isinstance(RowReader.at(dec, path), str) else rows)

    @staticmethod
    def at(x: Any, path: tuple) -> Any:
        for k in path:
            x = x[k]
        return x

    def named(self, rows: list[dict]) -> list[dict]:
        st = self.r.store
        ren: dict[str, str] = {}
        for k, slot in self.SLOT_OF.items():
            t = getattr(st, slot, None)
            if isinstance(t, str) and t and t != k and t not in ROW_FIELDS and t in st.row_keys:
                ren[k] = t
        if not ren or len(set(ren.values())) < len(ren):
            return rows
        return [{ren.get(k, k): v for k, v in x.items()} for x in rows]

    @staticmethod
    def number(v: Any) -> float | int | None:
        if isinstance(v, bool) or v is None or v == "":
            return None
        n = to_number(v)
        if n is None:
            return None
        return int(n) if float(n).is_integer() else n

    def segments(self, ids: list[str], texts: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for t in texts:
            spots = []
            for rid in ids:
                m = re.search(ROWS_EDGE + re.escape(rid) + ROWS_EDGE_END, t)
                if m and rid not in out:
                    spots.append((m.start(), m.end(), rid))
            spots.sort()
            for j, (a, b, rid) in enumerate(spots):
                lo = spots[j - 1][1] if j else 0
                hi = spots[j + 1][0] if j + 1 < len(spots) else len(t)
                out[rid] = t[lo:hi]
        return out

    def verified(self, d: Any, texts: list[str]) -> list[dict]:
        rows_in = d.get("rows") if isinstance(d, dict) else d
        if not isinstance(rows_in, list):
            return []
        raw = "\n".join(texts)
        cands: list[tuple[str, dict]] = []
        for x in rows_in[:80]:
            if not isinstance(x, dict):
                continue
            rid = x.get("id")
            rid = str(rid).strip() if isinstance(rid, (str, int)) and not isinstance(rid, bool) else ""
            if not rid or rid in {c for c, _ in cands} or len(rid) > 64 or any(c.isspace() for c in rid) or not self.found(rid, raw):
                continue
            cands.append((rid, x))
        segs = self.segments([c for c, _ in cands], texts)
        out: list[dict] = []
        for rid, x in cands:
            seg = segs.get(rid, raw)
            nums = numbers_in(seg)
            words = self.word_set(seg)
            row: dict | None = {"id": rid}
            for k in ROW_FIELDS[1:]:
                v = x.get(k)
                if k == "price":
                    if v is None or v == "":
                        continue
                    n = self.number(v)
                    if n is None or not any(abs(float(n) - m) < 1e-6 for m in nums):
                        row = None
                        break
                    row[k] = n
                elif k == "stock":
                    if isinstance(v, bool):
                        row[k] = v
                elif k == "facts":
                    if isinstance(v, list):
                        v = {a: b for e in v if isinstance(e, dict) and len(e) == 2 for a, b in [tuple(e.values())] if isinstance(a, str)}
                    if isinstance(v, dict):
                        fx = {self.restore(str(a).strip(), seg): rec_scalar(self.restore(str(b).strip(), seg)) for a, b in v.items()
                              if isinstance(b, (str, int, float)) and not isinstance(b, bool) and str(a).strip()
                              and self.said(str(a), words) and (not str(b).strip() or self.said(str(b), words))}
                        if fx:
                            row[k] = fx
                elif isinstance(v, str) and v.strip() and self.said(v, words):
                    row[k] = self.restore(v.strip(), seg)
            if row is None or not ("title" in row or "price" in row):
                continue
            out.append(row)
        return out

    @staticmethod
    def restore(v: str, raw: str) -> str:
        if not v or v in raw:
            return v
        toks = re.findall(r"[A-Za-z0-9]+", v)
        if not toks or len(toks) > 400:
            return v
        rx = ROWS_EDGE + r"[^A-Za-z0-9]{0,4}".join(re.escape(w) for w in toks) + ROWS_EDGE_END
        found = [m.group(0) for m in re.finditer(rx, raw)]
        return found[0] if found and len(set(found)) == 1 else v

    @staticmethod
    def locate(t: str, v: str, free: Callable[[int, int], bool]) -> int | None:
        edge = (ROWS_EDGE if v[:1].isalnum() else "") + re.escape(v) + (ROWS_EDGE_END if v[-1:].isalnum() else "")
        return next((m.start() for m in re.finditer(edge, t) if free(m.start(), m.end())), None)

    @staticmethod
    def facts_parse(v: str, join: str, sep: str) -> dict | None:
        fx: dict[str, Any] = {}
        for part in (v.split(sep) if sep else [v]):
            if join not in part:
                return None
            a, b = part.split(join, 1)
            if not a.strip():
                return None
            fx[a.strip()] = rec_scalar(b.strip())
        return fx

    @staticmethod
    def facts_span(t: str, facts: dict) -> tuple[int, int, str, str] | None:
        items = [(str(k), str(v)) for k, v in facts.items()]
        if not items:
            return None
        want = {k: rec_scalar(v) for k, v in items}
        for m in re.finditer(re.escape(items[0][0]), t):
            pos, spans = m.start(), []
            for k, v in items:
                kp = t.find(k, pos)
                vp = t.find(v, kp + len(k)) if kp >= 0 else -1
                if vp < 0:
                    spans = []
                    break
                spans.append((kp, kp + len(k), vp, vp + len(v)))
                pos = vp + len(v)
            if not spans or spans[0][0] != m.start():
                continue
            join = t[spans[0][1]:spans[0][2]]
            if not join or len(join) > 6:
                continue
            gaps = collections.Counter(t[spans[i][3]:spans[i + 1][0]] for i in range(len(spans) - 1))
            seps = [g for g, _ in gaps.most_common() if g.strip() and len(g) <= 4 and not re.search(r"[A-Za-z0-9]", g)]
            for sep in (seps[:1] or [""]):
                if RowReader.facts_parse(t[m.start():pos], join, sep) == want:
                    return m.start(), pos, join, sep
        return None

    def shape_of(self, t: str, x: dict) -> dict | None:
        spans: list[tuple[int, int, str]] = []
        free = lambda a, b: all(b <= s or a >= e for s, e, _ in spans)
        fx = None
        if x.get("facts"):
            loc = self.facts_span(t, x["facts"])
            if loc is None:
                return None
            spans.append((loc[0], loc[1], "facts"))
            fx = (loc[2], loc[3])
        for k in sorted((k for k in ROW_TEXT_FIELDS if isinstance(x.get(k), str) and x[k]), key=lambda k: -len(x[k])):
            p = self.locate(t, x[k], free)
            if p is None:
                return None
            spans.append((p, p + len(x[k]), k))
        if "price" in x:
            p = next(((m.start(), m.end()) for m in re.finditer(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\w)", t)
                      if free(m.start(), m.end()) and self.number(m.group(0)) is not None and abs(float(self.number(m.group(0))) - float(x["price"])) < 1e-6), None)
            if p is None:
                return None
            spans.append((p[0], p[1], "price"))
        spans.sort()
        lits, cur = [], 0
        for a, b, _ in spans:
            lits.append(t[cur:a])
            cur = b
        if any(not lit.strip() for lit in lits[1:]):
            return None
        rx = "^" + "".join(re.escape(lit) + "(.*?)" for lit in lits) + re.escape(t[cur:]) + "$"
        marks = [lit.strip()[-30:] for lit in lits[1:] + [t[cur:]] if len(lit.strip()) >= 4]
        labels = any(ROWS_LABEL_RX.search(" " + lit) for lit in lits[1:] + [t[cur:]])
        sh = {"rx": re.compile(rx, re.S), "fields": [k for _, _, k in spans], "facts": fx, "marks": marks, "labels": labels,
              "const": {"stock": x["stock"]} if "stock" in x else {}}
        return sh if self.apply(sh, t) == x else None

    def apply(self, sh: dict, t: str, avoid: Any = None) -> dict | None:
        m = sh["rx"].match(t)
        if not m:
            return None
        row: dict[str, Any] = dict(sh["const"])
        for k, v in zip(sh["fields"], m.groups()):
            if avoid is not None and k != "facts" and (any(f in v for f in avoid) or (sh["labels"] and ROWS_LABEL_RX.search(v))):
                return None
            if k == "facts":
                fx = self.facts_parse(v, *sh["facts"])
                if not fx:
                    return None
                row[k] = fx
            elif k == "price":
                n = self.number(v)
                if n is None or re.search(r"[A-Za-z]", v):
                    return None
                row[k] = n
            elif not v.strip() or v != v.strip() or (k == "id" and (any(c.isspace() for c in v) or len(v) > 64)):
                return None
            else:
                row[k] = v
        return {k: row[k] for k in ROW_FIELDS if k in row}

    def learn(self, texts: list[str], rows: list[dict], tool: str) -> None:
        pairs = []
        for t in texts:
            mine = [x for x in rows if self.found(x["id"], t)]
            if len(mine) == 1:
                pairs.append((t, mine[0]))
        fresh: list[dict] = []
        for t, x in pairs[:8]:
            try:
                sh = self.shape_of(t, x)
            except re.error:
                sh = None
            if sh and all(s["rx"].pattern != sh["rx"].pattern or s["const"] != sh["const"] or s["tool"] != tool for s in self.shapes + fresh):
                sh["tool"] = tool
                fresh.append(sh)
        for sh in fresh:
            if all(self.apply(sh, t) in (None, x) for t, x in pairs):
                self.shapes.append(sh)
        self.shapes = self.shapes[-ROWS_SHAPES:]

    def shape_rows(self, texts: list[str], tool: str) -> list[dict | None]:
        order = sorted((s for s in self.shapes if s["tool"] == tool), key=lambda s: -len(s["fields"]))
        if not order:
            return [None for _ in texts]
        marks = {f for s in order for f in s["marks"]}
        avoid = [marks - set(s["marks"]) for s in order]
        out: list[dict | None] = []
        for t in texts:
            best, top = None, None
            for sh, av in zip(order, avoid):
                g = self.apply(sh, t, av)
                if not g:
                    continue
                m = sh["rx"].match(t)
                score = (len(t) - sum(len(x) for x in m.groups()), len(sh["fields"]))
                if top is None or score > top:
                    best, top = g, score
            out.append(best if best and self.found(best["id"], t) else None)
        return out

    def user_of(self, texts: list[str]) -> str:
        body = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
        return "TEXTS:\n" + body[:ROWS_CHARS]

    @staticmethod
    def chunks(texts: list[str]) -> list[list[str]]:
        out: list[list[str]] = []
        size = 0
        for t in texts:
            if out and size + len(t) <= ROWS_CHUNK:
                out[-1].append(t)
                size += len(t)
            else:
                out.append([t])
                size = len(t)
        return out

    def by_model(self, batch: list[list[str]], tools: list[str]) -> list[list[dict]]:
        r = self.r
        res: list[list[dict]] = [[] for _ in batch]
        todo: dict[str, list[int]] = {}
        for j, texts in enumerate(batch):
            key = "\n".join(texts)[:ROWS_CHARS]
            if key in self.done:
                res[j] = self.done[key]
            else:
                todo.setdefault(key, []).append(j)
        if not todo or r.t() > GUARD_AT - ROWS_SPARE:
            return res
        parts: list[tuple[str, list[str]]] = []
        for key in todo:
            parts += [(key, c) for c in self.chunks(batch[todo[key][0]])]
        jobs = [("read", ROWS_SYS, self.user_of(c), {"max_tokens": min(ROWS_TOKENS, 600 + sum(len(t) for t in c) // 2),
                                                     "deadline": min(GUARD_AT, r.t() + ROWS_WAIT)}) for _, c in parts]
        answers = r.model_many(jobs)
        got: dict[str, list[dict]] = {}
        failed: set[str] = set()
        for (key, c), d in zip(parts, answers):
            self.asked += 1
            if d is None:
                failed.add(key)
            have = {x["id"] for x in got.get(key, [])}
            got.setdefault(key, []).extend(x for x in self.verified(d, c) if x["id"] not in have)
        for key, js in todo.items():
            rows = got.get(key) or []
            if key not in failed:
                self.done[key] = rows
            if rows:
                self.learn(batch[js[0]], rows, tools[js[0]])
            for j in js:
                res[j] = rows
        return res

    def fix_all(self, calls: list[tuple[str, dict]], obs_list: list) -> list:
        try:
            return self.fix_each(calls, obs_list)
        except NoTime:
            return obs_list
        except Exception as exc:
            self.r.notes.append(f"row reader: {type(exc).__name__}: {exc}"[:160])
            return obs_list

    def fix_each(self, calls: list[tuple[str, dict]], obs_list: list) -> list:
        r = self.r
        out = list(obs_list)
        jobs: list[tuple[int, tuple, list[str], list]] = []
        for i, ((n, a), o) in enumerate(zip(calls, obs_list)):
            role = r.role_of(n)
            hit = self.signal(role, a, o, r.store.id, set(r.cat.items), r.store.total)
            if hit is None:
                continue
            path, texts = hit
            self.texts.update(texts)
            known = self.shape_rows(texts, n)
            if all(known) and len({x["id"] for x in known if x}) == len(known):
                self.shaped += 1
                out[i] = self.place(o, path, self.named([x for x in known if x]), role)
            else:
                jobs.append((i, path, texts, known))
        if not jobs:
            return out
        got = self.by_model([[t for t, k in zip(texts, known) if not k] for _, _, texts, known in jobs], [calls[i][0] for i, _, _, _ in jobs])
        for (i, path, texts, known), rows in zip(jobs, got):
            merged: list[dict] = []
            seen: set[str] = set()
            for t, k in zip(texts, known):
                for x in ([k] if k else [x for x in rows if self.found(x["id"], t)]):
                    if x["id"] not in seen:
                        seen.add(x["id"])
                        merged.append(x)
            if merged:
                out[i] = self.place(obs_list[i], path, self.named(merged), r.role_of(calls[i][0]))
        r.notes.append(f"row reader: {len(jobs)} listing texts sent to the model ({sum(1 for _, _, _, k in jobs for x in k if not x)} unmatched pieces), "
                       f"{sum(len(x) for x in got)} rows kept; {self.shaped} decoded by learned shape so far")
        return out


class Env:

    QUIET = 60

    def __init__(self, run: Any) -> None:
        self.r = run
        self.last = 0.0
        self.conflict = False
        self.fails = 0
        self.spent: dict[str, Any] = {}
        self.quiet = 0

    @staticmethod
    def cast(spec: dict, v: Any) -> Any:
        typ = spec.get("type")
        try:
            if isinstance(v, bool):
                pass
            elif typ == "integer":
                v = int(round(float(v)))
            elif typ == "number":
                f = float(v)
                v = int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            pass
        if v is not None:
            if typ == "string" and not isinstance(v, str):
                v = str(v)
            elif typ == "array" and not isinstance(v, list):
                v = [v]
        if isinstance(v, str):
            choices = spec.get("enum")
            if choices:
                hit = next((e for e in choices if str(e).lower() == v.lower()), None)
                v = v if hit is None else hit
            if isinstance(spec.get("maxLength"), int):
                v = v[:spec["maxLength"]]
        elif isinstance(v, list) and isinstance(spec.get("maxItems"), int):
            v = v[:spec["maxItems"]]
        return v

    def shape(self, name: str, args: dict) -> dict:
        sch = self.r.schemas.get(name) or {}
        props = sch.get("properties") or {}
        strict = bool(props) or sch.get("additionalProperties") is False
        out: dict[str, Any] = {}
        for k, v in (args or {}).items():
            spec = props.get(k)
            if spec is not None:
                out[k] = self.cast(spec, v)
            elif not strict:
                out[k] = v
        for k in sch.get("required") or []:
            choices = (props.get(k) or {}).get("enum")
            if k not in out and choices:
                out[k] = self.enum_default(name, k, choices, out)
            elif k not in out and isinstance(props.get(k), dict):
                fill = self.filler(name, props[k], out)
                if fill is not None:
                    out[k] = fill
        return out

    def filler(self, name: str, spec: dict, args: dict) -> Any:
        r = self.r
        typ = spec.get("type")
        if typ == "string":
            role = r.call_role(name, args)
            if role == "decline":
                text = r.decline_note()
            elif role == "buy":
                held = r.order.held() if r.store.tm.get("cart") else {}
                text = "Placing the order for the requested lines" + (": " + ", ".join(f"{q} x {c}" for c, q in held.items()) if held else "") + "."
            else:
                text = Text.of(r.request)[:80] or "request"
            lo, hi = spec.get("minLength"), spec.get("maxLength")
            if isinstance(lo, int) and len(text) < lo:
                text = text + " " * (lo - len(text))
            return text[:hi] if isinstance(hi, int) and hi > 0 else text
        if typ in ("integer", "number"):
            lo = spec.get("minimum")
            return lo if isinstance(lo, (int, float)) else 0
        if typ == "boolean":
            return True
        if typ == "array":
            return []
        if typ == "object":
            return {}
        return None

    def enum_default(self, name: str, key: str, choices: list, args: dict) -> Any:
        r = self.r
        spec = ((r.schemas.get(name) or {}).get("properties") or {}).get(key, {})
        text = " ".join([r.descs.get(name, ""), Text.of(spec.get("description"))])
        used = set(Text(" ".join(Text.key(a) for a in args)).stems)
        pieces = re.split(r"[,;.]", text)
        best, top = None, 0
        for e in choices:
            mark = re.compile(r"\b" + re.escape(str(e).lower()) + r"\b")
            for piece in pieces:
                if not mark.search(piece.lower()):
                    continue
                shared = len(used & set(Text(piece).stems))
                if shared > top:
                    best, top = e, shared
        if best is not None:
            return best
        for m in r.tm.values():
            if m.get("name") == name and key in (m.get("args") or {}):
                return m["args"][key]
        return choices[0]

    def pace(self) -> None:
        wait = self.last + ENV_PACE - monotonic()
        if wait > 0:
            sleep(wait)

    def envelope(self, calls: list[tuple[str, dict]]) -> dict:
        env: dict[str, Any] = {**self.r.binding, "step": self.r.steps + 1}
        if len(calls) == 1:
            env["action"] = {"name": calls[0][0], "args": calls[0][1]}
        else:
            env["actions"] = [{"name": n, "args": a} for n, a in calls]
        return env

    def exchange(self, env: dict) -> Any:
        r = self.r
        data = None
        attempt = 0
        told: set[int] = set()
        while attempt < ENV_TRIES:
            left = TASK_SECONDS - 10 - r.t()
            if left < 3:
                break
            data, status, err = Wire.post("/environment/call", env, min(30.0, left))
            self.last = monotonic()
            if isinstance(data, dict):
                return data
            if status == 429:
                if left < 8:
                    break
                sleep(min(crowd_wait(r.sid), left - 6))
                continue
            attempt += 1
            if status == 402:
                r.exhausted = True
                break
            if status == 409:
                step = next((int(n) for n in re.findall(r"\d+", err or "") if int(n) != env.get("step")), None)
                if step is not None and step >= 1 and step not in told:
                    told.add(step)
                    r.steps = step - 1
                    env["step"] = step
                    continue
                self.conflict = True
                return None
            if TASK_SECONDS - 10 - r.t() < 8:
                break
            sleep(min(6.0, 1.0 + 1.5 * attempt))
        return data

    @staticmethod
    def row(name: str, args: dict, obs: Any) -> dict:
        return {"role": "assistant", "content": "",
                "tool_call": {"type": "function", "function": {"name": name, "arguments": Text.json(args)}},
                "environment_result": {"observation": obs}}

    def send(self, calls: list[tuple[str, dict]], retry: bool = False) -> list[Any]:
        r = self.r
        if not self.spent or not calls or self.quiet >= self.QUIET or not any(n in self.spent for n, _ in calls):
            return self.deliver(calls, retry)
        if r.finished or r.env_dead or r.t() > LAST_ENV_AT:
            return []
        live = [(n, a) for n, a in calls if n not in self.spent]
        got = iter(self.deliver(live, retry) if live else [])
        out: list[Any] = []
        held = 0
        for n, a in calls:
            if n in self.spent:
                out.append(json.loads(json.dumps(self.spent[n])))
                held += 1
                continue
            o = next(got, None)
            if o is None:
                break
            out.append(o)
        self.quiet += held
        r.plan.spend(r.phase, held)
        return out

    @staticmethod
    def over(obs: Any) -> bool:
        u = Store.unwrap(obs)[0]
        return isinstance(u, dict) and len(u) <= 2 and u.get("error") == "budget_exceeded"

    def spend_tool(self, name: str, obs: Any) -> None:
        r = self.r
        if name not in self.spent:
            r.notes.append(f"tool {name} answered budget_exceeded at step {r.steps}: treated as spent, not as a store error")
        self.spent[name] = obs

    def deliver(self, calls: list[tuple[str, dict]], retry: bool = False) -> list[Any]:
        r = self.r
        if r.finished or r.env_dead or not calls or r.t() > LAST_ENV_AT:
            return []
        room = max(0, min(16, r.steps_left()))
        calls = [(n, self.shape(n, a)) for n, a in calls if n in r.schemas][:room]
        if not calls:
            return []
        self.pace()
        data = self.exchange(self.envelope(calls))
        if not isinstance(data, dict):
            self.fails += 1
            if not r.finished and (self.conflict or r.exhausted):
                r.env_dead = True
            r.notes.append(f"environment call failed at step {r.steps} ({self.fails} so far)")
            return []
        self.fails = 0
        batch = data.get("observations")
        obs_list = batch if isinstance(batch, list) else [data.get("observation")]
        obs_list = [records_in(o) for o in obs_list]
        obs_list = r.prose.fix_all(calls, obs_list)
        try:
            r.steps = max(r.steps + len(obs_list), int(data.get("step") or 0))
        except (TypeError, ValueError):
            r.steps += len(obs_list)
        r.finished = bool(data.get("finished"))
        for (n, a), obs in zip(calls, obs_list):
            r.dialogue.append(self.row(n, a, obs))
            r.cat.ingest(n, a, obs)
            r.trace.append(f"s{r.steps} {n} {Text.json(a, 90)} -> {Text.json(obs, 140)}")
            if r.role_of(n) != "ask":
                if self.over(obs):
                    self.spend_tool(n, obs)
                else:
                    r.err_streak = r.err_streak + 1 if self.faulty(a, obs) else 0
            self.remember(obs)
        r.plan.spend(r.phase, len(obs_list))
        if not retry and not r.finished:
            obs_list = self.mend(calls, obs_list)
        if r.err_streak >= ERR_STREAK and r.mode == "pipeline" and not r.finished:
            r.notes.append(f"{r.err_streak} store errors in a row at step {r.steps}: handing over to the tool loop")
            raise StoreBroken()
        return obs_list

    def mend(self, calls: list[tuple[str, dict]], obs_list: list) -> list:
        r = self.r
        fix: list[tuple[int, str, dict]] = []
        for i, ((n, a), o) in enumerate(zip(calls, obs_list)):
            u = Store.unwrap(o)[0]
            if not (isinstance(u, dict) and u.get("error") == "invalid_arguments"):
                continue
            props = (r.schemas.get(n) or {}).get("properties") or {}
            miss = [x for x in re.findall(r"'([^']+)'", Text.of(u.get("detail"))) if x in props and x not in a]
            a2 = dict(a)
            for x in miss:
                v = self.filler(n, props[x], a2) if isinstance(props[x], dict) else None
                if v is not None:
                    a2[x] = v
            if a2 != a:
                fix.append((i, n, a2))
        if not fix or r.steps_left() < 1:
            return obs_list
        r.notes.append(f"invalid arguments: resent {[n for _, n, _ in fix]} with the named fields filled")
        again = self.send([(n, a2) for _, n, a2 in fix], retry=True)
        out = list(obs_list)
        for (i, _, _), o in zip(fix, again):
            out[i] = o
        return out

    def remember(self, obs: Any) -> None:
        stack = [(obs, 0)]
        while stack:
            x, d = stack.pop()
            if isinstance(x, dict) and d <= 3:
                for k, v in x.items():
                    if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                        self.r.seen_keys[Text.key(k)] = v
                    elif isinstance(v, dict):
                        stack.append((v, d + 1))

    @staticmethod
    def faulty(args: dict, obs: Any) -> bool:
        if obs is None:
            return True
        obs = Store.unwrap(obs)[0]
        if not isinstance(obs, dict) or not 1 <= len(obs) <= 2 or not all(isinstance(v, str) for v in obs.values()):
            return False
        passed = {v for v in (args or {}).values() if isinstance(v, str)}
        return not (set(obs.values()) & passed)


class Catalog:
    READ_ROLES = ("detail", "specs", "compare")

    def __init__(self, run: Any) -> None:
        self.r = run
        self.store = run.store
        self.items: dict[str, dict] = {}
        self.opened: dict[str, set] = {}
        self.aliases: dict[str, list[str]] = {}
        self.equiv: dict[tuple[str, str], bool] = {}
        self.pointers_done: set[str] = set()
        self.domains: dict[tuple[str, int], set[str]] = {}
        self.coded: dict[tuple, str | None] = {}
        self.shelf_codes: dict[tuple, bool] = {}

    @staticmethod
    def is_scalar(v: Any) -> bool:
        if isinstance(v, str):
            return len(v) <= 600
        return isinstance(v, (int, float, bool))

    @classmethod
    def row_lists(cls, obs: Any, depth: int = 0) -> list[tuple[str, list]]:
        found: list[tuple[str, list]] = []
        if not isinstance(obs, dict) or depth > 2:
            return found
        for k, v in obs.items():
            if isinstance(v, dict):
                found += cls.row_lists(v, depth + 1)
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                found.append((k, v))
        return found

    @staticmethod
    def id_in(d: dict, key: str | None) -> str | None:
        named = d.get(key) if key else None
        if isinstance(named, str):
            return named
        if isinstance(named, int) and not isinstance(named, bool):
            return str(named)
        for v in d.values():
            if isinstance(v, str) and " " not in v and not v.isdigit() and ID_RX.match(v):
                return v
        return None

    @staticmethod
    def boxes_of(d: dict, attrs_key: str | None) -> list:
        if attrs_key and attrs_key in d:
            return [d[attrs_key]]
        boxes = []
        for v in d.values():
            if isinstance(v, dict):
                if sum(1 for x in v.values() if Catalog.is_scalar(x) or isinstance(x, list)) >= 2:
                    boxes.append(v)
            elif isinstance(v, list) and v and all(isinstance(x, dict) and len(x) == 2 for x in v):
                boxes.append(v)
        return boxes

    @staticmethod
    def facts_in(d: dict, attrs_key: str | None) -> dict[str, str]:
        out: dict[str, str] = {}
        for box in Catalog.boxes_of(d, attrs_key):
            if isinstance(box, dict):
                for k, v in box.items():
                    if isinstance(v, list):
                        v = ", ".join(Text.of(x) for x in v if Catalog.is_scalar(x))
                    if Catalog.is_scalar(v) and str(v).strip():
                        out[str(k)] = " ".join(str(v).split())[:300]
            elif isinstance(box, list):
                for pair in box:
                    if isinstance(pair, str):
                        m = REC_PAIR_RX2.match(pair)
                        if m and m.group(2).strip():
                            out[m.group(1).strip()] = " ".join(m.group(2).split())[:300]
                        continue
                    if not isinstance(pair, dict):
                        continue
                    k, v = (list(pair.values()) + [None, None])[:2]
                    if len(pair) == 2 and Catalog.is_scalar(k) and Catalog.is_scalar(v):
                        out[str(k)] = " ".join(str(v).split())[:300]
        return out

    def blank(self, rid: str) -> dict:
        return self.items.setdefault(rid, {"id": rid, "row": {}, "rec": {}, "facts": {}, "read": False,
                                           "seen": self.r.steps, "queries": []})

    def scalars(self, d: dict) -> dict:
        return {k: money_of(v) for k, v in d.items() if self.is_scalar(money_of(v))}

    @staticmethod
    def keep_long(it: dict, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, str) and len(v) > 600:
                it.setdefault("long", {})[k] = v[:600]
                it.setdefault("blurb", {})[k] = v[:OPEN_TEXT]

    @staticmethod
    def merge_facts(it: dict, facts: dict) -> None:
        if any(it["facts"].get(k) != v for k, v in facts.items()):
            it.pop("jl", None)
            it.pop("jspan", None)
        it["facts"].update(facts)

    def mark_read(self, it: dict) -> None:
        it["read"] = True
        it["read_step"] = self.r.steps

    def note_opened(self, role: str, args: dict) -> None:
        m = self.r.tm.get(role) or {}
        v = args.get(m.get("ids") if role == "compare" else m.get("id") or "")
        for i in (v if isinstance(v, list) else [v]):
            if isinstance(i, str) and i:
                self.opened.setdefault(i, set()).add(role)

    def take_rows(self, role: str, args: dict, obs: Any) -> None:
        store = self.store
        rows_carry_facts = not store.reads_per_line() and not self.r.tm.get("compare")
        for _, rows in self.row_lists(obs):
            for row in rows:
                rid = self.id_in(row, store.id)
                if not rid:
                    continue
                it = self.blank(rid)
                it["row"].update(self.scalars(row))
                self.keep_long(it, row)
                facts = self.facts_in(row, store.attrs)
                if role not in self.READ_ROLES and role not in ("cart", "view", "remove", "quote"):
                    self.r.cheap.note_row(it, row, facts)
                if facts and (role in self.READ_ROLES or rows_carry_facts):
                    self.merge_facts(it, facts)
                    it["rec"].update(self.scalars(row))
                    self.mark_read(it)
                if role in ("search", "filter"):
                    m = self.r.tm.get(role) or {}
                    q = Text.of(args.get(m.get("query") or m.get("category") or "") or Text.json(args, 80)).lower()
                    if q not in it["queries"]:
                        it["queries"].append(q)

    def take_record(self, role: str, obs: dict) -> None:
        store = self.store
        for d in [obs] + [v for v in obs.values() if isinstance(v, dict)]:
            if store.id and store.id not in d:
                continue
            rid = self.id_in(d, store.id)
            flat = role in ("detail", "specs") and d is obs and sum(1 for v in d.values() if self.is_scalar(v)) >= 3
            nested = any(isinstance(v, (dict, list)) for v in d.values())
            if not rid or not (flat or nested):
                continue
            facts = self.facts_in(d, store.attrs)
            if not facts and role not in ("detail", "specs"):
                continue
            it = self.blank(rid)
            it["rec"].update(self.scalars(d))
            it["row"].update({k: v for k, v in self.scalars(d).items() if k not in it["row"]})
            self.keep_long(it, d)
            self.merge_facts(it, facts)
            if role == "detail":
                it["paged"] = True
            if role in self.READ_ROLES and (facts or not self.r.tm.get("specs") or role == "specs"):
                self.mark_read(it)

    def take_cart(self, obs: Any) -> None:
        r = self.r
        view = self.store.cart_lines(obs)
        if view is not None:
            top = Store.unwrap(obs)[0]
            nums = [v for v in (top.values() if isinstance(top, dict) else []) if isinstance(v, (int, float)) and not isinstance(v, bool)]
            r.cart_total = nums[0] if len(nums) == 1 else r.cart_total
            r.cart_rows = self.store.cart_rows(obs)
            r.cart = view
            r.cart_seen = True
            r.cart_stamp = r.steps

    def sentences_in(self, obs: Any) -> list[str]:
        out: list[str] = []
        stack = [(obs, 0)]
        while stack:
            x, depth = stack.pop()
            if depth > 2:
                continue
            if isinstance(x, dict):
                if self.store.id and self.store.id in x:
                    continue
                stack.extend((v, depth + 1) for v in reversed(list(x.values())))
            elif isinstance(x, list) and x and all(isinstance(s, str) for s in x):
                out.extend(s for s in x if len(s.split()) >= 4 and s not in self.r.prose.texts)
        return out

    def ingest(self, name: str, args: dict, obs: Any) -> None:
        r = self.r
        role = r.role_of(name)
        obs, notices = self.store.observe(name, args, obs)
        r.news.extend(s for s in notices if s not in r.news and s not in r.prose.texts)
        if role == "ask":
            return
        if role in self.READ_ROLES and isinstance(obs, dict) and (len(obs) >= 2 or self.row_lists(obs)):
            self.note_opened(role, args)
        self.take_rows(role, args, obs)
        r.capsweep.note(name, args, obs)
        if isinstance(obs, dict):
            self.take_record(role, obs)
        if role in ("cart", "view", "remove", "quote"):
            self.take_cart(obs)
        r.news.extend(s for s in self.sentences_in(obs) if s not in r.news)

    @staticmethod
    def field(it: dict, key: str | None) -> Any:
        if key:
            for box in (it["rec"], it["row"]):
                if key in box:
                    return box[key]
        return None

    def slot_text(self, it: dict, slot: str) -> str:
        return Text.of(self.field(it, getattr(self.store, slot)))

    def price(self, it: dict) -> float | None:
        v = money_of(self.field(it, self.store.price))
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            found = Text(v).numbers
            return found[0][0] if found else None
        return None

    def in_stock(self, it: dict) -> bool | None:
        v = self.field(it, self.store.stock)
        if isinstance(v, bool):
            return v == self.store.stock_true_available
        if isinstance(v, (int, float)):
            return v > 0
        word = v.strip().lower() if isinstance(v, str) else ""
        if not word or self.store.stock_yes is None:
            return None
        return word in self.store.stock_yes

    def category(self, it: dict) -> str:
        return self.slot_text(it, "category")

    def title(self, it: dict) -> str:
        return self.slot_text(it, "title")[:160]

    def seller(self, it: dict) -> str:
        return self.slot_text(it, "seller")

    def maker(self, it: dict) -> str:
        return self.slot_text(it, "maker")

    def notes_of(self, it: dict) -> str:
        parts: list[str] = []
        memo = self.field(it, self.store.notes)
        if isinstance(memo, str) and memo.strip():
            parts.append(memo)
        for k, v in list(it["rec"].items()) + list(it["row"].items()):
            if k != self.store.title and isinstance(v, str) and len(v.split()) >= 6 and v not in parts:
                parts.append(v)
        return " | ".join(sorted(parts, key=len))

    def full_text(self, it: dict) -> str:
        return " ".join([self.title(it)] + [str(v) for v in it["facts"].values()] + [self.notes_of(it)])

    def fact_values(self, it: dict, prop: str) -> list[tuple[str, str]]:
        p = Text.key_stems(prop)
        pool = dict(it["facts"])
        for k, v in list(it["rec"].items()) + list(it["row"].items()):
            if k not in pool and isinstance(v, (str, int, float)) and not isinstance(v, bool):
                pool[k] = str(v)
        exact = [(k, str(v)) for k, v in pool.items() if Text.key_stems(k) == p]
        if exact:
            return exact
        wider = []
        for k, v in pool.items():
            ks = Text.key_stems(k)
            if p and len(ks) > len(p) and ks[-len(p):] == p:
                wider.append((k, str(v)))
        if wider:
            return wider
        tail = [(k, str(v)) for k, v in pool.items()
                if 2 <= len(Text.key_stems(k)) < len(p) and p[-len(Text.key_stems(k)):] == Text.key_stems(k)]
        if tail:
            return tail
        alias = {Text.key(a) for a in self.aliases.get(Text.key(prop), [])}
        return [(k, str(v)) for k, v in pool.items() if Text.key(k) in alias]

    @staticmethod
    def wanted_of(req: dict) -> list[str]:
        vals = req.get("value")
        return [w for w in (vals if isinstance(vals, list) else [vals]) if Text.of(w).strip()]

    @staticmethod
    def wanted_text(req: dict) -> str:
        v = req.get("value")
        return v if isinstance(v, str) else " ".join(v or [])

    @staticmethod
    def initial_unit(u1: str, u2: str) -> bool:
        a, b = sorted((Text.of(u1).strip().strip(".\"'").lower().rstrip("s"), Text.of(u2).strip().strip(".\"'").lower().rstrip("s")), key=len)
        return len(a) == 1 and a.isalpha() and b.isalpha() and len(b) >= 4 and b[0] == a

    @staticmethod
    def number_verdict(req: dict, wanted: list, value: str, plain: bool = False) -> str:
        got = Text(value).numbers
        tgt = [n for w in wanted for n in Text(w).numbers]
        if not got or not tgt:
            return "unknown"
        unit = req.get("unit") or tgt[0][1]
        lo = tgt[0][0]
        hi = tgt[1][0] if len(tgt) > 1 else lo
        op = req.get("op")
        if plain and Text.of(unit).strip() and not re.search(r"\d\s*/\s*\d", value):
            real = [(g, gu) for g, gu in got if gu.strip().strip(".\"'")]
            if any(Text.unit_scale(gu, unit) is not None for _, gu in real):
                got = real
        comparable = True
        for g, gu in got:
            f = Text.unit_scale(gu, unit)
            if f is None and Catalog.initial_unit(gu, unit):
                f = 1.0
            if f is None:
                comparable = False
                continue
            x = g * f
            if op == "max":
                ok = x <= lo + 1e-9
            elif op == "min":
                ok = x >= lo - 1e-9
            elif op == "between":
                ok = min(lo, hi) - 1e-9 <= x <= max(lo, hi) + 1e-9
            else:
                ok = abs(x - lo) < 1e-6
            if ok:
                return "yes"
        return "no" if comparable else "unknown"

    @staticmethod
    def content_stems(w: Any) -> list[str]:
        st = Text(Text.of(w)).stems
        return st[1:] if len(st) > 1 and len(st[0]) == 1 and not st[0].isdigit() else st

    @staticmethod
    def lead_stems(w: Any, plain: bool) -> list[str]:
        return Text(Text.of(w)).stems if plain and re.match(r"\s*[A-Za-z]-", Text.of(w)) else Catalog.content_stems(w)

    @staticmethod
    def plus_marks(s: Any) -> tuple[set[str], set[str]]:
        plus: set[str] = set()
        bare: set[str] = set()
        for m in PLUS_RX.finditer(Text.of(s).lower()):
            if re.search(r"[a-z]", m.group(1)):
                (plus if m.group(2) else bare).add(m.group(1))
        return plus, bare

    @staticmethod
    def plus_clash(wanted: Any, value: Any) -> bool:
        wp, wb = Catalog.plus_marks(wanted)
        vp, vb = Catalog.plus_marks(value)
        return any(t in vb and t not in vp for t in wp) or any(t in vp and t not in vb for t in wb)

    @staticmethod
    def value_verdict(req: dict, value: str, plain: bool = False) -> str:
        op = req.get("op") or "is"
        wanted = Catalog.wanted_of(req)
        if op in ("is", "not") and wanted and all(Catalog.plus_clash(w, value) for w in wanted):
            return "yes" if op == "not" else "no"
        if op in ("max", "min", "between", "eq"):
            return Catalog.number_verdict(req, wanted, value, plain)
        hay = Text(value).stems
        found = any(Text.run_in(Catalog.lead_stems(w, plain), hay) for w in wanted)
        if op == "not":
            if not found and plain and len(hay) > 1:
                found = any("".join(hay) == "".join(Text(Text.of(w)).stems) for w in wanted)
            return "no" if found else "yes"
        if found:
            return "yes"
        own = set(Text.key_stems(req.get("prop") or ""))

        def keyish(x: str) -> bool:
            return x in own or any(min(len(x), len(k)) >= 4 and (k.startswith(x) or x.startswith(k)) for k in own)
        for w in wanted:
            full = Text(w).stems
            trimmed = [x for x in full if not keyish(x)]
            if trimmed and len(trimmed) < len(full) and Text.run_in(trimmed, hay):
                return "yes"
        if any(ch in value for ch in "&/,"):
            for w in wanted:
                full = Catalog.lead_stems(w, plain)
                for i in range(1, len(full) - 1):
                    if len(full[i]) <= 3 and full[i] not in hay and Text.run_in(full[:i] + full[i + 1:], hay):
                        return "yes"
        if all(Text(w).numbers for w in wanted) and Text(value).numbers:
            measure = all(len(Text(w).numbers) == 1 and sum(1 for x in Text(w).words if not any(c.isdigit() for c in x)) <= 1 for w in wanted)
            same = Catalog.number_verdict(dict(req, op="is"), wanted, value, plain) if measure else "unknown"
            split = False
            for w in wanted:
                own_w = set(Text(w).stems)
                at = [n for n, x in enumerate(hay) if x in own_w]
                if len(at) >= 2 and own_w <= set(hay) and any(x not in own_w for x in hay[min(at):max(at) + 1]):
                    split = True
            if same == "no" and named_figure(value) and all(named_figure(w) for w in wanted):
                return "no"
            return "yes" if same == "yes" and len(Text(value).numbers) == 1 and not split else "unknown"
        return "no"

    def mentions(self, it: dict, req: dict) -> bool:
        hay = Text(" | ".join([self.title(it)] + [str(v) for v in it["facts"].values()])).stems
        vals = req.get("value") if isinstance(req.get("value"), list) else [req.get("value")]
        return any(Text.run_in(Text(v).stems, hay) for v in vals if isinstance(v, str) and v.strip())

    def flag_in(self, it: dict, req: dict) -> bool:
        p = Text.key_stems(req.get("prop") or "")
        if not p or p != Text.key_stems(self.wanted_text(req)):
            return False
        return any(self.value_verdict(req, str(v)) == "yes" for v in it["facts"].values())

    def wider_states(self, it: dict, req: dict, vals: list) -> bool:
        p = Text.key_stems(req.get("prop") or "")
        if not p:
            return False
        for k, v in it["facts"].items():
            if (k, str(v)) in vals:
                continue
            ks = Text.key_stems(k)
            if len(ks) > len(p) and ks[-len(p):] == p and self.value_verdict(req, str(v)) == "yes":
                return True
        return False

    def judged_same(self, req: dict, vals: list) -> bool:
        w = Text.json(req.get("value") if isinstance(req.get("value"), list) else [req.get("value")])
        return any(self.equiv.get((w, v.lower()[:80])) and not self.named_apart(req, k, v) for k, v in vals)

    def named_apart(self, req: dict, key: str, value: str) -> bool:
        own = " ".join(Text(value).words)
        return any(d != own and self.value_verdict(req, d) == "yes" for d in self.domain(key))

    def verdict(self, it: dict, req: dict) -> str:
        sure = self.code_verdict(it, req)
        if sure:
            return sure
        jl = it.get("jl")
        if jl:
            lv = jl.get(Judge.sig(req))
            if lv in JUDGED:
                if JUDGED[lv] == "yes" and self.off_key(it, req):
                    return "close"
                return JUDGED[lv]
        got = self.read_verdict(it, req)
        return "yes" if got != "yes" and self.title_initials(it, req) else got

    def mark_clash(self, it: dict, req: dict) -> bool:
        if req.get("op") not in ("is", "eq"):
            return False
        wanted = self.wanted_of(req)
        hay = [v for _, v in self.fact_values(it, req.get("prop") or "")] or [self.title(it)]
        return bool(wanted) and all(all(self.plus_clash(w, v) for w in wanted) for v in hay)

    def title_initials(self, it: dict, req: dict) -> bool:
        if req.get("op") not in ("is", "eq") or self.fact_values(it, req.get("prop") or ""):
            return False
        toks = it.get("title_tokens")
        if toks is None or it.get("title_seen") != self.title(it):
            it["title_seen"] = self.title(it)
            toks = it["title_tokens"] = set(re.split(r"[^A-Za-z0-9]+", it["title_seen"]))
        for v in self.wanted_of(req):
            parts = [x for x in re.split(r"[^A-Za-z0-9]+", Text.of(v)) if x]
            if len(parts) >= 3 and "".join(x[0] for x in parts).upper() in toks:
                return True
        return False

    @staticmethod
    def flat_words(s: Any) -> str:
        return " ".join(re.sub(r"[-_/|,;()\[\]]", " ", Text.of(s).lower()).split())

    def title_states(self, it: dict, req: dict) -> bool:
        if req.get("op") != "is" or self.fact_values(it, req.get("prop") or ""):
            return False
        p = set(Text.key_stems(req.get("prop") or ""))
        if not p or any(p <= set(Text.key_stems(k)) for k in it["facts"]):
            return False
        title = self.flat_words(self.title(it))
        for v in self.wanted_of(req):
            w = self.flat_words(v)
            if not re.search(r"[a-z]", w) or Text(w).codes:
                continue
            pat = r"(?<![a-z0-9])" + r"\s*".join(re.escape(x) for x in w.split()) + r"(?:e?s)?(?![a-z0-9])"
            if re.search(pat, title):
                return True
        return False

    def code_verdict(self, it: dict, req: dict) -> str | None:
        if req.get("op") not in ("is", "eq"):
            return None
        wanted = self.wanted_text(req)
        want = fit_codes(wanted)
        if not want or not any(fit_codes(w) for w in self.wanted_of(req)):
            return None
        key = (it["id"], bool(it.get("read")), bool(it.get("paged")), wanted, len(self.items), FIT_STATE["version"])
        if key not in self.coded:
            self.coded[key] = None
            vals = self.fact_values(it, req.get("prop") or "")
            if not any(fit_related(c, w) for _, v in vals for c in fit_codes(v) for w in want):
                rec = self.fit_record(it)
                ptrs = fit_pointer_pages(rec)
                pages = [self.fit_record(x) for x in self.items.values() if x is not it and any(self.title_carries(x, c) for c in ptrs)]
                v, why = fit_judge(wanted, rec, pages)
                side = FitInfo(rec, "dep")
                kin = {fit_kind(c)[0] for c in side.compat | side.exc}
                if v == "yes" and why.startswith("match "):
                    self.coded[key] = "yes"
                elif v == "no" and kin & {fit_kind(w)[0] for w in want}:
                    self.coded[key] = "no"
                elif v == "unknown" and why == "no codes" and self.mute_fails(it, wanted, want):
                    self.coded[key] = "no"
        return self.coded[key]

    def mute_fails(self, it: dict, wanted: str, want: list[str]) -> bool:
        desc = self.store.desc
        if not desc or not any(desc in box for box in (it["row"], it["rec"])) or Text(self.full_text(it)).code_fits(wanted) == "yes":
            return False
        kinds = {fit_kind(w)[0] for w in want}
        shelf = self.category(it)
        key = (shelf, tuple(sorted(kinds)))
        if not self.shelf_codes.get(key):
            stale = (key, len(self.items))
            if stale not in self.shelf_codes:
                self.shelf_codes[stale] = any(x is not it and self.category(x) == shelf and kinds & {fit_kind(c)[0] for c in FitInfo(self.fit_record(x), "dep").compat}
                                              for x in self.items.values())
                if self.shelf_codes[stale]:
                    self.shelf_codes[key] = True
            return self.shelf_codes[stale] or bool(self.shelf_codes.get(key))
        return True

    def off_key(self, it: dict, req: dict) -> bool:
        if not it.get("facts") or req.get("op") not in ("is", "eq", "min", "max", "between"):
            return False
        prop = req.get("prop") or ""
        if self.fact_values(it, prop):
            return False
        names = {Text.key(a) for a in self.aliases.get(Text.key(prop), [])}
        if any(Text.key(k) in names for k in it["facts"]):
            return False
        if self.value_verdict(req, self.title(it)) == "yes" or self.title_initials(it, req):
            return False
        if req.get("op") in ("is", "eq") and any(self.value_verdict(req, str(v)) == "yes" for v in it["facts"].values()):
            return False
        shelf = self.category(it)
        p = Text.key_stems(prop)
        return any(x is not it and x.get("read") and self.category(x) == shelf and any(Text.key_stems(k) == p for k in x["facts"])
                   for x in self.items.values())

    def count_value(self, req: dict) -> bool:
        wanted = self.wanted_of(req)
        return (req.get("op") in ("is", "eq") and bool(wanted)
                and all(Text(Text.of(w)).numbers and not re.search(r"[A-Za-z]", Text.of(w)) for w in wanted))

    def stray_off(self, it: dict, req: dict) -> bool:
        prop = req.get("prop") or ""
        numeric = req.get("op") in ("min", "max", "between") or self.count_value(req)
        if not numeric or not it.get("facts") or self.fact_values(it, prop):
            return False
        names = {Text.key(a) for a in self.aliases.get(Text.key(prop), [])}
        return not any(Text.key(k) in names for k in it["facts"]) and self.stray_figure(it, req)

    def joined_off(self, it: dict, req: dict) -> bool:
        if req.get("op") not in ("is", "eq"):
            return False
        wanted = [Text.of(w) for w in self.wanted_of(req) if Text.of(w).strip()]
        vals = [Text.of(v) for _, v in self.fact_values(it, req.get("prop") or "")]
        return bool(wanted and vals) and joined_only(wanted, vals)

    def bare_off(self, it: dict, req: dict) -> bool:
        wanted = self.wanted_of(req)
        if req.get("op") not in ("is", "eq") or not wanted or not it.get("facts") or any(fit_codes(w) or named_figure(w) for w in wanted):
            return False
        p = {s for s in Text.key_stems(req.get("prop") or "") if len(s) > 2}
        if not p or any(set(Text.key_stems(k)) & p for k in it["facts"]):
            return False
        title = self.slot_text(it, "title")
        if self.value_verdict(req, title) == "yes" or self.title_states(it, req):
            return False
        hay = [{Text.stem(t) for t in e} for v in list(it["facts"].values()) + [title] for e in list_elements(v)]
        want = [{Text.stem(t) for t in x} for w in wanted for x in list_elements(w) if x]
        return not any(w <= h for h in hay for w in want)

    def dims_off(self, it: dict, req: dict) -> bool:
        prop = req.get("prop") or ""
        if req.get("op") not in ("min", "max", "between") or not it.get("facts") or self.fact_values(it, prop):
            return False
        if self.head_verdict(it, req) == "no":
            return True
        p, unit = self.figure_stems(req)
        if not p or any(p[0] in Text.key_stems(k) for k in it["facts"]):
            return False
        names = {Text.key(a) for a in self.aliases.get(Text.key(prop), [])}
        title = self.title(it)
        shown = any(u.strip(".\"' ") and (Text.unit_scale(u, unit) is not None or self.initial_unit(u, unit)) for _, u in Text(title).numbers)
        if any(Text.key(k) in names for k in it["facts"]) or (shown and self.value_verdict(req, title, True) == "yes"):
            return False
        head, _, cited = Text.of((it.get("jspan") or {}).get(Judge.sig(req))).partition(":")
        return any(Judge.flat(k) == Judge.flat(head) for k in it["facts"]) and bool(DIMS_RX.search(cited))

    def own_states(self, it: dict, req: dict) -> bool:
        wanted = self.wanted_of(req)
        if req.get("op") not in ("is", "eq") or not wanted:
            return False
        return any(phrase_states(w, v) for _, v in self.fact_values(it, req.get("prop") or "") for w in wanted)

    def head_off(self, it: dict, req: dict) -> bool:
        prop = req.get("prop") or ""
        wanted = self.wanted_of(req)
        if req.get("op") not in ("is", "eq") or not wanted or not it.get("facts") or self.fact_values(it, prop):
            return False
        if any(fit_codes(w) for w in wanted) or self.value_verdict(req, self.title(it)) == "yes" or self.title_states(it, req):
            return False
        p = [s for s in Text.key_stems(prop) if len(s) > 2]
        if len(p) < 2:
            return False
        kin = [v for k, v in it["facts"].items() if set(Text.key_stems(k)) & set(p)]
        heads = [v for k, v in it["facts"].items() if Text.key_stems(k) == p[-1:]]
        if any(len(Text.key_stems(k)) > 1 and Text.key_stems(k)[-1] == p[-1] for k in it["facts"]):
            return False
        return bool(heads) and all(self.value_verdict(req, Text.of(v), True) == "no" for v in kin)

    def figure_unit(self, req: dict) -> str:
        return Text.of(req.get("unit")).strip() or next((u for w in self.wanted_of(req) for _, u in Text(Text.of(w)).numbers if u.strip()), "")

    def measured(self, v: Any, unit: str) -> bool:
        return bool(unit) and any(u.strip(".\"' ") and Text.unit_scale(u, unit) is not None for _, u in Text(Text.of(v)).numbers)

    def figure_stems(self, req: dict) -> tuple[list[str], str]:
        unit = "" if self.count_value(req) else self.figure_unit(req)
        stems = [s for s in Text.key_stems(req.get("prop") or "") if len(s) > 2]
        heads = [s for s in stems if s.rstrip("s") != unit.lower().rstrip("s")]
        return (heads[-1:] + stems) if heads else stems, unit

    def names_figure(self, key: Any, value: Any, p: list[str], unit: str, loose: bool = False) -> bool:
        ks = Text.key_stems(key)
        if p[0] in ks:
            return True
        long_ = p[1:] or p
        named = (bool(set(ks) & set(long_)) or (bool(unit) and any(len(s) > 1 and Text.unit_scale(s, unit) is not None for s in ks + long_))
                 or any(len(w) > 3 and any(len(s) > 3 and (s.startswith(w) or w.startswith(s)) for s in long_) for w in Text(Text.of(value)).words))
        return named and (loose or (bool(unit) and self.measured(value, unit)))

    def stray_figure(self, it: dict, req: dict) -> bool:
        p, unit = self.figure_stems(req)
        if not p:
            return False
        count = self.count_value(req)
        related = {Judge.flat(k): v for k, v in it["facts"].items() if self.names_figure(k, v, p, unit, count)}
        span = Text.of((it.get("jspan") or {}).get(Judge.sig(req)))
        cited = Judge.flat(re.split(r"\s*[:=]\s*", span, maxsplit=1)[0]) if re.search(r"[:=]", span) else ""
        pure = {Judge.flat(k): v for k, v in it["facts"].items()
                if any(u.strip(".\"' ") for _, u in Text(Text.of(v)).numbers) and not re.search(r"[a-z0-9]", Text.AMOUNT.sub("", Text.of(v).lower()))}
        clean = {k for k, v in pure.items() if self.measured(v, unit)}
        if cited and cited not in related and (cited in (clean if unit else pure) or (count and any(Judge.flat(k) == cited for k in it["facts"]))):
            return not any(self.value_verdict(req, Text.of(v), True) == "yes" for v in related.values())
        return not related and bool(clean)

    def head_verdict(self, it: dict, req: dict) -> str:
        unit = self.figure_unit(req)
        p, _ = self.figure_stems(req)
        if not unit or not p:
            return "unknown"
        vals = [Text.of(v) for k, v in it["facts"].items() if p[0] in Text.key_stems(k)]
        if not vals or not all(len(Text(v).numbers) == 1 and self.measured(v, unit) for v in vals):
            return "unknown"
        return "no" if all(self.value_verdict(req, v, True) == "no" for v in vals) else "unknown"

    def firm_yes(self, it: dict, req: dict) -> bool:
        op = req.get("op")
        wanted = self.wanted_of(req)
        vals = [v for _, v in self.fact_values(it, req.get("prop") or "")]
        if op in ("min", "max", "between"):
            unit = self.figure_unit(req)
            if not vals and unit:
                p, _ = self.figure_stems(req)
                vals = [Text.of(v) for k, v in it["facts"].items() if p and p[0] in Text.key_stems(k)]
            return (bool(vals) and all(len(Text(v).numbers) == 1 for v in vals)
                    and all(self.value_verdict(req, v, True) == "yes" for v in vals) and (not unit or any(self.measured(v, unit) for v in vals)))
        if op not in ("is", "eq") or not wanted or not all(named_figure(w) for w in wanted):
            return False
        if vals:
            return all(self.value_verdict(req, v, True) == "yes" for v in vals)
        return self.title_code(it, req) and not self.mark_clash(it, req)

    def waives(self, it: dict, w: dict) -> bool:
        vals = [norm(v) for _, v in self.fact_values(it, w.get("prop") or "")]
        hit = any(phrase_states(v, s) for v in w.get("values") or [] for s in vals)
        return hit if w.get("mode") == "unless" else bool(vals) and not hit

    def title_code(self, it: dict, req: dict) -> bool:
        title = " ".join(self.title(it).lower().split())
        for w in self.wanted_of(req):
            parts = re.findall(r"[a-z0-9]+\+?", Text.of(w).lower())
            if parts and re.search(r"\d", "".join(parts)):
                pat = r"(?<![a-z0-9+])" + r"[\s-]?".join(re.escape(x) for x in parts) + r"(?![a-z0-9+])"
                if re.search(pat, title):
                    return True
        return False

    def read_verdict(self, it: dict, req: dict) -> str:
        op = req.get("op")
        literal = op in ("is", "eq")
        vals = self.fact_values(it, req.get("prop") or "")
        wanted = self.wanted_text(req)
        coded = literal and any(fit_codes(w) for w in self.wanted_of(req))
        if coded and not any(fit_related(c, w) for _, v in vals for c in fit_codes(v) for w in fit_codes(wanted)):
            vals = []
        if not vals:
            if coded:
                rec = self.fit_record(it)
                ptrs = fit_pointer_pages(rec)
                pages = [self.fit_record(x) for x in self.items.values() if x is not it and any(self.title_carries(x, c) for c in ptrs)]
                v, why = fit_judge(wanted, rec, pages)
                if v == "unknown" and "open " in why:
                    it["fit_open"] = [c for c in ptrs if fit_norm(c) not in self.pointers_done]
                    return "unknown"
                if v == "unknown" and why == "no codes":
                    v = Text(self.full_text(it)).code_fits(wanted)
                if v != "yes" and self.value_verdict(req, self.title(it)) == "yes" and (
                        not Text(self.full_text(it)).codes or (v == "unknown" and self.title_code(it, req))):
                    return "yes"
                return v if v != "unknown" else "no"
            if literal and Text(Text.of(req.get("value"))).codes:
                return Text(self.full_text(it)).code_fits(wanted)
            if literal and (self.value_verdict(req, self.title(it)) == "yes" or self.flag_in(it, req)):
                return "yes"
            if op == "not" and self.value_verdict(req, self.title(it)) == "no":
                return "no"
            if op == "not" and self.mentions(it, req):
                return "no"
            if op in ("min", "max", "between"):
                return self.head_verdict(it, req)
            return "unknown"
        answers = [self.value_verdict(req, v, True) for _, v in vals]
        p = Text.key_stems(req.get("prop") or "")
        if literal and not any(Text.key_stems(k) == p for k, _ in vals) and not any(Text.key(k) in {Text.key(a) for a in self.aliases.get(Text.key(req.get("prop") or ""), [])} for k, _ in vals):
            answers = ["unknown" if a == "yes" else a for a in answers]
        if literal and "yes" in answers:
            answers = ["no" if a == "yes" and self.bucket_miss(req, k, v) else a for a, (k, v) in zip(answers, vals)]
        if literal and "yes" not in answers:
            if self.wider_states(it, req, vals):
                answers.append("unknown")
            if self.judged_same(req, vals):
                answers.append("yes")
            if self.flag_in(it, req):
                answers.append("yes")
        if "yes" in answers:
            return "no" if op == "not" and "no" in answers else "yes"
        return "unknown" if "unknown" in answers else "no"

    def domain(self, key: str) -> set[str]:
        nk = Text.key(key)
        mark = (nk, sum(1 for it in self.items.values() if it["read"]))
        if mark not in self.domains:
            seen: set[str] = set()
            for it in self.items.values():
                for k, v in it["facts"].items():
                    if Text.key(k) == nk:
                        seen.add(" ".join(Text(v).words))
            self.domains[mark] = seen
        return self.domains[mark]

    def bucket_miss(self, req: dict, key: str, value: str) -> bool:
        if not re.search(r"\s(?:and|or|&)\s|/", value):
            return False
        whole = " ".join(Text(value).words)
        dom = self.domain(key)
        for w in self.wanted_of(req):
            ww = " ".join(Text(w).words)
            if ww and ww != whole and ww in dom and not any(" ".join(Text(part).words) == ww for part in re.split(r"[,;/]", value)):
                return True
        return False

    def fit_record(self, it: dict) -> tuple[str, str]:
        parts: list[str] = []
        long = it.get("long") or {}
        for v in (self.field(it, self.store.desc) or long.get(self.store.desc or ""), self.field(it, self.store.notes) or long.get(self.store.notes or "")):
            if isinstance(v, str) and v.strip() and v.strip() not in parts:
                parts.append(v.strip())
        for k, v in list(it["rec"].items()) + list(it["row"].items()):
            if k != self.store.title and isinstance(v, str) and len(v.split()) >= 6 and v.strip() not in parts:
                parts.append(v.strip())
        facts = [f"{k}: {v}." for k, v in it["facts"].items() if isinstance(v, str) and v.strip()]
        return self.slot_text(it, "title"), "\n".join(parts), "\n".join(facts)

    def title_carries(self, it: dict, code: str) -> bool:
        return fit_norm(code) in fit_codes(self.slot_text(it, "title"))

    def codes_of(self, it: dict) -> list[tuple[str, str]]:
        return Text(self.full_text(it)).codes

    def codes_agree(self, a: dict, b: dict) -> bool:
        ca, cb = self.codes_of(a), self.codes_of(b)
        if not ca or not cb:
            return True
        return any(Text.code_like(x, y) or Text.code_like(y, x) for x in ca for y in cb)

    def rows_of(self, obs: Any) -> tuple[list[str], int | None, Any]:
        store = self.store
        lists = self.row_lists(obs)
        if store.rows:
            lists = [kv for kv in lists if kv[0] == store.rows] or lists
        ids = [i for i in (self.id_in(row, store.id) for _, rows in lists[:1] for row in rows) if i]
        total = cursor = None
        if isinstance(obs, dict):
            if store.total and isinstance(obs.get(store.total), int):
                total = obs[store.total]
            if store.cursor and obs.get(store.cursor) not in (None, "", False):
                cursor = obs[store.cursor]
            if total == 0 and ids:
                total = None
        return ids, total, cursor


SLOTS = ("rows", "id", "title", "price", "stock", "category", "seller", "maker", "attrs", "notes", "desc", "total", "cursor")


def direct_buy(buy: Any) -> bool:
    return isinstance(buy, dict) and bool(buy.get("id") or buy.get("basket"))


def empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[,;.:()]", text) if c.strip()]


CLAUSE_TYPES = ("line", "value", "limit", "rule", "when", "decline", "reference", "message_cap", "step_cap",
                "budget", "priority", "confirm", "before_order", "fit", "stock", "hint")
OPS = ("is", "not", "min", "max", "between")
RELATIONS = ("same", "not", "min", "max", "near", "fit")
REPLY_KINDS = ("value", "reference", "no_preference", "not_understood", "ask_singly", "other")
SAME_VALUE_KINDS = ("spelling", "yesno")
NAMED_FIGURE_RX = re.compile(r"[A-Za-z]{1,6}[\s-]?\d+\+?")
PLUS_RX = re.compile(r"(?<![a-z0-9])([a-z0-9]*\d[a-z0-9]*)(\+)?")
DIMS_RX = re.compile(r"\d\s*[xX\u00d7*]\s*\d")


QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                           " ": " "})


def rd_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return str(x)
    return json.dumps(x, ensure_ascii=False, default=str)


def norm(s: Any) -> str:
    return " ".join(rd_text(s).translate(QUOTE_MAP).lower().split())


def tokens(s: Any) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*|[a-z]+", norm(s))


def to_number(s: Any) -> float | None:
    t = rd_text(s).strip().replace(" ", "")
    m = re.search(r"-?\d+(?:[.,]\d+)*", t)
    if not m:
        return None
    t = m.group(0)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    elif re.fullmatch(r"-?\d+,\d+", t):
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def numbers_in(s: Any) -> set[float]:
    out: set[float] = set()
    toks = tokens(s)
    for i, t in enumerate(toks):
        if t[0].isdigit():
            n = to_number(t)
            if n is not None:
                out.add(n)
    return out


def number_said(n: float, quote: str) -> bool:
    found = numbers_in(quote)
    return not found or float(n) in found


def named_figure(v: Any) -> bool:
    return bool(NAMED_FIGURE_RX.fullmatch(rd_text(v).strip()))


def same_token(a: str, b: str) -> bool:
    if a == b:
        return True
    if a[0].isdigit() or b[0].isdigit():
        na, nb = to_number(a), to_number(b)
        return na is not None and na == nb
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long_.startswith(short) and len(long_) - len(short) <= 2


def rd_contains(hay: list[str], needle: list[str], exact: bool = False) -> bool:
    if not needle:
        return False
    for i in range(len(hay) - len(needle) + 1):
        if all((hay[i + j] == needle[j]) if exact else same_token(hay[i + j], needle[j]) for j in range(len(needle))):
            return True
    return False


def quote_in(quote: Any, text: Any) -> bool:
    q, t = norm(quote), norm(text)
    if not q:
        return False
    if q in t:
        return True
    return rd_contains(tokens(t), tokens(q))


def value_in(value: Any, text: Any) -> bool:
    vt = tokens(value)
    if not vt:
        return False
    hay = tokens(text)
    if rd_contains(hay, vt):
        return True
    return all(any(same_token(h, v) for h in hay) for v in vt)


def sentences(text: Any) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", rd_text(text)) if s.strip()]


def find_quote(text: str, needles: list[Any]) -> str:
    need = [n for n in needles if rd_text(n).strip()]
    if not need:
        return ""
    hits = [s for s in sentences(text) if all(value_in(n, s) if not isinstance(n, (int, float)) else float(n) in numbers_in(s)
                                              for n in need)]
    return min(hits, key=len) if hits else ""


def rd_json(raw: Any) -> Any:
    s = rd_text(raw).strip()
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.S)
    for cand in (s, s[s.find("{"):s.rfind("}") + 1] if "{" in s else ""):
        if not cand:
            continue
        try:
            return json.loads(cand)
        except ValueError:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
            except ValueError:
                continue
    return Text.last_json(s)


def clean_list(v: Any, limit: int = 8) -> list[str]:
    if v is None:
        return []
    items = v if isinstance(v, list) else [v]
    out = []
    for x in items:
        s = rd_text(x).strip()
        if s and s.lower() not in ("null", "none", "any", "n/a") and s not in out:
            out.append(s[:120])
    return out[:limit]


def clean_op(op: Any, default: str = "is") -> str:
    o = rd_text(op).strip().lower()
    return o if o in OPS else ("is" if o == "eq" else default)


def clean_int(v: Any) -> int | None:
    n = to_number(v) if not isinstance(v, bool) else None
    return int(round(n)) if n is not None and 0 <= n <= 100000 else None




def relation_of(raw_rel: Any, text: str) -> str:
    rel = rd_text(raw_rel).strip().lower()
    return rel if rel in RELATIONS else "same"


def clean_with(w: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(w, dict):
        for k, v in w.items():
            if rd_text(k).strip() and rd_text(v).strip():
                val = rd_text(v).strip()
                if val.lower() in ("yes", "true", "y"):
                    val = rd_text(k).strip()
                out[rd_text(k).strip()[:60]] = val[:120]
    elif isinstance(w, list):
        for pair in w:
            if isinstance(pair, dict) and pair.get("property"):
                out[rd_text(pair["property"]).strip()[:60]] = rd_text(pair.get("value")).strip()[:120]
    return out


FIRST_SYS = """Task: turn a shopper's request to a shopping agent into a checklist. Answer: a single JSON object, no prose around it:
{"lines": [{"quote": "<request words naming the article>", "noun": "<article kind, shopper's own words>", "quantity": <units>}],
 "clauses": [{"quote": "<request words carrying this clause>", "type": "<type>", ...}, ...]}
Each line object and each clause object opens with "quote" (the request's words, copied letter for letter), then "type", then that type's fields.

Clause types, fields after "quote" and "type":
- value: {"article": <line index>, "property": "<shopper's wording>", "op": "is"|"not", "values": ["<each wanted value, copied>"], "unit": null}. A value the shopper copies from how listings label it, even one that itself negates ("not offered", "no entry"), keeps op "is" and that literal text. Two choices joined by "or" ("teal or mauve, either suits") form ONE clause holding both. Alternatives joined by "or" form one clause holding several values. A rejection of X ("not X", "anything but X") carries op "not".
- limit: {"article", "property", "op": "min"|"max"|"between", "values": ["<figure, copied>"], "unit": "<copied, else null>"}. Floor words (at least, minimum, upward of, or higher) give min; ceiling words (at most, maximum, not above, up to, under, or lower) give max; a range gives between with two figures. Units are never converted.
- reference: the value gets read off a DIFFERENT listing's page, whatever words send the agent there ("match whatever Xyzco's page shows", "copy the figure off Abc's listing", "or further along, wherever its page forwards you"). {"article", "property": "<the bought article's attribute whose value that listing decides, in shopper wording; always filled in>", "relation": "same"|"not"|"min"|"max"|"near"|"fit", "kind": "<what that other listing sells; frequently another article kind>", "with": {"<property>": "<value>"}, "seller": "<its seller; a name placed before the kind ('a Kestrelmoor whisk') is the seller>", "rule": "<tolerance, or a selection rule among several matches, copied; else null>"}. Relation: equal wording gives same; "unlike the one", "anything save the one" give not; "or above it" gives min; "or below it" gives max; "within N of it" gives near with N in rule; "a sort suited to it" gives fit. Each detail identifying that other listing (its colour, size, grade, bundle ...) belongs in "with", never among the article's values.
- when: a requirement holding only under a condition on a second article. {"mode": "if"|"unless", "cond": {"article", "property", "values": [...]}, "then": {"article", "property", "op", "values", "unit"}}. "If A's P ends up X, then B's Q must read Y" gives mode if; "B's Q must read Y, save when A's P is X" gives mode unless.
- fit: {"article": <the part>, "other": <the article it accompanies>} for "must accompany the chosen <other article>".
- stock: {} when sold-out stock is excluded, or only listings purchasable at once may be bought.
- budget: {"amount": <figure, copied>, "unit": "<copied>", "article": <line index when one article's price is capped; else null>}. Any money cap (a ceiling, a maximum spend, "nothing above N") counts as budget, never limit; price is no property.
- message_cap: {"count": <n>}: the maximum count of shopper messages.
- step_cap: {"count": <n>}: the maximum number of tool calls, steps or actions.
- priority: {"kind": "cheapest"|"best", "detail": "<what best means; else null>"} whenever a sentence says which fitting listing to choose (cheapest first, or best by some named measure). The least overall spend, when wished for, likewise counts as cheapest. The fallback instruction inside a decline sentence (how the fallback gets picked should no listing qualify) is no priority.
- decline: {"then": "<the closing words, copied>", "offer_instead": <true when that same clause also wants some listing proposed, shown or bought instead, else false>} for any clause saying what happens should no listing qualify. Excluding sold-out stock belongs to type stock, never decline.
- confirm: {} solely for a request stating outright the basket wants shopper approval ahead of ordering. Hoping for an immediate order, or inviting questions, is no confirm.
- before_order: {"instruction": "<copied>"} solely where the request plainly directs the agent, at checkout time, to report a stated thing or raise a stated question. A shopper musing aloud (should anything get mentioned, what might the agent require) yields a hint, never before_order; when unsure, hint.
- rule: {"text": "<copied>"} covering every remaining instruction changing what may be bought, or the agent's conduct.
- hint: {"article", "property": "<the property's store-style label, where the story points toward it; else null>"} for a story sentence pointing toward an unsettled property (the eventual user, the place, an earlier failure, a point beyond the shopper's judgement) without naming a value.

Reading rules:
- "quote" is copied letter for letter straight off the request wording: the sentence, or a fragment of it, carrying the clause. A clause lacking a quote, or whose quote is absent from that wording, gets thrown away; never paraphrase, never shorten.
- lines: one per distinct article kind wanted, in naming order. A count placed before the kind ("2 x", "three sets of", "Third: 2") gives the quantity; absent a count, 1. A shelf named with "and", "&" or "or" inside one phrase ("thimbles, bobbins and spools"; "footrests and ottomans") forms ONE line, whatever its length. The article kind and its department are never values. A reference's other listing never becomes a line, whatever it sells.
- "my van, a Q123 model, must take it", likewise "my bike is a Q123; the part belongs on it" give a value clause, property "vehicle", value Q123.
- Money, availability, unit counts, delivery wishes, greetings, thanks and story yield neither value nor limit. "the <prop> of <value>", likewise "with a <prop> of <value>", give a value clause, never a line.
- Values and figures are copied, the shopper's spelling kept intact; the property is named the shopper's way.
- "article" indexes lines from 0; a clause covering everything bought uses null."""

REPLY_SYS = """One message went from a shopping agent to its shopper, asking about one or more properties of one article; the shopper's reply follows and speaks to at most one of them. Determine the reply's upshot. Answer: a single JSON object, no prose around it:
{"kind": "value"|"reference"|"no_preference"|"not_understood"|"ask_singly"|"other",
 "property": "<the property the reply speaks of, worded the shopper's way; the asked property only when one was asked and none is named>",
 "op": "is"|"not"|"min"|"max"|"between", "values": ["<each value, copied>"], "unit": "<copied, else null>",
 "quote": "<the reply's words carrying the value>",
 "reference": {"relation": "same"|"not"|"min"|"max"|"near"|"fit", "kind": "<OTHER listing's article kind>", "seller": "<its seller>", "with": {"<property>": "<value>"}, "rule": "<else null>"} else null}

Kinds:
- value: the shopper states what a property must be. The reply may name a property other than the one asked; a requirement stated on any property is still kind value, "property" naming the one the reply speaks of. First settle which values the shopper ACCEPTS. Whenever the shopper echoes a listing's own label as the wanted value, even one that itself negates ("not offered", "no entry"), op stays "is" with that literal text. If X alone is accepted and all else refused, however phrased ("X, nothing besides passes", "X or I walk", "all but X gets tossed", "I refuse all but X", "show me nothing that reads other than X", "X, not a jot less"), the clause is op "is", value X. Op "not" applies only where X itself is refused ("not X", "anything but X", "X is out"). Read the whole sentence before choosing: a sentence that turns down every option except X requires X (op "is"), whatever negative words it uses. A rhetorical question naming the value ("Did I not say P means X?", "Why accept less than X on P?") is plainly value X. "X or Y" gives one clause holding two values. A floor figure gives min; a ceiling figure gives max; figure and unit copied. Values hold the value alone, never framing words ("other than", "apart from").
- reference: the shopper sends the agent to another listing's page to fetch the value ("copy whatever Abc's page displays", "or further along, wherever its page forwards you"): "values" stays empty; "reference" names that listing; details after the seller go in "with", as before.
- no_preference: whichever value, the shopper accepts; indifferent, leaving the matter open, or drifting to another topic, naming nothing concerning it.
- not_understood: the shopper lost the thread, requesting a repeat, plainer or shorter.
- ask_singly: the shopper requests one matter per message, in sequence, the questions split up.
- other: nothing listed above (a counter-question, unrelated news).
"values" holds words copied from the reply: never a whole sentence, never "any", never a value the shopper refuses."""

NOTICE_SYS = """Messages reached a shopping agent mid-task, shopper-sent or embedded within a tool result. Read them for changes to the job. Answer: a single JSON object, no prose around it:
{"budget": {"amount": <new figure, copied>, "unit": "<copied>", "quote": "<words>"} else null,
 "dropped": [{"article": <index else null>, "property": "<the property no longer wanted, worded as the message words it>", "requirement": "<the same property's name copied from REQUIREMENTS IN FORCE, else null>", "quote": "<words>"}],
 "quantity": [{"article": <index>, "count": <n>, "quote": "<words>"}],
 "gone": {"quote": "<words>"} else null,
 "price": {"quote": "<words>"} else null,
 "values": [{"article": <index else null>, "property": "...", "op": "is"|"not"|"min"|"max"|"between", "values": [...], "unit": null, "quote": "<words>"}]}
- budget: a fresh spending total, whether raised or cut.
- dropped: a withdrawn requirement ("forget the colour", "strike X off", "bin the size", "chuck it", "carry on without the material"). A message that tells you to let go of a requirement in force is a drop, whatever slang it uses, and never a new value.
- quantity: a changed unit count for an article.
- gone: the pick has become unavailable (taken by another buyer, sold out, vanished).
- price: the pick's price moved.
- values: a requirement newly stated inside a message, wanting a value; never the words that withdraw a requirement.
Every "quote" is copied letter for letter from the messages. Omit whatever the messages fail to say."""

WORDING_SYS = """Each numbered line pairs a wanted value against the words a store listing shows under that property, plus their placement (fact, title, property, description). The wanted words were already searched for inside the listing's words and turned up nowhere verbatim. Classify, per line, how the listing's words relate to the wanted value, choosing exactly one category:
- spelling: the identical words in a spelling variant ("fibre" / "fiber", "armour-plated" / "armor-plated").
- standard: the identical quantity or standard under its other standard name ("a dozen" for "twelve"; a resolution label for its pixel count).
- yesno: a yes/no fact answering a wanted "no ..." / "without ..." value ("no" beneath "assembly needed" for "no assembly").
- suffix: a wanted word appears with a suffix added or dropped ("glossed" for "gloss"; "silken" for "silk").
- inserted: an extra word sits inside the wanted phrase ("teak slat top" for "teak top").
- missing: a wanted word has no counterpart at all ("velvet trim" for "crimson velvet trim").
- sibling: a neighbouring value, a compound sharing a first word, a loose trade synonym, or another degree ("quick-fold" for "quick-lock"; "fast-charge" for "fast-dry"; "quick-release" for "quick-detach"; "soft" for "softest").
- abbreviation: shortened forms ("XS, M, XL" for "medium"; "wt" for "white").
- combination: joined with "+" or "and" ("tea + coffee" for "tea").
- part: a maker or part merely belonging to the wanted thing (a chip's series label for its maker).
- elsewhere: the words sit solely within description or small print rather than within a fact, title or property.
- different: anything else.
Answer: a single JSON object, no prose around it: {"kinds": {"<line number>": "<category>"}}"""

KEYMAP_SYS = """A shopper names a property in personal wording; a store's listings carry property keys. For each numbered line, list every key holding that very property, meaning the identical thing carrying another label (the shopper's "scent" becoming the store's "fragrance note"; "tint" its "finish tone"). A key for a sibling property on one part fails, even where trade talk blurs them: a differing last word marks another property ("sole pattern" differs from "sole thickness"; "lid style" differs from "lid shape"); nor does a key merely listing the variants on sale qualify. Copy keys exactly; give none when no key holds that property.
Answer: a single JSON object, no prose around it: {"keys": {"<line number>": ["<key>", ...]}}"""


def rd_tool_digest(tools: list[dict], limit: int = 2500) -> str:
    lines = []
    for t in tools or []:
        f = t.get("function", t) if isinstance(t, dict) else None
        if not isinstance(f, dict) or not f.get("name"):
            continue
        props = ((f.get("parameters") or {}).get("properties") or {})
        args = ", ".join(f"{k}: {(v or {}).get('type', '?')}" for k, v in props.items() if isinstance(v, dict))
        lines.append(f"- {f['name']}({args}): {rd_text(f.get('description'))[:160]}")
    return "\n".join(lines)[:limit]


def first_user(request: str, digest: str = "", opening: Any = None) -> str:
    parts = [f"REQUEST:\n{request[:7000]}"]
    if opening:
        parts.append(f"STORE OPENING NOTE:\n{rd_text(opening)[:1200]}")
    if digest:
        parts.append(f"STORE TOOLS (one tool call is one step):\n{digest}")
    return "\n\n".join(parts)


def read_request(call: Callable[[str, str, int], str], request: str, digest: str = "", opening: Any = None,
                 fix_call: Callable[[str, str, int], str] | None = None) -> dict:
    raw = call(FIRST_SYS, first_user(request, digest, opening), 3000)
    data = rd_json(raw)
    if not isinstance(data, dict):
        return {"clauses": [], "dropped": [{"why": "no json", "raw": rd_text(raw)[:300]}], "ok": False}
    out = ground_request(data, request, "\n".join([digest, rd_text(opening) if opening else ""]))
    name_reference_properties(fix_call or call, out["clauses"])
    out["ok"] = True
    return out


def name_reference_properties(call: Callable[[str, str, int], str], clauses: list[dict]) -> None:
    nouns = {c["article"]: c["noun"] for c in clauses if c["type"] == "line"}
    for c in clauses:
        if c["type"] != "reference" or c.get("property"):
            continue
        got = read_reply(call, "", "", c["quote"], nouns.get(c["article"], ""))
        prop = rd_text(got.get("property")).strip()[:60]
        if prop:
            c["property"] = prop


def clause_needles(raw: dict, typ: str) -> list[Any]:
    if typ in ("value", "limit"):
        return clean_list(raw.get("values") if raw.get("values") is not None else raw.get("value"))
    if typ == "reference":
        return [raw.get("seller")]
    if typ == "budget":
        return [to_number(raw.get("amount"))]
    if typ in ("message_cap", "step_cap"):
        return [clean_int(raw.get("count"))]
    if typ == "when":
        cond, then = raw.get("cond"), raw.get("then")
        return (clean_list(cond.get("values")) if isinstance(cond, dict) else []) + (clean_list(then.get("values")) if isinstance(then, dict) else [])
    if typ == "rule":
        return [raw.get("text")]
    if typ == "before_order":
        return [raw.get("instruction")]
    return []


def unit_after(vals: list[str], text: str, request: str) -> str | None:
    whole = norm(request)
    for v in vals:
        n = to_number(v)
        if n is None:
            continue
        figure = re.escape(rd_text(v).strip())
        m = re.search(figure + r"\s*([a-zA-Z%\u00b5\u00b0\"']{1,12})\b", norm(text))
        if m and to_number(m.group(1)) is None and unit_word(m.group(1), whole):
            return m.group(1)[:20]
    return None


def unit_word(word: str, text: str) -> bool:
    if not word.isalpha():
        return True
    seen = [x.start() for x in re.finditer(r"(?<![a-z0-9])" + re.escape(word.lower()) + r"(?![a-z0-9])", text)]
    return all(re.search(r"\d\s*$", text[:i]) for i in seen)


def joined_phrase(text: str, a: str, b: str) -> str:
    def part(x: str) -> str:
        return r"\W+".join(re.escape(t) + r"\w{0,2}" for t in tokens(x))
    if not tokens(a) or not tokens(b):
        return ""
    m = re.search(r"\b" + part(a) + r"\s*[&,]\s*" + part(b) + r"\b", norm(text))
    return m.group(0) if m else ""


def ends_with(long_noun: str, short_noun: str) -> bool:
    a, b = key_words(long_noun), key_words(short_noun)
    return bool(b) and len(a) > len(b) and a[-len(b):] == b


def counted_apart(a: dict, b: dict, request: str) -> bool:
    text = norm(request)
    spans = []
    for c in (a, b):
        q, words = norm(c.get("quote")), tokens(c.get("noun"))
        at = text.find(q) if q else -1
        noun = r"\W+".join(re.escape(w[:max(3, len(w) - 2)]) + r"\w{0,4}" for w in words)
        if at < 0 or not words or not re.search(r"(?<![\w.])\d+\W+(?:\w{1,2}\W+)?" + noun + r"(?!\w)", q):
            return False
        spans.append((at, at + len(q)))
    (s1, e1), (s2, e2) = sorted(spans)
    return e1 <= s2 and bool(re.search(r"[,.;!?]|\s-+\s", text[s1:s2]))


def merge_lines(lines: list[dict], request: str) -> list[dict]:
    out: list[dict] = []
    for c in lines:
        same = next((o for o in out if norm(o["noun"]) == norm(c["noun"]) or (o["quote"] and norm(o["quote"]) == norm(c["quote"]))
                     or ((ends_with(o["noun"], c["noun"]) or ends_with(c["noun"], o["noun"])) and not counted_apart(o, c, request))), None)
        if same:
            c["merged"] = out.index(same)
            same["quantity"] = max(same["quantity"], c["quantity"])
            if norm(same["quote"]) == norm(c["quote"]) and norm(same["noun"]) != norm(c["noun"]):
                same["noun"] = same["quote"][:80]
            elif len(key_words(c["noun"])) > len(key_words(same["noun"])):
                same["noun"] = c["noun"]
            continue
        prev = out[-1] if out else None
        if prev and c["quantity"] == 1 and prev["quantity"] == 1 and not numbers_in(c["quote"]):
            joined = joined_phrase(request, prev["noun"], c["noun"])
            if joined:
                c["merged"] = len(out) - 1
                prev["noun"] = joined[:80]
                prev["quote"] = joined if quote_in(joined, request) else prev["quote"]
                continue
        c["merged"] = len(out)
        out.append(dict(c, article=len(out)))
    return out


def cap_source(typ: str, quote: str, request: str, aside: str) -> str:
    if typ != "message_cap" or not aside.strip() or quote_in(quote, request) or not quote_in(quote, aside):
        return request
    return aside


def ground_request(data: dict, request: str, aside: str = "") -> dict:
    clauses: list[dict] = []
    dropped: list[dict] = []

    def drop(c: Any, why: str) -> None:
        dropped.append({"why": why, "clause": c})

    lines = data.get("lines") if isinstance(data.get("lines"), list) else []
    for raw in lines[:8]:
        if not isinstance(raw, dict):
            continue
        noun = rd_text(raw.get("noun")).strip()[:80]
        quote = rd_text(raw.get("quote")).strip()
        if not noun:
            drop(raw, "line without noun")
            continue
        if not quote_in(quote, request):
            quote = find_quote(request, [noun] + ([raw.get("quantity")] if clean_int(raw.get("quantity")) not in (None, 1) else []))
            if not quote and quote_in(noun, request):
                quote = noun
            if not quote:
                drop(raw, "line quote not in request")
                continue
        qty = clean_int(raw.get("quantity")) or 1
        if qty != 1 and not number_said(qty, quote) and float(qty) not in numbers_in(request):
            qty = 1
        clauses.append({"type": "line", "article": len(clauses), "noun": noun, "quantity": max(1, min(qty, 99)),
                        "quote": quote})
    unmerged = list(clauses)
    clauses = merge_lines(clauses, request)
    for c in clauses:
        c.pop("merged", None)
    remap = {c["article"]: c["merged"] for c in unmerged if "merged" in c}
    n_lines = len(clauses)
    if n_lines == 0:
        clauses.append({"type": "line", "article": 0, "noun": " ".join(tokens(request)[-3:]) or "item", "quantity": 1,
                        "quote": ""})
        n_lines = 1

    def art(v: Any, allow_none: bool = True) -> int | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        if isinstance(v, (int, float)) and int(v) in remap:
            v = remap[int(v)]
        if isinstance(v, (int, float)) and 0 <= int(v) < n_lines:
            return int(v)
        if isinstance(v, str) and v.strip().isdigit() and int(v) < n_lines:
            return int(v)
        return None if allow_none or n_lines > 1 else 0

    refs: list[dict] = []
    for raw in (data.get("clauses") if isinstance(data.get("clauses"), list) else [])[:80]:
        if not isinstance(raw, dict):
            continue
        typ = rd_text(raw.get("type")).strip().lower().replace(" ", "_").replace("-", "_")
        quote = rd_text(raw.get("quote")).strip()
        if typ not in CLAUSE_TYPES or typ == "line":
            drop(raw, "unknown type")
            continue
        src = cap_source(typ, quote, request, aside)
        if not quote_in(quote, src):
            quote = find_quote(src, clause_needles(raw, typ))
            if not quote:
                drop(raw, "quote not in request")
                continue
        c: dict[str, Any] = {"type": typ, "quote": quote}
        a = art(raw.get("article"), allow_none=True)
        if a is None and n_lines == 1 and typ in ("value", "limit", "reference", "hint", "fit"):
            a = 0
        if typ in ("value", "limit"):
            prop = rd_text(raw.get("property")).strip()[:60]
            vals = clean_list(raw.get("values") if raw.get("values") is not None else raw.get("value"))
            if a is None or not prop or not vals:
                drop(raw, "value without article, property or values")
                continue
            op = clean_op(raw.get("op"), "is" if typ == "value" else "max")
            if typ == "limit" and op in ("is", "not"):
                typ = "value"
            if typ == "value" and op in ("min", "max", "between"):
                typ = "limit"
            if not all(value_in(v, quote) or value_in(v, request) for v in vals):
                drop(raw, "value words not in quote")
                continue
            if op in ("min", "max", "between") and not any(to_number(v) is not None for v in vals):
                op = "is"
                typ = "value"
            if typ == "value" and op == "is":
                vals = [label_value(prop, v) for v in vals]
            c.update({"type": typ, "article": a, "property": prop, "op": op, "values": vals,
                      "unit": (rd_text(raw.get("unit")).strip()[:20] or None)})
            if c["unit"] and c["unit"].lower() in ("null", "none"):
                c["unit"] = None
            if typ == "limit" and not c["unit"]:
                c["unit"] = unit_after(vals, quote, request) or unit_after(vals, request, request)
        elif typ == "reference":
            prop = rd_text(raw.get("property")).strip()[:60]
            seller = rd_text(raw.get("seller")).strip()[:60]
            if seller.lower() in ("null", "none"):
                seller = ""
            if a is None:
                drop(raw, "reference without article")
                continue
            if seller and not value_in(seller, request):
                drop(raw, "seller not in request")
                continue
            rel = rd_text(raw.get("relation")).strip().lower()
            withs = {k: v for k, v in clean_with(raw.get("with")).items() if value_in(v, request)}
            if not seller and not withs and not rd_text(raw.get("kind")).strip():
                drop(raw, "reference names no listing")
                continue
            rule = rd_text(raw.get("rule")).strip()[:200]
            c.update({"article": a, "property": prop or None, "relation": relation_of(rel, quote),
                      "kind": rd_text(raw.get("kind")).strip()[:60] or None, "seller": seller or None, "with": withs,
                      "rule": rule if rule and rule.lower() not in ("null", "none") else None})
            refs.append(c)
        elif typ == "when":
            cond, then = raw.get("cond"), raw.get("then")
            if not isinstance(cond, dict) or not isinstance(then, dict):
                drop(raw, "when without cond/then")
                continue
            ca, ta = art(cond.get("article"), True), art(then.get("article"), True)
            cvals, tvals = clean_list(cond.get("values")), clean_list(then.get("values"))
            cprop, tprop = rd_text(cond.get("property")).strip()[:60], rd_text(then.get("property")).strip()[:60]
            if ta is None or not tprop or not tvals or not cprop or not cvals:
                drop(raw, "when incomplete")
                continue
            if not all(value_in(v, quote) for v in cvals + tvals):
                drop(raw, "when values not in quote")
                continue
            mode = rd_text(raw.get("mode")).strip().lower()
            c.update({"mode": mode if mode in ("if", "unless") else "if",
                      "cond": {"article": ca, "property": cprop, "values": cvals},
                      "then": {"article": ta, "property": tprop, "op": clean_op(then.get("op")), "values": tvals,
                               "unit": rd_text(then.get("unit")).strip()[:20] or None}})
        elif typ == "fit":
            other = art(raw.get("other"), True)
            if a is None or other is None or other == a:
                code = re.search(r"\b[A-Z]{2,}[ -]?\d{3,}\b", quote)
                if a is not None and code and not any(x["type"] in ("value", "limit") and x["article"] == a
                                                      and any(value_in(code.group(0), v) for v in x["values"]) for x in clauses):
                    c.update({"type": "value", "article": a, "property": "vehicle", "op": "is",
                              "values": [code.group(0)], "unit": None})
                    clauses.append(c)
                    continue
                drop(raw, "fit without two articles")
                continue
            c.update({"article": a, "other": other})
        elif typ == "budget":
            amount = to_number(raw.get("amount"))
            if amount is None or amount <= 0:
                drop(raw, "budget without amount")
                continue
            if not number_said(amount, quote):
                quote = find_quote(request, [amount])
                if not quote:
                    drop(raw, "budget amount not in quote")
                    continue
                c["quote"] = quote
            c.update({"amount": amount, "unit": rd_text(raw.get("unit")).strip()[:30] or None,
                      "article": a if n_lines > 1 else None, "per": "unit" if a is not None else "order"})
        elif typ in ("message_cap", "step_cap"):
            count = clean_int(raw.get("count"))
            if count is not None and not number_said(count, quote):
                quote = find_quote(src, [count])
                c["quote"] = quote
            if count is None or not quote:
                drop(raw, "cap count not in quote")
                continue
            c.update({"count": count})
        elif typ == "priority":
            kind = rd_text(raw.get("kind")).strip().lower()
            if kind not in ("cheapest", "best"):
                drop(raw, "priority kind unknown")
                continue
            detail = rd_text(raw.get("detail")).strip()[:200]
            c.update({"kind": kind, "detail": detail if detail.lower() not in ("", "null", "none") else None})
        elif typ == "decline":
            then = rd_text(raw.get("then")).strip()[:300]
            offer = raw.get("offer_instead")
            c.update({"then": then if then.lower() not in ("", "null", "none") else None,
                      "offer_instead": offer is True or rd_text(offer).strip().lower() == "true"})
        elif typ == "before_order":
            instr = rd_text(raw.get("instruction")).strip()[:300]
            if "?" in quote or "?" in instr:
                c.update({"type": "hint", "article": a, "property": None})
            else:
                c.update({"instruction": instr or quote})
        elif typ == "rule":
            c.update({"text": rd_text(raw.get("text")).strip()[:300] or quote})
        elif typ == "hint":
            prop = rd_text(raw.get("property")).strip()[:60]
            c.update({"article": a, "property": prop if prop.lower() not in ("", "null", "none") else None})
        elif typ in ("confirm", "stock"):
            pass
        clauses.append(c)

    kept: list[dict] = []
    seen: set[str] = set()
    for c in clauses:
        if c["type"] in ("value", "limit"):
            owner = [r for r in refs if r["article"] == c["article"]]
            shadow = False
            for r in owner:
                wv = {norm(v) for v in r["with"].values()}
                wk = {norm(k) for k in r["with"]}
                if any(norm(v) in wv for v in c["values"]) and (norm(c["property"]) in wk or quote_in(c["quote"], r["quote"])
                                                               or quote_in(r["quote"], c["quote"])):
                    shadow = True
                if norm(c["property"]) == norm(r["property"]) and quote_in(c["quote"], r["quote"]):
                    shadow = True
            if shadow:
                dropped.append({"why": "value belongs to a reference's other listing", "clause": c})
                continue
        key = json.dumps({k: v for k, v in c.items() if k != "quote"}, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return {"clauses": fix_line_counts(kept), "dropped": dropped}


def label_value(prop: str, value: str) -> str:
    words, said = tokens(prop), tokens(value)
    if len(said) != 1 or len(words) < 2 or set(words[1:]) & set(said):
        return value
    w, v = words[0], said[0]
    if w == v or len(w) < 5 or len(v) < 5 or w[:5] != v[:5] or abs(len(w) - len(v)) > 2:
        return value
    return " ".join([value.strip()] + words[1:])


def fix_line_counts(clauses: list[dict]) -> list[dict]:
    for c in clauses:
        if c.get("type") != "line" or c.get("quantity") in (None, 1):
            continue
        q = float(c["quantity"])
        held = [v for x in clauses if x.get("type") in ("value", "limit") and x.get("article") == c.get("article") for v in x.get("values") or []]
        if not any(q in numbers_in(v) for v in held):
            continue
        said, noun = tokens(c.get("quote")), tokens(c.get("noun"))
        at = next((i for i in range(len(said)) if noun and said[i:i + len(noun)] == noun), None)
        if at is not None and q not in numbers_in(" ".join(said[:at])):
            c["quantity"] = 1
    return clauses


def read_reply(call: Callable[[str, str, int], str], question: str, prop: str, reply: str, article: str = "") -> dict:
    user = (f"ARTICLE: {article}\n" if article else "") + f"PROPERTY ASKED: {prop}\nQUESTION: {question}\nREPLY: {reply}"
    data = rd_json(call(REPLY_SYS, user, 500))
    return ground_reply(data, prop, reply, article)


def asked_list(prop: Any) -> list[str]:
    return [k.strip() for k in rd_text(prop).split(",") if k.strip()]


def reply_property(p: str, prop: str, reply: str) -> str:
    if not p:
        return prop
    if value_in(p, reply) or norm(p) == norm(prop) or any(norm(p) == norm(k) for k in asked_list(prop)):
        return p
    said = tokens(reply)
    if any(len(w) >= 4 and any(same_token(w, h) for h in said) for w in tokens(p)):
        return p
    return prop


def ref_sound(ref: dict, data: dict, reply: str, article: str) -> bool:
    st = tokens(ref.get("seller"))
    if not st:
        return False
    if article and rd_contains(tokens(article), st):
        return False
    if rd_text(data.get("kind")).strip().lower() == "value":
        vals = [v for v in clean_list(data.get("values") if data.get("values") is not None else data.get("value")) if value_in(v, reply)]
        if any(rd_contains(tokens(v), st) for v in vals):
            return False
    return True


def plain_part(reply: Any, quote: Any, seller: Any) -> str:
    text = rd_text(reply)
    for cut in (rd_text(quote).strip(), rd_text(seller).strip()):
        at = text.lower().find(cut.lower()) if cut else -1
        if at >= 0:
            return " ".join((text[:at] + " " + text[at + len(cut):]).split())
    return ""


def ref_as_value(data: dict, ref: Any, prop: str, reply: str) -> dict:
    vals = [v for v in clean_list(data.get("values") if data.get("values") is not None else data.get("value")) if value_in(v, reply)]
    own = tokens(prop)
    if not vals and isinstance(ref, dict) and own:
        for k, v in clean_with(ref.get("with")).items():
            kt = tokens(k)
            if kt and (rd_contains(own, kt) or rd_contains(kt, own)) and value_in(v, reply):
                vals.append(v)
    if not vals:
        return dict(data, kind="other", reference=None)
    return dict(data, kind="value", values=vals, reference=None)


def ground_reply(data: Any, prop: str, reply: str, article: str = "") -> dict:
    out: dict[str, Any] = {"kind": "other", "property": prop, "op": "is", "values": [], "unit": None, "quote": "",
                           "reference": None}
    if not isinstance(data, dict):
        return out
    kind = rd_text(data.get("kind")).strip().lower().replace(" ", "_").replace("-", "_")
    out["kind"] = kind if kind in REPLY_KINDS else "other"
    p = rd_text(data.get("property")).strip()[:60]
    out["property"] = reply_property(p, prop, reply)
    quote = rd_text(data.get("quote")).strip()
    out["quote"] = quote if quote_in(quote, reply) else ""
    ref = data.get("reference")
    raw_ref = ref if isinstance(ref, dict) else {}
    if isinstance(ref, dict) and rd_text(ref.get("seller")).strip() and value_in(ref.get("seller"), reply) and not ref_sound(ref, data, reply, article):
        if out["kind"] == "value":
            out["kind"] = "other"
            return out
        data = ref_as_value(data, ref, out["property"], reply)
        out["kind"] = data["kind"]
        ref = None
    elif out["kind"] == "reference" and not (isinstance(ref, dict) and rd_text(ref.get("seller")).strip() and value_in(ref.get("seller"), reply)):
        data = ref_as_value(data, ref, out["property"], reply)
        out["kind"] = data["kind"]
        ref = None
    if kind == "reference" and out["kind"] == "other":
        out["lost_ref"] = {"seller": rd_text(raw_ref.get("seller")).strip(), "kind": rd_text(raw_ref.get("kind")).strip(),
                           "with": clean_with(raw_ref.get("with"))}
    if isinstance(ref, dict) and rd_text(ref.get("seller")).strip() and value_in(ref.get("seller"), reply):
        rel = rd_text(ref.get("relation")).strip().lower()
        rule = rd_text(ref.get("rule")).strip()
        out["reference"] = {"relation": relation_of(rel, reply), "prop": out["property"],
                            "kind": rd_text(ref.get("kind")).strip()[:60] or None,
                            "seller": rd_text(ref["seller"]).strip()[:60],
                            "with": {k: v for k, v in clean_with(ref.get("with")).items() if value_in(v, reply)},
                            "rule": rule if rule and rule.lower() not in ("null", "none") else None}
        out["kind"] = "reference"
        return out
    if out["kind"] == "value":
        vals = [v for v in clean_list(data.get("values") if data.get("values") is not None else data.get("value"))
                if value_in(v, reply) and len(tokens(v)) <= 8]
        if not vals:
            out["kind"] = "other"
            return out
        op = clean_op(data.get("op"))
        if op in ("min", "max", "between") and (not any(to_number(v) is not None for v in vals) or all(named_figure(v) for v in vals)):
            op = "is"
        unit = rd_text(data.get("unit")).strip()[:20]
        out.update({"op": op, "values": vals, "unit": unit if unit and unit.lower() not in ("null", "none") else None})
    return out


def read_notice(call: Callable[[str, str, int], str], texts: list[str], articles: list[str], reqs: list[str] | None = None) -> dict:
    names = list(dict.fromkeys(n for n in (reqs or []) if n.strip()))[:30]
    user = ("ARTICLES: " + "; ".join(f"{i}: {a}" for i, a in enumerate(articles))
            + ("\nREQUIREMENTS IN FORCE: " + "; ".join(names) if names else "") + "\nMESSAGES:\n"
            + "\n".join(f"- {t}" for t in texts[-6:]))
    data = rd_json(call(NOTICE_SYS, user, 600))
    out = ground_notice(data, " \n ".join(texts), len(articles))
    known = {Text.key(n): n for n in names}
    for d in out["dropped"]:
        d["requirement"] = known.get(Text.key(d.get("requirement")))
    return out


def ground_notice(data: Any, text: str, n_articles: int) -> dict:
    out: dict[str, Any] = {"budget": None, "dropped": [], "quantity": [], "gone": None, "price": None, "values": []}
    if not isinstance(data, dict):
        return out

    def art(v: Any) -> int | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)) and 0 <= int(v) < n_articles:
            return int(v)
        if isinstance(v, str) and v.strip().isdigit() and int(v) < n_articles:
            return int(v)
        return 0 if n_articles == 1 else None

    b = data.get("budget")
    if isinstance(b, dict):
        amount = to_number(b.get("amount"))
        quote = rd_text(b.get("quote")).strip()
        if amount and amount > 0 and quote_in(quote, text) and number_said(amount, quote):
            out["budget"] = {"amount": amount, "unit": rd_text(b.get("unit")).strip()[:30] or None, "quote": quote}
    for d in data.get("dropped") if isinstance(data.get("dropped"), list) else []:
        if isinstance(d, dict):
            prop, quote = rd_text(d.get("property")).strip()[:60], rd_text(d.get("quote")).strip()
            if prop and quote_in(quote, text) and value_in(prop, text):
                out["dropped"].append({"article": art(d.get("article")), "property": prop, "quote": quote,
                                       "requirement": rd_text(d.get("requirement")).strip()[:60] or None})
    for q in data.get("quantity") if isinstance(data.get("quantity"), list) else []:
        if isinstance(q, dict):
            count, quote = clean_int(q.get("count")), rd_text(q.get("quote")).strip()
            a = art(q.get("article"))
            if count and a is not None and quote_in(quote, text) and number_said(count, quote):
                out["quantity"].append({"article": a, "count": count, "quote": quote})
    for key in ("gone", "price"):
        g = data.get(key)
        if isinstance(g, dict) and quote_in(rd_text(g.get("quote")), text):
            out[key] = {"quote": rd_text(g.get("quote")).strip()}
    for v in data.get("values") if isinstance(data.get("values"), list) else []:
        if isinstance(v, dict):
            got = ground_reply(dict(v, kind="value"), rd_text(v.get("property")), text)
            if got["kind"] == "value" and got["property"]:
                out["values"].append({"article": art(v.get("article")), "property": got["property"], "op": got["op"],
                                      "values": got["values"], "unit": got["unit"], "quote": got["quote"]})
    return out


def list_elements(listing: Any) -> list[list[str]]:
    s = norm(listing).replace("_", " ").replace("-", " ").replace("/", " ")
    parts = [p for p in re.split(r"\s*[,;|]\s*", s) if p.strip()]
    return [tokens(p) for p in parts] or [[]]


def joined_only(wanted: list[str], vals: list[str]) -> bool:
    want = [" ".join(w.lower().split()) for w in wanted]
    vals = [v.lower() for v in vals]
    if not want or not vals or any("-" in w for w in want):
        return False
    alone = [re.compile(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])") for w in want]
    joined = [re.compile(r"\w-" + re.escape(w) + r"(?!\w)|(?<![\w-])" + re.escape(w) + r"-\w") for w in want]
    return not any(x.search(v) for x in alone for v in vals) and any(x.search(v) for x in joined for v in vals)


def phrase_states(wanted: Any, listing: Any) -> bool:
    w = tokens(norm(wanted).replace("_", " ").replace("-", " "))
    if not w:
        return False
    hay_all = tokens(norm(listing).replace("_", " ").replace("-", " ").replace("/", " "))
    if "+" in rd_text(listing) and len(hay_all) > len(w):
        return False
    return any(rd_contains(el, w, True) for el in list_elements(listing)) and rd_contains(hay_all, w, True)


def rd_judge_wording(call: Callable[[str, str, int], str], pairs: list[tuple[str, str, str]]) -> dict[int, bool]:
    out: dict[int, bool] = {}
    ask = []
    for n, (wanted, listing, where) in enumerate(pairs):
        if norm(where) == "description":
            out[n] = False
        elif phrase_states(wanted, listing):
            out[n] = True
        else:
            ask.append(n)
    if not ask:
        return out
    user = "\n".join(f"{n}. wanted: {pairs[n][0]} | listing says: {pairs[n][1]} | shown in: {pairs[n][2] or 'a property'}"
                     for n in ask[:40])
    data = rd_json(call(WORDING_SYS, user, 400))
    kinds = data.get("kinds") if isinstance(data, dict) and isinstance(data.get("kinds"), dict) else {}
    for n in ask:
        k = rd_text(kinds.get(str(n), kinds.get(n))).strip().lower()
        out[n] = n < 40 and k in SAME_VALUE_KINDS
    return out


def key_words(k: Any) -> list[str]:
    return tokens(norm(k).replace("_", " ").replace("-", " "))


def exact_keys(prop: str, keys: list[str]) -> list[str]:
    p = key_words(prop)
    return [k for k in keys if key_words(k) == p]


def wider_keys(prop: str, keys: list[str]) -> list[str]:
    p = key_words(prop)
    return [k for k in keys if p and len(key_words(k)) > len(p) and key_words(k)[-len(p):] == p]


def rd_map_keys(call: Callable[[str, str, int], str], items: list[tuple[str, list[str]]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    ask = []
    for n, (prop, keys) in enumerate(items):
        hit = exact_keys(prop, keys)
        if hit:
            out[n] = hit
        else:
            ask.append(n)
    if not ask:
        return out
    user = "\n".join(f"{n}. property: {items[n][0]} | keys: {', '.join(items[n][1][:60])}" for n in ask[:12])
    data = rd_json(call(KEYMAP_SYS, user, 400))
    got = data.get("keys") if isinstance(data, dict) and isinstance(data.get("keys"), dict) else {}
    for n in ask:
        ks = got.get(str(n), got.get(n))
        ks = ks if isinstance(ks, list) else ([ks] if ks else [])
        valid = [rd_text(k) for k in ks if rd_text(k) in items[n][1]]
        out[n] = valid[:3] or wider_keys(items[n][0], items[n][1])[:3]
    return out


def ledger_value(values: list[str]) -> Any:
    return values[0] if len(values) == 1 else list(values)


def to_ledger(clauses: list[dict]) -> list[dict]:
    out: list[dict] = []
    for c in clauses:
        if c["type"] in ("value", "limit"):
            out.append({"a": c["article"], "prop": c["property"], "op": c["op"], "value": ledger_value(c["values"]),
                        "unit": c.get("unit"), "source": "stated", "active": True, "ref": None, "quote": c["quote"]})
        elif c["type"] == "reference":
            out.append({"a": c["article"], "prop": c["property"], "op": "ref", "value": None, "unit": None,
                        "source": "reference", "active": True, "quote": c["quote"],
                        "ref": {"article": c["article"], "prop": c["property"], "relation": c["relation"],
                                "kind": c.get("kind"), "seller": c["seller"], "with": dict(c["with"]), "rule": c.get("rule")}})
        elif c["type"] == "when":
            t = c["then"]
            out.append({"a": t["article"], "prop": t["property"], "op": t["op"], "value": ledger_value(t["values"]),
                        "unit": t.get("unit"), "source": "stated", "active": False, "ref": None, "quote": c["quote"],
                        "when": {"mode": c["mode"], "article": c["cond"]["article"], "prop": c["cond"]["property"],
                                 "values": list(c["cond"]["values"])}})
    return out


def reply_to_ledger(got: dict, article: int, source: str = "revealed") -> dict | None:
    if got.get("kind") == "reference" and got.get("reference"):
        r = dict(got["reference"], article=article)
        return {"a": article, "prop": r["prop"], "op": "ref", "value": None, "unit": None, "source": "reference",
                "active": True, "ref": r, "quote": got.get("quote", ""), "heard": True}
    if got.get("kind") == "value" and got.get("values"):
        return {"a": article, "prop": got["property"], "op": got["op"], "value": ledger_value(got["values"]),
                "unit": got.get("unit"), "source": source, "active": True, "ref": None, "quote": got.get("quote", "")}
    return None


def when_applies(entry: dict, value_of: Callable[[int | None, str], list[str]]) -> bool:
    w = entry.get("when")
    if not w:
        return bool(entry.get("active"))
    seen = [norm(v) for v in value_of(w["article"], w["prop"])]
    hit = any(phrase_states(v, s) for v in w["values"] for s in seen)
    return hit if w["mode"] == "if" else not hit


def summary(clauses: list[dict]) -> dict:
    s: dict[str, Any] = {"lines": [], "message_cap": None, "step_cap": None, "budget": None, "unit_budgets": {},
                         "priority": "none", "priority_detail": None, "decline": None, "confirm": False,
                         "before_order": None, "in_stock": False, "together": [], "rules": [], "hints": [], "whens": [], "offer_instead": False}
    for c in clauses:
        t = c["type"]
        if t == "line":
            s["lines"].append({"i": c["article"], "noun": c["noun"], "qty": c["quantity"]})
        elif t == "message_cap":
            s["message_cap"] = c["count"] if s["message_cap"] is None else min(s["message_cap"], c["count"])
        elif t == "step_cap":
            s["step_cap"] = c["count"] if s["step_cap"] is None else min(s["step_cap"], c["count"])
        elif t == "budget":
            if c.get("article") is None:
                if s["budget"] is None or (s["budget"].get("per") == "unit" and c.get("per") == "order"):
                    s["budget"] = {"amount": c["amount"], "unit": c.get("unit"), "per": c.get("per", "order")}
            else:
                s["unit_budgets"][c["article"]] = {"amount": c["amount"], "unit": c.get("unit")}
        elif t == "priority":
            s["priority"], s["priority_detail"] = c["kind"], c.get("detail")
        elif t == "decline":
            if not c.get("offer_instead"):
                s["decline"] = {"quote": c["quote"], "then": c.get("then")}
            else:
                s["offer_instead"] = True
        elif t == "confirm":
            s["confirm"] = True
        elif t == "before_order":
            s["before_order"] = c["instruction"]
        elif t == "stock":
            s["in_stock"] = True
        elif t == "fit":
            pair = [c["article"], c["other"]]
            if pair not in s["together"]:
                s["together"].append(pair)
        elif t == "rule":
            s["rules"].append(c["text"])
        elif t == "hint":
            s["hints"].append({"a": c.get("article"), "prop": c.get("property"), "quote": c["quote"]})
        elif t == "when":
            s["whens"].append(c)
    return s



class Store:
    DOES = ("search", "browse", "page", "facts", "compare", "cart", "remove", "view", "quote", "order", "decline", "finish", "message", "usage", "other")
    ARGS = ("query", "category", "brand", "price", "page", "id", "ids", "qty", "lines", "mode", "text", "token", "other")
    VALUES = ("order", "decline", "add", "remove", "words", "browse", "other")
    CLOSING = ("order", "decline", "finish")
    SINGLE = ("order", "decline", "finish", "view", "usage", "message", "quote")
    ARG_FITS = {"query": ("str",), "text": ("str",), "token": ("str",), "category": ("str", "enum"), "brand": ("str", "enum"),
                "price": ("num",), "page": ("num", "str"), "id": ("str", "num"), "ids": ("arr",), "lines": ("arr",), "qty": ("num",),
                "mode": ("enum", "bool")}
    VALUE_FITS = {"order": CLOSING, "decline": CLOSING, "add": ("cart", "remove"), "remove": ("cart", "remove"),
                  "words": ("search", "browse"), "browse": ("search", "browse")}
    NEEDS = {"search": ("query",), "browse": ("category", "brand", "price"), "page": ("id",), "facts": ("id",),
             "compare": ("ids",), "cart": ("qty", "lines", "ids"), "message": ("text",)}

    def __init__(self, schemas: dict, descs: dict) -> None:
        self.schemas = schemas
        self.descs = descs
        self.tm: dict[str, dict] = {}
        self.roles: dict[str, str] = {}
        for k in SLOTS:
            setattr(self, k, None)
        self.stock_true_available = True
        self.stock_yes: set[str] | None = None
        self.stats: dict[str, dict] = {}
        self.row_keys: set[str] = set()
        self.rows_seen = 0
        self.tallied: set[str] = set()
        self.queries: list[str] = []
        self.views: dict[str, dict] = {}
        self.ids: set[str] = set()
        self.cart_qty: str | None = None
        self.open: dict[str, list[str]] = {}
        self.fixed: set[str] = set()
        self.confirmed = False
        self.budget_hint: float | None = None
        self.money_odd = False
        self.budget_lost = False
        self.cmp_cap: int | None = None
        self.search_alts: list[str] = []
        self.log: list[str] = []
        self.sense: dict[str, dict] = {}
        self.sense_doubts: list[str] = []
        self.unsure: set[str] = set()
        self.assign()

    def sense_rows(self, d: Any, depth: int = 0) -> dict:
        if isinstance(d, list):
            d = {Text.of(x.get("tool") or x.get("name")): x for x in d if isinstance(x, dict)}
        if not isinstance(d, dict):
            return {}
        if set(d) & set(self.schemas) or depth > 1:
            return d
        for v in d.values():
            got = self.sense_rows(v, depth + 1)
            if set(got) & set(self.schemas):
                return got
        return {}

    @staticmethod
    def arg_kind(spec: Any) -> str | None:
        if not isinstance(spec, dict):
            return None
        if isinstance(spec.get("enum"), list) and spec["enum"]:
            return "enum"
        typ = spec.get("type")
        if isinstance(typ, list):
            typ = next((t for t in typ if t != "null"), None)
        return {"integer": "num", "number": "num", "string": "str", "array": "arr", "boolean": "bool", "object": "obj"}.get(typ)

    def kinds_of(self, name: str) -> tuple[dict, list]:
        sch = self.schemas.get(name) or {}
        props = {k: v for k, v in (sch.get("properties") or {}).items() if isinstance(v, dict)}
        return {k: self.arg_kind(v) for k, v in props.items()}, [k for k in (sch.get("required") or []) if k in props]

    def does_fits(self, name: str, does: str | None) -> bool:
        kinds, req = self.kinds_of(name)
        have = set(kinds.values())
        open_ = None in have

        def has(*ks: str) -> bool:
            return open_ or bool(have & set(ks))
        if does in ("search", "message"):
            return has("str")
        if does == "browse":
            return bool(kinds)
        if does in ("page", "facts", "remove"):
            return has("str", "num") and "arr" not in have
        if does == "compare":
            return has("arr")
        if does == "cart":
            return has("str", "num", "arr")
        if does in ("view", "usage", "quote"):
            return not req and "arr" not in have
        if does == "finish":
            return has("enum", "bool")
        return True

    def sense_entry(self, name: str, x: Any) -> tuple[dict, list[str]]:
        empty = {"does": None, "ends": False, "args": {}, "values": {}}
        if not isinstance(x, dict):
            return empty, [f"{name}: not read"]
        kinds, _ = self.kinds_of(name)
        doubts: list[str] = []
        raw = Text.of(x.get("does")).strip().lower()
        does = raw if raw in self.DOES else None
        if does is None:
            doubts.append(f"{name}: does {raw!r} is no listed word")
        given = x.get("args") if isinstance(x.get("args"), dict) else {}
        args: dict[str, str] = {}
        for a, v in given.items():
            lab = Text.of(v).strip().lower()
            if a not in kinds or lab not in self.ARGS:
                continue
            if lab in self.ARG_FITS and kinds[a] is not None and kinds[a] not in self.ARG_FITS[lab]:
                doubts.append(f"{name}: {a} read as {lab}, its type is {kinds[a]}")
                continue
            args[a] = lab
        unread = [a for a in kinds if a not in given]
        if unread:
            doubts.append(f"{name}: arguments not read {unread}")
        labs = set(args.values())
        if does is None:
            does = ("search" if "query" in labs else "cart" if "lines" in labs else "compare" if "ids" in labs
                    else "browse" if labs & {"category", "brand", "price"} else None)
        if does is not None and not self.does_fits(name, does):
            doubts.append(f"{name}: read as {does}, its arguments do not fit")
            does = None
        if does == "other":
            doubts.append(f"{name}: read as other")
        need = self.NEEDS.get(does or "")
        if need and not labs & set(need):
            doubts.append(f"{name}: read as {does} without an argument read as {'/'.join(need)}")
        vals: dict[str, dict] = {}
        for a, m in (x.get("values") or {}).items() if isinstance(x.get("values"), dict) else []:
            allowed = (self.schemas.get(name) or {}).get("properties", {}).get(a, {}).get("enum") if kinds.get(a) == "enum" else None
            if not isinstance(m, dict) or not isinstance(allowed, list):
                continue
            row = {str(v): Text.of(m.get(str(v))).strip().lower() for v in allowed}
            row = {v: w for v, w in row.items() if w in self.VALUES}
            odd = [w for w in row.values() if w in self.VALUE_FITS and does is not None and does not in self.VALUE_FITS[w]]
            if odd:
                doubts.append(f"{name}: values of {a} read as {odd} do not fit {does}")
                continue
            vals[a] = row
        ends = x.get("ends") in (True, "true")
        if ends and does not in self.CLOSING + (None, "other"):
            doubts.append(f"{name}: read as {does} that ends the visit")
            ends = False
        if does == "finish" and not any({"order", "decline"} <= set(r.values()) for r in vals.values()):
            doubts.append(f"{name}: read as finish without an order and a decline value")
        return {"does": does, "ends": ends, "args": args, "values": vals}, doubts

    def read_sense(self, d: Any) -> tuple[dict, list[str]]:
        d = self.sense_rows(d)
        if not d:
            return {}, ["no reading"]
        got: dict[str, dict] = {}
        doubts: list[str] = []
        for nm in self.schemas:
            got[nm], why = self.sense_entry(nm, d.get(nm))
            doubts += why
        if not any(v["does"] for v in got.values()):
            return {}, doubts + ["no tool read"]
        count = collections.Counter(v["does"] for v in got.values())
        doubts += [f"{count[r]} tools read as {r}" for r in self.SINGLE if count[r] > 1]
        if count["finish"] and (count["order"] or count["decline"]):
            doubts.append("finish read beside order or decline")
        direct = any(v["does"] in ("order", "finish") and self.kinds_of(nm)[0] and set(self.kinds_of(nm)[0].values()) & {"arr", "num"}
                     for nm, v in got.items())
        for role, ok in (("search", count["search"]), ("order", count["order"] + count["finish"]), ("decline", count["decline"] + count["finish"]),
                         ("cart", count["cart"] or direct), ("message", count["message"])):
            if not ok:
                doubts.append(f"no tool read as {role}")
        return got, doubts

    def take_senses(self, answers: list) -> bool:
        best: tuple[dict, list[str]] | None = None
        for d in answers:
            got, doubts = self.read_sense(d)
            if got and (best is None or len(doubts) <= len(best[1])):
                best = (got, doubts)
        if best is None:
            return False
        self.sense, self.sense_doubts = best
        self.assign()
        self.log.append("roles from the model's tool reading: " + ", ".join(f"{nm}={v['does']}" for nm, v in self.sense.items())
                        + (f"; doubts kept: {'; '.join(self.sense_doubts)[:300]}" if self.sense_doubts else ""))
        return True

    def take_sense(self, d: Any) -> bool:
        return self.take_senses([d])

    def desc_words(self, name: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", Text.of(self.descs.get(name)).lower())

    def word_weights(self) -> dict:
        n = max(1, len(self.schemas))
        df = collections.Counter(w for nm in self.schemas for w in set(self.desc_words(nm)))
        return {w: max(0.0, 1.0 - c / n) for w, c in df.items()}

    def shared_run(self, a: str, b: str) -> tuple[float, set]:
        wa, wb = self.desc_words(a), self.desc_words(b)
        others = [" " + " ".join(self.desc_words(n)) + " " for n in self.schemas if n not in (a, b)]
        wt = self.word_weights()
        best: tuple[float, set] = (0.0, set())
        for i in range(len(wa)):
            for j in range(len(wb)):
                k = 0
                while i + k < len(wa) and j + k < len(wb) and wa[i + k] == wb[j + k]:
                    k += 1
                    run = set(wa[i:i + k])
                    score = sum(wt.get(w, 0.0) for w in run)
                    if k >= 2 and score > best[0] and not any(" " + " ".join(wa[i:i + k]) + " " in o for o in others):
                        best = (score, run)
        return best

    def other_overlap(self, name: str, pair: tuple, skip: set) -> float:
        wt = self.word_weights()
        others = [set(self.desc_words(nm)) for nm in self.schemas if nm not in pair]
        return sum(wt.get(w, 0.0) * sum(w in o for o in others) for w in set(self.desc_words(name)) - skip if wt.get(w, 0.0) > 0.3)

    def finish_tools(self, free: list[str]) -> list[tuple]:
        fins = []
        for nm in free:
            kinds, req = self.kinds_of(nm)
            enums = [k for k, v in kinds.items() if v == "enum"]
            vals = ((self.schemas[nm].get("properties") or {}).get(enums[0]) or {}).get("enum") if len(enums) == 1 else None
            if (vals and len(vals) == 2 and all(isinstance(v, str) for v in vals) and enums[0] in req
                    and all(v in ("enum", "str") for v in kinds.values()) and [a for a, v in kinds.items() if v == "str" and a != self.tokens.get(nm)][1:] == []):
                fins.append((nm, enums[0], vals))
        return fins

    def closing_pair(self, free: list[str]) -> tuple[str, str] | None:
        plain = [nm for nm in free if all(v == "str" for v in self.kinds_of(nm)[0].values()) and all(a == self.tokens.get(nm) for a in self.kinds_of(nm)[1])]
        pairs = sorted(((self.shared_run(a, b), a, b) for i, a in enumerate(plain) for b in plain[i + 1:]), key=lambda x: -x[0][0])
        if not pairs or not pairs[0][0][1] or (len(pairs) > 1 and pairs[1][0][0] >= pairs[0][0][0]):
            return None
        (_, run), a, b = pairs[0]
        if (a in self.tokens) != (b in self.tokens):
            return (a, b) if a in self.tokens else (b, a)
        sa, sb = self.other_overlap(a, (a, b), run), self.other_overlap(b, (a, b), run)
        return (a, b) if sa >= sb else (b, a)

    def cart_words(self, skip: set[str]) -> set[str]:
        wt = self.word_weights()
        names = {(self.tm.get(r) or {}).get("name") for r in ("cart", "view", "remove")} - {None}
        return {w for nm in names for w in self.desc_words(nm) if len(w) >= 3 and wt.get(w, 0.0) >= 0.5 and w not in skip}

    def pair_order(self, a: str, b: str) -> str | None:
        wa, wb = set(self.desc_words(a)), set(self.desc_words(b))
        cart = self.cart_words(wa & wb)
        votes = set()
        ca, cb = len(wa & cart), len(wb & cart)
        if ca != cb and min(ca, cb) == 0:
            votes.add(a if ca else b)
        note = {nm: any(kd == "str" and x != self.tokens.get(nm) for x, kd in self.kinds_of(nm)[0].items()) for nm in (a, b)}
        if note[a] != note[b]:
            votes.add(b if note[a] else a)
        return next(iter(votes)) if len(votes) == 1 else None

    def switch_order(self, nm: str, vals: list) -> Any:
        text = Text.of(self.descs.get(nm)).lower()
        spots = sorted((text.find(str(v).lower()), v) for v in vals if str(v).lower() in text)
        if len(spots) != len(vals):
            return None
        cart = self.cart_words(set())
        hits = {}
        for i, (at, v) in enumerate(spots):
            end = spots[i + 1][0] if i + 1 < len(spots) else len(text)
            seg = set(re.findall(r"[a-z0-9]+", text[at:end]))
            hits[v] = len(seg & cart)
        lead = max(vals, key=lambda v: hits[v])
        if any(hits[v] >= hits[lead] for v in vals if v != lead):
            return None
        if all(hits[v] == 0 for v in vals if v != lead) or lead == spots[0][1]:
            return lead
        return None

    def closing_guess(self) -> list[str]:
        tm = self.tm
        if tm.get("buy") and tm.get("decline"):
            b, d = tm["buy"], tm["decline"]
            if b.get("name") == d.get("name") and b.get("name") in self.schemas:
                k = next(iter(b.get("args") or {}), None)
                vals = [(b.get("args") or {}).get(k), (d.get("args") or {}).get(k)]
                told = self.switch_order(b["name"], vals) if k and None not in vals else None
                clash = told is not None and told != vals[0]
            else:
                told = self.pair_order(b.get("name"), d.get("name")) if b.get("name") in self.schemas and d.get("name") in self.schemas else None
                clash = told is not None and told != b.get("name")
            if clash:
                self.unsure |= {b.get("name"), d.get("name")}
                tm.pop("buy")
                tm.pop("decline")
                self.log.append("buy and decline from the label model contradict the store structure: both left unset")
            return []
        known = {r: tm[r]["name"] for r in ("buy", "decline") if tm.get(r) and tm[r].get("name") in self.schemas}
        busy = {m.get("name") for r, m in tm.items() if isinstance(m, dict) and r in ("search", "cart", "compare", "detail", "specs", "ask", "remove", "quote")}
        free = [nm for nm in self.schemas if nm not in busy]
        fins = self.finish_tools(free)
        finder = (tm.get("search") or {}).get("name")
        if not fins and not known and finder and self.finish_tools([finder]):
            kinds, _ = self.kinds_of(finder)
            notes = [a for a, kd in kinds.items() if kd == "str" and a != self.tokens.get(finder)]
            spec = ((self.schemas[finder].get("properties") or {}).get(notes[0]) or {}) if notes else {}
            if not notes or (spec.get("maxLength") or 0) >= 200:
                fins = self.finish_tools([finder])
                for r in ("search", "filter"):
                    if (tm.get(r) or {}).get("name") == finder:
                        tm.pop(r)
                alt = next((nm for nm in self.search_alts if nm != finder), None)
                if alt:
                    tm["search"] = self.search_map(alt, self.shape_of(alt))
                self.search_alts = [nm for nm in self.search_alts if nm not in (finder, alt)]
                free.append(finder)
        if known:
            fins = [f for f in fins if f[0] in known.values()]
        out: list[str] = []
        if len(fins) == 1:
            nm, k, vals = fins[0]
            text = Text.of(self.descs.get(nm)).lower()
            at = {v: (text.find(str(v).lower()) if str(v).lower() in text else 10 ** 6 + n) for n, v in enumerate(vals)}
            yes = min(vals, key=lambda v: at[v])
            sure = False
            for r, m in ((r, tm[r]) for r in known):
                if (m.get("args") or {}).get(k) in vals:
                    yes = m["args"][k] if r == "buy" else next(v for v in vals if v != m["args"][k])
                    sure = True
            if not sure:
                carted = self.switch_order(nm, vals)
                sure = carted is not None
                yes = carted if sure else yes
            no = next(v for v in vals if v != yes)
            if not sure:
                self.log.append(f"finish switch {nm}: order value not confirmed by structure; roles left to the label model")
                self.unsure.add(nm)
                fins = []
                nm = None
            if nm:
                note = next((a for a, kd in self.kinds_of(nm)[0].items() if kd == "str" and a != self.tokens.get(nm)), None)
                tm.setdefault("buy", {"name": nm, "args": {k: yes}, **({"token": self.tokens[nm]} if nm in self.tokens else {})})
                tm.setdefault("decline", {"name": nm, "args": {k: no}, **({"note": note} if note else {})})
                out.append(f"finish switch {nm}: {k}={yes} orders, {k}={no} declines")
        elif not fins:
            got = self.closing_pair(free)
            if got and set(known.values()) <= set(got):
                yes, no = got
                sure = bool(known)
                if known.get("buy") or (known.get("decline") and known["decline"] == yes):
                    yes, no = (known["buy"], next(x for x in got if x != known["buy"])) if known.get("buy") else (no, yes)
                if not sure:
                    told = self.pair_order(yes, no)
                    sure = told is not None
                    yes, no = (told, no if told == yes else yes) if sure else (yes, no)
                if not sure:
                    self.log.append(f"closing pair {yes}/{no}: which one orders is not confirmed by structure; roles left to the label model")
                    self.unsure |= {yes, no}
                    got = None
            if got and set(known.values()) <= set(got):
                note = next((x for x, kd in self.kinds_of(no)[0].items() if kd == "str"), None)
                tm.setdefault("buy", {"name": yes, "args": {}, **({"token": self.tokens[yes]} if yes in self.tokens else {})})
                tm.setdefault("decline", {"name": no, "args": {}, **({"note": note} if note else {})})
                out.append(f"closing pair by their shared clause: {yes} orders, {no} declines")
        taken = {tm[r]["name"] for r in ("buy", "decline") if tm.get(r)}
        for r in ("view", "usage"):
            if tm.get(r) and (tm[r].get("name") in taken or tm[r].get("name") in self.unsure):
                tm.pop(r)
        if not tm.get("view"):
            spare = next((nm for nm in free if nm not in taken and nm not in self.unsure and not self.kinds_of(nm)[0]
                          and nm != (tm.get("usage") or {}).get("name")), None)
            if spare:
                tm["view"] = {"name": spare, "args": {}}
        self.roles = {m["name"]: role for role, m in tm.items() if role != "filter" or m["name"] != (tm.get("search") or {}).get("name")}
        self.roles.update({m["name"]: "decline" for role, m in tm.items() if role == "decline" and m["name"] not in self.roles})
        self.log += out
        return out

    def sense_template(self) -> str:
        rows = []
        for nm, sch in self.schemas.items():
            props = {k: v for k, v in ((sch or {}).get("properties") or {}).items() if isinstance(v, dict)}
            vals = {a: {str(x): "?" for x in v["enum"]} for a, v in props.items() if isinstance(v.get("enum"), list) and 2 <= len(v["enum"]) <= 4}
            rows.append(json.dumps(nm) + ": " + json.dumps({"why": "?", "does": "?", "ends": "?", "args": {a: "?" for a in props}, "values": vals}))
        return "{\n" + ",\n".join(rows) + "\n}"

    def does(self, name: str) -> str | None:
        return (self.sense.get(name) or {}).get("does")

    def ends(self, name: str) -> bool:
        x = self.sense.get(name) or {}
        return x.get("does") in self.CLOSING or (bool(x.get("ends")) and x.get("does") in (None, "other"))

    def arg_is(self, name: str, arg: str | None, *roles: str) -> bool:
        return bool(arg) and ((self.sense.get(name) or {}).get("args") or {}).get(arg) in roles

    def value_is(self, name: str, arg: str, value: Any, *roles: str) -> bool:
        return (((self.sense.get(name) or {}).get("values") or {}).get(arg) or {}).get(str(value)) in roles

    def text_of(self, name: str) -> str:
        return (Text.key(name) + " " + Text.of(self.descs.get(name))).lower()

    def arg_text(self, name: str, arg: str) -> str:
        spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(arg) or {}
        own = (Text.key(arg) + " " + Text.of(spec.get("description") if isinstance(spec, dict) else "")).lower()
        named = [c for c in clauses(self.text_of(name)) if re.search(r"(?<![a-z0-9_])" + re.escape(arg.lower()) + r"(?![a-z0-9_])", c)]
        return own + " " + " ".join(named)

    def named_in(self, name: str, clause: str, pool: list[str]) -> list[str]:
        return [k for k in pool if re.search(r"(?<![a-z0-9_])" + re.escape(k.lower()) + r"(?![a-z0-9_])", clause)]

    def shape_of(self, name: str) -> dict:
        sch = self.schemas.get(name) or {}
        skip = getattr(self, "tokens", {}).get(name)
        props = {k: v for k, v in (sch.get("properties") or {}).items() if isinstance(v, dict) and k != skip}
        return {"props": props, "req": [k for k in (sch.get("required") or []) if k in props],
                "strs": [k for k, v in props.items() if v.get("type") == "string" and not v.get("enum")],
                "ints": [k for k, v in props.items() if v.get("type") in ("integer", "number")],
                "enums": [k for k, v in props.items() if isinstance(v.get("enum"), list) and v["enum"]],
                "arrs": [k for k, v in props.items() if v.get("type") == "array"], "text": self.text_of(name)}

    @staticmethod
    def idlike(spec: dict) -> bool | None:
        m = spec.get("maxLength")
        return None if not isinstance(m, int) else m <= 64

    def split_switch(self, name: str, key: str, vals: list, neg: str, pos: str = "") -> tuple:
        text = self.text_of(name) + " " + self.arg_text(name, key)
        hits = {}
        order = {}
        for n, v in enumerate(vals):
            hits[v] = 1 if self.value_is(name, key, v, neg) else (-1 if pos and self.value_is(name, key, v, pos) else 0)
            at = text.find(str(v).lower())
            order[v] = at if at >= 0 else 10 ** 6 + n
        no = max(vals, key=lambda v: (hits[v], order[v]))
        yes = next((v for v in vals if v != no), vals[0])
        return yes, no

    @staticmethod
    def names_in(text: str, name: str) -> bool:
        return bool(name) and re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text or "") is not None

    def find_quotes(self) -> dict[str, tuple[str, str]]:
        found: dict[str, tuple[str, str]] = {}
        for q, qs in self.schemas.items():
            if (qs or {}).get("properties"):
                continue
            qdesc = Text.of(self.descs.get(q))
            for b, bs in self.schemas.items():
                if b == q or b in found.values():
                    continue
                bdesc = Text.of(self.descs.get(b))
                for arg, spec in ((bs or {}).get("properties") or {}).items():
                    if not isinstance(spec, dict) or self.item_kind(spec) != "str" or not self.names_in(qdesc, arg):
                        continue
                    if self.names_in(bdesc, q) or self.names_in(Text.of(spec.get("description")), q):
                        found[q] = (b, arg)
                        break
                if q in found:
                    break
        return found

    def assign(self) -> None:
        tm: dict[str, dict] = {}
        reads: list[str] = []
        finders: list[str] = []
        self.tokens = {}
        quotes = self.find_quotes()
        for q, (b, arg) in quotes.items():
            self.tokens[b] = arg
            tm["quote"] = {"name": q, "token": arg, "for": b, "args": {}}
        baskets = [nm for nm in self.schemas if self.shape_of(nm)["arrs"] and self.basket_of(nm, self.shape_of(nm)) and not self.ends(nm)]
        singles: list[str] = []
        shelves: list[str] = []
        for name in self.schemas:
            if name in quotes:
                continue
            s = self.shape_of(name)
            props, req, strs, ints, enums, arrs = (s[k] for k in ("props", "req", "strs", "ints", "enums", "arrs"))
            n = len(props)
            does = self.does(name)
            long_str = [k for k in strs if (props[k].get("maxLength") or 0) >= 200]
            twos = [k for k in enums if len(props[k]["enum"]) == 2]
            ending = self.ends(name)
            basket = self.basket_of(name, s) if arrs else None
            fin = self.finish_switch(name, s, ending) if enums and not ints and not arrs and len(strs) <= 1 and n <= 3 else None
            ks = [k for k in strs if k in req] if len(strs) > 1 else strs
            cartish = (len(strs) > 1 and len(ks) == 1 and ints and not arrs and self.idlike(props[ks[0]]) is True
                       and not [k for k in long_str if k in ks] and bool(twos or [q for q in ints if q in req]))
            if basket and ending:
                tm.setdefault("buy", {"name": name, "args": {}, "basket": basket})
            elif basket:
                tm.setdefault("cart", basket)
            elif arrs and n <= 2:
                k = arrs[0]
                tm.setdefault("compare", {"name": name, "ids": k, "max": int(props[k].get("maxItems") or 4)})
            elif n == 0:
                if ending:
                    tm.setdefault("decline" if does == "decline" else "buy", {"name": name, "args": {}})
                elif does == "usage":
                    tm.setdefault("usage", {"name": name, "args": {}})
                elif does == "view" or not any(v.get("does") == "view" for v in self.sense.values()):
                    tm.setdefault("view", {"name": name, "args": {}})
            elif twos and not ints and not arrs and len(strs) <= 1 and n <= 2 and ending:
                k = twos[0]
                yes, no = self.split_switch(name, k, props[k]["enum"], "decline", "order")
                tm["buy"] = {"name": name, "args": {k: yes}}
                tm["decline"] = {"name": name, "args": {k: no}, **({"note": strs[0]} if strs else {})}
            elif fin:
                k, yes, no = fin
                tm.setdefault("buy", {"name": name, "args": {k: yes}})
                tm.setdefault("decline", {"name": name, "args": {k: no}, **({"note": strs[0]} if strs else {})})
            elif n == 1 and strs:
                k = strs[0]
                idk = self.idlike(props[k])
                if ending and does != "message":
                    tm.setdefault("decline" if does == "decline" else "buy", {"name": name, "args": {}, "note": k})
                elif (long_str and self.sense and does not in ("search", "browse", "page", "facts")) or does == "message":
                    tm.setdefault("ask", {"name": name, "text": k})
                elif long_str and not self.sense:
                    continue
                elif idk is False or (idk is None and does in ("search", "browse")):
                    singles.append(name)
                else:
                    reads.append(name)
            elif (len(strs) == 1 and ints and not arrs and not long_str and (
                    self.idlike(props[strs[0]]) is True or (self.idlike(props[strs[0]]) is None and does not in ("search", "browse")))) or cartish:
                k, q = ks[0], next((x for x in ints if x in req), ints[0])
                if ending:
                    tm.setdefault("buy", {"name": name, "args": {}, "id": k, "qty": q})
                elif baskets or does in ("search", "browse") or (q not in req and k in req and not twos and self.arg_is(name, q, "page")):
                    finders.append(name)
                else:
                    tm.setdefault("cart", self.cart_map(name, s, k, q, twos))
            elif n == 1 and ints and not ending:
                reads.append(name)
            elif len(ints) == 2 and not strs and not arrs and not ending and len(props) - len(enums) == 2:
                ranked = sorted(ints, key=lambda x: (isinstance(props[x].get("maximum"), (int, float)), x not in req))
                tm.setdefault("cart", self.cart_map(name, s, ranked[0], ranked[1], twos))
            elif long_str and len(strs) == 1 and not arrs and does == "message" and not ending:
                tm.setdefault("ask", {"name": name, "text": strs[0]})
            elif strs:
                finders.append(name)
            elif (not arrs and not ending and len(ints) <= 1 and any(len(props[k]["enum"]) > 4 for k in enums)
                  and does in (None, "browse", "search", "other")):
                shelves.append(name)
        spare = list(finders) + list(singles)
        if finders or tm.get("search"):
            reads.extend(singles)
        else:
            finders.extend(singles[:1])
            reads.extend(singles[1:])
        self.place_finders(tm, finders)
        for nm in shelves:
            if tm.get("filter") or not tm.get("search"):
                break
            s = self.shape_of(nm)
            f = self.filter_map(nm, s, {"page": self.page_key(nm, s), "query": None})
            if f:
                tm["filter"] = f
        self.search_alts = [nm for nm in spare if nm != (tm.get("search") or {}).get("name")]
        if not tm.get("buy"):
            self.buy_fallback(tm, reads)
        if tm.get("buy") and not tm.get("decline"):
            self.decline_fallback(tm, reads)
        cart = tm.get("cart") or {}
        if cart and cart.get("mode") != "basket" and not cart.get("switch"):
            ctext = Text.of(self.descs.get(cart["name"]))
            for nm in list(reads):
                arg = self.id_arg(nm)
                if self.names_in(ctext, nm) or arg == cart.get("id") or self.does(nm) == "remove":
                    tm["remove"] = {"name": nm, "line": arg, "by": "id" if arg == cart.get("id") else "ref", "ref_key": None, "tried": []}
                    reads.remove(nm)
                    break
        reads = [nm for nm in reads if self.does(nm) in (None, "page", "facts", "other")]
        reads.sort(key=lambda nm: {"page": -1, "facts": 1}.get(self.does(nm) or "", 0))
        if reads:
            tm["detail"] = {"name": reads[0], "id": self.id_arg(reads[0])}
        if len(reads) > 1:
            tm["specs"] = {"name": reads[-1], "id": self.id_arg(reads[-1])}
        cmp_ = tm.get("compare")
        if cmp_ and not tm.get("cart") and not (tm.get("buy") or {}).get("id") and not (tm.get("buy") or {}).get("basket"):
            if self.does(cmp_["name"]) == "cart":
                spec = (self.shape_of(cmp_["name"])["props"].get(cmp_["ids"]) or {})
                tm["cart"] = {"name": cmp_["name"], "mode": "basket", "shape": "repeat", "lines": cmp_["ids"], "args": {}, "qmin": 0, "qmax": None,
                              "min": int(spec.get("minItems") or 0), "alts": []}
                tm.pop("compare")
        for role in ("buy", "decline"):
            m = tm.get(role)
            if m and m["name"] in self.tokens:
                m["token"] = self.tokens[m["name"]]
                if m.get("note") == m["token"]:
                    m.pop("note")
        self.tm.clear()
        self.tm.update(tm)
        self.roles = {m["name"]: role for role, m in tm.items() if role != "filter" or m["name"] != (tm.get("search") or {}).get("name")}
        self.roles.update({m["name"]: "decline" for role, m in tm.items() if role == "decline" and m["name"] not in self.roles})

    def id_arg(self, name: str) -> str:
        s = self.shape_of(name)
        return (s["strs"] or s["ints"] or list(s["props"]))[0]

    def finish_switch(self, name: str, s: dict, ending: bool) -> tuple | None:
        props = s["props"]
        for k in s["enums"]:
            vals = props[k]["enum"]
            if not 2 <= len(vals) <= 4 or not all(isinstance(v, str) for v in vals):
                continue
            yes = next((v for v in vals if self.value_is(name, k, v, "order")), None)
            no = next((v for v in vals if self.value_is(name, k, v, "decline")), None)
            if yes is not None and no is not None and yes != no:
                return k, yes, no
        return None

    def buy_fallback(self, tm: dict, reads: list[str]) -> None:
        used = {m.get("name") for r, m in tm.items() if r not in ("view", "usage")} | set(reads)
        best, top = None, 0
        for nm, sch in self.schemas.items():
            s = self.shape_of(nm)
            if nm in used or s["strs"] or s["arrs"] or s["enums"]:
                continue
            sc = 1 if self.does(nm) == "order" else 0
            if sc > top:
                best, top = nm, sc
        if not best:
            return
        for r in ("view", "usage"):
            if (tm.get(r) or {}).get("name") == best:
                tm.pop(r)
        tm["buy"] = {"name": best, "args": {}, "extras": [k for k in self.shape_of(best)["props"]]}
        self.log.append(f"buy chosen by the model's reading among plain tools: {best}")

    def decline_fallback(self, tm: dict, reads: list[str]) -> None:
        used = {m.get("name") for m in tm.values() if isinstance(m, dict)} | set(reads)
        best = None
        for nm in sorted(self.schemas):
            s = self.shape_of(nm)
            if nm in used or s["arrs"] or s["enums"] or s["ints"] or len(s["strs"]) > 1:
                continue
            sc = 1 if self.does(nm) == "decline" else 0
            if sc > 0 and (best is None or sc > best[0]):
                best = (sc, nm, s["strs"])
        if best is None:
            return
        _, nm, strs = best
        tm["decline"] = {"name": nm, "args": {}, **({"note": strs[0]} if strs else {})}
        self.log.append(f"decline chosen by the model's reading among plain tools: {nm}")

    def place_finders(self, tm: dict, finders: list[str]) -> None:
        if not finders:
            return
        if not tm.get("search"):
            shapes = {nm: self.shape_of(nm) for nm in finders}
            maps = {nm: self.search_map(nm, shapes[nm]) for nm in finders}
            words = [nm for nm in finders if maps[nm]["query"] in shapes[nm]["req"]]

            def rank(nm: str) -> tuple:
                q = shapes[nm]["props"].get(maps[nm]["query"]) or {}
                chat = self.does(nm) == "message"
                return ((q.get("maxLength") or 0) >= 200 or chat, nm not in words)
            first = min(finders, key=lambda nm: (rank(nm), finders.index(nm)))
            tm["search"] = maps[first]
            f = self.filter_map(first, shapes[first], tm["search"])
            if f:
                tm.setdefault("filter", f)
        for nm in finders:
            if nm == tm["search"]["name"] or tm.get("filter"):
                continue
            s = self.shape_of(nm)
            f = self.filter_map(nm, s, {"page": self.page_key(nm, s), "query": None})
            if f:
                tm["filter"] = f

    def text_mode_arg(self, name: str, mode: str, vals: list, pool: list[str]) -> str | None:
        if not any(self.value_is(name, mode, v, "words") for v in vals):
            return None
        named = [k for k in pool if self.arg_is(name, k, "query")]
        return named[0] if len(named) == 1 else None

    def search_map(self, name: str, s: dict) -> dict:
        props, strs, enums, req = s["props"], s["strs"], s["enums"], s["req"]
        page = self.page_key(name, s)
        mode = next((k for k in enums if len(props[k]["enum"]) == 2), None)
        cat = next((k for k in strs if self.arg_is(name, k, "category")), None)
        cat = cat or next((k for k in enums if len(props[k]["enum"]) > 2), None)
        brand = next((k for k in strs if k != cat and self.arg_is(name, k, "brand")), None)
        rest = [k for k in strs if k not in (page, cat, brand)]
        named = self.text_mode_arg(name, mode, props[mode]["enum"], rest) if mode else None
        query = (named or next((k for k in rest if self.arg_is(name, k, "query")), None) or next((k for k in rest if k in req), None)
                 or (max(rest, key=lambda k: props[k].get("maxLength") or 0) if rest else (strs[0] if strs else None)))
        m: dict[str, Any] = {"name": name, "query": query, "page": page, "args": {}, "options": []}
        if mode:
            m["args"][mode] = self.mode_value(name, mode, props[mode]["enum"], query, True)
            m["mode"] = mode
        for q in [k for k in strs if k != page]:
            for v in (props[mode]["enum"] if mode else [None]):
                if q != query or (mode and v != m["args"].get(mode)):
                    m["options"].append((q, v))
        m["options"].sort(key=lambda o: (o[0] != query, o[0] in (cat, brand)))
        return m

    def mode_value(self, name: str, mode: str, vals: list, query: str | None, texty: bool) -> Any:
        want = next((v for v in vals if self.value_is(name, mode, v, "words" if texty else "browse")), None)
        if want is not None:
            return want
        text = self.text_of(name) + " " + self.arg_text(name, mode)
        tally = {}
        for v in vals:
            cs = [c for c in clauses(text) if re.search(r"(?<![a-z0-9_])" + re.escape(str(v).lower()) + r"(?![a-z0-9_])", c)]
            q = sum(1 for c in cs if query and re.search(r"\b" + re.escape(Text.key(query)) + r"\b", Text.key(c)))
            tally[v] = 2 * q + (1 if self.value_is(name, mode, v, "words") else -1 if self.value_is(name, mode, v, "browse") else 0)
        best = max(vals, key=lambda v: tally[v])
        if texty:
            return best
        return next((v for v in vals if v != best), vals[-1])

    def filter_map(self, name: str, s: dict, search: dict) -> dict | None:
        props, strs, ints, enums = s["props"], s["strs"], s["ints"], s["enums"]
        cat = next((k for k in strs if self.arg_is(name, k, "category")), None)
        cat = cat or next((k for k in enums if len(props[k]["enum"]) > 2), None)
        brand = next((k for k in strs if k != cat and self.arg_is(name, k, "brand")), None)
        price = (next((k for k in ints if self.arg_is(name, k, "price")), None)
                 or next((k for k in ints if props[k].get("type") == "number" and props[k].get("maximum") is None), None))
        if not cat and not price:
            return None
        f: dict[str, Any] = {"name": name, "category": cat, "brand": brand, "max_price": price, "page": search.get("page"), "args": {}}
        mode = next((k for k in enums if len(props[k]["enum"]) == 2), None)
        if mode:
            f["args"][mode] = self.mode_value(name, mode, props[mode]["enum"], search.get("query"), False)
        return f

    def page_key(self, name: str, s: dict) -> str | None:
        props, req = s["props"], s["req"]
        for k in s["ints"] + s["strs"]:
            if k not in req and self.arg_is(name, k, "page"):
                return k
        for k in s["ints"]:
            v = props[k]
            if k not in req and isinstance(v.get("maximum"), (int, float)) and v["maximum"] <= 50 and v.get("type") == "integer":
                return k
        for k in s["strs"]:
            v = props[k]
            if k not in req and isinstance(v.get("maxLength"), int) and v["maxLength"] <= 16:
                return k
        return None

    def page_arg(self, role: str) -> str | None:
        m = self.tm.get(role) or {}
        return m.get("page")

    def next_cursor(self, role: str, q: Any) -> Any:
        m = self.tm.get(role) or {}
        pg = m.get("page")
        if not pg:
            return None
        if q.cursor not in (None, "", False):
            return q.cursor
        spec = ((self.schemas.get(m.get("name") or "") or {}).get("properties") or {}).get(pg) or {}
        if spec.get("type") != "integer" or self.cursor:
            return None
        lo = spec.get("minimum")
        hi = spec.get("maximum")
        pages, n = q.pages or 0, len(q.ids or [])
        nxt = (n if lo == 0 else pages + 1) if isinstance(lo, int) else None
        if nxt is None or (isinstance(hi, (int, float)) and nxt > hi) or (q.total is not None and lo == 0 and nxt >= q.total):
            return None
        return nxt

    def cart_map(self, name: str, s: dict, k: str, q: str, twos: list) -> dict:
        props = s["props"]
        m: dict[str, Any] = {"name": name, "id": k, "qty": q, "args": {}}
        qmin = props[q].get("minimum")
        m["qmin"] = int(qmin) if isinstance(qmin, (int, float)) else 0
        m["qmax"] = int(props[q]["maximum"]) if isinstance(props[q].get("maximum"), (int, float)) else None
        if twos:
            sw = twos[0]
            up, down = self.split_switch(name, sw, props[sw]["enum"], "remove", "add")
            m.update({"switch": sw, "up": up, "down": down})
            m["args"][sw] = up
        m["mode"] = "add" if twos or m["qmin"] >= 1 else "set"
        return m

    @staticmethod
    def item_kind(spec: Any) -> str | None:
        if not isinstance(spec, dict):
            return None
        typ = spec.get("type")
        if isinstance(typ, list):
            typ = next((t for t in typ if t != "null"), None)
        if typ in ("integer", "number"):
            return "int"
        if typ == "string" and not spec.get("enum"):
            return "str"
        if typ == "object" or isinstance(spec.get("properties"), dict):
            return "obj"
        if typ == "array":
            return "arr"
        return None

    @staticmethod
    def bounds(spec: dict) -> tuple[int, int | None]:
        lo, hi = spec.get("minimum"), spec.get("maximum")
        return (int(lo) if isinstance(lo, (int, float)) else 0), (int(hi) if isinstance(hi, (int, float)) else None)

    def basket_of(self, name: str, s: dict) -> dict | None:
        props = s["props"]
        for k in s["arrs"]:
            spec = props[k]
            items = spec.get("items")
            kind = self.item_kind(items)
            base = {"name": name, "mode": "basket", "lines": k, "args": {}, "min": int(spec.get("minItems") or 0), "alts": []}
            if kind == "obj":
                ip = {a: v for a, v in (items.get("properties") or {}).items() if isinstance(v, dict)}
                ireq = [a for a in (items.get("required") or []) if a in ip]
                qs = [a for a in ip if self.item_kind(ip[a]) == "int"]
                ids = [a for a in ip if self.item_kind(ip[a]) == "str"]
                if not ids and len(qs) == 1 and len(ip) == 2:
                    ids = [a for a in ip if a not in qs]
                if not ids or not qs:
                    continue
                ids.sort(key=lambda a: (a not in ireq, self.idlike(ip[a]) is False))
                qs.sort(key=lambda a: (a not in ireq, ip[a].get("type") != "integer"))
                pairs = [(i, q) for i in ids[:3] for q in qs[:2]]
                lo, hi = self.bounds(ip[pairs[0][1]])
                return {**base, "shape": "objects", "item_id": pairs[0][0], "item_qty": pairs[0][1], "qmin": lo, "qmax": hi, "alts": pairs[1:4]}
            if kind == "arr":
                cols = items.get("prefixItems") if isinstance(items.get("prefixItems"), list) else items.get("items")
                kinds = [self.item_kind(c) for c in cols] if isinstance(cols, list) else []
                if kinds[:2] in (["str", "int"], ["int", "str"]):
                    lo, hi = self.bounds(cols[kinds.index("int")])
                    return {**base, "shape": "pairs", "order": kinds[:2], "qmin": lo, "qmax": hi}
            if kind == "str":
                twins = [q for q in s["arrs"] if q != k and self.item_kind(props[q].get("items")) == "int"]
                if twins:
                    lo, hi = self.bounds(props[twins[0]].get("items") or {})
                    return {**base, "shape": "parallel", "qtys": twins[0], "qmin": lo, "qmax": hi}
        return None

    def yields_facts(self, name: str) -> bool:
        v = self.views.get(name)
        return not v or v.get("facts", 0) > 0 or v.get("rows", 0) == 0

    def compare_max(self) -> int:
        cmp_ = self.tm.get("compare") or {}
        props = (self.schemas.get(cmp_.get("name") or "") or {}).get("properties") or {}
        spec = props.get(cmp_.get("ids") or "") or {}
        limits = []
        for c in (cmp_.get("max"), spec.get("maxItems"), self.cmp_cap):
            if isinstance(c, int) and c > 0:
                limits.append(c)
        if not limits:
            return 4
        return max(1, min(limits))

    def reads_per_line(self) -> int:
        return sum(1 for r in ("detail", "specs") if self.tm.get(r))

    def facts_role(self) -> str | None:
        if self.tm.get("specs"):
            return "specs"
        for r in ("detail", "compare"):
            m = self.tm.get(r)
            if m and (self.views.get(m["name"], {}).get("facts") or not self.views.get(m["name"])):
                return r
        return "detail" if self.tm.get("detail") else ("compare" if self.tm.get("compare") else None)

    @staticmethod
    def unwrap(obs: Any) -> tuple[Any, list[str]]:
        if isinstance(obs, list) and obs and all(isinstance(s, str) for s in obs):
            return {}, [s for s in obs if len(s.split()) >= 3]
        if isinstance(obs, dict) and 1 <= len(obs) <= 3:
            texts = [k for k, v in obs.items() if isinstance(v, list) and v and all(isinstance(s, str) for s in v) and any(len(s.split()) >= 4 for s in v)]
            rest = [k for k in obs if k not in texts]
            if texts and len(rest) <= 1:
                notes = [s for k in texts for s in obs[k]]
                return (obs[rest[0]] if rest else {}), notes
        return obs, []

    @staticmethod
    def ok_reply(obs: Any) -> bool | None:
        obs = Store.unwrap(obs)[0]
        if isinstance(obs, dict):
            if not obs:
                return False
            if any(v is True for v in obs.values()) or Catalog.row_lists(obs):
                return True
            if all(isinstance(v, str) for v in obs.values()):
                return False
        return None

    def observe(self, name: str, args: dict, obs: Any) -> tuple[Any, list[str]]:
        obs, notices = self.unwrap(obs)
        role = self.roles.get(name)
        try:
            if role in ("search", "filter"):
                m = self.tm.get(role) or {}
                self.learn_rows(obs, Text.of(args.get(m.get("query") or "")) if role == "search" else "")
            elif role in ("detail", "specs", "compare"):
                self.learn_view(name, args, obs)
            elif role in ("cart", "view", "buy", "remove", "quote"):
                self.learn_cart(obs)
        except Exception as exc:
            self.log.append(f"learn {role}: {type(exc).__name__}: {exc}"[:160])
        return obs, notices

    def tally(self, r: dict) -> None:
        self.rows_seen += 1
        for k, v in r.items():
            st = self.stats.setdefault(k, {"n": 0, "empty": 0, "bool": 0, "true": 0, "num": 0, "str": 0, "nest": 0, "vals": [], "set": set(),
                                           "len": 0, "digit": 0, "space": 0, "maxlen": 0, "nl": 0, "intitle": 0})
            st["n"] += 1
            self.row_keys.add(k)
            if empty(v):
                st["empty"] += 1
                continue
            v = money_of(v)
            if isinstance(v, bool):
                st["bool"] += 1
                st["true"] += int(v)
            elif isinstance(v, (int, float)):
                st["num"] += 1
                st["vals"].append(float(v))
            elif isinstance(v, str):
                st["str"] += 1
                s = v.strip()
                st["len"] += len(s)
                st["maxlen"] = max(st["maxlen"], len(s))
                st["digit"] += int(any(c.isdigit() for c in s))
                st["space"] += int(" " in s)
                st["nl"] += int("\n" in s)
                st["intitle"] += int(len(s) <= 40 and any(isinstance(x, str) and len(x) > len(s) + 10 and s.lower() in x.lower()
                                                           for kk, x in r.items() if kk != k))
                if len(st["set"]) < 400:
                    st["set"].add(s.lower()[:120])
                if len(st["vals"]) < 60:
                    st["vals"].append(s[:160])
            else:
                st["nest"] += 1

    def learn_rows(self, obs: Any, query: str) -> None:
        if not isinstance(obs, dict):
            return
        lists = Catalog.row_lists(obs)
        if not lists:
            return
        key, rows = max(lists, key=lambda kv: len(kv[1]))
        if self.rows is None or self.rows not in dict(lists):
            self.rows = key
        rows = dict(lists).get(self.rows) or rows
        for r in rows:
            mark = Text.json(r)
            if mark not in self.tallied:
                self.tallied.add(mark)
                self.tally(r)
        if query.strip():
            self.queries.append(query.strip().lower())
        top = {k: v for k, v in obs.items() if k != self.rows and (Catalog.is_scalar(v) or v is None)}
        self.learn_top(top, len(rows))
        self.decide_slots()

    def learn_top(self, top: dict, nrows: int) -> None:
        ints = {k: v for k, v in top.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        others = [k for k, v in top.items() if k not in ints and (v is None or (isinstance(v, str) and len(v) <= 24))]
        tot = cur = None
        if len(ints) >= 2:
            tot = max(ints, key=lambda k: ints[k])
            cur = next((k for k in ints if k != tot and ints[k] != nrows), None) or next(k for k in ints if k != tot)
        elif len(ints) == 1:
            k, v = next(iter(ints.items()))
            if v > nrows:
                tot = k
            elif v < nrows:
                cur = k
            else:
                tot = k
        if cur is None and others:
            cur = others[0]
        if tot and "total" not in self.fixed:
            self.total = tot
        if cur and "cursor" not in self.fixed:
            self.cursor = cur

    def decide_slots(self) -> None:
        st = self.stats

        def share(k: str, f: str) -> float:
            return st[k][f] / max(1, st[k]["n"] - st[k]["empty"])

        def mean_len(k: str) -> float:
            return st[k]["len"] / max(1, st[k]["str"])

        def distinct(k: str) -> float:
            return len(st[k]["set"]) / max(1, st[k]["str"])

        strk = [k for k in st if st[k]["str"] >= max(1, 0.6 * (st[k]["n"] - st[k]["empty"])) and st[k]["n"] - st[k]["empty"] > 0]
        idish = [k for k in strk if share(k, "space") == 0 and share(k, "digit") >= 0.9 and st[k]["maxlen"] <= 64]
        ids = [k for k in idish if st[k]["empty"] == 0 and distinct(k) >= 0.5]
        ids.sort(key=lambda k: (-(sum(1 for v in st[k]["vals"] if isinstance(v, str) and v in self.ids)), -(sum(1 for v in st[k]["vals"] if isinstance(v, str) and ID_RX.match(v))), mean_len(k)))
        nums = [k for k in st if st[k]["num"] >= 0.9 * max(1, st[k]["n"] - st[k]["empty"]) and st[k]["num"] > 0]
        if not ids:
            whole = [k for k in nums if st[k]["empty"] == 0 and st[k]["num"] >= 3 and all(float(v).is_integer() for v in st[k]["vals"])
                     and len(set(st[k]["vals"])) == len(st[k]["vals"])]
            if len(whole) >= 1 and len(nums) >= 2:
                ids = [max(whole, key=lambda k: min(st[k]["vals"]))]
        self.settle("id", ids[:1], ids)
        nums = [k for k in nums if k != self.id]
        nums.sort(key=lambda k: (-len(set(st[k]["vals"])), -max(st[k]["vals"] or [0])))
        self.settle("price", nums[:1], nums if len(nums) > 1 and len(set(st[nums[1]]["vals"])) > 3 else nums[:1])
        if not nums and self.price is None:
            coded = self.coded_money([k for k in strk if k not in idish])
            if coded:
                self.settle("price", coded[:1], coded[:1])
                if self.price == coded[0]:
                    self.log.append(f"price read from labelled amounts: {coded[0]!r}")
        bools = [k for k in st if st[k]["bool"] >= 0.9 * max(1, st[k]["n"] - st[k]["empty"]) and st[k]["bool"] > 0]
        if bools:
            self.settle("stock", bools[:1], bools)
            if self.stock in bools:
                self.stock_true_available = st[self.stock]["true"] >= 0.5 * st[self.stock]["bool"]
        else:
            flags = [k for k in strk if k not in idish and len(st[k]["set"]) <= 2 and st[k]["str"] >= 8]
            self.settle("stock", [], flags)
        texty = [k for k in strk if k not in idish and (st[k]["maxlen"] >= 60 or mean_len(k) >= 40)]
        top = max([distinct(k) for k in texty] or [0.0])
        text = [k for k in texty if distinct(k) > 0.3 or distinct(k) >= 0.5 * top or st[k]["str"] < 4] or texty
        text.sort(key=lambda k: (st[k]["empty"] > 0.1 * st[k]["n"], mean_len(k)))
        self.settle("title", text[:1], text[:1])
        rest = [k for k in text if k != self.title]
        rest.sort(key=lambda k: (-mean_len(k), st[k]["empty"]))
        self.settle("desc", rest[:1], rest[:1])
        always_empty = [k for k in st if st[k]["n"] >= 4 and st[k]["empty"] == st[k]["n"]]
        notes = [k for k in rest[1:]] + [k for k in always_empty if k not in text]
        if notes and self.notes is None:
            self.notes = notes[0]
        nests = [k for k in st if st[k]["nest"] >= 0.9 * max(1, st[k]["n"] - st[k]["empty"]) and st[k]["nest"] > 0]
        if nests and self.attrs is None:
            self.attrs = nests[0]
        low = [k for k in strk if k not in idish and k not in text and k != self.stock and k != self.price and st[k]["str"] >= 3 and mean_len(k) <= 50
               and (distinct(k) <= 0.9 or (share(k, "space") <= 0.3 and mean_len(k) <= 24 and share(k, "digit") <= 0.2))]
        qstems = [set(Text(q).stems) for q in self.queries]
        enum_vals = {str(e).lower() for name in (self.tm.get("search", {}).get("name"),) if name
                     for v in ((self.schemas.get(name) or {}).get("properties") or {}).values() if isinstance(v, dict)
                     for e in (v.get("enum") or []) if len(v.get("enum") or []) > 2}

        def catness(k: str) -> tuple:
            vals = [v for v in st[k]["vals"] if isinstance(v, str)]
            inq = sum(1 for v in vals if any(set(Text(v).stems) & q for q in qstems))
            ine = sum(1 for v in vals if v.lower() in enum_vals)
            return (ine > 0, inq / max(1, len(vals)), -distinct(k))
        cats = sorted(low, key=catness, reverse=True)
        best = cats[:1] if cats and (catness(cats[0])[0] or catness(cats[0])[1] > 0) else []
        tie = [k for k in cats[:2] if cats and catness(k)[:2] == catness(cats[0])[:2]]
        self.settle("category", best, tie if len(tie) > 1 else (cats[:2] if not best else best))
        rest = [k for k in low if k != self.category]
        makers = [k for k in rest if st[k]["empty"] >= 0.1 * st[k]["n"] or share(k, "intitle") >= 0.3]
        sellers = [k for k in rest if k not in makers]
        if len(sellers) > 1:
            sellers.sort(key=lambda k: share(k, "intitle"))
        self.settle("maker", makers[:1], makers)
        self.settle("seller", sellers[:1], sellers)

    def coded_money(self, keys: list[str]) -> list[str]:
        st = self.stats
        out: list[tuple[int, str]] = []
        for k in keys:
            vals = [v for v in st[k]["vals"] if isinstance(v, str)]
            if len(vals) < 3 or st[k]["str"] < 0.9 * max(1, st[k]["n"] - st[k]["empty"]):
                continue
            labels, amounts = set(), set()
            for v in vals:
                found = Text(v).numbers
                rest = re.sub(r"\d[\d.,]*", " ", v).strip()
                if len(found) != 1 or not rest or " " in rest or any(c.isdigit() for c in rest):
                    break
                labels.add(rest)
                amounts.add(found[0][0])
            else:
                if len(labels) == 1 and len(amounts) >= 3 and k != self.id:
                    out.append((sum(1 for v in vals if re.search(r"\d[.,]\d", v)), len(amounts), k))
        return [k for _, _, k in sorted(out, reverse=True)]

    def settle(self, slot: str, best: list[str], cands: list[str]) -> None:
        if slot in self.fixed:
            return
        cands = [c for c in dict.fromkeys(cands) if c]
        if len(cands) > 1:
            self.open[slot] = cands[:4]
        else:
            self.open.pop(slot, None)
        if best and best[0]:
            setattr(self, slot, best[0])
        elif cands and getattr(self, slot) is None and len(cands) == 1:
            setattr(self, slot, cands[0])

    def learn_view(self, name: str, args: dict, obs: Any) -> None:
        v = self.views.setdefault(name, {"n": 0, "facts": 0, "memo": 0, "row": 0, "rows": 0, "empty": 0, "id": None, "facts_key": None, "memo_keys": []})
        v["n"] += 1
        if not isinstance(obs, dict) or not obs:
            v["empty"] += 1
            self.settle_views()
            return
        passed = {str(x) for val in args.values() for x in (val if isinstance(val, list) else [val]) if isinstance(x, (str, int)) and not isinstance(x, bool)}
        lists = Catalog.row_lists(obs)
        recs = (dict(lists).get(self.rows) if self.rows in dict(lists) else lists[0][1]) if lists else [obs]
        if lists:
            v["rows"] += 1
        for d in recs:
            fk = max((k for k, x in d.items() if isinstance(x, dict) and sum(1 for y in x.values() if Catalog.is_scalar(y) or isinstance(y, list)) >= 2),
                     key=lambda k: len(d[k]), default=None)
            fk = fk or next((k for k, x in d.items() if isinstance(x, list) and x and all(isinstance(y, dict) and len(y) == 2 for y in x)), None)
            if fk:
                v["facts"] += 1
                v["facts_key"] = fk
                if self.attrs is None or self.attrs not in d:
                    self.attrs = fk
            shared = set(d) & self.row_keys
            if len(shared) >= 3:
                v["row"] += 1
            extra = [k for k in d if k not in self.row_keys and k != fk and (d[k] is None or isinstance(d[k], str)) and k != v["id"]
                     and not (isinstance(d[k], str) and d[k] in passed)]
            if lists and not self.rows:
                extra = []
            if extra:
                v["memo"] += 1
                v["memo_keys"] = list(dict.fromkeys(v["memo_keys"] + extra))[:6]
            for k, x in d.items():
                if isinstance(x, (str, int)) and not isinstance(x, bool) and str(x) in passed:
                    x = str(x)
                    v["id"] = k
                    self.ids.add(x)
                    if self.id is None or (k in self.row_keys and "id" in self.open):
                        self.id = k
                        self.fixed.add("id")
                        self.open.pop("id", None)
        memo_keys = [k for vv in self.views.values() for k in vv["memo_keys"] if k not in (self.title, self.desc)]
        if memo_keys and (self.notes is None or self.notes in self.row_keys):
            self.notes = memo_keys[0]
        self.settle_views()

    def settle_views(self) -> None:
        det, spc = self.tm.get("detail"), self.tm.get("specs")
        if not det or not spc:
            return
        a, b = self.views.get(det["name"]), self.views.get(spc["name"])
        if not a or not b:
            return
        tableness = lambda vv: (vv["facts"] > 0) - (vv["row"] > 0) - (vv["memo"] > 0)
        if tableness(a) > tableness(b):
            self.tm["detail"], self.tm["specs"] = {**spc}, {**det}
            self.roles[det["name"]], self.roles[spc["name"]] = "specs", "detail"
            self.log.append("swapped page and table tools")

    def learn_cart(self, obs: Any) -> None:
        for _, rows in Catalog.row_lists(obs):
            for r in rows:
                for k, x in r.items():
                    if isinstance(x, int) and not isinstance(x, bool) and k != self.price and 0 <= x <= 200 and k not in self.row_keys:
                        self.cart_qty = self.cart_qty or k
                if self.cart_qty is None:
                    k = next((k for k, x in r.items() if isinstance(x, int) and not isinstance(x, bool) and k != self.price and 0 <= x <= 200), None)
                    if k:
                        self.cart_qty = k

    def cart_lines(self, obs: Any) -> dict[str, int] | None:
        obs, _ = self.unwrap(obs)
        for _, rows in Catalog.row_lists(obs):
            lines = {}
            for r in rows:
                rid = Catalog.id_in(r, self.id)
                q = r.get(self.cart_qty) if self.cart_qty else None
                if not isinstance(q, int) or isinstance(q, bool):
                    q = next((x for k, x in r.items() if isinstance(x, int) and not isinstance(x, bool) and k != self.price and 0 <= x < 1000), None)
                if rid:
                    lines[rid] = q if q is not None else 1
            if lines:
                return lines
        if isinstance(obs, dict) and any(isinstance(v, list) and not v for v in obs.values()):
            return {}
        return None

    def cart_rows(self, obs: Any) -> dict[str, dict]:
        obs, _ = self.unwrap(obs)
        for _, rows in Catalog.row_lists(obs):
            got = {rid: row for row in rows for rid in [Catalog.id_in(row, self.id)] if rid}
            if got:
                return got
        return {}

    def questions(self, budget_raw: dict | None, prices: list[float]) -> dict:
        qs: dict[str, Any] = {}
        for slot, cands in self.open.items():
            if slot in ("rows",):
                continue
            qs[slot] = {k: [v for v in self.stats.get(k, {}).get("vals", [])[:3]] for k in cands}
        try:
            amt = float((budget_raw or {}).get("amount"))
        except (TypeError, ValueError):
            amt = None
        if amt and amt > 0 and prices:
            low = sum(1 for p in prices if p <= amt) / len(prices)
            self.money_odd = amt == int(amt) and any(abs(p - round(p)) > 1e-9 for p in prices)
            if low < 0.1 or amt > 100 * max(prices) or self.money_odd:
                qs["budget"] = {"stated": budget_raw, "sample_prices": sorted(prices)[:8]}
        keys = [k for k in ([self.stock] if self.stock else []) + list(self.open.get("stock", [])) if k in self.stats]
        texts = {k: sorted(self.stats[k]["set"])[:8] for k in dict.fromkeys(keys) if self.stats[k]["str"] > 0 and 0 < len(self.stats[k]["set"]) <= 8}
        if texts:
            qs["stock_values"] = texts
        return qs

    def ask_scale(self, run: Any, q: dict) -> Any:
        run.notes.append("budget scale question unanswered: asked once more on the strong chain")
        return run.model_json("label", CONFIRM_SYS, f"budget: {Text.json(q, 600)}", max_tokens=120,
                              deadline=TASK_SECONDS - 60, chain=STRONG)

    @staticmethod
    def same_digits(a: Any, b: Any) -> bool:
        try:
            x, y = float(a), float(b)
        except (TypeError, ValueError):
            return True
        if x <= 0 or y <= 0:
            return True
        return re.sub(r"\D", "", f"{x:.6f}").strip("0") == re.sub(r"\D", "", f"{y:.6f}").strip("0")

    def confirm(self, run: Any) -> None:
        if self.confirmed:
            return
        self.confirmed = True
        prices = [p for p in (run.cat.price(it) for it in run.cat.items.values()) if p is not None and p > 0]
        qs = self.questions(run.budget_raw, prices)
        if not qs:
            return
        user = "\n".join(f"{slot}: {Text.json(v, 600)}" for slot, v in qs.items())
        d = run.model_json("label", CONFIRM_SYS, user, max_tokens=250, deadline=TASK_SECONDS - 60)
        if not isinstance(d, dict) and "budget" in qs:
            d = self.ask_scale(run, qs["budget"])
            qs = {"budget": qs["budget"]} if isinstance(d, dict) else qs
        if not isinstance(d, dict):
            self.budget_lost = "budget" in qs
            return
        for slot in list(qs):
            if slot == "budget":
                try:
                    b = float(d.get("budget")) if d.get("budget") is not None else None
                except (TypeError, ValueError):
                    b = None
                if b and b > 0 and not self.same_digits(b, (qs["budget"].get("stated") or {}).get("amount")):
                    run.notes.append(f"budget scale answer {b:g} dropped: not a decimal shift of the stated amount")
                    b = None
                self.budget_hint = b if b and b > 0 else None
                continue
            if slot == "stock_values":
                yes = d.get("stock_available_values")
                if isinstance(yes, list):
                    self.stock_yes = {str(x).strip().lower() for x in yes if isinstance(x, (str, int))}
                continue
            pick = d.get(slot)
            if isinstance(pick, str) and pick in self.open.get(slot, []):
                setattr(self, slot, pick)
                self.fixed.add(slot)
                self.open.pop(slot, None)
            if slot == "stock" and isinstance(d.get("stock_true_means_available"), bool):
                self.stock_true_available = d["stock_true_means_available"]

    def slots(self) -> dict:
        return {k: getattr(self, k) for k in SLOTS} | {"stock_true_available": self.stock_true_available, "open": self.open,
                                                        "views": {k: {kk: vv for kk, vv in v.items()} for k, v in self.views.items()},
                                                        "cart_qty": self.cart_qty, "log": self.log}


class Plan:
    PHASES = ("find", "chains", "read", "ask", "proof", "sink", "commit")
    HOP = 3

    def __init__(self, run: Any) -> None:
        self.r = run
        self.alloc = {p: 0 for p in self.PHASES}
        self.used: collections.Counter = collections.Counter()
        self.mark: collections.Counter = collections.Counter()
        self.hops_seen = 0
        self.refused = 0
        self.extra = 0
        self.stage = "find"
        self.done: set[str] = set()
        self.log: list[dict] = []
        self.endgame = False
        self.fit_memo: tuple = ((), 0)

    def begin(self, phase: str, why: str = "") -> None:
        self.r.phase = phase
        if self.PHASES.index(phase) > self.PHASES.index(self.stage):
            self.stage = phase
        self.replan(why or phase)

    def past(self, phase: str) -> bool:
        return phase in self.done or self.PHASES.index(self.stage) > self.PHASES.index(phase)

    def spend(self, phase: str, n: int) -> None:
        self.used[phase] += n

    def room(self, phase: str) -> int:
        if phase == "sink":
            return max(0, self.r.steps_left() - self.finish_need(minimal=True) - self.repair_reserve())
        pool = ("chains", "read") if phase == "chains" else (phase,)
        after = self.PHASES[self.PHASES.index(pool[-1]) + 1:]
        later = sum(self.alloc[p] - (self.used[p] - self.mark[p]) for p in after if p in self.alloc)
        own = sum(self.alloc.get(p, 0) - (self.used[p] - self.mark[p]) for p in pool)
        if "read" in pool:
            own += self.extra
            later -= self.extra
        return max(0, min(own, self.r.steps_left() - max(0, later)))

    def fit_room(self) -> int:
        return max(0, self.r.steps_left() - self.finish_need(minimal=True) - self.repair_reserve() + self.fit_hold())

    def unsearched_reveal(self) -> bool:
        return any(x["active"] and x["source"] != "stated" and x["op"] not in ("not", "ref") and not x.get("searched")
                   for x in self.r.ledger)

    def chase_room(self) -> int:
        base = self.room("read")
        if self.refused:
            return base
        floor = self.finish_need(minimal=True) + self.lines() + 2
        least = min(2, self.r.steps_left() - floor)
        return max(base + 1, least, 0)

    def slack(self) -> int:
        return max(0, self.r.steps_left() - self.finish_need(minimal=True) - self.repair_reserve())

    def insure_cost(self) -> int:
        r = self.r
        pages = len(r.order.unpaged([a.get("pick") for a in r.articles]))
        return pages + (0 if r.order.cart_viewed() or not r.order.views() else 1)

    def repair_reserve(self) -> int:
        if self.refused:
            return self.r.order.swap_cost(1)
        return min(self.lines() + 1 if self.endgame else self.lines() + 3, self.r.order.swap_cost(1) + 2) + self.fit_hold()

    def fit_hold(self) -> int:
        r = self.r
        items = r.cat.items
        pending = bool(r.cheapest and r.together and not getattr(r.judge, "fit_solved", False))
        key = (tuple(a.get("pick") for a in r.articles), len(items), r.steps, len(r.ledger), r.judge.asked, pending, FIT_STATE["version"])
        if self.fit_memo[0] != key:
            bad = 0
            for pr in r.together:
                if len(pr) != 2 or not all(isinstance(x, int) and 0 <= x < len(r.articles) for x in pr):
                    continue
                x, h = (items.get(r.articles[n].get("pick") or "") for n in pr)
                bad += x is None or h is None or Judge.tier(*fit_judge(r.cat.fit_record(h), r.cat.fit_record(x), [])) >= 9
            for ai in r.judge.coded_lines():
                it = items.get(r.articles[ai].get("pick") or "")
                bad += it is None or r.basket.state(it, ai) != "v" or any(r.cat.code_verdict(it, q) != "yes" for q in r.judge.code_reqs(ai))
            self.fit_memo = (key, min(FIT_HOLD_CAP, r.step_cap // 5, FIT_HOLD * (max(bad, len(r.together)) if pending else bad)))
        return self.fit_memo[1]

    def chain_reserve(self) -> int:
        r = self.r
        ask = 0 if "ask" in self.done else max(0, self.alloc["ask"] - (self.used["ask"] - self.mark["ask"]))
        approval = 1 if r.confirm and r.msgs_total_left() > 0 else 0
        spare = r.steps_left() - self.finish_need(minimal=True) - (self.repair_reserve() if self.refused else 1 + self.fit_hold())
        return max(0, spare - ask - approval - self.reads_pending())

    def lines(self) -> int:
        return max(1, len(self.r.articles))

    def reads_pending(self) -> int:
        r = self.r
        if not any(r.store.tm.get(v) for v in ("detail", "specs")):
            return 0
        n = 0
        for art in r.articles:
            it = r.cat.items.get(art.get("pick") or "")
            if it is None or not (it.get("read") or it.get("paged")):
                n += 1
        return n

    def hop_relief(self) -> int:
        if self.refused:
            return 0
        spare = self.r.steps_left() - self.finish_need(minimal=True) - self.lines() - 1
        return max(0, min(2, spare))

    def stale(self) -> int:
        r = self.r
        picks = {a.get("pick") for a in r.articles if a.get("pick")}
        return sum(1 for cid, q in r.cart.items() if q > 0 and cid not in picks)

    def finish_need(self, minimal: bool = False) -> int:
        r = self.r
        buy = r.store.tm.get("buy") or {}
        held = r.order.held() if r.store.tm.get("cart") else {}
        dead = {a["pick"] for a in r.articles if a.get("pick") in held and r.order.dead(r.cat.items.get(a["pick"]) or {})}
        missing = sum(1 for a in r.articles if not a.get("pick") or a["pick"] not in held or a["pick"] in dead)
        basket = (r.store.tm.get("cart") or {}).get("mode") == "basket"
        need = 1 if direct_buy(buy) else (min(1, missing) if basket and held else (1 if basket else (missing if held else self.lines()))) + 1
        if r.store.tm.get("cart") and not direct_buy(buy) and not basket:
            need += self.stale() + len(dead)
        need += 1 if buy.get("token") and r.store.tm.get("quote") else 0
        if minimal:
            return need
        need += 1 if r.confirm and r.msgs_total_left() > 0 else 0
        need += self.repair_reserve()
        need += self.reads_pending()
        return need

    def chains_need(self) -> int:
        need = 0
        for x in self.r.ledger:
            if x["active"] and x["op"] == "ref" and not x.get("done"):
                done = len(x.get("opened") or [])
                need += self.HOP * max(2 if not done else 1, self.hops_seen + 1 - done)
        return need

    def replan(self, why: str = "") -> None:
        r = self.r
        left = r.steps_left()
        self.mark = collections.Counter(self.used)
        view = self.insure_cost()
        commit = self.finish_need() + view
        chains = self.chains_need()
        ask = r.msgs_left() if r.store.tm.get("ask") and "ask" not in self.done else 0
        proof = (2 * len(r.articles) + (1 if (r.store.tm.get("filter") or {}).get("max_price") else 0)) if r.cheapest and not self.past("proof") else 0
        rest = left - commit - chains - ask - proof
        approval = 1 if r.confirm and r.msgs_total_left() > 0 else 0
        reads = self.reads_pending()
        reserve = self.repair_reserve()
        chain_floor = chains
        ask_keep = min(ask, 1 if len(r.articles) == 1 else 2)
        keep = min(1, reserve) if chain_floor else 0
        order = [("view", view), ("proof", proof), ("reads", reads), ("chains", max(0, chains - chain_floor)), ("ask", max(0, ask - ask_keep)),
                 ("reserve", reserve - keep), ("ask", ask_keep), ("approval", approval), ("chains", chain_floor), ("reserve", keep)]
        for what, size in order:
            if rest >= 0:
                break
            cut = min(size, -rest)
            if cut <= 0:
                continue
            rest += cut
            if what == "proof":
                proof -= cut
            elif what == "chains":
                chains -= cut
            elif what == "ask":
                ask -= cut
            else:
                commit -= cut
        if rest < 2 and not any(it["read"] for it in r.cat.items.values()):
            short = 2 - rest
            cut = min(proof, short)
            proof -= cut
            rest += cut
            short -= cut
            cut = min(max(0, commit - self.finish_need(minimal=True) - self.lines() - 2), short) if not self.refused else 0
            commit -= cut
            rest += cut
        rest = max(0, rest)
        lacking = sum(1 for a in r.articles if not a.get("ids"))
        find = 0 if self.past("find") else max(min(rest, lacking), rest // 3)
        self.alloc = {"find": find, "chains": chains, "read": rest - find, "ask": ask, "proof": proof, "sink": 0, "commit": commit}
        self.log.append({"why": why, "step": r.steps, "left": left, **self.alloc})

    def note_hop(self, hops: int) -> None:
        if hops > self.hops_seen:
            self.hops_seen = hops
            self.replan("hop")


class Order:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.attempts = 0
        self.belief: dict[str, int] = {}
        self.stamp = -1
        self.verified = False
        self.first_cart_step: int | None = None
        self.audits = 0
        self.checked = 0
        self.op_checked = False
        self.wrote_ok = False
        self.no_lines = False
        self.err_shape: Any = False
        self.ref_owner: dict[str, str] = {}
        self.ref_key_of: dict[str, str] = {}

    def lines(self) -> dict[str, int]:
        r = self.r
        out: dict[str, int] = {}
        filled: list[int] = []
        for art in r.articles:
            pick = art.get("pick") if art.get("pick") in r.cat.items else None
            if pick and self.dead(r.cat.items[pick]):
                r.notes.append(f"article {art['i']}: gone pick {pick} dropped")
                pick = art["pick"] = None
            if not pick:
                ai = art["i"]
                c = [it for it in r.candidates(ai, read_only=False) if not r.contradicted(it, ai)]
                if not c and self.rescue_search(art):
                    c = [it for it in r.candidates(ai, read_only=False) if not r.contradicted(it, ai)]
                if not c:
                    c = [it for it in r.candidates(ai, read_only=False) if r.fit(it, ai)[1] == 0]
                if not c:
                    c = sorted((it for it in r.cat.items.values() if it["id"] in (art.get("ids") or []) and not it["read"] and not self.dead(it)),
                               key=lambda it: r.cat.price(it) if r.cat.price(it) is not None else 1e18) or r.candidates(ai)
                c = [it for it in c if not self.foreign(it, ai)]
                if c:
                    pick = c[0]["id"]
                    art["pick"] = pick
                    r.notes.append(f"article {art['i']} filled with {pick}")
                    if any(art["i"] in p for p in r.together):
                        filled.append(art["i"])
            if pick:
                out[pick] = out.get(pick, 0) + art["qty"]
        if filled and not r.finished:
            r.judge.see_pointers(filled, loose=True)
        return out

    def foreign(self, it: dict, ai: int) -> bool:
        r = self.r
        art = r.articles[ai]
        others = [o for o in r.articles if o["i"] != ai]
        if any(o.get("pick") == it["id"] for o in others):
            return True
        if it["id"] in (art.get("ids") or []):
            return False
        cat = r.cat.category(it)
        if cat and (art.get("cat") == cat or any(same_word(x, y) for x in Text(cat).words for y in Text(art["noun"]).words)):
            return False
        return any(it["id"] in (o.get("ids") or []) or (cat and o.get("cat") == cat) for o in others)

    def rescue_search(self, art: dict) -> bool:
        r = self.r
        ai = art["i"]
        if r.steps_left() - r.plan.finish_need(minimal=True) < 2 or r.t() > GUARD_AT - 20:
            return False
        stated = [x for x in r.reqs(ai) if x["source"] == "stated" and x["op"] in ("is", "eq") and x["value"] and not code_shaped(Catalog.wanted_text(x))]
        if not stated:
            return False
        read = [it for it in r.cat.items.values() if it["read"]]
        rare = min(stated, key=lambda x: sum(1 for it in read if r.cat.verdict(it, x) == "yes"))
        val = rare["value"] if isinstance(rare["value"], str) else rare["value"][0]
        q = " ".join([Text.of(val).strip()] + Text(art["noun"]).words[-1:])[:60]
        if r.finder.state(q) is not None:
            return False
        ids, _, _ = r.hunt.line_run(ai, q)
        r.hunt.add_ids(art, ids)
        pool = r.hunt.unread_pool(ai, ids)
        pool.sort(key=lambda it: 1e18 if r.cat.price(it) is None else r.cat.price(it))
        if pool and r.steps_left() - r.plan.finish_need(minimal=True) >= 1:
            r.hunt.open([pool[0]["id"]])
        r.notes.append(f"article {ai}: rescue search {q!r}, {len(ids)} rows")
        return bool(ids)

    @staticmethod
    def basket_args(b: dict, lines: dict[str, int]) -> dict:
        qmin, qmax = b.get("qmin") or 0, b.get("qmax")
        pos = [(cid, min(max(qmin, q), qmax) if isinstance(qmax, int) else max(qmin, q)) for cid, q in lines.items() if q > 0]
        out = dict(b.get("args") or {})
        shape = b.get("shape")
        if shape == "objects":
            out[b["lines"]] = [{b["item_id"]: cid, b["item_qty"]: q} for cid, q in pos]
        elif shape == "pairs":
            out[b["lines"]] = [[cid, q] if b.get("order", ["str"])[0] == "str" else [q, cid] for cid, q in pos]
        elif shape == "parallel":
            out[b["lines"]] = [cid for cid, _ in pos]
            out[b["qtys"]] = [q for _, q in pos]
        else:
            out[b["lines"]] = [cid for cid, q in pos for _ in range(q)]
        return out

    @staticmethod
    def basket_read(b: dict, args: dict) -> dict[str, int]:
        got: dict[str, int] = {}
        rows = args.get(b.get("lines") or "")
        rows = rows if isinstance(rows, list) else []
        shape = b.get("shape")
        if shape == "parallel":
            qs = args.get(b.get("qtys") or "")
            rows = [[cid, q] for cid, q in zip(rows, qs if isinstance(qs, list) else [])]
        for row in rows:
            if shape == "objects" and isinstance(row, dict):
                cid, q = row.get(b["item_id"]), row.get(b["item_qty"])
            elif shape in ("pairs", "parallel") and isinstance(row, list) and len(row) >= 2:
                cid, q = (row[0], row[1]) if shape == "parallel" or b.get("order", ["str"])[0] == "str" else (row[1], row[0])
            else:
                cid, q = row, 1
            if isinstance(cid, str) and isinstance(q, (int, float)) and q > 0:
                got[cid] = got.get(cid, 0) + int(q)
        return got

    def target(self, lines: dict[str, int]) -> dict[str, int]:
        b = self.r.store.tm.get("cart") or {}
        return self.basket_read(b, self.basket_args(b, lines))

    def calls_for(self, lines: dict[str, int], have: dict[str, int]) -> list[tuple[str, dict]]:
        r = self.r
        cart = r.store.tm.get("cart") or {}
        if cart.get("mode") == "basket":
            want = self.target(lines)
            if want == {c: q for c, q in have.items() if q > 0}:
                return []
            if not want and cart.get("min", 0) >= 1:
                r.notes.append("basket cannot be emptied: the tool needs at least one line")
                return []
            return [(cart["name"], self.basket_args(cart, lines))]
        name, cid_k, qty_k = cart.get("name"), cart.get("id"), cart.get("qty")
        sw, up, down = cart.get("switch"), cart.get("up"), cart.get("down")
        qmin, qmax = cart.get("qmin") or 0, cart.get("qmax")
        base = dict(cart.get("args") or {})
        rem = r.store.tm.get("remove")
        calls: list[tuple[str, dict]] = []

        def amount(k: int) -> int:
            k = max(qmin, k)
            return min(k, qmax) if isinstance(qmax, int) else k

        def more(cid: str, k: int) -> None:
            args = {**base, cid_k: cid}
            if sw:
                args[sw] = up
            if qty_k:
                args[qty_k] = amount(k)
            calls.append((name, args))

        def less(cid: str, k: int) -> None:
            if sw and qty_k:
                calls.append((name, {**base, sw: down, cid_k: cid, qty_k: amount(k)}))
            elif sw:
                calls.append((name, {**base, sw: down, cid_k: cid}))
            elif qty_k and qmin <= 0:
                calls.append((name, {**base, cid_k: cid, qty_k: 0}))
            elif rem:
                calls.append((rem["name"], {rem["line"]: self.ref_for(cid)}))
            else:
                r.notes.append(f"cannot remove {cid}: no switch and quantity floor {qmin}")

        for cid, q in have.items():
            if q > 0 and cid not in lines:
                less(cid, q)
        for cid, q in lines.items():
            h = have.get(cid, 0)
            if h == q:
                continue
            if cart.get("mode") == "add":
                if q > h:
                    more(cid, q - h)
                elif rem and not sw and not (qty_k and qmin <= 0):
                    less(cid, h)
                    more(cid, q)
                else:
                    less(cid, h - q)
            else:
                more(cid, q)
        return calls

    def apply(self, calls: list[tuple[str, dict]], obs: list) -> None:
        r = self.r
        cart = r.store.tm.get("cart") or {}
        sw, down, cid_k, qty_k = cart.get("switch"), cart.get("down"), cart.get("id"), cart.get("qty")
        rem = r.store.tm.get("remove") or {}
        for (n, a), o in zip(calls, obs):
            if self.refused(o):
                r.notes.append(f"cart call refused: {Text.json(o, 120)}")
                continue
            if rem and n == rem["name"]:
                v = a.get(rem["line"])
                self.belief[self.ref_owner.get(v, v)] = 0
                continue
            self.wrote_ok = True
            if cart.get("mode") == "basket":
                self.belief = self.basket_read(cart, a)
                continue
            cid = a.get(cid_k)
            q = a.get(qty_k) if qty_k else 1
            q = q if isinstance(q, int) else 1
            if not isinstance(cid, str):
                continue
            if cart.get("mode") == "add":
                if sw and a.get(sw) == down:
                    self.belief[cid] = max(0, self.belief.get(cid, 0) - q)
                else:
                    self.belief[cid] = self.belief.get(cid, 0) + q
            else:
                self.belief[cid] = q

    def ref_for(self, cid: str) -> str:
        r = self.r
        rem = r.store.tm["remove"]
        if rem["by"] == "id":
            return cid
        row = r.cart_rows.get(cid) or {}
        cands = [k for k, v in row.items() if isinstance(v, str) and v != cid and " " not in v and 0 < len(v) <= 64 and k not in rem["tried"]]
        cands.sort(key=lambda k: (k != rem.get("ref_key"), not set(Text.key_stems(k)) & set(Text.key_stems(rem["line"]))))
        if not cands:
            return cid
        val = row[cands[0]]
        self.ref_owner[val] = cid
        self.ref_key_of[val] = cands[0]
        return val

    def note_removes(self, calls: list[tuple[str, dict]], obs: list) -> None:
        r = self.r
        rem = r.store.tm.get("remove")
        if not rem:
            return
        for (n, a), o in zip(calls, obs):
            if n != rem["name"]:
                continue
            v = a.get(rem["line"])
            key = self.ref_key_of.get(v)
            if self.refused(o):
                if rem["by"] == "id":
                    rem["by"] = "ref"
                    r.notes.append("remove tool refused a listing id: switching to the line reference the basket shows")
                elif key:
                    rem["tried"].append(key)
                    r.notes.append(f"remove tool refused line field {key!r}")
            elif key and rem.get("ref_key") != key:
                rem["ref_key"] = key
                r.notes.append(f"remove tool verified with line field {key!r}")

    @staticmethod
    def runtime_fault(o: Any) -> bool:
        u = Store.unwrap(o)[0]
        return isinstance(u, dict) and isinstance(u.get("error"), str) and len(u) <= 2

    def refused(self, o: Any) -> bool:
        r = self.r
        if r.store.ok_reply(o) is not False:
            return False
        u = Store.unwrap(o)[0]
        if not isinstance(u, dict) or not u or self.runtime_fault(o):
            return True
        shape = frozenset(u)
        known = self.error_shape()
        if known is None:
            return True
        return shape == known

    def error_shape(self) -> frozenset | None:
        r = self.r
        if self.err_shape is not False:
            return self.err_shape
        self.err_shape = None
        m = r.store.tm.get("detail") or r.store.tm.get("specs")
        if not m or r.steps_left() < 3 or r.t() > LAST_ENV_AT - 5:
            return None
        got = r.env.send([(m["name"], {**(m.get("args") or {}), m["id"]: "zz-0-none"})])
        u = Store.unwrap(got[0])[0] if got else None
        if isinstance(u, dict) and u and all(isinstance(v, str) for v in u.values()):
            self.err_shape = frozenset(u)
            r.notes.append(f"store error shape learned: {sorted(u)}")
        else:
            self.err_shape = frozenset()
        return self.err_shape

    def fill_extras(self, buy: dict, args: dict, lines: dict[str, int]) -> None:
        r = self.r
        props = (r.schemas.get(buy["name"]) or {}).get("properties") or {}
        skip = {buy.get("token"), buy.get("note"), buy.get("id"), buy.get("qty")} | set(args)
        for k, spec in props.items():
            if k in skip or not isinstance(spec, dict) or (k not in (buy.get("extras") or []) and k not in ((r.schemas.get(buy["name"]) or {}).get("required") or [])):
                continue
            v = r.seen_keys.get(Text.key(k))
            if v is None and spec.get("type") in ("integer", "number"):
                v = r.cart_total
                if v is None:
                    prices = [(r.cat.price(r.cat.items[c]) if c in r.cat.items else None, q) for c, q in (self.held() or lines).items()]
                    v = sum(p * q for p, q in prices) if prices and all(p is not None for p, _ in prices) else None
            if v is not None:
                args[k] = v
                r.notes.append(f"buy arg {k!r} filled with {v!r}")

    @staticmethod
    def token_in(obs: Any, key: str) -> Any:
        stack, depth = [obs], 0
        while stack and depth < 200:
            depth += 1
            x = stack.pop(0)
            if isinstance(x, dict):
                for k, v in x.items():
                    if (k == key or Text.key(k) == Text.key(key)) and isinstance(v, (str, int)) and not isinstance(v, bool):
                        return v
                stack.extend(v for v in x.values() if isinstance(v, (dict, list)))
            elif isinstance(x, list):
                stack.extend(v for v in x if isinstance(v, (dict, list)))
        return None

    def held(self) -> dict[str, int]:
        r = self.r
        if r.cart_stamp > self.stamp:
            self.belief = dict(r.cart)
            self.stamp = r.cart_stamp
        return {k: q for k, q in self.belief.items() if q > 0}

    def unpaged(self, ids: list) -> list[str]:
        r = self.r
        det = r.tm.get("detail") or {}
        if not det.get("id") or not (r.tm.get("specs") or r.tm.get("compare")):
            return []
        return [c for c in dict.fromkeys(ids) if not (r.cat.items.get(c or "") or {}).get("paged")]

    def views(self) -> bool:
        r = self.r
        return bool(r.tm.get("view") and r.tm.get("cart") and not direct_buy(r.tm.get("buy") or {}))

    def cart_viewed(self) -> bool:
        r = self.r
        view = (r.tm.get("view") or {}).get("name")
        writes = {(r.tm.get(k) or {}).get("name") for k in ("cart", "remove")} - {None}
        seen = wrote = -1
        for n, d in enumerate(r.dialogue):
            name = ((d.get("tool_call") or {}).get("function") or {}).get("name")
            if name == view:
                seen = n
            elif name in writes:
                wrote = n
        return seen > wrote >= 0

    def insure(self, lines: dict[str, int]) -> bool:
        r = self.r
        keep = 1 + (0 if r.plan.refused else min(r.plan.repair_reserve(), self.swap_cost(1) + 1))
        ids = self.unpaged(list(lines))
        det = r.tm.get("detail") or {}
        room = r.steps_left() - keep
        shown: list[str] = []
        if ids and room > 0 and r.t() < LAST_ENV_AT - 3:
            shown = ids[:room]
            r.env.send([(det["name"], {**(det.get("args") or {}), det["id"]: c}) for c in shown])
            r.notes.append(f"own pages shown before checkout: {shown}")
        lost = [c for c in shown if c in r.cat.items and self.dead(r.cat.items[c])]
        if r.finished or (lost and r.steps_left() > self.swap_cost(1)):
            return bool(lost) and not r.finished
        view = r.tm.get("view") or {}
        if self.views() and lines and not self.cart_viewed() and r.steps_left() - keep >= 1 and r.t() < LAST_ENV_AT - 3:
            r.env.send([(view["name"], dict(view.get("args") or {}))])
            r.notes.append("cart viewed before checkout")
        return False

    def sync(self, lines: dict[str, int]) -> bool:
        r = self.r
        cart, view = r.store.tm.get("cart"), r.store.tm.get("view")
        if not cart:
            return True
        if cart.get("mode") == "basket":
            return self.sync_basket(lines)
        last: list = []
        for _attempt in range(2):
            have = self.held()
            calls = self.calls_for(lines, have)
            if not calls and have == lines:
                return True
            if calls == last:
                return False
            last = calls
            if calls:
                budget = max(0, r.steps_left() - 1)
                if len(calls) > budget:
                    kept = self.keep_held(lines, have, budget)
                    if kept is not None:
                        lines.clear()
                        lines.update(kept)
                        calls = self.calls_for(lines, have)
                    else:
                        calls = [] if cart.get("switch") else calls[:budget]
                if not calls:
                    return self.held() == lines
                n0 = r.cart_stamp
                obs = r.env.send(calls)
                self.note_removes(calls, obs)
                if obs and cart.get("switch") and not self.op_checked and all(self.refused(o) and not self.runtime_fault(o) for o in obs):
                    self.op_checked = True
                    cart["up"], cart["down"] = cart["down"], cart["up"]
                    cart["args"][cart["switch"]] = cart["up"]
                    r.notes.append(f"cart operation swapped: add is now {cart['up']!r}")
                    calls = self.calls_for(lines, self.held())[:max(0, r.steps_left() - 1)]
                    obs = r.env.send(calls) if calls else []
                if obs and self.first_cart_step is None:
                    self.first_cart_step = r.steps
                if r.cart_stamp <= n0:
                    self.apply(calls, obs)
                    self.stamp = r.steps
                else:
                    last = max((i for i, o in enumerate(obs) if r.store.cart_lines(o) is not None), default=None)
                    if last is not None and last < len(obs) - 1:
                        self.belief = dict(r.cart)
                        self.apply(calls[last + 1:], obs[last + 1:])
                        self.stamp = r.steps
            if view and r.cart_stamp < self.stamp and r.steps_left() > 1 and not r.finished and (
                    self.held() != lines or (cart.get("mode") == "add" and not self.verified)):
                r.env.send([(view["name"], dict(view.get("args") or {}))])
                self.verified = True
            if self.held() == lines:
                return True
        return self.held() == lines

    def keep_held(self, lines: dict[str, int], have: dict[str, int], budget: int, dead_ok: bool = False) -> dict[str, int] | None:
        r = self.r
        spare = [c for c, q in have.items() if q > 0 and c not in lines and c in r.cat.items and (dead_ok or not self.dead(r.cat.items[c]))]
        stand: dict[int, str] = {}
        for art in r.articles:
            if art.get("pick") not in lines or art.get("pick") in have:
                continue
            free = [c for c in spare if c not in stand.values()]
            old = (next((c for c in free if art.get("cat") and r.cat.category(r.cat.items[c]) == art["cat"]), None)
                   or next((c for c in free if not self.foreign(r.cat.items[c], art["i"])), None))
            if old is not None:
                stand[art["i"]] = old
        group = {a["i"]: {a["i"]} for a in r.articles}
        for a, b in r.together:
            if a in group and b in group and group[a] is not group[b]:
                joined = group[a] | group[b]
                for x in joined:
                    group[x] = joined
        out = dict(lines)
        swaps: list[tuple[dict, str]] = []
        for art in r.articles:
            if len(self.calls_for(out, have)) <= budget:
                break
            for ai in sorted(group[art["i"]]):
                a = r.articles[ai]
                if ai not in stand or a["pick"] not in out or any(s is a for s, _ in swaps):
                    continue
                out[a["pick"]] -= a["qty"]
                if out[a["pick"]] <= 0:
                    out.pop(a["pick"])
                out[stand[ai]] = out.get(stand[ai], 0) + a["qty"]
                swaps.append((a, stand[ai]))
        if not swaps or len(self.calls_for(out, have)) > budget:
            return None if dead_ok else self.keep_held(lines, have, budget, True)
        for art, old in swaps:
            r.notes.append(f"cart kept: line {art['i']} stays {old}, {budget} steps cannot pay the swap to {art['pick']}")
            art["backup"] = [y for y in [art["pick"]] + list(art.get("backup") or []) if y != old]
            art["pick"] = old
        return out

    def basket_send(self, lines: dict[str, int]) -> str:
        r = self.r
        view = r.store.tm.get("view")
        want = self.target(lines)
        calls = self.calls_for(lines, self.held())
        if not calls:
            return "same" if self.held() == want else "stuck"
        if r.steps_left() < 2 or r.t() > LAST_ENV_AT - 3:
            return "stuck"
        n0 = r.cart_stamp
        obs = r.env.send(calls)
        if not obs:
            return "stuck"
        if self.first_cart_step is None:
            self.first_cart_step = r.steps
        if self.refused(obs[0]):
            r.notes.append(f"basket write refused: {Text.json(obs[0], 120)}")
            return "refused"
        self.wrote_ok = True
        if r.cart_stamp <= n0:
            self.apply(calls, obs)
            self.stamp = r.steps
            if view and not self.verified and r.steps_left() > 1 and not r.finished:
                self.verified = True
                r.env.send([(view["name"], dict(view.get("args") or {}))])
        if r.cart_stamp > n0 or self.verified:
            self.verified = True
            got = self.held()
            if want and not got:
                self.wrote_ok = False
                return "empty"
            if want and not set(got) & set(want):
                self.wrote_ok = False
                return "empty"
        return "ok"

    def suspect(self, lines: dict[str, int]) -> str | None:
        r = self.r
        held = self.held()
        new = [c for c in lines if c not in held] or list(lines)
        if not new:
            return None
        new.sort(key=lambda c: (c in r.cat.items, bool((r.cat.items.get(c) or {}).get("read")), -list(lines).index(c)))
        return new[0]

    def sync_basket(self, lines: dict[str, int]) -> bool:
        r = self.r
        cart = r.store.tm["cart"]
        first = not self.wrote_ok
        keep = (cart.get("item_id"), cart.get("item_qty"), list(cart.get("alts") or []))
        orig = dict(lines)
        lines = dict(lines)
        mapping = first
        for _try in range(4 + len(keep[2])):
            got = self.basket_send(lines)
            if got in ("ok", "same"):
                if got == "ok" and cart.get("item_id") != keep[0]:
                    r.notes.append(f"basket fields verified: {cart.get('item_id')!r} and {cart.get('item_qty')!r}")
                break
            if got == "stuck":
                break
            if mapping and cart.get("alts"):
                r.notes.append(f"basket write {got} with {cart.get('item_id')!r}/{cart.get('item_qty')!r}; trying other item fields")
                cart["item_id"], cart["item_qty"] = cart["alts"].pop(0)
                self.wrote_ok = False
                continue
            if mapping:
                mapping = False
                cart["item_id"], cart["item_qty"], cart["alts"] = keep[0], keep[1], []
            if got != "refused" or len(lines) < 1:
                break
            bad = self.suspect(lines)
            if bad is None:
                break
            lines.pop(bad)
            it = r.cat.items.get(bad)
            if it is not None:
                it["gone"] = True
            r.notes.append(f"basket write refused: line {bad} dropped as the likely bad id, basket written again")
        return self.held() == self.target(orig)

    def cart_early(self) -> bool:
        r = self.r
        buy = r.store.tm.get("buy") or {}
        if not r.store.tm.get("cart") or not buy or direct_buy(buy) or self.held():
            return False
        lines: dict[str, int] = {}
        for art in r.articles:
            it = r.cat.items.get(art.get("pick") or "")
            if it is None or not it["read"] or r.contradicted(it, art["i"]):
                continue
            lines[it["id"]] = lines.get(it["id"], 0) + art["qty"]
        if not lines or r.steps_left() <= r.plan.finish_need(minimal=True) + len(lines) + 1:
            return False
        was = r.phase
        r.phase = "commit"
        self.sync(lines)
        r.phase = was
        r.notes.append(f"early cart {sorted(lines)} at step {r.steps}")
        r.plan.replan("early cart")
        return True

    def reread_first_line(self, since: int | None = None) -> bool:
        r = self.r
        held = self.held()
        first = next(iter(held), None) or next((a.get("pick") for a in r.articles if a.get("pick")), None)
        it = r.cat.items.get(first or "")
        mark = self.first_cart_step if since is None else since
        if it is None or mark is None or it.get("read_step", -1) >= mark or r.plan.slack() < 1 or r.plan.refused:
            return False
        two_tools = bool(r.tm.get("specs") and r.tm.get("detail"))
        was = r.phase
        r.phase = "commit"
        r.hunt.open([first], what="state" if two_tools else "facts", force=True)
        r.phase = was
        it["read_step"] = r.steps
        knocked = r.cat.in_stock(it) is False
        if knocked:
            it["gone"] = True
        r.notes.append(f"re-read first line {first}: stock {r.cat.in_stock(it)}, price {r.cat.price(it)}")
        return knocked

    def checklist(self) -> bool:
        r = self.r
        before = [a.get("pick") for a in r.articles]
        self.reread_first_line(since=self.stamp)
        r.judge.recheck()
        searched = False
        for art in r.articles:
            ai = art["i"]
            it = r.cat.items.get(art.get("pick") or "")
            if it is None:
                continue
            lost = not r.ok_basic(it, ai)
            if not lost and not r.contradicted(it, ai):
                continue
            cat = r.cat.category(it)
            good = [c for c in r.candidates(ai) if not r.contradicted(c, ai) and r.judge.fully_met(c, ai) and (not cat or r.cat.category(c) == cat)]
            if good:
                art["pick"] = good[0]["id"]
                r.notes.append(f"checklist: {'lost' if lost else 'incomplete'} line {it['id']} -> {good[0]['id']}")
                continue
            if not any(not r.contradicted(c, ai) for c in r.candidates(ai)) and r.plan.slack() >= 2 and self.rescue_search(art):
                searched = True
            elif r.plan.slack() >= 3 and r.hunt.prove_line(ai):
                searched = True
        if r.cheapest:
            r.phase = "sink"
            for art in r.articles:
                r.hunt.prove_pick(art["i"], 3)
            r.phase = "commit"
        if searched and r.t() < GUARD_AT - 20:
            r.judge.decide(strong=False)
        if r.tm.get("detail") and r.tm.get("specs"):
            unpaged = [a["pick"] for a in r.articles if a.get("pick") in r.cat.items and not r.cat.items[a["pick"]].get("paged")]
            room = r.plan.slack() - 1
            if unpaged and room > 0:
                r.hunt.open(unpaged[:room], what="state", force=True)
        if r.together and [a.get("pick") for a in r.articles] != before and r.t() < GUARD_AT - 20:
            r.judge.check_fits({a["i"]: r.candidates(a["i"])[:9] for a in r.articles})
        return [a.get("pick") for a in r.articles] != before

    def read_through(self, lines: dict[str, int]) -> None:
        r = self.r
        calls = []
        for cid in lines:
            it = r.cat.items.get(cid)
            if it is not None and (it.get("read") or it.get("paged")):
                continue
            m = r.store.tm.get("specs") or r.store.tm.get("detail")
            if m and m.get("id"):
                calls.append((m["name"], {**(m.get("args") or {}), m["id"]: cid}))
        room = r.steps_left() - 1 - (1 if r.confirm and r.approved_cart != lines and r.msgs_total_left() > 0 else 0)
        if calls and room > 0 and r.t() < LAST_ENV_AT - 3:
            r.env.send(calls[:room])

    def checkout(self, lines: dict[str, int]) -> bool:
        r = self.r
        buy = r.store.tm.get("buy")
        if not buy or r.finished or r.steps_left() < 1:
            return r.finished
        if self.insure(self.held() if r.tm.get("cart") and not direct_buy(buy) else lines):
            r.notes.append("checkout held back: a listing page showed an ordered line gone")
            return False
        if r.finished or r.steps_left() < 1:
            return r.finished
        args = dict(buy.get("args") or {})
        if buy.get("basket") and lines:
            args.update(self.basket_args(buy["basket"], lines))
        elif buy.get("id") and lines:
            cid, q = next(iter(lines.items()))
            args[buy["id"]] = cid
            if buy.get("qty"):
                args[buy["qty"]] = q
        if buy.get("extras") or [k for k in ((r.schemas.get(buy["name"]) or {}).get("required") or []) if k not in args and k != buy.get("token")]:
            view = r.store.tm.get("view")
            if view and r.steps_left() >= 3 and r.cart_stamp < r.steps - 1:
                r.env.send([(view["name"], dict(view.get("args") or {}))])
            self.fill_extras(buy, args, lines)
        qt = r.store.tm.get("quote")
        if qt and buy.get("token") and r.steps_left() >= 2:
            got = r.env.send([(qt["name"], dict(qt.get("args") or {}))])
            tok = self.token_in(r.store.unwrap(got[0])[0] if got else None, buy["token"])
            if tok is not None:
                args[buy["token"]] = tok
            else:
                r.notes.append(f"quote reply held no {buy['token']!r}: {Text.json(got[0] if got else None, 160)}")
            if r.finished or r.steps_left() < 1:
                return r.finished
        self.attempts += 1
        r.env.send([(buy["name"], args)])
        if r.finished:
            r.ordered = True
            return True
        r.plan.refused += 1
        r.notes.append(f"checkout {self.attempts} refused")
        return False

    def recheck_picks(self) -> None:
        r = self.r
        event = r.knocked
        r.knocked = None
        if not event:
            return
        picks = [a["pick"] for a in r.articles if a.get("pick") and a["pick"] in r.cat.items]
        if not picks:
            return
        if event["gone"] and len(picks) == 1 and not event.get("unknown"):
            r.notes.append("notice gone: single pick marked, no re-read")
            return
        what = "state" if (r.tm.get("specs") and r.tm.get("detail")) else "facts"
        room = r.steps_left() - r.plan.finish_need(minimal=True) - self.swap_cost(len(picks))
        if room >= 1:
            r.hunt.open(picks[:room], what=what, force=True)
        for pid in picks:
            it = r.cat.items[pid]
            if (event["gone"] or event.get("unknown")) and r.cat.in_stock(it) is False:
                it["gone"] = True
        r.notes.append(f"notice {'gone' if event['gone'] else 'price'}: picks re-read, lost "
                       + str([a['pick'] for a in r.articles if a.get('pick') in r.cat.items and not r.ok_basic(r.cat.items[a['pick']], a['i'])]))

    def move_calls(self, picks: dict[int, str]) -> int:
        lines: dict[str, int] = {}
        for art in self.r.articles:
            pid = picks.get(art["i"], art.get("pick"))
            if pid:
                lines[pid] = lines.get(pid, 0) + max(1, int(art.get("qty") or 1))
        return len(self.calls_for(lines, self.held())) if lines else 0

    def move_extra(self, picks: dict[int, str]) -> int:
        return max(0, self.move_calls(picks) - self.move_calls({}))

    def swap_cost(self, lines: int = 1) -> int:
        cart = self.r.store.tm.get("cart") or {}
        per = 2 if (cart.get("switch") or cart.get("mode") in ("add", "set")) else 1
        return per * max(0, lines)

    def dead(self, it: dict) -> bool:
        return bool(it.get("gone")) or self.r.cat.in_stock(it) is False

    def by_verdict(self, cands: list[dict], ai: int) -> list[dict]:
        r = self.r
        sure = [it for it in cands if r.basket.state(it, ai) == "v"]
        if not sure:
            return cands
        if r.cheapest:
            sure = sorted(sure, key=lambda it: (1e18 if r.cat.price(it) is None else r.cat.price(it), it["id"]))
        return sure + [it for it in cands if it not in sure]

    def thin_repair(self) -> bool:
        r = self.r
        before = [a.get("pick") for a in r.articles]
        afford = max(0, (r.steps_left() - 1) // max(1, self.swap_cost(1)))
        for art in r.articles:
            ai = art["i"]
            cur = r.cat.items.get(art.get("pick") or "")
            if cur is not None and r.ok_basic(cur, ai):
                continue
            if cur is not None and self.dead(cur) and afford <= 0 and r.steps_left() >= self.swap_cost(1) + 1:
                afford = 1
            if afford <= 0:
                r.notes.append(f"thin repair: article {ai} kept, no steps to replace its line")
                continue
            good = self.by_verdict([c for c in r.candidates(ai) if r.ok_basic(c, ai) and not r.contradicted(c, ai)]
                                   or [c for c in r.candidates(ai) if r.ok_basic(c, ai) and r.fit(c, ai)[1] == 0]
                                   or [c for c in r.candidates(ai) if r.ok_basic(c, ai)], ai)
            if good:
                art["pick"] = good[0]["id"]
                afford -= 1
        r.notes.append("thin repair: no reads, the second checkout kept")
        r.plan.replan("thin repair")
        return [a.get("pick") for a in r.articles] != before

    def settled(self) -> bool:
        r = self.r
        if r.cheapest or Text.of(r.spec.get("priority")) == "best":
            return False
        for art in r.articles:
            it = r.cat.items.get(art.get("pick") or "")
            if it is None or not r.ok_basic(it, art["i"]) or r.contradicted(it, art["i"]) or r.fit(it, art["i"])[1] > 0:
                return False
        return True

    def repair(self) -> bool:
        r = self.r
        before = [a.get("pick") for a in r.articles]
        r.asker.catch_up()
        r.cheap.after_loss()
        if r.steps_left() <= r.plan.finish_need(minimal=True) + self.swap_cost(1):
            return self.thin_repair() or [a.get("pick") for a in r.articles] != before
        last = r.notices[-1] if r.notices else None
        if (r.knocked is None and self.attempts and r.plan.refused
                and not (isinstance(last, dict) and any(last.get(k) for k in ("budget", "dropped", "quantity", "gone", "price", "values")))):
            r.knocked = {"gone": False, "price": False, "unknown": True}
        drop = r.dropped_since
        self.recheck_picks()
        r.plan.replan("refusal")
        r.phase = "read"
        if r.dropped_since and self.settled():
            r.dropped_since = False
            r.notes.append("drop notice: every pick still meets every need and nothing ranks the picks, no new search")
        if r.dropped_since and r.t() < GUARD_AT - 40 and r.plan.room("read") >= 3:
            r.dropped_since = False
            n = max(1, len(r.articles))
            for art in r.articles:
                art["proved_below"] = None
                r.phase = "find"
                r.plan.alloc["find"] = r.plan.alloc["read"] // 3
                r.hunt.find(art["i"], max(1, r.plan.room("find") // n))
                r.phase = "read"
                r.hunt.wave1(art["i"], max(1, r.plan.room("read") // n))
            if r.cheapest:
                r.phase = "proof"
                r.plan.alloc["proof"] = r.plan.room("read")
                r.hunt.prove_cheapest()
            r.judge.decide()
        elif r.dropped_since and r.t() < GUARD_AT - 25:
            r.dropped_since = False
            r.judge.decide()
        else:
            lost = [art for art in r.articles if r.cat.items.get(art.get("pick") or "") is None
                    or not r.ok_basic(r.cat.items[art["pick"]], art["i"]) or r.fit(r.cat.items[art["pick"]], art["i"])[1] > 0]
            if lost and r.t() < GUARD_AT - 30 and r.plan.room("read") >= 2:
                for art in lost:
                    if any(not r.contradicted(it, art["i"]) for it in r.candidates(art["i"])):
                        continue
                    spare = r.steps_left() - r.plan.finish_need(minimal=True)
                    if not (spare >= 4 and r.hunt.refind(art["i"])):
                        r.hunt.wave1(art["i"], max(1, r.plan.room("read") // len(lost)))
                r.judge.decide()
                if any(a.get("judged_none") for a in r.articles) and r.hunt.expand():
                    r.judge.decide()
            else:
                for art in r.articles:
                    ai = art["i"]
                    cur = r.cat.items.get(art.get("pick") or "")
                    fits = [it for it in r.candidates(ai) if r.fit(it, ai)[1] == 0]
                    if cur is None or not r.ok_basic(cur, ai) or r.fit(cur, ai)[1] > 0:
                        c = self.by_verdict([it for it in fits if not r.contradicted(it, ai)] or fits or r.candidates(ai), ai)
                        art["pick"] = c[0]["id"] if c else art.get("pick")
                    elif r.cheapest and fits and fits[0]["id"] != cur["id"] and (r.cat.price(fits[0]) or 1e18) < (r.cat.price(cur) or 1e18):
                        art["pick"] = fits[0]["id"]
        r.cheap.after_event(drop)
        if r.cheapest and r.steps_left() > r.plan.finish_need(minimal=True) + 2:
            r.phase = "sink"
            for art in r.articles:
                r.hunt.prove_pick(art["i"], 3)
        r.phase = "commit"
        r.plan.replan("repaired")
        return [a.get("pick") for a in r.articles] != before

    def run(self) -> None:
        r = self.r
        if not r.store.tm.get("buy"):
            return
        r.phase = "commit"
        for attempt in range(4):
            if r.finished or r.steps_left() < 1:
                return
            lines = self.lines()
            if not lines:
                self.no_lines = True
                return
            self.sync(lines)
            if r.finished or r.steps_left() < 1:
                return
            self.read_through(lines)
            if r.judge.stamp and r.judge.ledger_mark() != r.judge.stamp and r.t() < GUARD_AT - 25 and attempt < 3:
                touched = r.judge.changed_lines()
                keep = {a["i"]: (a.get("pick"), a.get("backup")) for a in r.articles if a["i"] not in touched}
                if touched:
                    r.judge.decide(strong=False)
                    for a in r.articles:
                        if a["i"] in keep:
                            a["pick"], a["backup"] = keep[a["i"]]
                    r.notes.append(f"re-decide before cart for lines {sorted(touched)}")
                    r.judge.metric_pick("re-decide")
                else:
                    r.judge.line_stamp = r.judge.line_marks()
                    r.judge.stamp = r.judge.ledger_mark()
                if [a.get("pick") for a in r.articles if a.get("pick")] and self.lines() != lines:
                    continue
            if self.checked < 2 and r.t() < GUARD_AT - 20:
                self.checked += 1
                if self.checklist():
                    continue
            if self.audits < 2 and r.t() < GUARD_AT - 20 and r.chains.audit():
                self.audits += 1
                r.judge.decide(strong=False)
                continue
            if attempt < 3 and r.asker.catch_up() and self.repair():
                continue
            if (r.confirm or r.pre_note) and r.approved_cart != lines and r.msgs_total_left() > 0 and r.steps_left() > 1:
                r.asker.ask_approval(lines)
                if r.finished or r.steps_left() < 1:
                    return
                if attempt < 3 and r.asker.catch_up() and self.repair():
                    continue
            if attempt < 3 and any([r.judge.fit_exact("guard"), r.judge.code_guard("guard")]):
                continue
            if self.checkout(lines):
                return
            if attempt < 3:
                self.repair()

    def finish(self) -> None:
        r = self.r
        if r.finished or r.env_dead or r.t() > LAST_ENV_AT:
            return
        buy, dec = r.store.tm.get("buy"), r.store.tm.get("decline")
        r.phase = "commit"
        held = self.held()
        lines = self.lines()
        if (buy and r.store.tm.get("cart") and not direct_buy(buy) and held != lines and lines and r.steps_left() >= 2
                and not (r.drive_carted and not self.wrote_ok)):
            self.sync(lines)
            held = self.held()
        if buy and (held or direct_buy(buy) or r.drive_carted) and r.steps_left() >= 1 and not r.finished:
            if r.steps_left() > len(held) + 1:
                self.read_through(held or lines)
            if (r.confirm or r.pre_note) and r.approved_cart != (held or lines) and r.msgs_total_left() > 0 and r.steps_left() >= 2:
                r.asker.ask_approval(held or lines)
            self.checkout(held or lines)
            if r.finished:
                return
        if dec and not r.finished and r.steps_left() >= 1 and r.decline_ok and not held:
            for _ in range(3):
                r.env.send([(dec["name"], r.decline_args())])
                r.declined = r.finished
                if r.finished or r.steps_left() < 1 or r.t() > LAST_ENV_AT:
                    break
                r.notes.append("decline refused: news read, decline sent again")
                r.asker.catch_up()


FIT_CODE = r"(?:(?-i:[A-Z])[A-Za-z]{1,5} |[A-Za-z]{2,6}-?)\d[\dxX]{0,5}(?:-\d{2,6})?"
FIT_CODE_RX = re.compile(rf"\b({FIT_CODE})\b")
FIT_TITLE_RX = re.compile(rf"[\(\[]\s*(?:[A-Za-z]{{1,8}}\.?\s+|#\s*)?({FIT_CODE})\s*[\)\]]")
FIT_GLUE_RX = re.compile(r"^[\s,/;&+]*$")
FIT_SLOT_RX = re.compile(r"<\d+>")
FIT_ROLES: dict[str, dict[int, str]] = {}
FIT_ASKED: set[str] = set()
FIT_STATE = {"version": 0}
FIT_ROLE_NAMES = ("own", "fits", "not", "see", "about", "theirs", "none")
FIT_READS = 3
FIT_READ_BATCH = 20
FIT_SENT_RX = re.compile(r"(?<=[.!?;\n])\s+|(?<=\.)(?=[A-Z])")
SHAPE_RX = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z][A-Za-z]{1,5} |[A-Za-z]{2,6}-?)\d[\dxX]{1,5}(?![A-Za-z0-9])")
TOKEN_RX = re.compile(r"[A-Za-z0-9]+")
SHAPE_DENSITY = 4
FIT_LOOSE = 5
SERIES_RX = re.compile(r"(?<![A-Za-z0-9-])([A-Z]{2,6}\d{1,2})(?![A-Za-z0-9-])")


def fit_norm(code: str) -> str:
    return re.sub(r"[\s-]", "", code).upper()


def fit_series(code: str) -> str | None:
    m = re.fullmatch(r"([A-Za-z]{2,6})\s?(\d{1,2})-(\d{2,6})", code.strip())
    return (m.group(1) + m.group(2)).upper() if m else None


def fit_slots(sent: str) -> list[list[str]]:
    slots: list[list[str]] = []
    last = None
    for m in FIT_CODE_RX.finditer(sent):
        if last is not None and FIT_GLUE_RX.match(sent[last:m.start()]):
            slots[-1].append(m.group(1))
        else:
            slots.append([m.group(1)])
        last = m.end()
    return slots


def fit_forms(body: str) -> list[tuple[str, list[list[str]], str]]:
    body = re.sub(rf"(?<=[A-Za-z] )([A-Za-z]{{1,3}})\.\s*(?=(?:{FIT_CODE})|\d)", r"\1 ", body or "")
    out = []
    for sent in FIT_SENT_RX.split(body):
        sent = sent.strip()
        if not sent or not re.search(r"\d", sent):
            continue
        slots = fit_slots(sent)
        if not slots:
            continue
        tokens = len(TOKEN_RX.findall(sent))
        covered = sum(len(TOKEN_RX.findall(c)) for slot in slots for c in slot)
        if tokens - covered > FIT_LOOSE and covered * SHAPE_DENSITY < tokens:
            continue
        out.append((sent, slots, fit_template(sent)))
    return out


def fit_template(sent: str) -> str:
    out, last, n = [], 0, 0
    for m in FIT_CODE_RX.finditer(sent):
        if out and FIT_GLUE_RX.match(sent[last:m.start()]):
            last = m.end()
            continue
        n += 1
        out.append(sent[last:m.start()] + f"<{n}>")
        last = m.end()
    return " ".join(("".join(out) + sent[last:]).lower().split())


def fit_codes(text: str) -> list[str]:
    return [fit_norm(c) for c in FIT_CODE_RX.findall(text or "")]


def fit_kind(code: str) -> tuple[str, int]:
    m = re.match(r"([A-Z]+)", code)
    letters = m.group(1) if m else ""
    return letters, len(code) - len(letters)


def fit_kind_digits(code: str) -> str:
    return code[len(fit_kind(code)[0]):]


def fit_related(a: str, b: str) -> bool:
    ka, kb = fit_kind(a), fit_kind(b)
    return ka[0] == kb[0] or (len(ka[0]), ka[1]) == (len(kb[0]), kb[1])


def fit_same(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    return all(x == y or x == "X" or y == "X" for x, y in zip(a, b))


PAIR_TOKEN = re.compile(r"[a-z]+|[0-9]+(?:\.[0-9]+)?")
PAIR_CUT = re.compile(r"[,;:.!?]")


def pair_tokens(text: str) -> list[tuple[str, int, int, int]]:
    low = text.lower()
    out = []
    for m in PAIR_TOKEN.finditer(low):
        w = m.group(0)
        out.append((Text.stem(w) if w.isalpha() else w, m.start(), m.end(), len(PAIR_CUT.findall(low[:m.start()]))))
    return out


def pair_partner(sentence: str, own: str, vocab: Any, keys: Any) -> str | None:
    o = tuple(w for w, *_ in pair_tokens(own))
    if not o:
        return None
    st = pair_tokens(sentence)
    words = [w for w, *_ in st]
    others = {k for k in keys if k}
    marks: list[tuple[int, int, tuple]] = []
    taken = [False] * len(st)
    for v in sorted({tuple(w for w, *_ in pair_tokens(Text.of(x))) for x in list(vocab) + [own]} - {()}, key=lambda x: -len(x)):
        n = len(v)
        for i in range(len(st) - n + 1):
            if tuple(words[i:i + n]) == v and not any(taken[i:i + n]) and len({st[j][3] for j in range(i, i + n)}) == 1:
                cl = st[i][3]
                after = lambda k: tuple(words[i + n:i + n + len(k)]) == k and all(st[j][3] == cl for j in range(i + n, i + n + len(k)))
                before = lambda k: i >= len(k) and tuple(words[i - len(k):i]) == k and all(st[j][3] == cl for j in range(i - len(k), i))
                if any(after(k) or before(k) for k in others):
                    continue
                marks.append((i, i + n, v))
                for j in range(i, i + n):
                    taken[j] = True
    marks.sort()
    span = lambda a, b: sentence[st[a][1]:st[b - 1][2]]
    hit = [n for n, x in enumerate(marks) if x[2] == o]
    if hit and len(marks) >= 2 and len(marks) % 2 == 0:
        for n in hit:
            if n % 2 == 0 and ((len(marks) == 2 and marks[n + 1][2] != o) or st[marks[n][0]][3] == st[marks[n + 1][0]][3]):
                return span(marks[n + 1][0], marks[n + 1][1])
        return None
    if not hit:
        if len({x[2] for x in marks}) != 1:
            return None
        a = marks[0][0]
        last = a
        while last + 1 < len(st) and st[last + 1][3] == st[a][3]:
            last += 1
        return span(a, last + 1)
    conn = None
    for (a1, b1, v1), (a2, b2, v2) in zip(marks, marks[1:]):
        if st[a1][3] == st[a2][3] and b1 < a2 and o not in (v1, v2):
            conn = tuple(words[b1:a2])
            break
    if not conn:
        return None
    for a, b, v in marks:
        if v != o:
            continue
        end = b + len(conn)
        if tuple(words[b:end]) == conn and end < len(st) and st[end][3] == st[b][3]:
            last = end
            while last + 1 < len(st) and st[last + 1][3] == st[end][3]:
                last += 1
            return span(end, last + 1)
    return None


def pair_runs(seq: tuple, run: tuple) -> list[int]:
    n = len(run)
    return [k for k in range(len(seq) - n + 1) if seq[k:k + n] == run]


def pair_clauses(sentence: str, own: str, known: Any, named: bool, shelf: Any = (), prop: str = "", keys: Any = ()) -> str | None:
    o = tuple(w for w, *_ in pair_tokens(own))
    if not o:
        return None
    mine = {w for w, *_ in pair_tokens(Text.key(prop))} if prop else set()
    cls = []
    for m in re.finditer(r"[^,;:!?]+?(?=[,;:!?]|\.(?!\d)|$)", sentence):
        toks = [(w, m.start() + a, m.start() + b) for w, a, b, _ in pair_tokens(m.group(0))]
        if toks:
            cls.append(toks)
    words = [tuple(w for w, *_ in c) for c in cls]
    if len(words) < 2:
        return None
    oset = set(o) | mine
    best: tuple = ()
    score = (0, 0)
    for c in words:
        for i in range(len(c)):
            for j in range(i + 1, min(len(c), i + 6) + 1):
                run = c[i:j]
                if oset & set(run):
                    continue
                per = [pair_runs(x, run) for x in words]
                if any(len(p) > 1 for p in per):
                    continue
                hits = [(n, p[0]) for n, p in enumerate(per) if p]
                if len(hits) >= 2 and (len(hits), len(run)) > score and all(k > 0 and k + len(run) < len(words[n]) for n, k in hits):
                    best, score = run, (len(hits), len(run))
    if not best:
        return None
    n = len(best)
    toks_of = lambda xs: {tuple(w for w, *_ in pair_tokens(Text.of(v))) for v in xs} - {()}
    known_t = toks_of(known)
    wide = known_t | toks_of(shelf)
    rows = [n_ for n_, x in enumerate(words) if pair_runs(x, best)]
    for n_ in rows:
        k = pair_runs(words[n_], best)[0]
        wide.add(words[n_][k + n:])
    keyset = {tuple(k) for k in keys}

    def span(c: list, a: int, b: int) -> str:
        return sentence[c[a][1]:c[b - 1][2]]

    def fine(part: tuple) -> bool:
        return bool(part) and not set(part) <= mine and part not in keyset
    lefts = []
    sides = [words[row][pair_runs(words[row], best)[0] + n:] for row in rows]
    listed = o in sides
    vals = {w for x in sides for w in x}
    for row in rows:
        k = pair_runs(words[row], best)[0]
        at = pair_runs(words[row][:k], o)
        tail = words[row][at[-1] + len(o):k] if at else ()
        if at and (not tail or (listed and len(tail) <= 2 and not set(tail) & vals)):
            lefts.append((0 if tail else 1, -row, k))
    if lefts:
        _, row, k = max(lefts)
        c, x = cls[-row], words[-row]
        if fine(x[k + n:]) and (named or x[k + n:] in known_t):
            return span(c, k + n, len(x))
        return None
    for row in rows:
        c, x = cls[row], words[row]
        k = pair_runs(x, best)[0]
        left, right = x[:k], x[k + n:]
        if left[-len(o):] == o:
            if fine(right) and (named or right in known_t):
                return span(c, k + n, len(x))
            continue
        if right != o:
            continue
        for s in range(len(left)):
            if left[s:] in wide:
                if fine(left[s:]) and (named or left[s:] in known_t):
                    return span(c, s, k)
                break
    return None


def pair_same(a: Any, b: Any) -> bool:
    return Text.key(Text.of(a)) == Text.key(Text.of(b)) or [w for w, *_ in pair_tokens(Text.of(a))] == [w for w, *_ in pair_tokens(Text.of(b))]


def fit_same_pairs(xs: Any, ys: Any) -> list[tuple[str, str]]:
    return [(x, y) for x in sorted(xs) for y in sorted(ys) if fit_same(x, y)]


def fit_related_any(xs: Any, ys: Any) -> bool:
    return any(fit_related(x, y) for x in xs for y in ys)


class FitInfo:

    def __init__(self, rec: Any, role: str | None = None) -> None:
        self.role = role
        self.own: set[str] = set()
        self.compat: set[str] = set()
        self.exc: set[str] = set()
        self.ext: list[tuple[str, set[str]]] = []
        self.pointers: list[str] = []
        self.resolved: set[str] = set()
        self.weak: set[str] = set()
        self.paged: set[str] = set()
        self.series: set[str] = set()
        self.sown: set[str] = set()
        title, body = ("", rec) if isinstance(rec, str) else (rec[0], rec[1])
        self.has_title = bool(title)
        facts = rec[2] if isinstance(rec, tuple) and len(rec) > 2 else ""
        if isinstance(rec, str) and re.fullmatch(rf"\s*{FIT_CODE}\s*", rec):
            self.add_own(rec)
        for c in FIT_TITLE_RX.findall(title):
            self.add_own(c)
        self.titled = set(self.own) if title else set()
        self.parse(body)
        if facts and not (self.compat or self.exc or self.ext or self.pointers or self.own - self.titled):
            self.parse(facts)

    def add_own(self, raw: str) -> None:
        self.own.add(fit_norm(raw))
        s = fit_series(raw)
        if s:
            self.own.add(s)
            self.sown.add(s)

    def parse(self, body: str) -> None:
        for sent, slots, tpl in fit_forms(body):
            roles = self.shape_roles(slots)
            read = FIT_ROLES.get(tpl) or {}
            self.apply({n: ("not" if read.get(n) == "not" and v in ("fits", "main", "own") else v) for n, v in roles.items()}, slots)

    def shape_roles(self, slots: list[list[str]]) -> dict[int, str]:
        codes = [fit_norm(c) for slot in slots for c in slot]
        wild = [c for c in codes if "X" in fit_kind_digits(c)]
        dependent = bool(self.titled) if (self.has_title or self.role is None) else self.role != "main"
        if wild:
            out = {c for c in codes if c not in wild and any(fit_same(w, c) for w in wild)}
            return {n: ("not" if all(fit_norm(c) in out for c in slot) else "fits" if dependent else "own")
                    for n, slot in enumerate(slots, 1)}
        kinds = [{fit_kind(fit_norm(c)) for c in slot} for slot in slots]
        if len(slots) == 2 and len(slots[0]) == 1 and not kinds[0] & kinds[1]:
            return {n: ("about" if n == 1 else "theirs") for n in range(1, len(slots) + 1)}
        if not dependent:
            return {n: "main" for n in range(1, len(slots) + 1)}
        mine = {fit_kind(c) for c in self.titled}
        roles = {}
        for n, slot in enumerate(slots, 1):
            norm = [fit_norm(c) for c in slot]
            if all(self.is_own(c) for c in norm):
                roles[n] = "none"
            elif all(fit_kind(c) in mine for c in norm):
                roles[n] = "see"
            else:
                roles[n] = "fits"
        if "fits" in roles.values() and list(roles.values()).count("see") == 1:
            roles = {n: ("about" if v == "see" else "theirs" if v == "fits" else v) for n, v in roles.items()}
        return roles

    def apply(self, roles: dict[int, str], slots: list[list[str]]) -> None:
        about = None
        theirs: set[str] = set()
        for n, slot in enumerate(slots, 1):
            role = roles.get(n, "none")
            norm = [fit_norm(c) for c in slot]
            if role == "fits":
                self.compat.update(norm)
            elif role == "not":
                self.exc.update(norm)
            elif role == "own":
                for c in slot:
                    self.add_own(c)
            elif role == "see":
                self.pointers += [c.strip() for c in slot]
            elif role == "about" and about is None:
                about = norm[0]
            elif role == "theirs":
                theirs.update(norm)
            elif role == "main":
                for c, raw in zip(norm, slot):
                    if SERIES_RX.fullmatch(raw.strip()):
                        if not self.is_own(c):
                            self.series.add(c)
                    elif fit_series(raw):
                        self.pointers.append(raw.strip())
                    elif "X" not in fit_kind_digits(c):
                        self.add_own(raw)
        if about is not None and theirs:
            self.ext.append((about, theirs))

    def copy(self) -> "FitInfo":
        c = FitInfo.__new__(FitInfo)
        c.role, c.titled, c.has_title = self.role, set(self.titled), self.has_title
        c.own, c.compat, c.exc, c.weak, c.paged = set(self.own), set(self.compat), set(self.exc), set(self.weak), set(self.paged)
        c.ext, c.pointers, c.resolved = list(self.ext), list(self.pointers), set(self.resolved)
        c.series, c.sown = set(self.series), set(self.sown)
        return c

    def is_own(self, key: str) -> bool:
        return any(fit_same(key, o) for o in self.own)

    def absorb(self, page: "FitInfo") -> None:
        for key, lst in page.ext:
            if self.is_own(key):
                self.compat.update(lst)
                self.paged.update(lst)
        for p in self.pointers:
            if page.is_own(fit_norm(p)):
                self.resolved.add(fit_norm(p))

    def settle(self) -> None:
        for key, lst in self.ext:
            if self.is_own(key):
                self.compat.update(lst)
            else:
                self.weak.update(lst)

    def open_pointers(self) -> list[str]:
        return [p for p in self.pointers if fit_norm(p) not in self.resolved]


PART_CODE_RX = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,6}-?\d[\dxX]{2,5}(?:-\d{2,6})?(?![A-Za-z0-9])")
SPACED_CODE_RX = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,6} \d{3,6}(?![A-Za-z0-9])")


def code_shaped(value: Any) -> bool:
    text = Text.of(value)
    rx = next((x for x in (PART_CODE_RX, SPACED_CODE_RX) if x.search(text)), None)
    if rx is None:
        return False
    rest = rx.sub(" ", text)
    return len([w for w in Text(rest).words if not w.isdigit()]) <= 1


LINE_SEARCH_STEPS = 6
RESCUE_SYS = "You help a shopper search an online store. Reply with JSON only: {\"phrases\": [\"...\"]}."
READ_TRIES = 2
RESCUE_ALTS = 3


def noun_heads(noun: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"\s*(?:&|/|,|\band\b)\s*", Text.of(noun).lower()):
        w = Text(part).words
        if w and w[-1] not in out:
            out.append(w[-1])
    return out


def search_steps(terms: dict, traits: dict | None) -> list[tuple[tuple[str, ...], int]]:
    noun = [w for w in Text(terms.get("noun") or "").words if len(w) > 1]
    heads = terms.get("heads") or noun[-1:]
    head = heads[0] if heads else (noun[-1] if noun else "")
    ints = list(terms.get("ints") or [])
    decs = list(terms.get("decs") or [])
    values = [v for v in terms.get("values") or [] if v]
    revealed = [v for v in terms.get("revealed") or [] if v]
    alts = [v for v in terms.get("alts") or [] if v]
    vwords: list[str] = []
    for v in values:
        vwords += [w for w in Text(v).words if w not in vwords]
    rest = [w for w in noun if w != head]

    def q(*parts: Any) -> tuple[str, ...]:
        out: list[str] = []
        for part in parts:
            for w in (part if isinstance(part, list) else [part]):
                if w and w not in out:
                    out.append(w)
        return tuple(out)
    first = q(head, ints, vwords, rest)
    steps: list[tuple[tuple[str, ...], int]] = [(first, 1)]
    if traits is None:
        return steps
    limit = traits.get("limit")
    short = limit is not None and limit <= 2
    early = 2 if limit is None else (1 if (traits.get("price") or short) else 0)
    if limit:
        early = min(early, limit - 1)
    steps += [(q(head, Text(v).words, [] if short else rest), 1) for v in alts]
    steps += [(first, n) for n in range(2, 2 + early)]
    if short:
        toks = [t for t in dict.fromkeys(ints + decs + vwords + [w for v in revealed + alts for w in Text(v).words]) if t != head]
        steps += [(q(head, t), 1) for t in toks[1:]]
        steps += [(q(h, toks[0] if toks else ""), 1) for h in heads[1:]]
        steps += [(q(head, t), 2) for t in toks[1:]]
        if traits.get("shelf"):
            steps += [(("@shelf",), 1), (("@shelf",), 2)]
    else:
        steps += [(q(head, n, rest), 1) for n in ints]
        if len(ints) + len(values) > 1:
            steps += [(q(head, Text(v).words, rest), 1) for v in values]
        steps += [(q(head, Text(v).words, rest), 1) for v in alts]
        steps += [(q(head, Text(v).words, Text(values[0]).words if values else [], rest), 1) for v in revealed]
        steps += [(q(head, Text(v).words, rest), 1) for v in revealed]
        if traits.get("criteria"):
            steps += [(("@criteria",), 1), (("@halve",), 1), (("@halve",), 1)]
        if traits.get("shelf"):
            steps += [(("@shelf",), 1), (("@shelf",), 2)]
        last = min(limit, 3) if limit else 6
        steps += [(first, n) for n in range(2 + early, last + 1)]
        if traits.get("price"):
            steps += [(q(head, Text(v).words, rest), 2) for v in values] + [(q(head, rest), 1), (q(head, rest), 2)]
    seen: set = set()
    out = []
    for st in steps:
        if st[0] and st not in seen:
            seen.add(st)
            out.append(st)
    return out


def store_traits_from(words: list[str], texts: list[str], price_sorted: bool, limit: int | None, criteria: bool) -> dict:
    stems = [set(Text(t).stems) for t in texts]
    ws = [Text.stem(w) for w in words]

    def has(row: set, w: str) -> bool:
        return w in row or any(Text.alike(x, w) for x in row)
    cover = [[has(row, w) for w in ws] for row in stems]
    all_rows = bool(cover) and all(all(c) for c in cover)
    cap2 = len(ws) >= 3 and bool(cover) and all(any(c[:2]) for c in cover) and not any(any(c[2:]) for c in cover)
    return {"or": bool(cover) and not all_rows, "cap2": cap2, "price": price_sorted, "limit": limit, "criteria": criteria}



def same_word(a: str, b: str) -> bool:
    if a == b or Text.stem(a) == Text.stem(b) or Text.alike(a, b) or Text.alike(b, a):
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 3 and long_.startswith(short) and len(long_) - len(short) <= 2


def sibling_name(noun: str, dept: str) -> bool:
    wa = [w for w in Text(noun).words if len(w) > 3]
    wb = [w for w in Text(dept).words if len(w) > 3]
    left_a = [w for w in wa if not any(same_word(w, x) for x in wb)]
    left_b = [w for w in wb if not any(same_word(w, x) for x in wa)]
    return bool(left_a) and bool(left_b)


def wider_name(noun: str, dept: str) -> bool:
    wa, wb = Text(noun).words, Text(dept).words
    return bool(wa) and all(any(same_word(w, x) for x in wb) for w in wa) and any(len(x) > 3 and not any(same_word(x, w) for w in wa) for x in wb)


def same_name(noun: str, dept: str) -> bool:
    wa, wb = Text(noun).words, Text(dept).words
    if not wa or not wb:
        return False
    left_a = [w for w in wa if not any(same_word(w, x) for x in wb)]
    left_b = [w for w in wb if not any(same_word(w, x) for x in wa)]
    matched = len(wa) - len(left_a)
    return matched >= 1 and matched * 2 > len(wa) and all(len(w) <= 3 for w in left_a + left_b) and len(left_a + left_b) <= 1 + (len(wa) >= 3)


def fit_info(rec: Any, role: str | None = None) -> FitInfo:
    return rec.copy() if isinstance(rec, FitInfo) else FitInfo(rec, role)


def fit_pointer_pages(record: Any) -> list[str]:
    seen, out = set(), []
    for p in fit_info(record).pointers:
        if fit_norm(p) not in seen:
            seen.add(fit_norm(p))
            out.append(p)
    return out


def fit_judge(main_record: Any, accessory_record: Any, extra_pages: Any = ()) -> tuple[str, str]:
    M, A = fit_info(main_record, "main"), fit_info(accessory_record, "dep")
    pages = [fit_info(p) for p in extra_pages]
    for side in (M, A):
        side.settle()
        for pg in pages:
            side.absorb(pg)
    exc = fit_same_pairs(A.exc, M.own) + fit_same_pairs(M.exc, A.own)
    if exc:
        return "no", "excluded " + ",".join(f"{x}~{y}" for x, y in exc[:3])
    hit = fit_same_pairs(A.compat - A.paged, M.own) + fit_same_pairs(M.compat - M.paged, A.own)
    if hit:
        return "yes", "match " + ",".join(f"{x}~{y}" for x, y in hit[:3])
    hit = fit_same_pairs(A.paged, M.own) + fit_same_pairs(M.paged, A.own)
    if hit:
        return "yes", "match via pointer page " + ",".join(f"{x}~{y}" for x, y in hit[:3])
    ser = sorted((M.series & A.sown) | (A.series & M.sown))
    if ser:
        return "yes", "series " + ",".join(ser[:3])
    if {fit_kind(c)[0] for c in M.series} & {fit_kind(c)[0] for c in A.sown} or {fit_kind(c)[0] for c in A.series} & {fit_kind(c)[0] for c in M.sown}:
        return "no", "series differs"
    notes = []
    weak = (fit_same_pairs(A.weak, M.own) if not A.titled else []) + (fit_same_pairs(M.weak, A.own) if not M.titled else [])
    if weak:
        notes.append("own page line for another part names " + ",".join(y for x, y in weak[:3]))
    opened = A.open_pointers() + M.open_pointers()
    if opened:
        notes.append("open " + ",".join(opened))
    if notes:
        return "unknown", "; ".join(notes)
    if fit_related_any(A.compat, M.own):
        return "no", "not listed by accessory"
    if fit_related_any(M.compat, A.own):
        return "no", "not listed by main"
    return "unknown", "no codes"


def fit_best(edges: list[tuple[int, int]], pools: dict[int, list[dict]], cur: dict[int, Any], tier: Callable, price: Callable,
             qty: dict[int, int], unsure: Callable, classify: Callable, others: float, budget: float | None, cheapest: bool) -> tuple | None:
    hubs = sorted({b for b, _ in edges})
    accs = sorted({a for _, a in edges} - set(hubs))
    ups = {a: [b for b, x in edges if x == a] for a in accs}
    inner = [(b, a) for b, a in edges if a in hubs]

    def vec(ts: list[int], us: list[int]) -> tuple[int, int, int, int, int]:
        return (sum(t >= 9 for t in ts), sum(u - 1 for u in us if u >= 2), sum(t == 2 for t in ts), sum(u == 1 for u in us), sum(t == 1 for t in ts))
    reps: dict[int, list[dict]] = {}
    for b in hubs:
        groups: dict[Any, list[dict]] = {}
        for it in pools.get(b) or []:
            groups.setdefault(classify(it, b), []).append(it)
        keep = []
        for its in groups.values():
            level: dict[int, dict] = {}
            for it in sorted(its, key=lambda it: (price(it), it["id"])):
                level.setdefault(unsure(it, b), it)
            keep += list(level.values())
        if cur.get(b) is not None:
            keep.append(cur[b])
        reps[b] = sorted({it["id"]: it for it in keep}.values(), key=lambda it: (unsure(it, b), price(it), it["id"]))
        if not reps[b]:
            return None
    best = None
    for n, hc in enumerate(itertools.product(*(reps[b] for b in hubs))):
        if n >= FIT_HUB_COMBOS:
            break
        chosen = dict(zip(hubs, hc))
        base = vec([tier(chosen[b], chosen[a]) for b, a in inner], [unsure(chosen[b], b) for b in hubs])
        cost = sum(price(chosen[b]) * qty[b] for b in hubs)
        fronts = []
        for a in accs:
            opts = []
            for x in pools.get(a) or []:
                opts.append((vec([tier(chosen[b], x) for b in ups[a]], [unsure(x, a)]), price(x) * qty[a], cur.get(a) is not x, x["id"], x))
            opts.sort(key=lambda o: o[:4])
            front: list[tuple] = []
            low = None
            for o in opts:
                if low is None or o[1] < low - 1e-9:
                    front.append(o)
                    low = o[1]
            front = front[:FIT_FRONT]
            mine = [o for o in opts if not o[2]]
            if mine and mine[0] not in front:
                front.append(mine[0])
            fronts.append(front)
        if any(not f for f in fronts):
            continue
        for combo in itertools.product(*fronts):
            total = others + cost + sum(o[1] for o in combo)
            if budget and total > budget + 1e-9:
                continue
            v = tuple(map(sum, zip(base, *(o[0] for o in combo))))
            full = dict(chosen)
            full.update({a: o[4] for a, o in zip(accs, combo)})
            changes = sum(1 for ai, it in full.items() if cur.get(ai) is None or cur[ai]["id"] != it["id"])
            ids = tuple(full[ai]["id"] for ai in sorted(full))
            key = v + ((total, changes) if cheapest else (changes, total)) + (ids,)
            if best is None or key < best[0]:
                best = (key, full, total)
    return best


class Query:
    __slots__ = ("text", "total", "cursor", "pages", "ids", "used", "by_price", "fetched", "first")

    def __init__(self, text: str) -> None:
        self.text = text
        self.total: int | None = None
        self.cursor: Any = None
        self.pages = 0
        self.ids: list[str] = []
        self.used: list = []
        self.by_price: bool | None = None
        self.fetched = 0
        self.first = 0

    def export(self) -> dict:
        out: dict[str, Any] = {"total": self.total, "cursor": self.cursor, "pages": self.pages}
        if self.used:
            out["used"] = self.used
        out["n"] = len(self.ids)
        return out


class Finder:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.queries: dict[str, Query] = {}
        self.first_totals: list[tuple[frozenset, int]] = []
        self.verified = False
        self.traits: dict | None = None
        self.shape = {"and": 0, "or": 0}
        self.live_memo: dict[tuple, list[dict]] = {}
        self.pair_and: bool | None = None
        self.stem_memo: dict[str, frozenset] = {}
        self.decimal_dead = False

    def state(self, text: str) -> Query | None:
        return self.queries.get(text.lower())

    def open_queries(self) -> list[Query]:
        return [q for q in self.queries.values() if q.cursor and self.lead(q) is q and not self.stale(q)]

    def id_ordered(self) -> bool:
        votes = [q.ids[:q.first] == sorted(q.ids[:q.first]) for q in self.queries.values() if q.first >= 3]
        return bool(votes) and all(votes)

    def stale(self, q: Query) -> bool:
        if not isinstance(q.total, int) or q.total < 1 or q.first < 1 or not self.id_ordered() or self.capped_total(q.total):
            return False
        mine = self.stem_set(q.text)
        n = q.first
        for o in self.queries.values():
            if (o is q or not isinstance(o.total, int) or o.total < 1 or o.cursor not in (None, "", False)
                    or o.pages < 2 or o.fetched >= o.total):
                continue
            theirs = self.stem_set(o.text)
            window = min(q.total, o.fetched)
            if o.total <= q.total and mine < theirs and o.ids[:n] == q.ids[:n]:
                if window / q.total <= o.fetched / o.total and (window - q.fetched) * (1 - o.total / q.total) < 1:
                    return True
            elif theirs < mine and q.total <= o.total and self.match() != "or":
                pages = -(-max(0, window - q.fetched) // max(1, q.first))
                new = window - max(q.fetched, q.total * o.fetched / o.total)
                if pages and new < pages:
                    return True
        return False

    def stem_set(self, text: str) -> frozenset:
        got = self.stem_memo.get(text)
        if got is None:
            got = self.stem_memo[text] = frozenset(Text(text).stems)
        return got

    def capped_total(self, total: int) -> bool:
        tots = [q.total for q in self.queries.values() if isinstance(q.total, int)]
        if total >= max(tots, default=0):
            return True
        same = [q for q in self.queries.values() if q.total == total and q.pages >= 1 and q.first > 0]
        n = min((q.first for q in same), default=0)
        return len({tuple(q.ids[:n]) for q in same}) > 1

    def same_rows(self, a: Query, b: Query) -> bool:
        if a is b or a.pages < 1 or b.pages < 1 or not isinstance(a.total, int) or a.total != b.total or a.total < 1:
            return False
        n = min(a.first, b.first)
        if n < 1 or a.ids[:n] != b.ids[:n]:
            return False
        sa, sb = self.stem_set(a.text), self.stem_set(b.text)
        return (sa <= sb or sb <= sa) and not self.capped_total(a.total)

    def lead(self, q: Query) -> Query:
        best = q
        for o in self.queries.values():
            if o.fetched > best.fetched and self.same_rows(q, o):
                best = o
        return best

    def class_text(self, q: Query) -> str:
        return " ".join([q.text] + [o.text for o in self.queries.values() if o is not q and self.same_rows(q, o)])

    def sync(self, lead: Query) -> None:
        done = lead.cursor in (None, "", False)
        for m in self.queries.values():
            if m is lead or m.fetched >= lead.fetched or not self.same_rows(m, lead):
                continue
            m.ids += [i for i in lead.ids if i not in m.ids]
            if done:
                m.cursor, m.fetched, m.pages, m.used = lead.cursor, lead.fetched, lead.pages, list(lead.used)

    def record(self, text: str, ids: list[str], total: int | None, cursor: Any) -> None:
        q = self.queries.setdefault(text.lower(), Query(text.lower()))
        if total is not None:
            q.total = total
        q.cursor = cursor
        q.pages += 1
        items = self.r.cat.items
        for n, i in enumerate(ids):
            it = items.get(i)
            if it is not None:
                it["rank"] = min(it.get("rank", 1e9), len(q.ids) + n)
        q.ids += [i for i in ids if i not in q.ids]
        q.fetched += len(ids)
        if q.pages == 1:
            q.first = len(ids)
        if q.pages == 1 and isinstance(total, int):
            self.first_totals.append((frozenset(Text(text).stems), total))
            self.vote(q)
        prices = [self.r.cat.price(items[i]) for i in ids if i in items and self.r.cat.price(items[i]) is not None]
        if q.pages == 1 and len(prices) >= 3:
            q.by_price = all(a <= b for a, b in zip(prices, prices[1:]))
        if q.pages == 1 and not ids and not total and any(self.decimal(w) for w in Text(text).words):
            self.decimal_dead = True
        self.sync(self.lead(q))

    def unsearchable(self, value: Any) -> bool:
        return self.decimal_dead and any(self.decimal(w) for w in Text(Text.of(value)).words)

    def price_ordered(self) -> bool:
        votes = [q.by_price for q in self.queries.values() if q.by_price is not None]
        return bool(votes) and sum(votes) * 2 > len(votes)

    @staticmethod
    def decimal(w: str) -> bool:
        return "." in w and w.replace(".", "", 1).isdigit()

    def value_tokens(self, value: Any) -> list[str]:
        return [w for w in dict.fromkeys(Text(value).words) if not self.decimal(w)]

    def vote(self, q: Query) -> None:
        mine = frozenset(Text(q.text).stems)
        if not 1 <= len(mine) <= 2:
            return
        for o in list(self.queries.values()):
            if o is q or o.pages < 1 or not isinstance(o.total, int):
                continue
            other = frozenset(Text(o.text).stems)
            if not 1 <= len(other) <= 2 or not (other < mine or mine < other):
                continue
            sub, sup = (o, q) if other < mine else (q, o)
            if sup.total < sub.total:
                self.shape["and"] += 1
            elif sup.total > sub.total:
                self.shape["or"] += 1

    def match(self) -> str | None:
        a, o = self.shape["and"], self.shape["or"]
        return "and" if a > o else ("or" if o > a else None)

    def sas(self) -> bool:
        return self.match() == "and"

    def complete(self, q: Query | None) -> bool:
        return (q is not None and isinstance(q.total, int) and q.pages >= 1 and q.cursor in (None, "", False)
                and q.fetched >= q.total)

    def rows_per_page(self) -> int:
        return max([q.first for q in self.queries.values()] + [1])

    def reach(self) -> int:
        rows = self.rows_per_page()
        cut = [q.fetched for q in self.queries.values()
               if q.pages > 1 and q.cursor in (None, "", False) and isinstance(q.total, int) and q.fetched < q.total]
        if cut:
            return max(rows, min(cut))
        lim = self.page_limit(rows)
        return rows * (lim if lim else 8)

    def df_table(self) -> collections.Counter:
        cat = self.r.cat
        c: collections.Counter = collections.Counter()
        for it in cat.items.values():
            c.update(set(Text(" ".join([cat.title(it), cat.category(it), cat.seller(it), cat.maker(it)])).stems))
        return c

    @staticmethod
    def sig(req: dict) -> str:
        vals = req["value"] if isinstance(req["value"], list) else [req["value"]]
        return Text.json([Text.key(req.get("prop")), sorted(Text.of(v).lower() for v in vals)])

    def live_sets(self, ai: int) -> list[dict]:
        r = self.r
        if not self.sas() or not 0 <= ai < len(r.articles):
            return []
        art = r.articles[ai]
        sets = art.get("sas_sets") or []
        if not sets:
            return []
        key = (ai, r.steps, r.judge.asked, len(r.ledger), sum(1 for x in r.ledger if x["active"]), art.get("cat"))
        if key in self.live_memo:
            return self.live_memo[key]
        sigs = {self.sig(x) for x in r.reqs(ai) if x["op"] in ("is", "eq")}
        out = []
        for s in sets:
            qs = [self.state(t) for t in s["qs"]]
            if any(not self.complete(q) for q in qs) or not set(s["need"]) <= sigs:
                continue
            if s["cat"] and s["cat"] != art.get("cat"):
                continue
            ids = {i for q in qs for i in q.ids}
            if any(it["id"] not in ids and r.judge.line_level(it, ai) == "exact" for it in r.owners(ai)):
                continue
            out.append({**s, "ids": ids})
        self.live_memo = {k: v for k, v in self.live_memo.items() if k[1] == r.steps}
        self.live_memo[key] = out
        return out

    def lacks(self, it: dict, ai: int) -> bool:
        if it.get("read") or not self.sas():
            return False
        return any(it["id"] not in s["ids"] for s in self.live_sets(ai))

    def run(self, text: str, extra: dict | None = None) -> tuple[list[str], int | None, Any]:
        r = self.r
        m = r.tm.get("search")
        if not m or not m.get("query"):
            return [], None, None
        twin = None if extra else self.twin(text)
        if twin is not None:
            self.queries[text.lower()] = twin
            return list(twin.ids), twin.total, twin.cursor
        args = {**(m.get("args") or {}), m["query"]: self.fit_length(m["name"], m["query"], text), **(extra or {})}
        obs = r.env.send([(m["name"], args)])
        ids, total, cursor = r.cat.rows_of(obs[0] if obs else None)
        if obs and not ids and not extra and not self.verified and not r.cat.items and self.plain(text):
            ids, total, cursor = self.verify(text)
        if obs:
            self.record(text, ids, total, cursor)
        return ids, total, cursor

    def twin(self, text: str) -> Query | None:
        key = sorted(Text(text).words)
        if not key or len(key) > 6:
            return None
        return next((q for t, q in self.queries.items() if q.pages >= 1 and isinstance(q.total, int) and sorted(Text(t).words) == key), None)

    def plain(self, text: str) -> bool:
        stems = set(Text(text).stems)
        return not self.r.articles or any(stems <= set(Text(a["noun"]).stems) for a in self.r.articles)

    def fit_length(self, tool: str, arg: str, text: str) -> str:
        spec = ((self.r.schemas.get(tool) or {}).get("properties") or {}).get(arg) or {}
        top = spec.get("maxLength")
        if not isinstance(top, int) or len(text) <= top:
            return text
        words = text.split()
        while len(words) > 1 and len(" ".join(words)) > top:
            words.pop()
        return " ".join(words)

    def page_limit(self, page_rows: int) -> int | None:
        r = self.r
        m = r.tm.get("search") or {}
        pg = m.get("page")
        spec = ((r.schemas.get(m.get("name") or "") or {}).get("properties") or {}).get(pg or "") or {}
        if not pg or spec.get("type") != "integer":
            return None
        hi, lo = spec.get("maximum"), spec.get("minimum")
        if not isinstance(hi, (int, float)):
            return None
        if lo == 0:
            return int(hi) // max(1, page_rows) + 1
        return int(hi)

    def learn_traits(self, words: list[str], ids: list[str]) -> None:
        r = self.r
        if self.traits is not None and (self.traits.get("seen") or not ids):
            return
        cat = r.cat
        texts = [" ".join([cat.title(cat.items[i]), cat.category(cat.items[i])] + [v for v in cat.items[i]["row"].values() if isinstance(v, str)])
                 for i in ids if i in cat.items]
        f = r.tm.get("filter")
        criteria = bool(f and f.get("category") and f.get("max_price"))
        shelf = bool(f and f.get("category") and not f.get("max_price"))
        prices = [cat.price(cat.items[i]) for i in ids if i in cat.items and cat.price(cat.items[i]) is not None]
        by_price = len(prices) >= 3 and all(a <= b for a, b in zip(prices, prices[1:]))
        self.traits = store_traits_from(words, texts, by_price, self.page_limit(len(ids) or 1) if ids else self.page_limit(10), criteria)
        self.traits["shelf"] = shelf
        self.traits["seen"] = bool(ids)
        r.notes.append(f"store traits: {self.traits}")

    def verify(self, text: str) -> tuple[list[str], int | None, Any]:
        r = self.r
        m = r.tm["search"]
        self.verified = True
        props = (r.schemas.get(m["name"]) or {}).get("properties") or {}
        options = [o for o in (m.get("options") or []) if o[0] in props][:5]
        for q, v in options:
            if r.steps_left() <= r.plan.finish_need(minimal=True) + 2 or r.t() > GUARD_AT - 60:
                break
            args = {**(m.get("args") or {}), q: self.fit_length(m["name"], q, text)}
            if m.get("mode") and v is not None:
                args[m["mode"]] = v
            if q != m["query"]:
                args.pop(m["query"], None)
            obs = r.env.send([(m["name"], args)])
            ids, total, cursor = r.cat.rows_of(obs[0] if obs else None)
            if ids:
                old_q, old_v = m["query"], (m.get("args") or {}).get(m.get("mode"))
                m["query"] = q
                if m.get("mode") and v is not None:
                    m["args"][m["mode"]] = v
                    f = r.tm.get("filter")
                    if f and f.get("name") == m["name"] and (f.get("args") or {}).get(m["mode"]) == v:
                        f["args"][m["mode"]] = old_v
                r.notes.append(f"search verified: query arg {old_q!r} -> {q!r}, mode {old_v!r} -> {v!r}")
                return ids, total, cursor
        return self.reelect(text)

    def reelect(self, text: str) -> tuple[list[str], int | None, Any]:
        r = self.r
        m = r.tm["search"]
        for alt in list(r.store.search_alts):
            if r.steps_left() <= r.plan.finish_need(minimal=True) + 2 or r.t() > GUARD_AT - 60:
                break
            m2 = r.store.search_map(alt, r.store.shape_of(alt))
            if not m2.get("query"):
                continue
            obs = r.env.send([(alt, {**(m2.get("args") or {}), m2["query"]: text})])
            if not obs or not Catalog.row_lists(Store.unwrap(obs[0])[0]):
                continue
            old = m["name"]
            r.store.search_alts = [a for a in r.store.search_alts if a != alt] + [old]
            m.clear()
            m.update(m2)
            r.store.roles[alt] = "search"
            if r.store.roles.get(old) == "search":
                r.store.roles.pop(old)
                if not r.tm.get("detail") and r.store.shape_of(old)["strs"]:
                    r.tm["detail"] = {"name": old, "id": r.store.id_arg(old)}
                    r.store.roles[old] = "detail"
            r.notes.append(f"search re-elected: {old} returned no rows, {alt} did")
            return r.cat.rows_of(obs[0])
        return [], None, None

    def strict(self) -> bool:
        tots = [t for q, t in dict(self.first_totals).items() if len(q) >= 3]
        if len(tots) < 3:
            return True
        top, n = collections.Counter(tots).most_common(1)[0]
        flat = top >= 300 and n >= 3 and n >= 0.6 * len(tots)
        return not flat

    def page_on(self, text: str, pages: int) -> list[str]:
        r = self.r
        pg = r.store.page_arg("search")
        q = self.state(text)
        if q is not None:
            q = self.lead(q)
        out: list[str] = []
        for _ in range(pages):
            nxt = r.store.next_cursor("search", q) if (pg and q) else None
            if nxt is None or nxt in q.used or r.plan.room(r.phase) <= 0:
                break
            q.used.append(nxt)
            ids, _, _ = self.run(text if q.text == text.lower() else q.text, {pg: nxt})
            out += ids
            if not ids:
                break
        return out

    def value_groups(self, ai: int) -> list[list[list[str]]]:
        groups = []
        for req in self.r.reqs(ai):
            if req["op"] not in ("is", "eq"):
                continue
            vals = req["value"] if isinstance(req["value"], list) else [req["value"]]
            g = [Text(self.r.hunt.written(ai, req, v)).stems for v in vals if isinstance(v, str)]
            g = [x for x in g if x and not all(w.replace(".", "").isdigit() for w in x)]
            if g:
                groups.append(g)
        return groups

    def value_hits(self, it: dict, groups: list[list[list[str]]]) -> int:
        title = Text(self.r.cat.title(it)).stems
        asked = [set(Text(q).stems) for q in it.get("queries") or []] if self.strict() else []

        def stated(v: list[str]) -> bool:
            if Text.run_in(v, title):
                return True
            need = set(v)
            return any(need <= q for q in asked)
        return sum(1 for g in groups if any(stated(v) for v in g))

    @staticmethod
    def carries(phrase: str, groups: list[list[list[str]]]) -> int:
        have = set(Text(phrase).stems)
        hits = 0
        for g in groups:
            if any(set(v) <= have for v in g):
                hits += 1
        return hits

    def combined_phrase(self, ai: int) -> str | None:
        groups = [g for g in self.value_groups(ai) if not any(code_shaped(" ".join(v)) for v in g)]
        noun = Text(self.r.articles[ai]["noun"]).stems[-1:]
        if len(groups) < 2 or not noun:
            return None
        shortest = [min(g, key=len) for g in groups[:2]]
        if max(len(v) for v in shortest) > 2:
            return None
        return " ".join(shortest[0] + shortest[1] + noun)

    def order_key(self, groups: list[list[list[str]]]) -> Callable[[dict], tuple]:
        strict = self.strict()
        cat = self.r.cat
        by_price = self.r.cheapest or strict

        def key(it: dict) -> tuple:
            p = cat.price(it)
            p = 1e18 if p is None else p
            title = set(Text(cat.title(it)).stems)
            figures = sum(1 for g in groups if any(len(n := [x for x in v if Text(x).numbers]) == 1 and len(v) > 1 and n[0] in title for v in g))
            hits = (-self.value_hits(it, groups), -figures)
            return (hits, p) if by_price else (hits, it.get("rank", 1e9), p)
        return key

    def export(self) -> dict:
        return {text: q.export() for text, q in self.queries.items()}


class Hunt:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.written_memo: dict[tuple, tuple] = {}
        self.ranked: dict[tuple, dict[str, tuple[int, int]]] = {}
        self.rank_tried: set[tuple] = set()
        self.rank_calls = 0
        self.open_log: list[str] = []

    def rank_sig(self, ai: int) -> tuple:
        r = self.r
        return (r.judge.line_key(r.reqs(ai)), Text.of(r.spec.get("priority")), Text.of(r.spec.get("priority_detail")))

    def rank_allowed(self) -> bool:
        r = self.r
        return (not r.exhausted and r.finder.sas() and self.rank_calls < OPEN_TASK_CAP and r.t() < GUARD_AT - 40
                and r.spent + OPEN_UNIT_USD < OPEN_STOP_USD)

    def rank_rule(self) -> str:
        r = self.r
        detail = Text.of(r.spec.get("priority_detail")).strip()
        if r.cheapest:
            return "the cheapest listing that meets every requirement"
        if Text.of(r.spec.get("priority")) == "best":
            return f"the best listing by {detail or 'the measure the request names'} among those that meet every requirement"
        return "a listing that meets every requirement"

    def rank_row(self, n: int, it: dict) -> str:
        cat = self.r.cat
        st = cat.store
        skip = {st.id, st.title, st.price, st.stock, st.category, st.attrs, st.rows, st.total, st.cursor}
        parts = [f"#{n}", f"title: {' '.join(cat.slot_text(it, 'title').split())[:240]}", f"price: {cat.price(it)}"]
        if cat.category(it):
            parts.append(f"shelf: {cat.category(it)}")
        stock = cat.in_stock(it)
        if stock is not None:
            parts.append("in stock" if stock else "out of stock")
        texts: list[str] = []
        for k, v in list(it["row"].items()) + list((it.get("blurb") or {}).items()):
            text = " ".join(Text.of(v).split())
            if k in skip or not text:
                continue
            if len(text.split()) < 6 and k not in (st.notes, st.desc):
                parts.append(f"{k}: {text[:80]}")
            elif not any(text in t for t in texts):
                texts = [t for t in texts if t not in text] + [text]
        if it["facts"]:
            parts.append("facts: " + "; ".join(f"{k}={Text.of(v)[:80]}" for k, v in it["facts"].items())[:400])
        if texts:
            parts.append("text: " + " / ".join(texts)[:OPEN_TEXT])
        return " | ".join(parts)

    def rank_prompt(self, ai: int, rows: list[dict]) -> str:
        r = self.r
        art = r.articles[ai]
        reqs = r.reqs(ai)
        lines = [f"PRODUCT LINE: {art['noun']} (quantity {art['qty']})"]
        if art.get("cat"):
            lines.append(f"LINE SHELF: {art['cat']}")
        lines.append("REQUIREMENTS:")
        lines += [f"{n + 1}. {Judge.req_text(q)}" for n, q in enumerate(reqs)] or ["none known yet"]
        limits = []
        if art.get("unit_cap"):
            limits.append(f"each unit at most {art['unit_cap'] * r.money_scale:g}")
        if r.budget:
            limits.append(f"the whole basket at most {r.budget}")
        if limits:
            lines.append("PRICE LIMITS: " + "; ".join(limits) + " (every row below already fits them)")
        lines.append(f"THE SHOPPER BUYS: {self.rank_rule()}")
        if r.request:
            lines += ["", "SHOPPER'S REQUEST (context; it may also cover other product lines):", r.request[:OPEN_REQUEST]]
        lines += ["", f"ROWS ({len(rows)}):"] + [self.rank_row(n + 1, it) for n, it in enumerate(rows)]
        return "\n".join(lines)

    @staticmethod
    def rank_parse(got: Any, rows: list[dict]) -> dict[str, tuple[int, int]]:
        if not isinstance(got, dict):
            return {}
        ids = {it["id"]: it["id"] for it in rows}
        out: dict[str, tuple[int, int]] = {}
        pos = 0
        for lv, name in enumerate(OPEN_LEVELS):
            vals = got.get(name)
            if not isinstance(vals, list):
                continue
            for v in vals:
                rid = None
                if isinstance(v, str) and v.strip() in ids:
                    rid = v.strip()
                else:
                    n = v.strip().lstrip("#") if isinstance(v, str) else v
                    if isinstance(n, str) and n.isdigit():
                        n = int(n)
                    if isinstance(n, float) and n.is_integer():
                        n = int(n)
                    if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(rows):
                        rid = rows[n - 1]["id"]
                if rid is None or rid in out:
                    continue
                out[rid] = (lv, pos)
                pos += 1
        return out

    def rank_job(self, ai: int, pool: list[dict], k: int) -> tuple | None:
        r = self.r
        if k <= 0 or len(pool) <= k or not self.rank_allowed():
            return None
        if not r.reqs(ai) and Text.of(r.spec.get("priority")) != "best":
            return None
        sig = self.rank_sig(ai)
        have = self.ranked.get((ai, sig)) or {}
        fresh = [it for it in pool if it["id"] not in have][:OPEN_ROWS]
        if len(fresh) < 2:
            return None
        mark = (ai, sig, frozenset(it["id"] for it in fresh))
        if mark in self.rank_tried:
            return None
        self.rank_tried.add(mark)
        return ai, sig, fresh, self.rank_prompt(ai, fresh)

    def rank_run(self, jobs: list[tuple | None]) -> None:
        r = self.r
        work = [j for j in jobs if j][:max(0, OPEN_TASK_CAP - self.rank_calls)]
        if not work:
            return
        self.rank_calls += len(work)
        t0, s0 = r.t(), r.spent
        due = min(GUARD_AT - 30, t0 + OPEN_WAIT)
        got = r.model_many([("pick", OPEN_SYS, j[3], {"max_tokens": OPEN_TOKENS, "deadline": due, "chain": OPEN_CHAIN}) for j in work])
        got = list(got) + [None] * (len(work) - len(got))
        for (ai, sig, fresh, _), g in zip(work, got):
            marks = self.rank_parse(g, fresh)
            if not marks:
                self.open_log.append(f"a{ai} {len(fresh)} rows: no answer")
                continue
            cache = self.ranked.setdefault((ai, sig), {})
            for n, it in enumerate(fresh):
                cache[it["id"]] = marks.get(it["id"], (1, 10 ** 6 + n))
            tally = collections.Counter(lv for lv, _ in marks.values())
            self.open_log.append(f"a{ai} {len(fresh)} rows: " + "/".join(str(tally[n]) for n in range(len(OPEN_LEVELS)))
                                 + f" at {t0:.0f}s")
        self.open_log.append(f"{len(work)} calls {r.t() - t0:.1f}s ${r.spent - s0:.4f}")

    def model_order(self, ai: int, pool: list[dict], k: int, base: Callable[[dict], Any] | None = None, force: bool = False) -> list[dict]:
        r = self.r
        pool = sorted(pool, key=base or r.finder.order_key(r.finder.value_groups(ai)))
        room = r.plan.slack() if force else r.plan.room(r.phase)
        k = min(k, room * (r.store.compare_max() if r.store.facts_role() == "compare" else 1))
        if k <= 0 or len(pool) <= max(1, k):
            return pool
        self.rank_run([self.rank_job(ai, pool, k)])
        marks = self.ranked.get((ai, self.rank_sig(ai))) or {}
        if not marks:
            return pool
        old = {it["id"]: n for n, it in enumerate(pool)}

        def key(it: dict) -> tuple:
            lv, pos = marks.get(it["id"], (1, 2 * 10 ** 6))
            if r.cheapest:
                p = r.cat.price(it)
                return (lv, 1e18 if p is None else p, old[it["id"]])
            return (lv, pos, old[it["id"]])
        return sorted(pool, key=key)

    def prime(self, wants: list[tuple[int, int]]) -> None:
        r = self.r
        jobs = []
        for ai, k in wants:
            pool = sorted(self.unread_pool(ai, r.articles[ai].get("ids") or []), key=r.finder.order_key(r.finder.value_groups(ai)))
            jobs.append(self.rank_job(ai, pool, k))
        self.rank_run(jobs)

    def read_calls(self, ids: list[str], what: str) -> list[tuple[str, dict]]:
        tm = self.r.tm
        cmp_, det, spec = tm.get("compare"), tm.get("detail"), tm.get("specs")

        def per_id(m: dict) -> list[tuple[str, dict]]:
            return [(m["name"], {**(m.get("args") or {}), m["id"]: i}) for i in ids]

        calls: list[tuple[str, dict]] = []
        if spec and spec.get("id") and what in ("facts", "all"):
            calls += per_id(spec)
        if calls and what not in ("state", "all"):
            return calls
        if det and det.get("id") and what in ("state", "all"):
            calls += per_id(det)
        elif cmp_ and cmp_.get("ids") and self.r.store.yields_facts(cmp_["name"]) and not (spec and what == "all"):
            k = self.r.store.compare_max()
            for i in range(0, len(ids), k):
                calls.append((cmp_["name"], {**(cmp_.get("args") or {}), cmp_["ids"]: ids[i:i + k]}))
        elif det and det.get("id"):
            calls += per_id(det)
        return calls

    def shrink_compare(self, ids: list[str], what: str, force: bool, calls: list, obs: list) -> None:
        r = self.r
        cmp_ = r.tm.get("compare")
        if not cmp_:
            return
        key = cmp_["ids"]
        batched = any(isinstance(a.get(key), list) and len(a[key]) > 1 for n, a in calls if n == cmp_["name"])
        if not batched:
            return
        empty = any(isinstance(a.get(key), list) and len(a[key]) > 1 and not Catalog.row_lists(r.store.unwrap(o)[0])
                    for (n, a), o in zip(calls, obs) if n == cmp_["name"])
        if not empty:
            return
        was = r.store.compare_max()
        r.store.cmp_cap = max(1, min(4, was // 2))
        left = [i for i in ids if not r.cat.items.get(i, {}).get("read")]
        if left and not force and r.store.cmp_cap < was:
            self.open(left, what)

    def open(self, ids: list[str], what: str = "facts", force: bool = False, room: int | None = None) -> None:
        r = self.r
        items = r.cat.items
        ids = [i for i in dict.fromkeys(ids) if i and (force or what != "facts" or not items.get(i, {}).get("read"))]
        if what == "facts":
            ids = [i for i in ids if items.get(i, {}).get("read") or items.get(i, {}).get("tries", 0) < READ_TRIES]
        if not ids:
            return
        calls = self.read_calls(ids, what)
        if not calls:
            return
        room = room if room is not None else (r.plan.slack() if force else r.plan.room(r.phase))
        obs = r.env.send(calls[:room])
        self.count_tries(ids, what, calls[:room])
        self.shrink_compare(ids, what, force, calls, obs)

    def count_tries(self, ids: list[str], what: str, sent: list[tuple[str, dict]]) -> None:
        if what != "facts" or not sent:
            return
        items = self.r.cat.items
        blob = Text.json([a for _, a in sent], 20000)
        for i in ids:
            it = items.get(i)
            if it is not None and not it.get("read") and i in blob:
                it["tries"] = it.get("tries", 0) + 1

    def unread_pool(self, ai: int, ids: list[str], gate: bool = True) -> list[dict]:
        r = self.r
        items = r.cat.items
        coded = r.judge.code_reqs(ai) if 0 <= ai < len(r.articles) else []
        return [items[i] for i in ids if i in items and not items[i]["read"] and r.ok_basic(items[i], ai)
                and not (gate and r.finder.lacks(items[i], ai))
                and not any(r.cat.code_verdict(items[i], q) == "no" for q in coded)]

    def add_ids(self, art: dict, ids: list[str]) -> None:
        have = art.setdefault("ids", [])
        have += [i for i in ids if i not in have]

    def exact_department(self, ai: int) -> str | None:
        r = self.r
        noun = r.articles[ai]["noun"]
        seen = collections.Counter(r.cat.category(it) for it in r.cat.items.values())
        seen.pop("", None)
        hits = [(n, c) for c, n in seen.items() if same_name(noun, c)]
        return max(hits)[1] if hits else None

    def learn_category(self, ai: int) -> None:
        r = self.r
        art = r.articles[ai]
        if not r.store.category:
            return
        exact = self.exact_department(ai)
        if exact:
            if art.get("cat") != exact:
                art["cat"] = exact
                r.notes.append(f"article {ai}: department named like the noun: {exact!r}")
            return
        if art.get("cat"):
            return
        cats = collections.Counter(r.cat.category(r.cat.items[i]) for i in art.get("ids") or [] if i in r.cat.items)
        cats.pop("", None)
        if not cats or sum(cats.values()) < 3:
            return
        noun = set(Text(art["noun"]).stems)
        named = [(len(noun & set(Text(c).stems)), n, c) for c, n in cats.items() if noun & set(Text(c).stems)]
        if named:
            art["cat"] = max(named)[2]
            return
        c, n = cats.most_common(1)[0]
        if n >= 0.6 * sum(cats.values()):
            art["cat"] = c

    @staticmethod
    def notation_budget(amt: float, prices: list[float]) -> float | None:
        if len(prices) < 3 or amt != int(amt) or any(p > amt for p in prices):
            return None
        d = max(len(f"{p:.4f}".rstrip("0").split(".")[1]) for p in prices)
        if d < 1:
            return None
        b = amt / 10 ** d
        n = sum(1 for p in prices if p <= b)
        mid = sorted(prices)[len(prices) // 2]
        return b if 0 < n < len(prices) and abs(math.log(b / mid)) < abs(math.log(amt / mid)) else None

    def settle_budget(self) -> None:
        r = self.r
        try:
            amt = float((r.budget_raw or {}).get("amount"))
        except (TypeError, ValueError):
            amt = None
        hint = r.store.budget_hint
        r.budget = amt if amt and amt > 0 else hint
        prices = [p for p in (r.cat.price(it) for it in r.cat.items.values()) if p is not None and p > 0]
        if not hint and amt and r.store.money_odd and r.store.budget_lost:
            hint = self.notation_budget(amt, prices)
            if hint:
                r.notes.append(f"budget scale unresolved: {amt:g} sits above every price; read in the prices' smallest written unit: {hint:g}")
        if not prices or not amt or not hint or abs(hint - amt) < 1e-9:
            return

        def share(v: float) -> float:
            return sum(1 for p in prices if p <= v) / len(prices)
        if share(amt) < 0.1 <= share(hint) or (r.store.money_odd and hint < amt):
            r.budget = hint
        r.money_scale = r.budget / amt

    def probe(self) -> None:
        r = self.r
        finder = r.finder
        a0 = r.articles[0]
        tried: list[str] = []
        first_obs = None
        first_ph = None
        terms = self.search_terms(0)
        first_words = search_steps(terms, None)[0][0]
        firsts = [" ".join(first_words)] + [p_ for p_ in [Text(a0["noun"]).singular, a0["noun"]] if p_]
        multi = {Text.of(v) for x in r.reqs(0) if x["op"] in ("is", "eq") and isinstance(x["value"], list) and len(x["value"]) > 1
                 for v in x["value"]}
        own = {Text.of(x["value"]): set(Text.key_stems(x["prop"])) for x in r.reqs(0) if isinstance(x["value"], str)}
        kept = [" ".join([w for w in Text(v).words if Text.stem(w) not in own.get(v, set())] or Text(v).words)
                for v in terms["values"] if v not in multi]
        sq = r.tm.get("search") or {}
        cap = (((r.schemas.get(sq.get("name") or "") or {}).get("properties") or {}).get(sq.get("query") or "") or {}).get("maxLength")
        if multi and kept and not (isinstance(cap, int) and len(firsts[0]) > cap):
            firsts.insert(1, " ".join(search_steps({**terms, "values": kept}, None)[0][0]))
        for ph in firsts[:3]:
            if ph.lower() in tried:
                continue
            tried.append(ph.lower())
            n0 = len(r.dialogue)
            ids, _, _ = finder.run(ph)
            if r.steps > 0:
                a0["search_steps"] = a0.get("search_steps", 0) + 1
            self.add_ids(a0, ids)
            if ids:
                first_obs = r.dialogue[n0]["environment_result"]["observation"]
                first_ph = ph
                if ph == firsts[0]:
                    finder.learn_traits(list(first_words), ids)
                break
        if a0.get("ids"):
            r.phase = "read"
            self.open(a0["ids"][:r.store.compare_max() if r.store.facts_role() == "compare" else 1])
            r.phase = "find"
        if first_obs is not None:
            r.store.confirm(r)
            self.settle_budget()
            for d in list(r.dialogue):
                tc = (d.get("tool_call") or {}).get("function") or {}
                if tc.get("name"):
                    r.cat.ingest(tc["name"], Text.loads(tc.get("arguments")) or {}, (d.get("environment_result") or {}).get("observation"))
            ids, total, cursor = r.cat.rows_of(first_obs)
            q = finder.state(first_ph) if first_ph else None
            if q is not None:
                q.total, q.cursor, q.ids = total, cursor, ids
                q.fetched = q.first = len(ids)
        a0["done"] = sorted(set(a0.get("done") or []) | set(tried))
        self.learn_category(0)
        self.store_probe()

    def store_probe(self) -> None:
        r = self.r
        f = r.finder
        if f.match() or r.steps_left() - r.plan.finish_need() < 8 or r.t() > GUARD_AT - 80:
            return
        df = f.df_table()
        for art in r.articles:
            groups = [g for g in self.sas_groups(art["i"]) if g["stated"]]
            if groups:
                break
        else:
            return
        heads = self.shelf_words(art["i"], df)
        h = heads[0][0] if heads else ""
        toks = [w for g in groups for w in g["alts"][0] if len(w) > 1 and Text.stem(w) != Text.stem(h)]
        if not toks or not h:
            return
        a = min(toks, key=lambda w: (not any(c.isalpha() for c in w), df.get(Text.stem(w), 0), -len(w), w))
        for q in (f"{a} {h}", h, a):
            if f.match() or f.state(q) is not None or r.steps_left() - r.plan.finish_need() < 6:
                continue
            if q == h:
                pair = f.state(f"{a} {h}")
                if pair is None or not isinstance(pair.total, int) or pair.total > f.reach():
                    continue
            ids, _, _ = self.line_run(art["i"], q, force=True)
            self.add_ids(art, ids)
        self.learn_category(art["i"])
        r.notes.append(f"sas: store {f.match()} (votes {f.shape}) from {a!r}/{h!r}; price order {f.price_ordered()}")

    def sas_groups(self, ai: int) -> list[dict]:
        r = self.r
        f = r.finder
        out = []
        for req in r.reqs(ai):
            if req["op"] not in ("is", "eq"):
                continue
            vals = req["value"] if isinstance(req["value"], list) else [req["value"]]
            alts: list[list[str]] = []
            for v in vals:
                t = self.written(ai, req, v)
                if not t.strip() or code_shaped(t):
                    continue
                toks = f.value_tokens(t)
                if toks and toks not in alts:
                    alts.append(toks)
            if alts and len(alts) <= 4:
                out.append({"sig": f.sig(req), "alts": alts, "stated": req["source"] == "stated"})
        return out

    def shelf_words(self, ai: int, df: collections.Counter) -> list[tuple[str, bool]]:
        art = self.r.articles[ai]
        cat = Text.of(art.get("cat")).strip()
        ws = [w for w in dict.fromkeys(Text(cat or art["noun"]).words) if len(w) > 2 and any(c.isalpha() for c in w)]
        ws.sort(key=lambda w: (df.get(Text.stem(w), 0), -len(w), w))
        return [(w, bool(cat)) for w in ws]

    def sas_plan(self, ai: int, groups: list[dict], df: collections.Counter) -> list[tuple]:
        art = self.r.articles[ai]

        def rare(toks: list[str]) -> list[str]:
            return sorted(toks, key=lambda w: (not any(c.isalpha() for c in w), df.get(Text.stem(w), 0), -len(w), w))

        def cost(qs: list[str]) -> float:
            return sum(df.get(Text.stem(w), 0) for q in qs for w in q.split()) / max(1, len(qs))
        out = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                A, B = groups[i], groups[j]
                qs: list[str] = []
                for x in A["alts"]:
                    for y in B["alts"]:
                        a = rare(x)[0]
                        b = next((w for w in rare(y) if Text.stem(w) != Text.stem(a)), None)
                        if b is not None and f"{a} {b}" not in qs:
                            qs.append(f"{a} {b}")
                if qs and len(qs) <= 4:
                    out.append(((len(qs), cost(qs), A["stated"] + B["stated"]), qs, [A["sig"], B["sig"]], None))
        heads = self.shelf_words(ai, df)
        for g in groups:
            if len(g["alts"]) == 1 and len(g["alts"][0]) >= 2:
                t = " ".join(rare(g["alts"][0])[:2])
                out.append(((2, cost([t]), 2 * g["stated"]), [t], [g["sig"]], None))
            qs, used, named = [], False, True
            for x in g["alts"]:
                t = rare(x)
                if len(t) >= 2:
                    q = " ".join(t[:2])
                else:
                    h = next(((w, c) for w, c in heads if Text.stem(w) != Text.stem(t[0])), None)
                    if h is None:
                        continue
                    q = f"{t[0]} {h[0]}"
                    used, named = True, named and h[1]
                if q not in qs:
                    qs.append(q)
            if qs and len(qs) == len(g["alts"]):
                out.append(((3 * len(qs), 1e9, 2 * g["stated"]), qs, [g["sig"]], ((art.get("cat") or "") if named else False) if used else None))
        out.sort(key=lambda c: (c[0], c[1]))
        return out

    def lattice(self, ai: int, allowance: int, stop_at_set: bool = True) -> int:
        r = self.r
        f = r.finder
        art = r.articles[ai]
        art.setdefault("ids", [])
        groups = self.sas_groups(ai)
        if not groups or allowance <= 0:
            return 0
        start = r.steps
        df = f.df_table()
        rows = f.rows_per_page()
        reach = f.reach()
        done = set(art.get("sas_done") or [])
        fresh = 0
        was = r.phase
        for _score, qs, need, cat in self.sas_plan(ai, groups, df):
            key = " | ".join(sorted(need)) + (" @" if cat is not None else "")
            if key in done:
                continue
            if was in ("find", "read"):
                r.phase = "find" if r.plan.room("find") > 0 else "read"
            left = allowance - (r.steps - start)
            if left <= 0 or r.plan.room(r.phase) <= 0 or r.t() > GUARD_AT - 60:
                break
            todo = [q for q in qs if f.state(q) is None]
            if len(todo) > left:
                continue
            for q in todo:
                ids, _, _ = self.line_run(ai, q, force=True)
                self.add_ids(art, ids)
            self.learn_category(ai)
            got = [f.state(q) for q in qs]
            if any(g is None or not isinstance(g.total, int) for g in got):
                done.add(key)
                continue
            tot = sum(g.total for g in got)
            if tot == 0 or tot > reach:
                done.add(key)
                continue
            pages = sum(max(0, -(-g.total // max(1, rows)) - g.pages) for g in got if not f.complete(g))
            if pages > allowance - (r.steps - start):
                break
            done.add(key)
            for g in got:
                if not f.complete(g):
                    if was in ("find", "read"):
                        r.phase = "find" if r.plan.room("find") > 0 else "read"
                    self.add_ids(art, f.page_on(g.text, max(0, -(-g.total // max(1, rows)) - g.pages)))
            if cat is not False and all(f.complete(g) for g in got):
                art.setdefault("sas_sets", []).append({"qs": list(qs), "need": list(need), "cat": cat or ""})
                fresh += 1
                r.notes.append(f"sas: line {ai} complete set {qs} with {tot} rows")
                if stop_at_set:
                    break
        art["sas_done"] = sorted(done)
        r.phase = was
        return fresh

    def sas_share(self, lines_left: int) -> int:
        plan = self.r.plan
        return max(2, (plan.room("find") + plan.room("read")) // max(1, 3 * lines_left))

    def sas_certified(self, ai: int) -> bool:
        r = self.r
        if not r.cheapest or not r.finder.sas():
            return False
        for s in r.finder.live_sets(ai):
            rows = [r.cat.items[i] for i in s["ids"] if i in r.cat.items and r.ok_basic(r.cat.items[i], ai)]
            rows.sort(key=lambda it: (r.judge.price_of(it), it["id"]))
            for it in rows:
                st = r.basket.state(it, ai)
                if st == "v":
                    return True
                if st == "u":
                    break
        return False

    def sorted_open(self, ai: int, k: int) -> None:
        r = self.r
        art = r.articles[ai]
        for _ in range(k):
            if r.plan.room(r.phase) <= 0 or r.t() > GUARD_AT - 40:
                return
            bar = min((r.judge.price_of(it) for it in r.candidates(ai) if r.basket.state(it, ai) == "v"), default=None)
            pool = [it for it in self.unread_pool(ai, art.get("ids") or [])
                    if (bar is None or r.judge.price_of(it) < bar) and r.basket.state(it, ai) != "x"]
            if not pool:
                return
            it = min(pool, key=lambda x: (r.judge.price_of(x), x["id"]))
            self.open([it["id"]])
            if not it.get("read"):
                return
            if r.judge.line_level(it, ai) is None and r.judge.can_judge(1):
                r.judge.judge_batch([(it, ai)])
            if r.basket.state(it, ai) == "v":
                return

    def clean(self, ai: int, words: Any) -> str:
        r = self.r
        art = r.articles[ai]
        cat = r.cat
        seen = set(Text(art.get("cat") or "").words)
        for it in cat.items.values():
            if art.get("cat") and cat.category(it) == art["cat"]:
                seen.update(Text(cat.title(it)).words)
        return " ".join(w for w in words if len(w) > 2 or any(c.isdigit() for c in w) or w in seen)

    def long_text(self, ai: int) -> str:
        terms = self.search_terms(ai)
        return self.clean(ai, search_steps({**terms, "values": terms["values"] + terms["revealed"]}, None)[0][0])

    def sas_leftover(self) -> bool:
        r = self.r
        if not r.finder.sas():
            return False
        for art in r.articles:
            ai = art["i"]
            if art.get("sas_long") or self.sas_certified(ai):
                continue
            if not r.cheapest and any(r.basket.state(it, ai) == "v" for it in r.candidates(ai)):
                continue
            art["sas_long"] = True
            text = self.long_text(ai)
            if not text or r.finder.state(text) is not None:
                continue
            ids, _, _ = self.line_run(ai, text, force=True)
            self.add_ids(art, ids)
            cur = r.cat.items.get(art.get("pick") or "")
            bar = r.cat.price(cur) if cur is not None else None
            pool = [it for it in self.unread_pool(ai, ids) if bar is None or (r.cat.price(it) is not None and r.cat.price(it) < bar)]
            pool = self.model_order(ai, pool, 2, base=lambda it: r.judge.price_of(it))
            self.open([it["id"] for it in pool[:2]])
            r.notes.append(f"sas: line {ai} leftover long query {text!r}, {len(ids)} rows")
            return True
        return False

    def phrases_for(self, ai: int) -> list[str]:
        art = self.r.articles[ai]
        phrases = [p for p in art["search"] if p]
        combo = self.r.finder.combined_phrase(ai)
        if combo and combo.lower() not in {p.lower() for p in phrases}:
            phrases.insert(0, combo)
        phrases.append(art["noun"])
        return phrases

    def line_run(self, ai: int, q: str, force: bool = False) -> tuple[list[str], int | None, Any]:
        r = self.r
        art = r.articles[ai] if isinstance(ai, int) and 0 <= ai < len(r.articles) else None
        if art is not None and not force and art.get("search_steps", 0) >= LINE_SEARCH_STEPS:
            return [], None, None
        n0 = r.steps
        out = r.finder.run(q)
        if art is not None and r.steps > n0:
            art["search_steps"] = art.get("search_steps", 0) + (r.steps - n0)
        return out

    def prove_line(self, ai: int) -> bool:
        r = self.r
        art = r.articles[ai]
        words = search_steps(self.search_terms(ai), None)[0][0]
        text = " ".join(words)
        found = False
        for _ in range(2):
            if r.plan.slack() < 3 or r.t() > GUARD_AT - 25:
                break
            q = r.finder.state(text)
            ids = self.line_run(ai, text, force=True)[0] if q is None else r.finder.page_on(text, 1)
            if not ids:
                break
            self.add_ids(art, ids)
            k = min(3, max(1, r.plan.slack() - 2))
            pool = self.model_order(ai, self.unread_pool(ai, ids), k, base=lambda it: 1e18 if r.cat.price(it) is None else r.cat.price(it),
                                    force=True)
            self.open([it["id"] for it in pool[:k]], force=True)
            if any(r.judge.fully_met(it, ai) for it in r.candidates(ai)):
                found = True
                break
        r.notes.append(f"prove line {ai}: {'found' if found else 'none'} under {text!r}")
        return found

    def written(self, ai: int, req: dict, v: Any) -> str:
        text = Text.of(v).strip()
        if not text.isdigit():
            return Text.of(v)
        unit = Text.of(req.get("unit")).strip()
        if unit and unit.replace(" ", "").isalpha():
            return f"{text} {unit}"
        r = self.r
        prop = Text.of(req.get("prop"))
        stamp = (r.steps, len(r.cat.items))
        got = self.written_memo.get((ai, prop, text))
        if got is not None and got[0] == stamp:
            return got[1]
        cat = r.articles[ai].get("cat") if isinstance(ai, int) and 0 <= ai < len(r.articles) else None
        tails: collections.Counter = collections.Counter()
        for it in r.cat.items.values():
            if not it.get("read") or (cat and r.cat.category(it) != cat):
                continue
            for _k, val in r.cat.fact_values(it, prop):
                ws = Text(val).words
                if len(ws) >= 2 and ws[0].isdigit() and ws[1].isalpha():
                    tails[ws[1]] += 1
        if not tails:
            own = set(Text.key_stems(prop))
            for it in r.cat.items.values():
                if cat and r.cat.category(it) != cat:
                    continue
                ws = Text(r.cat.title(it)).words
                for x, y in zip(ws, ws[1:]):
                    if x.isdigit() and y.isalpha() and Text.stem(y) in own:
                        tails[y] += 1
        out = text
        if tails:
            word, n = tails.most_common(1)[0]
            if n * 2 > sum(tails.values()):
                out = f"{text} {word}"
        self.written_memo[(ai, prop, text)] = (stamp, out)
        return out

    def rewritten(self, ai: int) -> bool:
        for req in self.r.reqs(ai):
            if req["op"] not in ("is", "eq"):
                continue
            vals = req["value"] if isinstance(req["value"], list) else [req["value"]]
            if any(Text.of(v).strip().isdigit() and self.written(ai, req, v) != Text.of(v) for v in vals):
                return True
        return False

    def refresh(self, ai: int) -> None:
        r = self.r
        art = r.articles[ai]
        if not self.rewritten(ai) or r.plan.room(r.phase) < 2 or r.t() > GUARD_AT - 60:
            return
        words = search_steps(self.search_terms(ai), None)[0][0]
        text = self.clean(ai, words) if r.finder.sas() else " ".join(words)
        if not text or any(r.finder.stem_set(t) == r.finder.stem_set(text) for t in list(r.finder.queries)):
            return
        ids, total, _ = self.line_run(ai, text, force=True)
        self.add_ids(art, ids)
        self.learn_category(ai)
        r.notes.append(f"article {ai}: value written as the store writes it, searched {text!r}, {len(ids)} rows of {total}")

    def capped_carrier(self, ai: int) -> bool:
        r = self.r
        f = r.finder
        groups = f.value_groups(ai)
        if not groups:
            return False
        rows = f.rows_per_page()
        lim = f.page_limit(rows)
        for q in dict.fromkeys(f.queries.values()):
            if not isinstance(q.total, int) or f.lead(q) is not q or not f.carries(f.class_text(q), groups):
                continue
            if (q.cursor in (None, "", False) and q.pages >= 2 and q.fetched < q.total) or (lim and q.total > rows * lim):
                return True
        return False

    def verified_bar(self, ai: int) -> float | None:
        r = self.r
        if not r.cheapest:
            return None
        return min((r.judge.price_of(it) for it in r.candidates(ai) if r.basket.state(it, ai) == "v"), default=None)

    def cheaper_unread(self, ai: int, ids: list[str]) -> list[dict]:
        bar = self.verified_bar(ai)
        return [it for it in self.unread_pool(ai, ids) if bar is None or self.r.judge.price_of(it) < bar]

    def value_queries(self, ai: int) -> list[str]:
        picks: list[list[str]] = []
        for req in self.r.reqs(ai):
            if req["op"] not in ("is", "eq"):
                continue
            vals = req["value"] if isinstance(req["value"], list) else [req["value"]]
            raw = [Text.of(v) for v in vals if Text.of(v).strip() and not code_shaped(v) and not self.r.finder.unsearchable(v)]
            one, rest = Reach.one_of([self.written(ai, req, v) for v in raw])
            if one:
                picks.append([one] + rest + [v for v in raw if v not in rest and v != one])
        if not picks:
            return []
        terms = self.search_terms(ai)
        base = [p[0] for p in picks]
        sets = [base] + [base[:k] + [alt] + base[k + 1:] for k, p in enumerate(picks) for alt in p[1:]]
        out: list[str] = []
        for group in sets:
            words = search_steps({**terms, "values": group, "revealed": [], "alts": [], "ints": []}, None)[0][0]
            text = self.clean(ai, words) if self.r.finder.sas() else " ".join(words)
            if text and text not in out:
                out.append(text)
        return out

    def search_terms(self, ai: int) -> dict:
        r = self.r
        art = r.articles[ai]
        values: list[str] = []
        revealed: list[str] = []
        alts: list[str] = []
        ints: list[str] = []
        decs: list[str] = []
        for req in r.reqs(ai):
            vals = req["value"] if isinstance(req["value"], list) else [req["value"]]
            if req["op"] in ("is", "eq"):
                kept = [self.written(ai, req, v) for v in vals if Text.of(v).strip() and not code_shaped(v) and not r.finder.unsearchable(v)]
                one, rest = Reach.one_of(kept)
                if one:
                    (values if req["source"] == "stated" else revealed).append(one)
                alts += rest
            elif req["op"] in ("min", "max", "between"):
                for v in vals:
                    for n, _ in Text(Text.of(v)).numbers:
                        word = str(int(n)) if float(n).is_integer() else str(n)
                        (ints if float(n).is_integer() else decs).append(word)
        return {"noun": art["noun"], "heads": noun_heads(art["noun"]), "values": values, "revealed": revealed,
                "alts": list(dict.fromkeys(alts)), "ints": list(dict.fromkeys(ints)), "decs": list(dict.fromkeys(decs))}

    def store_fuses(self, unit: str) -> bool:
        joined = spaced = 0
        for it in self.r.cat.items.values():
            words = Text(self.r.cat.title(it)).words
            for k, w in enumerate(words):
                if w.endswith(unit) and Text(w[:-len(unit)]).words == [w[:-len(unit)]] and w[:-len(unit)].replace(".", "").isdigit():
                    joined += 1
                elif w.replace(".", "").isdigit() and k + 1 < len(words) and words[k + 1] == unit:
                    spaced += 1
        return joined > spaced

    def criteria_step(self, art: dict, halve: bool) -> list[str]:
        r = self.r
        f = r.tm.get("filter")
        ai = art["i"]
        if not (f and f.get("category") and f.get("max_price") and art.get("cat")):
            return []
        if halve:
            last = art.get("criteria")
            if not last or last[1] is None or last[1] <= last[2]:
                return []
            ceiling = last[0] / 2
        else:
            fits = [r.cat.price(it) for it in r.cat.items.values() if r.cat.category(it) == art["cat"] and r.cat.price(it) is not None
                    and r.ok_basic(it, ai) and not (it["read"] and r.contradicted(it, ai))]
            if not fits:
                return []
            ceiling = min(fits)
        ceiling = int(ceiling) if float(ceiling).is_integer() else round(ceiling, 2)
        args = {**(f.get("args") or {}), f["category"]: art["cat"], f["max_price"]: ceiling}
        obs = r.env.send([(f["name"], args)])
        ids, total, _ = r.cat.rows_of(obs[0] if obs else None)
        reach = max(1, len(ids)) * (r.finder.page_limit(max(1, len(ids))) or 3)
        art["criteria"] = (ceiling, total if isinstance(total, int) else len(ids), reach)
        return ids

    def shelf_step(self, art: dict, page: int) -> list[str] | None:
        r = self.r
        f = r.tm.get("filter")
        if not (f and f.get("category") and art.get("cat")):
            return None
        args = {**(f.get("args") or {}), f["category"]: art["cat"]}
        if page > 1:
            pg = f.get("page")
            last = art.get("shelf_page")
            if not pg or last != page - 1:
                return None
            spec = ((r.schemas.get(f["name"]) or {}).get("properties") or {}).get(pg) or {}
            args[pg] = page if spec.get("minimum", 1) != 0 else len(art.get("shelf_ids") or [])
        obs = r.env.send([(f["name"], args)])
        ids, _, _ = r.cat.rows_of(obs[0] if obs else None)
        art["shelf_page"] = page
        art["shelf_ids"] = (art.get("shelf_ids") or []) + ids
        return ids

    def plan_step(self, art: dict, words: tuple[str, ...], page: int) -> list[str] | None:
        r = self.r
        finder = r.finder
        if words and words[0] == "@criteria":
            return self.criteria_step(art, False) if r.tm.get("filter") else None
        if words and words[0] == "@halve":
            return self.criteria_step(art, True) if art.get("criteria") else None
        if words and words[0] == "@shelf":
            return self.shelf_step(art, page)
        text = self.clean(art["i"], words) if finder.sas() else " ".join(words)
        if not text:
            return None
        q = finder.state(text)
        if page == 1:
            if q is not None:
                return None
            ids, _, _ = finder.run(text)
            finder.learn_traits(list(words), ids)
            return ids
        if q is None or q.pages >= page or q.pages < page - 1:
            return None
        if q.total is not None and q.total <= len(q.ids):
            return None
        return finder.page_on(text, 1)

    def find(self, ai: int, allowance: int) -> None:
        r = self.r
        finder = r.finder
        art = r.articles[ai]
        art.setdefault("ids", [])
        start = r.steps
        done: set[str] = set(art.get("done") or [])
        valued = 0

        if finder.sas() and self.sas_groups(ai):
            self.lattice(ai, allowance)
            if art.get("ids") or r.steps - start >= allowance:
                return
            allowance -= r.steps - start
            start = r.steps

        def budget_left() -> bool:
            return (r.steps - start < allowance and r.plan.room("find") > 0
                    and art.get("search_steps", 0) < LINE_SEARCH_STEPS)

        tried = set(art.get("plan_done") or [])
        for _round in range(LINE_SEARCH_STEPS * 2):
            if not budget_left():
                break
            steps = [st for st in search_steps(self.search_terms(ai), finder.traits) if st not in tried]
            if not steps:
                break
            st = steps[0]
            tried.add(st)
            n0 = r.steps
            ids = self.plan_step(art, st[0], st[1])
            if r.steps > n0:
                art["search_steps"] = art.get("search_steps", 0) + (r.steps - n0)
            if ids:
                self.add_ids(art, ids)
                valued += len(ids)
                self.learn_category(ai)
                if st[1] == 1 and not st[0][0].startswith("@"):
                    done.add(" ".join(st[0]))
        art["plan_done"] = sorted(tried)
        self.browse_fallback(art)
        if art.get("cat") and not (set(Text(art["cat"]).stems) & set(Text(art["noun"]).stems)) and r.plan.room("find") > 0:
            bare = Text(art["noun"]).words[-1:]
            if bare and bare[0] not in done:
                done.add(bare[0])
                ids, _, _ = self.line_run(ai, bare[0])
                self.add_ids(art, ids)
                art["cat"] = None
                self.learn_category(ai)
        written = " ".join(Text(art["noun"]).words)
        if not art.get("cat") and r.store.category and written and written not in done and r.plan.room("find") > 0:
            done.add(written)
            ids, _, _ = self.line_run(ai, written)
            self.add_ids(art, ids)
            self.learn_category(ai)
            if art.get("cat"):
                r.notes.append(f"article {ai}: department from the noun as written: {art['cat']!r}")
        art["done"] = sorted(done)

    def own_department(self, ai: int) -> None:
        r = self.r
        art = r.articles[ai]
        if not art.get("cat") or self.exact_department(ai) or not any(
                o is not art and self.exact_department(o["i"]) == art["cat"] for o in r.articles):
            return
        bare = " ".join(Text(art["noun"]).words)
        done = set(art.get("done") or [])
        if not bare or bare in done:
            return
        ids, _, _ = self.line_run(ai, bare, force=True)
        art["done"] = sorted(done | {bare})
        self.add_ids(art, ids)
        taken = art["cat"]
        art["cat"] = None
        self.learn_category(ai)
        if art.get("cat") == taken:
            art["cat"] = None
        r.notes.append(f"article {ai}: department {taken!r} belongs to another line; searched {bare!r}, now {art.get('cat')!r}")

    def sibling_department(self, ai: int) -> None:
        r = self.r
        art = r.articles[ai]
        if not r.store.category or art.get("sibling_checked") or self.exact_department(ai):
            return
        cat = art.get("cat")
        if cat and not sibling_name(art["noun"], cat) and not wider_name(art["noun"], cat):
            return
        art["sibling_checked"] = True
        bare = " ".join(Text(art["noun"]).words)[:60].strip()
        if not bare or r.finder.state(bare) is not None or r.steps_left() - r.plan.finish_need() < 4 or r.t() > GUARD_AT - 70:
            return
        ids, _, _ = self.line_run(ai, bare, force=True)
        self.add_ids(art, ids)
        self.learn_category(ai)
        r.notes.append(f"article {ai}: shelf {cat!r} is a sibling or unknown; searched {bare!r}, now {art.get('cat')!r}")

    def browse_fallback(self, art: dict) -> None:
        r = self.r
        f = r.tm.get("filter") or {}
        if (not f.get("category") or f.get("name") == (r.tm.get("search") or {}).get("name") or art.get("browsed")
                or len(art.get("ids") or []) >= BROWSE_FEW or r.plan.room("find") <= 0 or r.t() > GUARD_AT - 70):
            return
        art["browsed"] = True
        spec = ((r.schemas.get(f["name"]) or {}).get("properties") or {}).get(f["category"]) or {}
        labels = [art.get("cat")] + ([] if spec.get("enum") else [art["noun"]])
        for label in dict.fromkeys(x for x in labels if x):
            args = {**(f.get("args") or {}), f["category"]: label}
            obs = r.env.send([(f["name"], args)])
            ids, _, _ = r.cat.rows_of(obs[0] if obs else None)
            r.notes.append(f"article {art['i']}: browse {f['name']} {label!r}, {len(ids)} rows")
            if ids:
                self.add_ids(art, ids)
                self.learn_category(art["i"])
                return
            if r.plan.room("find") <= 0:
                return

    def fit_ceiling(self, ai: int) -> float | None:
        r = self.r
        if not r.budget:
            return None
        others = 0.0
        for a in r.articles:
            if a["i"] == ai:
                continue
            ps = [r.cat.price(it) for it in r.cat.items.values() if r.cat.price(it) is not None and r.basket.state(it, a["i"]) != "x"]
            others += min(ps, default=0.0) * max(1, int(a.get("qty") or 1))
        art = r.articles[ai]
        top = (r.budget - others) / max(1, int(art.get("qty") or 1))
        if art.get("unit_cap"):
            top = min(top, art["unit_cap"] * r.money_scale)
        return top if top > 0 else None

    def fit_scan(self, ai: int, allowance: int, ok: Callable[[dict], bool] | None = None, cap: float | None = None) -> int:
        r = self.r
        art = r.articles[ai]
        coded = r.judge.code_reqs(ai)
        if ok is None and not coded:
            return 0
        items = r.cat.items
        good = ok or (lambda it: all(r.cat.code_verdict(it, q) == "yes" for q in coded))
        afford = self.fit_afford(ai) if ok is None else None

        def live() -> int:
            return sum(1 for i in art.get("ids") or [] if i in items and r.basket.state(items[i], ai) != "x" and good(items[i])
                       and (afford is None or r.judge.price_of(items[i]) <= afford + 1e-9))
        f = r.tm.get("filter") or {}
        shelf = bool(f.get("category") and art.get("cat"))
        top = self.fit_ceiling(ai) if shelf and f.get("max_price") else None
        if cap is not None and shelf and f.get("max_price"):
            top = cap if top is None else min(top, cap)
        slot = "fit_cursor" if cap is None else f"fit_cursor@{cap}"
        terms = self.search_terms(ai)
        vals = terms["values"] + terms["revealed"]
        head = self.line_word(ai)
        phrases = list(dict.fromkeys(" ".join(list(dict.fromkeys(w for v in group for w in Text(v).words)) + [head]).strip()
                                     for group in ([vals] if len(vals) > 1 else []) + [[v] for v in vals]))
        loose = [w for v in vals if len([x for x in Text(v).words if x.isalpha() and len(x) > 2]) >= 2
                 for w in Text(v).words if w.isalpha() and len(w) > 2]
        phrases += [p for p in dict.fromkeys(f"{w} {head}".strip() for w in loose) if p not in phrases]
        for v in vals:
            parts = Text(v).words
            if len(parts) > 1 and not any(c.isspace() for c in Text.of(v).strip()):
                phrases += [x for x in (f"{w} {head}".strip() for w in parts if w.isalpha() and len(w) > 2) if x not in phrases]
            phrases += [x for x in (f"{int(n)} {head}".strip() for n, unit in Text(v).numbers
                                    if unit and float(n).is_integer() and any(c.isalpha() for c in Text.of(v)) and str(int(n)) not in parts) if x not in phrases]
        rows_pp = r.finder.rows_per_page()
        lim = r.finder.page_limit(rows_pp)
        start, pages = r.steps, 0
        while r.steps - start < allowance and live() < FIT_SCAN_KEEP and r.steps_left() > r.plan.finish_need(minimal=True) + 2 and r.t() < GUARD_AT - 45:
            n0 = r.steps
            q, nxt = self.fit_query(ai)
            carried = q is not None and r.finder.carries(q.text, r.finder.value_groups(ai)) > 0
            phrase = next((x for x in phrases if r.finder.state(x) is None), "")
            first = cap is not None and shelf and art.get(slot) is not False
            if carried and phrase and lim and isinstance(q.total, int) and q.total > rows_pp * lim:
                carried = False
            if not first and (carried or (q is not None and not (shelf and art.get(slot) is not False) and not phrase)):
                q.used.append(nxt)
                ids, _, _ = r.finder.run(q.text, {r.store.page_arg("search"): nxt})
            elif phrase and not first:
                ids, _, _ = self.line_run(ai, phrase, force=True)
            elif shelf and art.get(slot) is not False:
                args = {**(f.get("args") or {}), f["category"]: art["cat"]}
                if top is not None:
                    args[f["max_price"]] = int(top) if float(top).is_integer() else round(top, 2)
                if art.get(slot) is not None:
                    if not f.get("page"):
                        art[slot] = False
                        continue
                    args[f["page"]] = art[slot]
                obs = r.env.send([(f["name"], args)])
                ids, _, cursor = r.cat.rows_of(obs[0] if obs else None)
                art[slot] = cursor if ids and cursor not in (None, "", False) else False
            elif (split := self.fit_split(ai)) is not None:
                ids, _, _ = self.line_run(ai, split[0], force=True) if split[1] is None else r.finder.run(split[0], {r.store.page_arg("search"): split[1]})
            else:
                break
            if r.steps == n0:
                break
            pages += 1
            self.add_ids(art, ids)
        if pages:
            r.notes.append(f"fit scan: line {ai} {pages} pages, {live()} fitting rows")
        return r.steps - start

    def line_word(self, ai: int) -> str:
        r = self.r
        art = r.articles[ai]
        own = Text(art.get("cat") or art["noun"]).words
        theirs = {x for o in r.articles if o is not art for x in Text(" ".join([o.get("cat") or "", o["noun"]])).stems}
        return next((w for w in own if len(w) > 2 and Text.stem(w) not in theirs), None) or (noun_heads(art["noun"]) or [""])[0]

    def fit_focus(self, q: Query, ai: int) -> bool:
        r = self.r
        art = r.articles[ai]
        rows = [r.cat.items[i] for i in q.ids if i in r.cat.items]
        if not rows:
            return False
        if art.get("cat"):
            return sum(1 for it in rows if r.cat.category(it) == art["cat"]) * 2 >= len(rows)
        return bool(set(Text(q.text).stems) & {Text.stem(w) for w in noun_heads(art["noun"])})

    def fit_query(self, ai: int) -> tuple[Query | None, Any]:
        r = self.r
        if not r.store.page_arg("search"):
            return None, None
        groups = r.finder.value_groups(ai)
        best = None
        for q in r.finder.open_queries():
            nxt = r.store.next_cursor("search", q)
            if nxt is None or nxt in q.used or not self.fit_focus(q, ai):
                continue
            k = (r.finder.carries(q.text, groups), -(q.total if isinstance(q.total, int) else 10 ** 9), q.text)
            if best is None or k > best[0]:
                best = (k, q, nxt)
        return (best[1], best[2]) if best else (None, None)

    def fit_split(self, ai: int) -> tuple[str, Any] | None:
        r = self.r
        art = r.articles[ai]
        tried = art.setdefault("fit_splits", [])
        capped = [q for q in r.finder.queries.values() if isinstance(q.total, int) and q.total > len(q.ids) and not q.cursor
                  and self.fit_focus(q, ai) and len(q.text.split()) < 6]
        if not capped:
            return None
        groups = r.finder.value_groups(ai)
        capped = [q for q in capped if q.text not in tried and r.finder.lead(q) is q] or capped
        q = max(capped, key=lambda q: (r.finder.carries(q.text, groups), q.total, q.text))
        got = self.split_pick(art, q, tried)
        if got is None:
            return None
        _, text, start = got
        tried.append(text)
        return text, (start or None)

    def fit_afford(self, ai: int) -> float | None:
        r = self.r
        if not r.budget:
            return None
        price = r.judge.price_of
        others = 0.0
        for a in r.articles:
            if a["i"] == ai:
                continue
            table = r.basket.table(a["i"])
            sure = [price(it) for it in table if r.basket.state(it, a["i"]) == "v"]
            others += min(sure or [price(it) for it in table if r.basket.state(it, a["i"]) != "x"] or [0.0]) * max(1, int(a.get("qty") or 1))
        return (r.budget - others) / max(1, int(r.articles[ai].get("qty") or 1))

    def fit_open(self, ai: int) -> bool:
        r = self.r
        coded = r.judge.code_reqs(ai)
        if not coded or r.articles[ai].get("fit_dry"):
            return False
        top = self.fit_afford(ai)
        return not any(r.basket.state(it, ai) == "v" and all(r.cat.code_verdict(it, q) == "yes" for q in coded)
                       and (top is None or r.judge.price_of(it) <= top + 1e-9) for it in r.candidates(ai))

    def fit_gap(self) -> int | None:
        return next((ai for ai in self.r.judge.coded_lines() if self.fit_open(ai)), None)

    def fit_feed(self, ai: int) -> None:
        r = self.r
        art = r.articles[ai]
        n0 = r.steps
        coded = r.judge.code_reqs(ai)
        order = {i: n for n, i in enumerate(art.get("ids") or [])}
        top = self.fit_afford(ai)
        live = lambda: sorted((it for it in self.unread_pool(ai, art.get("ids") or []) if all(r.cat.code_verdict(it, q) == "yes" for q in coded)
                               and (top is None or r.judge.price_of(it) <= top + 1e-9)),
                              key=lambda it: (r.judge.price_of(it) if r.cheapest else order.get(it["id"], len(order)), it["id"]))
        spare = lambda: r.plan.fit_room()
        if not live():
            self.fit_scan(ai, min(2, max(0, spare() - 3)))
        pool = live()[:max(0, min(2, spare() - 2))]
        for it in pool:
            r.judge.read_one(it)
        art["fit_fruitless"] = 0 if pool else art.get("fit_fruitless", 0) + 1
        if r.steps == n0 or art["fit_fruitless"] >= FIT_FRUITLESS:
            art["fit_dry"] = True

    def merge_same_kind(self) -> None:
        r = self.r
        keep, remap = [], {}
        for art in r.articles:
            cat = art.get("cat") or ""
            cs = set(Text(cat).stems)
            mine = set(Text(art["noun"]).stems)
            twin = next((k for k in keep if cat and k.get("cat") == cat and set(Text(k["noun"]).stems) <= cs and mine <= cs
                         and set(Text(k["noun"]).stems) - mine and mine - set(Text(k["noun"]).stems)
                         and not (k.get("counted") and art.get("counted"))), None)
            if twin is None:
                keep.append(art)
                continue
            remap[art["i"]] = twin["i"]
            twin["qty"] = max(twin["qty"], art["qty"])
            twin["ids"] = list(dict.fromkeys((twin.get("ids") or []) + (art.get("ids") or [])))
            twin["open"] = list(dict.fromkeys(twin["open"] + art["open"]))
        if not remap:
            return
        new = {art["i"]: n for n, art in enumerate(keep)}
        for old, tgt in remap.items():
            new[old] = new[tgt]
        for n, art in enumerate(keep):
            art["i"] = n
        r.articles = keep
        for x in r.ledger:
            if isinstance(x.get("a"), int) and x["a"] in new:
                x["a"] = new[x["a"]]
        r.together = [[new.get(a, a), new.get(b, b)] for a, b in r.together if new.get(a, a) != new.get(b, b)]
        self.ranked.clear()
        r.notes.append(f"merged {sorted(remap)} into one article each")

    def wave1(self, ai: int, k: int) -> None:
        r = self.r
        art = r.articles[ai]
        if k >= 2:
            self.refresh(ai)
            self.chase_first(ai)
            k -= self.page_short(ai, k // 2)
        pool = self.model_order(ai, self.unread_pool(ai, art.get("ids") or []), k)
        if len(pool) < k and r.finder.live_sets(ai):
            kept = {it["id"] for it in pool}
            pool += sorted((it for it in self.unread_pool(ai, art.get("ids") or [], gate=False) if it["id"] not in kept),
                           key=r.finder.order_key(r.finder.value_groups(ai)))
        if r.together and k > 0 and FIT_READ_LEAD:
            lead = [r.cat.items[i] for i in r.judge.fit_lead(ai) if i in r.cat.items and not r.cat.items[i].get("read")]
            if lead:
                ids = {it["id"] for it in lead}
                pool = lead + [it for it in pool if it["id"] not in ids]
        self.open([it["id"] for it in pool[:k]])

    def chase_first(self, ai: int) -> None:
        r = self.r
        art = r.articles[ai]
        for req in r.reqs(ai):
            if req["source"] == "stated" or req.get("searched") or req.get("chased") or req["op"] not in ("is", "eq"):
                continue
            if r.plan.room(r.phase) < 3 or r.t() > GUARD_AT - 60:
                return
            req["chased"] = True
            val = req["value"] if isinstance(req["value"], str) else (req["value"][0] if req["value"] else "")
            val = self.written(ai, req, val)
            if not re.search(r"[a-z0-9]{2}", Text.of(val).lower()) or code_shaped(val) or r.finder.unsearchable(val):
                continue
            stated = self.search_terms(ai)["values"][:1]
            words = search_steps({"noun": art["noun"], "heads": noun_heads(art["noun"]), "values": [val] + stated}, None)[0][0]
            lead = [w for v in [val] + stated for w in Text(v).words[:1]]
            q = " ".join(dict.fromkeys(lead + list(words)))
            if not q.strip() or any(r.finder.stem_set(t) == r.finder.stem_set(q) for t in list(r.finder.queries)):
                continue
            ids, total, _ = self.line_run(ai, q, force=True)
            self.add_ids(art, ids)
            r.notes.append(f"article {ai}: revealed value searched before reads, {q!r}, {len(ids)} rows of {total}")

    def page_short(self, ai: int, budget: int) -> int:
        r = self.r
        art = r.articles[ai]
        f = r.finder
        groups = f.value_groups(ai)
        if budget <= 0 or not groups or not r.reach.lacking(ai) or not r.store.page_arg("search"):
            return 0
        rows = max(1, f.rows_per_page())
        best = None
        for q in self.carrying_open(art, groups):
            if not isinstance(q.total, int):
                continue
            pages = -(-max(0, min(q.total, rows * (f.page_limit(rows) or q.total)) - q.fetched) // rows)
            key = (-f.carries(f.class_text(q), groups), pages)
            if 0 < pages <= budget and (best is None or key < best[0]):
                best = (key, q, pages)
        if best is None:
            return 0
        n0 = r.steps
        self.add_ids(art, f.page_on(best[1].text, best[2]))
        r.notes.append(f"article {ai}: paged the value search {best[1].text!r} before reads, {best[1].fetched} of {best[1].total} rows")
        return r.steps - n0

    def chase_revealed(self, art: dict) -> bool:
        r = self.r
        ai = art["i"]
        for req in r.reqs(ai):
            if req["source"] == "stated" or req.get("searched") or req["op"] in ("not", "ref"):
                continue
            if not r.cheapest and any(r.cat.verdict(it, req) == "yes" and r.fit(it, ai)[1] == 0 and r.cat.fact_values(it, req.get("prop") or "")
                                      for it in r.owners(ai)):
                continue
            room = r.plan.chase_room()
            if room < 2 or r.t() > GUARD_AT - 40:
                return False
            req["searched"] = True
            if r.finder.sas() and req["op"] in ("is", "eq"):
                r.plan.extra = max(0, room - r.plan.room("read"))
                try:
                    before = set(art.get("ids") or [])
                    self.lattice(ai, max(1, room // 2))
                    sets = r.finder.live_sets(ai)
                    src = [i for s_ in sets for i in s_["ids"]] if sets else [i for i in art.get("ids") or [] if i not in before]
                    pool = self.model_order(ai, self.unread_pool(ai, src), 4)
                    self.open([it["id"] for it in pool[:4]])
                finally:
                    r.plan.extra = 0
                continue
            val = req["value"] if isinstance(req["value"], str) else (req["value"][0] if req["value"] else "")
            if req["op"] in ("is", "eq"):
                val = self.written(ai, req, val)
            figures = Text(Text.of(val)).numbers if req["op"] in ("min", "max", "between") else []
            unit = ((figures[0][1] if figures else "") or Text.of(req.get("unit"))).strip().lower()
            if len(figures) == 1 and unit.isalpha() and self.store_fuses(unit):
                n = figures[0][0]
                val = (str(int(n)) if float(n).is_integer() else str(n)) + unit
            if not re.search(r"[a-z0-9]{2}", val.lower()) or code_shaped(val) or r.finder.unsearchable(val):
                continue
            stated = self.search_terms(ai)["values"][:1]
            words = search_steps({"noun": art["noun"], "heads": noun_heads(art["noun"]), "values": [val] + stated}, None)[0][0]
            lead = [w for v in [val] + stated for w in Text(v).words[:1]]
            q = " ".join(dict.fromkeys(lead + list(words)))
            if not q.strip() or r.finder.state(q) is not None:
                continue
            r.plan.extra = max(0, room - r.plan.room("read"))
            try:
                ids, _, _ = self.line_run(ai, q, force=True)
                self.add_ids(art, ids)
                self.learn_category(ai)
                pool = self.model_order(ai, self.unread_pool(ai, ids), 4)
                self.open([it["id"] for it in pool[:4]])
            finally:
                r.plan.extra = 0
        return True

    def chase_pair(self, art: dict) -> None:
        r = self.r
        ai = art["i"]
        if r.finder.sas():
            return

        def firsts(source_is_stated: bool) -> list[str]:
            return [req["value"] if isinstance(req["value"], str) else req["value"][0] for req in r.reqs(ai)
                    if req["op"] in ("is", "eq") and req["value"] and (req["source"] == "stated") == source_is_stated]
        vals = firsts(False) + firsts(True)
        if len(vals) < 2 or art.get("combined") == vals[:2]:
            return
        art["combined"] = vals[:2]
        n_req = len(r.reqs(ai))
        if any(r.fit(it, ai)[1] == 0 and r.fit(it, ai)[0] >= n_req for it in r.owners(ai)):
            return
        if r.plan.room("read") < 3 or r.t() > GUARD_AT - 40:
            return
        q = " ".join(Text(vals[0]).words[:1] + Text(vals[1]).words[:1] + Text(art["noun"]).words[-1:])
        ids, _, _ = self.line_run(ai, q)
        self.add_ids(art, ids)
        order = {i: n for n, i in enumerate(ids)}
        pool = self.model_order(ai, self.unread_pool(ai, ids), 6, base=lambda it: order.get(it["id"], len(order)))
        self.open([it["id"] for it in pool[:6]])

    def wave2(self) -> None:
        r = self.r
        self.waves = getattr(self, "waves", 0) + 1
        for art in r.articles:
            self.chase_revealed(art)
        for art in r.articles:
            self.chase_pair(art)
        r.chains.walk()
        if r.together and self.waves >= 2 and FIT_EARLY:
            r.judge.fit_solve()
        if r.cheapest:
            r.plan.begin("proof", "proof")
            self.prove_cheapest()
        r.plan.begin("read", "after proof")

    def filter_sweep(self, art: dict, bp: float, pool: dict, cheaper: Callable[[str], bool]) -> None:
        r = self.r
        f = r.tm["filter"]
        ceiling = bp - 1 if float(bp).is_integer() else round(bp - 0.01, 2)
        seen_fit = [r.cat.price(it) for it in r.cat.items.values() if r.cat.category(it) == art["cat"] and r.cat.price(it) is not None
                    and r.ok_basic(it, art["i"]) and not (it["read"] and r.contradicted(it, art["i"]))]
        if seen_fit:
            ceiling = min(ceiling, min(seen_fit))
        args = {**(f.get("args") or {}), f["category"]: art["cat"], f["max_price"]: ceiling}
        pg = r.store.page_arg("filter")
        narrowed = 0
        for _round in range(6):
            if r.plan.room("proof") <= 0 or r.t() > GUARD_AT - 30:
                break
            obs = r.env.send([(f["name"], args)])
            ids, total, cursor = r.cat.rows_of(obs[0] if obs else None)
            self.add_ids(art, ids)
            pool.update({i: r.cat.items[i] for i in ids if cheaper(i)})
            reach = len(ids) * (r.finder.page_limit(max(1, len(ids))) or 3)
            if isinstance(total, int) and ids and total > reach and narrowed < 3 and ceiling > 1:
                narrowed += 1
                ceiling = int(ceiling / 2) if float(ceiling).is_integer() else round(ceiling / 2, 2)
                args = {k: v for k, v in args.items() if k != pg}
                args[f["max_price"]] = ceiling
                continue
            if not cursor or not pg:
                break
            args = {**args, pg: cursor}

    def alt_firsts(self, art: dict, budget: int) -> tuple[list[str], int]:
        r = self.r
        terms = self.search_terms(art["i"])
        alts = [Text(v).words for v in terms.get("alts") or []]
        alts = [a for a in alts if a]
        if not alts or budget <= 0 or r.finder.traits is None:
            return [], 0
        got: list[str] = []
        used = 0
        tried = set(art.get("plan_done") or [])
        for st in search_steps(terms, r.finder.traits):
            if used >= budget:
                break
            if st[1] != 1 or st in tried or not st[0] or st[0][0].startswith("@") or not any(all(w in st[0] for w in a) for a in alts):
                continue
            tried.add(st)
            n0 = r.steps
            ids = self.plan_step(art, st[0], 1)
            used += r.steps - n0
            art["search_steps"] = art.get("search_steps", 0) + (r.steps - n0)
            if ids:
                got += ids
        art["plan_done"] = sorted(tried)
        if got:
            r.notes.append(f"article {art['i']}: alternatives searched before deeper pages ({used} steps)")
        return got, used

    def short_query(self, art: dict, groups: list, share: int) -> Query | None:
        r = self.r
        finder = r.finder
        if finder.price_ordered():
            return None
        noun = Text(art["noun"]).stems[-1:]
        rows = max(1, finder.rows_per_page())
        room = r.plan.room("proof") // max(1, share)
        best: tuple[int, Query] | None = None
        for q in finder.open_queries():
            text = finder.class_text(q)
            if not isinstance(q.total, int) or not finder.carries(text, groups) or (noun and noun[0] not in Text(text).stems):
                continue
            pages = -(-max(0, q.total - q.fetched) // rows)
            if 0 < pages <= room and (best is None or pages < best[0]):
                best = (pages, q)
        return best[1] if best else None

    def dept_share(self, q: Query, art: dict) -> float:
        r = self.r
        cat = art.get("cat")
        if not cat:
            return 1.0
        cats = [r.cat.category(r.cat.items[i]) for i in q.ids if i in r.cat.items and r.cat.category(r.cat.items[i])]
        return (sum(1 for c in cats if c == cat) + 1) / (len(cats) + 2)

    def carrying_open(self, art: dict, groups: list) -> list[Query]:
        finder = self.r.finder
        noun = Text(art["noun"]).stems[-1:]
        return [q for q in finder.open_queries() if finder.carries(finder.class_text(q), groups) and (not noun or noun[0] in Text(finder.class_text(q)).stems)]

    def page_sweep(self, art: dict, groups: list, pool: dict, cheaper: Callable[[str], bool], share: int = 1) -> None:
        r = self.r
        finder = r.finder
        open_qs = self.carrying_open(art, groups)
        open_qs.sort(key=lambda q: (-finder.carries(finder.class_text(q), groups), -self.dept_share(q, art), q.total or 1e9))
        for q in open_qs[:1]:
            budget = min(3, max(0, r.plan.room("proof") - 1) // max(1, share))
            ids, used = self.alt_firsts(art, budget)
            if ids:
                self.add_ids(art, ids)
                pool.update({i: r.cat.items[i] for i in ids if cheaper(i)})
            ids = finder.page_on(q.text, budget - used)
            self.add_ids(art, ids)
            pool.update({i: r.cat.items[i] for i in ids if cheaper(i)})

    def prove_cheapest(self) -> None:
        r = self.r
        for k, art in enumerate(r.articles):
            ai = art["i"]
            if art.get("fit_verified"):
                continue
            if r.finder.sas():
                if self.sas_certified(ai) or r.t() > GUARD_AT - 30:
                    continue
                if r.finder.price_ordered():
                    share = max(1, r.plan.room("proof") // max(1, len(r.articles) - k))
                    self.sorted_open(ai, share)
                    r.reach.price_walk(ai, max(0, r.plan.room("proof") // max(1, len(r.articles) - k)))
                    continue
                self.lattice(ai, max(1, r.plan.room("proof") // max(1, len(r.articles) - k) // 2))
            want = [x for x in r.reqs(ai) if x["op"] in ("is", "eq")]
            cands = r.candidates(ai)
            fits = ([it for it in cands if r.fit(it, ai)[1] == 0 and all(r.cat.mentions(it, x) for x in want)]
                    or [it for it in cands if r.fit(it, ai)[1] == 0] or cands[:1])
            if not fits or r.plan.room("proof") <= 0 or r.t() > GUARD_AT - 30:
                continue
            bp = r.cat.price(fits[0])
            if bp is None or art.get("proved_below") == bp:
                continue
            art["proved_below"] = bp
            groups = r.finder.value_groups(ai)

            def cheaper(i: str) -> bool:
                it = r.cat.items.get(i)
                if not it or it["read"] or not r.ok_basic(it, ai) or r.finder.lacks(it, ai):
                    return False
                p = r.cat.price(it)
                return (1e18 if p is None else p) < bp
            pool = {i: r.cat.items[i] for i in art.get("ids") or [] if cheaper(i)}
            vqs = self.value_queries(ai) if groups and r.finder.match() != "or" and r.store.page_arg("search") and not r.finder.live_sets(ai) else []
            for vq in vqs:
                if (r.plan.room("proof") < 2 or r.t() > GUARD_AT - 40
                        or any(r.finder.stem_set(t) <= r.finder.stem_set(vq) and r.finder.carries(t, groups) >= r.finder.carries(vq, groups)
                               for t in list(r.finder.queries))):
                    continue
                ids, _, _ = self.line_run(ai, vq, force=True)
                self.add_ids(art, ids)
                pool.update({i: r.cat.items[i] for i in ids if cheaper(i)})
                r.notes.append(f"proof: line {ai} value query {vq!r}, {len(ids)} rows")
            f = r.tm.get("filter")
            short = self.short_query(art, groups, len(r.articles) - k) if groups and r.store.page_arg("search") else None
            if short is not None:
                ids = r.finder.page_on(short.text, r.plan.room("proof"))
                self.add_ids(art, ids)
                pool.update({i: r.cat.items[i] for i in ids if cheaper(i)})
                r.notes.append(f"proof: line {ai} paged the value search {short.text!r} to its end, {short.fetched} of {short.total} rows")
            elif groups and r.store.page_arg("search") and not r.finder.price_ordered() and r.finder.sas() and self.carrying_open(art, groups):
                self.page_sweep(art, groups, pool, cheaper, len(r.articles) - k)
            elif f and f.get("max_price") and f.get("category") and art.get("cat"):
                self.filter_sweep(art, bp, pool, cheaper)
            elif groups and r.store.page_arg("search") and not r.finder.price_ordered():
                self.page_sweep(art, groups, pool, cheaper, len(r.articles) - k)

            def key(it: dict) -> tuple:
                p = r.cat.price(it)
                return (-r.finder.value_hits(it, groups), 1e18 if p is None else p)
            ranked = sorted(pool.values(), key=key)
            if groups:
                hit = [it for it in ranked if r.finder.value_hits(it, groups)]
                ranked = hit + [it for it in ranked if it not in hit][:max(0, 6 - len(hit))]
            spot = {it["id"]: n for n, it in enumerate(ranked)}
            share = max(1, r.plan.room(r.phase) // max(1, len(r.articles) - k))
            ranked = self.model_order(ai, ranked[:12], min(12, share), base=lambda it: spot[it["id"]])
            self.open([it["id"] for it in ranked[:min(12, share)]])

    def prove_pick(self, ai: int, k: int = 3) -> bool:
        r = self.r
        art = r.articles[ai]
        cur = r.cat.items.get(art.get("pick") or "")
        bar = r.cat.price(cur) if cur is not None else None
        if not r.cheapest or bar is None or art.get("proved_pick") == bar or self.sas_certified(ai):
            return False
        cat = r.cat.category(cur)
        pool = [it for it in r.cat.items.values() if not it["read"] and r.ok_basic(it, ai) and (not cat or r.cat.category(it) == cat)
                and r.cat.price(it) is not None and r.cat.price(it) < bar and not r.finder.lacks(it, ai)]
        n = min(k, max(0, r.plan.room(r.phase)))
        take = self.model_order(ai, pool, n, base=lambda it: r.cat.price(it))[:n]
        if not take:
            art["proved_pick"] = bar
            return False
        self.open([it["id"] for it in take])
        art["proved_pick"] = bar
        r.notes.append(f"article {ai}: {len(take)} cheaper rows read under the pick")
        return True

    def other_name(self, ai: int) -> bool:
        r = self.r
        art = r.articles[ai]
        if art.get("other_named") or not r.cheapest or self.capped_carrier(ai):
            return False
        art["other_named"] = True
        cur = r.cat.items.get(art.get("pick") or "")
        bar = r.cat.price(cur) if cur is not None else None
        noun = Text(art["noun"]).words[-1:]
        names = []
        if art.get("cat") and Text(art["cat"]).stems != Text(art["noun"]).stems:
            names.append(art["cat"])
        if noun and noun[0] != art["noun"].lower():
            names.append(noun[0])
        if cur is not None:
            names.append(" ".join(Text(r.cat.title(cur)).words[:2]))
        q = next((n for n in names if n and r.finder.state(n) is None), None)
        if not q:
            return False
        ids, _, _ = self.line_run(ai, q)
        self.add_ids(art, ids)
        pool = [it for it in self.unread_pool(ai, ids) if bar is None or (r.cat.price(it) is not None and r.cat.price(it) < bar)]
        pool = self.model_order(ai, pool, 2, base=lambda it: 1e18 if r.cat.price(it) is None else r.cat.price(it))
        self.open([it["id"] for it in pool[:2]])
        r.notes.append(f"article {ai}: other name {q!r}, {len(ids)} rows")
        return True

    def unsearched_value(self) -> tuple[int, str] | None:
        r = self.r
        for later in (False, True):
            for art in [a for a in r.articles if a.get("search_steps", 0) < LINE_SEARCH_STEPS]:
                for req in r.reqs(art["i"]):
                    if req["op"] not in ("is", "eq") or not req["value"]:
                        continue
                    vals = [req["value"]] if isinstance(req["value"], str) else list(req["value"])
                    for v in (vals[1:] if later else vals[:1]):
                        if not isinstance(v, str) or code_shaped(v) or r.finder.unsearchable(v):
                            continue
                        v = self.written(art["i"], req, v)
                        q = " ".join(Text(v).words[:2] + Text(art["noun"]).words[-1:])
                        if q.strip() and r.finder.state(q) is None and not any(Text(q).stems == Text(d).stems for d in r.finder.queries):
                            return art["i"], q
        return None

    def relax_value(self) -> bool:
        r = self.r
        if r.t() > GUARD_AT - 40:
            return False
        for art in r.articles:
            ai = art["i"]
            if any(r.basket.state(it, ai) == "v" for it in r.candidates(ai)):
                continue
            done = art.setdefault("relaxed", [])
            terms = self.search_terms(ai)
            words: list[str] = []
            for req in r.reqs(ai):
                if req["op"] not in ("is", "eq") or not req["value"]:
                    continue
                vals = [req["value"]] if isinstance(req["value"], str) else list(req["value"])
                for v in vals:
                    if not isinstance(v, str) or code_shaped(v) or r.finder.unsearchable(v):
                        continue
                    ws = [w for w in Text(self.written(ai, req, v)).words if w.isalpha() and len(w) > 2]
                    if len(ws) >= 2:
                        words += [w for w in ws if w not in words]
            if len(done) >= 2 or not words:
                continue
            df = r.finder.df_table()
            spot = {w: n for n, w in enumerate(words)}
            words = sorted(words, key=lambda w: (df.get(Text.stem(w), 0), spot[w]))
            for w in words[:2]:
                text = " ".join(search_steps({**terms, "values": [w], "revealed": [], "alts": [], "ints": []}, None)[0][0])
                if not text or text in done or any(r.finder.stem_set(t) == r.finder.stem_set(text) for t in list(r.finder.queries)):
                    continue
                done.append(text)
                ids, total, _ = self.line_run(ai, text, force=True)
                self.add_ids(art, ids)
                coded = r.judge.code_reqs(ai)
                pool = self.unread_pool(ai, ids)
                pool = self.model_order(ai, pool, 1, base=lambda it: (sum(1 for q in coded if r.cat.code_verdict(it, q) != "yes"),
                                                                      r.judge.price_of(it)))
                self.open([it["id"] for it in pool[:1]])
                r.notes.append(f"relaxed value search: line {ai} {text!r}, {len(ids)} rows of {total}")
                return True
        return False

    def split_pick(self, art: dict, q: Query, tried: Any) -> tuple[tuple, str, int] | None:
        r = self.r
        f = r.finder
        items = r.cat.items
        cat = art.get("cat")
        base = min([q.text] + [o.text for o in f.queries.values() if f.same_rows(q, o)], key=lambda t: (len(t.split()), t))
        if len(base.split()) >= 6 or not isinstance(q.total, int) or not q.ids:
            return None
        rows = [items[i] for i in q.ids if i in items]
        have = f.stem_set(f.class_text(q))
        df: collections.Counter = collections.Counter()
        mine: collections.Counter = collections.Counter()
        known: collections.Counter = collections.Counter()
        read = 0
        for it in rows:
            facts = it.get("facts") or {}
            text = " ".join([r.cat.title(it)] + [Text.of(v) for v in facts.values()])
            ws = {w for w in Text(text).words if w.isalpha() and len(w) > 2 and Text.stem(w) not in have}
            df.update(ws)
            if facts:
                read += 1
                known.update(ws)
            if not cat or r.cat.category(it) == cat:
                mine.update(ws)
        size = max(1, q.first)
        jump = bool(r.store.cursor) and q.used and all(isinstance(u, int) and not isinstance(u, bool) for u in q.used)
        best = None
        for w, seen in df.items():
            text = f"{base} {w}"
            n = max(seen, known[w] * len(rows) / read) if read else seen
            if text in tried or f.state(text) is not None or n >= len(rows) or not mine[w]:
                continue
            reach = min(n * q.total / max(1, len(q.ids)), q.fetched)
            start = min((int(min(n, reach - 1)) // size) * size, max(q.used)) if jump else 0
            new = (reach - n) * mine[w] / seen
            pages = max(1, -(-int(reach - start) // size))
            if new <= 0:
                continue
            key = (new / pages, new, w)
            if best is None or key > best[0]:
                best = (key, text, start)
        return best

    def tail_split(self) -> bool:
        r = self.r
        f = r.finder
        pg = r.store.page_arg("search")
        if not pg or r.t() > GUARD_AT - 40:
            return False
        paired = {x for p in r.together for x in p}
        best = None
        for art in r.articles:
            ai = art["i"]
            if ai in paired or not r.reach.lacking(ai):
                continue
            tried = set(art.get("tail_tried") or [])
            spent = [q for q in dict.fromkeys(f.queries.values())
                     if isinstance(q.total, int) and q.cursor in (None, "", False) and q.pages >= 2 and q.fetched < q.total
                     and f.lead(q) is q and q.text not in tried and r.reach.fit(q, ai) is not None]
            if not spent:
                continue
            def unseen(x: Query) -> float:
                last = max(x.ids) if x.ids else ""
                mine_ = f.stem_set(x.text)
                tail = {i for o in f.queries.values() if mine_ < f.stem_set(o.text) for i in o.ids if i > last}
                return x.total - x.fetched - len(tail)
            q = max(spent, key=lambda x: (r.reach.fit(x, ai)[:2], unseen(x)))
            got = self.split_pick(art, q, tried)
            if got is not None and (best is None or got[0] > best[0]):
                best = (got[0], art, got[1], got[2])
        if best is None:
            return False
        _, art, text, start = best
        art.setdefault("tail_tried", []).append(text)
        ai = art["i"]
        ids, total, _ = self.line_run(ai, text, force=True) if not start else f.run(text, {pg: start})
        self.add_ids(art, ids)
        r.notes.append(f"tail split: line {ai} {text!r} from row {start}, {len(ids)} rows of {total}")
        pool = self.model_order(ai, self.cheaper_unread(ai, ids), 1, base=lambda it: r.judge.price_of(it))
        self.open([it["id"] for it in pool[:1]])
        return True

    def sink(self) -> None:
        r = self.r
        r.plan.begin("sink", "sink")
        fed = False
        for _ in range(FIT_FEEDS):
            gap = self.fit_gap()
            if gap is None or r.plan.fit_room() <= 4 or r.t() > GUARD_AT - 30:
                break
            n0 = r.steps
            self.fit_feed(gap)
            fed = fed or r.steps > n0
        if Text.of(r.spec.get("priority")) == "best":
            for art in r.articles:
                room = r.plan.room("sink") - 2
                order = {i: n for n, i in enumerate(art.get("ids") or [])}
                pool = self.model_order(art["i"], self.unread_pool(art["i"], art.get("ids") or []), room,
                                        base=lambda it: order.get(it["id"], len(order))) if room > 0 else []
                if pool and room > 0:
                    self.open([it["id"] for it in pool[:room]])
                    fed = True
        for _ in range(12):
            if r.plan.room("sink") <= 2 or r.t() > GUARD_AT - 30:
                break
            n0 = r.steps
            weak = [ai for ai in r.basket.weak_lines() if ai not in {x for p in r.together for x in p}]
            if weak and r.basket.verify_loop(1, note=False, keep=1, only=weak):
                fed = True
            elif r.chains.audit():
                fed = True
            elif any(it.get("fit_open") and not all(fit_norm(c) in r.cat.pointers_done for c in it["fit_open"])
                     for a in r.articles for it in r.candidates(a["i"])[:6]):
                self.resolve_pointers(1)
                fed = True
            elif r.cheapest and any(self.prove_pick(a["i"], 3) or r.steps > n0 for a in r.articles):
                fed = True
            elif any(not (r.cat.items.get(a.get("pick") or "") or {}).get("read") for a in r.articles if a.get("pick") in r.cat.items):
                for a in r.articles:
                    it = r.cat.items.get(a.get("pick") or "")
                    if it is not None and not it["read"]:
                        self.open([it["id"]])
                fed = True
            elif (gap := self.fit_gap()) is not None:
                self.fit_feed(gap)
                fed = True
            elif self.sas_leftover():
                fed = True
            elif self.relax_value():
                fed = True
            elif (found := self.unsearched_value()) is not None:
                ai, q = found
                ids, _, _ = self.line_run(ai, q)
                self.add_ids(r.articles[ai], ids)
                pool = self.model_order(ai, self.unread_pool(ai, ids), 1)
                self.open([it["id"] for it in pool[:1]])
                fed = True
            elif r.cheapest and r.plan.room("sink") >= 14 + 2 * len(r.articles) and any(self.other_name(a["i"]) for a in r.articles):
                fed = True
            else:
                got = r.reach.page()
                if got is not None:
                    ai, ids = got
                    pool = self.model_order(ai, self.cheaper_unread(ai, ids), 1, base=lambda it: r.judge.price_of(it)) if r.cheapest else \
                        self.model_order(ai, self.unread_pool(ai, ids), 1)
                    self.open([it["id"] for it in pool[:1]])
                    fed = True
                    if r.steps == n0:
                        break
                    continue
                if self.tail_split():
                    fed = True
                    if r.steps == n0:
                        break
                    continue
                big = max((q for q in r.finder.open_queries()), key=lambda q: q.total or 0, default=None)
                if big is None:
                    break
                ids = r.finder.page_on(big.text, 1)
                if not ids:
                    break
                for a in r.articles:
                    pool = self.model_order(a["i"], self.unread_pool(a["i"], ids), 1)
                    self.open([it["id"] for it in pool[:1]])
                fed = True
            if r.steps == n0:
                break
        if fed and r.t() < GUARD_AT - 25:
            r.judge.decide(strong=False)
            r.notes.append(f"sink: {r.plan.used['sink']} steps")

    def refind(self, ai: int) -> bool:
        r = self.r
        art = r.articles[ai]
        stated = [x for x in r.reqs(ai) if x["source"] == "stated" and x["op"] in ("is", "eq") and x["value"] and not code_shaped(Catalog.wanted_text(x))
                  and not r.finder.unsearchable(Catalog.wanted_text(x))]
        revealed = [x for x in r.reqs(ai) if x["source"] != "stated" and x["op"] in ("is", "eq") and x["value"] and not code_shaped(Catalog.wanted_text(x))
                    and not r.finder.unsearchable(Catalog.wanted_text(x))]
        first = lambda x: Text.of(x["value"] if isinstance(x["value"], str) else x["value"][0])
        noun = Text(art["noun"]).words[-1:]
        tries = []
        if stated and revealed:
            tries.append(" ".join(Text(first(stated[0])).words[:5] + Text(first(revealed[0])).words[:3] + noun))
        for x in stated[:1] + revealed[:1]:
            tries.append(" ".join(Text(first(x)).words[:5] + noun))
        tries.append(" ".join(noun))
        for q in tries:
            if not q.strip() or r.finder.state(q) is not None or r.plan.slack() < 2:
                continue
            ids, _, _ = self.line_run(ai, q, force=True)
            self.add_ids(art, ids)
            pool = self.unread_pool(ai, ids)
            if not pool:
                continue
            k = min(3, max(1, r.plan.slack() - 1))
            pool = self.model_order(ai, pool, k, base=lambda it: 1e18 if r.cat.price(it) is None else r.cat.price(it), force=True)
            self.open([it["id"] for it in pool[:k]], force=True)
            r.notes.append(f"refind article {ai}: {q!r}, {len(ids)} rows")
            return True
        return False

    def cover(self) -> None:
        r = self.r
        for art in r.articles:
            ai = art["i"]
            have = len(r.owners(ai))
            if have >= 2 or r.t() > GUARD_AT - 60:
                continue
            if r.steps_left() - r.plan.finish_need(minimal=True) - r.plan.repair_reserve() < 3:
                break
            pool = self.unread_pool(ai, art.get("ids") or [])
            if not pool and not have:
                for q in r.search_phrases(art)[::-1]:
                    if r.finder.state(q) is not None:
                        continue
                    ids, _, _ = self.line_run(ai, q, force=True)
                    self.add_ids(art, ids)
                    if ids:
                        break
                pool = self.unread_pool(ai, art.get("ids") or [])
            if not pool and not art.get("capped"):
                pool = self.capped_pool(art)
            pool = self.model_order(ai, pool, 2 - have, force=True)
            if pool:
                self.open([it["id"] for it in pool[:2 - have]], force=True)
                r.notes.append(f"cover: article {ai} read up to two listings before its questions")

    def capped_pool(self, art: dict) -> list[dict]:
        r = self.r
        f = r.tm.get("filter")
        art["capped"] = True
        if not (f and f.get("category") and f.get("max_price") and art.get("cat") and r.budget):
            return []
        ceiling = r.budget / max(1, art["qty"])
        ceiling = int(ceiling) if float(ceiling).is_integer() else round(ceiling, 2)
        obs = r.env.send([(f["name"], {**(f.get("args") or {}), f["category"]: art["cat"], f["max_price"]: ceiling})])
        ids, _, _ = r.cat.rows_of(obs[0] if obs else None)
        self.add_ids(art, ids)
        r.notes.append(f"cover: article {art['i']} capped at {ceiling}, {len(ids)} rows")
        return self.unread_pool(art["i"], ids)

    def resolve_pointers(self, k: int = 3) -> None:
        r = self.r
        cat = r.cat
        pending: list[str] = []
        for art in r.articles:
            for it in r.candidates(art["i"])[:6]:
                for code in it.get("fit_open") or []:
                    if code not in pending and fit_norm(code) not in cat.pointers_done:
                        pending.append(code)
        two = bool(r.tm.get("specs") and r.tm.get("detail"))
        for code in pending[:k]:
            if r.plan.slack() < 3 or r.t() > GUARD_AT - 30:
                break
            cat.pointers_done.add(fit_norm(code))
            r.finder.run(code)
            carriers = [it["id"] for it in cat.items.values() if cat.title_carries(it, code) and not (it.get("paged") or it.get("read"))]
            if carriers:
                self.open(carriers[:2], what="state" if two else "facts", force=True)
            r.notes.append(f"pointer {code}: {len(carriers)} pages")

    def miss_phrases(self, art: dict, cands: list[dict]) -> list[str]:
        r = self.r
        phrases: list[str] = []
        noun = Text(art["noun"]).words[-2:]
        for req in r.reqs(art["i"]):
            if req["op"] not in ("is", "eq", "min", "max", "between") or not req["value"]:
                continue
            if any(r.cat.verdict(it, req) == "yes" for it in cands):
                continue
            v = req["value"] if isinstance(req["value"], str) else req["value"][0]
            if Text(v).numbers and len(Text(v).words) <= 2:
                continue
            head = Text(v).words[:2]
            phrases += [" ".join(head + noun), " ".join(head + noun[-1:])]
        return phrases

    def expand(self) -> bool:
        r = self.r
        finder = r.finder
        before = sum(1 for it in r.cat.items.values() if it["read"])
        for art in r.articles:
            ai = art["i"]
            if not art.get("judged_none") or r.plan.room("read") < 2 or r.t() > GUARD_AT - 45:
                continue
            if r.judge.code_reqs(ai) and not art.get("fit_dry"):
                while r.plan.room("read") >= 5 and r.t() < GUARD_AT - 45 and self.fit_open(ai):
                    n0 = r.steps
                    self.fit_feed(ai)
                    if r.steps == n0:
                        break
                if not art.get("fit_dry"):
                    continue
            done = set(art.get("done") or [])
            groups = finder.value_groups(ai)
            fresh = [p for p in dict.fromkeys(self.miss_phrases(art, r.candidates(ai))) if p and p.lower() not in done][:3]
            for ph in fresh:
                if r.plan.room("read") < 2:
                    break
                done.add(ph.lower())
                ids, _, _ = self.line_run(ai, ph)
                self.add_ids(art, ids)
                pool = self.model_order(ai, self.unread_pool(ai, ids), 6, base=finder.order_key(groups))
                self.open([it["id"] for it in pool[:6]])
            art["done"] = sorted(done)
            if fresh:
                continue
            noun = Text(art["noun"]).stems[-1:]
            open_qs = [q for q in finder.open_queries() if not noun or noun[0] in Text(q.text).stems]
            open_qs.sort(key=lambda q: (-finder.carries(q.text, groups), q.total or 1e9))
            got: list[str] = []
            if open_qs:
                got += self.alt_firsts(art, min(2, max(0, r.plan.room("read") - 2)))[0]
            for q in open_qs[:2]:
                got += finder.page_on(q.text, 1 if len(open_qs) > 1 else 2)
            self.add_ids(art, got)
            pool = self.model_order(ai, self.unread_pool(ai, art["ids"]), 6, base=finder.order_key(groups))
            self.open([it["id"] for it in pool[:6]])
        return sum(1 for it in r.cat.items.values() if it["read"]) > before

    def mop_up(self) -> None:
        r = self.r
        r.plan.done.add("proof")
        r.plan.done.add("ask")
        r.plan.begin("read", "mop-up")
        if r.plan.room("read") < 1 or r.t() > GUARD_AT - 30:
            return
        before = {a["i"]: a.get("pick") for a in r.articles}
        fresh = 0
        for art in r.articles:
            ai = art["i"]
            if r.plan.room("read") < 1:
                break
            cur = r.cat.items.get(art.get("pick") or "")
            if cur is not None and r.fit(cur, ai)[1] == 0 and r.fit(cur, ai)[2] == 0 and not r.cheapest:
                continue
            pool = self.unread_pool(ai, art.get("ids") or [])
            if r.cheapest and cur is not None and r.cat.price(cur) is not None:
                bar = r.cat.price(cur)
                pool = [it for it in pool if (1e18 if r.cat.price(it) is None else r.cat.price(it)) < bar]
            share = max(1, r.plan.room("read") // max(1, len(r.articles)))
            take = self.model_order(ai, pool, share)[:share]
            if take:
                n0 = sum(1 for it in r.cat.items.values() if it["read"])
                self.open([it["id"] for it in take])
                fresh += sum(1 for it in r.cat.items.values() if it["read"]) - n0
        if not fresh:
            return
        better = False
        for art in r.articles:
            ai = art["i"]
            cur = r.cat.items.get(art.get("pick") or "")
            fits = [it for it in r.candidates(ai) if r.fit(it, ai)[1] == 0 and it.get("read_step", -1) >= r.steps - fresh]
            if not fits:
                continue
            if cur is None or r.fit(cur, ai)[1] > 0 or (r.cheapest and (r.cat.price(fits[0]) or 1e18) < (r.cat.price(cur) or 1e18)):
                better = True
        if better and r.t() < GUARD_AT - 25:
            r.judge.decide()
            r.notes.append(f"mop-up: {fresh} new reads, picks {before} -> {[a.get('pick') for a in r.articles]}")

    def bounds_clash(self) -> bool:
        r = self.r
        for art in r.articles:
            lows: dict[str, float] = {}
            highs: dict[str, float] = {}
            for x in r.reqs(art["i"]):
                if x["source"] != "stated" or x["op"] not in ("min", "max"):
                    continue
                nums = Text(Catalog.wanted_text(x)).numbers
                if not nums:
                    continue
                key = Text.key(x["prop"])
                if x["op"] == "min":
                    lows[key] = max(lows.get(key, -1e18), nums[0][0])
                else:
                    highs[key] = min(highs.get(key, 1e18), nums[0][0])
            if any(lows[k] > highs[k] + 1e-9 for k in lows if k in highs):
                return True
        return False

    def over_cap(self) -> bool:
        r = self.r
        if not r.budget:
            return False
        total = 0.0
        for art in r.articles:
            ai = art["i"]
            if art.get("search_steps", 0) < LINE_SEARCH_STEPS:
                return False
            budget, r.budget = r.budget, None
            try:
                pool = r.candidates(ai)
            finally:
                r.budget = budget
            prices = [r.cat.price(it) for it in pool if r.judge.tier_of(it, ai) <= 1 and r.cat.price(it) is not None]
            if not prices:
                return False
            total += min(prices) * art["qty"]
        return total > r.budget + 1e-9

    def should_decline(self) -> bool:
        r = self.r
        if not (r.decline_ok or r.offer_decline) or not r.tm.get("decline"):
            return False
        if r.basket.over_budget():
            r.notes.append("decline: no basket of verified listings fits the budget, the over-budget basket is not ordered")
            return True
        if self.bounds_clash() or self.over_cap():
            r.notes.append("decline: stated bounds clash or the cheapest valid set exceeds the cap")
            return True
        pending: list[int] = []
        for art in r.articles:
            ai = art["i"]

            def proven(it: dict) -> bool:
                return all(r.cat.verdict(it, x) == "yes" for x in r.reqs(ai))
            pick = r.cat.items.get(art.get("pick") or "")
            lost = (art.get("judged_none") and pick is not None and r.contradicted(pick, ai)
                    and any(r.fit(it, ai)[1] == 0 and not r.ok_basic(it, ai) for it in r.owners(ai)))
            if not (art.get("no_pick") or lost) or any(proven(it) for it in r.candidates(ai)):
                continue
            if any(not r.contradicted(it, ai) for it in r.candidates(ai)):
                continue
            noun = Text(art["noun"]).words[-1:]
            for req in r.reqs(ai):
                if req["op"] in ("not", "ref") or r.plan.room("read") < 2:
                    continue
                if any(r.cat.verdict(it, req) == "yes" for it in r.owners(ai)):
                    continue
                val = req["value"] if isinstance(req["value"], str) else " ".join(req["value"][:1])
                for q in (val, " ".join([val] + noun)):
                    ids, _, _ = self.line_run(ai, q[:60], force=True)
                    pool = [r.cat.items[i] for i in ids if i in r.cat.items and r.ok_basic(r.cat.items[i], ai)]
                    self.open([it["id"] for it in pool[:4]])
            if any(proven(it) or not r.contradicted(it, ai) for it in r.candidates(ai)):
                continue
            unsearched = []
            for req in r.reqs(ai):
                if req["op"] in ("not", "ref") or req["source"] != "stated" or any(r.cat.verdict(it, req) == "yes" for it in r.owners(ai)):
                    continue
                val = req["value"] if isinstance(req["value"], str) else " ".join(Text.of(x) for x in req["value"][:1])
                if all(r.finder.state(q[:60]) is None for q in (val, " ".join([val] + noun))):
                    unsearched.append(req["prop"])
            if unsearched:
                r.notes.append(f"decline: line {ai} has no fitting row but {', '.join(unsearched)} was never searched; ordering")
                continue
            r.notes.append(f"decline check: line {ai} has no fitting row after searching every value")
            pending.append(ai)
        if pending:
            r.judge.weak = True
            try:
                r.basket.judge_all(pending)
            finally:
                r.judge.weak = False
            r.judge.last_look(only=pending)
        return self.proven_empty()

    def proven_empty(self) -> bool:
        r = self.r
        if r.msgs_left() > 0 and not all(a.get("asked") for a in r.articles):
            return False
        for art in r.articles:
            ai = art["i"]
            if art.get("look") != "none":
                continue
            cands = r.candidates(ai)
            if not cands or not all(r.judge.judged_no(it, ai) for it in cands):
                continue
            rows = [it for it in r.basket.table(ai) if not it.get("read") and r.basket.state(it, ai) == "u" and r.basket.chance(it, ai) >= 0.5]
            if rows and r.plan.slack() >= 2:
                r.notes.append(f"decline proof: line {ai} still has {len(rows)} likely unopened rows")
                continue
            if self.rescue_line(ai):
                continue
            if not r.summary.get("decline") and self.unproven(ai):
                continue
            r.notes.append(f"decline: line {ai} has {len(cands)} opened listings, each judged failing, the final check found none, likely rows spent")
            return True
        return False

    def unproven(self, ai: int) -> bool:
        r = self.r
        cands = r.candidates(ai)
        said = {it["id"]: [r.cat.read_verdict(it, q) for q in r.reqs(ai)] for it in cands}
        if not cands or all("no" in v for v in said.values()):
            return False
        art = r.articles[ai]
        best = min(cands, key=lambda it: (said[it["id"]].count("no"), -said[it["id"]].count("yes")))

        def cost(a: dict, pid: Any) -> float:
            it = r.cat.items.get(pid or "")
            return ((r.cat.price(it) if it is not None else 0) or 0) * max(1, int(a.get("qty") or 1))
        total = sum(cost(a, a.get("pick")) for a in r.articles) - cost(art, art.get("pick")) + cost(art, best["id"])
        if best["id"] != art.get("pick") and (not r.budget or total <= r.budget + 1e-9):
            art["backup"] = [x for x in [art.get("pick")] + list(art.get("backup") or []) if x and x != best["id"]]
            art["pick"] = best["id"]
        r.notes.append(f"no decline: line {ai} has a listing its page does not contradict; ordering {art.get('pick')}")
        return True

    def rescue_line(self, ai: int) -> bool:
        r = self.r
        art = r.articles[ai]
        if art.get("rescued") or r.t() > GUARD_AT - 35 or r.finished:
            return bool(self.rescue_live(ai, set(), judge=False))
        art["rescued"] = True
        plan = r.plan
        phase, start = r.phase, r.steps
        base = {it["id"] for it in r.owners(ai)}
        tried: list[str] = []
        asked = False
        plan.endgame = True
        r.phase = "sink"
        try:
            queue, wants, broad = self.rescue_queries(ai)
            while plan.slack() >= 1 and r.t() < GUARD_AT - 35 and not r.finished:
                if self.rescue_live(ai, base):
                    break
                n0 = r.steps
                hot, cold = self.rescue_rows(ai, wants)
                q = next((x for x in queue if x not in tried and r.finder.state(x) is None), None)
                if q is None and not hot and not asked:
                    asked = True
                    queue += self.rescue_phrases(ai) + broad
                    q = next((x for x in queue if x not in tried and r.finder.state(x) is None), None)
                if hot:
                    self.open([hot[0]["id"]], force=True)
                elif q is not None:
                    tried.append(q)
                    ids, _, _ = self.line_run(ai, q, force=True)
                    self.add_ids(art, ids)
                    got = set(ids)
                    new = [it for it in sum(self.rescue_rows(ai, wants), []) if it["id"] in got]
                    if new and plan.slack() >= 1:
                        self.open([new[0]["id"]], force=True)
                elif cold:
                    self.open([cold[0]["id"]], force=True)
                else:
                    pages = self.rescue_pages(ai)
                    if not pages:
                        break
                    self.add_ids(art, pages)
                if r.steps == n0:
                    break
            live = self.rescue_live(ai, base)
        finally:
            plan.endgame = False
            r.phase = phase
        r.notes.append(f"rescue line {ai}: {r.steps - start} steps, queries {tried}, new reads "
                       f"{sum(1 for it in r.owners(ai) if it['id'] not in base)}, live {[it['id'] for it in live[:3]]}")
        if not live:
            return False
        old = art.get("pick")
        art["pick"] = live[0]["id"]
        art["backup"] = [x for x in ([old] if old else []) + list(art.get("backup") or []) if x and x != live[0]["id"]]
        art["look"] = "rescued"
        return True

    def rescue_live(self, ai: int, base: set[str], judge: bool = True) -> list[dict]:
        r = self.r
        cands = [it for it in r.candidates(ai) if it["id"] not in base and not r.judge.judged_no(it, ai) and r.fit(it, ai)[1] == 0]
        sure = lambda it: r.judge.line_level(it, ai) == "exact" or all(r.cat.verdict(it, x) == "yes" for x in r.reqs(ai))
        live = [it for it in cands if sure(it)]
        if live or not judge:
            return live
        fresh = [it for it in cands if r.judge.line_level(it, ai) is None][:2]
        if fresh and r.judge.can_judge(len(fresh)):
            r.judge.judge_batch([(it, ai) for it in fresh])
        return [it for it in cands if sure(it)]

    def rescue_rows(self, ai: int, wants: list[list[str]]) -> tuple[list[dict], list[dict]]:
        r = self.r
        art = r.articles[ai]
        ids = list(dict.fromkeys((art.get("ids") or []) + [it["id"] for it in r.candidates(ai, read_only=False) if not it["read"]]))
        pool = [it for it in self.unread_pool(ai, ids, gate=False)
                if it.get("tries", 0) < READ_TRIES and not any(r.cat.verdict(it, q) == "no" for q in r.reqs(ai))]
        price = lambda it: 1e18 if r.cat.price(it) is None else r.cat.price(it)
        hits = {it["id"]: sum(1 for w in wants if Text.run_in(w, Text(r.cat.title(it)).stems)) for it in pool}
        ev = self.query_evidence(ai)
        pool.sort(key=lambda it: (-ev.get(it["id"], 0), -hits[it["id"]], price(it), it["id"]))
        full = max(1, len([x for x in r.reqs(ai) if x["op"] in ("is", "eq", "min", "max")]))
        return ([it for it in pool if hits[it["id"]] > 0 or ev.get(it["id"], 0) >= full],
                [it for it in pool if hits[it["id"]] == 0 and ev.get(it["id"], 0) < full])

    def query_evidence(self, ai: int) -> dict[str, int]:
        r = self.r
        groups = []
        for req in r.reqs(ai):
            if req["op"] not in ("is", "eq", "min", "max"):
                continue
            alts = [x for x in (Text(Text.of(v)).stems for v in Catalog.wanted_of(req)) if x]
            if alts:
                groups.append(alts)
        out: dict[str, int] = {}
        if not groups:
            return out
        for q in r.finder.queries.values():
            have = set(Text(q.text).stems)
            k = sum(1 for alts in groups if any(set(a) <= have for a in alts))
            if k:
                for i in q.ids:
                    if out.get(i, 0) < k:
                        out[i] = k
        return out

    def rescue_queries(self, ai: int) -> tuple[list[str], list[list[str]], list[str]]:
        r = self.r
        art = r.articles[ai]
        noun = Text(art["noun"]).words[-1:]
        owners = r.owners(ai)
        reqs = [x for x in r.reqs(ai) if x["op"] in ("is", "eq", "not")]
        yes = [sum(1 for it in owners if r.cat.verdict(it, x) == "yes") for x in reqs]
        order = sorted(range(len(reqs)), key=lambda k: (yes[k], k))
        texts: list[list[list[str]]] = []
        for k in order:
            x = reqs[k]
            vals = self.other_values(ai, x) if x["op"] == "not" else Catalog.wanted_of(x)[:2]
            texts.append([Text(Text.of(v)).words[:4] for v in vals if not code_shaped(Text.of(v)) and Text(Text.of(v)).words])
        firsts = [t[0] for t in texts if t]
        out: list[list[str]] = []
        if len(firsts) >= 2:
            out.append(firsts[0] + firsts[1] + noun)
        out += [w + noun for t in texts for w in t] + [w[:1] + noun for t in texts for w in t if len(w) > 1] + [w for t in texts for w in t]
        wants = [[x for x in Text(" ".join(w)).stems if not x.replace(".", "").isdigit()] for w in (texts[0] if texts else [])]
        broad = [" ".join(Text(x).words)[:60].strip() for x in (art["noun"], Text.of(art.get("cat")))]
        return [q for q in dict.fromkeys(" ".join(w)[:60].strip() for w in out) if q], [w for w in wants if w], [q for q in broad if q]

    def other_values(self, ai: int, req: dict) -> list[str]:
        r = self.r
        bad = set(Text(Catalog.wanted_text(req)).stems)
        seen: collections.Counter = collections.Counter()
        for it in r.cat.items.values():
            if it.get("read") and r.cat.verdict(it, req) == "yes":
                for _, v in r.cat.fact_values(it, req["prop"]):
                    if v.strip() and not set(Text(v).stems) <= bad:
                        seen[v.strip().lower()] += 1
        return [v for v, _ in seen.most_common(RESCUE_ALTS)]

    def rescue_phrases(self, ai: int) -> list[str]:
        r = self.r
        if r.t() > GUARD_AT - 45:
            return []
        art = r.articles[ai]
        owners = r.owners(ai)
        reqs = r.reqs(ai)
        if not reqs:
            return []
        fails = [f"- {r.judge.req_text(x)}: {sum(1 for it in owners if r.cat.verdict(it, x) == 'no' or r.judge.level(it, x) == 'no')} "
                 f"of {len(owners)} read listings fail it" for x in reqs]
        user = (f"PRODUCT LINE: {art['noun']}\nREQUIREMENTS AND HOW THE LISTINGS READ SO FAR DID:\n" + "\n".join(fails)
                + f"\nSuggest up to {RESCUE_ALTS} short store search phrases (2 to 5 words each) likely to find a listing of this "
                  "product that meets every requirement. Use words a listing title would carry; for a requirement that excludes a "
                  "value, name an allowed value instead; for a number limit, name a product style that usually meets it.")
        got = r.model_json("read", RESCUE_SYS, user, max_tokens=150, deadline=GUARD_AT - 40)
        out = []
        for v in (got.get("phrases") if isinstance(got, dict) else None) or []:
            q = " ".join(Text(Text.of(v)).words[:6])[:60].strip()
            if q and q not in out:
                out.append(q)
        r.notes.append(f"rescue line {ai}: model phrases {out}")
        return out[:RESCUE_ALTS]

    def rescue_pages(self, ai: int) -> list[str]:
        r = self.r
        finder = r.finder
        groups = finder.value_groups(ai)
        noun = Text(r.articles[ai]["noun"]).stems[-1:]
        qs = [q for q in finder.open_queries() if not noun or noun[0] in Text(q.text).stems]
        qs.sort(key=lambda q: (-finder.carries(q.text, groups), q.total or 1e9))
        for q in qs:
            got = finder.page_on(q.text, 1)
            if got:
                return got
        return []


def words_in(needle: list[str], hay: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def fact_rows(facts: list[dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for f in facts:
        row: dict[str, str] = {}
        for k, v in f.items():
            key, val = Text.key(k), " ".join(Text.of(v).lower().split())[:80]
            if key and val and key not in row:
                row[key] = val
        rows.append(row)
    return rows


def same_reading(rows: list[dict[str, str]], a: str, b: str) -> bool:
    both = [row for row in rows if a in row and b in row]
    if not both:
        return True
    same = 0
    for row in both:
        wa, wb = set(Text(row[a]).words), set(Text(row[b]).words)
        same += 1 if wa <= wb or wb <= wa else 0
    return 2 * same >= len(both)


def alike_values(rows: list[dict[str, str]], a: str, b: str) -> bool:
    va = {row[a] for row in rows if a in row}
    vb = {row[b] for row in rows if b in row}
    return bool(va and vb) and len(va & vb) >= VALUE_TWIN * len(va | vb)


def spelling_of(ks: list[str], os_: list[str]) -> bool:
    if not ks or len(os_) < len(ks) or not set(ks) <= set(os_):
        return False
    return ks != os_ if len(os_) == len(ks) else os_[-1] == ks[-1]


def spelling_heads(rows: list[dict[str, str]]) -> dict[str, str]:
    cover = collections.Counter(k for row in rows for k in row)
    keys = sorted(cover)
    stems = {k: Text(k).stems for k in keys}
    outer: dict[str, set[str]] = {}
    for k in keys:
        outer[k] = {o for o in keys if o != k and spelling_of(stems[k], stems[o]) and same_reading(rows, k, o)}
    up: dict[str, str] = {}
    for k in keys:
        tops = sorted(o for o in outer[k] if not outer[o] & outer[k])
        up[k] = tops[0] if len(tops) == 1 and (k not in outer[tops[0]] or tops[0] < k) else k
    root: dict[str, str] = {}
    for k in keys:
        x, seen = k, {k}
        while up[x] != x and up[x] not in seen:
            seen.add(up[x])
            x = up[x]
        root[k] = x
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(root[k], []).append(k)
    head: dict[str, str] = {}
    for members in groups.values():
        best = sorted(members, key=lambda k: (-len(stems[k]), -cover[k], k))[0]
        for k in members:
            head[k] = best
    return head


def key_table(rows: list[dict[str, str]], notes: list[str], request: str) -> list[dict]:
    n = len(rows)
    if not n:
        return []
    heads = spelling_heads(rows)
    merged: list[dict[str, str]] = []
    for row in rows:
        m: dict[str, str] = {}
        for k in sorted(row, key=lambda k: (heads[k] != k, k)):
            m.setdefault(heads[k], row[k])
        merged.append(m)
    cover = collections.Counter(k for row in merged for k in row)
    least = 1 if n < 2 else max(2.0, KEY_MIN_SHARE * n)
    req_words, req_stems = Text(request).words, set(Text(request).stems)
    note_words = [Text(x).words for x in notes]
    noted = [i for i, w in enumerate(note_words) if w]
    weights = KEY_W20 if n >= KEY_W20_PAGES else KEY_W10
    out: list[dict] = []
    for k in sorted(cover):
        if cover[k] < least or len(k.split()) > KEY_WORDS:
            continue
        vals = [row[k] for row in merged if k in row]
        seen = collections.Counter(vals)
        if len(vals) >= VARIANT_MIN and len(seen) >= VARIANT_SPREAD * len(vals) and \
                sum(1 for v in vals if VARIANT_RX.search(v)) >= VARIANT_SHARE * len(vals):
            continue
        kw = Text(k).words
        named = 1.0 if all(s in req_stems for s in Text(k).stems) else 0.0
        valreq = 1.0 if any(len(v) >= 3 and not NUMERIC_RX.fullmatch(v) and words_in(Text(v).words, req_words) for v in seen) else 0.0
        vnote = sum(1 for i in noted if k in merged[i] and words_in(Text(merged[i][k]).words, note_words[i])) / max(1, len(noted))
        knote = 1.0 if any(words_in(kw, note_words[i]) for i in noted) else 0.0
        feats = [len(vals) / n, 0.0, named, valreq, 1.0 if len(seen) == 1 else 0.0, len(seen) / len(vals),
                 max(seen.values()) / len(vals), vnote, knote]
        ex = [v[:30] for v, _ in sorted(seen.items(), key=lambda x: (-x[1], x[0]))[:4]]
        out.append({"key": k, "f": feats, "ex": ex, "pages": len(vals),
                    "spellings": sorted(x for x in heads if heads[x] == k and x != k)})
    for rank, m in enumerate(sorted(out, key=lambda m: (-m["f"][0], m["key"]))):
        m["f"][1] = rank / max(1, len(out) - 1)
    for m in out:
        m["S"] = sum(w * x for w, x in zip(weights, m["f"]))
    out.sort(key=lambda m: (-m["S"], m["key"]))
    return out


def story_order(tab: list[dict], order: list[str]) -> list[tuple[float, str]]:
    pos = {k: i for i, k in enumerate(order)}
    tail = len(order) + 2
    scored = [(m["S"] - (math.log(1 + pos.get(m["key"], tail)) if order else 0.0), m["key"]) for m in tab]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


def rank_prior(i: int) -> float:
    return RANK_PRIOR[i] if i < len(RANK_PRIOR) else RANK_TAIL


def key_belief(keys: list[str], miss: dict[str, float], gone: set[str]) -> list[tuple[str, float]]:
    raw = [(k, rank_prior(i) * (0.0 if k in gone else miss.get(k, 1.0))) for i, k in enumerate(keys)]
    rest = max(0.0, 1.0 - sum(rank_prior(i) for i in range(len(keys))))
    total = sum(p for _, p in raw) + rest
    if total <= 0:
        return []
    out = [(k, p / total) for k, p in raw if p > 0]
    out.sort(key=lambda x: -x[1])
    return out


def kin_rate(asked: str, other: str) -> float:
    a, o = set(Text.key_stems(asked)), set(Text.key_stems(other))
    if not a or not o or a == o:
        return 0.0
    if o < a:
        return KIN_NARROW
    return KIN_BROAD if a < o else 0.0


def after_miss(pa: float, probs: list[float]) -> float:
    q = min(1.0, sum(p * SINGLE_RECALL for p in probs))
    stay = pa * (1 - q)
    return stay / max(1e-9, stay + 1 - pa)


class Asker:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.story: dict[int, list[str]] | None = None
        self.hints: dict[int, str] = {}
        self.tables: dict[int, list[dict]] = {}
        self.pa: dict[int, float] = {}
        self.miss: dict[int, dict[str, float]] = {}
        self.gone: dict[int, set[str]] = {}
        self.single = True
        self.single_order: dict[int, list[str]] | None = None
        self.sent: set[str] = set()

    def total_left(self) -> int:
        r = self.r
        cap = DEFAULT_MSGS if r.msg_cap is None else r.msg_cap
        return max(0, cap - r.msgs)

    def left(self) -> int:
        r = self.r
        reserved = 1 if r.confirm and not r.approved_cart else 0
        return max(0, self.total_left() - reserved)

    def revealed(self, ai: int) -> int:
        heard = (x for x in self.r.ledger if x["source"] == "revealed" or x.get("heard"))
        return sum(1 for x in heard if x["a"] == ai)

    def may_ask(self) -> bool:
        r = self.r
        return self.left() > 0 and r.t() < ASK_BY and r.plan.room("ask") > 0

    def page_set(self, ai: int) -> tuple[list[dict[str, str]], list[str]]:
        r = self.r
        pages = r.owners(ai)
        rows = fact_rows([it["facts"] for it in pages])
        notes = [Text.of(r.cat.field(it, r.store.notes)) if r.store.notes else "" for it in pages]
        keep = [i for i, row in enumerate(rows) if row]
        return [rows[i] for i in keep], [notes[i] for i in keep]

    def stated_names(self, ai: int) -> list[list[str]]:
        r = self.r
        props = [x["prop"] for x in r.reqs(ai) if x["source"] == "stated"]
        for c in r.clauses:
            if c.get("type") != "when":
                continue
            for side in (c.get("cond"), c.get("then")):
                if isinstance(side, dict) and side.get("article") == ai:
                    props.append(Text.of(side.get("property")))
        return [s for s in (Text.key_stems(p) for p in props) if s]

    @staticmethod
    def same_name(ks: list[str], ps: list[str]) -> bool:
        return ks == ps or (len(ps) >= 2 and Text.run_in(ps, ks)) or (len(ks) >= 2 and Text.run_in(ks, ps))

    def table(self, ai: int) -> list[dict]:
        rows, notes = self.page_set(ai)
        tab = key_table(rows, notes, self.r.request)
        said = self.stated_names(ai)
        for m in tab:
            ks = Text.key_stems(m["key"])
            m["stated"] = any(self.same_name(ks, ps) for ps in said)
        self.tables[ai] = tab
        return tab

    def story_rank(self, tables: dict[int, list[dict]]) -> None:
        if self.story is not None:
            return
        self.story = {}
        r = self.r
        lines = [ai for ai in sorted(tables) if tables[ai]]
        if not lines:
            return
        parts = []
        for n, ai in enumerate(lines, 1):
            art = r.articles[ai]
            shown = sorted(tables[ai][:STORY_KEYS], key=lambda m: m["key"])
            shelf = Text.of(art.get("cat")).strip()
            head = f"{n}. {art['noun']}" + (f" (store shelf: {shelf})" if shelf and Text.key(shelf) != Text.key(art["noun"]) else "")
            parts.append(head + "".join(f"\n   - {m['key']} (e.g. {', '.join(m['ex'])})" for m in shown))
        user = f"REQUEST:\n{r.request[:5000]}\n\nLINES AND CANDIDATE PROPERTIES (example values seen in the store):\n" + "\n".join(parts)
        d = r.model_json("rank", STORY_SYS, user, max_tokens=1200, deadline=ASK_BY - 5, chain=STORY_MODELS)
        got = d.get("lines") if isinstance(d, dict) else d
        for x in got if isinstance(got, list) else []:
            if not isinstance(x, dict):
                continue
            try:
                n = int(x.get("line"))
            except (TypeError, ValueError):
                continue
            if not 1 <= n <= len(lines) or lines[n - 1] in self.story:
                continue
            ai = lines[n - 1]
            names = {Text.key(sp): m["key"] for m in tables[ai] for sp in m["spellings"] + [m["key"]]}
            order: list[str] = []
            for k in x.get("ranking") if isinstance(x.get("ranking"), list) else []:
                hit = names.get(Text.key(Text.of(k)))
                if hit and hit not in order:
                    order.append(hit)
            if order:
                self.story[ai] = order
                self.hints[ai] = Text.of(x.get("hint"))[:200]
        r.rank_log.append({"story": {str(ai): v[:STORY_KEYS] for ai, v in self.story.items()},
                           "hints": {str(ai): h for ai, h in self.hints.items()}, "user": user[:6000]})

    def ranked(self, ai: int) -> list[str]:
        tab = self.tables.get(ai)
        if tab is None:
            tab = self.table(ai)
        order = (self.story or {}).get(ai) or []
        first = order[0] if order else None
        keep = [m for m in tab if not self.covered(ai, m["key"]) and (not m["stated"] or m["key"] == first)]
        base = [k for _, k in story_order(keep, order)]
        if not self.single or not SINGLE_LEAD:
            return base
        if self.single_order is None:
            self.single_rank()
        lead = [k for k in (self.single_order or {}).get(ai) or [] if not self.covered(ai, k)]
        seen = {Text.key(k) for k in lead}
        return lead + [k for k in base if Text.key(k) not in seen]

    def menu_known(self, ai: int) -> set[str]:
        r = self.r
        known = {Text.key(x["prop"]) for x in r.reqs(ai) if x["op"] not in ("min", "max")}
        known |= {Text.key(x["prop"]) for x in r.ledger if x["op"] == "ref" and x["a"] in (ai, None)}
        return known

    def menu_for(self, ai: int, req_stems: list[str]) -> list[dict]:
        r = self.r
        art = r.articles[ai]
        known = self.menu_known(ai)
        pool = r.owners(ai)
        if not pool:
            return []
        per: dict[str, dict[str, str]] = {}
        family: dict[str, list[str]] = {}
        for it in pool:
            for k, v in it["facts"].items():
                fam = " ".join(sorted(Text.key_stems(k)))
                family.setdefault(fam, [])
                if k not in family[fam]:
                    family[fam].append(k)
                per.setdefault(fam, {}).setdefault(it["id"], Text.of(v))
        groups: dict[frozenset, list[str]] = {}
        for fam, m in per.items():
            groups.setdefault(frozenset((i, Text.key(v)) for i, v in m.items()), []).extend(family[fam])
        menu = []
        for keys in groups.values():
            keys.sort(key=lambda k: (len(Text.key(k)), k))
            k = keys[0]
            fam = " ".join(sorted(Text.key_stems(k)))
            nk = Text.key(k)
            if any(Text.key(x) in known for x in keys) or any(
                    (len(Text(kk).stems) >= 2 and Text.run_in(Text(kk).stems, Text(nk).stems)) for kk in known):
                continue
            vals = list(per[fam].values())
            n = len(vals)
            cov = n / len(pool)
            top = max(collections.Counter(Text.key(v) for v in vals).values()) / n
            mlen = sum(len(v) for v in vals) / n
            digits = sum(1 for v in vals if any(ch.isdigit() for ch in v)) / n
            named = 1.0 if Text.run_in(Text(nk).stems, req_stems) else 0.0
            listy = sum(1 for v in vals if "/" in v or "," in v) / n
            prior = cov - 0.5 * top - 0.5 * min(1.0, mlen / 15) - 0.25 * digits - 0.25 * named - (0.5 if listy >= 0.5 else 0.0)
            menu.append({"prop": k, "n": n, "distinct": len({Text.key(v) for v in vals}), "prior": round(prior, 3),
                         "ex": [v[:30] for v in list(dict.fromkeys(vals))[:3]]})
        for o in art.get("open") or []:
            for m in menu:
                if set(Text.key_stems(o)) & set(Text.key_stems(m["prop"])):
                    m["hint"] = True
        menu.sort(key=lambda m: -m["prior"])
        return menu

    def menu_role(self, ai: int) -> str:
        pairs = [p for p in self.r.together if len(p) == 2]
        if any(p[1] == ai for p in pairs):
            return "hub"
        return "acc" if any(p[0] == ai for p in pairs) else ""

    def names_other_line(self, ai: int, stems: list[str]) -> bool:
        arts = self.r.articles

        def heads(art: dict) -> set[str]:
            return {w for n in (art["noun"], art.get("cat") or "") for h in noun_heads(n) for w in Text(h).stems}
        own = heads(arts[ai])
        for art in arts:
            if art["i"] == ai:
                continue
            for y in heads(art) - own:
                if any(Text.alike(x, y) or Text.alike(y, x) for x in stems):
                    return True
        return False

    def menu_settled(self, ai: int, m: dict, examples: dict[int, dict[str, list[str]]]) -> bool:
        r = self.r
        stems = Text.key_stems(m["prop"])
        if not stems:
            return False
        for c in r.clauses:
            if c.get("type") != "when":
                continue
            for side in (c.get("cond"), c.get("then")):
                if isinstance(side, dict) and side.get("article") == ai and Text.key_stems(Text.of(side.get("property"))) == stems:
                    return True
        if self.names_other_line(ai, stems):
            return True
        if self.menu_role(ai) != "acc":
            return False
        ex = [w for v in m.get("ex") or [] for w in Text(v).stems]
        for a, b in r.together:
            if a != ai or not 0 <= b < len(r.articles):
                continue
            hub = r.articles[b]
            names = Text(hub["noun"]).stems + Text(hub.get("cat") or "").stems
            if any(Text.alike(x, y) or Text.alike(y, x) for x in stems + ex for y in names):
                return True
            if not any(ch.isdigit() for v in m.get("ex") or [] for ch in v):
                continue
            for k in examples.get(b) or {}:
                ks = Text.key_stems(k)
                if len(ks) >= 2 and Text.run_in(ks, stems):
                    return True
        return False

    def single_rank(self) -> None:
        r = self.r
        self.single_order = {}
        req_stems = Text(r.request).stems
        raw = {art["i"]: self.menu_for(art["i"], req_stems) for art in r.articles}
        examples = {ai: {m["prop"]: list(m.get("ex") or []) for m in menu} for ai, menu in raw.items()}
        menus: dict[int, list[dict]] = {}
        others: dict[int, list[str]] = {}
        for ai, menu in raw.items():
            menu = [m for m in menu if not self.menu_settled(ai, m, examples)]
            if menu:
                menus[ai] = menu[:MENU_SIZE]
                others[ai] = [m["prop"] for m in menu[MENU_SIZE:MENU_SIZE + 30]]
        if not menus or r.t() > ASK_BY - 10:
            return
        lines = []
        for ai, menu in menus.items():
            art = r.articles[ai]
            st = "; ".join(f"{x['prop']} {x['op']} {Text.json(x['value'])}" for x in r.reqs(ai)) or "none"
            cands = "; ".join(f"{m['prop']} (on {m['n']} listings, {m['distinct']} values, e.g. {' / '.join(m.get('ex') or [])}, prior {m['prior']:+.2f}"
                              + (", the story hints at it)" if m.get("hint") else ")") for m in menu)
            missed = [k for h in r.heard if h.get("a") == ai and len(h.get("keys") or []) == 1
                      and h.get("kind") not in ("value", "reference", "ask_singly") for k in h["keys"]]
            got = [x["prop"] for x in r.ledger if x["a"] == ai and (x["source"] == "revealed" or x.get("heard"))]
            hist = (f"\n  asked already, nothing revealed: {', '.join(dict.fromkeys(missed))}" if missed else "") + (
                f"\n  already revealed: {', '.join(dict.fromkeys(got))}" if got else "")
            lines.append(f"article {ai}: {art['noun']} (x{art['qty']}); stated: {st}{hist}\n  candidates: {cands}"
                         + (f"\n  other listed properties: {', '.join(others[ai])}" if others.get(ai) else ""))
        user = f"REQUEST:\n{r.request[:5000]}\n\nARTICLES ({len(r.articles)} in all):\n" + "\n".join(lines)
        d = r.model_json("rank", SHOPPER_SYS, user, max_tokens=900, deadline=ASK_BY - 5, chain=STORY_MODELS)
        for ai, menu in menus.items():
            names = {Text.key(m["prop"]): m["prop"] for m in menu}
            names.update({Text.key(k): k for k in others.get(ai) or []})
            got = d.get(str(ai)) if isinstance(d, dict) else None
            order = got.get("order") if isinstance(got, dict) else None
            chosen: list[str] = []
            for k in order if isinstance(order, list) else []:
                hit = names.get(Text.key(Text.of(k)))
                if hit and hit not in chosen:
                    chosen.append(hit)
            self.single_order[ai] = chosen or [m["prop"] for m in menu]
        r.rank_log.append({"single_rank": {str(k): v[:10] for k, v in self.single_order.items()}})

    def line_chance(self, ai: int) -> float:
        if ai not in self.pa:
            n = max(1, len(self.r.articles))
            self.pa[ai] = SECOND_NEED if self.revealed(ai) else LINE_PRIOR[min(n, len(LINE_PRIOR)) - 1]
        return self.pa[ai]

    def offer(self, ai: int) -> tuple[float, list[str], list[float]]:
        probs = key_belief(self.ranked(ai), self.miss.get(ai, {}), self.gone.get(ai, set()))
        if not probs:
            return 0.0, [], []
        return self.line_chance(ai) * probs[0][1], [probs[0][0]], [probs[0][1]]

    def fold_miss(self, ai: int, keys: list[str], probs: list[float], miss: dict[str, float], pa: float) -> float:
        others = [m["key"] for m in self.tables.get(ai) or [] if m["key"] not in keys]
        rows = self.page_set(ai)[0] if others else []
        for k in keys:
            miss[k] = miss.get(k, 1.0) * (1 - SINGLE_RECALL)
            for o in others:
                f = kin_rate(k, o)
                if f > 0 and any(k in row and o in row for row in rows) and same_reading(rows, k, o):
                    f = 1.0
                if f > 0:
                    miss[o] = miss.get(o, 1.0) * (1 - SINGLE_RECALL * f)
        return after_miss(pa, probs)

    def spelled(self, ai: int, key: str) -> str:
        plain = " ".join(key.replace("_", " ").split())
        tab = self.tables.get(ai) or []
        m = next((x for x in tab if x["key"] == key), None)
        if m is None:
            return plain
        rows = self.page_set(ai)[0]
        cover = collections.Counter(k for row in rows for k in row)
        names = [key] + [s for s in m.get("spellings") or [] if s != key]
        ks = Text.key_stems(key)
        for o in tab:
            ok = o["key"]
            if ok == key or ok in names:
                continue
            os_ = Text.key_stems(ok)
            near = RevealAsker.twin_key(ks, os_) or kin_rate(key, ok) > 0 or kin_rate(ok, key) > 0
            if near and ((any(key in row and ok in row for row in rows) and same_reading(rows, key, ok)) or alike_values(rows, key, ok)):
                names.append(ok)
        top = max(names, key=lambda k: (cover[k], k == key, -len(k), k))
        if not cover[top] or words_in(Text(top).words, Text(plain).words):
            return plain
        return " ".join(top.split())

    def ask_item(self, ai: int) -> str:
        art = self.r.articles[ai]
        base = self.ask_noun(ai)
        props = [Text(x).words for m in self.tables.get(ai) or [] for x in [m["key"]] + list(m.get("spellings") or [])]
        bw = Text(base).words

        def adds_prop(n: str) -> bool:
            nw = Text(n).words
            return any(p and words_in(p, nw) and not words_in(p, bw) for p in props)
        names = [base] + [n for n in (Text.of(x).strip().lower() for x in (art["noun"], art.get("cat"))) if n and not adds_prop(n)]

        def need(n: str) -> set[str]:
            return {w for w in Text(n).stems if len(w) > 2 or not w.isalpha()}
        whole = set().union(*(need(n) for n in names))
        for n in names:
            if whole <= need(n):
                return n
        out, have = base, need(base)
        for n in names[1:]:
            if not need(n) <= have:
                out += f" ({n})"
                have |= need(n)
        return out

    def phrase(self, ai: int, key: str) -> str:
        said = self.spelled(ai, key)
        ask = self.r.tm.get("ask") or {}
        spec = ((getattr(self.r, "schemas", None) or {}).get(ask.get("name")) or {}).get("properties") or {}
        cap = (spec.get(ask.get("text")) or {}).get("maxLength")
        limit = min(QUESTION_CHARS, cap) if isinstance(cap, int) and cap > 0 else QUESTION_CHARS
        items = [self.ask_item(ai), self.ask_noun(ai), (noun_heads(self.ask_noun(ai)) or [""])[-1]]
        text = ""
        for long_form in (True, False):
            for item in items:
                text = f"What {said} would you like for the {item}?" if long_form else f"Which {said} for the {item}?"
                if len(text) <= limit:
                    return text
        return text

    def plan_round(self, budget: int, only: list[int] | None = None) -> None:
        r = self.r
        ask = r.tm.get("ask")
        if not ask or not ask.get("text") or budget <= 0 or r.t() > ASK_BY:
            return
        tabs = {a["i"]: self.table(a["i"]) for a in r.articles}
        lines = [ai for ai, tab in tabs.items() if tab and (only is None or ai in only)]
        if not lines:
            return
        self.story_rank(tabs)
        spent = tries = 0
        while spent < budget and tries < ASK_TRIES and self.may_ask():
            tries += 1
            best: tuple[float, int, str, float] | None = None
            for ai in lines:
                gain, keys, probs = self.offer(ai)
                if keys and (best is None or gain > best[0] + 1e-12):
                    best = (gain, ai, keys[0], probs[0])
            if best is None or best[0] <= 0:
                break
            gain, ai, key, prob = best
            text = self.phrase(ai, key)
            if text in self.sent:
                miss = self.miss.setdefault(ai, {})
                miss[key] = miss.get(key, 1.0) * (1 - SINGLE_RECALL)
                continue
            if not self.put(ai, [key], [prob], text, gain):
                break
            spent += 1

    def spot(self, keys: list[str], prop: str, reply: str) -> int | None:
        ps = Text.key_stems(prop)
        for j, k in enumerate(keys):
            if ps and Text.key_stems(k) == ps:
                return j
        said = Text(reply).stems
        hits = [j for j, k in enumerate(keys) if self.said_exactly(Text.key_stems(k), said)]
        if hits:
            return max(hits, key=lambda j: (len(Text.key_stems(keys[j])), -j))
        for j, k in enumerate(keys):
            ks = set(Text.key_stems(k))
            if ps and ks and (ks <= set(ps) or set(ps) <= ks):
                return j
        return None

    def put(self, ai: int, keys: list[str], probs: list[float], text: str, gain: float) -> bool:
        r = self.r
        ask = r.tm["ask"]
        obs = r.env.send([(ask["name"], {ask["text"]: text})])
        if not obs:
            return False
        r.msgs += 1
        self.sent.add(text)
        reply = self.reply_text(r.store.unwrap(obs[0])[0])
        art = r.articles[ai]
        art["asked"].extend(k for k in keys if k not in art["asked"])
        label = keys[0] if len(keys) == 1 else ", ".join(keys)

        def call(sy: str, u: str, m: int) -> str:
            return r.model("read", sy, u, max_tokens=m, deadline=TASK_SECONDS - 40).text
        def strong(sy: str, u: str, m: int) -> str:
            return r.model("decide", sy, u, max_tokens=m, deadline=TASK_SECONDS - 40).text
        item = self.ask_noun(ai)
        got = read_reply(call, text, label, reply, item)
        lost = got if got.get("lost_ref") is not None else None
        if got.get("kind") in ("no_preference", "not_understood", "other"):
            def second(sy: str, u: str, m: int) -> str:
                return r.model("read", sy, u, max_tokens=m, deadline=TASK_SECONDS - 40, chain=SECOND_READER).text
            again = read_reply(second, text, label, reply, item)
            if again.get("kind") in ("value", "reference"):
                got = again
        if got.get("kind") == "other" and got.get("quote"):
            other = self.named_prop(label, reply)
            if Text.key(other) != Text.key(label):
                again = read_reply(call, text, other, reply, item)
                if again.get("kind") in ("value", "reference"):
                    got = again
        if got.get("kind") == "other" and got.get("quote"):
            again = read_reply(strong, text, label, reply, item)
            if again.get("kind") in ("value", "reference"):
                got = again
        if got.get("kind") == "value" and got.get("op") == "not":
            second = read_reply(strong, text, label, reply, item)
            if second.get("kind") in ("value", "reference"):
                got = second
        if lost and got.get("kind") not in ("value", "reference"):
            got = self.plain_read(ai, text, label, reply, lost.get("quote"), lost["lost_ref"]) or got
        kind = got["kind"]
        pa = self.line_chance(ai)
        listed = len(keys) > 1 and Text.key(got["property"]) == Text.key(label)
        at = self.spot(keys, "" if listed else got["property"], reply) if kind in ("value", "reference") else None
        if listed:
            fix = keys[at] if at is not None else self.reply_key(keys, reply)
            got = dict(got, property=fix)
            if isinstance(got.get("reference"), dict):
                got["reference"] = dict(got["reference"], prop=fix)
        entry = reply_to_ledger(got, ai)
        gone = self.gone.setdefault(ai, set())
        owner = self.line_named(reply, ai) if entry else ai
        if entry and owner != ai:
            entry = dict(entry, a=owner)
            r.notes.append(f"reply names line {owner}, not asked line {ai}: need filed there")
            if entry["op"] == "ref":
                self.take_reference(owner, self.ref_named(entry, reply), entry["prop"])
            else:
                prop = self.named_prop(entry["prop"], reply, Catalog.wanted_of(entry))
                r.add_req(owner, {"prop": prop, "op": entry["op"], "value": entry["value"], "unit": entry.get("unit"),
                                  "quote": entry.get("quote")}, "revealed")
            self.pa[owner] = SECOND_NEED
            miss = self.miss.setdefault(ai, {})
            for k in keys:
                miss[k] = miss.get(k, 1.0) * (1 - SINGLE_RECALL)
            self.pa[ai] = after_miss(pa, probs)
        elif entry:
            if at is not None:
                gone.update(keys[:at + 1])
            if entry["op"] == "ref":
                self.take_reference(ai, self.ref_named(entry, reply), entry["prop"])
            else:
                prop = self.named_prop(entry["prop"], reply, Catalog.wanted_of(entry))
                r.add_req(ai, {"prop": prop, "op": entry["op"], "value": entry["value"], "unit": entry.get("unit"),
                               "quote": entry.get("quote")}, "revealed")
            self.pa[ai] = SECOND_NEED
        else:
            self.pa[ai] = self.fold_miss(ai, keys, probs, self.miss.setdefault(ai, {}), pa)
        r.heard.append({"a": ai, "q": text, "r": reply, "prop": label, "kind": kind, "keys": keys, "at": at,
                        "read": {k: got.get(k) for k in ("property", "op", "values", "quote")}})
        r.rank_log.append({"line": ai, "keys": keys, "kind": kind, "at": at, "pa": round(pa, 3), "pa_after": round(self.pa[ai], 3),
                           "gain": round(gain, 3), "form": "single", "step": r.steps})
        r.plan.replan("reply")
        return True

    def plain_read(self, ai: int, question: str, label: str, reply: str, quote: Any, ref: dict | None) -> dict | None:
        r = self.r
        ref = ref or {}
        text = plain_part(reply, quote, ref.get("seller"))
        told = set(tokens(" ".join([question, label, Text.of(ref.get("seller")), Text.of(ref.get("kind"))]
                                   + [Text.of(v) for v in (ref.get("with") or {}).values()] + self.article_names())))
        if not text or not set(tokens(text)) - told or r.t() > GUARD_AT - 40:
            return None

        raw: list[str] = []

        def call(sy: str, u: str, m: int) -> str:
            raw.append(r.model("read", sy, u, max_tokens=m, deadline=TASK_SECONDS - 40).text)
            return raw[-1]
        got = read_reply(call, question, label, text, self.ask_noun(ai))
        vals = [v for v in got.get("values") or [] if set(tokens(v)) - told]
        data = rd_json(raw[-1]) if raw else None
        said = set(Text.key_stems(data.get("property") if isinstance(data, dict) and rd_text(data.get("property")).strip() else got.get("property")))
        if got.get("kind") != "value" or not vals or not any(
                q and said and (said <= q or q <= said) for q in (set(Text.key_stems(label)), set(Text.key_stems(ref.get("prop"))))):
            return None
        r.notes.append(f"reference reply on line {ai} with no listing to trace; its own value {vals} used")
        return dict(got, values=vals)

    def reply_key(self, keys: list[str], reply: str) -> str:
        runs = [(len(self.said_run(Text.key_stems(k), reply).split()), -j) for j, k in enumerate(keys)]
        n, j = max(runs)
        return keys[-j] if n else keys[0]

    def line_named(self, reply: str, ai: int) -> int:
        r = self.r
        said = Text(reply).stems
        hits: dict[int, int] = {}
        for art in r.articles:
            for name in (art.get("cat"), art["noun"], self.ask_noun(art["i"])):
                st = Text(Text.of(name)).stems
                if st and self.said_exactly(st, said):
                    hits[art["i"]] = max(hits.get(art["i"], 0), len(st))
        if not hits or ai in hits:
            return ai
        top = max(hits.values())
        named = [b for b, n in hits.items() if n == top]
        return named[0] if len(named) == 1 else ai

    def ask_noun(self, ai: int) -> str:
        art = self.r.articles[ai]
        names = [n for n in (art.get("cat"), art["noun"]) if n]
        for n in names:
            if len(noun_heads(n)) < 2 or any(ch in n for ch in "&/"):
                return n.lower()
        name = names[0].lower()
        parts: list[str] = []
        at = 0
        for h in noun_heads(name):
            m = re.search(r"\b" + re.escape(h) + r"\b", name[at:])
            if not m:
                continue
            gap, at = name[at:at + m.end()], at + m.end()
            run = Text(gap).words
            if parts and "," not in gap:
                run = run[1:]
            if run:
                parts.append(Text(" ".join(run)).singular)
        if len(parts) >= 2 and len(parts[0].split()) == 1 and len(parts[-1].split()) > 1:
            return parts[-1]
        out: list[str] = []
        for part in parts:
            if not set(part.split()) <= {w for o in out for w in o.split()}:
                out.append(part)
        return " or ".join(out) or name

    def reply_text(self, obs: Any) -> str:
        if isinstance(obs, str):
            return obs
        if isinstance(obs, list):
            return " ".join(s for s in obs if isinstance(s, str))
        if isinstance(obs, dict):
            strs: list[str] = []
            listed: list[str] = []

            def walk(x: Any, inlist: bool) -> None:
                if isinstance(x, str) and x.strip():
                    (listed if inlist else strs).append(x)
                elif isinstance(x, dict):
                    for y in x.values():
                        walk(y, inlist)
                elif isinstance(x, list):
                    for y in x:
                        walk(y, True)
            walk(obs, False)
            if strs or listed:
                return max(strs or listed, key=len)
        return Text.json(obs, 400)

    def take_reference(self, ai: int, entry: dict, prop: str) -> None:
        r = self.r
        ref = entry["ref"] or {}
        rp = Text.key(Text.of(ref.get("prop") or prop))
        rs = Text.of(ref.get("seller")).strip().lower()
        for x in r.ledger:
            if x["op"] == "ref" and x["a"] == ai and Text.key(x["prop"]) == rp and Text.of((x.get("ref") or {}).get("seller")).strip().lower() == rs:
                return
        r.ledger.append(dict(entry, step=r.steps))

    def covered(self, ai: int, prop: str) -> bool:
        p = set(Text.key_stems(prop))
        if not p:
            return False
        for x in self.r.ledger:
            if not x["active"] or x["a"] != ai or x["source"] == "stated":
                continue
            q = set(Text.key_stems(x["prop"]))
            if q and (p <= q or q <= p):
                return True
        return False

    def named_prop(self, prop: Any, reply: str, values: list | tuple = ()) -> str:
        prop = Text.of(prop)
        keys = dict.fromkeys(k for it in self.r.cat.items.values() if it["read"] for k in it["facts"])
        own = Text.key_stems(prop)
        full = Text(reply).stems
        if own and self.said_exactly(own, full):
            longer = [k for k in keys if len(Text.key_stems(k)) > len(own) and self.said_exactly(Text.key_stems(k), full)
                      and Text.run_in(own, Text.key_stems(k))]
            return max(longer, key=lambda k: len(Text.key(k))) if longer else prop
        names = {w for n in self.article_names() for w in Text(n).stems}
        spoken = [k for k in keys if len(Text.key_stems(k)) >= 2 and self.said_exactly(Text.key_stems(k), full)
                  and not set(Text.key_stems(k)) <= names]
        if spoken:
            return max(spoken, key=lambda k: len(Text.key(k)))
        for v in list(values) + self.article_names():
            v = Text.of(v).strip()
            if v:
                reply = re.sub(re.escape(v), " ", reply, flags=re.I)
        said = Text(reply).stems
        if Text.run_in(own, said):
            exact = [k for k in keys if Text.key_stems(k) and self.said_exactly(Text.key_stems(k), said) and Text.run_in(own, Text.key_stems(k))]
            if self.said_exactly(own, said):
                longer = [k for k in exact if len(Text.key_stems(k)) > len(own)]
                return max(longer, key=lambda k: len(Text.key(k))) if longer else prop
            return max(exact, key=lambda k: len(Text.key(k))) if exact else prop
        hits = [k for k in keys if len(Text.key_stems(k)) >= 1 and Text.run_in(Text.key_stems(k), said)]
        return self.kin_named(own, hits, reply) or (max(hits, key=lambda k: len(Text.key(k))) if hits else prop)

    def article_names(self) -> list[str]:
        names = [Text.of(n).strip() for a in self.r.articles for n in (a.get("cat"), a.get("noun")) if Text.of(n).strip()]
        return sorted(set(names), key=len, reverse=True)

    def kin_named(self, own: list[str], hits: list[str], reply: str) -> str:
        kin = [k for k in hits if any(Text.alike(a, b) for a in Text.key_stems(k) for b in own)]
        if kin:
            return max(kin, key=lambda k: len(Text.key(k)))
        return self.said_run(own, reply)

    @staticmethod
    def said_run(own: list[str], reply: str) -> str:
        words = Text(reply).words
        said = [Text.stem(w) for w in words]
        best: tuple[int, int] | None = None
        for i in range(len(said)):
            for j in range(len(own)):
                n = 0
                while i + n < len(said) and j + n < len(own) and Text.alike(said[i + n], own[j + n]):
                    n += 1
                if n and any(len(x) >= 4 and not x.isdigit() for x in said[i:i + n]) and (best is None or n > best[1]):
                    best = (i, n)
        return " ".join(words[best[0]:best[0] + best[1]]) if best else ""

    @staticmethod
    def said_exactly(st: list[str], said: list[str]) -> bool:
        return any(said[i:i + len(st)] == st for i in range(len(said) - len(st) + 1))

    def ref_named(self, entry: dict, reply: str) -> dict:
        ref = entry.get("ref") or {}
        prop = Text.of(entry.get("prop"))
        said = Text(reply).stems
        if not ref or Text.run_in(Text.key_stems(prop), said):
            return entry
        told = Text(" ".join([Text.of(ref.get("kind")), Text.of(ref.get("seller"))]
                             + [f"{k} {Text.of(v)}" for k, v in (ref.get("with") or {}).items()])).stems
        keys = dict.fromkeys(k for it in self.r.cat.items.values() if it["read"] for k in it["facts"])
        hits = [k for k in keys if Text.key_stems(k) and Text.run_in(Text.key_stems(k), said) and not Text.run_in(Text.key_stems(k), told)]
        if not hits:
            return entry
        best = max(hits, key=lambda k: len(Text.key(k)))
        if not all(Text.run_in(Text.key_stems(k), Text.key_stems(best)) for k in hits):
            return entry
        rs = Text.of(ref.get("seller")).strip().lower()
        if any(x["op"] == "ref" and x["a"] == entry["a"] and Text.key(x["prop"]) == Text.key(best)
               and Text.of((x.get("ref") or {}).get("seller")).strip().lower() == rs for x in self.r.ledger):
            return entry
        return dict(entry, prop=best, ref=dict(ref, prop=best))

    def apply_changes(self, ch: Any) -> None:
        r = self.r
        if not isinstance(ch, dict):
            return
        self.move_budget(ch.get("budget"))
        for prop in ch.get("dropped") or []:
            for x in r.ledger:
                if x["active"] and Text.key(x["prop"]) == Text.key(Text.of(prop)):
                    x["active"] = False
                    r.dropped_since = True
                    if isinstance(x.get("a"), int) and 0 <= x["a"] < len(r.articles):
                        art_x = r.articles[x["a"]]
                        art_x["search_steps"] = min(art_x.get("search_steps", 0), LINE_SEARCH_STEPS - 2)
        q = ch.get("quantity")
        if isinstance(q, dict):
            for k, v in q.items():
                try:
                    ai, n = int(k), int(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= ai < len(r.articles) and 1 <= n <= 99:
                    r.articles[ai]["qty"] = n

    def move_budget(self, raw: Any) -> None:
        r = self.r
        try:
            b = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return
        if not b or b <= 0:
            return
        r.budget = b * r.money_scale

    def catch_up(self) -> bool:
        r = self.r
        fresh = r.news[r.news_read:]
        if not fresh:
            return False
        r.news_read = len(r.news)
        self.absorb(fresh)
        return True

    def restated_drops(self, got: dict) -> dict:
        r = self.r
        keep = []
        for v in got.get("values") or []:
            own = set(Text(Text.of(v.get("property"))).stems)
            said = own | set(Text(" ".join(Text.of(x) for x in v.get("values") or [])).stems)
            tgt = v.get("article")
            hit = next((x["prop"] for x in r.ledger if x["active"] and x["source"] != "news" and (tgt is None or x["a"] in (tgt, None))
                        and (ps := set(Text(x["prop"]).stems)) and not ps <= own and ps <= said), None)
            if hit is None:
                keep.append(v)
                continue
            got.setdefault("dropped", []).append({"article": tgt, "property": hit, "quote": v.get("quote")})
            r.notes.append(f"notice value {v.get('property')!r}={v.get('values')!r} only restates {hit!r}: read as a drop")
        got["values"] = keep
        return got

    def absorb(self, texts: list[str]) -> None:
        r = self.r

        def call(sy: str, u: str, m: int) -> str:
            return r.model("read", sy, u, max_tokens=m, deadline=GUARD_AT).text
        got = self.restated_drops(read_notice(call, texts, [a["noun"] for a in r.articles], [Text.of(x.get("prop")) for x in r.ledger if x["active"]]))
        r.notices.append(got)
        b = got.get("budget")
        if b and b.get("amount"):
            self.apply_changes({"budget": b["amount"]})
        if got.get("dropped"):
            self.apply_changes({"dropped": [n for d in got["dropped"] for n in (d["property"], d.get("requirement")) if n]})
        if got.get("quantity"):
            self.apply_changes({"quantity": {str(q["article"]): q["count"] for q in got["quantity"]}})
        picks = [a["pick"] for a in r.articles if a.get("pick")]
        if got.get("gone") or got.get("price"):
            r.knocked = {"gone": bool(got.get("gone")), "price": bool(got.get("price"))}
        if got.get("gone") and len(picks) == 1 and picks[0] in r.cat.items:
            r.cat.items[picks[0]]["gone"] = True
        for v in got.get("values") or []:
            tgt = v["article"] if isinstance(v.get("article"), int) else (0 if len(r.articles) == 1 else None)
            r.add_req(tgt, {"prop": v["property"], "op": v["op"], "value": ledger_value(v["values"]), "unit": v.get("unit"),
                            "quote": v.get("quote")}, "news")
        r.plan.replan("notice")

    def basket_text(self, lines: dict[str, int], ids: bool = False) -> str:
        r = self.r
        parts, total = [], 0.0
        for cid, q in lines.items():
            it = r.cat.items.get(cid) or {}
            p = r.cat.price(it) if it else None
            total += (p or 0) * q
            name = r.cat.title(it)[:70] if it else ""
            tag = f" ({cid})" if ids and name else ""
            parts.append(f"{q} x {name or cid}{tag}" + (f" at {p:g} each" if p is not None else ""))
        return "Here is the final basket: " + "; ".join(parts) + f". Total {total:g}."

    def ask_approval(self, lines: dict[str, int]) -> None:
        r = self.r
        ask = r.tm.get("ask")
        if not ask or not ask.get("text"):
            return
        text = self.basket_text(lines)
        if r.pre_note:
            missing = [a["noun"] for a in r.articles if not a.get("pick")]
            d = r.model_json("read", PRE_ORDER_SYS, f"INSTRUCTION: {r.pre_note[:400]}\nFINAL BASKET: {self.basket_text(lines, True)}\n"
                             f"NOT FOUND: {', '.join(missing) or 'nothing'}\nASK FOR APPROVAL TOO: {r.confirm}",
                             max_tokens=250, deadline=LAST_ENV_AT - 8)
            extra = Text.of(d.get("message")).strip() if isinstance(d, dict) else ""
            text = extra if len(extra) > 20 else text
        if r.confirm:
            text += " Do you approve this basket? With your yes I will place the order."
        obs = r.env.send([(ask["name"], {ask["text"]: text[:780]})])
        if obs:
            r.msgs += 1
            r.approved_cart = dict(lines)
            r.heard.append({"a": None, "q": text[:200], "r": self.reply_text(obs[0])[:200], "prop": "(approval)", "kind": "approval"})


RANK_KEYS = 16
ASK_TRIES = 64
MENU_SIZE = 10
QUESTION_CHARS = 140
VALUE_TWIN = 0.3
SINGLE_RECALL = 0.9
LINE_PRIOR = (0.81, 0.78, 0.49, 0.31, 0.30)
SECOND_NEED = 0.08
RANK_RATIO = 0.6
RANK_MASS = 0.97
RANK_PRIOR = tuple(RANK_MASS * (1 - RANK_RATIO) * RANK_RATIO ** i / (1 - RANK_RATIO ** RANK_KEYS) for i in range(RANK_KEYS))
RANK_TAIL = 0.002
SINGLE_LEAD = False
SECOND_READER = [(MISTRAL, None)]
KIN_NARROW = 0.997
KIN_BROAD = 0.673
PRE_ASK_PAGES = 10
PRE_ASK_STEPS = 99
KEY_MIN_SHARE = 0.05
KEY_WORDS = 5
KEY_W10 = (1.53, -0.83, -0.58, -0.85, -0.64, -1.74, -1.51, 0.44, 0.85)
KEY_W20 = (2.13, -0.84, -0.75, -0.77, -1.60, -2.23, -1.77, 0.58, 0.97)
KEY_W20_PAGES = 15
VARIANT_RX = re.compile(r"\s/\s|\+|\|")
VARIANT_MIN, VARIANT_SPREAD, VARIANT_SHARE = 4, 0.5, 0.3
NUMERIC_RX = re.compile(r"[<>~]?\s*-?\d+(?:[.,]\d+)?\s*[a-z\"'%/0-9 .]{0,12}")
STORY_KEYS = 12
STORY_MODELS = [(GEMMA, None), (HAIKU, "none")]

SHOPPER_SYS = """Play the shopper whose request follows. A shopping agent will question you; every question names one property
of one article. For some articles (not every one) you hold one wish left unwritten, usually one property (seldom
two), and you voice it only when asked about exactly that property. Read the request through its author's eyes: the worry
behind it, whom or which place each article serves, what you cannot judge alone, what it ought to pair with or stand
beside, what disappointed you earlier, what guests would notice. Decide per article: which ONE property would this
shopper insist on without having written it? Rank the article's candidate properties from the surest wish downward;
a property under "other listed properties" may go anywhere when the story points there. A property the request
settles never is the wish, nor a field merely listing purchase variants, nor a property already asked without a
reveal (a sibling of it then ranks higher). Numeric properties weigh like worded ones. Each candidate carries
a prior computed over the store rows (higher = likelier); the story outranks the prior.
Answer with ONLY JSON: {"<article index>": {"why": "<a short reason behind your top pick>", "order": ["<property>", "<property>", ...]}}"""

STORY_SYS = """You help a shopping assistant. A shopper's request names one or more numbered product lines and states some
requirements openly. For some lines the shopper also holds ONE unstated must-have about one property of that product.
They reveal it only when asked about that exact property, so the assistant must guess which property to ask about.

How such requests are written:
- Openly stated requirements (values, limits, "only A or B", references to other listings, "has to work with ...",
  budget, stock) are NOT hints. The hidden property is usually a different one; only now and then is it a stated
  property again, when the story stresses it once more.
- Story sentences are often riddles about the hidden property: they describe a situation in which one particular
  property of the product decides whether it suits the shopper, without naming it. Invented examples: "the hose never
  reached the far end of the allotment" points at hose length; "my grandmother can no longer work fiddly clasps" points
  at closure type; "my son only wants to know what sits under the bonnet" points at engine size; "the neighbours knock
  on the wall when it runs at night" points at noise level; "the last one gave up halfway up the hill" points at
  battery life.
- Other story sentences say the shopper does NOT mind about something ("I'm not fussy about the brand", "I haven't
  decided where it goes"). The property they talk about is then NOT the hidden one; rank it low.
- A property that only lists the variants on sale (option codes, or bundles of other properties) is never the hidden
  one; rank it last.
- Many lines carry no story hint. Then rank by which property this shopper would most plausibly insist on for this
  product and situation, avoiding properties already stated.

For every numbered line, rank ALL its candidate properties from most to least likely to be the hidden must-have.
Answer with JSON only: {"lines": [{"line": 1, "hint": "<story sentence used, or empty>", "ranking": ["<candidate>", ...]}]}
Use the candidate names exactly as given."""


class RevealAsker(Asker):

    def __init__(self, run: Any) -> None:
        super().__init__(run)
        self.persona: str | None = None

    def ranked(self, ai: int) -> list[str]:
        seen: list[list[str]] = []
        out: list[str] = []
        for k in super().ranked(ai):
            ks = Text.key_stems(k)
            if ks and any(self.twin_key(ks, o) for o in seen):
                continue
            seen.append(ks)
            out.append(k)
        return out

    @staticmethod
    def twin_key(a: list[str], b: list[str]) -> bool:
        if len(a) != len(b):
            return False
        for x, y in zip(sorted(a), sorted(b)):
            if x != y and not (min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x))):
                return False
        return True

    def put(self, ai: int, keys: list[str], probs: list[float], text: str, gain: float) -> bool:
        r = self.r
        n_heard = len(r.heard)
        ok = super().put(ai, keys, probs, text, gain)
        if not ok or len(r.heard) <= n_heard:
            return ok
        kind = r.heard[-1].get("kind")
        if self.persona is None and kind in ("no_preference", "not_understood"):
            self.persona = kind
        r.rank_log.append({"reveal_engine": {"line": ai, "kind": kind, "persona": self.persona}})
        return ok


class Chains:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.log: list[dict] = []
        self.traced: set = set()

    def room(self) -> int:
        return self.r.plan.room("chains")

    @staticmethod
    def toks(v: str) -> list[str]:
        return [w for w in Text(v).words if (any(c.isalpha() for c in w) and len(w) >= 2) or (w.isdigit() and len(w) >= 2)]

    def store_words(self) -> set[str]:
        cat = self.r.cat
        out: set[str] = set()
        for it in cat.items.values():
            out |= set(Text(cat.title(it) + " " + cat.category(it)).stems)
        return out

    def joining_words(self) -> set[str]:
        cats = {Text.of(self.r.cat.category(it)) for it in self.r.cat.items.values()} - {""}
        seen: collections.Counter = collections.Counter()
        edge: set[str] = set()
        for c in cats:
            ws = Text(c).words
            seen.update(set(ws))
            edge |= set(ws[:1] + ws[-1:])
        return {w for w, n in seen.items() if (n >= 3 or len(w) <= 3) and w not in edge}

    def queries_for(self, seller: str, kind: str, withs: dict) -> list[str]:
        known = self.store_words()
        joins = self.joining_words()
        kind_words = [w for w in Text(kind).words if Text.stem(w) in known and w not in joins] or ([] if known else Text(kind).words)
        noun = Text(" ".join(Text(" ".join(kind_words)).singular.split())).words[-1:]
        wv = [Text.of(v) for v in withs.values() if Text.of(v).strip()]
        keys = {w for it in self.r.cat.items.values() for k in it["facts"] for w in Text(k).words}
        vt = sorted(dict.fromkeys(t for v in wv for t in self.toks(v) if t not in noun and t not in kind_words and (t not in keys or t in known)),
                    key=lambda t: (-any(c.isdigit() for c in t), -len(t)))
        sw = Text(seller).words[:2]
        self.valued = set(vt)
        queries = [" ".join(sw + kind_words)] if kind_words else []
        traits = self.r.finder.traits or {}
        short = traits.get("limit") is not None and traits.get("limit") <= 2
        if vt and short:
            queries.append(" ".join(sw[:1] + vt[:1]))
            queries.append(" ".join(sw[:1] + vt[:1] + noun))
        elif vt:
            queries.append(" ".join(sw[:1] + vt[:1] + noun))
            queries.append(" ".join(sw + kind_words[-2:] + vt[:6]))
        queries.append(" ".join(sw + noun))
        if len(kind_words) >= 2 and kind_words[0] not in noun:
            queries.append(" ".join(sw[:1] + kind_words[:1]))
        if len(vt) >= 2:
            queries.append(" ".join(vt[:2] + noun))
        queries += [" ".join(self.toks(v)[:2] + noun) for v in wv[:2]]
        for v in wv[:2]:
            tv = Text(v).words
            if len(tv) >= 2 and tv[0].isdigit() and tv[1].isalpha():
                queries.append(" ".join(sw[:1] + [tv[0] + tv[1]] + noun))
        return queries

    def sieve_words(self, kind: str, withs: dict) -> list[str]:
        plain = set(Text(kind).words)
        per = []
        for k, v in withs.items():
            keyw = set(Text(k).words)
            best = []
            parts = [w for w in (p.strip(",;:()[]\"'") for p in re.split(r"[\s/-]+", Text.of(v).lower())) if w]
            for n, w in enumerate(parts):
                after_number = n > 0 and bool(re.fullmatch(r"[\d.,:/x-]+", parts[n - 1]))
                if not re.fullmatch(r"[^\W_]+", w):
                    level = None
                elif w.isdigit():
                    level = 3 if len(w) >= 2 else None
                elif w.isalpha():
                    level = None if len(w) < 2 else (4 if after_number else (1 if len(w) >= 3 else 2))
                else:
                    level = 0
                if level is not None and w not in plain and w not in keyw:
                    best.append((level, w))
            best.sort(key=lambda x: x[0])
            if best:
                per.append(best)
        per.sort(key=lambda b: b[0][0])
        self.sieve_level = {w: lv for b in per for lv, w in b}
        if not per or per[0][0][0] >= 4:
            return []
        flat = [b[0][1] for b in per] + [x[1] for b in per for x in b[1:]]
        return list(dict.fromkeys(flat))

    def sieve_pool(self, seller: str, kind: str, withs: dict) -> list[dict] | None:
        r = self.r
        cat = r.cat
        self.sieve_log: list[str] = []
        if r.finder.pair_and is False or not r.tm.get("search"):
            return None
        name = Text(seller).words
        words = [w for w in self.sieve_words(kind, withs) if w not in name]
        if len(name) != 1 or not words:
            return None
        low = seller.lower()

        def mine(it: dict) -> bool:
            return low in cat.seller(it).lower() or low in cat.maker(it).lower()

        found: dict[str, int] = {}

        def row_hits(it: dict) -> int:
            text = " ".join([cat.title(it)] + [v for v in it["row"].values() if isinstance(v, str)])
            have = set(Text(text).words)
            return sum(1 for w in words if w in have)

        def row_rare(it: dict) -> float:
            text = " ".join([cat.title(it)] + [v for v in it["row"].values() if isinstance(v, str)])
            have = set(Text(text).words) | set(Text(self.spaced(text)).words)
            return sum(1.0 / max(1, found[w]) for w in words if w in found and w in have)
        sets: list[tuple[set[str], bool]] = []
        tags: list[frozenset] = []
        group = {w: next((k for k, v in withs.items() if w in re.split(r"[\s/,;:()\[\]\"'-]+", Text.of(v).lower())), w) for w in words}
        pool: dict[str, dict] = {}
        seen_order: list[str] = []
        strong = [w for w in words if self.sieve_level.get(w, 4) < 4]
        combo = [next(w for w in strong if group[w] == g) for g in dict.fromkeys(group[w] for w in strong)][:3]
        for ws in [[w] for w in words]:
            if len(self.sieve_log) >= SIEVE_QUERIES or self.room() < 3 or r.t() > GUARD_AT - 30:
                break
            q = " ".join([name[0]] + ws)
            ids, total, _ = r.finder.run(q)
            self.sieve_log.append(q)
            rows = [cat.items[i] for i in ids if i in cat.items]
            if not rows:
                found[ws[0]] = 0
                continue
            own = [it for it in rows if mine(it)]
            if len(own) < len(rows):
                if not sets:
                    r.finder.pair_and = False
                    r.notes.append(f"sieve: '{q}' returned other sellers, falling back")
                    return None
                continue
            r.finder.pair_and = True
            found[ws[0]] = len(own)
            complete = isinstance(total, int) and total <= len(ids)
            for it in own:
                pool.setdefault(it["id"], it)
                if it["id"] not in seen_order:
                    seen_order.append(it["id"])
            sets.append(({it["id"] for it in own}, complete))
            tags.append(frozenset(group[w] for w in ws))
            full = [x for x, c in sets if c]
            both = set.intersection(*full) if full else set()
            if full and 0 < len(both) <= SIEVE_STOP:
                break
        full = [x for x, c in sets if c]
        both = set.intersection(*full) if full else set()
        if sets and not full and len(combo) >= 2 and self.room() >= 4 and r.t() <= GUARD_AT - 30:
            q = " ".join([name[0]] + combo)
            ids, total, _ = r.finder.run(q)
            self.sieve_log.append(q)
            got = [cat.items[i] for i in ids if i in cat.items]
            if got and all(mine(it) for it in got) and isinstance(total, int) and total <= len(ids) and len(got) <= SIEVE_STOP:
                for it in got:
                    pool.setdefault(it["id"], it)
                    if it["id"] not in seen_order:
                        seen_order.append(it["id"])
                sets.append(({it["id"] for it in got}, True))
                tags.append(frozenset(group[w] for w in combo))
                both = {it["id"] for it in got}
        if not pool:
            return None
        if not both and self.room() >= 4 and len(self.sieve_log) < SIEVE_QUERIES + 1:
            ids, total, _ = r.finder.run(name[0])
            self.sieve_log.append(name[0])
            for i in ids:
                it = cat.items.get(i)
                if it is not None and mine(it):
                    pool.setdefault(i, it)
                    if i not in seen_order:
                        seen_order.append(i)
        self.valued = set(words)
        self.valued_ids = set(both)

        def key(it: dict) -> tuple:
            hit = [n for n, (x, c) in enumerate(sets) if c and it["id"] in x]
            votes = len(frozenset().union(*[tags[n] for n in hit]))
            rare = sum(1.0 / max(1, len(sets[n][0])) for n in hit)
            loose = sum(1 for x, c in sets if not c and it["id"] in x)
            return (-votes, -rare, -loose, -row_hits(it), -row_rare(it), seen_order.index(it["id"]))
        return sorted(pool.values(), key=key)

    def seller_pool(self, seller: str, queries: list[str]) -> list[dict]:
        r = self.r
        cat = r.cat
        low = seller.lower()
        sw = Text(seller).words[:1]

        def mine(it: dict) -> bool:
            return low in cat.seller(it).lower() or low in cat.maker(it).lower()

        def named(it: dict) -> bool:
            texts = [cat.title(it)] + [v for v in it["row"].values() if isinstance(v, str)]
            return any(Text(t).has(seller) for t in texts)
        own: list[dict] = []
        first: list[dict] = []
        pool: list[dict] = []
        found = 0
        valued = getattr(self, "valued", set())

        def hits(it: dict) -> int:
            text = " ".join([cat.title(it)] + [v for v in it["row"].values() if isinstance(v, str)]).lower()
            words = set(Text(re.sub(r"(\d)([a-z])", r"\1 \2", text)).words)
            return sum(1 for t in valued if t in words)
        for q in [x for x in dict.fromkeys(queries) if x.strip()][:5]:
            if self.room() < 2:
                break
            ids, _, cursor = r.finder.run(q)
            pool = [cat.items[i] for i in ids if i in cat.items]
            got = [it for it in pool if mine(it)]
            if got and len(got) == len(pool) and cursor and self.room() >= 3 and not any(hits(it) >= max(1, len(valued)) for it in got):
                got += [cat.items[i] for i in r.finder.page_on(q, 1) if i in cat.items and mine(cat.items[i])]
            new = [it for it in got if it not in own]
            own += new
            if valued & set(Text(q).words) and got:
                first += [it for it in got if it not in first]
                break
            found += bool(new)
            if found >= 2 or (valued and any(hits(it) >= len(valued) for it in own)) or (got and self.room() < 4):
                break
            if not own and pool and sw and sw[0] in Text(q).words:
                own = [it for it in pool if named(it)]
                if own:
                    self.log.append({"seller": seller, "fallback": "title/row match"})
                    break
        order = {it["id"]: n for n, it in enumerate(own)}
        self.valued_ids = {it["id"] for it in first}
        return sorted(own, key=lambda it: (it not in first, -hits(it), order[it["id"]]))

    @staticmethod
    def spaced(v: Any) -> str:
        return re.sub(r"(\d)([a-z])", r"\1 \2", Text.of(v).lower())

    def detail_words(self, text: str, seller: str, kind: str, prop: str) -> list[str]:
        cat = self.r.cat
        pages = [it for it in cat.items.values() if it.get("read") and it["facts"]]
        if not pages:
            return []
        skip = set(Text(seller).stems) | set(Text(kind).stems) | set(Text(prop).stems)
        vals = [set(Text(self.spaced(" ".join(Text.of(v) for v in it["facts"].values()))).stems) for it in pages]
        prose = [set(Text(self.spaced(cat.notes_of(it))).stems) for it in pages]
        keys = [set(Text(" ".join(Text.key(k) for k in it["facts"])).stems) for it in pages]
        out = []
        for w in dict.fromkeys(Text(self.spaced(text)).stems):
            if w in skip or (len(w) < 2 and not w.isdigit()):
                continue
            seen_vals = sum(1 for x in vals if w in x)
            if any(ch.isdigit() for ch in w) or (seen_vals >= sum(1 for x in prose if w in x) and seen_vals >= sum(1 for x in keys if w in x)):
                out.append(w)
        return out

    def lead_rows(self, seller: str, values: list[str], tight: bool, alt: str = "") -> tuple[str, list[dict]]:
        r = self.r
        cat = r.cat
        name = Text(seller).words
        parts: list[str] = []
        said = set(name)
        for v in values:
            ws = Text(v).words
            if ws and not set(ws) <= said:
                parts.append(Text.of(v).strip())
                said |= set(ws)
        if not name or not parts:
            return "", []
        q = " ".join([seller.strip()] + parts)
        if r.finder.state(q) is None:
            if r.t() > GUARD_AT - 30 or not (self.room() >= 2 or (tight and r.plan.chain_reserve() >= 2)):
                return "", []
            r.finder.run(q)
        st = r.finder.state(q)
        low = seller.strip().lower()
        ids = st.ids if st is not None else []
        out = [cat.items[i] for i in ids if i in cat.items and (low in cat.seller(cat.items[i]).lower() or low in cat.maker(cat.items[i]).lower())]
        again = " ".join(name + [alt]) if alt and alt not in name else ""
        if not out and again and again != q.lower() and r.t() <= GUARD_AT - 30 and (self.room() >= 2 or (tight and r.plan.chain_reserve() >= 2)):
            if r.finder.state(again) is None:
                r.finder.run(again)
            st = r.finder.state(again)
            out = [cat.items[i] for i in (st.ids if st is not None else []) if i in cat.items
                   and (low in cat.seller(cat.items[i]).lower() or low in cat.maker(cat.items[i]).lower())]
            q = again
        return q, out

    @staticmethod
    def is_rule(rule: str, pointed: bool = False) -> bool:
        return bool(rule) and (pointed or bool(re.search(r"\d", rule)))

    def pick_listing(self, desc: str, cands: list[dict]) -> dict | None:
        r = self.r
        cat = r.cat
        lines = [f"{it['id']}: {cat.title(it)[:100]} | {cat.category(it)} | {Text.json(it['facts'], 400) if it.get('read') else '(no table read)'}"
                 + (f" | notes: {cat.notes_of(it)[:240]}" if cat.notes_of(it) else "") for it in cands]
        d = r.model_json("read", PICK_SYS, f"DESCRIPTION: {desc[:600]}\nLISTINGS:\n" + "\n".join(lines),
                         max_tokens=120, deadline=GUARD_AT - 25)
        pid = Text.of(d.get("id")) if isinstance(d, dict) else ""
        return next((it for it in cands if it["id"] == pid), None)

    def hop_call(self, role: str, best: dict, prop: str, desc: str, notes: str, facts: dict) -> Any:
        r = self.r
        cat = r.cat
        raw = r.model_json(role, HOP_SYS, f"PROPERTY: {prop or '(unnamed)'}\nDESCRIBED AS: {desc[:400]}\nTHIS LISTING'S SELLER: {cat.seller(best)}\n"
                           f"LISTING: {cat.title(best)}\nSELLER NOTES: "
                           f"{notes[:1500]}\nPROPERTIES: {Text.json(facts, 1500)}",
                           max_tokens=HOP_TOKENS, deadline=GUARD_AT - 20)
        return raw if isinstance(raw, dict) and set(raw) & set(HOP_FIELDS) else None

    def named_seller(self, text: str, page: dict) -> str | None:
        cat = self.r.cat
        own = Text.of(cat.seller(page)).strip().lower()
        known = {Text.of(cat.seller(it)).strip() for it in cat.items.values()} - {""}
        hit = [x for x in known if x.lower() != own and re.search(rf"(?<!\w){re.escape(x)}(?!\w)", text)]
        if not hit:
            shelves = {w for it in cat.items.values() for w in Text(cat.category(it)).stems}
            titled = set(Text(cat.title(page)).stems) | set(Text(cat.maker(page)).stems)
            hit = [w for w in dict.fromkeys(re.findall(r"(?<=\s)[A-Z][a-z]{3,}(?!\w)", text))
                   if w.lower() != own and Text.stem(w.lower()) not in shelves and Text.stem(w.lower()) not in titled]
        return hit[0] if len(hit) == 1 else None

    def names_other(self, text: str, page: dict, loose: bool) -> bool:
        cat = self.r.cat
        own = Text.of(cat.seller(page)).strip().lower()
        known = {Text.of(cat.seller(it)).strip() for it in cat.items.values()} - {""}
        if any(x.lower() != own and re.search(rf"(?<!\w){re.escape(x)}(?!\w)", text) for x in known):
            return True
        return loose and self.named_seller(text, page) is not None

    def pointer_sentence(self, page: dict, prop: str) -> tuple[str, str] | None:
        if not Text(prop).stems:
            return None
        for part in re.split(r"(?<=[.!?])\s+|\s+\|\s+", self.r.cat.notes_of(page)):
            if not Text(part).has(prop):
                continue
            seller = self.named_seller(part, page)
            if seller:
                return part.strip(), seller
        return None

    def read_pointer(self, best: dict, prop: str, desc: str, notes: str, table: bool = True, role: str = "read") -> tuple[Any, str | None]:
        r = self.r
        cat = r.cat
        raw = self.hop_call(role, best, prop, desc, notes, best["facts"] if table else {})
        self.mapped = isinstance(raw, dict) and bool(Text.of(raw.get("mapping_quote")).strip()) and Text.of(raw.get("mapping_quote")).strip().lower() not in ("null", "none")
        if isinstance(raw, dict) and Text.of(raw.get("pointer_quote")).strip() and not Text.of(raw.get("seller")).strip():
            quote = Text.of(raw["pointer_quote"])
            known = {Text.of(cat.seller(it)).strip() for it in cat.items.values()} - {""}
            hit = [x for x in known if re.search(rf"(?<!\w){re.escape(x)}(?!\w)", quote)]
            hit = hit or list(dict.fromkeys(re.findall(r"(?<=\s)[A-Z][a-z]{3,}(?!\w)", quote)))
            if len(hit) == 1:
                raw = {**raw, "seller": hit[0]}
                r.notes.append(f"pointer seller taken from the quote: {hit[0]!r}")
        pointer, value = hop_adapter(raw, prop or "")
        return self.vet(pointer, best, desc, notes), value

    def vet(self, pointer: Any, page: dict, desc: str, notes: str) -> Any:
        if not isinstance(pointer, dict) or not Text.of(pointer.get("seller")).strip():
            return pointer
        quote = Text.of(pointer.get("description"))
        said = Text(quote).stems
        if said and not Text.run_in(said, Text(notes).stems) and Text.run_in(said, Text(desc).stems):
            self.r.notes.append("pointer dropped: its quote restates the description, not this page")
            return {"denies": False}
        if (Text.of(pointer["seller"]).strip().lower() == Text.of(self.r.cat.seller(page)).strip().lower()
                and not pointer.get("with") and not pointer.get("hint")):
            self.r.notes.append("pointer dropped: it names this page's own seller and no detail")
            return {"denies": False}
        if quote and not Text(quote).has(pointer["seller"]):
            other = self.named_seller(quote, page)
            if other and other.strip().lower() != Text.of(pointer["seller"]).strip().lower():
                self.r.notes.append(f"pointer seller {pointer['seller']!r} is not in its quote; taking {other!r}")
                return {**pointer, "seller": other}
        return pointer

    def guard(self, best: dict, prop: str, desc: str, pointer: Any, memo: str | None, trusted: bool) -> tuple[Any, str | None]:
        r = self.r
        if not trusted or (isinstance(pointer, dict) and pointer.get("seller")):
            return pointer, memo
        found = self.pointer_sentence(best, prop)
        if found is None:
            return pointer, memo
        sentence, seller = found
        self.mapped = False
        if r.t() < GUARD_AT - 25:
            again, _ = hop_adapter(self.hop_call("read", best, prop, desc, sentence, {}), prop or "")
            if isinstance(again, dict) and again.get("seller"):
                r.notes.append(f"pointer re-read from its sentence: {again['seller']!r}")
                return {**again, "denies": False}, None
        r.notes.append(f"pointer taken from the note sentence: {seller!r}")
        return {"seller": seller, "kind": None, "with": {}, "rule": None, "description": sentence, "denies": False}, None

    def read_value(self, it: dict, prop: str) -> str | None:
        r = self.r
        cat = r.cat
        if r.t() > GUARD_AT - 20:
            return None
        raw = r.model_json("read", VALUE_SYS, f"PROPERTY: {prop}\nLISTING: {cat.title(it)}\nPROPERTIES: {Text.json(it['facts'], 1500)}\n"
                           f"SELLER NOTES: {cat.notes_of(it)[:1500]}", max_tokens=120, deadline=GUARD_AT - 20)
        v = Text.of(raw.get("value")).strip() if isinstance(raw, dict) else ""
        if not v or v.lower() in ("null", "none", "unknown"):
            return None
        r.notes.append(f"chain value read for {prop!r}: {v!r}")
        return v[:80]

    def paired(self, page: dict | None, prop: str | None, value: Any) -> Any:
        cat = self.r.cat
        if page is None or not prop or not Text.of(value).strip():
            return value
        table = cat.fact_values(page, prop) if page.get("facts") else []
        if table:
            own = Text.of(table[0][1])
        elif self.own_shown(page, Text.of(value)):
            own = Text.of(value)
        else:
            return value
        mine = tuple(w for w, *_ in pair_tokens(Text.key(prop)))
        vocab = {own, Text.of(value)}
        shelf: set = set()
        keys = set()
        for it in cat.items.values():
            if not it.get("facts"):
                continue
            for k, v in it["facts"].items():
                kt = tuple(w for w, *_ in pair_tokens(Text.key(k)))
                if kt and kt != mine:
                    keys.add(kt)
                if isinstance(v, (str, int, float)) and not isinstance(v, bool) and len(Text.of(v)) <= 60:
                    shelf.add(Text.of(v))
            vocab.update(Text.of(v) for _, v in cat.fact_values(it, prop))
        memo = cat.field(page, cat.store.notes) if cat.store.notes else None
        memo = memo if isinstance(memo, str) and memo.strip() else cat.notes_of(page)
        ot = [w for w, *_ in pair_tokens(own)]
        named: list[str] = []
        for part in re.split(r"(?<=[.!?])\s+", memo):
            have = [w for w, *_ in pair_tokens(part)]
            names = bool(mine) and set(mine) <= set(have)
            holds = any(have[i:i + len(ot)] == ot for i in range(len(have)))
            if not names and not holds:
                continue
            got = pair_clauses(part, own, vocab, names, shelf, prop, keys) or pair_partner(part, own, vocab, keys)
            if not got:
                if names:
                    named.append(part)
                continue
            if not pair_same(got, value):
                self.r.notes.append(f"chain value paired for {prop!r}: {Text.of(value)!r} -> {got!r} (own {own!r})")
                return got
            return value
        got = self.pair_read(page, prop, own, named, keys) if named else None
        if got and not pair_same(got, value):
            self.r.notes.append(f"chain value paired by reading for {prop!r}: {Text.of(value)!r} -> {got!r} (own {own!r})")
            return got
        return value

    def own_shown(self, page: dict, value: str) -> bool:
        cat = self.r.cat
        want = [w for w, *_ in pair_tokens(value)]
        if not want:
            return False
        for text in [cat.title(page)] + [Text.of(v) for v in (page.get("facts") or {}).values()]:
            have = [w for w, *_ in pair_tokens(Text.of(text))]
            if any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1)):
                return True
        return False

    def pair_read(self, page: dict, prop: str, own: str, parts: list[str], keys: Any = ()) -> str | None:
        r = self.r
        if r.t() > GUARD_AT - 20:
            return None
        note = " ".join(parts)[:900]
        raw = r.model_json("read", PAIR_SYS, f"PROPERTY: {prop}\nOWN VALUE: {own}\nNOTE: {note}", max_tokens=60, deadline=GUARD_AT - 20)
        got = Text.of(raw.get("partner")).strip().strip(".,;:") if isinstance(raw, dict) else ""
        if not got or len(got) > 60 or got.lower() not in note.lower():
            return None
        gt = [w for w, *_ in pair_tokens(got)]
        ot = [w for w, *_ in pair_tokens(own)]
        mine = [w for w, *_ in pair_tokens(Text.key(prop))]
        if not gt or gt == ot or any(gt[i:i + len(ot)] == ot for i in range(len(gt) - len(ot) + 1)) or (mine and set(gt) <= set(mine)):
            return None
        if tuple(gt) in {tuple(k) for k in keys} or any(pair_same(got, v) for v in (page.get("facts") or {}).values()):
            return None
        return got

    def walk(self) -> None:
        r = self.r
        r.plan.begin("chains", "chains")
        two_tools = bool(r.tm.get("specs") and r.tm.get("detail"))
        for entry in [x for x in r.ledger if x["active"] and x["op"] == "ref" and not x.get("done")]:
            if not entry["active"]:
                continue
            entry["done"] = True
            ref = entry.get("ref") or {}
            twin = (entry["a"], Text.key(ref.get("seller")), frozenset((Text.key(k), Text.key(v)) for k, v in clean_with(ref.get("with")).items()))
            if twin[1] and twin in self.traced:
                entry["active"] = False
                r.notes.append(f"chain reference to {ref.get('seller')!r} repeats one already traced; skipped")
                continue
            mark = len(self.log)
            prop, value = self.trace(entry, two_tools)
            raw = value
            self.restate(self.log[mark:])
            if not value and entry.get("heard"):
                self.plain_value(entry, self.log[mark:])
            if value:
                self.traced.add(twin)
            if value and prop:
                value = self.paired(r.cat.items.get(entry.get("settled") or ""), prop, value)
                ref = entry.get("ref") or {}
                rel = Text.of(ref.get("relation")).lower()
                ai = entry["a"] if isinstance(entry["a"], int) else 0
                req = self.ref_req(ref, rel, prop, value)
                if rel == "near" and req["op"] != "between":
                    req = self.near_req(r.cat.items.get(entry.get("settled") or ""), ref, prop, value, req, two_tools, pair_same(raw, value))
                r.add_req(ai, req, "reference")
            r.plan.replan("chain done")

    def restate(self, hops: list) -> None:
        cat = self.r.cat
        for hop in hops:
            page = cat.items.get(Text.of(hop.get("best"))) if isinstance(hop, dict) else None
            if page is None or not page.get("read") or page["id"] not in (hop.get("full") or []):
                continue
            for k, v in (hop.get("with") or {}).items():
                v = Text.of(v).strip()
                keys = [fk for fk, _ in cat.fact_values(page, k) if fk in page["facts"]]
                if v and len(keys) == 1 and len(Text(v).numbers) == 1 and self.unit_blind(page, k, v):
                    page["facts"][keys[0]] = f"{page['facts'][keys[0]]} ({v})"
                    self.r.notes.append(f"chain page {page['id']}: {keys[0]!r} also stated as {v!r} by the shopper")

    def plain_value(self, entry: dict, hops: list) -> None:
        r = self.r
        ref = entry.get("ref") or {}
        seller = Text.of(ref.get("seller")).strip()
        first = next((h for h in hops if isinstance(h, dict)), {})
        if first.get("best") and seller and clean_with(ref.get("with")):
            return
        quote = Text.of(entry.get("quote")).strip()
        heard = [h for h in r.heard if h.get("kind") == "reference"
                 and ((quote and quote_in(quote, h.get("r"))) or (not quote and seller and value_in(seller, h.get("r"))))]
        if not heard:
            return
        h = heard[-1]
        got = r.asker.plain_read(h["a"], h["q"], h["prop"], h["r"], quote, ref)
        if not got:
            return
        ai = entry["a"] if isinstance(entry["a"], int) else 0
        r.add_req(ai, {"prop": entry["prop"], "op": got["op"], "value": ledger_value(got["values"]), "unit": got.get("unit"),
                       "quote": got.get("quote")}, "revealed")
        entry["value_ok"] = True

    def near_req(self, page: dict | None, ref: dict, prop: str, value: str, req: dict, two_tools: bool, own: bool = False) -> dict:
        r = self.r
        if page is None:
            return req
        if two_tools and not page.get("read") and r.steps_left() > r.plan.finish_need(minimal=True) + 1 and r.t() <= GUARD_AT - 30:
            r.hunt.open([page["id"]], what="facts")
        alts = [v for _, v in r.cat.fact_values(page, prop) if Text(v).numbers and not pair_same(v, value)]
        if own and page.get("read") and not any(Text.key_stems(k) == Text.key_stems(prop) for k in page["facts"]):
            page["facts"][prop] = Text.of(value)
        for v in alts:
            got = self.ref_req(ref, "near", prop, v)
            if got["op"] == "between":
                return got
        if not alts:
            return req
        r.notes.append(f"chain value for {prop!r}: tolerance unit differs from the page's units; any of the page's own values")
        return {**req, "value": list(dict.fromkeys([Text.of(value)] + alts))}

    def settle_entry(self, entry: dict, prop: str | None, value: str) -> None:
        r = self.r
        ref = entry.get("ref") or {}
        rel = Text.of(ref.get("relation")).lower()
        ai = entry["a"] if isinstance(entry["a"], int) else 0
        r.add_req(ai, self.ref_req(ref, rel, prop, value), "reference")
        entry["value_ok"] = True

    @staticmethod
    def ref_req(ref: dict, rel: str, prop: str | None, value: str) -> dict:
        if rel == "near":
            span = Chains.near_span(Text.of(ref.get("rule")), value)
            if span is not None:
                lo, hi, unit = span
                return {"prop": prop, "op": "between", "value": [f"{lo} {unit}".strip(), f"{hi} {unit}".strip()], "unit": unit or None}
        return {"prop": prop, "op": rel if rel in ("not", "min", "max") else "is", "value": value}

    @staticmethod
    def near_span(rule: str, value: str) -> tuple[str, str, str] | None:
        tol = Text(rule).numbers
        got = Text(value).numbers
        if not tol or not got:
            return None
        n, nu = tol[0]
        v, vu = got[0]
        if nu and vu and Text.stem(nu.lower()) != Text.stem(vu.lower()):
            return None

        def show(x: float) -> str:
            return f"{round(x, 4):.4f}".rstrip("0").rstrip(".")
        return show(v - n), show(v + n), vu

    def audit(self) -> bool:
        r = self.r
        cat = r.cat
        two_tools = bool(r.tm.get("specs") and r.tm.get("detail"))
        what = "state" if two_tools else "facts"
        changed = False
        for entry in [x for x in r.ledger if x["active"] and x["op"] == "ref" and not x.get("value_ok")]:
            if entry.get("audits", 0) >= 2:
                continue
            opened = set(entry.get("opened") or [])
            address = {k: Text.of(v) for k, v in (entry.get("last_with") or {}).items()
                       if Text.of(v).strip() and Text.key_stems(k) != Text.key_stems(entry.get("prop") or "")}

            def addressed(it: dict) -> bool:
                return all(cat.verdict(it, {"prop": k, "op": "is", "value": v}) == "yes" for k, v in address.items())
            pending = [i for i in entry.get("last_own") or [] if i in cat.items and i not in opened]
            if not pending:
                continue
            entry["audits"] = entry.get("audits", 0) + 1
            prop = entry["prop"]
            for pid in pending:
                if r.plan.slack() < 1 or r.t() > GUARD_AT - 25:
                    break
                it = cat.items[pid]
                if not (it.get("paged") or (not two_tools and it.get("read"))):
                    r.hunt.open([pid], what=what)
                entry.setdefault("opened", []).append(pid)
                changed = True
                if not (it.get("paged") or it.get("read")):
                    continue
                notes = cat.notes_of(it)
                if notes:
                    desc = " ".join([Text.of((entry.get("ref") or {}).get("kind")) or r.articles[entry["a"] if isinstance(entry["a"], int) else 0]["noun"]]
                                    + [f"{k} {v}" for k, v in ((entry.get("ref") or {}).get("with") or {}).items()])
                    pointer, memo = self.read_pointer(it, prop, desc, notes)
                    pointer, memo = self.guard(it, prop, desc, pointer, memo, bool(address) and addressed(it))
                    table = cat.fact_values(it, prop) if prop and it.get("read") else []
                    if memo and table and not getattr(self, "mapped", False):
                        memo = table[0][1]
                    if memo and prop and addressed(it):
                        self.settle_entry(entry, prop, self.paired(it, prop, memo))
                        entry["settled"] = pid
                        break
                    if isinstance(pointer, dict) and pointer.get("seller") and r.plan.slack() >= 3:
                        entry["ref"] = {**(entry.get("ref") or {}), "seller": pointer["seller"], "kind": pointer.get("kind"),
                                        "with": pointer.get("with") or {}, "rule": pointer.get("rule")}
                        prop2, value = self.trace(entry, two_tools)
                        if value and prop2:
                            self.settle_entry(entry, prop2, self.paired(cat.items.get(entry.get("settled") or ""), prop2, value))
                        break
                if not addressed(it):
                    continue
                vals = cat.fact_values(it, prop) if prop else self.quoted_fact(it, entry)
                if vals:
                    self.settle_entry(entry, prop or vals[0][0], self.paired(it, prop or vals[0][0], vals[0][1]))
                    entry["settled"] = pid
                    break
        return changed

    def quoted_fact(self, it: dict, entry: dict) -> list[tuple[str, str]]:
        said = Text(Text.of(entry.get("quote"))).stems
        named = [k for k in it["facts"] if Text.key_stems(k) and Text.run_in(Text.key_stems(k), said)]
        if not named:
            return []
        key = max(named, key=lambda k: len(Text.key(k)))
        return [(key, str(it["facts"][key]))]

    def trace(self, entry: dict, two_tools: bool) -> tuple[str | None, str | None]:
        try:
            return self.trace_hops(entry, two_tools)
        finally:
            self.r.plan.extra = 0

    def trace_hops(self, entry: dict, two_tools: bool) -> tuple[str | None, str | None]:
        r = self.r
        cat = r.cat
        ref = entry.get("ref") or {}
        ai = entry["a"] if isinstance(entry["a"], int) else 0
        seller = Text.of(ref.get("seller")).strip()
        kind = Text.of(ref.get("kind")).strip() or r.articles[ai]["noun"]
        withs = ref.get("with") if isinstance(ref.get("with"), dict) else {}
        rule = Text.of(ref.get("rule")).strip()
        prop = entry["prop"]
        desc = " ".join([kind] + [f"{k} {v}" for k, v in withs.items()])
        what = "state" if two_tools else "facts"

        def seen(it: dict) -> bool:
            return bool(it.get("read") or it.get("paged"))

        came_from = None
        hint: dict = {}
        src = None
        tried: set[str] = set()
        for hop_n in range(8):
            if seller and self.room() < 3:
                r.plan.extra = r.plan.hop_relief()
            tight = bool(seller) and self.room() < 3 and r.plan.chain_reserve() >= 2
            if not seller or (self.room() < 3 and not tight) or r.t() > GUARD_AT - 30:
                r.plan.extra = 0
                self.log.append({"prop": prop, "stop": "no seller" if not seller else "budget", "room": self.room(),
                                 "reserve": r.plan.chain_reserve()})
                return prop, None
            withs = {k: v for k, v in withs.items() if Text.of(v).strip()}
            words = [] if withs else self.detail_words(desc, seller, kind, prop)
            needs = [(k, Text.of(v)) for k, v in withs.items()] or [("", w) for w in words]
            asked = withs or ({prop or kind: " ".join(words)} if words else {})
            self.valued_ids = set()
            lead_q, lead = (self.lead_rows(seller, [v for _, v in needs], tight, (self.sieve_words(kind, withs) or [""])[0])
                            if tight or r.finder.match() == "or" else ("", []))
            lead = [it for it in lead if it["id"] != came_from]
            hop = {"prop": prop, "seller": seller, "kind": kind, "with": withs, "rule": rule, "queries": [lead_q] if lead_q else [],
                   "own": [], "sieve": False}
            if words:
                hop["words"] = words
            if tight:
                hop["tight"] = True
            self.log.append(hop)
            wwords = [Text(self.spaced(v)).stems for _, v in needs]
            real_rule = self.is_rule(rule, hop_n > 0)

            def row_text(it: dict) -> str:
                return " ".join([cat.title(it), cat.category(it)] + [str(v) for v in it["row"].values() if isinstance(v, str)])
            dwords = {w for w in Text(desc + " " + " ".join(Text.of(v) for v in hint.values())).stems if len(w) >= 3 or w.isdigit()}

            said_by: dict = {}

            def own_said(it: dict, text: str, loose: bool) -> str:
                if it.get("read"):
                    return text
                if (it["id"], text, loose) not in said_by:
                    said_by[(it["id"], text, loose)] = " ".join(p for p in re.split(r"(?<=[.!?])\s+", text) if not self.names_other(p, it, loose))
                return said_by[(it["id"], text, loose)]

            def positive_text(it: dict) -> str:
                plain = " ".join([cat.title(it)] + [str(v) for v in it["facts"].values()])
                return plain + " " + " . ".join(own_said(it, v, k == cat.store.notes) for k, v in it["rec"].items() if isinstance(v, str))

            def with_score(it: dict) -> int:
                text = positive_text(it)
                hay = Text(text).stems
                flat = Text(self.spaced(text)).stems
                memo_text = own_said(it, Text.of(cat.field(it, cat.store.notes)), True)
                memo = Text(self.spaced(memo_text)).stems
                bits = ([set(memo)] if it.get("read") or it["facts"]
                        else [set(Text(self.spaced(b)).stems) for b in re.split(r"[;!?]|[.,](?!\d)", memo_text)])
                shelf = set(Text(self.spaced(cat.title(it) + " " + cat.category(it))).stems)
                own_key = Text.key_stems(prop or "")
                score = 0
                blinds = 0
                for k, v in needs:
                    say = "unknown" if not k or (own_key and Text.key_stems(k)[-len(own_key):] == own_key) else cat.verdict(it, {"prop": k, "op": "is", "value": v})
                    if say == "no" and any(self.frac_same(v, fv) for _, fv in cat.fact_values(it, k)):
                        say = "yes"
                    vs = Text(self.spaced(v)).stems
                    said = (Text.run_in(Text(v).stems, hay) or (not k and Text.run_in(vs, flat)) or self.counted_in(v, hay)
                            or (len(vs) > 1 and any(set(vs) <= b for b in bits)))
                    if say == "yes" or (say != "no" and said) or (say == "no" and Text.run_in(vs, memo) and not set(vs) <= shelf):
                        score += 1
                    elif say == "unknown" and k and names_prop(it) and self.unit_blind(it, k, v):
                        blinds += 1
                return score + (1 if blinds == 1 and score >= 1 else 0)

            def contradicted(it: dict) -> bool:
                own_key = Text.key_stems(prop or "")
                shown = Text(" ".join([cat.title(it)] + [str(v) for v in it["facts"].values()])).stems
                for k, v in withs.items():
                    v = Text.of(v)
                    vs = Text(v).stems
                    if not vs or Text.run_in(vs, shown) or (own_key and Text.key_stems(k)[-len(own_key):] == own_key):
                        continue
                    req = {"prop": k, "op": "is", "value": v}
                    vals = cat.fact_values(it, k) or (self.near_keys(it, k) if Text(v).numbers else [])
                    if not vals or cat.verdict(it, req) == "yes" or any(self.frac_same(v, fv) for _, fv in vals):
                        continue
                    nums = bool(Text(v).numbers)
                    kk = set(Text.key_stems(k))
                    if all(self.num_no(req, v, fv) if nums and Text(fv).numbers
                           else not nums and not Text(fv).numbers and self.apart(kk, v, fv) for _, fv in vals):
                        return True
                return False

            def denied(it: dict) -> bool:
                return contradicted(it)

            def names_prop(it: dict) -> bool:
                return bool(prop) and Text(cat.notes_of(it)).has(prop)

            def bare(it: dict) -> bool:
                return not names_prop(it) and not Text.of(cat.field(it, cat.store.notes)).strip()

            def weight(it: dict) -> float:
                hay = set(Text(" ".join([cat.title(it), cat.category(it)] + list(it["facts"].values()) + list(it["facts"]))).stems)
                return with_score(it) + (2.0 if names_prop(it) else 0.0) + 0.2 * len(dwords & hay) - (5 if it.get("decoy") else 0)

            best = None
            full: list[dict] = []
            after_full = 0
            scanned: set[str] = set()

            def scan(cands: list[dict]) -> bool:
                nonlocal best, after_full
                for cand in cands:
                    budget = max(self.room(), r.plan.chain_reserve()) if tight else self.room()
                    lean = bool(full) and all(bare(x) for x in full)
                    if budget < (2 if (full and not lean) or not needs else 1) and best is not None:
                        return True
                    if full and (budget < (1 if lean else 3) or after_full >= 2):
                        return True
                    scanned.add(cand["id"])
                    if not seen(cand):
                        if tight and self.room() < 1 and r.plan.chain_reserve() < 1:
                            return True
                        r.hunt.open([cand["id"]], what=what, force=tight and self.room() < 1)
                        after_full += bool(full)
                    if not seen(cand):
                        continue
                    if denied(cand):
                        cand["decoy"] = True
                        hop.setdefault("decoys", []).append(cand["id"])
                        continue
                    if needs and with_score(cand) >= len(needs):
                        full.append(cand)
                        lean = all(bare(x) for x in full)
                        more = [it for it in cands if it is not cand and not seen(it) and not it.get("decoy")
                                and (lean or it["id"] in self.valued_ids or all(Text.run_in(v, Text(self.spaced(row_text(it))).stems) for v in wwords if v))]
                        if not real_rule and (not more or len(full) >= 3 or budget < (2 if lean else 3)):
                            return True
                    if best is None or weight(cand) > weight(best):
                        best = cand
                    if not needs and not rule and (names_prop(cand) or (len(cands) == 1 and cat.fact_values(cand, prop))):
                        return True
                return False

            own = list(lead)
            done = scan(lead[:LEAD_ROWS] if not tight else lead[:10])
            sieved = None
            if not done and not full and not tight:
                sieved = self.sieve_pool(seller, kind, asked)
                queries = list(self.sieve_log) if sieved is not None else self.queries_for(seller, kind, asked)
                pool = sieved if sieved is not None else self.seller_pool(seller, queries)
                rank = {it["id"]: n for n, it in enumerate(pool)} if sieved is not None else {}
                pool = [it for it in pool if it["id"] != came_from]
                pool.sort(key=lambda it: (rank.get(it["id"], 0), -sum(1 for v in wwords if v and Text.run_in(v, Text(self.spaced(row_text(it))).stems))))
                have = {it["id"] for it in pool}
                own = pool + [it for it in lead if it["id"] not in have]
                hop["queries"] = (hop["queries"] + queries)[:5]
                hop["sieve"] = sieved is not None
                done = scan([it for it in own if it["id"] not in scanned][:10])
            hop["own"] = [it["id"] for it in own][:10]
            entry["last_own"] = [it["id"] for it in own][:10]
            entry["last_with"] = dict(withs)
            if not own:
                fixed = self.repair_seller(seller, src, prop) if seller.lower() not in tried else None
                if fixed and fixed.lower() not in tried:
                    tried.update({seller.lower(), fixed.lower()})
                    r.notes.append(f"chain hop seller {seller!r} found no rows; trying {fixed!r}")
                    seller = fixed
                    continue
                if tight:
                    return prop, None
            if not done and not full and not tight and self.room() >= 3 and r.t() <= GUARD_AT - 30:
                rest = [it for it in lead + own if it["id"] not in scanned and not seen(it) and not it.get("decoy")]
                rest = [it for n, it in enumerate(rest) if it["id"] not in {x["id"] for x in rest[:n]}]
                if sieved is not None:
                    taken = {it["id"] for it in own}
                    rest += [it for it in self.seller_pool(seller, self.queries_for(seller, kind, asked))
                             if it["id"] != came_from and it["id"] not in taken and not it.get("decoy")]
                if rest:
                    hop["widened"] = [it["id"] for it in rest][:10]
                    own += [it for it in rest if it not in own]
                    scan(rest[:10])
            if needs and not full and not tight:
                name = " ".join(Text(seller).words)
                low = seller.strip().lower()
                marks = {t for _, v in needs for t in Text(self.spaced(v)).stems if any(c.isdigit() for c in t)}

                def fits_row(it: dict) -> tuple[int, int]:
                    hay = Text(self.spaced(row_text(it))).stems
                    return sum(1 for v in wwords if v and Text.run_in(v, hay)), len(marks & set(hay))
                for n_ in range(4):
                    if full or not name or self.room() < 3 or r.t() > GUARD_AT - 30:
                        break
                    st = r.finder.state(name)
                    ids = list(st.ids) if n_ == 0 and st is not None else (r.finder.run(name)[0] if st is None else r.finder.page_on(name, 1))
                    if not ids and n_:
                        break
                    fresh = [cat.items[i] for i in dict.fromkeys(ids) if i in cat.items and i not in scanned and i != came_from
                             and not cat.items[i].get("decoy") and (low in cat.seller(cat.items[i]).lower() or low in cat.maker(cat.items[i]).lower())]
                    own += [it for it in fresh if it not in own]
                    hop.setdefault("dug", []).append(len(fresh))
                    ranked = sorted((it for it in fresh if any(fits_row(it))), key=fits_row, reverse=True)
                    if ranked:
                        scan(ranked[:3])
                    elif fresh and self.room() >= 10:
                        scan(fresh[:self.room() - 6])
            if two_tools and not tight and len(full) >= 2:
                for it in sorted(full, key=weight, reverse=True):
                    if not it.get("read") and self.room() >= 2 and r.t() <= GUARD_AT - 30:
                        r.hunt.open([it["id"]], what="facts")
                        hop.setdefault("compared", []).append(it["id"])
                    if it.get("read") and denied(it) and any(x is not it and not x.get("decoy") for x in full):
                        it["decoy"] = True
                        hop.setdefault("decoys", []).append(it["id"])
                full = [it for it in full if not it.get("decoy")]
            if full:
                best = max(full, key=weight)
            hop["full"] = [it["id"] for it in full]
            if needs and not full and best is not None and not names_prop(best) and with_score(best) < max(1, len(needs) - 1):
                hop["weak"] = best["id"]
                r.notes.append(f"chain hop {seller!r}: best page {best['id']} matches too few details; not settled")
                best = None
            if needs and not full and best is not None:
                hop["unverified"] = True
                r.notes.append(f"chain hop {seller!r} settled on an unverified page {best['id']}")
            if len(full) >= 2 and ((real_rule and self.room() >= 1) or (not real_rule and r.t() <= GUARD_AT - 30)):
                pick = self.pick_listing(desc + (f" RULE: {rule}" if real_rule else ""), full)
                if pick is not None:
                    best = pick
            pointer = None
            memo_value = None
            tries = 0
            while best is not None and seen(best) and tries < 3:
                tries += 1
                entry.setdefault("opened", []).append(best["id"])
                best["chain"] = True
                hop.update({"best": best["id"], "weight": round(weight(best), 2)})
                lazy = (two_tools and not best.get("read") and best in full and bool(cat.notes_of(best))
                        and not any(it is not best and not it.get("decoy") for it in full))
                if two_tools and not best.get("read") and not lazy and (self.room() >= 1 or (tight and r.plan.chain_reserve() >= 1)):
                    r.hunt.open([best["id"]], what="facts", force=self.room() < 1)
                    if best.get("read") and denied(best) and any(it is not best and not it.get("decoy") for it in full):
                        best["decoy"] = True
                        hop.setdefault("decoys", []).append(best["id"])
                        best = max([it for it in full if not it.get("decoy")], key=weight)
                        continue
                    if best.get("read") and denied(best) and needs and best not in full:
                        best["decoy"] = True
                        hop.setdefault("decoys", []).append(best["id"])
                        r.notes.append(f"chain hop {seller!r}: unverified page {best['id']} contradicts the pointer; not settled")
                        best = None
                        break
                notes = cat.notes_of(best)
                pointer = memo_value = None
                if notes:
                    pointer, memo_value = self.read_pointer(best, prop, desc, notes)
                    away = (isinstance(pointer, dict) and Text.of(pointer.get("seller")).strip()
                            and Text.key(pointer.get("seller")) != Text.key(cat.seller(best)))
                    if lazy and not away and (self.room() >= 1 or (tight and r.plan.chain_reserve() >= 1)):
                        r.hunt.open([best["id"]], what="facts", force=self.room() < 1)
                        if best.get("read"):
                            pointer, memo_value = self.read_pointer(best, prop, desc, notes)
                    if (isinstance(pointer, dict) and pointer.get("denies") is True and not pointer.get("seller")
                            and best in full and not denied(best) and best["facts"]):
                        again, again_value = self.read_pointer(best, prop, desc, notes, table=False)
                        if isinstance(again, dict) and again.get("seller"):
                            pointer, memo_value = again, again_value
                    pointer, memo_value = self.guard(best, prop, desc, pointer, memo_value, best in full or (not needs and len(own) == 1))
                    hop["next"] = pointer
                    if isinstance(pointer, dict) and pointer.get("denies") is True and len(own) > 1 and (not pointer.get("seller") or denied(best)):
                        best["decoy"] = True
                        hop.setdefault("decoys", []).append(best["id"])
                        full = [it for it in full if not it.get("decoy")]
                        more = [it for it in own[:10] if not it.get("decoy") and it["id"] not in scanned]
                        if needs and not full and more and self.room() >= 2 and r.t() <= GUARD_AT - 30:
                            scan(more)
                            full = [it for it in full if not it.get("decoy")]
                        if full:
                            best = max(full, key=weight)
                            pointer = memo_value = None
                            continue
                        unseen = [it for it in own[:10] if not it.get("decoy") and not seen(it)]
                        rest = [it for it in own[:10] if not it.get("decoy")]
                        best = unseen[0] if unseen else (max(rest, key=weight) if rest else None)
                        if best is not None and not seen(best) and self.room() >= 2:
                            r.hunt.open([best["id"]], what=what)
                        pointer = memo_value = None
                        continue
                    if not isinstance(pointer, dict) or not pointer.get("seller"):
                        pointer = None
                    elif self.self_loop(pointer, seller, kind, withs) or self.describes(pointer, best):
                        pointer = None
                break
            if best is None or not seen(best):
                return prop, None
            if two_tools and not best.get("paged") and r.plan.slack() >= 1 and r.t() <= GUARD_AT - 30:
                r.hunt.open([best["id"]], what="state", force=True)
            if (pointer is None and not memo_value and prop and best in full and names_prop(best)
                    and not cat.fact_values(best, prop) and self.room() >= 0):
                again, again_value = self.read_pointer(best, prop, desc, cat.notes_of(best), role="decide")
                if again_value:
                    memo_value = again_value
                elif isinstance(again, dict) and again.get("seller") and not self.self_loop(again, seller, kind, withs):
                    pointer = again
                    hop["next"] = pointer
            if pointer is None:
                table = cat.fact_values(best, prop) if prop and best.get("read") else []
                if memo_value and prop and (getattr(self, "mapped", False) or not table):
                    hop["value"] = memo_value
                    entry["settled"] = best["id"]
                    entry["value_ok"] = bool(best.get("paged")) or not two_tools
                    return prop, memo_value
                if not table and two_tools and not best.get("read") and self.room() >= 1:
                    r.hunt.open([best["id"]], what="facts")
                if not table and r.tm.get("detail") and not best.get("paged") and r.plan.slack() >= 1:
                    r.hunt.open([best["id"]], what="state", force=True)
                vals = cat.fact_values(best, prop) if prop else self.quoted_fact(best, entry)
                if vals:
                    hop["value"] = vals[0][1]
                    hop["prop"] = prop or vals[0][0]
                    hop["from"] = "table"
                    entry["settled"] = best["id"]
                    entry["value_ok"] = bool(best.get("paged")) or not two_tools
                    return prop or vals[0][0], vals[0][1]
                found = self.read_value(best, prop) if prop else None
                if found:
                    hop["value"] = found
                    hop["from"] = "value read"
                    entry["settled"] = best["id"]
                    entry["value_ok"] = bool(best.get("paged")) or not two_tools
                    return prop, found
                return prop, None
            r.plan.note_hop(hop_n + 1)
            src = best
            seller, kind = Text.of(pointer.get("seller")), Text.of(pointer.get("kind")) or kind
            withs = pointer.get("with") if isinstance(pointer.get("with"), dict) else {}
            hint = pointer.get("hint") if isinstance(pointer.get("hint"), dict) else {}
            came_from = None if withs and all(cat.verdict(best, {"prop": k, "op": "is", "value": Text.of(v)}) == "yes"
                                              for k, v in withs.items()) else best["id"]
            rule = Text.of(pointer.get("rule")).strip()
            desc = Text.of(pointer.get("description")) or " ".join([kind] + [f"{k} {v}" for k, v in withs.items()])
        return prop, None

    @staticmethod
    def abbrev(a: str, b: str) -> bool:
        short, long_ = sorted((a.strip().strip(".").lower(), b.strip().strip(".").lower()), key=len)
        return len(short) == 1 and long_.isalpha() and len(long_) >= 3 and long_[0] == short

    def num_no(self, req: dict, v: str, fv: str) -> bool:
        got = Catalog.number_verdict(req, [v], fv)
        if got != "unknown":
            return got == "no"
        want = Text(v).numbers
        same = [x for x, u in Text(fv).numbers if len(want) == 1 and u.strip() and self.abbrev(want[0][1], u)]
        return bool(same) and all(abs(x - want[0][0]) > 1e-6 for x in same)

    def unit_blind(self, it: dict, k: str, v: str) -> bool:
        want = Text(v).numbers
        if len(want) != 1 or not want[0][1].strip():
            return False
        wu = want[0][1]
        texts = [fv for _, fv in self.r.cat.fact_values(it, k)]
        texts += [p for p in re.split(r"(?<=[.!?])\s+", self.r.cat.notes_of(it)) if Text(p).has(k)]
        for t in texts:
            got = [u for _, u in Text(t).numbers if u.strip()]
            if got and all(Text.unit_scale(u, wu) is None and not self.abbrev(u, wu) for u in got):
                return True
        return False

    @staticmethod
    def near_keys(it: dict, k: str) -> list[tuple[str, str]]:
        want = Text.key_stems(k)
        out = []
        for fk, fv in it["facts"].items():
            ks = Text.key_stems(fk)
            if ks and want and ks[-1] == want[-1] and (set(ks) <= set(want) or set(want) <= set(ks)):
                out.append((fk, str(fv)))
        return out

    @staticmethod
    def frac_same(a: Any, b: Any) -> bool:
        def nums(s: str) -> list[float]:
            got = [int(x) / int(y) for x, y in re.findall(r"(?<![\d.,/])(\d{1,3})\s*/\s*(\d{1,3})(?![\d.,/])", s) if int(y)]
            plain = Text(s).numbers
            return got or ([plain[0][0]] if len(plain) == 1 else [])
        sa, sb = Text.of(a), Text.of(b)
        if not re.search(r"\d\s*/\s*\d", sa + " " + sb):
            return False
        return any(abs(x - y) < 1e-6 for x in nums(sa) for y in nums(sb))

    @staticmethod
    def apart(kk: set, v: str, fv: str) -> bool:
        a, b = set(Catalog.content_stems(v)), set(Text(fv).stems)
        if a - kk and b - kk:
            a, b = a - kk, b - kk
        return not a & b

    def repair_seller(self, seller: str, src: dict | None, prop: str | None) -> str | None:
        cat = self.r.cat
        name = seller.strip()
        if not name:
            return None
        known = {Text.of(cat.seller(it)).strip() for it in cat.items.values()} - {""}
        low = {k.lower(): k for k in known}
        if name.lower() not in low:
            ws = name.split()
            for k in range(len(ws) - 1, 0, -1):
                pre = " ".join(ws[:k]).lower()
                if pre in low:
                    return low[pre]
        if src is not None and name.lower() == Text.of(cat.seller(src)).strip().lower():
            got = self.pointer_sentence(src, prop) if prop else None
            other = got[1] if got else self.named_seller(cat.notes_of(src), src)
            if other and other.strip().lower() != name.lower():
                return other.strip()
        return None

    @staticmethod
    def counted_in(value: str, hay: list[str]) -> bool:
        st = Text(value).stems
        runs = [st[i:i + 2] for i in range(len(st) - 1) if any(c.isdigit() for c in st[i]) and not any(c.isdigit() for c in st[i + 1])]
        return len(runs) >= 2 and all(Text.run_in(x, hay) for x in runs)

    def describes(self, pointer: dict, it: dict) -> bool:
        cat = self.r.cat
        if Text.of(pointer.get("seller")).strip().lower() != Text.of(cat.seller(it)).strip().lower():
            return False
        withs = pointer.get("with") if isinstance(pointer.get("with"), dict) else {}
        return bool(withs) and all(cat.verdict(it, {"prop": k, "op": "is", "value": Text.of(v)}) == "yes" for k, v in withs.items())

    @staticmethod
    def self_loop(pointer: dict, seller: str, kind: str, withs: dict) -> bool:
        if Text.of(pointer.get("seller")).strip().lower() != seller.strip().lower():
            return False
        if Text.key(Text.of(pointer.get("kind")) or kind) != Text.key(kind):
            return False
        theirs = {Text.key(k): Text.key(Text.of(v)) for k, v in (pointer.get("with") or {}).items()}
        ours = {Text.key(k): Text.key(Text.of(v)) for k, v in withs.items()}
        return theirs == ours


class Judge:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.decisions: list[dict] = []
        self.stamp: tuple = ()
        self.tried: set[tuple] = set()
        self.asked = 0
        self.log: list[str] = []
        self.weak = False
        self.fit_reads = 0

    @staticmethod
    def sig(req: dict) -> tuple:
        return (Text.key(req.get("prop")), Text.of(req.get("op")), Text.json(req.get("value")), Text.of(req.get("unit")).strip().lower())

    def judgeable(self, ai: int) -> list[dict]:
        keys = {tuple(Text.key_stems(k)) for it in self.r.owners(ai) for k in it["facts"]}
        out = []
        for q in self.r.reqs(ai):
            wanted = Catalog.wanted_of(q)
            if any(code_shaped(w) for w in wanted):
                continue
            if any(Text(Text.of(w)).codes for w in wanted) and tuple(Text.key_stems(q.get("prop"))) not in keys:
                continue
            out.append(q)
        return out

    @staticmethod
    def level(it: dict, req: dict) -> str | None:
        jl = it.get("jl")
        return jl.get(Judge.sig(req)) if jl else None

    def line_level(self, it: dict, ai: int, reqs: list[dict] | None = None) -> str | None:
        if reqs is None and any(self.r.cat.code_verdict(it, q) == "no" for q in self.code_reqs(ai)):
            return "no"
        reqs = self.judgeable(ai) if reqs is None else reqs
        got = [self.level(it, q) for q in reqs]
        if not got or None in got:
            return None
        return max(got, key=JUDGE_LEVELS.index)

    def judged(self, it: dict, ai: int) -> bool:
        return bool(it.get("jl")) and any(self.level(it, q) is not None for q in self.judgeable(ai))

    def tier_of(self, it: dict, ai: int) -> int:
        r = self.r
        if not r.ok_basic(it, ai) or r.contradicted(it, ai):
            return 3
        seen = {r.cat.verdict(it, q) for q in r.reqs(ai)}
        if "unknown" in seen:
            return 2
        return 1 if "close" in seen else 0

    @staticmethod
    def flat(s: Any) -> str:
        return " ".join(norm(s).replace("_", " ").split())

    @staticmethod
    def found(span: str, page: str) -> bool:
        hay = Judge.flat(page)
        rows = [Judge.flat(x) for x in page.splitlines()]
        parts = [p.strip(" .,;:-'\"") for p in re.split(r"\.\.\.|…", Judge.flat(span))]
        parts = [p for p in parts if p]
        if not parts:
            return False
        for p in parts:
            if p in hay:
                continue
            cut = re.split(r"\s*[:=]\s*", p, maxsplit=1)
            if len(cut) == 2 and all(cut) and any(cut[0] in row and cut[1] in row for row in rows):
                continue
            return False
        return True

    def page_of(self, it: dict) -> str:
        cat = self.r.cat
        st = cat.store
        skip = {st.id, st.title, st.notes, st.desc, st.price, st.stock, st.category, st.attrs, st.rows, st.total, st.cursor}
        lines = [f"TITLE: {cat.slot_text(it, 'title')[:300]}", f"SHELF: {cat.category(it)}", "FACTS:"]
        keys: set[str] = set()
        for k, v in it["facts"].items():
            lines.append(f"- {k}: {Text.of(v)[:300]}")
            keys.add(Text.key(k))
        for k, v in list(it["rec"].items()) + list(it["row"].items()):
            text = Text.of(v).strip()
            if k in skip or Text.key(k) in keys or not text or len(text.split()) >= 6:
                continue
            keys.add(Text.key(k))
            lines.append(f"- {k}: {text[:300]}")
        long = it.get("long") or {}
        notes = Text.of(cat.field(it, st.notes) or long.get(st.notes or "")).strip()
        desc = Text.of(cat.field(it, st.desc) or long.get(st.desc or "")).strip()
        if not notes and not desc:
            notes = cat.notes_of(it)
        if notes:
            lines.append(f"GOOD TO KNOW: {notes[:600]}")
        if desc and desc != notes:
            lines.append(f"DESCRIPTION: {desc[:500]}{'...' if len(desc) > 500 else ''}")
        return "\n".join(lines)

    @staticmethod
    def req_text(req: dict) -> str:
        prop = Text.of(req.get("prop")).strip()
        vals = [Text.of(v).strip() for v in Catalog.wanted_of(req)]
        unit = Text.of(req.get("unit")).strip()

        def u(v: str) -> str:
            return f"{v} {unit}" if unit and unit.lower() not in v.lower() else v
        op = req.get("op")
        if op == "not":
            text = f"{prop} must not be " + " or ".join(vals)
        elif op == "min":
            text = f"{prop} at least {u(vals[0])}"
        elif op == "max":
            text = f"{prop} at most {u(vals[0])}"
        elif op == "between":
            text = f"{prop} between {vals[0]} and {u(vals[1])}" if len(vals) > 1 else f"{prop} between {u(vals[0])}"
        elif len(vals) == 1:
            text = f"{prop} must be {u(vals[0])}"
        else:
            text = f"{prop} must be one of: " + " or ".join(u(v) for v in vals)
        quote = Text.of(req.get("quote")).strip()
        if req.get("source") == "reference":
            text += " (value resolved from a referenced listing)"
        elif quote:
            text += f' (shopper\'s words: "{quote[:200]}")'
        return text

    def ask_text(self, ai: int, reqs: list[dict], page: str, again: bool) -> str:
        lines = [f"PRODUCT LINE: {self.r.articles[ai]['noun']}", "REQUIREMENTS:"]
        lines += [f"{n}. {self.req_text(q)}" for n, q in enumerate(reqs)]
        lines += ["", "LISTING:", page]
        if again:
            lines += ["", f"Give exactly one entry for each requirement index from 0 to {len(reqs) - 1}."]
        return "\n".join(lines)

    def parse(self, got: Any, n: int, page: str) -> dict[int, tuple[str, str]] | None:
        rows = got.get("r") if isinstance(got, dict) else got
        if not isinstance(rows, list):
            return None
        out: dict[int, tuple[str, str]] = {}
        for x in rows:
            if not isinstance(x, dict):
                continue
            i = x.get("i")
            if isinstance(i, str) and i.strip().isdigit():
                i = int(i.strip())
            if isinstance(i, float) and i.is_integer():
                i = int(i)
            if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < n or i in out:
                continue
            lv = Text.of(x.get("level")).strip().lower()
            if lv not in JUDGE_LEVELS:
                continue
            span = Text.of(x.get("span")).strip()[:300]
            if lv == "exact" and not self.found(span, page):
                lv = "close"
            out[i] = (lv, span)
        return out

    @staticmethod
    def own_phrase(it: dict, req: dict) -> bool:
        if (req.get("op") or "is") != "is":
            return False
        vals = [Text.of(v) for v in Catalog.wanted_of(req) if Text.of(v).strip()]
        stems = Text.key_stems(req.get("prop"))
        if not vals or not stems:
            return False
        for k, v in it["facts"].items():
            if Text.key_stems(k) == stems and any(phrase_states(w, Text.of(v)) for w in vals):
                return True
        return False

    @staticmethod
    def cited_agrees(it: dict, req: dict, span: str) -> bool:
        op = req.get("op") or "is"
        if op not in ("is", "eq", "min", "max", "between") or ":" not in span:
            return False
        head, _, quoted = span.partition(":")
        head, quoted = Judge.flat(head), Judge.flat(quoted)
        p = Text.key_stems(req.get("prop"))
        wanted = Catalog.wanted_of(req)
        if not head or not quoted or not p or not wanted:
            return False
        pool = dict(it["facts"])
        for k, v in list(it.get("rec", {}).items()) + list(it.get("row", {}).items()):
            if k not in pool and isinstance(v, (str, int, float)) and not isinstance(v, bool):
                pool[k] = str(v)
        plain = any(Text.key_stems(k) == p for k in pool)
        unit = Text.of(req.get("unit")).strip() or next((u for w in wanted for _, u in Text(Text.of(w)).numbers if u.strip()), "")
        for k, v in pool.items():
            val = Text.of(v)
            if Judge.flat(k) != head or quoted not in Judge.flat(val):
                continue
            ks = Text.key_stems(k)
            whole = ks == p
            if not whole and (plain or not set(p) <= set(ks)):
                continue
            if op == "is":
                if any(phrase_states(w, val) for w in wanted):
                    return True
                continue
            nums = Text(val).numbers
            if len(nums) != 1 or (not whole and not nums[0][1].strip()):
                continue
            if nums[0][1].strip() and not (unit and Text.unit_scale(nums[0][1], unit) is not None):
                continue
            if Catalog.number_verdict(dict(req, unit=unit or None), wanted, val) == "yes":
                return True
        return False

    def close_states(self, it: dict, req: dict) -> bool:
        if (req.get("op") or "is") not in ("is", "eq") or self.r.cat.fact_values(it, req.get("prop") or ""):
            return False
        vals = [Text.of(v) for v in Catalog.wanted_of(req) if Text.of(v).strip()]
        return any(phrase_states(w, Text.of(v)) for v in it["facts"].values() for w in vals)

    @staticmethod
    def cites_fact(it: dict, span: str) -> bool:
        if ":" not in span:
            return False
        head = Judge.flat(span.split(":", 1)[0])
        return bool(head) and any(Judge.flat(k) == head for k in it["facts"])

    def cites_kin(self, it: dict, ai: int, req: dict, span: str) -> bool:
        if not self.cites_fact(it, span):
            return False
        r = self.r
        head = Judge.flat(span.split(":", 1)[0])
        if any(Judge.flat(k) == head for k in r.cat.aliases.get(Text.key(req.get("prop"))) or []):
            return True
        vals = [Text.of(v) for v in Catalog.wanted_of(req) if Text.of(v).strip()]
        said = [Text.of(v) for v in it["facts"].values() if any(phrase_states(w, Text.of(v)) for w in vals)]
        if not said or joined_only(vals, said):
            return True
        return not any(r.cat.fact_values(o, req.get("prop") or "") for o in r.owners(ai) if o is not it and o.get("read"))

    def keep(self, it: dict, ai: int, reqs: list[dict], res: dict[int, tuple[str, str]]) -> None:
        for i, (lv, span) in sorted(res.items()):
            if lv != "exact" and (self.own_phrase(it, reqs[i]) or self.r.cat.title_initials(it, reqs[i]) or self.r.cat.own_states(it, reqs[i])):
                lv = "exact"
            if lv == "close" and self.close_states(it, reqs[i]):
                lv = "exact"
            if lv != "exact" and not self.cites_kin(it, ai, reqs[i], span) and self.r.cat.title_states(it, reqs[i]):
                lv = "exact"
            if lv != "exact" and self.cited_agrees(it, reqs[i], span):
                lv = "exact"
            if lv != "exact" and self.r.cat.firm_yes(it, reqs[i]):
                lv = "exact"
            key = self.sig(reqs[i])
            it.setdefault("jspan", {})[key] = span
            if lv == "exact" and self.r.cat.mark_clash(it, reqs[i]):
                lv = "no"
            elif lv == "exact" and (self.r.cat.stray_off(it, reqs[i]) or self.r.cat.head_off(it, reqs[i]) or self.r.cat.dims_off(it, reqs[i])
                                       or self.r.cat.bare_off(it, reqs[i]) or self.r.cat.joined_off(it, reqs[i])):
                lv = "close"
            it.setdefault("jl", {})[key] = lv
        lv = self.line_level(it, ai, reqs)
        self.log.append(f"a{ai} {it['id']} {lv or 'partial ' + str(len(res)) + '/' + str(len(reqs))}")

    def line_key(self, reqs: list[dict]) -> tuple:
        return tuple(sorted(self.sig(q) for q in reqs))

    def can_judge(self, n: int = 1) -> bool:
        r = self.r
        cap = JUDGE_TASK_CAP + (WEAK_JUDGE_EXTRA if self.weak else 0)
        return (not r.exhausted and self.asked + n <= cap and r.t() < GUARD_AT - 40
                and r.spent + n * JUDGE_UNIT_USD < CHEAP_ONLY_AT - JUDGE_RESERVE_USD)

    def call_many(self, work: list[tuple[dict, int, list[dict], str]], again: bool) -> list[Any]:
        jobs = [("judge", JUDGE_SYS, self.ask_text(ai, reqs, page, again),
                 {"max_tokens": JUDGE_TOKENS, "deadline": GUARD_AT - 30, "chain": JUDGE_CHAIN}) for it, ai, reqs, page in work]
        self.asked += len(jobs)
        got = self.r.model_many(jobs)
        return list(got) + [None] * (len(jobs) - len(got))

    def judge_batch(self, pairs: list[tuple[dict, int]]) -> None:
        work = []
        for it, ai in pairs:
            reqs = self.judgeable(ai)
            self.tried.add((it["id"], self.line_key(reqs)))
            work.append((it, ai, reqs, self.page_of(it)))
        retry = []
        for (it, ai, reqs, page), got in zip(work, self.call_many(work, False)):
            res = self.parse(got, len(reqs), page)
            if res is not None and len(res) == len(reqs):
                self.keep(it, ai, reqs, res)
            else:
                retry.append((it, ai, reqs, page, res or {}))
        if not retry:
            return
        if not self.can_judge(len(retry)):
            for it, ai, reqs, page, first in retry:
                self.keep(it, ai, reqs, first)
            return
        for (it, ai, reqs, page, first), got in zip(retry, self.call_many([w[:4] for w in retry], True)):
            res = self.parse(got, len(reqs), page) or {}
            self.keep(it, ai, reqs, res if len(res) == len(reqs) else {**first, **res})

    def hint(self, it: dict, ai: int) -> int:
        r = self.r
        cat = r.cat
        bad = [q for q in r.reqs(ai) if cat.verdict(it, q) == "no" or r.blind(it, q)]
        if not bad:
            return 0
        hay = set(Text(cat.title(it) + " " + " ".join(Text.of(v) for v in it["facts"].values())).stems)
        for q in bad:
            hit = False
            for w in Catalog.wanted_of(q):
                words = Text(w).words
                if any(len(x) > 2 and x in hay for x in Catalog.content_stems(w)) or (len(words) > 1 and "".join(x[0] for x in words) in hay):
                    hit = True
            if not hit:
                return 2
        return 1

    def price_of(self, it: dict) -> float:
        p = self.r.cat.price(it)
        return 1e18 if p is None else p

    def open_for(self, it: dict, reqs: list[dict], key: tuple) -> bool:
        got = [self.level(it, q) for q in reqs]
        return (it["id"], key) not in self.tried and None in got and "no" not in got

    def bar_of(self, ai: int, cands: list[dict], reqs: list[dict]) -> float:
        return min((self.price_of(it) for it in cands if self.line_level(it, ai, reqs) == "exact"), default=1e18)

    def targets(self, ai: int) -> list[dict]:
        r = self.r
        reqs = self.judgeable(ai)
        if not reqs:
            return []
        key = self.line_key(reqs)
        cands = r.candidates(ai)
        pool = [it for it in cands if self.open_for(it, reqs, key)]
        if r.cheapest:
            bar = self.bar_of(ai, cands, reqs)
            pool = [it for it in pool if self.price_of(it) < bar or bar >= 1e18]
            pool.sort(key=lambda it: (self.hint(it, ai), self.price_of(it), it["id"]))
        pick = r.cat.items.get(r.articles[ai].get("pick") or "")
        if pick is not None and pick in pool:
            pool.remove(pick)
            pool.insert(0, pick)
        return pool

    def judge_lines(self, lines: list[int] | None = None) -> int:
        r = self.r
        lines = [a["i"] for a in r.articles] if lines is None else lines
        if not self.can_judge():
            return 0
        pools = {ai: self.targets(ai) for ai in lines}
        reqs = {ai: self.judgeable(ai) for ai in lines}
        keys = {ai: self.line_key(reqs[ai]) for ai in lines}
        bars = {ai: 1e18 for ai in lines}
        room = {ai: JUDGE_LINE_CAP for ai in lines}
        limit = max(JUDGE_BATCH, (JUDGE_TASK_CAP - self.asked) * 2 // 3)
        done = 0
        while self.can_judge() and done < limit:
            queues = {}
            for ai in lines:
                q = [it for it in pools[ai] if self.open_for(it, reqs[ai], keys[ai])
                     and not (r.cheapest and bars[ai] < 1e18 and self.price_of(it) >= bars[ai])]
                queues[ai] = q[:max(0, room[ai])]
            pairs: list[tuple[dict, int]] = []
            for k in range(JUDGE_LINE_CAP):
                for ai in lines:
                    q = queues[ai]
                    if k < len(q) and len(pairs) < JUDGE_BATCH:
                        pairs.append((q[k], ai))
            pairs = pairs[:max(0, min(JUDGE_TASK_CAP - self.asked, limit - done))]
            if not pairs or not self.can_judge(len(pairs)):
                break
            self.judge_batch(pairs)
            done += len(pairs)
            for it, ai in pairs:
                room[ai] -= 1
                if self.line_level(it, ai, reqs[ai]) == "exact":
                    bars[ai] = min(bars[ai], self.price_of(it))
        return done

    def policy(self, pick: dict | None, cands: list[dict], ai: int, order: list[str] | None = None) -> dict | None:
        r = self.r
        if not cands or not any(self.judged(it, ai) for it in cands):
            return pick
        price = self.price_of
        tiers = {it["id"]: self.tier_of(it, ai) for it in cands}
        best = min(tiers.values())
        if best > 1:
            return pick
        cur = None if pick is None else (tiers[pick["id"]] if pick["id"] in tiers else self.tier_of(pick, ai))
        top = [it for it in cands if tiers[it["id"]] == best]
        clean = [it for it in top if not self.own_no(it, ai)]
        flags = self.own_flags(pick, ai) if pick is not None else []
        if flags:
            clean = [it for it in clean if self.own_states(it, flags)]
        stale = pick is not None and bool(clean) and bool(flags)
        top = clean or top
        room = self.room_left(ai)
        if room is not None and pick is not None and price(pick) * self.qty_of(ai) <= room + 1e-9:
            within = [it for it in top if price(it) * self.qty_of(ai) <= room + 1e-9]
            if not within:
                return pick
            top = within
        if r.cheapest:
            choice = min(top, key=lambda it: (price(it), -self.own_score(it, ai), it["id"]))
            if pick is not None and cur == best and not stale and (price(pick) < price(choice) or (
                    price(pick) == price(choice) and self.own_score(pick, ai) >= self.own_score(choice, ai))):
                return pick
            return choice
        if pick is not None and cur is not None and cur <= best and not stale:
            return pick
        if pick is not None and cur is not None and cur <= 1 and not stale and self.line_level(pick, ai) in ("exact", "close"):
            seen = [it for it in top if self.judged(it, ai)]
            if not seen:
                return pick
            top = seen
        ids = {it["id"]: it for it in top}
        ranked = [ids[x] for x in (order or []) if x in ids]
        if ranked:
            return ranked[0]
        spot = {it["id"]: n for n, it in enumerate(top)}
        return min(top, key=lambda it: (-self.own_score(it, ai), spot[it["id"]]))

    def own_flags(self, it: dict, ai: int) -> list[dict]:
        out = []
        for q in self.r.reqs(ai):
            if q.get("op") not in ("is", "eq", "min", "max", "between", "not") or empty(q.get("value")):
                continue
            p = Text.key_stems(Text.of(q.get("prop")))
            if not p:
                continue
            vals = [Text.of(v) for k, v in it["facts"].items() if Text.key_stems(k) == p and Text.of(v).strip()]
            if vals and all(Catalog.value_verdict(q, v, True) == "no" for v in vals):
                out.append(q)
        return out

    def own_states(self, it: dict, flags: list[dict]) -> bool:
        for q in flags:
            p = Text.key_stems(Text.of(q.get("prop")))
            vals = [Text.of(v) for k, v in it["facts"].items() if Text.key_stems(k) == p and Text.of(v).strip()]
            if not vals or all(Catalog.value_verdict(q, v, True) == "no" for v in vals):
                return False
        return True

    def own_no(self, it: dict, ai: int) -> bool:
        for q in self.r.reqs(ai):
            if q.get("op") not in ("is", "eq", "min", "max", "between", "not") or empty(q.get("value")):
                continue
            p = Text.key_stems(Text.of(q.get("prop")))
            if not p:
                continue
            vals = [Text.of(v) for k, v in it["facts"].items() if Text.key_stems(k) == p and Text.of(v).strip()]
            if vals and all(Catalog.value_verdict(q, v, True) == "no" for v in vals):
                return True
        return False

    def qty_of(self, ai: int) -> int:
        return max(1, int(self.r.articles[ai].get("qty") or 1))

    def room_left(self, ai: int) -> float | None:
        r = self.r
        if not r.budget:
            return None
        others = 0.0
        for art in r.articles:
            o = r.cat.items.get(art.get("pick") or "") if art["i"] != ai else None
            if o is not None and self.price_of(o) < 1e17:
                others += self.price_of(o) * max(1, int(art.get("qty") or 1))
        return r.budget - others

    def own_score(self, it: dict, ai: int) -> int:
        cat = self.r.cat
        title = Text.key_stems("title")
        n = 0
        for q in self.judgeable(ai):
            if self.level(it, q) != "exact":
                continue
            k, sep, _ = Text.of((it.get("jspan") or {}).get(self.sig(q))).partition(":")
            ks = Text.key_stems(k)
            if not sep or not ks:
                continue
            if ks == title or any(Text.key_stems(fk) == ks for fk, _ in cat.fact_values(it, Text.of(q.get("prop")))):
                n += 1
        return n

    def judged_no(self, it: dict, ai: int) -> bool:
        return any(self.level(it, q) == "no" for q in self.judgeable(ai))

    def recheck(self) -> bool:
        r = self.r
        if not r.articles or r.t() > GUARD_AT - 40:
            return False
        self.judge_lines()
        paired = {x for p in r.together for x in p}
        moved = False
        for art in r.articles:
            ai = art["i"]
            if ai in paired or r.steps_left() <= r.plan.finish_need(minimal=True) + r.order.swap_cost(1) + 1:
                continue
            pick = r.cat.items.get(art.get("pick") or "")
            if pick is None:
                continue
            cands = r.candidates(ai)[:9]
            new = self.policy(pick, cands, ai, [pick["id"]] + [Text.of(x) for x in art.get("backup") or []])
            if new is not None and new is not pick:
                r.notes.append(f"judge: line {ai} {pick['id']} ({self.line_level(pick, ai)}) -> {new['id']} ({self.line_level(new, ai)})")
                art["backup"] = [x for x in [pick["id"]] + list(art.get("backup") or []) if x != new["id"]]
                art["pick"] = new["id"]
                moved = True
        return moved

    def audit(self) -> None:
        r = self.r
        cat = r.cat
        if not r.articles or r.t() > GUARD_AT - 45 or r.spent > CHEAP_ONLY_AT - 0.004:
            return
        paired = {x for p in r.together for x in p}
        lines, cmap = [], {}
        for art in r.articles:
            ai = art["i"]
            pick = cat.items.get(art.get("pick") or "")
            reqs = r.reqs(ai)
            if pick is None or ai in paired or not reqs:
                continue
            alts = [it for it in r.candidates(ai) if it is not pick and r.ok_basic(it, ai) and it.get("read")]
            alts.sort(key=lambda it: (self.price_of(it), it["id"]))
            cands = [pick] + alts[:AUDIT_ALTS]
            cmap[ai] = {it["id"]: it for it in cands}
            rl = "\n".join(f"    {n}. {self.req_text(q)}" for n, q in enumerate(reqs))
            rows = "\n".join(f"    {'PICK ' if it is pick else ''}{it['id']}: price {cat.price(it)}, title {cat.title(it)[:160]!r}, facts {Text.json(it['facts'], 700)}"
                             for it in cands)
            lines.append(f"article {ai}: {art['noun']} x{art['qty']}\n  requirements:\n{rl}\n  listings:\n{rows}")
        if not lines:
            return
        rule = "cheapest that meets every requirement" if r.cheapest else (Text.of(r.spec.get("priority")) or "any listing that meets every requirement")
        user = f"REQUEST:\n{r.request[:3500]}\n\nSELECTION RULE: {rule}\nBUDGET: {r.budget}\n\n" + "\n\n".join(lines)
        d = r.model_json("audit", AUDIT_SYS, user, max_tokens=2500, deadline=GUARD_AT - 20, chain=AUDIT_CHAIN)
        rows = d.get("articles") if isinstance(d, dict) else None
        flagged = [x.get("article") for x in rows or [] if isinstance(x, dict) and (x.get("pick_ok") is False or x.get("replace_with"))]
        r.notes.append(f"audit: {len(cmap)} articles checked, reply {'ok' if isinstance(rows, list) else 'missing'}, flagged {flagged}")
        for x in rows if isinstance(rows, list) else []:
            if not isinstance(x, dict):
                continue
            try:
                ai = int(x.get("article"))
            except (TypeError, ValueError):
                continue
            if ai not in cmap or r.steps_left() <= r.plan.finish_need(minimal=True) + r.order.swap_cost(1) + 1:
                continue
            art = r.articles[ai]
            pick = cat.items.get(art.get("pick") or "")
            alt = cmap[ai].get(Text.of(x.get("replace_with")))
            if pick is None or alt is None or alt is pick or r.contradicted(alt, ai):
                continue
            reqs = r.reqs(ai)
            quotes = x.get("replace_quotes") if isinstance(x.get("replace_quotes"), dict) else {}
            if not all(self.level(alt, q) == "exact" or self.own_phrase(alt, q) or self.quote_ok(alt, quotes.get(str(n)) or quotes.get(n), q)
                       for n, q in enumerate(reqs)):
                r.notes.append(f"audit: line {ai} swap to {alt['id']} not verified")
                continue
            fails = x.get("failed") if isinstance(x.get("failed"), list) else []
            proven = any(isinstance(f, dict) and self.pick_fails(pick, reqs, f) for f in fails)
            cheaper = self.price_of(alt) < self.price_of(pick)
            better = Text.of(r.spec.get("priority")) == "best" and self.better_by(alt, pick, x.get("better_by"))
            if not (proven or (r.cheapest and cheaper) or better):
                continue
            r.notes.append(f"audit: line {ai} {pick['id']} -> {alt['id']} ({'pick fails' if proven else 'cheaper' if cheaper and r.cheapest else 'better by rule'})")
            art["backup"] = [y for y in [pick["id"]] + list(art.get("backup") or []) if y != alt["id"]]
            art["pick"] = alt["id"]

    def fit_solve(self) -> None:
        r = self.r
        if not r.together or r.t() > GUARD_AT - 25:
            return
        self.read_fit_wording()
        hubs: dict[int, list[int]] = {}
        for pr in r.together:
            if len(pr) == 2 and all(isinstance(x, int) and 0 <= x < len(r.articles) for x in pr) and pr[0] != pr[1]:
                hubs.setdefault(pr[1], []).append(pr[0])
        for b, accs in hubs.items():
            accs = list(dict.fromkeys(a for a in accs if a != b))
            if not accs:
                continue
            if not self.fit_pivot(b, accs):
                self.fit_pairs_solve(b, accs)
        self.fit_exact("solve")
        self.see_pointers([a for accs in hubs.values() for a in accs])
        self.fit_solved = True

    def fit_exact(self, tag: str) -> bool:
        r = self.r
        cat = r.cat
        if not r.together or r.finished or r.t() > GUARD_AT - 15:
            return False
        edges = [(b, a) for b, accs in self.fit_hubs().items() for a in accs if a != b]
        if not edges:
            return False
        lines = sorted({x for e in edges for x in e})
        cur = {ai: cat.items.get(r.articles[ai].get("pick") or "") for ai in lines}
        index = self.title_index()
        infos: dict[tuple, FitInfo] = {}
        tiers: dict[tuple, int] = {}

        def info(it: dict, role: str | None) -> FitInfo:
            key = (it["id"], bool(it.get("read")), bool(it.get("paged")), role)
            if key not in infos:
                infos[key] = FitInfo(cat.fit_record(it), role)
            return infos[key]

        def tier(h: dict, x: dict) -> int:
            key = (h["id"], x["id"], bool(h.get("read")), bool(x.get("read")))
            if key not in tiers:
                pages = [info(q, None) for c in info(x, "dep").pointers for q in index.get(fit_norm(c), []) if q is not x]
                tiers[key] = self.tier(*fit_judge(info(h, "main"), info(x, "dep"), pages))
            return tiers[key]
        if all(cur[ai] is not None for ai in lines):
            ts = [tier(cur[b], cur[a]) for b, a in edges]
            if not any(t >= 9 or t == 2 for t in ts):
                return False
        else:
            ts = [9] * (len(edges) + 1)
        sub = {a for _, a in edges}

        def hard(it: dict, ai: int) -> bool:
            return bool(it.get("gone") or it.get("decoy")) or not r.ok_basic(it, ai)

        def classify(it: dict, b: int) -> tuple:
            out = []
            for role in ("main", "dep") if b in sub else ("main",):
                f = info(it, role)
                out.append((tuple(sorted(f.own)), tuple(sorted(f.compat)), tuple(sorted(f.exc)), tuple(sorted(f.weak)),
                            tuple(sorted((k, tuple(sorted(v))) for k, v in f.ext)), tuple(sorted(fit_norm(p) for p in f.open_pointers()))))
            return tuple(out)
        dead: set[tuple[int, str]] = set()

        def state(it: dict, ai: int) -> str:
            return "x" if (ai, it["id"]) in dead else r.basket.state(it, ai)
        price = self.price_of
        qty = {a["i"]: max(1, int(a.get("qty") or 1)) for a in r.articles}
        others = sum(price(cat.items[art["pick"]]) * qty[art["i"]] for art in r.articles
                     if art["i"] not in lines and cat.items.get(art.get("pick") or "") is not None)
        opened = 0
        while True:
            memo: dict[tuple[int, str], str] = {}

            def seen_state(it: dict, ai: int) -> str:
                k = (ai, it["id"])
                if k not in memo:
                    memo[k] = state(it, ai)
                return memo[k]
            def doubt(it: dict, ai: int) -> int:
                st = seen_state(it, ai)
                return 0 if st == "v" else 1 if st == "u" else 1 + max(1, r.fit(it, ai)[1])
            now = (sum(t >= 9 for t in ts), sum(doubt(cur[ai], ai) - 1 for ai in lines if cur[ai] is not None and doubt(cur[ai], ai) >= 2),
                   sum(t == 2 for t in ts))
            pools = {}
            for ai in lines:
                spoilt = cur[ai] is None or seen_state(cur[ai], ai) == "x"
                pool = [it for it in r.basket.table(ai) if (ai, it["id"]) not in dead and not hard(it, ai)
                        and (spoilt or seen_state(it, ai) != "x")][:FIT_POOL]
                if cur[ai] is not None and all(it["id"] != cur[ai]["id"] for it in pool):
                    pool.append(cur[ai])
                pools[ai] = pool
            best = fit_best(edges, pools, cur, tier, price, qty, doubt, classify, others, r.budget, r.cheapest)
            if best is None or best[0][:3] >= now:
                r.notes.append(f"fit exact ({tag}): misfit {now} kept, best {best[0][:5] if best else None}, opened {opened}")
                return False
            chosen = best[1]
            moved = [ai for ai in lines if cur[ai] is None or chosen[ai]["id"] != cur[ai]["id"]]
            room = r.steps_left() - r.plan.finish_need(minimal=True) - 1 - max(len(moved) * r.order.swap_cost(1), r.order.move_extra({ai: chosen[ai]["id"] for ai in moved}))
            unsure = [(ai, chosen[ai]) for ai in moved if state(chosen[ai], ai) == "u"]
            if not unsure:
                break
            if opened >= FIT_VERIFY_CAP or room < 2 or r.t() > GUARD_AT - 25:
                blind = [(ai, it) for ai, it in unsure if cur[ai] is not None and state(cur[ai], ai) == "v"]
                if not blind:
                    break
                dead.update((ai, it["id"]) for ai, it in blind)
                continue
            ai, it = min(unsure, key=lambda p: (price(p[1]), p[1]["id"]))
            if not it.get("read"):
                self.read_one(it)
                opened += 1
            if it.get("read") and self.line_level(it, ai) is None and self.can_judge(1):
                self.judge_batch([(it, ai)])
            if state(it, ai) != "v":
                dead.add((ai, it["id"]))
        if room < 0:
            r.notes.append(f"fit exact ({tag}): fix needs {len(moved)} swaps, no room")
            return False
        for ai in moved:
            art = r.articles[ai]
            old = cur[ai]["id"] if cur[ai] is not None else None
            art["backup"] = [y for y in ([old] if old else []) + list(art.get("backup") or []) if y != chosen[ai]["id"]]
            art["pick"] = chosen[ai]["id"]
        r.notes.append(f"fit exact ({tag}): misfit {now} -> {best[0][:5]}, lines {moved}, total {best[2]}, opened {opened}")
        self.see_pointers([ai for ai in moved if ai in sub], loose=True)
        return True

    def code_guard(self, tag: str) -> bool:
        r = self.r
        cat = r.cat
        if r.finished or r.t() > GUARD_AT - 15:
            return False
        price = self.price_of
        state = r.basket.state
        changed = False
        for ai in self.coded_lines():
            reqs = [q for q in r.reqs(ai) if q.get("op") in ("is", "eq") and any(fit_codes(Text.of(w)) for w in Catalog.wanted_of(q))]
            cur = cat.items.get(r.articles[ai].get("pick") or "")
            if cur is None or not (any(cat.code_verdict(cur, q) == "no" for q in reqs) or state(cur, ai) == "x"):
                continue
            qty = max(1, int(r.articles[ai].get("qty") or 1))
            others = sum(price(cat.items[a["pick"]]) * max(1, int(a.get("qty") or 1)) for a in r.articles
                         if a["i"] != ai and cat.items.get(a.get("pick") or "") is not None)
            dead: set[str] = set()
            opened = 0

            def pool(strict: bool = True) -> list[dict]:
                return [it for it in r.basket.table(ai) if it["id"] not in dead and (state(it, ai) != "x" or not strict)
                        and not (it.get("gone") or it.get("decoy") or not r.ok_basic(it, ai))
                        and not any(cat.code_verdict(it, q) == "no" for q in reqs)
                        and not (r.budget and others + price(it) * qty > r.budget + 1e-9)][:FIT_POOL]

            def doubt(it: dict) -> int:
                return sum(cat.code_verdict(it, q) != "yes" for q in reqs)
            scans = 0
            while opened < FIT_VERIFY_CAP and r.t() < GUARD_AT - 25:
                ranked = sorted(pool(), key=lambda it: (doubt(it), price(it), it["id"]))
                room = r.steps_left() - r.plan.finish_need(minimal=True) - 1 - r.order.swap_cost(1)
                if not ranked and scans < 2 and room >= 4:
                    scans += 1
                    if r.hunt.fit_scan(ai, min(FIT_SCAN_PAGES, (room - 2) // 2), ok=lambda it: it["id"] not in dead and doubt(it) == 0):
                        continue
                if not ranked or state(ranked[0], ai) == "v" or room < 2:
                    break
                top = ranked[0]
                if not top.get("read"):
                    self.read_one(top)
                    opened += 1
                if top.get("read") and self.line_level(top, ai) is None and self.can_judge(1):
                    self.judge_batch([(top, ai)])
                if state(top, ai) != "v":
                    dead.add(top["id"])
            left = pool() or pool(False)
            if not left or r.steps_left() - r.plan.finish_need(minimal=True) - 1 < r.order.swap_cost(1):
                r.notes.append(f"code guard ({tag}): line {ai} pick {cur['id']} fails a stated code, no replacement (opened {opened})")
                continue
            pick = min(left, key=lambda it: ("vux".index(state(it, ai)) if state(it, ai) in "vux" else 2, r.fit(it, ai)[1], doubt(it), price(it), it["id"]))
            if state(pick, ai) != "v" and not any(cat.code_verdict(cur, q) == "no" for q in reqs):
                continue
            art = r.articles[ai]
            art["backup"] = [y for y in [cur["id"]] + list(art.get("backup") or []) if y != pick["id"]]
            art["pick"] = pick["id"]
            r.notes.append(f"code guard ({tag}): line {ai} {cur['id']} -> {pick['id']} (opened {opened})")
            self.see_pointers([ai], loose=True)
            changed = True
        return changed

    def read_fit_wording(self) -> int:
        r = self.r
        cat = r.cat
        fitted = set(self.coded_lines()) | {x for pr in r.together for x in pr if isinstance(x, int)}
        if not fitted or self.fit_reads >= FIT_READS or r.t() > GUARD_AT - 40:
            return 0
        shelves = {r.articles[ai].get("cat") for ai in fitted if 0 <= ai < len(r.articles)} - {None, ""}
        seen: dict[str, list] = {}
        listed = 0
        for it in list(cat.items.values()):
            if not shelves or cat.category(it) in shelves:
                found = [(sent, tpl) for part in cat.fit_record(it)[1:] for sent, _, tpl in fit_forms(part)]
                listed += bool(found)
                for sent, tpl in found:
                    seen.setdefault(tpl, [0, sent])[0] += 1
        pend = sorted(((n, t, ex) for t, (n, ex) in seen.items() if t not in FIT_ROLES and t not in FIT_ASKED),
                      key=lambda z: (-z[0], z[1]))[:FIT_READ_BATCH]
        if not pend:
            return 0
        self.fit_reads += 1
        user = "SENTENCES:\n" + "\n".join(f"{k}. {t}\n   example: {ex[:200]}" for k, (_, t, ex) in enumerate(pend, 1))
        d = r.model_json("read", FITREAD_SYS, user, max_tokens=60 * len(pend) + 200, deadline=GUARD_AT - 30, chain=JUDGE_CHAIN)
        rows = d.get("answers") if isinstance(d, dict) else None
        stored = 0
        for x in rows if isinstance(rows, list) else []:
            try:
                k = int(x.get("id"))
                got = {int(a): Text.of(b).strip().lower() for a, b in (x.get("roles") or {}).items()}
            except (AttributeError, TypeError, ValueError):
                continue
            if not 1 <= k <= len(pend):
                continue
            t = pend[k - 1][1]
            n = len(FIT_SLOT_RX.findall(t))
            if pend[k - 1][0] * 2 > listed:
                got = {a: ("none" if b == "not" else b) for a, b in got.items()}
            vals = set(got.values())
            if not got or not all(1 <= a <= n and b in FIT_ROLE_NAMES for a, b in got.items()) or ("about" in vals) != ("theirs" in vals):
                continue
            FIT_ROLES[t] = got
            stored += 1
        if isinstance(rows, list):
            FIT_ASKED.update(t for _, t, _ in pend)
        if stored:
            FIT_STATE["version"] += 1
        r.notes.append(f"fit wording: {len(pend)} sentence forms asked, {stored} read: "
                       + "; ".join(f"{t} = {FIT_ROLES[t]}" for _, t, _ in pend if t in FIT_ROLES)[:600])
        return stored

    def coded_lines(self) -> list[int]:
        return [a["i"] for a in self.r.articles if self.code_reqs(a["i"])]

    def code_reqs(self, ai: int) -> list[dict]:
        return [q for q in self.r.reqs(ai) if q.get("op") in ("is", "eq") and any(fit_codes(Text.of(w)) for w in Catalog.wanted_of(q))]

    def fit_hubs(self) -> dict[int, list[int]]:
        r = self.r
        hubs: dict[int, list[int]] = {}
        for pr in r.together:
            if len(pr) == 2 and all(isinstance(x, int) and 0 <= x < len(r.articles) for x in pr) and pr[0] != pr[1]:
                hubs.setdefault(pr[1], [])
                if pr[0] not in hubs[pr[1]]:
                    hubs[pr[1]].append(pr[0])
        return hubs

    def fit_lead(self, ai: int) -> list[str]:
        r = self.r
        cat = r.cat
        price = self.price_of
        for b, accs in self.fit_hubs().items():
            if ai != b and ai not in accs:
                continue
            lines = [b] + accs
            table = {x: [it for it in r.basket.table(x) if r.basket.state(it, x) != "x"][:FIT_POOL] for x in lines}
            if any(not table[x] for x in lines):
                return []
            infos: dict[tuple, FitInfo] = {}

            def info(it: dict, role: str) -> FitInfo:
                key = (it["id"], role)
                if key not in infos:
                    infos[key] = FitInfo(cat.fit_record(it), role)
                return infos[key]
            index = self.title_index()

            def pages(x: dict) -> list[FitInfo]:
                return [info(q, "page") for c in info(x, "dep").pointers for q in index.get(fit_norm(c), []) if q is not x]
            classes: dict[tuple, list[dict]] = {}
            for h in table[b]:
                ids = tuple(sorted(info(h, "main").own))
                if ids:
                    classes.setdefault(ids, []).append(h)
            ranked = []
            for ids, hs in classes.items():
                h = min(hs, key=lambda it: (price(it), it["id"]))
                per = [[h] + sorted((x for x in hs if x is not h), key=lambda it: (price(it), it["id"]))[:1]]
                ts = []
                for a in accs:
                    opts = [(self.tier(*fit_judge(info(h, "main"), info(x, "dep"), pages(x))), price(x), x["id"], x) for x in table[a]]
                    opts = sorted((o for o in opts if o[0] < 9), key=lambda o: ((o[0] == 2) if r.cheapest and FIT_CLAIMS_PRICED else o[0], o[1], o[2]))
                    if not opts:
                        break
                    per.append([o[3] for o in opts[:2]])
                    ts.append(opts[0][0])
                else:
                    total = sum(price(p[0]) * max(1, int(r.articles[x].get("qty") or 1)) for x, p in zip(lines, per))
                    ranked.append(((ts.count(2), total if r.cheapest else sum(ts), sum(ts), ids), per))
            ranked.sort(key=lambda z: z[0])
            out: list[str] = []
            k = lines.index(ai)
            for n, (_, per) in enumerate(ranked[:3]):
                for it in (per[k] if n == 0 else per[k][:1]):
                    if it["id"] not in out:
                        out.append(it["id"])
            return out
        return []

    def see_pointers(self, accs: list[int], loose: bool = False) -> None:
        r = self.r
        cat = r.cat
        index = self.title_index()
        for ai in dict.fromkeys(accs):
            it = cat.items.get(r.articles[ai].get("pick") or "")
            if it is None:
                continue
            for code in fit_info(cat.fit_record(it), "dep").pointers:
                room = r.steps_left() - r.plan.finish_need(minimal=True) - 1 if loose else r.plan.slack()
                if r.t() > GUARD_AT - 20 or room < 2:
                    return
                held = [q for q in index.get(fit_norm(code), []) if q is not it]
                if not held:
                    self.see_part(code)
                    index = self.title_index()
                elif not any(q.get("read") or q.get("paged") for q in held) and r.plan.slack() >= 4:
                    r.hunt.open([held[0]["id"]], force=True)

    def title_index(self) -> dict[str, list[dict]]:
        cat = self.r.cat
        out: dict[str, list[dict]] = {}
        for it in cat.items.values():
            for c in dict.fromkeys(fit_codes(cat.slot_text(it, "title"))):
                out.setdefault(c, []).append(it)
        return out

    def fit_pivot(self, b: int, accs: list[int]) -> bool:
        r = self.r
        cat = r.cat
        price = self.price_of
        lines = [b] + accs
        qty = {art["i"]: max(1, int(art.get("qty") or 1)) for art in r.articles}
        picks = {ai: cat.items.get(r.articles[ai].get("pick") or "") for ai in lines}
        recs: dict[tuple, tuple[str, str]] = {}
        infos: dict[tuple, FitInfo] = {}
        dead: set[tuple[int, str]] = set()
        tiers: dict[tuple, int] = {}

        def mark(it: dict) -> tuple:
            return (it["id"], bool(it.get("read")), bool(it.get("paged")))

        def rec(it: dict) -> tuple[str, str]:
            if mark(it) not in recs:
                recs[mark(it)] = cat.fit_record(it)
            return recs[mark(it)]

        def info(it: dict, role: str | None) -> FitInfo:
            key = mark(it) + (role,)
            if key not in infos:
                infos[key] = FitInfo(rec(it), role)
            return infos[key]

        def state(it: dict, ai: int) -> str:
            return "x" if (ai, it["id"]) in dead else r.basket.state(it, ai)
        index = self.title_index()

        def pages(x: dict) -> list[FitInfo]:
            return [info(q, None) for c in info(x, "dep").pointers for q in index.get(fit_norm(c), []) if q is not x]

        def tier(h: dict, x: dict) -> int:
            key = (mark(h), mark(x), len(index))
            if key not in tiers:
                tiers[key] = self.tier(*fit_judge(info(h, "main"), info(x, "dep"), pages(x)))
            return tiers[key]

        loose: set[int] = set()

        def order(it: dict, ai: int) -> tuple:
            st = state(it, ai)
            if ai in loose:
                return (r.fit(it, ai)[1], price(it), it["id"])
            return (price(it), st != "v", it["id"]) if r.cheapest else (st != "v", price(it), it["id"])
        others = sum(price(cat.items[art["pick"]]) * qty[art["i"]] for art in r.articles
                     if art["i"] not in lines and cat.items.get(art.get("pick") or "") is not None)

        if not any(info(h, "main").own for h in r.basket.table(b)[:FIT_POOL]):
            return False
        def rank(chosen: list[dict], ts: list[int]) -> tuple:
            total = others + sum(price(it) * qty[ai] for ai, it in zip(lines, chosen))
            changes = sum(1 for ai, it in zip(lines, chosen) if it is not picks[ai])
            if r.cheapest and FIT_CLAIMS_PRICED:
                return (ts.count(2), total, sum(ts), changes)
            if r.cheapest:
                return (sum(ts), total, changes, 0)
            return (sum(ts), changes, total, 0)

        def mark_verified() -> None:
            for ai in lines:
                r.articles[ai]["fit_verified"] = True
        cur_ok = all(picks[ai] is not None and state(picks[ai], ai) == "v" for ai in lines) and all(tier(picks[b], picks[a]) < 9 for a in accs)
        if cur_ok and not r.cheapest:
            mark_verified()
            return True
        broken = all(picks[x] is not None for x in lines) and any(
            fit_judge(info(picks[b], "main"), info(picks[a], "dep"), pages(picks[a]))[0] == "no" for a in accs)

        def room() -> int:
            return r.steps_left() - r.plan.finish_need(minimal=True) - 1 if broken else r.plan.slack() + r.plan.fit_hold()
        cur_ts = [tier(picks[b], picks[a]) for a in accs] if cur_ok else []
        cur_total = others + sum(price(picks[ai]) * qty[ai] for ai in lines) if cur_ok else 0.0
        last: dict = {}

        def solve(strict: bool = False) -> tuple | None:
            table = {ai: sorted((it for it in r.basket.table(ai) if state(it, ai) == "v" or (state(it, ai) != "x" and not strict)),
                                key=lambda it: state(it, ai) != "v")[:FIT_POOL] for ai in lines}
            for ai in lines:
                if not table[ai] and ai != b:
                    loose.add(ai)
                    table[ai] = [it for it in r.candidates(ai) if not it.get("decoy")][:FIT_POOL]
            classes: dict[tuple, list[dict]] = {}
            for h in table[b]:
                ids = tuple(sorted(info(h, "main").own))
                if ids:
                    classes.setdefault(ids, []).append(h)
            best = None
            cover = {a: 0 for a in accs}
            for ids, hs in classes.items():
                h = min(hs, key=lambda it: order(it, b))
                chosen = [h]
                ts = []
                for a in accs:
                    opts = [(tier(h, x), order(x, a), x) for x in table[a]]
                    opts = [o for o in opts if o[0] < 9]
                    if a in loose:
                        clean = [o for o in opts if not r.contradicted(o[2], a)]
                        opts = clean or ([] if picks[a] is not None and not r.contradicted(picks[a], a) else opts)
                    if opts:
                        cover[a] += 1
                    if not opts:
                        break
                    t, _, x = min(opts, key=lambda o: (o[1][0] if a in loose else 0,) + ((o[0] == 2, o[1]) if r.cheapest and FIT_CLAIMS_PRICED else (o[0], o[1])))
                    chosen.append(x)
                    ts.append(t)
                else:
                    total = others + sum(price(it) * qty[ai] for ai, it in zip(lines, chosen))
                    if r.budget and total > r.budget + 1e-9:
                        continue
                    key = (sum(1 for ai, it in zip(lines, chosen) if ai in loose and r.contradicted(it, ai)),) + rank(chosen, ts)
                    if cur_ok and (total >= cur_total - 1e-9 or ts.count(2) > cur_ts.count(2) or (sum(ts) > sum(cur_ts) and not FIT_CLAIMS_PRICED)):
                        continue
                    key = key + (ids,)
                    if best is None or key < best[0]:
                        best = (key, chosen, table, ids, total)
            last.update(table=table, classes=list(classes), cover=cover)
            return best
        opened = searched = widened = sought = cheaper = 0
        best = None
        for _ in range(FIT_ROUNDS):
            if r.t() > GUARD_AT - 30:
                break
            best = solve()
            if best is None:
                if sought < FIT_SEEK and self.fit_seek(b, accs, last.get("table") or {}, tier, room() if broken else (room() - 4) // 4):
                    sought += 1
                    index = self.title_index()
                    continue
                if widened < FIT_WIDEN and self.fit_widen(b, accs, last):
                    widened += 1
                    index = self.title_index()
                    continue
                break
            _, chosen, table, _, _ = best
            unsure = [(ai, it) for ai, it in zip(lines, chosen) if state(it, ai) == "u" and ai not in loose]
            if unsure and opened < FIT_VERIFY_CAP and room() >= 3:
                ai, it = min(unsure, key=lambda p: (len(table[p[0]]), price(p[1]), p[1]["id"]))
                if not it.get("read"):
                    self.read_one(it)
                    opened += 1
                if it.get("read") and r.judge.line_level(it, ai) is None and r.judge.can_judge(1):
                    r.judge.judge_batch([(it, ai)])
                if state(it, ai) != "v":
                    dead.add((ai, it["id"]))
                continue
            if unsure or not r.cheapest or room() < 4:
                break
            if cheaper < FIT_SEEK and self.fit_cheaper(chosen, lines, tier, room()):
                cheaper += 1
                index = self.title_index()
                continue
            if searched >= FIT_VOI or r.plan.slack() < 4:
                break
            code = self.pointer_worth(chosen, accs, table, tier, info, index, price)
            if not code:
                break
            self.see_part(code)
            searched += 1
            index = self.title_index()
        if best is None and cur_ok:
            mark_verified()
            return True
        if best is None:
            r.notes.append(f"fit pivot: hub {b} no fitting code class (opened {opened}, searched {searched}, widened {widened}, sought {sought})")
            return False
        _, chosen, _, best_ids, best_total = best
        if any(state(it, ai) != "v" for ai, it in zip(lines, chosen) if ai not in loose):
            sure = solve(True)
            if sure is None and r.cheapest and cur_ok and cheaper < FIT_SEEK and room() >= 6 and r.t() < GUARD_AT - 40:
                cheaper += 1
                self.fit_cheaper([picks[x] for x in lines], lines, tier, room(), read=True)
                sure = solve(True)
            if sure is None or any(state(it, ai) != "v" for ai, it in zip(lines, sure[1]) if ai not in loose):
                r.notes.append(f"fit pivot: hub {b} best class not verified (opened {opened}, searched {searched})")
                return cur_ok
            best = sure
            _, chosen, _, best_ids, best_total = best
            r.notes.append(f"fit pivot: hub {b} falls back to the best verified class {best_ids}")
        changes = sum(1 for ai, it in zip(lines, chosen) if it is not picks[ai])
        room = max(0, r.steps_left() - r.plan.finish_need(minimal=True) - 1)
        if max(changes * r.order.swap_cost(1), r.order.move_extra({ai: it["id"] for ai, it in zip(lines, chosen)})) > room:
            r.notes.append(f"fit pivot: hub {b} fix needs {changes} swaps, no room")
            return True
        for ai, it in zip(lines, chosen):
            if it is not picks[ai]:
                art = r.articles[ai]
                old = picks[ai]["id"] if picks[ai] is not None else None
                r.notes.append(f"fit pivot: line {ai} {old} -> {it['id']}")
                art["backup"] = [y for y in ([old] if old else []) + list(art.get("backup") or []) if y != it["id"]]
                art["pick"] = it["id"]
        mark_verified()
        r.notes.append(f"fit pivot: hub {b} class {best_ids} total {best_total} (opened {opened}, searched {searched}, widened {widened})")
        return True

    def read_one(self, it: dict) -> None:
        r = self.r
        calls = r.hunt.read_calls([it["id"]], "facts")
        if calls:
            obs = r.env.send(calls[:1])
            r.hunt.shrink_compare([it["id"]], "facts", True, calls[:1], obs)

    def fit_widen(self, b: int, accs: list[int], last: dict) -> bool:
        r = self.r
        if not r.finder.sas() or r.t() > GUARD_AT - 40:
            return False
        allowance = min(FIT_WIDEN_STEPS, r.plan.slack() - 3 - len(accs))
        if allowance < 2:
            return False
        cover = last.get("cover") or {}
        lines = [a for a in accs if not r.articles[a].get("fit_widened")]
        if not last.get("classes") and not r.articles[b].get("fit_widened"):
            lines = [b]
        if not lines:
            return False
        target = min(lines, key=lambda a: (cover.get(a, 0), a))
        r.articles[target]["fit_widened"] = True
        before = r.steps
        was, extra = r.phase, r.plan.extra
        r.phase = "read"
        r.plan.extra = max(0, allowance - r.plan.room("read"))
        try:
            r.hunt.lattice(target, allowance, stop_at_set=False)
        finally:
            r.phase, r.plan.extra = was, extra
        r.notes.append(f"fit widen: line {target} searched {r.steps - before} steps (classes {len(last.get('classes') or [])}, cover {cover})")
        return r.steps > before

    def fit_cheaper(self, chosen: list[dict], lines: list[int], tier: Callable, allow: int, read: bool = False) -> bool:
        r = self.r
        price = self.price_of
        qty = {art["i"]: max(1, int(art.get("qty") or 1)) for art in r.articles}
        h = chosen[0]
        start = r.steps
        stop = r.steps + min(FIT_SEEK_STEPS, max(0, allow - 4))
        order = sorted(range(len(lines)), key=lambda k: (-price(chosen[k]) * qty[lines[k]], k))
        found = False
        for k in order:
            if stop - r.steps < 2:
                break
            ai, cur = lines[k], chosen[k]
            tried = r.articles[ai].setdefault("fit_cheaper", [])
            if price(cur) in tried or price(cur) <= 0:
                continue
            tried.append(price(cur))
            if k == 0:
                ok = lambda x, cur=cur: price(x) < price(cur) and all(tier(x, y) < 9 for y in chosen[1:])
            else:
                ok = lambda x, cur=cur: price(x) < price(cur) and tier(h, x) < 9
            r.hunt.fit_scan(ai, min(3, stop - r.steps - 1), ok=ok, cap=price(cur) - 0.01)
            pool = sorted((x for x in r.hunt.unread_pool(ai, r.articles[ai].get("ids") or []) if ok(x)), key=lambda x: (price(x), x["id"]))
            found = found or bool(pool)
            for x in pool[:2] if read else []:
                if stop - r.steps < 1:
                    break
                self.read_one(x)
        if r.steps > start:
            r.notes.append(f"fit cheaper: {r.steps - start} steps, cheaper fitting rows {'found' if found else 'none'}")
        return found

    def fit_seek(self, b: int, accs: list[int], table: dict, tier: Callable, allow: int) -> bool:
        r = self.r
        cat = r.cat
        price = self.price_of
        if r.t() > GUARD_AT - 40:
            return False
        good = {a: [x for x in table.get(a) or [] if r.basket.state(x, a) == "v"] for a in accs}
        if any(not good[a] and (not getattr(self, "fit_solved", False) or len([x for x in table.get(a) or [] if r.basket.state(x, a) != "x"]) < FIT_SCAN_KEEP)
               for a in accs):
            return False
        firm = min(accs, key=lambda a: (len(good[a]), a))
        stop = r.steps + min(FIT_SEEK_STEPS, allow - r.order.move_extra({ai: f"@{ai}" for ai in [b] + accs if ai != firm}))
        if stop - r.steps < 2:
            return False
        qty = {art["i"]: max(1, int(art.get("qty") or 1)) for art in r.articles}
        others = sum(price(cat.items[o["pick"]]) * qty[o["i"]] for o in r.articles
                     if o["i"] not in [b] + accs and cat.items.get(o.get("pick") or "") is not None)

        def cost(h: dict) -> float:
            return others + price(h) * qty[b] + sum(qty[a] * min((price(x) for x in good[a] if tier(h, x) < 9), default=0.0) for a in accs)
        hubs = [h for h in table.get(b) or [] if r.basket.state(h, b) == "v" and not (r.budget and cost(h) > r.budget + 1e-9)]
        lone = [a for a in accs if good[a] and not any(tier(h, x) < 9 for h in hubs for x in good[a])
                and f"@{a}" not in (r.articles[b].get("fit_sought") or [])]
        if lone:
            a = min(lone, key=lambda a: (len(good[a]), a))
            r.articles[b]["fit_sought"] = list(r.articles[b].get("fit_sought") or []) + [f"@{a}"]
            start = r.steps
            r.hunt.fit_scan(b, min(FIT_SCAN_PAGES, stop - r.steps - 2), ok=lambda h, a=a: any(tier(h, x) < 9 for x in good[a]))
            pool = sorted((h for h in r.hunt.unread_pool(b, r.articles[b].get("ids") or []) if any(tier(h, x) < 9 for x in good[a])),
                          key=lambda h: (price(h), h["id"]))
            if pool and stop > r.steps:
                r.hunt.open([h["id"] for h in pool[:min(2, stop - r.steps)]], force=True)
            r.notes.append(f"fit seek: line {a} has no fitting hub, hub scan {r.steps - start} steps, {len(pool)} fitting")
            if r.steps > start:
                return True
        reach = {a: sum(1 for x in table.get(a) or [] if any(tier(h, x) < 9 for h in hubs)) for a in accs}
        held = {ai: cat.items.get(r.articles[ai].get("pick") or "") for ai in [b] + accs}
        ranked = []
        for h in hubs:
            f = fit_info(cat.fit_record(h), "main")
            covered = [a for a in accs if any(tier(h, x) < 9 for x in table.get(a) or [])]
            keys = sorted(f.series) or sorted(f.own)
            gaps = [a for a in accs if a not in covered and any(k not in (r.articles[a].get("fit_sought") or []) for k in keys)]
            moves = (held[b] is None or held[b]["id"] != h["id"]) + sum(1 for a in accs if held[a] is None or tier(h, held[a]) >= 9)
            if gaps:
                ranked.append(((-len(covered), moves, -sum(reach[a] for a in gaps), price(h), h["id"]), h, f, keys, gaps))
        if not ranked:
            return False
        _, h, f, keys, gaps = min(ranked, key=lambda z: z[0])
        start = r.steps
        for a in gaps:
            art = r.articles[a]
            key = next(k for k in keys if k not in (art.get("fit_sought") or []))
            art["fit_sought"] = list(art.get("fit_sought") or []) + [key]
            if stop - r.steps < 2:
                break
            ids: list[str] = []
            if f.series:
                mine = set(Text(art.get("cat") or art["noun"]).stems)
                theirs = {s for o in r.articles if o is not art for s in Text(" ".join([o.get("cat") or "", o["noun"]])).stems}
                head = next((w for w in Text(art.get("cat") or art["noun"]).words if Text.stem(w) in mine - theirs), None) or (noun_heads(art["noun"]) or [""])[0]
                vals = r.hunt.search_terms(a)["values"]
                ids, _, _ = r.hunt.line_run(a, " ".join([key.lower()] + (Text(vals[0]).words[:2] if vals else []) + [head]).strip(), force=True)
                r.hunt.add_ids(art, ids)
            if not any(tier(h, x) < 9 for x in r.hunt.unread_pool(a, ids)) and stop - r.steps >= 3:
                r.hunt.fit_scan(a, min(FIT_SCAN_PAGES, stop - r.steps - 2), ok=lambda x, h=h: tier(h, x) < 9)
                ids = list(art.get("ids") or [])
            pool = sorted((x for x in r.hunt.unread_pool(a, ids) if tier(h, x) < 9), key=lambda x: (price(x), x["id"]))
            if pool and stop > r.steps:
                r.hunt.open([x["id"] for x in pool[:min(2, stop - r.steps)]], force=True)
            r.notes.append(f"fit seek: hub {h['id']} key {key} line {a}, {len(ids)} rows, {len(pool)} fitting")
        return r.steps > start

    def pointer_worth(self, chosen: list[dict], accs: list[int], table: dict, tier: Callable, info: Callable, index: dict, price: Callable) -> str | None:
        h = chosen[0]
        for a, cur in zip(accs, chosen[1:]):
            for x in table[a]:
                if price(x) >= price(cur) or tier(h, x) < 9:
                    continue
                for code in info(x, "dep").open_pointers():
                    if not index.get(fit_norm(code)):
                        return code
        return None

    def see_part(self, code: str) -> None:
        r = self.r
        was = r.phase
        r.phase = "proof"
        try:
            ids, _, _ = r.finder.run(code)
            carriers = [r.cat.items[i] for i in ids if i in r.cat.items and r.cat.title_carries(r.cat.items[i], code)]
            unread = [it["id"] for it in carriers if not (it.get("read") or it.get("paged"))]
            if unread and r.plan.slack() >= 4 and r.t() < GUARD_AT - 20:
                r.hunt.open(unread[:1], force=True)
        finally:
            r.phase = was

    def fit_pairs_solve(self, b: int, accs: list[int]) -> None:
        r = self.r
        cat = r.cat
        price = self.price_of
        qty = {art["i"]: max(1, int(art.get("qty") or 1)) for art in r.articles}
        tiers: dict[tuple[str, str], int] = {}
        lines = [b] + accs
        picks = {ai: cat.items.get(r.articles[ai].get("pick") or "") for ai in lines}
        if any(v is None for v in picks.values()):
            return
        depts = {cat.category(picks[a]) for a in accs}
        pages = [cat.fit_record(it) for it in cat.items.values() if (it.get("read") or it.get("paged")) and cat.category(it) in depts]

        def tier(h: dict, x: dict) -> int:
            key = (h["id"], x["id"])
            if key not in tiers:
                tiers[key] = self.tier(*fit_judge(cat.fit_record(h), cat.fit_record(x), pages))
            return tiers[key]
        if all(tier(picks[b], picks[a]) < 9 for a in accs):
            return
        cands = {}
        for ai in lines:
            pool = [it for it in r.candidates(ai) if it is not picks[ai] and r.ok_basic(it, ai) and not r.contradicted(it, ai)
                    and self.line_level(it, ai) == "exact"]
            pool.sort(key=lambda it: (price(it), it["id"]))
            cands[ai] = [picks[ai]] + pool[:FIT_ALTS]
        others = sum(price(cat.items[art["pick"]]) * qty[art["i"]] for art in r.articles
                     if art["i"] not in lines and cat.items.get(art.get("pick") or "") is not None)
        best = None
        for h in cands[b]:
            per = []
            for a in accs:
                ok = [x for x in cands[a] if tier(h, x) < 9]
                if not ok:
                    break
                per.append(ok)
            else:
                for combo in itertools.product(*per):
                    chosen = [h] + list(combo)
                    total = others + sum(price(it) * qty[ai] for ai, it in zip(lines, chosen))
                    if r.budget and total > r.budget + 1e-9:
                        continue
                    changes = sum(1 for ai, it in zip(lines, chosen) if it is not picks[ai])
                    score = sum(tier(h, x) for x in combo)
                    key = (score, total if r.cheapest else changes, changes, tuple(it["id"] for it in chosen))
                    if best is None or key < best[0]:
                        best = (key, chosen)
        if best is None:
            r.notes.append(f"fit solver: hub {b} has no confirmed fitting basket among verified listings")
            return
        room = max(0, r.steps_left() - r.plan.finish_need(minimal=True) - 1)
        changes = best[0][2]
        if max(changes * r.order.swap_cost(1), r.order.move_extra({ai: it["id"] for ai, it in zip(lines, best[1])})) > room:
            r.notes.append(f"fit solver: hub {b} fix needs {changes} swaps, no room")
            return
        for ai, it in zip(lines, best[1]):
            if it is not picks[ai]:
                art = r.articles[ai]
                r.notes.append(f"fit solver: line {ai} {picks[ai]['id']} -> {it['id']}")
                art["backup"] = [y for y in [picks[ai]["id"]] + list(art.get("backup") or []) if y != it["id"]]
                art["pick"] = it["id"]

    def solve_budget(self) -> None:
        r = self.r
        cat = r.cat
        if not r.budget or not r.articles:
            return
        picks = {art["i"]: cat.items.get(art.get("pick") or "") for art in r.articles}
        if any(v is None for v in picks.values()):
            return
        qty = {art["i"]: max(1, int(art.get("qty") or 1)) for art in r.articles}
        price = self.price_of
        if sum(price(it) * qty[ai] for ai, it in picks.items()) <= r.budget + 1e-9:
            return
        paired = {x for p in r.together for x in p}
        keys = sorted(picks)
        options = []
        for ai in keys:
            cur = picks[ai]
            alts = [] if ai in paired else [it for it in r.candidates(ai) if it is not cur and r.ok_basic(it, ai)
                                            and not r.contradicted(it, ai) and self.line_level(it, ai) == "exact"
                                            and price(it) < price(cur)]
            alts.sort(key=lambda it: (price(it), it["id"]))
            options.append([cur] + alts[:SOLVE_ALTS])
        room = max(0, r.steps_left() - r.plan.finish_need(minimal=True) - 1)
        best = None
        for combo in itertools.product(*options):
            total = sum(price(it) * qty[ai] for ai, it in zip(keys, combo))
            if total > r.budget + 1e-9:
                continue
            changes = sum(1 for ai, it in zip(keys, combo) if it is not picks[ai])
            if changes * r.order.swap_cost(1) > room:
                continue
            key = (changes, total, tuple(it["id"] for it in combo))
            if best is None or key < best[0]:
                best = (key, combo)
        if best is None:
            r.notes.append("solver: no basket within budget among verified listings")
            return
        for ai, it in zip(keys, best[1]):
            if it is not picks[ai]:
                art = r.articles[ai]
                r.notes.append(f"solver: line {ai} {picks[ai]['id']} -> {it['id']} to meet the budget")
                art["backup"] = [y for y in [picks[ai]["id"]] + list(art.get("backup") or []) if y != it["id"]]
                art["pick"] = it["id"]

    def better_by(self, alt: dict, pick: dict, by: Any) -> bool:
        if not isinstance(by, dict):
            return False
        key = Text.of(by.get("key")).strip()
        way = Text.of(by.get("direction")).strip().lower()
        if not key or way not in ("higher", "lower"):
            return False

        def number(it: dict) -> float | None:
            vals = self.r.cat.fact_values(it, key)
            if not vals:
                return None
            nums = Text(Text.of(vals[0][1])).numbers
            return float(nums[0][0]) if nums else None
        a, b = number(alt), number(pick)
        if a is None or b is None:
            return False
        return a > b if way == "higher" else a < b

    def metric_pick(self, tag: str) -> None:
        r = self.r
        if Text.of(r.spec.get("priority")) != "best" or r.t() > GUARD_AT - 40 or r.spent > CHEAP_ONLY_AT - 0.004:
            return
        detail = Text.of(r.spec.get("priority_detail")).strip()
        if not detail or r.together:
            return
        price = self.price_of
        seen = getattr(self, "metric_seen", None)
        if seen is None:
            seen = self.metric_seen = {}
        for art in r.articles:
            ai = art["i"]
            if r.steps_left() <= r.plan.finish_need(minimal=True) + r.order.swap_cost(1) + 1:
                return
            cur = r.cat.items.get(art.get("pick") or "")
            qty = max(1, int(art.get("qty") or 1))
            rest = sum(price(r.cat.items[a["pick"]]) * max(1, int(a.get("qty") or 1)) for a in r.articles
                       if a is not art and a.get("pick") in r.cat.items)
            pool = [it for it in r.candidates(ai) if it.get("read") and r.ok_basic(it, ai) and not r.contradicted(it, ai)
                    and self.line_level(it, ai) == "exact" and (not r.budget or rest + price(it) * qty <= r.budget + 1e-9)]
            pool.sort(key=lambda it: (price(it), it["id"]))
            pool = pool[:METRIC_ROWS]
            if cur is not None and cur not in pool:
                pool.append(cur)
            if len(pool) < 2:
                continue
            key = (ai, tuple(sorted(it["id"] for it in pool)), cur["id"] if cur else "")
            if key not in seen:
                rows = "\n".join(f"- {it['id']}{' (CURRENT PICK)' if it is cur else ''}: price {r.cat.price(it)}, title {r.cat.title(it)[:140]!r}, "
                                  f"facts {Text.json(it['facts'], 600)}" for it in pool)
                user = f"CHOOSING RULE: {detail}\nPRODUCT LINE: {art['noun']}\nLISTINGS (each already meets every requirement):\n{rows}"
                seen[key] = r.model_json("audit", METRIC_SYS, user, max_tokens=300, deadline=GUARD_AT - 25, chain=AUDIT_CHAIN)
            got = seen[key]
            if not isinstance(got, dict):
                continue
            ids = {it["id"]: it for it in pool}
            best = ids.get(Text.of(got.get("best")).strip())
            quote = Text.of(got.get("value")).strip()
            if best is None or best is cur or best not in pool or self.line_level(best, ai) != "exact":
                continue
            if not quote or not self.found(quote, self.page_of(best)):
                r.notes.append(f"metric {tag}: line {ai} choice {best['id']} without a quoted value, kept {cur['id'] if cur else None}")
                continue
            r.notes.append(f"metric {tag}: line {ai} {cur['id'] if cur else None} -> {best['id']} ({quote[:60]}) by '{detail[:40]}'")
            art["backup"] = [y for y in ([cur["id"]] if cur else []) + list(art.get("backup") or []) if y != best["id"]]
            art["pick"] = best["id"]

    def pick_fails(self, pick: dict, reqs: list[dict], f: dict) -> bool:
        try:
            n = int(f.get("req"))
        except (TypeError, ValueError):
            return False
        if not 0 <= n < len(reqs):
            return False
        req = reqs[n]
        if self.level(pick, req) == "exact" or self.own_phrase(pick, req):
            return False
        q = Text.of(f.get("quote"))
        if "=" not in q:
            return False
        k, v = (y.strip() for y in q.split("=", 1))
        if not v or not self.found(q, self.page_of(pick)):
            return False
        own = [key for key, _ in self.r.cat.fact_values(pick, req.get("prop") or "")]
        if own and not any(Text.key_stems(key) == Text.key_stems(k) for key in own):
            return False
        return Catalog.value_verdict(req, v) == "no"

    def export(self) -> dict:
        r = self.r
        picks = {}
        for art in r.articles:
            it = r.cat.items.get(art.get("pick") or "")
            if it is not None:
                picks[str(art["i"])] = self.line_level(it, art["i"])
        return {"asked": self.asked, "log": self.log[-40:], "picks": picks}

    def ledger_mark(self) -> tuple:
        r = self.r
        return (tuple((x["a"], x["prop"], Text.json(x["value"]), x["active"]) for x in r.ledger), r.budget)

    def line_marks(self) -> dict:
        r = self.r
        out = {}
        for art in r.articles:
            out[art["i"]] = (tuple((x["prop"], x["op"], Text.json(x["value"]), x["active"]) for x in r.ledger if x["a"] in (art["i"], None)),
                             r.budget, art.get("qty"))
        return out

    def changed_lines(self) -> set:
        r = self.r
        now = self.line_marks()
        was = getattr(self, "line_stamp", {})
        out = {ai for ai, m in now.items() if was.get(ai) != m}
        for art in r.articles:
            it = r.cat.items.get(art.get("pick") or "")
            if it is None or not r.ok_basic(it, art["i"]):
                out.add(art["i"])
        return out

    def map_keys(self) -> None:
        r = self.r
        cat = r.cat
        items: list[tuple[str, list[str]]] = []
        meta: list[tuple[int, str]] = []
        for art in r.articles:
            ai = art["i"]
            keys = sorted({k for it in r.owners(ai) for k in it["facts"]})
            if not keys:
                continue
            for req in r.reqs(ai):
                nk = Text.key(req["prop"])
                if nk in cat.aliases or (ai, nk) in meta or any(cat.fact_values(it, req["prop"]) for it in r.owners(ai)):
                    continue
                items.append((req["prop"], keys[:60]))
                meta.append((ai, nk))
        items, meta = items[:12], meta[:12]
        if not items or r.t() > GUARD_AT - 40:
            return

        def call(sy: str, u: str, m: int) -> str:
            return r.model("read", sy, u, max_tokens=m, deadline=GUARD_AT - 30).text
        got = rd_map_keys(call, items)
        for n, (ai, nk) in enumerate(meta):
            cat.aliases[nk] = [Text.of(k) for k in (got.get(n) or []) if Text.of(k) in items[n][1]][:3]

    def judge_wording(self) -> None:
        r = self.r
        cat = r.cat
        triples: list[tuple[str, str, str]] = []
        keys: list[tuple[str, str]] = []
        for art in r.articles:
            ai = art["i"]
            for req in r.reqs(ai):
                if req["op"] not in ("is", "eq"):
                    continue
                wanted = req["value"] if isinstance(req["value"], list) else [req["value"]]
                w = Text.json(wanted)
                for it in r.candidates(ai)[:14]:
                    if self.level(it, req) is not None:
                        continue
                    for k, v in cat.fact_values(it, req["prop"]):
                        key = (w, v.lower()[:80])
                        if key not in cat.equiv and key not in keys and cat.value_verdict(req, v) == "no":
                            keys.append(key)
                            triples.append((" or ".join(Text.of(x) for x in wanted), v, k))
        triples, keys = triples[:40], keys[:40]
        if not triples or r.t() > GUARD_AT - 25:
            return

        def call(sy: str, u: str, m: int) -> str:
            return r.model("read", sy, u, max_tokens=m, deadline=GUARD_AT - 15).text
        got = rd_judge_wording(call, triples)
        for n, key in enumerate(keys):
            cat.equiv[key] = bool(got.get(n))

    def activate_whens(self) -> None:
        r = self.r

        def value_of(ai: Any, prop: str) -> list[str]:
            if not isinstance(ai, int) or not 0 <= ai < len(r.articles):
                return []
            art = r.articles[ai]
            it = r.cat.items.get(art.get("pick") or "") or next(iter(r.candidates(ai)), None)
            return [v for _, v in r.cat.fact_values(it, prop)] if it else []
        for e in r.ledger:
            if e.get("when"):
                e["active"] = when_applies(e, value_of)

    def describe(self, art: dict, cands: list[dict]) -> str:
        r = self.r
        cat = r.cat
        ai = art["i"]
        rl = "; ".join(f"#{n} {x['prop']} {x['op']} {Text.json(x['value'])}" for n, x in enumerate(r.reqs(ai))) or "none stated"
        rows = [f"  {it['id']}: price {cat.price(it)}, stock {cat.in_stock(it)}, category {cat.category(it)!r}, "
                f"title {cat.title(it)!r}; properties {Text.json(it['facts'], 900)}"
                + (f"; check={self.line_level(it, ai)}" if self.line_level(it, ai) else "") for it in cands]
        return f"article {ai}: {art['noun']} x{art['qty']}\n requirements: {rl}\n candidates:\n" + "\n".join(rows)

    @staticmethod
    def wanted_unit(req: dict) -> str:
        if req.get("unit"):
            return req["unit"]
        stated = req["value"] if isinstance(req["value"], str) else ""
        for _, u in Text(stated).numbers:
            if u:
                return u
        return ""

    def converted_fails(self, req: dict, quote: Any) -> bool:
        q = Text.of(quote)
        if "=" not in q:
            return False
        unit = self.wanted_unit(req)
        if not unit:
            return False
        with_units = [(n, u) for n, u in Text(q.split("=")[-1]).numbers if u]
        if not with_units:
            return False
        n, u = with_units[-1]
        if Text.unit_scale(u, unit) is None:
            return False
        return Catalog.value_verdict(req, f"{n} {u}") == "no"

    def quote_ok(self, it: dict, quote: Any, req: dict | None = None) -> bool:
        cat = self.r.cat
        q = Text.of(quote)
        if "=" not in q:
            return False
        k, v = (x.strip() for x in q.split("=", 1))
        vs = Text(v).stems
        if not vs:
            return False
        own = [key for key, _ in cat.fact_values(it, req.get("prop") or "")] if req else []
        if own and not any(Text.key_stems(key) == Text.key_stems(k) for key in own):
            return False
        if Text.key(k) in {"title", Text.key(cat.store.title or "title")} and Text.run_in(vs[:6], Text(cat.title(it)).stems):
            return not req or Catalog.value_verdict(req, cat.title(it)) != "no"
        for key, val in it["facts"].items():
            if Text.key_stems(key) == Text.key_stems(k) and Text.run_in(vs[:6], Text(val).stems):
                return not req or Catalog.value_verdict(req, val) != "no"
        return False

    def vetoes(self, it: dict, ai: int, first: bool, ev: dict) -> list[int]:
        r = self.r
        cat = r.cat
        bad = []
        for k, req in enumerate(r.reqs(ai)):
            lv = self.level(it, req)
            if lv is not None:
                if lv == "no":
                    bad.append(k)
                continue
            shown = ev.get(str(k)) or ev.get(k)
            v = cat.verdict(it, req)
            numeric = req["op"] in ("min", "max", "between")
            failing = v == "no" or (v == "unknown" and req["op"] != "not" and not cat.mentions(it, req) and not (numeric and cat.fact_values(it, req["prop"])))
            if failing and not (first and self.quote_ok(it, shown, req)):
                bad.append(k)
            elif failing and v == "unknown":
                it.setdefault("quoted", []).append(Text.json(req, 200))
            elif first and numeric and self.converted_fails(req, shown):
                bad.append(k)
        return bad

    def nearest_miss(self, cands: list[dict], ai: int, raw: dict | None) -> dict | None:
        r = self.r
        good = [it for it in cands if not r.contradicted(it, ai)]
        if good:
            return good[0]
        unknown = [it for it in cands if r.fit(it, ai)[1] == 0]
        if unknown:
            return min(unknown, key=lambda it: 1e18 if r.cat.price(it) is None else r.cat.price(it))
        return raw

    def settle(self, art: dict, cands: list[dict], got: Any) -> None:
        ai = art["i"]
        ids = {it["id"]: it for it in cands}
        order = [got.get("pick")] + list(got.get("backups") or []) if isinstance(got, dict) else []
        ev = got.get("evidence") if isinstance(got, dict) and isinstance(got.get("evidence"), dict) else {}
        pick = None
        for n, pid in enumerate(order):
            it = ids.get(Text.of(pid))
            if it is not None and not self.vetoes(it, ai, n == 0, ev):
                pick = it
                break
        if pick is None and cands:
            pick = self.nearest_miss(cands, ai, ids.get(Text.of(order[0])) if order else None)
        pick = self.cheapest_full(pick, cands, ai)
        pick = self.policy(pick, cands, ai, [Text.of(x) for x in order])
        pick = self.exact_over_miss(pick, cands, ai, got, ev, [Text.of(x) for x in order])
        if pick is not None and order and Text.of(order[0]) == pick["id"] and got.get("all_met") is True \
                and all(ev.get(str(k)) for k in range(len(self.r.reqs(ai)))):
            art.setdefault("model_met", {})[pick["id"]] = Text.json(self.r.reqs(ai), 2000)
        art["pick"] = pick["id"] if pick else None
        art["backup"] = [Text.of(x) for x in order[1:] if Text.of(x) in ids]
        art["judged_none"] = (isinstance(got, dict) and got.get("all_met") is False) or (
            pick is not None and self.judged(pick, ai) and self.tier_of(pick, ai) == 1)
        art["no_pick"] = isinstance(got, dict) and not got.get("pick")

    def exact_over_miss(self, pick: dict | None, cands: list[dict], ai: int, got: Any, ev: dict, order: list[str]) -> dict | None:
        r = self.r
        if pick is None or not isinstance(got, dict) or got.get("all_met") is not False or self.line_level(pick, ai) == "exact":
            return pick
        others = 0.0
        for art in r.articles:
            o = r.cat.items.get(art.get("pick") or "") if art["i"] != ai else None
            if o is not None and self.price_of(o) < 1e17:
                others += self.price_of(o) * max(1, int(art.get("qty") or 1))
        qty = max(1, int(r.articles[ai].get("qty") or 1))
        sure = [it for it in cands if it is not pick and self.line_level(it, ai) == "exact" and r.ok_basic(it, ai)
                and not r.contradicted(it, ai) and r.fit(it, ai)[1] == 0 and not self.vetoes(it, ai, False, ev)
                and (not r.budget or self.price_of(it) * qty + others <= r.budget)]
        if not sure:
            return pick
        spot = {x: n for n, x in enumerate(order)}
        if r.cheapest:
            choice = min(sure, key=lambda it: (self.price_of(it), it["id"]))
        else:
            choice = min(sure, key=lambda it: (spot.get(it["id"], len(spot)), self.price_of(it), it["id"]))
        r.notes.append(f"line {ai}: reply marked all_met false on {pick['id']}; exact candidate {choice['id']} taken instead")
        return choice

    def look_marks(self, it: dict, reqs: list[dict]) -> str:
        spans = it.get("jspan") or {}
        out = []
        for n, q in enumerate(reqs):
            span = Text.of(spans.get(self.sig(q))).strip()
            out.append(f"#{n} {self.level(it, q) or 'unchecked'}" + (f" ({span[:120]})" if span else ""))
        return "; ".join(out)

    def last_look(self, only: list[int] | None = None) -> None:
        r = self.r
        cat = r.cat
        if not r.articles or r.t() > GUARD_AT - 40 or r.spent > CHEAP_ONLY_AT:
            return
        paired = {x for p in r.together for x in p}
        blocks, cmap = [], {}
        for art in r.articles:
            ai = art["i"]
            pick = cat.items.get(art.get("pick") or "")
            reqs = self.judgeable(ai)
            if not reqs or (only is None and (pick is None or not self.judged_no(pick, ai))) or (only is not None and ai not in only):
                continue
            alts = [it for it in r.candidates(ai) if it is not pick and not self.judged_no(it, ai)]
            alts.sort(key=lambda it: (self.line_level(it, ai) != "exact", self.price_of(it), it["id"]))
            alts = alts[:LOOK_ALTS]
            if pick is None and not alts:
                alts = sorted(r.candidates(ai), key=lambda it: (self.price_of(it), it["id"]))[:LOOK_ALTS]
            cmap[ai] = {it["id"]: it for it in ([pick] if pick is not None else []) + alts}
            rl = "\n".join(f"    #{n} {self.req_text(q)}" for n, q in enumerate(reqs))
            rows = ([f"  CHOSEN {pick['id']}, price {cat.price(pick)}; check: {self.look_marks(pick, reqs)}\n{self.page_of(pick)}"]
                    if pick is not None else ["  CHOSEN none"])
            rows += [f"  OTHER {it['id']}, price {cat.price(it)}; check: {self.look_marks(it, reqs)}\n{self.page_of(it)}" for it in alts]
            blocks.append(f"ARTICLE {ai}: {art['noun']} x{art['qty']}\n  requirements:\n{rl}\n" + "\n".join(rows))
        if not blocks:
            return
        if r.cheapest:
            rule = "the cheapest listing that meets every requirement"
        elif Text.of(r.spec.get("priority")) == "best":
            rule = f"the best by {Text.of(r.spec.get('priority_detail')) or 'the measure the request names'} among listings that meet every requirement"
        else:
            rule = "any listing that meets every requirement"
        user = f"REQUEST:\n{r.request[:3500]}\n\nSELECTION RULE: {rule}\nBUDGET: {r.budget}\n\n" + "\n\n".join(blocks)
        d = r.model_json("decide", LOOK_SYS, user, max_tokens=LOOK_TOKENS, deadline=GUARD_AT - 20)
        got = d if isinstance(d, dict) else {}
        for ai, opts in cmap.items():
            art = r.articles[ai]
            x = got.get(str(ai))
            reqs = self.judgeable(ai)
            pick = opts.get(art.get("pick") or "")
            if not isinstance(x, dict):
                art["look"] = "unanswered"
                r.notes.append(f"last look: line {ai} no answer")
                continue
            verdict = Text.of(x.get("verdict")).strip().lower()
            quotes = x.get("quotes") if isinstance(x.get("quotes"), dict) else {}

            def proven(it: dict) -> bool:
                return all(self.level(it, q) == "exact" or self.own_phrase(it, q) or self.quote_ok(it, quotes.get(str(n)) or quotes.get(n), q)
                           for n, q in enumerate(reqs))
            alt = opts.get(Text.of(x.get("id")))
            if verdict == "none":
                art["look"] = "none"
            elif verdict == "keep" and pick is not None and proven(pick):
                art["look"] = "keep"
            elif (verdict == "switch" and alt is not None and alt is not pick and ai not in paired and proven(alt)
                  and r.steps_left() > r.plan.finish_need(minimal=True) + r.order.swap_cost(1)):
                art["backup"] = [y for y in ([pick["id"]] if pick is not None else []) + list(art.get("backup") or []) if y != alt["id"]]
                art["pick"] = alt["id"]
                art["look"] = "switch"
            else:
                art["look"] = "unproven"
            r.notes.append(f"last look: line {ai} {pick['id'] if pick is not None else 'none'} {verdict or 'no verdict'} -> {art['look']} {art.get('pick')}")

    def separate_departments(self, cmap: dict) -> None:
        r = self.r
        cat = r.cat
        for art in r.articles:
            ai = art["i"]
            pick = cat.items.get(art.get("pick") or "")
            if pick is None:
                continue
            dep = cat.category(pick)
            own = set(Text(art["noun"]).stems)
            if not dep or own & set(Text(dep).stems):
                continue
            rivals = [b for b in r.articles if b["i"] != ai and cat.category(cat.items.get(b.get("pick") or "") or {"rec": {}, "row": {}}) == dep]
            if not rivals:
                continue
            alt = [c for c in (cmap.get(ai) or r.candidates(ai)) if c is not pick and not r.contradicted(c, ai)
                   and cat.category(c) != dep and (own & set(Text(cat.category(c)).stems))]
            if alt:
                art["pick"] = alt[0]["id"]
                r.notes.append(f"department: article {ai} moved off {dep!r} to {cat.category(alt[0])!r}")

    def fully_met(self, it: dict, ai: int) -> bool:
        y, n, u = self.r.fit(it, ai)
        return n == 0 and u == 0

    def verified(self, it: dict | None, ai: int) -> bool:
        r = self.r
        if it is None or not r.ok_basic(it, ai) or r.contradicted(it, ai):
            return False
        y, n, u = r.fit(it, ai)
        return y > 0 and n == 0 and u == 0

    def equiv_only(self, it: dict | None, ai: int) -> bool:
        cat = self.r.cat
        if it is None or not cat.equiv or self.line_level(it, ai) is not None:
            return False
        reqs = [q for q in self.r.reqs(ai) if q.get("op") in ("is", "eq") and self.level(it, q) is None]
        got = [cat.verdict(it, q) for q in reqs]
        if "yes" not in got:
            return False
        saved, cat.equiv = cat.equiv, {}
        try:
            return any(a == "yes" and cat.verdict(it, q) != "yes" for a, q in zip(got, reqs))
        finally:
            cat.equiv = saved

    def cheaper_met(self, it_o: dict | None, it_n: dict | None, art: dict) -> bool:
        r = self.r
        ai = art["i"]
        if it_o is None or it_n is None or (art.get("model_met") or {}).get(it_o["id"]) != Text.json(r.reqs(ai), 2000):
            return False
        po, pn = r.cat.price(it_o), r.cat.price(it_n)
        return (po is not None and pn is not None and po < pn and r.ok_basic(it_o, ai) and self.line_level(it_o, ai) in (None, "exact")
                and r.fit(it_o, ai)[1] == 0 and not any(r.cat.verdict(it_o, q) == "no" for q in r.reqs(ai)))

    def keep_verified(self, before: dict) -> None:
        r = self.r
        for art in r.articles:
            ai = art["i"]
            old, new = before.get(ai), art.get("pick")
            if not old or old == new:
                continue
            it_o = r.cat.items.get(old)
            it_n = r.cat.items.get(new or "")
            if r.cheapest and self.cheaper_met(it_o, it_n, art):
                art["pick"] = old
                art["backup"] = [x for x in [new] + list(art.get("backup") or []) if x and x != old]
                r.notes.append(f"kept model-met cheaper pick {old} on line {ai} over {new}")
                continue
            if not self.verified(it_o, ai):
                continue
            if self.equiv_only(it_o, ai):
                r.notes.append(f"kept the model's pick {new} on line {ai}: {old} matched only by the wording check")
                continue
            if self.verified(it_n, ai) and not (r.cheapest and r.cat.price(it_o) is not None and r.cat.price(it_o) == r.cat.price(it_n)):
                continue
            art["pick"] = old
            art["backup"] = [x for x in [new] + list(art.get("backup") or []) if x and x != old]
            r.notes.append(f"kept verified pick {old} on line {ai} over {new}")

    def cheapest_full(self, pick: dict | None, cands: list[dict], ai: int) -> dict | None:
        r = self.r
        if not r.cheapest or not cands:
            return pick
        full = [it for it in cands if self.fully_met(it, ai)]
        if not full:
            return pick

        def price(it: dict) -> float:
            p = r.cat.price(it)
            return 1e18 if p is None else p
        best = min(full, key=lambda it: (price(it), -self.own_score(it, ai), it["id"]))
        if pick is None or r.fit(pick, ai)[1] > 0 or (self.fully_met(pick, ai) and price(pick) > price(best)):
            return best
        return pick

    def decide_tokens(self) -> int:
        return min(DECIDE_TOKENS_MAX, DECIDE_TOKENS_BASE + DECIDE_TOKENS_LINE * max(1, len(self.r.articles)))

    def decide(self, strong: bool = True) -> None:
        r = self.r
        cat = r.cat
        self.activate_whens()
        self.map_keys()
        self.judge_lines()
        self.judge_wording()
        cmap = {art["i"]: r.candidates(art["i"])[:9] for art in r.articles}
        blocks = [self.describe(art, cmap[art["i"]]) for art in r.articles]
        if not any(cmap.values()):
            return
        pr = Text.of(r.spec.get("priority")) or "none"
        if r.together:
            blocks = [b + "".join(f"\n  {it['id']} notes: {cat.notes_of(it)[:260]}" for it in cmap.get(art["i"], []))
                      for b, art in zip(blocks, r.articles)]
            blocks.append("MUST WORK TOGETHER (article pairs): " + Text.json(r.together))
        user = (f"REQUEST:\n{r.request[:4000]}\n\nPRIORITY: {pr} {Text.of(r.spec.get('priority_detail'))}\n"
                f"BUDGET (store units): {r.budget}\n\n" + "\n\n".join(blocks))
        d = decide_adapter(r.model_json("decide" if strong else "read", DECIDE_SYS, user, max_tokens=self.decide_tokens(), deadline=GUARD_AT - 8))
        if not isinstance(d, dict):
            d = decide_adapter(r.model_json("read", DECIDE_SYS, user, max_tokens=self.decide_tokens(), deadline=GUARD_AT - 8))
        if not isinstance(d, dict):
            r.notes.append("decide: no reply, picks kept")
            for art in r.articles:
                if not art.get("pick") and cmap.get(art["i"]):
                    art["pick"] = cmap[art["i"]][0]["id"]
                cur = r.cat.items.get(art.get("pick") or "")
                new = self.policy(cur, cmap.get(art["i"]) or [], art["i"], [art["pick"]] if cur is not None else [])
                if new is not None and new is not cur:
                    art["pick"] = new["id"]
            self.check_fits(cmap)
            return
        self.decisions.append({"in": {str(ai): [(it["id"], cat.price(it), r.fit(it, ai), cat.category(it)) for it in c]
                                      for ai, c in cmap.items()}, "out": d})
        before = {art["i"]: art.get("pick") for art in r.articles}
        sound = self.fit_sound(before)
        for art in r.articles:
            self.settle(art, cmap.get(art["i"]) or [], d.get(str(art["i"])))
        self.keep_verified(before)
        self.keep_fitting(before, sound)
        self.separate_departments(cmap)
        self.settle_whens()
        self.stamp = self.ledger_mark()
        self.line_stamp = self.line_marks()
        self.check_fits(cmap)

    def settle_whens(self) -> None:
        r = self.r
        cat = r.cat
        n = len(r.articles)
        paired = {x for p in r.together for x in p}
        for e in r.ledger:
            w = e.get("when")
            a, b = (w or {}).get("article"), e.get("a")
            if not w or not isinstance(a, int) or not isinstance(b, int) or a == b or not (0 <= a < n and 0 <= b < n) or a in paired:
                continue
            rest = [q for q in r.reqs(b) if q is not e]
            fine = [x for x in r.candidates(b) if not any(cat.verdict(x, q) == "no" or r.blind(x, q) for q in rest)]
            if not fine or any(cat.verdict(x, e) == "yes" for x in fine):
                continue
            alts = [x for x in r.candidates(a) if cat.waives(x, w) and not r.contradicted(x, a)]
            if not alts:
                continue
            art = r.articles[a]
            art["waives"] = [x for x in art.get("waives") or [] if x != w] + [w]
            cur = cat.items.get(art.get("pick") or "")
            if cur is not None and cat.waives(cur, w):
                continue
            best = min(alts, key=lambda x: (self.tier_of(x, a), self.price_of(x) if r.cheapest else 0, x["id"]))
            art["backup"] = [y for y in ([cur["id"]] if cur is not None else []) + list(art.get("backup") or []) if y != best["id"]]
            art["pick"] = best["id"]
            r.notes.append(f"when: line {a} -> {best['id']} so line {b} may skip {e['prop']}")
        self.activate_whens()

    def fit_line(self, it: dict) -> str:
        cat = self.r.cat
        return f"{it['id']}: {cat.title(it)[:90]} | {Text.json(it['facts'], 500)} | notes: {cat.notes_of(it)[:600]}"

    def fit_pairs(self, a: int, b: int, ca: list[dict], cb: list[dict], known_b: list[dict] | None = None) -> tuple[list, Any]:
        r = self.r
        user = (f"GROUP A ({r.articles[a]['noun']}):\n" + "\n".join(self.fit_line(it) for it in ca)
                + f"\n\nGROUP B ({r.articles[b]['noun']}, must work with A):\n" + "\n".join(self.fit_line(it) for it in cb))
        d = r.model_json("read", FIT_SYS, user, max_tokens=500, deadline=GUARD_AT - 10)
        pairs = d.get("pairs") if isinstance(d, dict) else None
        if not isinstance(pairs, list):
            return [], d
        ida = {it["id"]: it for it in ca}
        idb = {it["id"]: it for it in (known_b or []) + cb}
        ok = [(ida[Text.of(p[0])], idb[Text.of(p[1])]) for p in pairs if isinstance(p, list) and len(p) == 2
              and Text.of(p[0]) in ida and Text.of(p[1]) in idb]
        return [pr for pr in ok if r.cat.codes_agree(pr[0], pr[1])], d

    def fit_search(self, b: int, ca: list[dict], d: Any) -> list[dict]:
        r = self.r
        models = d.get("models") if isinstance(d, dict) and isinstance(d.get("models"), dict) else {}
        names = [Text.of(x) for it in ca for x in (models.get(it["id"]) or []) if Text.of(x).strip()]
        names = [x for x in dict.fromkeys(names) if re.search(r"\d", x)][:3]
        noun = Text(r.articles[b]["noun"]).words[-1:]
        found: list[str] = []
        for nm in names:
            if r.plan.slack() <= 4 or r.t() > GUARD_AT - 30:
                break
            ids, _, _ = r.finder.run(" ".join(Text(nm).words[:2] + noun))
            found += [i for i in ids if i not in found]
        pool = r.hunt.unread_pool(b, found)
        if not pool:
            return []
        r.hunt.open([it["id"] for it in pool[:6]])
        return pool

    def note_pages(self, listings: list[dict]) -> None:
        r = self.r
        if not (r.tm.get("specs") and r.tm.get("detail")):
            return
        need = [it for it in listings if not it.get("paged") and not it.get("noted")][:8]
        room = min(len(need), r.plan.slack() - 2)
        if room > 0:
            r.hunt.open([it["id"] for it in need[:room]], what="state", force=True)
            for it in need[:room]:
                it["noted"] = True

    def match_table(self, mains: list[dict], accs: list[dict], pages: list) -> dict[tuple[str, str], tuple[str, str]]:
        cat = self.r.cat
        table = {}
        for m in mains:
            for a in accs:
                table[(m["id"], a["id"])] = fit_judge(cat.fit_record(m), cat.fit_record(a), pages)
        return table

    def pointer_codes(self, table: dict, mains: list[dict], accs: list[dict]) -> list[str]:
        cat = self.r.cat
        by_id = {it["id"]: it for it in mains + accs}
        codes: list[str] = []
        for (mid, aid), (_, why) in table.items():
            if "open " not in why:
                continue
            for code in fit_pointer_pages(cat.fit_record(by_id[mid])) + fit_pointer_pages(cat.fit_record(by_id[aid])):
                if fit_norm(code) not in {fit_norm(c) for c in codes}:
                    codes.append(code)
        return codes

    def open_pointer_pages(self, codes: list[str]) -> list[tuple[str, str]]:
        r = self.r
        cat = r.cat
        what = "state" if (r.tm.get("specs") and r.tm.get("detail")) else "facts"
        pages: list[tuple[str, str]] = []
        was = r.phase
        r.phase = "chains"
        for code in codes:
            if r.plan.room("chains") < 2 or r.t() > GUARD_AT - 30:
                break
            ids, _, _ = r.finder.run(code)
            carriers = [it for it in cat.items.values() if cat.title_carries(it, code)]
            unseen = [it["id"] for it in carriers if not (it.get("read") or it.get("paged"))]
            if unseen and r.plan.room("chains") >= 1:
                r.hunt.open(unseen, what=what)
            pages += [cat.fit_record(it) for it in carriers]
        r.phase = was
        return pages

    @staticmethod
    def tier(verdict: str, why: str) -> int:
        if verdict == "yes":
            return 0 if "pointer" not in why else 1
        if verdict == "unknown" and why.startswith("own page line"):
            return 2
        return 9

    def pair_cost(self, pr: tuple[dict, dict], a: int, b: int) -> int:
        r = self.r
        return sum(n + u for _, n, u in (r.fit(pr[0], a), r.fit(pr[1], b)))

    def shared_cover(self, m: dict, later: list[tuple[int, list[dict]]]) -> int:
        cat = self.r.cat
        return sum(1 for _, ca2 in later if any(self.tier(*fit_judge(cat.fit_record(m), cat.fit_record(x), [])) < 9 for x in ca2))

    def choose_pair(self, ok: list[tuple[dict, dict]], a: int, b: int) -> tuple[dict, dict]:
        r = self.r
        cat = r.cat
        low = min(self.pair_cost(pr, a, b) for pr in ok)
        ok = [pr for pr in ok if self.pair_cost(pr, a, b) == low]
        if r.cheapest:
            price = lambda it: 1e18 if cat.price(it) is None else cat.price(it)
            acc = min((pr[0] for pr in ok), key=price)
            mains = [pr[1] for pr in ok if pr[0] is acc]
            return acc, min(mains, key=price)
        cur = next((pr for pr in ok if pr[0]["id"] == r.articles[a]["pick"] and pr[1]["id"] == r.articles[b]["pick"]), None)
        return cur or ok[0]

    def model_pairs(self, a: int, b: int, ca: list[dict], cb: list[dict]) -> list[tuple[dict, dict]]:
        r = self.r
        ok, d = self.fit_pairs(a, b, ca, cb)
        if d is None or not isinstance(d, dict) or not isinstance(d.get("pairs"), list):
            return []
        r.notes.append(f"fit {a}+{b}: {len(ok)} fitting pairs of {len(ca)}x{len(cb)}")
        if ok or r.articles[b].get("fit_searched"):
            return ok
        r.articles[b]["fit_searched"] = True
        if not self.fit_search(b, ca, d):
            return ok
        known = {it["id"] for it in cb}
        more = [it for it in r.candidates(b) if not r.contradicted(it, b) and it["id"] not in known][:6]
        if not more:
            return ok
        table = self.match_table(more, ca, [])
        ok = [(ida, idb) for ida in ca for idb in more if self.tier(*table[(idb["id"], ida["id"])]) < 9]
        if not ok and all(v == "unknown" for v, _ in table.values()):
            ok, _ = self.fit_pairs(a, b, ca, more, cb)
        r.notes.append(f"fit {a}+{b} after model search: {len(ok)} pairs of {len(ca)}x{len(more)}")
        return ok

    def fit_sound(self, picks: dict) -> set[int]:
        r = self.r
        cat = r.cat
        index = self.title_index()
        out: set[int] = set()
        for b, accs in self.fit_hubs().items():
            group = [b] + [a for a in accs if a != b]
            its = {x: cat.items.get(picks.get(x) or "") for x in group}
            if any(it is None or r.basket.state(it, x) != "v" for x, it in its.items()):
                continue
            h = its[b]
            pages = lambda x: [cat.fit_record(q) for c in fit_info(cat.fit_record(x), "dep").pointers for q in index.get(fit_norm(c), []) if q is not x]
            if all(self.tier(*fit_judge(cat.fit_record(h), cat.fit_record(its[a]), pages(its[a]))) < 9 for a in group[1:]):
                out.update(group)
        return out

    def keep_fitting(self, before: dict, sound: set[int]) -> None:
        r = self.r
        if not sound:
            return
        now = {art["i"]: art.get("pick") for art in r.articles}
        if sound <= self.fit_sound(now):
            return
        moved = [ai for ai in sorted(sound) if now.get(ai) != before.get(ai)]
        for ai in moved:
            art = r.articles[ai]
            art["backup"] = [x for x in [now.get(ai)] + list(art.get("backup") or []) if x and x != before.get(ai)]
            art["pick"] = before.get(ai)
        if moved:
            r.notes.append(f"fit basket kept on lines {moved}: the new picks do not fit together")

    def check_fits(self, cmap: dict) -> None:
        r = self.r
        if not r.together or r.t() > GUARD_AT - 20:
            return
        fixed: set[int] = set()
        sound = self.fit_sound({art["i"]: art.get("pick") for art in r.articles})
        pairs = [pr for pr in r.together[:4] if not (pr[0] in sound and pr[1] in sound)]
        for k, (a, b) in enumerate(pairs):
            ca = [it for it in cmap.get(a, []) if not r.contradicted(it, a)][:6]
            cb = [it for it in cmap.get(b, []) if not r.contradicted(it, b)][:6]
            if not ca or not cb:
                r.notes.append(f"fit {a}+{b}: contradicted candidates used on {'A' if not ca else ''}{'B' if not cb else ''}")
                ca = ca or cmap.get(a, [])[:6]
                cb = cb or cmap.get(b, [])[:6]
            if a in fixed:
                ca = [it for it in ca if it["id"] == r.articles[a]["pick"]] or ca
            if b in fixed:
                cb = [it for it in cb if it["id"] == r.articles[b]["pick"]] or cb
            if not ca or not cb:
                continue
            self.note_pages(ca + cb)
            table = self.match_table(cb, ca, [])
            codes = self.pointer_codes(table, cb, ca)
            if codes:
                pages = self.open_pointer_pages(codes)
                if pages:
                    table = self.match_table(cb, ca, pages)
            ranked = sorted(((self.tier(*table[(m["id"], x["id"])]), n, x, m) for n, (x, m) in enumerate((x, m) for m in cb for x in ca)),
                            key=lambda t: (t[0], t[1]))
            ok = [(x, m) for t, _, x, m in ranked if t < 9]
            clean = [pr for pr in ok if not r.contradicted(pr[0], a)]
            if clean:
                ok = clean
            verdicts = [v for v, _ in table.values()]
            r.notes.append(f"fit {a}+{b}: matcher {len(ok)} acceptable of {len(ca)}x{len(cb)}; "
                           + ", ".join(f"{v}:{verdicts.count(v)}" for v in ("yes", "unknown", "no") if verdicts.count(v)))
            if not ok and r.plan.slack() > 4 and not r.articles[a].get("code_searched"):
                r.articles[a]["code_searched"] = True
                mains = [m for m in cb if m["id"] == r.articles[b].get("pick")] or cb[:1]
                fam = {fit_kind(c)[0] for x in ca for c in fit_info(r.cat.fit_record(x)).sown}
                ser = [c for m in mains for c in sorted(fit_info(r.cat.fit_record(m), "main").series) if fit_kind(c)[0] in fam]
                codes = list(dict.fromkeys(ser + [c for m in mains for c in sorted(fit_info(r.cat.fit_record(m)).own)]))[:2]
                head = noun_heads(r.articles[a]["noun"])[:1]
                found = []
                for code in codes:
                    ids, _, _ = r.hunt.line_run(a, " ".join([code.lower()] + head), force=True)
                    found += ids
                pool = r.hunt.unread_pool(a, found)
                if pool:
                    r.hunt.open([it["id"] for it in pool[:3]], force=True)
                    more = [it for it in r.cat.items.values() if it["id"] in found and it.get("read") and r.ok_basic(it, a)]
                    ca = list({it["id"]: it for it in ca + more}.values())[:9]
                    table = self.match_table(cb, ca, [])
                    ranked = sorted(((self.tier(*table[(m["id"], x["id"])]), n, x, m) for n, (x, m) in enumerate((x, m) for m in cb for x in ca)),
                                    key=lambda t: (t[0], t[1]))
                    ok = [(x, m) for t, _, x, m in ranked if t < 9]
                    verdicts = [v for v, _ in table.values()]
                    r.notes.append(f"fit {a}+{b}: searched main codes {codes}, {len(ok)} acceptable")
            if not ok and all(v == "unknown" for v in verdicts):
                ok = self.model_pairs(a, b, ca, cb)
            elif not ok:
                cur = table.get((r.articles[b]["pick"], r.articles[a]["pick"]))
                plain = [(x, m) for m in cb for x in ca if table[(m["id"], x["id"])][0] == "unknown"]
                if cur is not None and cur[0] == "no" and plain:
                    ok = plain
            if not ok:
                continue
            later = [(a2, [it for it in cmap.get(a2, []) if not r.contradicted(it, a2)][:6] or cmap.get(a2, [])[:6])
                     for a2, b2 in pairs[k + 1:] if b2 == b and a2 != a and a2 not in fixed]
            if later and b not in fixed:
                cover = {m["id"]: self.shared_cover(m, later) for m in {pr[1]["id"]: pr[1] for pr in ok}.values()}
                top = max(cover.values())
                ok = [pr for pr in ok if cover[pr[1]["id"]] == top]
            best = self.choose_pair(ok, a, b)
            r.articles[a]["pick"], r.articles[b]["pick"] = best[0]["id"], best[1]["id"]
            fixed |= {a, b}


class SegMin:

    def __init__(self, keys: list[float]) -> None:
        n = 1
        while n < len(keys):
            n *= 2
        self.n = n
        self.t: list[tuple[float, int]] = [(math.inf, -1)] * (2 * n)
        for i, k in enumerate(keys):
            self.t[n + i] = (k, i)
        for i in range(n - 1, 0, -1):
            self.t[i] = min(self.t[2 * i], self.t[2 * i + 1])

    def kill(self, i: int) -> None:
        i += self.n
        self.t[i] = (math.inf, -1)
        i //= 2
        while i:
            self.t[i] = min(self.t[2 * i], self.t[2 * i + 1])
            i //= 2

    def least(self, lo: int, hi: int) -> tuple[float, int]:
        best = (math.inf, -1)
        lo += self.n
        hi += self.n
        while lo < hi:
            if lo & 1:
                best = min(best, self.t[lo])
                lo += 1
            if hi & 1:
                hi -= 1
                best = min(best, self.t[hi])
            lo //= 2
            hi //= 2
        return best


class Basket:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.log: list[str] = []
        self.stats = collections.Counter()
        self.broke = False

    def state(self, it: dict, ai: int) -> str:
        r = self.r
        if not r.ok_basic(it, ai) or it.get("decoy"):
            return "x"
        if not it.get("read"):
            return "x" if any(r.cat.verdict(it, q) == "no" for q in r.reqs(ai)) or r.finder.lacks(it, ai) else "u"
        if r.contradicted(it, ai):
            return "x"
        lv = r.judge.line_level(it, ai)
        if lv == "exact":
            return "v"
        if lv in ("no", "close"):
            return "x"
        y, n, u = r.fit(it, ai)
        if n > 0:
            return "x"
        if it.get("read") and y > 0 and u == 0:
            return "v"
        return "u"

    def chance(self, it: dict, ai: int) -> float:
        y, n, u = self.r.fit(it, ai)
        return (y + 1) / (y + n + u + 2)

    def table(self, ai: int) -> list[dict]:
        r = self.r
        art = r.articles[ai]
        pool = {it["id"]: it for it in r.candidates(ai, read_only=False) if not it.get("gone")}
        for pid in art.get("ids") or []:
            it = r.cat.items.get(pid)
            if it is not None and pid not in pool and not it.get("gone") and not it.get("read") and r.ok_basic(it, ai):
                pool[pid] = it
        out = list(pool.values())
        out.sort(key=lambda it: (r.judge.price_of(it), it["id"]))
        return out

    def strict_state(self, it: dict, ai: int) -> str:
        st = self.state(it, ai)
        if st == "v" and self.r.judge.line_level(it, ai) != "exact":
            return "u"
        return st

    def verify_loop(self, cap: int = 0, note: bool = True, keep: int = 0, only: list[int] | None = None, steps: int | None = None,
                    dip: int = 0) -> int:
        r = self.r
        cap = cap or VERIFY_CAP
        if not r.articles or r.t() > GUARD_AT - 50:
            return 0
        paired = {x for p in r.together for x in p}
        lines = [a["i"] for a in r.articles if a["i"] not in paired] if only is None else list(only)
        look = self.state if only is None else self.strict_state
        price = r.judge.price_of
        qty = {a["i"]: max(1, int(a.get("qty") or 1)) for a in r.articles}
        cands = {ai: self.table(ai) for ai in lines}
        states = {ai: [look(it, ai) for it in cands[ai]] for ai in lines}
        trees = {}
        for ai in lines:
            tree = SegMin([price(it) * qty[ai] for it in cands[ai]])
            for j, st in enumerate(states[ai]):
                if st != "u":
                    tree.kill(j)
            trees[ai] = tree
        opened = judged = 0
        census = {ai: "".join(f"{k}{states[ai].count(k)}" for k in "vux") for ai in lines}
        for _ in range(cap):
            room = r.steps_left() - r.plan.finish_need(minimal=True) - max(0, r.plan.repair_reserve() - dip)
            if r.t() > GUARD_AT - 45 or room < 2 + keep or (steps is not None and opened >= steps):
                break
            pick = None
            for ai in lines:
                ver = [j for j, st in enumerate(states[ai]) if st == "v"]
                if ver and not r.cheapest:
                    continue
                top = ver[0] if ver else len(cands[ai])
                if only is not None and not ver:
                    js = [k for k, st in enumerate(states[ai]) if st == "u"]
                    j = min(js, key=lambda k: (-self.chance(cands[ai][k], ai), price(cands[ai][k]), cands[ai][k]["id"])) if js else -1
                    cost = price(cands[ai][j]) * qty[ai] if js else 0.0
                else:
                    cost, j = trees[ai].least(0, top)
                if j < 0:
                    continue
                gap = (price(cands[ai][ver[0]]) * qty[ai] if ver else (r.budget or cost * 2 + 1)) - cost
                gain = self.chance(cands[ai][j], ai) * max(gap, 0.0) * (1 if ver else VERIFY_EMPTY)
                key = (gain, -cost, -ai)
                if gain > 0 and (pick is None or key > pick[0]):
                    pick = (key, ai, j)
            if pick is None:
                break
            _, ai, j = pick
            it = cands[ai][j]
            trees[ai].kill(j)
            if not it.get("read"):
                before = r.steps
                r.hunt.open([it["id"]])
                opened += r.steps > before
                if not it.get("read"):
                    states[ai][j] = "x"
                    continue
            if r.judge.line_level(it, ai) is None and r.judge.can_judge(1):
                r.judge.judge_batch([(it, ai)])
                judged += 1
            states[ai][j] = look(it, ai)
            if states[ai][j] == "u":
                states[ai][j] = "x"
        moved = []
        for ai in lines if only is None else []:
            ver = [cands[ai][j] for j, st in enumerate(states[ai]) if st == "v"]
            art = r.articles[ai]
            cur = r.cat.items.get(art.get("pick") or "")
            if not ver:
                continue
            if cur is not None and r.judge.line_level(cur, ai) == "close":
                soft = [it["id"] for it in ver if r.judge.equiv_only(it, ai)]
                ver = [it for it in ver if it["id"] not in soft]
                if soft:
                    r.notes.append(f"verify loop: line {ai} kept close pick {cur['id']} over {soft} matched only by the wording check")
                if not ver:
                    continue
            best = min(ver, key=lambda it: (price(it), -r.judge.own_score(it, ai), it["id"]))
            cur_ok = cur is not None and self.state(cur, ai) == "v"
            if cur is best or (cur_ok and not r.cheapest) or (cur_ok and price(cur) <= price(best)):
                continue
            art["backup"] = [y for y in ([cur["id"]] if cur else []) + list(art.get("backup") or []) if y != best["id"]]
            art["pick"] = best["id"]
            moved.append(ai)
        self.stats.update({"opened": opened, "judged": judged, "moved": len(moved), "calls": 1})
        if note:
            r.notes.append(f"verify loop: states {census}, opened {opened}, judged {judged}, picks moved on lines {moved}; totals {dict(self.stats)}")
        return opened + judged

    def unpaired_weak(self) -> list[int]:
        paired = {x for p in self.r.together for x in p}
        return [ai for ai in self.weak_lines() if ai not in paired]

    def weak_lines(self) -> list[int]:
        r = self.r
        out = []
        for a in r.articles:
            ai = a["i"]
            if not r.judge.judgeable(ai):
                continue
            it = r.cat.items.get(a.get("pick") or "")
            if it is None or not r.ok_basic(it, ai) or r.judge.line_level(it, ai) != "exact":
                out.append(ai)
        return out

    def exact_ids(self, ai: int) -> set[str]:
        r = self.r
        return {it["id"] for it in r.candidates(ai) if r.judge.line_level(it, ai) == "exact"}

    def weak_pass(self, steps: int, tag: str) -> bool:
        r = self.r
        judge = r.judge
        weak = self.weak_lines()
        if not weak or r.t() > GUARD_AT - 60:
            return False
        before = {ai: self.exact_ids(ai) for ai in weak}
        n0, a0 = r.steps, judge.asked
        judge.weak = True
        try:
            self.judge_all(weak)
            if steps > 0:
                bare = any(r.articles[ai].get("pick") in r.cat.items and (judge.judged_no(r.cat.items[r.articles[ai]["pick"]], ai)
                                                                          or judge.line_level(r.cat.items[r.articles[ai]["pick"]], ai) == "close") for ai in weak)
                self.verify_loop(VERIFY_CAP, note=False, only=weak, steps=steps, dip=WEAK_DIP if bare else 0)
        finally:
            judge.weak = False
        fresh = [ai for ai in weak if self.exact_ids(ai) - before[ai]]
        r.notes.append(f"weak pass {tag}: lines {weak}, judged {judge.asked - a0}, opened {r.steps - n0}, new exact on {fresh}")
        return bool(fresh)

    def judge_all(self, lines: list[int]) -> None:
        r = self.r
        judge = r.judge
        queues = []
        for ai in lines:
            reqs = judge.judgeable(ai)
            key = judge.line_key(reqs)
            pool = [it for it in r.candidates(ai) if judge.open_for(it, reqs, key)]
            pool.sort(key=lambda it: (r.fit(it, ai)[1] > 0, r.fit(it, ai)[2] > 0, judge.price_of(it), it["id"]))
            queues.append([(it, ai) for it in pool])
        pairs = [q[k] for k in range(max((len(q) for q in queues), default=0)) for q in queues if k < len(q)]
        while pairs and r.t() < GUARD_AT - 50:
            chunk = pairs[:2 * JUDGE_BATCH]
            if not judge.can_judge(len(chunk)):
                break
            judge.judge_batch(chunk)
            pairs = pairs[len(chunk):]

    def fit_ok(self, hub: dict, acc: dict, pages: list) -> bool:
        cat = self.r.cat
        return self.r.judge.tier(*fit_judge(cat.fit_record(hub), cat.fit_record(acc), pages)) < 9

    def group_options(self, hub: int, accs: list[int], picks: dict) -> list[tuple[float, int, tuple[tuple[int, str], ...]]]:
        r = self.r
        cat = r.cat
        price = r.judge.price_of
        qty = {a["i"]: max(1, int(a.get("qty") or 1)) for a in r.articles}
        lines = [hub] + accs
        depts = {cat.category(picks[a]) for a in accs}
        pages = [cat.fit_record(it) for it in cat.items.values() if (it.get("read") or it.get("paged")) and cat.category(it) in depts]
        pools = {}
        for ai in lines:
            alts = [it for it in self.table(ai) if it is not picks[ai] and it.get("read") and self.state(it, ai) == "v"]
            pools[ai] = [picks[ai]] + alts[:FIT_ALTS]
        out = []
        for h in pools[hub]:
            per = [[x for x in pools[a] if self.fit_ok(h, x, pages)] for a in accs]
            if any(not p for p in per):
                continue
            for combo in itertools.product(*per):
                chosen = [h] + list(combo)
                cost = sum(price(it) * qty[ai] for ai, it in zip(lines, chosen))
                ch = sum(1 for ai, it in zip(lines, chosen) if it is not picks[ai])
                out.append((cost, ch, tuple((ai, it["id"]) for ai, it in zip(lines, chosen))))
        cur = (sum(price(picks[ai]) * qty[ai] for ai in lines), 0, tuple((ai, picks[ai]["id"]) for ai in lines))
        if cur not in out:
            out.append(cur)
        return out

    def knapsack(self) -> None:
        r = self.r
        cat = r.cat
        self.broke = False
        if not r.budget or not r.articles:
            return
        price = r.judge.price_of
        qty = {a["i"]: max(1, int(a.get("qty") or 1)) for a in r.articles}
        picks = {a["i"]: cat.items.get(a.get("pick") or "") for a in r.articles}
        if any(v is None for v in picks.values()):
            return
        if sum(price(it) * qty[ai] for ai, it in picks.items()) <= r.budget + 1e-9:
            return
        paired = {x for p in r.together for x in p}
        hubs: dict[int, list[int]] = {}
        for pr in r.together:
            if len(pr) == 2 and all(isinstance(x, int) and x in picks for x in pr) and pr[0] != pr[1]:
                hubs.setdefault(pr[1], []).append(pr[0])
        seen: collections.Counter = collections.Counter(x for b, accs in hubs.items() for x in {b, *accs})
        groups = {b: list(dict.fromkeys(accs)) for b, accs in hubs.items() if all(seen[x] == 1 for x in {b, *accs})}
        grouped = {x for b, accs in groups.items() for x in [b] + accs}
        fixed = sum(price(picks[ai]) * qty[ai] for ai in picks if ai in paired and ai not in grouped)
        cap = r.budget - fixed
        units: list[list[tuple[float, int, tuple[tuple[int, str], ...]]]] = []
        for ai in sorted(picks):
            if ai in paired:
                continue
            cur = picks[ai]
            opts = [(price(cur) * qty[ai], 0, ((ai, cur["id"]),))]
            for it in self.table(ai):
                if it is not cur and it.get("read") and self.state(it, ai) == "v":
                    opts.append((price(it) * qty[ai], 1, ((ai, it["id"]),)))
            units.append(opts)
        for b in sorted(groups):
            if r.t() > GUARD_AT - 30:
                units.append([(sum(price(picks[x]) * qty[x] for x in [b] + groups[b]), 0, tuple((x, picks[x]["id"]) for x in [b] + groups[b]))])
                continue
            units.append(self.group_options(b, groups[b], picks))
        frontier: list[tuple[float, int, tuple[tuple[int, str], ...]]] = [(0.0, 0, ())]
        for opts in units:
            merged = [(c + oc, ch + och, ids + oids) for c, ch, ids in frontier for oc, och, oids in opts if c + oc <= cap + 1e-9]
            merged.sort(key=lambda x: (x[0], x[1], x[2]))
            frontier = []
            low = None
            for cost, ch, ids in merged:
                if low is None or ch < low:
                    frontier.append((cost, ch, ids))
                    low = ch
            if not frontier:
                break
        if not frontier or len(frontier[0][2]) != len(picks) - len(paired - grouped):
            r.notes.append("knapsack: no basket within budget among verified listings")
            self.broke = True
            return
        cost, ch, ids = min(frontier, key=lambda x: (x[1], x[0], x[2]))
        room = max(0, r.steps_left() - r.plan.finish_need(minimal=True) - 1)
        if ch * r.order.swap_cost(1) > room:
            r.notes.append(f"knapsack: budget fix needs {ch} swaps, no room")
            return
        for ai, pid in ids:
            if pid != picks[ai]["id"]:
                art = r.articles[ai]
                r.notes.append(f"knapsack: line {ai} {picks[ai]['id']} -> {pid} to meet the budget")
                art["backup"] = [y for y in [picks[ai]["id"]] + list(art.get("backup") or []) if y != pid]
                art["pick"] = pid

    def budget_retry(self) -> None:
        r = self.r
        if not self.broke or r.t() > GUARD_AT - 50:
            return
        keep = r.cheapest
        r.cheapest = True
        try:
            got = self.verify_loop(note=False)
        finally:
            r.cheapest = keep
        self.knapsack()
        r.notes.append(f"budget retry: {got} reads or judgements, basket {'still over the budget' if self.broke else 'fits the budget'}")

    def over_budget(self) -> bool:
        r = self.r
        if not self.broke or not r.budget or not r.articles or Text.of((r.budget_raw or {}).get("per")).strip().lower() == "unit":
            return False
        total = 0.0
        for art in r.articles:
            it = r.cat.items.get(art.get("pick") or "")
            p = None if it is None else r.cat.price(it)
            if p is None:
                return False
            total += p * max(1, int(art.get("qty") or 1))
        return total > r.budget + 1e-9


class Reach:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.stats: collections.Counter = collections.Counter()

    @staticmethod
    def one_of(vals: list[str]) -> tuple[str | None, list[str]]:
        if not vals:
            return None, []
        one = min(vals, key=lambda v: (len(Text(v).words), vals.index(v)))
        return one, [v for v in vals if v != one]

    def lacking(self, ai: int) -> bool:
        r = self.r
        art = r.articles[ai]
        if art.get("fit_verified"):
            return False
        if not any(r.basket.state(it, ai) == "v" for it in r.candidates(ai)):
            return True
        return bool(r.cheapest) and not r.hunt.sas_certified(ai)

    def fit(self, q: Query, ai: int) -> tuple | None:
        r = self.r
        art = r.articles[ai]
        text = r.finder.class_text(q)
        have = set(Text(text).stems)
        heads = {Text.stem(w) for w in noun_heads(art["noun"])}
        named = bool(have & heads)
        carried = r.finder.carries(text, r.finder.value_groups(ai))
        if not named and not carried:
            return None
        cat = art.get("cat")
        reached = int(not cat or any(r.cat.category(r.cat.items[i]) == cat for i in q.ids if i in r.cat.items))
        return (carried + named, reached, -(q.total if isinstance(q.total, int) else 10 ** 9))

    def same_set(self, q: Query) -> bool:
        f = self.r.finder
        mine = set(Text(q.text).stems)
        for c in f.queries.values():
            if c is q or not f.complete(c) or c.total != q.total:
                continue
            other = set(Text(c.text).stems)
            if mine <= other or other <= mine:
                return True
        return False

    def best_open(self, ai: int) -> Query | None:
        r = self.r
        pg = r.store.page_arg("search")
        if not pg:
            return None
        best, key = None, None
        needs_carry = r.hunt.capped_carrier(ai)
        for q in r.finder.open_queries():
            if r.finder.complete(q):
                continue
            nxt = r.store.next_cursor("search", q)
            if nxt is None or nxt in q.used or self.same_set(q):
                continue
            k = self.fit(q, ai)
            if k is not None and needs_carry and not r.finder.carries(r.finder.class_text(q), r.finder.value_groups(ai)):
                continue
            if k is not None and (key is None or k > key):
                best, key = q, k
        return best

    def page(self) -> tuple[int, list[str]] | None:
        r = self.r
        best, key = None, None
        for art in r.articles:
            ai = art["i"]
            if not self.lacking(ai):
                continue
            q = self.best_open(ai)
            if q is None:
                continue
            verified = any(r.basket.state(it, ai) == "v" for it in r.candidates(ai))
            k = (art.get("reach_pages", 0) + int(verified), ai)
            if key is None or k < key:
                best, key = (art, q), k
        if best is None:
            return None
        art, q = best
        got, used = r.hunt.alt_firsts(art, 1)
        if used:
            self.stats["pages"] += 1
            r.hunt.add_ids(art, got)
            return art["i"], got
        ids = r.finder.page_on(q.text, 1)
        self.stats["pages"] += 1
        art["reach_pages"] = art.get("reach_pages", 0) + 1
        r.hunt.add_ids(art, ids)
        r.notes.append(f"reach: line {art['i']} page {q.pages} of {q.text!r} ({q.total}), {len(ids)} rows")
        return art["i"], ids

    def price_walk(self, ai: int, allowance: int) -> None:
        r = self.r
        art = r.articles[ai]
        start = r.steps
        while r.steps - start < allowance and r.plan.room(r.phase) > 0 and r.t() < GUARD_AT - 40:
            q = self.best_open(ai)
            if q is None:
                return
            bar = min((r.judge.price_of(it) for it in r.candidates(ai) if r.basket.state(it, ai) == "v"), default=None)
            items = r.cat.items
            last = [r.cat.price(items[i]) for i in q.ids[-1:] if i in items and r.cat.price(items[i]) is not None]
            if bar is not None and (not last or last[0] >= bar):
                return
            n0 = r.steps
            got, used = r.hunt.alt_firsts(art, 1)
            if used:
                r.hunt.add_ids(art, got)
                continue
            ids = r.finder.page_on(q.text, 1)
            self.stats["walk"] += 1
            r.hunt.add_ids(art, ids)
            r.notes.append(f"reach: line {ai} price walk page {q.pages} of {q.text!r}, {len(ids)} rows")
            if not ids:
                return
            r.hunt.sorted_open(ai, max(1, allowance - (r.steps - start)))
            if r.steps == n0:
                return


class Cheap:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.tried: set[str] = set()
        self.memo: dict[tuple, str] = {}
        self.stats: collections.Counter = collections.Counter()

    def note_row(self, it: dict, row: dict, facts: dict) -> None:
        flat = {k: (v[:OPEN_TEXT] if isinstance(v, str) else v) for k, v in row.items()
                if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        it["srow"] = {**(it.get("srow") or {}), **flat}
        if facts:
            it["sfacts"] = {**(it.get("sfacts") or {}), **facts}

    def lines(self) -> list[int]:
        r = self.r
        paired = {x for p in r.together for x in p}
        return [a["i"] for a in r.articles if a["i"] not in paired and not a.get("fit_verified") and not r.hunt.sas_certified(a["i"])]

    def bar(self, ai: int) -> float | None:
        r = self.r
        price = r.judge.price_of
        ver = [price(it) for it in r.candidates(ai) if r.basket.state(it, ai) == "v"]
        if ver and min(ver) < 1e18:
            return min(ver)
        cur = r.cat.items.get(r.articles[ai].get("pick") or "")
        if cur is not None and r.ok_basic(cur, ai) and not r.contradicted(cur, ai) and r.cat.price(cur) is not None:
            return r.cat.price(cur)
        return None

    def pool(self, ai: int, bar: float) -> list[dict]:
        r = self.r
        out = []
        for it in r.basket.table(ai):
            if it.get("read") or it["id"] in self.tried or it.get("gone") or it.get("decoy"):
                continue
            p = r.cat.price(it)
            if p is None or p >= bar or r.basket.state(it, ai) != "u":
                continue
            out.append(it)
        return out[:CHEAP_POOL]

    def seen_row(self, it: dict) -> dict:
        return {"id": it["id"], "row": it.get("srow") or {}, "rec": {}, "facts": dict(it.get("sfacts") or {}), "read": False, "queries": []}

    def bucket(self, it: dict, req: dict) -> str:
        if not it.get("srow"):
            return ""
        key = (it["id"], Judge.sig(req), len(it["srow"]), len(it.get("sfacts") or {}))
        if key not in self.memo:
            self.memo[key] = self.row_bucket(it, req)
        return self.memo[key]

    def row_bucket(self, it: dict, req: dict) -> str:
        v = self.r.cat.read_verdict(self.seen_row(it), req)
        if v in ("yes", "no"):
            return v
        if req.get("op") not in ("is", "eq"):
            return "u"
        hay = set(Text(" ".join(Text.of(x) for x in it["srow"].values())).stems)
        share = 0.0
        for w in Catalog.wanted_of(req):
            st = Catalog.content_stems(w)
            if st:
                share = max(share, sum(1 for s in st if s in hay) / len(st))
        return "u-all" if share >= 1 else ("u-some" if share > 0 else "u-none")

    def rates(self, ai: int, reqs: list[dict]) -> dict:
        r = self.r
        owners = [it for it in r.owners(ai) if it.get("srow")]
        out: dict = {}
        for q in reqs:
            tally: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
            for it in owners:
                truth = r.cat.verdict(it, q)
                if truth in ("yes", "no"):
                    tally[self.bucket(it, q)][0 if truth == "yes" else 1] += 1
            out[Judge.sig(q)] = {b: (y + 1) / (y + n + 2) for b, (y, n) in tally.items()}
        return out

    def chance(self, it: dict, reqs: list[dict], rates: dict) -> float:
        c = 1.0
        for q in reqs:
            c *= rates.get(Judge.sig(q), {}).get(self.bucket(it, q), 0.5)
        return c

    @staticmethod
    def curve(bar: float, rows: list[tuple[float, float]], m: int) -> tuple[list[float], list[list[int]]]:
        f: list[float] = [bar] + [math.inf] * m
        took: list[list[bool]] = []
        for j, (p, c) in enumerate(rows):
            mark = [False] * (m + 1)
            for k in range(min(m, j + 1), 0, -1):
                if f[k - 1] == math.inf:
                    continue
                v = c * p + (1 - c) * f[k - 1]
                if v < f[k] - 1e-12:
                    f[k] = v
                    mark[k] = True
            took.append(mark)

        def chosen(k: int) -> list[int]:
            out = []
            for j in range(len(rows) - 1, -1, -1):
                if k > 0 and took[j][k]:
                    out.append(j)
                    k -= 1
            return out
        best, size = [bar], [0]
        for k in range(1, m + 1):
            if f[k] < best[-1]:
                best.append(f[k])
                size.append(k)
            else:
                best.append(best[-1])
                size.append(size[-1])
        memo: dict[int, list[int]] = {}
        for z in size:
            if z not in memo:
                memo[z] = chosen(z)
        return best, [memo[z] for z in size]

    @staticmethod
    def allot(gains: list[list[float]], total: int) -> list[int]:
        h = [0.0] * (total + 1)
        picks: list[list[int]] = []
        for g in gains:
            nh = [-1.0] * (total + 1)
            ch = [0] * (total + 1)
            for b in range(total + 1):
                for m in range(min(b, len(g) - 1) + 1):
                    v = h[b - m] + g[m]
                    if v > nh[b] + 1e-12:
                        nh[b], ch[b] = v, m
            h = nh
            picks.append(ch)
        out: list[int] = []
        b = total
        for ch in reversed(picks):
            out.append(ch[b])
            b -= ch[b]
        return out[::-1]

    def next_reads(self, steps: int, per: int) -> list[tuple[int, str]]:
        r = self.r
        total = max(0, steps) * max(1, per)
        if total <= 0:
            return []
        price = r.judge.price_of
        work = []
        for ai in self.lines():
            bar = self.bar(ai)
            if bar is None:
                continue
            pool = self.pool(ai, bar)
            if not pool:
                continue
            reqs = r.reqs(ai)
            rates = self.rates(ai, reqs)
            qty = max(1, int(r.articles[ai].get("qty") or 1))
            rows = sorted(((price(it), self.chance(it, reqs, rates), it) for it in pool), key=lambda x: (-x[0], x[2]["id"]))
            best, sets = self.curve(bar, [(p, c) for p, c, _ in rows], min(total, len(rows)))
            work.append((ai, qty, bar, rows, best, sets))
        if not work:
            return []
        total = min(total, sum(len(w[3]) for w in work))
        alloc = self.allot([[(w[2] - b) * w[1] for b in w[4]] for w in work], total)
        ranked = []
        for (ai, qty, bar, rows, best, sets), m in zip(work, alloc):
            for j in sets[m]:
                p, c, it = rows[j]
                ranked.append((c * (bar - p) * qty, -p, ai, it["id"]))
        ranked.sort(reverse=True)
        return [(ai, pid) for g, _, ai, pid in ranked if g > 0][:max(1, per)]

    def repick(self) -> list[int]:
        r = self.r
        price = r.judge.price_of
        moved = []
        for ai in self.lines():
            ver = [it for it in r.candidates(ai) if r.basket.state(it, ai) == "v"]
            if not ver:
                continue
            best = min(ver, key=lambda it: (price(it), -r.judge.own_score(it, ai), it["id"]))
            art = r.articles[ai]
            cur = r.cat.items.get(art.get("pick") or "")
            if cur is best:
                continue
            ok = cur is not None and r.ok_basic(cur, ai) and not r.contradicted(cur, ai)
            if ok and (price(cur) < price(best) or (price(cur) == price(best) and r.basket.state(cur, ai) == "v")):
                continue
            art["backup"] = [y for y in ([cur["id"]] if cur is not None else []) + list(art.get("backup") or []) if y != best["id"]]
            art["pick"] = best["id"]
            r.notes.append(f"cheap: line {ai} {cur['id'] if cur is not None else None} -> {best['id']} at {price(best):g}")
            moved.append(ai)
        return moved

    def sweep(self, budget: int, tag: str, own: bool = False) -> bool:
        r = self.r
        if not r.cheapest or budget <= 0 or not r.articles or r.finished:
            return False
        cmp_ = r.tm.get("compare") or {}
        batch = not (r.tm.get("specs") or {}).get("id") and cmp_.get("ids") and r.store.yields_facts(cmp_["name"])
        per = r.store.compare_max() if batch else 1
        used = 0
        got: list[str] = []
        while used < budget and r.t() < GUARD_AT - 40 and not r.finished and not r.env_dead:
            want = self.next_reads(budget - used, per)
            if not want:
                break
            ids = [pid for _, pid in want]
            self.tried.update(ids)
            n0 = r.steps
            was = r.phase
            r.phase = "sink"
            r.hunt.open(ids, force=True, room=budget - used if own else None)
            r.phase = was
            if r.steps == n0:
                break
            used += r.steps - n0
            fresh = [(ai, pid) for ai, pid in want if r.cat.items.get(pid, {}).get("read")]
            got += [pid for _, pid in fresh]
            lines = sorted({ai for ai, _ in fresh})
            if lines and r.judge.can_judge():
                r.judge.judge_lines(lines)
        moved = self.repick()
        self.stats.update({"steps": used, "read": len(got), "moved": len(moved), "runs": 1})
        r.notes.append(f"cheap {tag}: budget {budget}, steps {used}, read {len(got)}, moved {moved}")
        return bool(moved)

    def before_sink(self) -> None:
        r = self.r
        if not r.cheapest or r.t() > GUARD_AT - 50:
            return
        refs = sum(1 for x in r.ledger if x["active"] and x["op"] == "ref" and not x.get("value_ok"))
        budget = r.plan.slack() - r.order.swap_cost(1) - len(r.basket.weak_lines()) - 2 * refs
        if budget > 0:
            self.sweep(budget, "before sink")

    def widen(self, steps: int) -> None:
        r = self.r
        n0 = r.steps
        f = r.tm.get("filter") or {}
        for ai in self.lines():
            if r.steps - n0 >= steps or r.t() > GUARD_AT - 40 or r.finished:
                break
            art = r.articles[ai]
            bar = self.bar(ai)
            if f.get("max_price") and f.get("category") and art.get("cat") and bar is not None and bar < 1e18:
                ceiling = bar - 1 if float(bar).is_integer() else round(bar - 0.01, 2)
                obs = r.env.send([(f["name"], {**(f.get("args") or {}), f["category"]: art["cat"], f["max_price"]: ceiling})])
                ids, _, _ = r.cat.rows_of(obs[0] if obs else None)
                r.hunt.add_ids(art, ids)
                continue
            q = next((p for p in r.search_phrases(art) if p.strip() and r.finder.state(p) is None), None)
            if q:
                ids, _, _ = r.hunt.line_run(ai, q, force=True)
                r.hunt.add_ids(art, ids)
        if r.steps > n0:
            r.notes.append(f"cheap: widened after a drop with {r.steps - n0} steps")

    def after_loss(self) -> bool:
        r = self.r
        f = r.tm.get("filter") or {}
        if not r.cheapest or r.finished or not (f.get("max_price") and f.get("category")) or r.t() > GUARD_AT - 40:
            return False
        go = False
        for ai in self.lines():
            art = r.articles[ai]
            cur = r.cat.items.get(art.get("pick") or "")
            p0 = r.cat.price(cur) if cur is not None else None
            bar = self.bar(ai)
            if cur is None or not r.order.dead(cur) or p0 is None or not art.get("cat") or (bar is not None and bar <= p0):
                continue
            room = r.steps_left() - r.plan.finish_need(minimal=True)
            if room < 1 or r.finished:
                break
            go = True
            if room < 2:
                continue
            obs = r.env.send([(f["name"], {**(f.get("args") or {}), f["category"]: art["cat"], f["max_price"]: r.capsweep.cap_arg(f, p0)})])
            ids, _, _ = r.cat.rows_of(obs[0] if obs else None)
            r.hunt.add_ids(art, ids)
            r.notes.append(f"cheap: line {ai} lost its pick at {p0:g}, {len(ids)} rows at or under that price")
        if not go:
            return False
        spare = r.steps_left() - r.plan.finish_need(minimal=True)
        budget = max(min(spare, 1), r.plan.slack())
        return self.sweep(budget, "after loss", own=True) if budget > 0 else False

    def after_event(self, drop: bool) -> bool:
        r = self.r
        if not r.cheapest or r.finished or r.t() > GUARD_AT - 40:
            return False
        keep = r.plan.finish_need(minimal=True) + r.order.swap_cost(1)
        if drop and r.steps_left() - keep >= 3:
            self.widen(r.steps_left() - keep - 2)
        budget = min(r.steps_left() - keep, r.plan.slack())
        return self.sweep(budget, "after event") if budget > 0 else False


class CapSweep:

    def __init__(self, run: Any) -> None:
        self.r = run
        self.proved: dict[int, float] = {}
        self.tried: set[str] = set()
        self.known: list[tuple[float, int | None]] = []
        self.counts: dict[tuple[str, Any], int] = {}
        self.broken = False
        self.stats: collections.Counter = collections.Counter()

    def note(self, name: str, args: Any, obs: Any) -> None:
        f = self.tool()
        if f is None or name != f["name"] or not isinstance(args, dict) or not args.get(f["category"]):
            return
        cap = args.get(f["max_price"])
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or any(args.get(k) != v for k, v in (f.get("args") or {}).items()):
            return
        if f.get("page") and args.get(f["page"]) not in (None, "", 0):
            return
        _, total, _ = self.r.cat.rows_of(obs)
        if isinstance(total, int):
            self.counts[(Text.of(args[f["category"]]).lower(), cap)] = total

    def tool(self) -> dict | None:
        r = self.r
        f = r.tm.get("filter") or {}
        if not (r.cheapest and f.get("name") and f.get("category") and f.get("max_price")):
            return None
        return f

    def out(self) -> bool:
        r = self.r
        return bool(self.broken or r.finished or r.env_dead or r.t() > GUARD_AT - 45)

    def bar(self, ai: int) -> float | None:
        r = self.r
        cur = r.cat.items.get(r.articles[ai].get("pick") or "")
        if cur is None or r.basket.state(cur, ai) != "v":
            return None
        p = r.cat.price(cur)
        return p if p is not None and 0 < p < 1e18 else None

    def cap_arg(self, f: dict, v: float) -> Any:
        spec = ((self.r.schemas.get(f["name"]) or {}).get("properties") or {}).get(f["max_price"]) or {}
        if spec.get("type") == "integer" or float(v).is_integer():
            return int(math.ceil(v))
        return round(v, 2)

    def fetch(self, art: dict, f: dict, cap: Any, page: Any = None) -> tuple[list[str], int | None, Any, bool]:
        r = self.r
        args = {**(f.get("args") or {}), f["category"]: art["cat"], f["max_price"]: cap}
        if page is not None and f.get("page"):
            args[f["page"]] = page
        n0 = r.steps
        obs = r.env.send([(f["name"], args)])
        ids, total, cursor = r.cat.rows_of(obs[0] if obs else None)
        r.hunt.add_ids(art, ids)
        self.stats["pages"] += 1
        dear = [p for p in (r.cat.price(r.cat.items[i]) for i in ids if i in r.cat.items) if p is not None and p > cap + 0.01]
        if dear:
            self.broken = True
            r.notes.append(f"cap sweep: rows above the cap {cap} ({len(dear)}), the cap is not a price filter here; sweep stopped")
        return ids, total, cursor, r.steps > n0 and not dear

    def next_page(self, f: dict, q: dict) -> Any:
        r = self.r
        pg = f.get("page")
        if not pg or not q["last"] or (q["total"] is not None and q["seen"] >= q["total"]):
            return None
        if q["cursor"] not in (None, "", False):
            return q["cursor"]
        spec = ((r.schemas.get(f["name"]) or {}).get("properties") or {}).get(pg) or {}
        if r.store.cursor or spec.get("type") != "integer" or not isinstance(spec.get("minimum"), int):
            return None
        nxt = q["seen"] if spec["minimum"] == 0 else q["pages"] + 1
        hi = spec.get("maximum")
        return None if isinstance(hi, (int, float)) and nxt > hi else nxt

    def open_query(self, art: dict, f: dict, cap: Any) -> dict:
        ids, total, cursor, sent = self.fetch(art, f, cap)
        q = {"cap": cap, "total": total if isinstance(total, int) else None, "cursor": cursor, "pages": int(sent), "seen": len(ids),
             "last": ids, "rpp": len(ids), "sent": sent}
        if sent:
            self.known.append((cap, q["total"] if q["total"] is not None else (len(ids) if self.next_page(f, q) is None else None)))
        return q

    def walk(self, art: dict, f: dict, q: dict, steps: int) -> bool:
        r = self.r
        n0 = r.steps
        while r.steps - n0 < steps and not self.out():
            nxt = self.next_page(f, q)
            if nxt is None:
                break
            ids, total, cursor, sent = self.fetch(art, f, q["cap"], nxt)
            if not sent:
                break
            q["pages"] += 1
            q["seen"] += len(ids)
            q["last"], q["cursor"] = ids, cursor
            if isinstance(total, int):
                q["total"] = total
        return bool(q["sent"]) and self.next_page(f, q) is None and (q["total"] is None or q["seen"] >= q["total"])

    def guess(self, art: dict, bar: float, target: int) -> float | None:
        r = self.r
        prices = sorted(p for p in (r.cat.price(it) for it in r.cat.items.values() if r.cat.category(it) == art["cat"])
                        if p is not None and p < bar)
        pts = [(c, n) for c, n in self.known if isinstance(n, int)]
        lows = [p for p in pts if p[1] < target]
        highs = [p for p in pts if p[1] >= target]
        if not highs:
            return None
        lo = max(lows) if lows else None
        hi = min(highs)
        n_lo = lo[1] if lo else 0
        floor = lo[0] if lo else 0
        if hi[1] <= n_lo or hi[0] <= floor:
            return None
        mid = [p for p in prices if p > floor and p <= hi[0]]
        c = floor + (hi[0] - floor) * (target - n_lo) / (hi[1] - n_lo)
        if len(mid) >= 2:
            k = math.ceil((target - n_lo) / (hi[1] - n_lo) * len(mid)) - 1
            c = min(c, mid[max(0, min(len(mid) - 1, k))])
        return c if floor < c < hi[0] else None

    def pair_tier(self, it: dict, ai: int, memo: dict) -> int:
        r = self.r
        cat = r.cat
        jd = r.judge
        worst = 0
        for b, accs in jd.fit_hubs().items():
            if ai == b:
                pairs = [(it, cat.items.get(r.articles[a].get("pick") or "")) for a in accs if a != b]
            elif ai in accs:
                pairs = [(cat.items.get(r.articles[b].get("pick") or ""), it)]
            else:
                continue
            for h, x in pairs:
                if h is None or x is None:
                    continue
                if "index" not in memo:
                    memo["index"] = jd.title_index()
                dep = FitInfo(cat.fit_record(x), "dep")
                pages = [FitInfo(cat.fit_record(p), None) for c in dep.pointers for p in memo["index"].get(fit_norm(c), []) if p is not x]
                worst = max(worst, jd.tier(*fit_judge(FitInfo(cat.fit_record(h), "main"), dep, pages)))
        return worst

    def plausible(self, it: dict, reqs: list[dict]) -> bool:
        cheap = self.r.cheap
        return all(cheap.bucket(it, q) not in ("no", "u-none") for q in reqs if q.get("op") in ("is", "eq"))

    def verify(self, ai: int, steps: int, ceiling: float) -> None:
        r = self.r
        art = r.articles[ai]
        price = r.judge.price_of
        reqs = r.reqs(ai)
        rates = r.cheap.rates(ai, reqs)
        n0 = r.steps
        while r.steps - n0 < steps and not self.out():
            bar = self.bar(ai)
            cur = r.cat.items.get(art.get("pick") or "")
            if bar is None or cur is None:
                return
            memo: dict = {}
            base = self.pair_tier(cur, ai, memo)
            pool = [it for it in r.basket.table(ai) if not it.get("read") and it["id"] not in self.tried and price(it) < bar
                    and price(it) <= ceiling and r.basket.state(it, ai) == "u" and self.plausible(it, reqs)]
            pool = [it for it in pool if self.pair_tier(it, ai, memo) <= base]
            if not pool:
                return
            best = max(pool, key=lambda it: (r.cheap.chance(it, reqs, rates) * (bar - price(it)), -price(it), it["id"]))
            self.tried.add(best["id"])
            m0 = r.steps
            r.judge.read_one(best)
            if r.steps == m0:
                return
            self.stats["read"] += 1
            if best.get("read") and r.judge.line_level(best, ai) is None and r.judge.can_judge(1):
                r.judge.judge_batch([(best, ai)])
            if r.basket.state(best, ai) == "v" and price(best) < bar and self.pair_tier(best, ai, {}) <= base:
                old = art.get("pick")
                art["backup"] = [y for y in ([old] if old else []) + list(art.get("backup") or []) if y != best["id"]]
                art["pick"] = best["id"]
                self.stats["moved"] += 1
                r.notes.append(f"cap sweep: line {ai} {old} -> {best['id']} at {price(best):g}")
                if r.tm.get("detail") and r.tm.get("specs") and not best.get("paged") and r.plan.slack() >= 1:
                    r.hunt.open([best["id"]], what="state", force=True)

    def climb(self, ai: int, art: dict, f: dict, q: dict, left: Callable[[], int]) -> None:
        r = self.r
        while left() > 0 and not self.out():
            self.verify(ai, max(0, left() - 1), q["cap"])
            now = self.bar(ai)
            tail = [r.cat.price(r.cat.items[i]) for i in q["last"] if i in r.cat.items and r.cat.price(r.cat.items[i]) is not None]
            if now is None:
                return
            if (tail and max(tail) >= now) or (q["sent"] and self.next_page(f, q) is None and (q["total"] is None or q["seen"] >= q["total"])):
                self.proved[ai] = now
                return
            n0 = r.steps
            self.walk(art, f, q, 1)
            if r.steps == n0:
                return

    def line(self, ai: int, allowance: int) -> None:
        r = self.r
        f = self.tool()
        art = r.articles[ai]
        bar = self.bar(ai)
        if f is None or bar is None or not art.get("cat") or self.proved.get(ai) == bar:
            return
        start = r.steps
        shelf = Text.of(art["cat"]).lower()
        self.known = [(c, n) for (sh, c), n in self.counts.items() if sh == shelf]

        def left() -> int:
            return allowance - (r.steps - start)
        q = self.open_query(art, f, self.cap_arg(f, bar))
        if not q["sent"]:
            return
        rpp = max(1, q["rpp"])
        shown = [r.cat.price(r.cat.items[i]) for i in q["last"] if i in r.cat.items]
        rising = len(shown) >= 4 and None not in shown and all(x <= y for x, y in zip(shown, shown[1:])) and len(set(shown)) > 1
        if rising:
            self.climb(ai, art, f, q, left)
            r.notes.append(f"cap sweep: line {ai} {r.steps - start} steps in price order, proved {self.proved.get(ai)}")
            return
        descents = 0
        floor: tuple[Any, int] | None = None
        while left() > 0 and not self.out():
            bar = self.bar(ai)
            if bar is None:
                break
            while descents < 3 and q["total"] is not None and math.ceil(q["total"] / rpp) - q["pages"] > max(1, left() - 2):
                c = self.guess(art, bar, max(rpp * max(1, (left() - 2) // 2), floor[1] + rpp if floor else 0))
                if c is None or self.cap_arg(f, c) >= q["cap"] or (floor and self.cap_arg(f, c) <= floor[0]):
                    break
                q = self.open_query(art, f, self.cap_arg(f, c))
                descents += 1
                if not q["sent"]:
                    return
            done = self.walk(art, f, q, max(0, left() - 2))
            self.verify(ai, max(0, left()), q["cap"])
            now = self.bar(ai)
            if now is None:
                break
            if done and (now <= q["cap"] or q["cap"] >= self.cap_arg(f, now)):
                self.proved[ai] = now
                break
            if not done or left() < 2:
                break
            floor = (q["cap"], q["seen"])
            c = self.guess(art, now, max(2 * q["seen"], q["seen"] + 2 * rpp))
            cap = self.cap_arg(f, c) if c is not None and self.cap_arg(f, c) > q["cap"] else self.cap_arg(f, now)
            if cap <= q["cap"]:
                break
            q = self.open_query(art, f, cap)
            descents = 0
            if not q["sent"]:
                break
        r.notes.append(f"cap sweep: line {ai} {r.steps - start} steps, known {self.known[:6]}, proved {self.proved.get(ai)}")

    def run(self) -> None:
        r = self.r
        if self.tool() is None or r.finished or not r.articles or r.t() > GUARD_AT - 50:
            return
        refs = sum(1 for x in r.ledger if x["active"] and x["op"] == "ref" and not x.get("value_ok"))
        budget = r.plan.slack() - r.order.swap_cost(1) - len(r.basket.weak_lines()) - 2 * refs
        qty = {a["i"]: max(1, int(a.get("qty") or 1)) for a in r.articles}
        lines = [(self.bar(a["i"]), a["i"]) for a in r.articles if a.get("cat") and not r.hunt.sas_certified(a["i"])]
        lines = sorted(((b * qty[ai], ai) for b, ai in lines if b is not None), reverse=True)
        start = r.steps
        for k, (_, ai) in enumerate(lines):
            room = budget - (r.steps - start)
            if room < 3 or self.out():
                break
            self.line(ai, max(3, room - 3 * (len(lines) - k - 1)))
        self.stats["steps"] += r.steps - start


class Run:

    def __init__(self, problem_data: dict) -> None:
        self.began = monotonic()
        env = problem_data.get("environment") or {}
        self.binding = dict(env.get("binding") or {})
        pv = env.get("policy_view") or {}
        self.request = Text.of(pv.get("query"))
        self.opening = pv.get("opening")
        self.tools = [t for t in (pv.get("tools") or []) if isinstance(t, dict)]
        self.schemas: dict[str, dict] = {}
        self.descs: dict[str, str] = {}
        for t in self.tools:
            f = t.get("function", t)
            if isinstance(f, dict) and f.get("name"):
                self.schemas[f["name"]] = f.get("parameters") or {}
                self.descs[f["name"]] = Text.of(f.get("description"))
        try:
            self.max_steps = max(1, int(pv.get("max_steps") or 40))
        except (TypeError, ValueError):
            self.max_steps = 40
        self.step_cap = self.max_steps
        self.finished = bool(pv.get("finished"))
        self.steps = 0
        self.msg_cap: int | None = None
        self.msgs = 0
        self.spent = 0.0
        self.calls = 0
        self.lock = threading.Lock()
        BUCKET[0] = Bucket(0.9, 5.0)
        self.exhausted = False
        self.env_dead = False
        self.sid = str(self.binding.get("session_id") or problem_data.get("problem_id") or id(self))[:64]
        self.dialogue: list[dict] = []
        self.trace: list[str] = []
        self.store = Store(self.schemas, self.descs)
        self.tm = self.store.tm
        self.cat = Catalog(self)
        self.prose = RowReader(self)
        self.env = Env(self)
        self.finder = Finder(self)
        self.hunt = Hunt(self)
        self.asker = RevealAsker(self)
        self.chains = Chains(self)
        self.judge = Judge(self)
        self.basket = Basket(self)
        self.cheap = Cheap(self)
        self.reach = Reach(self)
        self.capsweep = CapSweep(self)
        self.plan = Plan(self)
        self.order = Order(self)
        self.phase = "find"
        self.cart_stamp = -1
        self.spec: dict = {}
        self.articles: list[dict] = []
        self.ledger: list[dict] = []
        self.heard: list[dict] = []
        self.news: list[str] = []
        self.cart: dict[str, int] = {}
        self.cart_seen = False
        self.ordered = False
        self.declined = False
        self.notes: list[str] = []
        self.rank_log: list[dict] = []
        self.news_read = 0
        self.confirm = False
        self.pre_note = ""
        self.together: list[list[int]] = []
        self.approved_cart: dict[str, int] | None = None
        self.dropped_since = False
        self.knocked: dict | None = None
        self.notices: list[dict] = []
        self.clauses: list[dict] = []
        self.dropped_clauses: list[dict] = []
        self.summary: dict = {}
        self.budget: float | None = None
        self.budget_raw: dict = {}
        self.cheapest = False
        self.decline_ok = False
        self.offer_decline = False
        self.mode = "setup"
        self.cart_rows: dict[str, dict] = {}
        self.cart_total: Any = None
        self.seen_keys: dict[str, Any] = {}
        self.err_streak = 0
        self.drive_carted = False

    def t(self) -> float:
        return monotonic() - self.began

    def steps_left(self) -> int:
        return self.step_cap - self.steps

    def model(self, role: str, system: str, user: str, **kw: Any) -> Reply:
        return Call(self, role, system, user, **kw).send()

    def model_json(self, role: str, system: str, user: str, **kw: Any) -> Any:
        return Call(self, role, system, user, **kw).json()

    def model_many(self, jobs: list[tuple[str, str, str, dict]]) -> list[Any]:
        if not jobs:
            return []
        run = lambda j: Call(self, j[0], j[1], j[2], **(j[3] if len(j) > 3 else {})).json()
        if len(jobs) == 1:
            return [run(jobs[0])]
        with ThreadPoolExecutor(max_workers=min(POOL_SIZE, len(jobs))) as ex:
            return list(ex.map(run, jobs))

    def role_of(self, name: str) -> str | None:
        return self.store.roles.get(name)

    def msgs_left(self) -> int:
        return self.asker.left()

    def msgs_total_left(self) -> int:
        return self.asker.total_left()

    def read_the_request(self) -> dict:
        digest = rd_tool_digest(self.tools)
        got: dict = {}
        for chain in (STRONG[:1], STRONG[1:2], CHEAP[:1]):
            def call(sy: str, u: str, m: int, c: list = chain) -> str:
                return self.model("setup", sy, u, max_tokens=m, deadline=SETUP_BY + 15, chain=c).text
            got = read_request(call, self.request, digest, self.opening, fix_call=self.small_reader)
            if got.get("ok") and any(c["type"] == "line" and c.get("quote") for c in got.get("clauses") or []):
                break
        return got

    def small_reader(self, system: str, user: str, max_tokens: int) -> str:
        return self.model("read", system, user, max_tokens=max_tokens, deadline=SETUP_BY + 15).text

    def new_article(self, noun: str, qty: int, cap: Any, opens: list[str]) -> dict:
        return {"i": len(self.articles), "noun": noun, "qty": qty, "unit_cap": cap, "search": [], "open": opens,
                "cat": None, "asked": [], "pick": None, "backup": []}

    def take_lines(self, sm: dict) -> None:
        for ln in sm["lines"][:6]:
            noun = Text.of(ln["noun"]).strip()[:60]
            if not noun:
                continue
            cap = (sm["unit_budgets"].get(ln["i"]) or {}).get("amount")
            opens = [Text.of(h["prop"]).strip()[:40] for h in sm["hints"] if h["a"] == ln["i"] and h["prop"]][:6]
            self.articles.append(self.new_article(noun, ln["qty"] if 1 <= ln["qty"] <= 99 else 1, cap if cap and cap > 0 else None, opens))
            quote = next((c.get("quote") for c in self.clauses if c.get("type") == "line" and c.get("article") == ln["i"]), "")
            self.articles[-1]["counted"] = bool(numbers_in(quote or ""))
        if not self.articles:
            self.articles.append(self.new_article(" ".join(Text(self.request).words[-3:]) or "item", 1, None, []))

    def take_ledger(self, clauses: list[dict]) -> None:
        n = len(self.articles)
        line_quote = {c.get("article"): Text.of(c.get("quote")) for c in clauses if c.get("type") == "line"}
        for e in to_ledger(clauses):
            a = e.get("a")
            if isinstance(a, int) and not 0 <= a < n:
                continue
            vals = e["value"] if isinstance(e["value"], list) else [e["value"]]
            own = set(Text(self.articles[a]["noun"]).stems) if isinstance(a, int) else set()
            if e["op"] in ("is", "eq") and all(set(Text(v).stems) <= own for v in vals if Text.of(v).strip()):
                said = {w for v in vals for w in Text(v).stems}
                noun = Text(self.articles[a]["noun"]).stems if isinstance(a, int) else []
                hit = [i for i, w in enumerate(noun) if w in said] if len(vals) > 1 else []
                rest = {w for i, w in enumerate(noun) if not (hit and hit[0] <= i <= hit[-1])} - said
                apart = bool(Text.of(e.get("quote")).strip()) and not Text(line_quote.get(a, "")).has(e.get("quote"))
                if not (rest and apart):
                    continue
            e["step"] = 0
            self.ledger.append(e)

    def search_phrases(self, art: dict) -> list[str]:
        phrases: list[str] = []
        single = Text(art["noun"]).singular
        last = Text(single).words[-1:]
        alone: list[str] = []
        for req in self.reqs(art["i"]):
            if req["op"] not in ("is", "eq"):
                continue
            for v in (req["value"] if isinstance(req["value"], list) else [req["value"]])[:2]:
                if code_shaped(v):
                    continue
                ph = " ".join(Text(Text.of(v)).words[:2] + last).strip()
                if ph and ph not in phrases:
                    phrases.append(ph)
                whole = " ".join(Text(Text.of(v)).words[:5]).strip()
                if len(Text(whole).words) >= 2 and whole not in phrases and whole not in alone:
                    alone.append(whole)
        nouns = [single] if Text(single).stems == Text(art["noun"]).stems else [single, art["noun"]]
        return phrases[:5] + list(dict.fromkeys(nouns)) + alone[:3]

    def roles_missing(self) -> list[str]:
        tm = self.tm
        miss = [r for r in ("search", "buy") if not tm.get(r)]
        if not tm.get("cart") and not direct_buy(tm.get("buy")):
            miss.append("cart")
        if not tm.get("ask") and not self.store.sense:
            miss.append("ask")
        return miss

    def label_roles(self) -> None:
        miss = self.roles_missing()
        if not miss:
            return
        d = self.model_json("label", LABEL_SYS, f"STORE TOOLS (name, description, argument schema):\n{self.drive_catalogue()}\n\nMISSING: {', '.join(miss)}",
                            max_tokens=500, deadline=SETUP_BY + 20)
        if not isinstance(d, dict):
            return
        st = self.store
        taken = {m.get("name") for m in self.tm.values() if isinstance(m, dict)}

        def tool_of(role: str) -> tuple[str | None, dict]:
            x = d.get(role)
            if not isinstance(x, dict) or x.get("tool") not in self.schemas:
                return None, {}
            return x["tool"], x
        props = lambda nm: (self.schemas.get(nm) or {}).get("properties") or {}
        name, x = tool_of("search")
        if "search" in miss and name and x.get("query") in props(name):
            m = st.search_map(name, st.shape_of(name))
            m["query"] = x["query"]
            self.tm["search"] = m
        name, x = tool_of("cart")
        if "cart" in miss and name and name not in taken:
            sh = st.shape_of(name)
            basket = st.basket_of(name, sh) if sh["arrs"] else None
            if basket:
                self.tm["cart"] = basket
            elif x.get("id") in props(name) and x.get("qty") in props(name):
                self.tm["cart"] = st.cart_map(name, sh, x["id"], x["qty"], [k for k in sh["enums"] if len(sh["props"][k]["enum"]) == 2])
        for role in ("buy", "decline"):
            name, x = tool_of(role)
            if not name or self.tm.get(role):
                continue
            args = {k: v for k, v in (x.get("args") or {}).items() if k in props(name) and isinstance(v, (str, int, float, bool))}
            self.tm[role] = {"name": name, "args": args, "extras": [k for k in props(name) if k not in args]}
        name, x = tool_of("ask")
        closing = {(self.tm.get(r) or {}).get("name") for r in ("buy", "decline")}
        if "ask" in miss and name and name not in taken and name not in closing and x.get("text") in props(name):
            self.tm["ask"] = {"name": name, "text": x["text"]}
        st.roles.update({m["name"]: r for r, m in self.tm.items() if isinstance(m, dict) and m.get("name") and m["name"] not in st.roles})
        self.notes.append(f"roles labelled by the model for {miss}: {sorted(r for r in miss if self.tm.get(r))}")

    def fold_details(self, clauses: list) -> list:
        low = self.request.lower()
        gone: set[int] = set()
        for ref in clauses:
            if not isinstance(ref, dict) or ref.get("type") != "reference" or ref.get("with") or not Text.of(ref.get("quote")).strip():
                continue
            at = low.find(Text.of(ref["quote"]).lower())
            if at < 0:
                continue
            end = at + len(ref["quote"])
            folded: dict = {}
            for n, c in enumerate(clauses):
                if n in gone or not isinstance(c, dict) or c.get("type") != "value" or c.get("article") != ref.get("article") or c.get("op") != "is":
                    continue
                q = Text.of(c.get("quote")).lower()
                start = low.find(q, end) if q else -1
                if start < 0 or re.search(r"[.!?;]", low[end:start]) or len(Text(low[end:start]).words) > 1:
                    continue
                vals = [Text.of(v) for v in c.get("values") or [] if Text.of(v).strip()]
                if not vals or not Text.of(c.get("property")).strip():
                    continue
                folded[Text.of(c["property"])] = " ".join(vals)
                gone.add(n)
                end = start + len(q)
            if folded:
                ref["with"] = folded
        return [c for n, c in enumerate(clauses) if n not in gone]

    def sense_catalogue(self) -> str:
        out = []
        for nm, sch in self.schemas.items():
            props = (sch or {}).get("properties") or {}
            req = set((sch or {}).get("required") or [])
            args = []
            for a, v in props.items():
                v = v if isinstance(v, dict) else {}
                bits = [Text.of(v.get("type") or "?")] + (["required"] if a in req else [])
                for k in ("minimum", "maximum", "maxLength", "minItems", "maxItems"):
                    if isinstance(v.get(k), (int, float)):
                        bits.append(f"{k} {v[k]}")
                vals = v.get("enum")
                if isinstance(vals, list):
                    bits.append("allowed " + (Text.json(vals, 300) if len(vals) <= 8 else f"{len(vals)} values such as {Text.json(vals[:4], 200)}"))
                items = v.get("items")
                if isinstance(items, dict):
                    bits.append("items " + Text.json({k: x for k, x in items.items() if k != "description"}, 400))
                desc = Text.of(v.get("description")).strip()
                args.append(f"    {a}: {'; '.join(bits)}" + (f" | {desc[:240]}" if desc else ""))
            out.append(f"- {nm}: {Text.of(self.descs.get(nm))[:700]}\n" + ("\n".join(args) if args else "    (no arguments)"))
        return "\n".join(out)

    def sense_tools(self) -> None:
        try:
            st = self.store
            user = f"STORE TOOLS:\n{self.sense_catalogue()}\n\nANSWER TEMPLATE:\n{st.sense_template()}"
            answers = [self.model_json("label", SENSE_SYS, user, max_tokens=4000, deadline=SETUP_BY + 15, chain=SENSE_CHAIN)]
            got, doubts = st.read_sense(answers[0])
            if not got or doubts:
                self.notes.append("tool reading doubted, asking a second model: " + "; ".join(doubts)[:300])
                answers.append(self.model_json("label", SENSE_SYS, user, max_tokens=4000, deadline=SETUP_BY + 20, chain=SENSE_SECOND))
            if st.take_senses(answers):
                self.notes.append("tool roles read by the model: " + ", ".join(f"{r}={m.get('name')}" for r, m in self.tm.items() if isinstance(m, dict)))
            else:
                self.notes.append("the model's tool reading failed: roles from structure only")
        except Exception as exc:
            self.notes.append(f"tool reading: {type(exc).__name__}: {exc}"[:200])

    def closing_net(self) -> None:
        if self.store.sense:
            return
        try:
            got = self.store.closing_guess()
            self.notes.append("closing roles from structure only: " + ("; ".join(got) if got else "none found"))
        except Exception as exc:
            self.notes.append(f"closing roles: {type(exc).__name__}: {exc}"[:200])

    def setup(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as ex:
            sensing = ex.submit(self.sense_tools)
            got = self.read_the_request()
            sensing.result()
        self.label_roles()
        self.closing_net()
        clauses = self.fold_details(got.get("clauses") or [])
        self.clauses = clauses
        self.dropped_clauses = got.get("dropped") or []
        sm = summary(clauses)
        self.summary = sm
        self.spec = {"priority": sm["priority"], "priority_detail": sm["priority_detail"], "in_stock": True, "rules": sm["rules"]}
        read_steps = sm["step_cap"] if isinstance(sm["step_cap"], int) and sm["step_cap"] > 0 else None
        if read_steps:
            self.step_cap = min(self.max_steps, read_steps)
        read_msgs = sm["message_cap"] if isinstance(sm["message_cap"], int) and sm["message_cap"] >= 0 else None
        self.msg_cap = read_msgs
        self.take_lines(sm)
        self.take_ledger(clauses)
        for art in self.articles:
            art["search"] = self.search_phrases(art)
        self.cheapest = sm["priority"] == "cheapest"
        self.decline_ok = bool(sm["decline"]) or (not sm.get("offer_instead") and bool(self.tm.get("decline")))
        self.offer_decline = bool(sm.get("offer_instead")) and bool(self.tm.get("decline"))
        self.confirm = bool(sm["confirm"])
        self.pre_note = Text.of(sm["before_order"]).strip()
        n = len(self.articles)
        self.together = [p for p in sm["together"] if isinstance(p, list) and len(p) == 2 and all(isinstance(x, int) and 0 <= x < n for x in p)
                         and p[0] != p[1]]
        self.budget_raw = sm["budget"] or {}
        self.budget = None
        self.money_scale = 1.0

    def add_req(self, ai: int | None, req: Any, source: str) -> None:
        if not isinstance(req, dict) or not Text.of(req.get("prop")).strip():
            return
        op = Text.of(req.get("op") or "is").lower()
        if op not in ("is", "not", "min", "max", "between", "eq", "ref"):
            op = "is"
        val = req.get("value")
        if isinstance(val, list):
            val = [Text.of(v) for v in val if Text.of(v).strip()][:8]
        else:
            val = Text.of(val).strip()
        if not val and op != "ref":
            return
        if op in ("min", "max", "between") and (not re.search(r"\d", Text.of(val)) or all(named_figure(v) for v in (val if isinstance(val, list) else [val]))):
            op = "is"
        prop = Text.of(req["prop"]).strip()[:60]
        if source != "stated":
            val = self.retire(ai, prop, op, val, source)
        self.ledger.append({"a": ai, "prop": prop, "op": op, "value": val, "unit": Text.of(req.get("unit")) or None,
                            "source": source, "active": True, "step": self.steps, "quote": Text.of(req.get("quote")).strip()[:240]})

    def retire(self, ai: int | None, prop: str, op: str, val: Any, source: str) -> Any:
        for old in self.ledger:
            if not old["active"] or old["a"] != ai or Text.key(old["prop"]) != Text.key(prop):
                continue
            merge = (source == "revealed" and old["source"] == "revealed" and op in ("is", "eq") and old["op"] in ("is", "eq")
                     and old.get("step") == self.steps)
            if merge:
                val = list(dict.fromkeys((old["value"] if isinstance(old["value"], list) else [old["value"]])
                                         + (val if isinstance(val, list) else [val])))[:8]
            old["active"] = False
        return val

    def reqs(self, ai: int) -> list[dict]:
        return [x for x in self.ledger if x["active"] and x["op"] != "ref" and x["a"] in (ai, None)]

    def blind(self, it: dict, req: dict) -> bool:
        if self.cat.verdict(it, req) != "unknown" or self.cat.mentions(it, req) or req.get("op") == "not":
            return False
        if Text.json(req, 200) in (it.get("quoted") or []):
            return False
        return not (req["op"] in ("min", "max", "between") and self.cat.fact_values(it, req["prop"]))

    def contradicted(self, it: dict, ai: int) -> bool:
        for req in self.reqs(ai):
            if self.cat.verdict(it, req) == "no" or self.blind(it, req):
                return True
        return False

    def fit(self, it: dict, ai: int) -> tuple[int, int, int]:
        tally = collections.Counter(self.cat.verdict(it, req) for req in self.reqs(ai))
        return tally["yes"], tally["no"], tally["unknown"] + tally["close"]

    def ok_basic(self, it: dict, ai: int) -> bool:
        art = self.articles[ai]
        if it.get("gone") or (self.spec.get("in_stock", True) and self.cat.in_stock(it) is False):
            return False
        p = self.cat.price(it)
        if self.budget and p is not None:
            if p * art["qty"] > self.budget + 1e-9:
                return False
            cap = art.get("unit_cap")
            if cap and p > cap * self.money_scale + 1e-9:
                return False
        cat = art.get("cat")
        if cat and self.cat.category(it) and self.cat.category(it) != cat:
            return False
        return all(self.cat.waives(it, w) for w in art.get("waives") or [])

    def candidates(self, ai: int, read_only: bool = True) -> list[dict]:
        art = self.articles[ai]
        own = set(Text(art["noun"]).stems)
        pool = [it for it in self.cat.items.values() if (it["read"] or not read_only) and self.ok_basic(it, ai)
                and not (it.get("chain") and it["id"] not in (art.get("ids") or []) and not (own & set(Text(self.cat.category(it)).stems))
                         and not (art.get("cat") and self.cat.category(it) == art["cat"]))]
        if art.get("cat"):
            pool = [it for it in pool if self.cat.category(it) == art["cat"]] or pool
        elif art.get("ids"):
            pool = [it for it in pool if it["id"] in art["ids"]] or pool
        literal = [req for req in self.reqs(ai) if req["op"] in ("is", "eq")]

        def key(it: dict) -> tuple:
            y, n, u = self.fit(it, ai)
            plaus = sum(1 for req in literal if self.cat.verdict(it, req) == "yes" or self.cat.mentions(it, req))
            blind = sum(1 for req in self.reqs(ai) if self.blind(it, req))
            near = sum(1 for req in self.reqs(ai) if self.cat.verdict(it, req) == "close") if it.get("jl") else 0
            p = self.cat.price(it)
            p = 1e18 if p is None else p
            return (n + blind, near, -plaus, p, -y) if self.cheapest else (n + blind, near, -plaus, -y, u, p)
        return sorted(pool, key=key)

    def owners(self, ai: int) -> list[dict]:
        art = self.articles[ai]
        wanted_cat = art.get("cat")
        mine = art.get("ids") or []
        out = []
        for it in self.cat.items.values():
            if not it["read"]:
                continue
            cat = self.cat.category(it)
            if wanted_cat and cat:
                if cat == wanted_cat:
                    out.append(it)
            elif it["id"] in mine:
                out.append(it)
        return out

    def drive_status(self, goal: str) -> str:
        cat = self.cat
        known = sorted(cat.items.values(), key=lambda it: (not it["read"], 1e18 if cat.price(it) is None else cat.price(it)))[:25]
        table = "\n".join(f"- {it['id']}: {cat.title(it)} | price {cat.price(it)} | stock {cat.in_stock(it)} | {Text.json(it['facts'], 300)}"
                          for it in known)
        reqs = "; ".join(f"{x['prop']} {x['op']} {Text.json(x['value'])}" for x in self.ledger if x["active"] and x["op"] != "ref")
        lines = "; ".join(f"{a['qty']} x {a['noun']}" + (f" (current pick {a['pick']})" if a.get("pick") else "") for a in self.articles)
        cap = DEFAULT_MSGS if self.msg_cap is None else self.msg_cap
        cart = Text.json(self.cart, 400) if self.cart_seen else "not seen yet"
        return (f"GOAL: {goal}\nSTATUS: step {self.steps} of {self.step_cap}; shopper messages {self.msgs} of "
                f"{cap}; {int(TASK_SECONDS - self.t())} s left.\nLINES WANTED: {lines or 'see request'}\n"
                f"REQUIREMENTS: {reqs or 'see request'}\nCART AS LAST SEEN: {cart}\nLISTINGS SEEN:\n{table or '(none)'}\n"
                + ("ONE STEP LEFT: finish now with the purchase tool.\n" if self.steps_left() <= 1 else "")
                + "Make one or two tool calls.")

    def shortfalls(self, it: dict, ai: int) -> tuple[int, int]:
        reqs = self.reqs(ai)
        no = sum(1 for x in reqs if self.cat.verdict(it, x) == "no" or self.judge.level(it, x) == "no")
        met = sum(1 for x in reqs if self.cat.verdict(it, x) == "yes" or self.judge.level(it, x) == "exact")
        return no, len(reqs) - met

    def nearest_pool(self, ai: int) -> list[dict]:
        pool = self.candidates(ai, read_only=False)
        read = [it for it in pool if it.get("read")]
        if not self.offer_decline or not read:
            return pool
        return sorted(read, key=lambda it: (self.shortfalls(it, ai), 1e18 if self.cat.price(it) is None else self.cat.price(it), it["id"]))

    def decline_note(self) -> str:
        near = []
        for art in self.articles:
            ai = art["i"]
            pool = self.nearest_pool(ai)
            if not pool:
                pool = sorted((it for it in self.cat.items.values() if it["id"] in (art.get("ids") or [])),
                              key=lambda it: (self.fit(it, ai)[1], 1e18 if self.cat.price(it) is None else self.cat.price(it)))
            if not pool:
                continue
            it = pool[0]
            short = [x["prop"] for x in self.reqs(ai) if self.cat.verdict(it, x) == "no" or self.judge.level(it, x) == "no"]
            if self.cat.in_stock(it) is False:
                short.append("stock")
            near.append(f"{art['noun']}: nearest listing {it['id']} ({self.cat.title(it)[:50]})" + (f", short on {', '.join(short[:3])}" if short else ""))
        return "No listing meets every requirement of the request" + ("; " + "; ".join(near) if near else "") + "."

    def decline_args(self) -> dict:
        dec = self.tm.get("decline") or {}
        args = dict(dec.get("args") or {})
        if dec.get("note") and dec.get("note") != dec.get("token"):
            args[dec["note"]] = self.decline_note()
        return args

    def call_role(self, name: str, args: dict) -> str | None:
        buy, dec = self.tm.get("buy") or {}, self.tm.get("decline") or {}
        if buy.get("name") == name and dec.get("name") == name:
            dargs = dec.get("args") or {}
            return "decline" if dargs and all(args.get(k) == v for k, v in dargs.items()) else "buy"
        return self.role_of(name)

    def drive_role(self, name: str, args: dict) -> str | None:
        return self.call_role(name, args)

    def drive_gate(self, name: Any, args: Any, held: list[bool], asks: list[int]) -> str | None:
        if name not in self.schemas or not isinstance(args, dict):
            return "unknown tool or bad arguments; nothing was done"
        role = self.drive_role(name, args)
        if role == "ask":
            if self.msgs_total_left() - asks[0] <= 0:
                return "no shopper messages left; nothing was sent"
            asks[0] += 1
        if role == "decline" and not self.decline_ok and not held[0]:
            held[0] = True
            return "not sent: the request gives no rule for sending nothing; buy the closest match"
        if self.steps_left() <= 1 and role not in ("buy", "decline"):
            return "not sent: one step remains and it belongs to the purchase tool"
        return None

    def drive_tools(self) -> list[dict]:
        out = []
        for t in self.tools:
            f = t.get("function", t)
            if isinstance(f, dict) and f.get("name") in self.schemas:
                out.append(t if isinstance(t.get("function"), dict) else {"type": "function", "function": f})
        return out

    def drive_catalogue(self) -> str:
        return "\n".join(f"- {n}: {Text.of(self.descs.get(n))[:600]} | parameters: {Text.json(sch, 1500)}" for n, sch in self.schemas.items())

    @staticmethod
    def drive_tail(history: list[list[dict]], turns: int = 6) -> list[dict]:
        return [m for turn in history[-turns:] for m in turn]

    def drive_pick(self, native: bool, goal: str, deadline: float, history: list, log: list[str], chain: list | None = None) -> tuple[Any, list[dict], str]:
        head = f"REQUEST:\n{self.request[:5000]}"
        if native:
            reply = self.model("drive", DRIVE_SYS, self.drive_status(goal), history=[{"role": "user", "content": head}] + self.drive_tail(history),
                               tools=self.drive_tools(), max_tokens=900, deadline=deadline)
            if not reply.ok:
                return None, [], ""
            return True, [c for c in reply.calls if isinstance(c, dict) and isinstance(c.get("function"), dict)][:3], reply.text
        user = (f"{head}\n\nSTORE TOOLS (name: description | JSON schema of the arguments):\n{self.drive_catalogue()}\n\n"
                f"CALLS SO FAR (latest last):\n" + ("\n".join([x[:400] for x in log[-14:-4]] + log[-4:]) or "(none)")
                + f"\n\n{self.drive_status(goal)}")
        reply = self.model("drive", DRIVE_SYS + DRIVE_TEXT_SYS, user, max_tokens=900, deadline=deadline, chain=chain)
        if not reply.ok:
            return None, [], ""
        got = Text.loads(reply.text)
        rows = got.get("calls") if isinstance(got, dict) else got
        rows = rows if isinstance(rows, list) else ([got] if isinstance(got, dict) and got.get("tool") else [])
        calls = []
        for n, row in enumerate(rows[:3]):
            if isinstance(row, dict) and isinstance(row.get("tool"), str):
                args = row.get("args")
                args = Text.loads(args) if isinstance(args, str) else args
                calls.append({"id": f"t{len(log)}-{n}", "type": "function",
                              "function": {"name": row["tool"], "arguments": Text.json(args if isinstance(args, dict) else {})}})
        return True, calls, reply.text

    def drive(self, goal: str, deadline: float) -> None:
        was = self.mode
        self.mode = "drive"
        history: list[list[dict]] = []
        log: list[str] = []
        held = [False]
        asks = [0]
        native = bool(self.drive_tools())
        idle = blocked = 0
        chains = [None, CHEAP[1:], STRONG[:1]]
        ci = 0
        while (not self.finished and not self.env_dead and self.steps_left() > 0 and self.t() < deadline and not self.exhausted
               and self.spent < STOP_AT_USD):
            ok, wanted, text = self.drive_pick(native, goal, deadline, history, log, chains[ci] if not native else None)
            if ok is None:
                if self.exhausted or self.t() > deadline - 8:
                    break
                if native:
                    native = False
                    self.notes.append("drive: tool-calling request failed; switched to the JSON protocol")
                    continue
                ci += 1
                if ci >= len(chains):
                    break
                self.notes.append(f"drive: model reply failed; trying model set {ci}")
                continue
            if not wanted:
                idle += 1
                if idle >= 2 and native:
                    native, idle = False, 0
                    self.notes.append("drive: tool-calling replies held no call; switched to the JSON protocol")
                    continue
                if idle >= 2 and not native and ci + 1 < len(chains):
                    ci += 1
                    idle = 0
                    self.notes.append(f"drive: no usable call; trying model set {ci}")
                    continue
                if idle >= 2:
                    break
                if native:
                    history.append([{"role": "assistant", "content": text or "(no tool call)"},
                                    {"role": "user", "content": "No tool was called. Call a store tool now."}])
                else:
                    log.append("(previous reply held no tool call: answer with the JSON object and at least one call)")
                continue
            idle = 0
            asks[0] = 0
            for n, c in enumerate(wanted):
                c["id"] = c.get("id") or f"call{len(history)}x{n}"
            pairs = [((c.get("function") or {}).get("name"), Text.loads((c.get("function") or {}).get("arguments")) or {}) for c in wanted]
            notes = [self.drive_gate(n, a, held, asks) for n, a in pairs]
            todo = [i for i, m in enumerate(notes) if m is None]
            obs = self.env.send([pairs[i] for i in todo])
            results = {i: obs[j] for j, i in enumerate(todo) if j < len(obs)}
            for i in todo[:len(obs)]:
                n, a = pairs[i]
                role = self.drive_role(n, a)
                if role == "ask":
                    self.msgs += 1
                elif role in (None, "cart") or (role == "compare" and not self.tm.get("cart")):
                    if self.store.ok_reply(results[i]) is not False and not Env.faulty(a, results[i]):
                        self.drive_carted = True
                if self.finished and role == "buy":
                    self.ordered = True
                elif self.finished and role == "decline":
                    self.declined = True
            turn = [{"role": "assistant", "content": text or "", "tool_calls": wanted}] if native else []
            for i, c in enumerate(wanted):
                out = notes[i] if notes[i] is not None else (Text.json(results[i], 2500) if i in results else "not run: out of steps or time")
                if native:
                    turn.append({"role": "tool", "tool_call_id": c.get("id") or f"x{len(history)}-{i}", "content": out})
                log.append(f"{pairs[i][0]} {Text.json(pairs[i][1], 300)} -> {out[:2500]}")
            history.append(turn)
            if not obs and all(m is None for m in notes):
                break
            blocked = blocked + 1 if not todo else 0
            if blocked >= 3:
                break
        self.mode = was

    def pipeline(self) -> None:
        plan, hunt, asker = self.plan, self.hunt, self.asker
        plan.begin("find", "start")
        hunt.probe()
        plan.replan("probed")
        n = max(1, len(self.articles))
        tight = self.step_cap < 50 and n >= 3
        sas = self.finder.sas()
        for art in (self.articles if sas else self.articles[1:]):
            if self.t() > GUARD_AT - 70:
                break
            self.phase = "find"
            hunt.find(art["i"], hunt.sas_share(n) if sas else 1)
        for art in self.articles:
            hunt.own_department(art["i"])
            hunt.sibling_department(art["i"])
        plan.begin("read", "first reads")
        per_step = self.store.compare_max() if self.store.facts_role() == "compare" else 1
        room = plan.room("read")
        deep = (room - (2 if tight else 3) * n) // n
        menu_reads = max(2, min(PRE_ASK_PAGES, PRE_ASK_STEPS * per_step, max(room // (n + 1), deep)))
        if self.t() <= GUARD_AT - 70:
            hunt.prime([(a["i"], menu_reads) for a in self.articles])
        for art in self.articles:
            if self.t() > GUARD_AT - 70:
                break
            hunt.wave1(art["i"], menu_reads)
        asker.catch_up()
        hunt.cover()
        ready = [a["i"] for a in self.articles if len(self.owners(a["i"])) >= min(4, menu_reads)]
        if ready:
            plan.begin("ask", "ask early")
            hold = min(len(self.articles) - len(ready), max(0, self.msgs_left() - 1))
            asker.plan_round(min(1, self.msgs_left() - hold), only=ready)
        plan.begin("find", "finds")
        for k, art in enumerate(self.articles):
            if self.t() > GUARD_AT - 70:
                break
            self.phase = "find"
            hunt.find(art["i"], hunt.sas_share(n - k) if self.finder.sas() else min(2 if tight else 99, max(1, plan.room("find") // max(1, n - k))))
        hunt.merge_same_kind()
        for art in self.articles:
            if hunt.exact_department(art["i"]):
                hunt.learn_category(art["i"])
        self.judge.read_fit_wording()
        coded = self.judge.coded_lines()
        for ai in coded:
            hunt.fit_scan(ai, min(FIT_SCAN_PAGES, max(0, plan.slack() - 2 * n - self.msgs_left()) // len(coded)))
        plan.done.add("find")
        asker.catch_up()
        self.chains.walk()
        plan.begin("read", "reads")
        n = max(1, len(self.articles))
        first = plan.room("read") // n * (self.store.compare_max() if self.store.facts_role() == "compare" else 1)
        if self.t() <= GUARD_AT - 60:
            hunt.prime([(a["i"], max(min(8, first), min(2 if tight else 3, plan.room("read")))) for a in self.articles])
        for k, art in enumerate(self.articles):
            if self.t() > GUARD_AT - 60:
                break
            share = plan.room("read") // max(1, n - k)
            per = min(8, share * (self.store.compare_max() if self.store.facts_role() == "compare" else 1))
            per = max(per, min(2 if tight else 3, plan.room("read")))
            if per > 0:
                hunt.wave1(art["i"], per)
        asker.catch_up()
        hunt.cover()
        plan.begin("ask", "ask")
        asker.plan_round(min(1, self.msgs_left()))
        plan.begin("read", "wave2")
        hunt.wave2()
        asker.catch_up()
        plan.begin("ask", "ask2")
        asker.plan_round(self.msgs_left())
        plan.begin("read", "wave2b")
        hunt.wave2()
        if self.t() > LAST_ENV_AT - 10:
            return
        asker.catch_up()
        plan.begin("read", "decide")
        for art in self.articles:
            if hunt.exact_department(art["i"]):
                hunt.learn_category(art["i"])
        self.judge.read_fit_wording()
        hunt.resolve_pointers()
        self.judge.decide()
        if self.cheapest:
            for art in self.articles:
                hunt.prove_pick(art["i"], 3)
        if self.order.cart_early():
            knocked = self.order.reread_first_line()
            if asker.catch_up() or knocked:
                self.order.repair()
        if self.basket.weak_pass(WEAK_STEPS, "after decide") and self.t() < GUARD_AT - 50:
            self.judge.decide()
        self.basket.verify_loop(only=None if self.finder.sas() else self.basket.unpaired_weak())
        self.judge.fit_solve()
        for _round in range(3):
            if (not any(a.get("judged_none") for a in self.articles) or self.t() > GUARD_AT - 50
                    or plan.room("read") < 3 or not hunt.expand()):
                break
            self.judge.decide()
        hunt.mop_up()
        self.capsweep.run()
        self.cheap.before_sink()
        hunt.sink()
        self.basket.verify_loop(only=None if self.finder.sas() else self.basket.unpaired_weak())
        if self.basket.weak_pass(WEAK_STEPS, "before order") and self.t() < GUARD_AT - 45:
            self.judge.decide()
        self.judge.audit()
        self.judge.fit_solve()
        self.basket.knapsack()
        self.basket.budget_retry()
        self.judge.metric_pick("pipeline")
        self.judge.last_look()
        self.judge.code_guard("pre")
        fitted = [a for accs in self.judge.fit_hubs().values() for a in accs] + self.judge.coded_lines()
        if fitted:
            self.judge.see_pointers(fitted, loose=True)
        if hunt.should_decline():
            dec = self.tm["decline"]
            self.env.send([(dec["name"], self.decline_args())])
            self.declined = self.finished
            if self.finished:
                return
        self.order.run()

    def rescue_due(self) -> bool:
        if self.finished or self.env_dead or self.exhausted or not self.articles:
            return False
        if self.t() > GUARD_AT - 20 or self.steps_left() < 3:
            return False
        if self.err_streak >= ERR_STREAK:
            return True
        if self.order.no_lines:
            return False
        return not self.order.wrote_ok or self.order.attempts >= 2

    def run(self) -> list[dict]:
        goal = "buy what the request asks for, within every stated limit"
        try:
            self.setup()
            if all(self.tm.get(k) for k in ("search", "buy")) and (self.tm.get("cart") or direct_buy(self.tm["buy"])):
                self.mode = "pipeline"
                try:
                    self.pipeline()
                except StoreBroken:
                    pass
                self.mode = "finish"
                if self.rescue_due():
                    self.notes.append(f"pipeline ended at step {self.steps} without a landed cart write: tool loop takes over")
                    self.drive(goal + "; the planned flow stopped without a working basket, so finish the purchase", GUARD_AT)
            else:
                self.notes.append(f"store roles incomplete ({sorted(self.tm)}): tool loop drives")
                self.drive(goal, GUARD_AT)
        except Exception as exc:
            self.notes.append(f"crash: {type(exc).__name__}: {exc}"[:240])
        self.mode = "finish"
        try:
            if not self.finished:
                self.order.finish()
        except Exception as exc:
            self.notes.append(f"finish: {type(exc).__name__}: {exc}"[:240])
        try:
            self.dialogue.append(self.meta_row())
        except Exception as exc:
            self.dialogue.append({"role": "assistant", "content": "", "meta": {"steps": self.steps, "spent_usd": round(self.spent, 4),
                                                                              "notes": self.notes + [f"meta: {type(exc).__name__}: {exc}"[:200]]}})
        return self.dialogue

    def meta_row(self) -> dict:
        return {"role": "assistant", "content": "", "meta": {
            "elapsed_s": round(self.t(), 1), "steps": self.steps, "step_cap": self.step_cap, "asked": self.msgs,
            "spent_usd": round(self.spent, 4), "calls": self.calls, "driver": HAIKU, "ordered": self.ordered,
            "declined": self.declined, "articles": [{k: a.get(k) for k in ("noun", "qty", "cat", "pick", "asked", "look")} for a in self.articles],
            "ledger": [{k: x.get(k) for k in ("a", "prop", "op", "value", "source", "active", "ref", "opened")}
                       for x in self.ledger], "refs": self.chains.log,
            "searches": self.finder.export(),
            "heard": self.heard, "notes": self.notes, "tools": self.tm, "shape": self.store.slots(),
            "clauses": self.clauses, "dropped_clauses": self.dropped_clauses[:12], "notices": self.notices,
            "plan": self.plan.log[-12:], "phase_steps": dict(self.plan.used),
            "budget": self.budget, "decisions": self.judge.decisions, "ranks": self.rank_log, "judge": self.judge.export(),
            "confirm": self.confirm, "pre_note": self.pre_note, "together": self.together,
            "opening": {"calls": self.hunt.rank_calls, "log": self.hunt.open_log[-30:]}}}


DEFAULT_MSGS = 3


def tidy_with(withs: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(withs, dict):
        return out
    for k, v in withs.items():
        k, v = Text.of(k).strip()[:60], Text.of(v).strip()[:120]
        if v.lower() in ("yes", "true", "y"):
            v = k
        if k and v and v.lower() not in ("null", "none"):
            out[k] = v
    return out


def hop_adapter(d: Any, prop: str = "") -> tuple[dict | None, str | None]:
    if not isinstance(d, dict):
        return None, None
    kind = str(d.get("kind") or "").strip()
    seller = str(d.get("seller") or "").strip()
    quote = str(d.get("pointer_quote") or d.get("description") or "").strip()
    if d.get("denies") is True and not (seller and quote):
        return {"denies": True}, None
    value = d.get("value")
    value = str(value).strip() if isinstance(value, (str, int, float)) and str(value).strip() else None
    if not seller or not quote:
        return {"denies": False}, value
    withs = tidy_with(d.get("with"))
    said = set(Text(quote).stems) | set(Text(Chains.spaced(quote)).stems)
    kept = {k: v for k, v in withs.items() if set(Text(v).stems) <= said or set(Text(Chains.spaced(v)).stems) <= said}
    return {"seller": seller, "kind": kind or None, "with": kept, "hint": {} if kept else withs, "rule": d.get("rule"), "description": quote,
            "denies": d.get("denies") is True}, None


def decide_adapter(d: Any) -> Any:
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        ev = v.get("evidence") if isinstance(v.get("evidence"), dict) else {}
        out[str(k)] = {"pick": v.get("pick"), "backups": [b for b in (v.get("backups") or []) if isinstance(b, str)],
                       "evidence": {re.sub(r"^#\s*(\d+)$", r"\1", str(a).strip()): str(b) for a, b in ev.items() if isinstance(b, str) and "=" in b},
                       "all_met": v.get("all_met") is True}
    return out


SENSE_SYS = """An online store hands a shopping agent the tools in STORE TOOLS: name, description, and per argument its
type, limits, allowed values and description. Fill in ANSWER TEMPLATE: copy it, keep every tool name, argument name and
allowed value exactly as written, and replace each "?" from the lists below. Reply with the filled JSON object only.

why: up to 8 words copied from the tool's own description that show what it does.
does: exactly one word from this list (never a word from the args or values lists):
- search: finds listings from typed words (also when one argument switches it to listing a section instead)
- browse: lists the listings of a section, shelf, brand or price range and takes no typed words
- page: shows one listing's page
- facts: shows one listing's property table, kept apart from its page
- compare: shows several chosen listings together
- cart: puts listings into the cart or changes their unit counts, one line at a time or the whole cart at once
- remove: only takes a line out of the cart
- view: shows the cart
- quote: returns a quote or token that the ordering call needs
- order: places the order and closes the visit
- decline: closes the visit without buying
- finish: closes the visit, one argument choosing between ordering and declining
- message: sends the shopper a message and returns the reply
- usage: reports used or remaining allowances, such as messages, steps or budget
- other: none of these
ends: true when calling the tool closes the visit, else false.
args: for every argument one word from this list: query (the typed search words), category (section or shelf), brand
(maker), price (price limit), page (page number, row offset or cursor of a result list), id (one listing id, or one cart
line reference), ids (several listing ids), qty (unit count), lines (the whole list of cart lines), mode (picks what the
tool does), text (free message or note), token (a quote or confirmation token), other.
Judge each argument by what it holds, from the descriptions, its type and its limits, not by its name alone: an integer
that sets how many units is qty even when its name speaks of a line, row or number; a row offset or cursor is page.
values: for every allowed value one word from this list: order (places the order), decline (closes without buying),
add (adds units), remove (takes units out), words (typed-word search), browse (lists a section without typed words),
other."""

LABEL_SYS = """An online store hands a shopping agent the tools in STORE TOOLS. The agent's own reading of them left the roles
in MISSING unfilled. Using the descriptions and argument schemas only, name the tool for each role:
- search: finds listings from typed words; "query" is the argument that takes the words.
- cart: puts listings into the basket; "id" takes the listing id and "qty" the unit count (omit both when the tool takes
  a whole list of basket lines).
- buy: completes the purchase of the basket; "args" holds any fixed argument values that select buying.
- decline: ends the request without buying; "args" holds the fixed values that select that.
- ask: sends a message to the shopper and returns the reply without ending anything; "text" is the argument that takes the
  message. A tool that ends the request is never this role.
Output nothing but this JSON object, leaving out any role no tool fits:
{"search": {"tool": "<name>", "query": "<arg>"}, "cart": {"tool": "<name>", "id": "<arg>", "qty": "<arg>"},
 "buy": {"tool": "<name>", "args": {}}, "decline": {"tool": "<name>", "args": {}}, "ask": {"tool": "<name>", "text": "<arg>"}}"""

CONFIRM_SYS = """A shopping agent read an online store's search and product results and could not settle a few field names by
structure alone. Each line names a slot and the candidate keys with sample values. Reply with ONLY JSON naming the one
key per slot that holds that thing ({"<slot>": "<key or null>"}): price = the listing's current price; title = its
name; stock = its availability (add "stock_true_means_available": true or false when a stock slot is asked); category =
its shelf or department; seller = the merchant; maker = the brand; desc = the long description; cursor = the next-page
token or number; total = the number of matches. A "budget" line gives the shopper's stated budget and sample prices:
answer "budget": <the budget converted into the store's price unit, a number> or null when the unit is already the same.
A "stock_values" line gives, per availability key, its distinct values: answer "stock_available_values": [the values,
copied, that mean the listing can be ordered now]."""

DECIDE_SYS = """Settle which listing to buy per article. Input: REQUEST, PRIORITY, BUDGET, then per article its numbered
requirements and candidate listings (id, price, stock, category, title, property table, sometimes notes).

Test every candidate against every requirement, pair by pair:
T1 Deciding fact. Use the table key carrying the requirement's property name (case, underscores and ordinary
   synonyms of that name are irrelevant; where a plain key coexists alongside a longer look-alike, that plain key rules:
   finish over finish_group). A present key settles the matter alone; title, other keys and prose cannot overrule
   it. Lacking such a key, title or a further key may carry the value. Blurb prose, bullet text and seller memos are
   never facts.
T2 Stating the value. The wanted words must appear inside the fact consecutively, same order (space, hyphen and
   underscore are interchangeable): "dusk teal" carries teal, "quartz gen7" carries quartz, yet "20k rose gold" fails to
   carry "20k gold" (split words) and "woollen" does not carry "wool". A comma-separated fact is a list whose every
   entry counts ("lime, navy, cream" carries navy); a "+" fuses parts into one combined thing ("gloss + matte" is
   not gloss). Different wording with identical meaning counts (sofa for couch, colour for color); an abbreviation,
   initial or code does not ("M" fails medium, "blk" fails black); a neighbouring term of different meaning does
   not (clip-latch is not latch-free; dim glow is not unlit). Words that only repeat the key's own name may be
   omitted from the wanted value (wanting "twin bundle" under key "bundle" needs just "twin"). A range written as
   words ("30 and above") is matched as words.
T3 A negative requirement ("not X") fails when the deciding fact carries X, passes when it carries something else.
T4 A numeric limit requires a number inside the deciding fact; convert units beforehand, keeping the conversion
   for evidence.
T5 An options / variants / choices key counts only while that property has no proper key, and solely for the
   value literally named.
T6 Silence fails: a requirement the listing states nowhere is unmet; hence a listing lacking any condition entry
   is neither new nor non-refurbished.
T7 The category fixes which article kind each listing belongs to; other kinds are unpickable. Stock False means unorderable:
   it loses to every orderable listing passing as much.
T8 A candidate marked check=exact passed a careful separate test of its requirements, check=close came near without
   passing, check=no failed one; such a mark settles T1-T6 for that candidate.

With MUST WORK TOGETHER pairs, look in titles, tables and notes for proof the two fit: a model code printed on one and
cited by its partner after works-with / fits / compatible wording; a code pattern using x as wildcard (QBT-41xx covers
QBT-4130) minus exclusions stated in that sentence (except QBT-4130); an alias (also marketed as Rallo 390). A remark
concerning some other part (part KD2207 fits ...) proves nothing regarding its host page. Prefer textual proof to
plausibility.

Ranking: a full pass always beats a partial pass regardless of price. Among full passes, PRIORITY cheapest = lowest
price, best = the PRIORITY detail, none = any. With no full pass, the pick is still made: the candidate with the
fewest failures, all_met false. pick is null solely for an article with no candidates or whose candidates are all
of some other kind. Missing data, odd requests or an empty candidate list never change the reply format.

The reply starts with { and ends with }: nothing but this JSON object, no walkthrough, one entry per article index:
{"<article index>": {"pick": "<id or null>", "backups": ["<runner-up ids, best first>"],
  "evidence": {"<requirement #>": "<key> = <value verbatim>"}, "all_met": true|false}}
Evidence copies the deciding key and value; a converted number adds the conversion after a second "="
(girth = 1.07 m = 107 cm); a title-borne value reads title = <those words>; failed requirements get no line."""

JUDGE_SYS = """You check whether one store listing satisfies a shopper's requirements, the way a strict store auditor does. Judge each requirement on its own.

Where the value may come from:
1. A fact whose key names the requirement's property. Keys that are spelling variants of the same property count (spaces, underscores, case, singular or plural, a shorter or longer name for the same thing, e.g. "lens" or "lens_coating" for lens coating).
2. Only if no fact names the property: the title.
3. Nothing else counts: not a fact about a different property, not the description, and not good-to-know sentences that point to another listing, mention a version that "exists elsewhere", say what this item is not, or tell the reader to take a value from somewhere else.
If neither 1 nor 2 states the property, the requirement is not met.

When values match:
4. The listing's value must be the shopper's value. Allowed differences: case, singular or plural, hyphen, space or underscore, a small filler word ("below the elbow" = "below elbow"), and a standard abbreviation of the whole value ("HD" for high definition).
5. A different value fails even when it is close in meaning: "matte" is not "satin", "hand sewn" is not "hand stitched", "unglazed" is not "no glaze", "timber" is not "wood".
6. The shopper's words must appear whole and in order inside the value: "light sky blue" holds "sky blue"; "sky and sea blue" does not. A value with extra words still counts if it holds them whole.
7. A fact listing several items separated by commas meets the requirement if one item holds the value. Items joined by "+" form one combined value ("60% wool+40% acrylic" is not "wool").
8. Numbers: take the number from the property's own fact (or the title when no fact names it), convert units if they differ, then compare. "at least", "at most" and "between" include their bounds. A number written without a unit is in the shopper's unit.
9. "must not be X": met only when the listing states the property and its value is not X.
10. "one of: A or B": met if the value matches any of them.

For each requirement give a level:
- "exact": the rules above say it is met.
- "close": the rules above say it is not met, yet this listing's own fact (under a related key such as "options") or its title gives a value a reasonable shopper would call the same thing (a synonym or paraphrase).
- "no": a different value, or nothing on the listing states it.
Reply with JSON only, no prose:
{"r":[{"i":<requirement index>,"span":"<the exact listing text that decides, or empty>","level":"exact|close|no"}]}"""

AUDIT_SYS = """A shopping agent is about to buy one listing per article. Check the basket before purchase, article by article.

For each article you get its numbered requirements, the current PICK and some alternative listings with price, title and
property table. Decide with the store's facts only:
- The value comes from the fact whose key names the requirement's property (spelling variants count). Only when no fact
  names it, the title may state it. Descriptions and seller notes never count.
- A wanted value counts when its words appear whole and in order inside the fact; a comma list counts if one item holds
  it; numbers are compared after converting units, bounds included.
Then:
1. pick_ok is false only when a requirement is plainly not met by the PICK's own fact; list each such requirement with the
   deciding fact copied as "key = value".
2. replace_with names one alternative that meets EVERY requirement and, under the selection rule, beats the PICK: it is
   needed when the PICK fails, when the rule wants the cheapest and the alternative is cheaper, or when the rule wants the
   best by some measure and the alternative scores better on it. For each requirement index give the deciding fact of that
   alternative as "key = value" (or "title = <words>" when only the title states it). When the rule ranks by a measure,
   better_by names the fact key holding that measure and whether a higher or a lower number wins. Otherwise
   replace_with is null.
Answer with JSON only:
{"articles": [{"article": <index>, "pick_ok": true|false, "failed": [{"req": <index>, "quote": "key = value"}],
  "replace_with": "<listing id or null>", "replace_quotes": {"<requirement index>": "key = value"},
  "better_by": {"key": "<fact key>", "direction": "higher"|"lower"} or null}]}"""

METRIC_SYS = """A shopper wants, among listings that all meet the requirements, the one that ranks first under a choosing rule
(for example the most or the least of some measure). You get the rule and the listings with their facts.
Find the fact that holds the rule's measure on each listing, compare the values (convert units before comparing), and
choose the listing that ranks first. When the CURRENT PICK ties for first, choose it.
Answer with JSON only: {"best": "<listing id>", "value": "<the measure's fact on that listing, copied as key: value>"}"""

OPEN_SYS = """A shopping agent searched a store and now chooses which listings to open. Opening a listing costs one store step and steps are few, so it opens first the listings most likely to satisfy every requirement of the product line.

You get the product line, its requirements, how the shopper chooses among fitting listings, the shopper's request as context, and numbered search-result rows. A row shows only part of a listing; its full page holds more facts.

Put every row into one level:
- "likely": the row states values that meet one or more requirements, and nothing in it breaks any requirement.
- "maybe": the row says nothing that settles the requirements either way.
- "unlikely": the row states a value that breaks a requirement (a different value, or a number outside the allowed range once units agree), or the row is on a shelf other than the line shelf.

How to read a row:
1. Use the title, the short fields and the text. Sentences about other items ("goes with", "also fits", "more on part") say nothing about this listing.
2. A value is stated when the row gives the shopper's words whole, allowing case, plural, hyphen, spacing and standard abbreviations. Extra words around them are fine.
3. A row that offers several options ("8 / 10 / 12 inch", "red, blue") states each of them.
4. A "must not be" requirement is broken only when the row states the excluded value.
5. Rows on the line shelf count as that product even when they look like parts or accessories; judge them on the requirements alone.
6. The request only helps you understand the requirements and the shopper's choice. When it plainly sets a requirement for this line that the list lacks, use it as one more requirement.

Inside each level, list rows best first: rows that show more requirements met come first; when the shopper picks the best by some measure, rows stronger on that measure come first; rows that look alike keep their given order.

Reply with JSON only, every row number exactly once:
{"likely":[<row numbers>],"maybe":[<row numbers>],"unlikely":[<row numbers>]}"""

HOP_SYS = """While tracing a reference, a shopping agent opened a store page. The target was a listing fitting DESCRIBED AS,
whose PROPERTY the shopper requires. Read SELLER NOTES (title and property table are context) and say which of three
things the notes do. Pointers originate in the notes alone; DESCRIBED AS and PROPERTIES never supply one.

FORWARD. The notes send the reader to another listing for PROPERTY (shaped like: for <property>, use the <kind> by
<seller> having <details>, over this page; or: follow the <seller> <kind> at <detail>). Such a sentence is a pointer
and never a denial, whatever its closing words. Copy from that sentence only: pointer_quote = the sentence verbatim,
seller = the name tied onto the other listing (it may equal this page's seller), kind = the article kind written
there, with = one entry per detail given, rule = any tie-break for choosing among several matches (largest capacity
of those matching ...), description = pointer_quote.

DENY. Beyond any pointer, the notes state that a detail of DESCRIBED AS belongs to a different page, in words like
"the teal one is listed elsewhere", "a left-threaded version is sold by Marrow & Co", "this page is not the 775 g one".
Then denies is true and the pointer fields all stay null, pointer or no pointer. Without such a statement, denies
is false.

STAY. No pointer, no denial. value = what the notes settle for PROPERTY, also when it sits inside a longer sentence about this page ("offered in oak, with a teal finish" settles finish = teal), else null. DESCRIBED AS only says which page was wanted; it is never a pointer from this page. A pairing or mapping sentence
(our <property> rule links alpha with beta, gamma with delta) forms no pointer: mapping_quote = that sentence, and
value becomes the partner of this page's own PROPERTY value, as plain value words.

Output nothing but this JSON object:
{"denies": true|false, "pointer_quote": "<sentence or null>", "seller": "<seller or null>", "kind": "<kind or null>",
 "with": {"<property>": "<value>"}, "rule": "<tie-break or null>", "description": "<same as pointer_quote>",
 "mapping_quote": "<sentence or null>", "value": "<settled value or null>"}
pointer_quote and mapping_quote are copied out of SELLER NOTES; absent such a sentence, write null there and keep the
dependent fields null. Every with entry is a detail written inside pointer_quote itself (a size, a condition, a pack, a
colour, a style), never one taken from DESCRIBED AS or PROPERTIES. The reply begins with the opening brace: no reasoning
before or after the object."""

VALUE_SYS = """A shopping agent opened a store page because the shopper needs its PROPERTY value. Read the LISTING title, the
PROPERTIES table and the SELLER NOTES, and give the value this listing itself has for PROPERTY, in the page's own words. The
value may sit only in the title or inside a sentence of the notes ("our peach juice in a bottle" gives flavor = peach).
Sentences about some other listing never supply it. A pairing or mapping sentence for PROPERTY (our <property> rule links
alpha with beta) turns the value into the partner of this page's own PROPERTY value. No value on the page gives null.
Output nothing but this JSON object: {"value": "<value or null>"}"""

PAIR_SYS = """A store page NOTE may state a pairing rule for a PROPERTY: it links a value of PROPERTY with a partner value.
OWN VALUE is this page's own PROPERTY value; the note may name it or call it "its" value. Give the partner the note
links with OWN VALUE, copied exactly as the note writes it. A note that only describes the page, or links other values
only, gives null.
Output nothing but this JSON object: {"partner": "<text or null>"}"""

PICK_SYS = """Given a DESCRIPTION of a listing cited by a store page or the shopper, plus LISTINGS of that seller (id:
title | category | property table or "(no table read)" | notes), find which listing is meant.
Order of checks:
1 Decoys out. Every listing whose notes place a described detail on another page, or call itself not the <detail>
  version, is disqualified, whatever else matches.
2 Count described details each remaining listing carries within its own text: title, category, table, plus any
  memo sentence inside its notes (a memo naming the seller while restating the details weighs most). Other
  wording or units are fine (a sixpack equals six units; a dozen equals twelve); an unread table is no bar when title
  and notes carry the details.
3 Keep the listings carrying all details. If the DESCRIPTION ends with "RULE: ...", apply it among them (smallest
  diameter, heaviest pack, ...) and take the winner; without a rule take the memo matching best.
4 Should none carry every detail, take the non-decoy carrying most; null if none carries any.
Output nothing but this JSON object, no reasoning text: {"id": "<listing id or null>"}"""

LOOK_SYS = """A shopping agent is about to buy one listing per article. For each article below, a strict per-listing check rated the CHOSEN listing as failing at least one requirement. You get the article's numbered requirements, the CHOSEN listing and the OTHER opened listings of that article, each with the check's verdict per requirement and the text it relied on, followed by the listing's page.

Read the pages yourself, one requirement at a time:
1. The value must come from a fact whose key names the requirement's property; only when no fact names it, from the title. Facts about other properties, descriptions and notes that point to other listings do not count.
2. The shopper's words must appear in that value whole and in order (case, plural, hyphen and spacing aside); a different value fails even when close in meaning.
3. Numbers are compared after converting units; "at least", "at most" and "between" include their bounds.
4. "must not be X" is met only when the listing states the property with a value other than X.

Then give one verdict per article:
- "keep": the CHOSEN listing meets every requirement after all.
- "switch": the CHOSEN listing fails, and an OTHER listing meets every requirement; give its id, following the SELECTION RULE among those that do.
- "none": no listing shown meets every requirement.
For "keep" and "switch", quote for every requirement the deciding text of the listing you name as "<key> = <value as written>" (a title-borne value as "title = <those words>").

Reply with JSON only, one entry per article index shown:
{"<article index>": {"verdict": "keep|switch|none", "id": "<listing id or null>", "quotes": {"<requirement #>": "<key> = <value>"}}}"""

FITREAD_SYS = """You read short sentences taken from online store listings. In each sentence the product codes are replaced by numbered slots such as <1> and <2>. A slot holds one code or a list of codes. An example of the real sentence is given under each one.
For every sentence, say what it states about the listing it was taken from, slot by slot, with exactly one role per slot:
own: the listing's own model, part or series code.
fits: the listing works with, fits, suits, takes or supports what the slot names.
not: the listing does not work with what the slot names, or the slot is left out of a group it otherwise covers.
see: another part or listing to look at for more of this listing's details.
about: another part or model that the sentence makes a claim about.
theirs: what that other part or model works with; use it only together with about.
none: anything else.
Reply with JSON only: {"answers": [{"id": 1, "roles": {"1": "fits", "2": "not"}}]}"""

FIT_SYS = """GROUP A and GROUP B list store items (id: title | property table | notes). Each B item must suit some A item.
Using the texts alone, decide which pairs are proved fitting, and list the codes identifying each A item.
Proof of a fit (any one suffices):
- a code or model designation printed on one item (model ZN-7700; [KD2207]; also marketed as Rallo 390) which the other cites
  after works-with, fits, compatible-with or for wording;
- a code pattern using x as wildcard covering the other item's code (every QBT-41xx unit covers QBT-4130), unless
  that sentence excludes it (except QBT-4130);
- an alias (also marketed as ...) the other item cites.
Not proof: a remark about some other part (part KD2207 fits Rallo 390 speaks for KD2207, never for its host page);
agreement in generic words only (type, shape or size vocabulary); a shared brand.
Output nothing but this JSON object: {"pairs": [["<A id>", "<B id>"], ...],
 "models": {"<A id>": ["<each code, model designation or alias the A item claims>", ...]}}
pairs holds every proved pair ([] when none); models has an entry per A item (empty list when it claims no code)."""

PRE_ORDER_SYS = """One step remains, then the order goes in. The request demanded a message first; INSTRUCTION holds its words. FINAL
BASKET lists the lines queued for ordering, with unit prices and the sum; NOT FOUND lists articles never found; ASK
FOR APPROVAL TOO says if approval is also required beforehand.
Write that single message. It (1) does exactly what INSTRUCTION asks, in the request's wording (the sum, what was
dropped, which listings, whatever they wanted told); (2) names every basket line by its product name, quantity and
unit price, plus the sum, keeping the bracketed listing codes out unless INSTRUCTION itself asks for them; (3) if
ASK FOR APPROVAL TOO is True, closes on a yes-answerable question containing the word
"approve"; (4) asks nothing further and promises nothing past the order. Plain wording, two to four
sentences, no greeting.
Output nothing but this JSON object: {"message": "<the message>"}"""

DRIVE_SYS = """You take over a shopping agent whose planned flow stopped short, and operate the store tool set yourself until the
purchase is done. Each turn brings the REQUEST, a STATUS line (steps and messages used, seconds left), the LINES
WANTED with any current pick, the REQUIREMENTS gathered to date, the CART AS LAST SEEN plus LISTINGS SEEN.
Your job: an order holding exactly the articles and quantities asked, each meeting all requirements stated or later
revealed, obeying every rule the request sets, inside its limits on steps and messages. Working rules:
1 Search phrases hold two words or three, drawn from an article's values plus its noun; each word needs a match.
2 Judge by a listing's property table and category, never by title or blurb.
3 Open a listing before leaning on it; cart nothing unread.
4 Each shopper message covers one property of one named article; no pleasantries.
5 A basket line failing a requirement gets swapped, never supplemented. Read each tool's description and schema: a
  tool that replaces the whole basket takes every wanted line in one call, each time.
6 A reply that only reports a problem changed nothing; correct the arguments to fit the schema and call again.
7 Close out with steps to spare: exact quantities carted, cart checked, purchase completed. Ending unpurchased is
  permitted solely where the request itself prescribes that outcome should nothing fit, and indeed nothing does.
Per turn: one tool call or two, nothing more."""

DRIVE_TEXT_SYS = """
Tool protocol for this turn: native tool calls are unavailable, so STORE TOOLS lists every tool with the JSON schema of
its arguments, and CALLS SO FAR shows each earlier call with what the store returned. Answer with nothing but this JSON
object: {"calls": [{"tool": "<a name from STORE TOOLS>", "args": {<arguments that match its schema>}}]}, holding one
call or two. Array arguments take a JSON array whose elements follow the schema's items."""


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        run = Run(problem_data if isinstance(problem_data, dict) else {})
    except Exception as exc:
        return [{"role": "assistant", "content": "", "meta": {"error": f"{type(exc).__name__}: {exc}"[:240]}}]
    return run.run()


__all__ = ["agent_main"]
