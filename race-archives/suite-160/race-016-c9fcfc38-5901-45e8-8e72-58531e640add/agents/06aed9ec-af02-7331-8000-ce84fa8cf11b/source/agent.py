"""Sentinel 2: an evidence-ledger shopper with a hard clock.

Flow: label the store's tools -> compile the request into a ledger of wanted items and conditions -> gather listings
(searches, shelves, cursor pages) -> read the most promising records -> map each item's conditions onto the store's
fields once -> score every record in code -> interview the shopper on the likeliest unstated property -> settle the
basket (cart sync, cart check, basket message, order) and repair around refusals. Every network call is bounded by the
clock; when the finish reserve is reached, the episode is closed from what is already known.
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

DEEP = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-sol", "anthropic/claude-haiku-5.5"]
QUICK = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4-flash"]
TOTAL_S = 268.0        # our share of the 300 s episode
CLOSE_S = 42.0         # kept for the close: cart lines, basket message, order
HURRY_S = 36.0         # an ordinary tool call with less than this left hands over to the close
ASK_GROUP = 1         # properties named in one question (race shoppers turn away a message touching two of their subjects)
SURE_NONE = 8         # read records each clearly breaking a condition, none left unread: nothing fits
KIN_IGNORE = {"general", "main", "overall", "kind"}   # words that leave a property name's meaning unchanged
SLOW_S = 18.0         # a model reply slower than this means the proxy is loaded
MONEY_SOFT, MONEY_HARD, MONEY_STOP = 0.03, 0.05, 0.08    # USD per episode (run caps: $2 qualifying, $5 race)
LISTING_ID = re.compile(r"^[A-Za-z]{1,3}-[0-9a-f]{6,}$")


RACE_ALWAYS = True   # race-world behaviour (close misses bought, shopper asked early) whatever the message limit


class Hurry(Exception):
    """The finish reserve has been reached."""


# ======================================================================================================== text utils
def dumps(x: Any, limit: int = 4000) -> str:
    s = json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def tokens(s: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s or "").lower())


def plain_key(k: Any) -> str:
    """A store attribute key as plain words: separators become spaces and short code-like leading parts are dropped."""
    t = tokens(re.sub(r"[_.\-/]+", " ", str(k or "")))
    while len(t) > 1 and t[0] not in FILLER and (len(t[0]) <= 2 or (len(t[0]) == 3 and len(t) >= 3)):
        t = t[1:]
    while len(t) > 2 and len(t[-1]) <= 2 and not t[-1].isdigit():   # a short trailing tag ("storage type gr")
        t = t[:-1]
    return " ".join(t)


def root(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


FILLER = {"the", "a", "an", "of", "for", "and", "with", "to", "in", "on", "or", "by"}
COUNT_WORDS = {"item", "from", "set", "pair", "piece", "unit", "one", "some", "any"}   # how an item is counted, not what it is
VARIANT_WORDS = {"option", "available", "choice", "variant", "selection"}
CATCH_ALL = {"feature", "extra", "condition", "warranty", "certification"}   # umbrella names a shopper hears as many questions
BROAD = {"type", "kind", "style", "form", "category", "class", "option", "detail", "feature"}
NAMED_ARG = {"words": re.compile(r"text|query|term|keyword|search|words|phrase|^q$|^q_", re.I),
             "department": re.compile(r"categor|department|group|label|section|aisle|shelf", re.I),
             "brand": re.compile(r"brand|maker|vendor|manufact", re.I)}
MEASURE = re.compile(r"^\s*~?\d+(?:[.,]\d+)?(?:\s*(?:-|~|to)\s*\d+(?:[.,]\d+)?)?\s*[a-z]{1,12}\.?(?:\s*\(.*\))?\s*$", re.I)


def roots(s: Any) -> set[str]:
    return {root(w) for w in tokens(re.sub(r"\(.*?\)", " ", str(s or ""))) if w not in FILLER}


def alike(a: set[str], b: set[str]) -> bool:
    """Two property names for one thing: every word of one is in the other (a word or its longer form)."""
    same = lambda x, y: x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)))
    return bool(a and b) and (all(any(same(x, y) for y in b) for x in a) or all(any(same(x, y) for y in a) for x in b))


def near(x: str, y: str) -> bool:
    """One word, allowing one slip (a letter added, dropped or changed) in words of five letters or more: listings misspell
    ("polyster")."""
    if x == y:
        return True
    if min(len(x), len(y)) < 5 or abs(len(x) - len(y)) > 1 or any(c.isdigit() for c in x + y):
        return False   # codes and figures match exactly (QRX4400 is not QRX4410)
    if len(x) == len(y):
        return sum(a != b for a, b in zip(x, y)) == 1
    a, b = (x, y) if len(x) < len(y) else (y, x)
    return any(b[:k] + b[k + 1:] == a for k in range(len(b)))


def split_figures(text: Any) -> set[str]:
    """Word roots with figures split from the letters glued to them ("10FT" reads as "10 ft")."""
    return roots(re.sub(r"(\d)([a-z])", r"\1 \2", str(text or "").lower()))


def counts(text: Any) -> list[tuple[float, str]]:
    """Figures with the word written after them ("2pcs", "1 piece", "6 pcs/set"), the word cut to its consonants
    without a plural s ("pcs", "pc" and "piece" all give "pc") so different spellings of one word meet."""
    out = []
    # a size written as one figure by another ("54x78inch", "100*200cm") is a dimension, not a count
    flat = re.sub(r"\d+(?:\.\d+)?\s*[a-z]{0,3}\s*[x*\u00d7]\s*\d+(?:\.\d+)?\s*[a-z]*", " ", str(text or "").lower())
    for n, w in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", flat):
        sk = w[0] + re.sub(r"[aeiou]", "", w[1:])
        out.append((float(n), sk[:-1] if len(sk) > 2 and sk.endswith("s") else sk))
    return out


def same_word(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a)


def as_number(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(x))
    if not m:
        return None
    t = m.group(0)
    t = t.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", t) else t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


MONEY = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*(?:us\s+|u\.s\.\s+)?(cents?|pence|pennies|dollars?|bucks|euros?|pounds?|quid)\b"
                   r"|[$€£]\s?(\d[\d,]*(?:\.\d+)?)", re.I)


def money_said(text: str) -> list[tuple[float, bool]]:
    """The sums of money a text names, each with whether it is counted in cents (a hundredth of the main unit)."""
    out = []
    for m in MONEY.finditer(text or ""):
        n = as_number(m.group(1) or m.group(3))
        if n is not None:
            out.append((n, bool(m.group(2)) and m.group(2).lower().startswith(("cent", "pen"))))
    return out


def parse_json(text: Any) -> Any:
    """The first decodable JSON value in a model reply."""
    if not isinstance(text, str):
        return None
    dec = json.JSONDecoder()
    start = min([k for k in (text.find("{"), text.find("[")) if k >= 0], default=-1)
    if start >= 0:
        try:
            return dec.raw_decode(text[start:])[0]
        except ValueError:
            fixed = close_brackets(text[start:])
            if fixed:
                try:
                    return dec.raw_decode(fixed)[0]
                except ValueError:
                    pass
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return dec.raw_decode(text[i:])[0]
            except ValueError:
                continue
    return None


def close_brackets(text: str) -> str | None:
    """A reply whose JSON lost closing brackets (a list left open before the outer keys) with the missing ones added
    where the text ends; None when it cannot be balanced that way."""
    stack: list[str] = []
    in_str = esc = False
    end = len(text)
    for k, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
            if not stack:
                return None   # already balanced: the fault is elsewhere
            end = k + 1
    if in_str or not stack:
        return None
    return text[:end].rstrip().rstrip(",") + "".join(reversed(stack))


def names_something(words: list[str]) -> bool:
    """False for an empty value or a bare yes/no flag, which no search can look for."""
    return any(not re.fullmatch(r"yes|no|true|false|[yn]", w) for w in words)


def voice_lines(obs: Any) -> list[str]:
    """Sentences in a reply that are not listing data: shopper speech, store notices, refusals."""
    found: list[str] = []

    def walk(x: Any, depth: int, key: str) -> None:
        if depth > 7:
            return
        if isinstance(x, dict):
            record = any(isinstance(v, str) and LISTING_ID.match(v) for v in x.values())
            if record and depth:
                return
            for k, v in x.items():
                # a record handed back on its own is data too; only notes riding beside it in lists are speech
                if record and not (isinstance(v, list) and all(isinstance(e, str) for e in v)):
                    continue
                walk(v, depth + 1, str(k).lower())
        elif isinstance(x, list):
            for v in x:
                walk(v, depth + 1, key)
        elif isinstance(x, str):
            if len(x) > 220 and re.search(r"desc|about|text|blurb|detail|summary|title|name", key):
                return
            if len(re.findall(r"[A-Za-z]{2,}", x)) >= 4:
                found.append(x.strip())

    walk(obs, 0, "")
    return found


def direct_reply(obs: Any) -> list[str]:
    """The answer itself: the reply when it is plain text, or the plain-text values at its top level (notices come
    as lists beside it)."""
    if isinstance(obs, str):
        return [obs.strip()]
    if isinstance(obs, dict):
        return [v.strip() for v in obs.values() if isinstance(v, str)]
    return []


def shorten(text: str, cap: int) -> str:
    """Leading whole words of text, at most cap characters."""
    words_ = text.split()
    while len(words_) > 1 and len(" ".join(words_)) > cap:
        words_.pop()
    return " ".join(words_)[:cap]


def refused(obs: Any) -> bool:
    """A reply that turned the call down: bad arguments, a fault such as an empty basket, a blocked step (also when it
    rides inside a news wrapper)."""
    if not isinstance(obs, dict):
        return False
    if obs.get("error") or any(re.search(r"fault|error", str(k), re.I) and v for k, v in obs.items()):
        return True
    return any(isinstance(v, dict) and (v.get("error") or any(re.search(r"fault", str(k), re.I) and x for k, x in v.items()))
               for v in obs.values())


FIX_PROMPT = """A call to an online store's tool was refused. Write the arguments again so the call does what it was meant
to do and fits the tool's schema exactly (types, required fields, nested objects, lengths, bounds).
Reply with JSON only: {"args": {...}}
"""


# ======================================================================================================== gateway
class Gateway:
    """The episode's only door to the world and to the models; every call is bounded by the clock."""

    def __init__(self, env: dict[str, Any]):
        self.t0 = monotonic()
        self.caps: dict[tuple[str, str], int] = {}   # (tool, argument) -> longest text the store takes
        self.binding = env["binding"]
        pv = env["policy_view"]
        self.request = pv.get("query") or ""
        self.opening = pv.get("opening")
        self.schemas = [t.get("function", t) for t in (pv.get("tools") or [])]
        self.step_cap = int(pv.get("max_steps") or 30)
        self.steps = 0
        self.done = bool(pv.get("finished"))
        self.closing = False
        self.transcript: list[dict[str, Any]] = []
        self.unread: list[str] = []
        self.heard: list[str] = []
        self.cost = 0.0
        self.llm_calls = 0
        self.slow_hits = 0
        self.dead_models: set[str] = set()
        self.next_slot = 0.0
        self.calm_until = 0.0
        self.lock = threading.Lock()
        self.env_client = ProxyClient(timeout=25, max_retries=1, retry_delay=0.5)

    # ---------------------------------------------------------------- clock
    def left(self) -> float:
        return TOTAL_S - (monotonic() - self.t0)

    def steps_left(self) -> int:
        return self.step_cap - self.steps

    def _slot(self) -> None:
        with self.lock:
            gap = 0.8 if monotonic() < self.calm_until else 0.25
            wait = self.next_slot - monotonic()
            if wait > 0:
                sleep(wait)
            self.next_slot = monotonic() + gap

    @staticmethod
    def _pause(attempt: int, status: Any, salt: int) -> float:
        base = 1.5 if status in (429, 503) else 0.6
        jitter = 0.75 + ((salt * 131 + attempt * 977) % 50) / 100.0
        return min(6.0, base * 2 ** attempt) * jitter

    # ---------------------------------------------------------------- world
    def tool(self, name: str, args: dict[str, Any]) -> Any:
        for k, cap in self.caps.items():   # limits learned from the store's own complaints
            if k[0] == name and isinstance(args.get(k[1]), str) and len(args[k[1]]) > cap:
                args = {**args, k[1]: shorten(args[k[1]], cap)}
        obs = self.step_once(name, args)
        why = str(obs.get("detail") or "") if isinstance(obs, dict) and obs.get("error") else ""
        hit = re.search(r"\$\.(\w+): string length out of range", why)
        if hit and isinstance(args.get(hit.group(1)), str) and " " in args[hit.group(1)].strip():
            k = (name, hit.group(1))   # a length the schema never stated: once more, a few words shorter
            self.caps[k] = min(self.caps.get(k, 10 ** 6), max(8, len(args[k[1]]) * 2 // 3))
            obs = self.step_once(name, {**args, k[1]: shorten(args[k[1]], self.caps[k])})
        return obs

    def step_once(self, name: str, args: dict[str, Any]) -> Any:
        if self.done or self.steps >= self.step_cap:
            return None
        if not self.closing and self.left() < HURRY_S:
            raise Hurry()
        self.steps += 1
        body = {**self.binding, "step": self.steps, "action": {"name": name, "args": args}}
        reply = None
        for attempt in range(4):
            left = self.left()
            if attempt and left < 4:
                break
            self._slot()
            self.env_client.timeout = int(max(3, min(25, left - 1)))
            res = self.env_client.post_verbose("/environment/call", body)
            if isinstance(res.data, dict) and "observation" in res.data:
                reply = res.data
                break
            status = (res.error or {}).get("status")
            if status == 409:
                break
            if status in (429, 503):
                self.calm_until = monotonic() + 20
            pause = self._pause(attempt, status, self.steps + len(name))
            if self.left() < pause + 6:
                break
            sleep(pause)
        if reply is None:
            return None
        obs = unpack(reply.get("observation"))
        self.done = bool(reply.get("finished"))
        self.transcript.append({"role": "assistant", "tool_call": {"name": name, "arguments": args},
                                "environment_result": reply})
        for line in voice_lines(obs):
            if line not in self.heard[-50:]:
                self.heard.append(line)
                self.unread.append(line)
        return obs

    def drain(self) -> list[str]:
        out, self.unread = self.unread, []
        return out

    # ---------------------------------------------------------------- models
    def think(self, models: list[str], prompt: str, max_tokens: int = 700, as_json: bool = True,
              cap_s: float = 40.0, messages: list | None = None, tools: list | None = None) -> Any:
        if self.cost > MONEY_STOP:
            return None
        ladder = list(models)
        if self.cost > MONEY_HARD or self.slow_hits >= 2:
            ladder = [m for m in QUICK if m not in self.dead_models][:1]
        elif self.cost > MONEY_SOFT or self.slow_hits:
            ladder = [m for m in ladder if m in QUICK] or QUICK[:1]
        keep = 6.0 if self.closing else CLOSE_S
        for model in [m for m in ladder if m not in self.dead_models][:2]:
            room = self.left() - keep
            if room < 8:
                return None
            # a model that thinks first spends its allowance before answering: a small cap returns no text at all
            body: dict[str, Any] = {"model": model, "temperature": 0, "max_tokens": max(max_tokens, 1200),
                                    "messages": messages or [{"role": "user", "content": prompt}]}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            self._slot()
            t = monotonic()
            wait = min(cap_s, 25.0) if self.slow_hits else cap_s
            res = ProxyClient(timeout=int(max(5, min(wait, room))), max_retries=1, retry_delay=0.5).post_verbose(
                "/inference/chat/completions", body)
            if monotonic() - t > SLOW_S:
                self.slow_hits += 1
            with self.lock:
                self.llm_calls += 1
            data = res.data
            if isinstance(data, dict) and data.get("choices"):
                with self.lock:
                    self.cost += as_number((data.get("usage") or {}).get("cost")) or 0.0
                msg = data["choices"][0].get("message") or {}
                if tools:
                    return msg
                text = msg.get("content") or ""
                if not as_json:
                    return text
                got = parse_json(text)
                if got is not None:
                    return got
                continue
            status = (res.error or {}).get("status")
            if status in (400, 401, 402, 403, 404):
                self.dead_models.add(model)
            elif status in (429, 503):
                self.calm_until = monotonic() + 20
        return None

    def think_all(self, jobs: list[tuple[list[str], str, int]]) -> list[Any]:
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            futs = [ex.submit(self.think, m, p, n) for m, p, n in jobs]
            return [f.result() for f in futs]


# ======================================================================================================== listings
PRICE_KEY = re.compile(r"price|cost|amount|fee|cents", re.I)
STOCK_KEY = re.compile(r"stock|avail|orderable|on.?hand|purchas|sellable|buyable|in_store|instore|ready", re.I)
SHELF_KEY = re.compile(r"categor|department|shelf|aisle|section|range|group|collection|family|division", re.I)
TITLE_KEY = re.compile(r"title|name|heading|caption|label|headline", re.I)
MAKER_KEY = re.compile(r"brand|maker|manufacturer|company|label_name", re.I)
SELLER_KEY = re.compile(r"seller|vendor|merchant|shop|store|sold.?by", re.I)
TEXT_KEY = re.compile(r"desc|about|blurb|summary|detail|text|overview|story", re.I)
NOTE_KEY = re.compile(r"remark|note|fine.?print|small.?print|caveat|disclaim|aside", re.I)


def stock_flag(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        s = v.lower()
        if re.search(r"out of stock|sold out|unavailable|not available|back.?order|none left|no stock|discontinu|^no$|^false$|^0$", s):
            return False
        if re.search(r"in stock|available|yes|true|ready|on hand|^\d+ left", s):
            return True
    return None


class Listing:
    def __init__(self, id: str, raw: dict[str, Any] | None = None):
        self.id = id
        self.title = ""
        self.price: float | None = None
        self.stock: bool | None = None
        self.shelf = ""
        self.maker = ""
        self.text = ""
        self.notes = ""   # a listing's remarks: where trails and fit rules sit, never one of its properties
        self.attrs: dict[str, str] = {}
        self.raw: dict[str, Any] = raw or {}
        self.full = False
        self.gone = False

    def blob(self) -> str:
        return " ".join([self.title, self.shelf, self.maker, self.text, " ".join(f"{k} {v}" for k, v in self.attrs.items()),
                         dumps({k: v for k, v in self.raw.items() if isinstance(v, (str, int, float))}, 3000)])


def flatten_attrs(v: Any, prefix: str = "", out: dict[str, str] | None = None, depth: int = 0) -> dict[str, str]:
    """Attribute maps in any of the usual shapes: {name: value}, nested groups, or [{name/key, value}] pairs."""
    out = {} if out is None else out
    if depth > 3:
        return out
    if isinstance(v, dict):
        keys = {str(k).lower() for k in v}
        if len(v) <= 3 and keys & {"name", "key", "label", "attribute"} and keys & {"value", "val"}:
            nm = next(str(v[k]) for k in v if str(k).lower() in ("name", "key", "label", "attribute"))
            val = next(v[k] for k in v if str(k).lower() in ("value", "val"))
            if isinstance(val, (str, int, float, bool)):
                out[(prefix + " " + nm).strip()] = str(val)
            return out
        for k, x in v.items():
            name = (prefix + " " + str(k)).strip()
            if isinstance(x, (str, int, float)) and not isinstance(x, bool):
                out[name] = str(x)
            elif isinstance(x, bool):
                out[name] = "yes" if x else "no"
            elif isinstance(x, list) and x and all(isinstance(e, (str, int, float)) for e in x):
                out[name] = ", ".join(str(e) for e in x)
            elif isinstance(x, (dict, list)):
                flatten_attrs(x, name, out, depth + 1)
    elif isinstance(v, list):
        for x in v:
            flatten_attrs(x, prefix, out, depth + 1)
    return out


def to_listing(d: dict[str, Any]) -> Listing | None:
    lid = next((v for v in d.values() if isinstance(v, str) and LISTING_ID.match(v)), None)
    if not lid:
        return None
    item = Listing(id=lid, raw=d)
    loose_text = []
    plain_price, flags = None, []   # a "list" price stands in when it is the only one; a lone yes/no field is stock
    for k, v in d.items():
        key = str(k)
        if v == lid or v is None:
            continue
        if isinstance(v, (dict, list)) and not (isinstance(v, list) and all(isinstance(e, str) for e in v)):
            item.attrs.update(flatten_attrs(v))
            continue
        if isinstance(v, list):
            v = ", ".join(v)
        if PRICE_KEY.search(key) and not re.search(r"total|range", key, re.I) and as_number(v) is not None:
            if re.search(r"was|old|list|msrp", key, re.I):
                plain_price = as_number(v) if plain_price is None else plain_price
            elif item.price is None:
                item.price = as_number(v)
        elif STOCK_KEY.search(key) and stock_flag(v) is not None and item.stock is None:
            item.stock = stock_flag(v)
        elif NOTE_KEY.search(key) and isinstance(v, str):
            item.notes = (item.notes + " " + v).strip()[:1500]
        elif SHELF_KEY.search(key) and isinstance(v, str) and 0 < len(v) <= 80 and not item.shelf:
            item.shelf = v
        elif SELLER_KEY.search(key) and isinstance(v, str):
            item.attrs.setdefault(key, v)   # before the title test: a seller-name key names the seller, not the item
        elif TITLE_KEY.search(key) and isinstance(v, str) and not item.title:
            item.title = v
        elif MAKER_KEY.search(key) and isinstance(v, str) and not item.maker:
            item.maker = v
        elif TEXT_KEY.search(key) and isinstance(v, str):
            item.text = (item.text + " " + v).strip()[:1500]
        elif isinstance(v, str):
            loose_text.append(v)
            if len(v) <= 60:
                item.attrs.setdefault(key, v)
        elif isinstance(v, bool):
            flags.append(v)
        elif isinstance(v, (int, float)):
            item.attrs.setdefault(key, str(v))
    if item.price is None:
        item.price = plain_price
    if item.stock is None and len(flags) == 1:
        item.stock = flags[0]
    if not item.title:
        cand = [t for t in loose_text if 8 <= len(t) <= 300]
        item.title = cand[0] if cand else ""
    return item


FACTS_FIELD = re.compile(r"fact|spec|attr|propert|feature|trait|particular|detail|characteristic", re.I)


def plain_value(text: str, field: str = "") -> Any:
    """A value written out as text: none/yes/no flags, plain numbers, and name=value lists ("a=1; b=2") as a map. A piece
    with no "=" belongs to the value before it (a value may hold ";"); one lone pair counts only in an attribute field."""
    t = text.strip()
    if t.lower() in ("none", "null", "n/a", ""):
        return None
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return float(t) if "." in t else int(t)
    out: dict[str, str] = {}
    last = None
    for b in t.split(";"):
        m = re.match(r"\s*([^=;]{1,60}?)\s*=(.*)\Z", b, re.S)
        if m and re.match(r"[A-Za-z]", m.group(1)):
            last = m.group(1).strip()
            out[last] = m.group(2).strip()
        elif last is None:
            return t
        else:
            out[last] = (out[last] + ";" + b).strip()
    if len(out) >= 2 or (out and FACTS_FIELD.search(field)):
        return out
    return t


# a record written as one line of text: "name: value | name: value", one per line, or "name=value; name=value";
# snake_case names first (a title may hold a stray " | Something: "), then looser ones; the last field may be empty
RECORD_SPLITS = (re.compile(r"\s+\|\s+(?=[a-z][a-z0-9_]{0,40}:(?:\s|\Z))"),
                 re.compile(r"\s+\|\s+(?=[A-Za-z][\w ]{0,40}?:(?:\s|\Z))"),
                 re.compile(r"\n\s*(?=[A-Za-z][\w ]{0,40}?:(?:\s|\Z))"), re.compile(r";\s+(?=[A-Za-z][\w ]{0,40}?[:=])"),
                 re.compile(r"\s+\|\s+(?=[A-Za-z][\w ]{0,40}?=)"))
ID_IN_TEXT = re.compile(r"(?<![\w-])[A-Za-z]{1,3}-[0-9a-f]{6,}(?![\w-])")


def text_record(text: str) -> Any:
    """The record (or list) a text value spells out, when it carries a listing id; None otherwise."""
    t = text.strip()
    if len(t) < 12 or not ID_IN_TEXT.search(t):
        return None
    if t[0] in "{[":
        try:
            v = unpack(json.loads(t))
            return v if listing_dicts(v) else None
        except ValueError:
            pass
    for rx in RECORD_SPLITS:
        parts = rx.split(t)
        if len(parts) < 3:
            continue
        rec: dict[str, Any] = {}
        for part in parts:
            m = re.match(r"([A-Za-z][\w ]{0,40}?)\s*[:=]\s*(.*)\Z", part, re.S)
            if not m:
                rec = {}
                break
            rec[m.group(1).strip()] = plain_value(m.group(2), m.group(1))
        if rec and any(isinstance(v, str) and LISTING_ID.match(v) for v in rec.values()):
            return rec
    return None


def unpack(obs: Any, depth: int = 0) -> Any:
    """A reply with records written as text turned into the same reply with records as maps (any depth)."""
    if depth > 8:
        return obs
    if isinstance(obs, str):
        rec = text_record(obs)
        return rec if rec is not None else obs
    if isinstance(obs, list):
        return [unpack(v, depth + 1) for v in obs]
    if isinstance(obs, dict):
        return {k: unpack(v, depth + 1) for k, v in obs.items()}
    return obs


def listing_dicts(obs: Any, depth: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if depth > 7:
        return out
    if isinstance(obs, dict):
        if any(isinstance(v, str) and LISTING_ID.match(v) for v in obs.values()):
            out.append(obs)
        for v in obs.values():
            if isinstance(v, (dict, list)):
                out += listing_dicts(v, depth + 1)
    elif isinstance(obs, list):
        for v in obs:
            out += listing_dicts(v, depth + 1)
    return out


def cursor_of(obs: Any) -> tuple[Any, int | None]:
    """The next-page token a listing reply hands back (if any) and the total match count (if stated)."""
    token, total, _ = cursor_state(obs)
    return token, total


def cursor_state(obs: Any) -> tuple[Any, int | None, bool]:
    """cursor_of plus whether the reply carries a next-page field at all (an empty one means the last page)."""
    if not isinstance(obs, dict):
        return None, None, False
    body = obs
    inner = [v for v in obs.values() if isinstance(v, dict)]
    if len(obs) <= 3 and len(inner) == 1 and not listing_dicts({k: v for k, v in obs.items() if not isinstance(v, dict)}):
        if any(isinstance(v, list) for v in inner[0].values()):
            body = inner[0]
    token, total, spare = None, None, []
    marked = False
    for k, v in body.items():
        kl = str(k).lower()
        if isinstance(v, (list, dict)):
            continue
        if re.search(r"next|onward|continu|after|cursor|more|token|follow|further|forward|resume|marker|ahead|subsequent", kl):
            marked = True
            if v not in (None, "", False):
                token = v
        elif isinstance(v, int) and not isinstance(v, bool) and re.search(r"total|count|hits|match|found|number|num", kl):
            total = v
        elif v is None or (isinstance(v, str) and 0 < len(v) <= 80 and " " not in v and not LISTING_ID.match(v)):
            spare.append(v)
    if not marked and len(spare) == 1 and any(isinstance(v, list) for v in body.values()):
        token = spare[0]
    return token, total, marked


# ======================================================================================================== tool roles
LABEL_PROMPT = """Below are the tools of an online store, as function schemas. Give every tool exactly one role, and say
what each of its arguments is for. Roles:
 search (find products by words) | shelf (list one department or category) | maker (list one brand) |
 inspect (one product's full attribute record) | peek (one product's short view) | multi_inspect (records of several products) |
 cart_add | cart_drop | cart_show | order (place the order) | give_up (end with no purchase) | talk (send a message to
 the shopper) | combined (one tool doing several of these, chosen by an argument value) | other
Argument purposes: words | department | brand | product_id | product_ids | quantity | message | cursor | page_number |
 offset | price_max | price_min | sort | mode | remark | other
For a "combined" tool, list under "modes" each role it can play with the argument values that select it.
Reply with JSON only: {"tools": [{"name": "...", "role": "...", "args": {"<arg>": "<purpose>"},
 "modes": [{"role": "...", "set": {"<arg>": <value>}}], "quantity_sets_line": true|false}]}
quantity_sets_line is true when the quantity argument replaces the line's quantity (0 removes it), false when it adds.
TOOLS:
"""


READER_WORDS = re.compile(r"detail|spec|technical|inspect|view|page|info|data|propert|particular|fact|sheet|record|"
                          r"describe|examine|attribute|look")
CART_WORDS = re.compile(r"\badd|cart|bag|basket|hold|stash|remove|drop|delete|buy|order|checkout|check out|purchase|"
                        r"decline|refuse|message|ask|tell|units|quantity")


class Roles:
    """Which tool does what, and with which arguments; built from one labelling call checked against the schemas."""

    def __init__(self, schemas: list[dict[str, Any]]):
        self.schemas = {s.get("name"): s for s in schemas}
        self.by_role: dict[str, dict[str, Any]] = {}

    def props(self, tool: str) -> dict[str, Any]:
        return ((self.schemas.get(tool) or {}).get("parameters") or {}).get("properties") or {}

    def required(self, tool: str) -> list[str]:
        return list(((self.schemas.get(tool) or {}).get("parameters") or {}).get("required") or [])

    def learn(self, labels: Any) -> None:
        entries = labels.get("tools") if isinstance(labels, dict) else None
        for e in entries or []:
            if not isinstance(e, dict) or e.get("name") not in self.schemas:
                continue
            props = self.props(e["name"])
            args = {a: str(p).strip("<> ") for a, p in (e.get("args") or {}).items() if a in props}
            plays = []
            if e.get("role") and e["role"] not in ("combined", "other"):
                plays.append((e["role"], {}))
            for m in e.get("modes") or []:
                if isinstance(m, dict) and m.get("role"):
                    # a value copied from the reply template ("<department>") is no value at all
                    plays.append((m["role"], {k: v for k, v in (m.get("set") or {}).items() if k in props
                                              and not (isinstance(v, str) and re.fullmatch(r"\s*<[^<>]*>\s*", v))}))
            for role, fixed in plays:
                if role in self.by_role:
                    continue
                self.by_role[role] = {"tool": e["name"], "args": args, "fixed": fixed,
                                      "sets_line": bool(e.get("quantity_sets_line"))}
        self.guess()

    def guess(self) -> None:
        """Roles the labels missed, from names, descriptions and argument shapes."""
        for name, s in self.schemas.items():
            words_ = (name + " " + str(s.get("description") or "")).lower()
            props = self.props(name)
            req = self.required(name)
            strs = [a for a, p in props.items() if p.get("type") == "string" and not p.get("enum")]
            idish = [a for a in strs if re.search(r"id|code|ref|sku|item|product|listing|target", a.lower())]

            def put(role: str, args: dict[str, str]) -> None:
                if role not in self.by_role:
                    self.by_role[role] = {"tool": name, "args": args, "fixed": {}, "sets_line": False}

            if re.search(r"no purchase|give up|give_up|decline|abandon|without buying|stand down", words_):
                put("give_up", {strs[0]: "remark"} if strs else {})
            elif re.search(r"message|shopper|customer|say|tell|chat|note|write|ask", words_) and strs and not idish \
                    and not re.search(r"search|find|query|look", words_):
                put("talk", {strs[0]: "message"})
            elif not props and re.search(r"cart|bag|basket|trolley|selection", words_) and \
                    not re.search(r"order|checkout|check out|purchase|settle|buy|decline|give up|abandon", words_):
                put("cart_show", {})
            elif not req and re.search(r"checkout|check out|purchase|place|settle|finali|complete|buy", words_) and \
                    not re.search(r"no purchase|give up|decline|abandon|without", words_):
                put("order", {})
            elif len(req) == 1 and req[0] in idish and re.search(r"detail|spec|sheet|attribute|propert|fact|inspect", words_):
                put("inspect", {req[0]: "product_id"})
        if "inspect" not in self.by_role and "peek" in self.by_role:
            self.by_role["inspect"] = self.by_role["peek"]
        for name in self.schemas:
            for arg in self.required(name):
                # one finishing tool, its required choice deciding: order with the buying value, give up with the other
                vals = [str(v) for v in (self.props(name).get(arg) or {}).get("enum") or []]
                buy = next((v for v in vals if re.search(r"order|buy|purchase|check ?out|commit|confirm", v, re.I)), None)
                end = next((v for v in vals if re.search(r"pass|declin|none|cancel|abandon|skip|refus|give ?up", v, re.I)), None)
                if not (buy and end):
                    continue
                for role, val in (("order", buy), ("give_up", end)):
                    e = self.by_role.get(role)
                    if not e:
                        note = next((a for a, p in self.props(name).items() if p.get("type") == "string" and not p.get("enum")), None)
                        self.by_role[role] = {"tool": name, "args": {note: "remark"} if note else {}, "fixed": {arg: val},
                                              "sets_line": False}
                    elif e["tool"] == name and arg not in e.get("fixed", {}):
                        e["fixed"] = {**e.get("fixed", {}), arg: val}
            lines = self.lines_shape(name)
            add = self.by_role.get("cart_add")
            if lines and (not add or add["tool"] == name):
                # the whole basket in one call: every line goes in each time, and a line left out is removed
                self.by_role["cart_add"] = {"tool": name, "args": {}, "fixed": {}, "sets_line": True, "lines": lines}
                if (self.by_role.get("cart_drop") or {}).get("tool") == name:
                    del self.by_role["cart_drop"]

    def lines_shape(self, tool: str) -> tuple[str, str, str] | None:
        """(list argument, id field, count field) of a cart tool taking line objects, from the schema alone."""
        for a, p in self.props(tool).items():
            item = (p or {}).get("items") or {}
            if (p or {}).get("type") != "array" or item.get("type") != "object":
                continue
            ip = item.get("properties") or {}
            ids = [k for k, v in ip.items() if (v or {}).get("type") == "string"
                   and re.search(r"id|code|ref|sku|item|product|listing|target", k, re.I)]
            counts = [k for k, v in ip.items() if (v or {}).get("type") in ("integer", "number")]
            if ids and counts:
                return a, ids[0], counts[0]
        return None

    def has(self, role: str) -> bool:
        return role in self.by_role

    def arg(self, role: str, purpose: str) -> str | None:
        e = self.by_role.get(role) or {}
        # a line or slot number says where a line sits, never how many units it holds
        return next((a for a, p in e.get("args", {}).items() if p == purpose and not (
            purpose == "quantity" and re.search(r"line|position|index|slot|row|seq", a, re.I))), None)

    def shape(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Coerce argument values to the schema (types, enums, bounds) and fill required enums."""
        props = self.props(tool)
        out: dict[str, Any] = {}
        for k, v in args.items():
            p = props.get(k)
            if p is None or v is None:
                continue
            if p.get("enum") and v not in p["enum"]:
                hit = [x for x in p["enum"] if str(x).lower() == str(v).lower()]
                if not hit:
                    continue
                v = hit[0]
            t = p.get("type")
            if t == "integer":
                v = int(round(as_number(v) or 0))
            elif t == "number":
                v = as_number(v) or 0
            elif t == "string":
                v, cap = str(v), int(p.get("maxLength") or 2000)
                if len(v) > cap:   # a short search box keeps whole words
                    v = v[:cap].rsplit(" ", 1)[0] if cap < 200 and " " in v[:cap] else v[:cap]
            elif t == "array":
                v = (v if isinstance(v, list) else [v])[: int(p.get("maxItems") or 50)]
            if t in ("integer", "number"):
                if p.get("minimum") is not None:
                    v = max(v, p["minimum"])
                if p.get("maximum") is not None:
                    v = min(v, p["maximum"])
            out[k] = v
        for k in self.required(tool):
            if k not in out and (props.get(k) or {}).get("enum"):
                out[k] = props[k]["enum"][0]
        return out


# ======================================================================================================== request ledger
LEDGER_PROMPT = """Read a shopper's message to a store assistant and write down exactly what they asked for. The message
mixes a story with real conditions; take every condition, wherever it sits.
Reply with JSON only:
{"items": [{"what": "the product, in the shopper's words",
            "department": "the department / range / category the shopper names, verbatim, or null",
            "queries": ["2-4 short search phrases a product title would match: the product noun alone, and noun with one wanted value"],
            "qty": 1,
            "conditions": [{"prop": "attribute as the shopper names it", "rel": "is"|"not"|"min"|"max"|"any_of"|"has",
                            "value": "wanted value, units kept", "quote": "the words it came from", "ref": null}]}],
 "budget": number or null, "budget_per_unit": false, "goal": "cheapest"|"best"|"any", "best_by": "measure or null",
 "in_stock": true, "message_limit": number or null, "step_limit": number or null,
 "confirm_first": false, "rules": ["each procedural instruction, verbatim"],
 "silent_decline": false, "story": ["properties the story makes likely to matter, not stated as conditions"]}
Notes:
- conditions never include the product kind or its department, the price/budget, stock, or limits on messages/steps.
- "X-style" or "in the X style" attached to the product is a condition: style is "X" (a word like "kind" or "sort" is
  not: it only names the product).
- prop is the noun the shopper ties the value to: "for a sporty occasion" -> occasion is "sporty" (not "style").
- "not": an exclusion ("anything but glass"). "min"/"max": bounds ("at least 2 litres", "under 40 cm").
  A range the product must reach ("zoom that goes up to 30x", "covers up to 50 m2") is "min"; a cap the shopper
  sets on the product ("up to 40 cm long", "no heavier than 2 kg") is "max".
- A detail the shopper promises to check personally is a condition.
- budget is in the store's money unit exactly as written (a budget in cents stays in cents).
- confirm_first only when the shopper wants to see or approve the basket before the order is placed.
- silent_decline when the shopper asks not to be offered substitutes or named products if nothing fits.
- A model or compatibility code the shopper ties to a product ("the hinges have to be compatible with QRX4410",
  "fits my QRX4810") is a condition on THAT product only: {"prop": "compatible with", "rel": "has", "value": the code}.
- A property whose value is to come from ANOTHER listing ("for the plug type, go by the fairy lights from Vozenan
  (10 ft, warm white)", "the colour should be whatever the X from Y shows", "anything other than what the X from Y has")
  is a condition with "value": "" and "ref": {"department": "the other listing's product/department", "seller": "the
  shop/brand named", "with": {"property": "value" for each detail given about that listing}, "same": true} ("same":
  false for "anything other than"). That other listing is never itself an item to buy.
MESSAGE:
"""


class Need:
    def __init__(self, prop: str, rel: str, value: str, hidden: bool = False):
        self.prop = prop
        self.rel = rel
        self.value = value
        self.hidden = hidden
        self.off = False
        self.ref: dict[str, Any] | None = None   # the value is another listing's: {department, seller, with, same}


class Want:
    def __init__(self, what: str, department: str, queries: list[str], qty: int, needs: list[Need]):
        self.what = what
        self.department = department
        self.queries = queries
        self.qty = qty
        self.needs = needs

    def live(self) -> list[Need]:
        return [n for n in self.needs if not n.off]


# ======================================================================================================== condition mapping
MAP_PROMPT = """A store's records for one kind of product carry the attribute fields below (with sample values). For each
numbered condition, name the fields that state that property (other spellings, qualified names and usual alternative
names count), the sample values that satisfy it, and the sample values that clearly do not.
A value satisfies only if it states the wanted value, a spelling variant, or the same thing in other words; a related
but different value does not. For a "not" condition, "yes" lists the EXCLUDED values. For min/max give fields only.
Reply with JSON only: {"map": [{"n": 1, "fields": [...], "yes": [...], "no": [...]}]}
CONDITIONS:
"""


class Rule:
    """A condition bound to this store's vocabulary; checks records in code."""

    def __init__(self, need: Need, fields_: list[str], yes: list[str], no: list[str]):
        self.need = need
        self.fields = {" ".join(tokens(f)) for f in fields_ if tokens(f)}
        opts = [need.value]
        if need.rel == "any_of":
            opts = [o for o in re.split(r",|\bor\b|/", need.value) if o.strip()] or [need.value]
        self.yes = [roots(v) for v in yes + opts if roots(v)]
        if need.rel == "not":   # only spellings of the excluded value itself; the model sometimes lists the allowed ones
            base = roots(need.value)
            self.yes = [g for g in self.yes if g & base] or ([base] if base else [])
        self.no = [roots(v) for v in no if roots(v)]

    def field_hits(self, rec: Listing) -> list[str]:
        p = roots(self.need.prop)
        hits, menus = [], []
        for k, v in rec.attrs.items():
            nk = " ".join(tokens(k))
            if nk in self.fields or (p and p <= roots(k)):
                (menus if roots(k) & VARIANT_WORDS else hits).append(v)
        # "color options" / "available sizes" list what can be ordered, not what this record is: only a fallback
        return hits or menus

    def says(self, text: Any, groups: list[set[str]]) -> bool:
        # a figure glued to its unit or not ("250ml" / "250 ml") is one value: both spellings are compared
        t = roots(text) | split_figures(text)
        return any(h <= t or (h and all(w in t or any(near(w, x) for x in t) for w in h))
                   for g in groups for h in (g, split_figures(" ".join(sorted(g)))) if h)

    def check(self, rec: Listing) -> str:
        """'yes' | 'no' | 'unknown'."""
        rel = self.need.rel
        code = self.need.value.strip()
        if is_code(code) and rel in ("is", "has", "any_of"):
            # a compatibility code sits in the listing's text ("Compatible with: QRX4410, QRX6120"), not in a field named
            # like the wish (a "compatible model" field names other things): the whole record must carry the exact code
            text = rec.blob() + " " + dumps(rec.raw, 6000)
            if code.lower() in tokens(text):
                return "no" if code_fit(code, text) is False else "yes"
            fit = code_fit(code, text)
            if fit is not None:
                return "yes" if fit else "no"
            return "no" if rec.full or rec.text else "unknown"
        hits = self.field_hits(rec)
        if rel in ("min", "max"):
            want = as_number(self.need.value)
            vals = [as_number(v) for v in hits if as_number(v) is not None]
            if want is None or not vals:
                return "unknown"
            return "yes" if any((x >= want) if rel == "min" else (x <= want) for x in vals) else "no"
        if rel == "not":
            if any(self.says(v, self.yes) for v in hits):
                return "no"
            if hits:
                return "yes"
            return "no" if self.says(rec.title, self.yes) else "yes"
        if hits:
            if any(self.says(v, self.yes) for v in hits):
                return "yes"
            # a narrower value ("single" for "single can") counts when the record states the rest elsewhere ("cans")
            rest = roots(" ".join([rec.title] + [str(v) for v in rec.attrs.values()]))
            return "yes" if any(roots(v) and roots(v) < g and (g - roots(v)) <= rest for v in hits for g in self.yes) else "no"
        if self.says(rec.title, self.yes) or self.says(rec.text, self.yes):
            return "yes"
        if self.need.hidden and rec.full:
            return "no"   # a wish the shopper revealed must be stated by the listing itself; a fully read silent one fails
        return "unknown"


# ======================================================================================================== prompts (interview, notices)
PICK_PROMPT = """A shopper holds one or more firm wishes about a product that they only reveal when asked about that exact
property. Their story hides the clue. Read each story detail and decide which listed property it points at, e.g.:
checking tags, labels or small print -> what it is made of; photos, looks, matching clothes or a room -> its pattern,
colour or style, whichever the message leaves open; something it must fit or sit on (a wall, a wrist, a desk) -> that
shape or size; buying for another person, or how it suits the one wearing it -> who it is for;
how it feels on the skin -> material; the place or activity it is for -> its style or type; opening and closing it often
-> its fastening. Weather, timing and money worries are background, not clues. Order the listed properties by how
strongly the story points at them; with no clue, prefer what sets one listing apart from another: its colour or finish,
the kind of item within its department (its type), where it goes or is fitted, a key measure (how much it holds, how
deep or wide it is), and who it is for (for anything worn). Never pick a catch-all (features, options, extras,
condition, warranty, certifications, eco claims): a shopper hears it as several questions at once. Never list a property
the message already settles; clothing sizes rarely. Reply with the JSON alone, no reasoning text:
{"ask": ["property copied from the list", ...]}
MESSAGE: """

ANSWER_PROMPT = """A store assistant asked the shopper: {q}
The shopper answered: {a}
What does the answer mean? Reply with JSON only:
{"meaning": "wish" | "indifferent" | "confused" | "too_many" | "other",
 "wishes": [{"item": "which product it concerns", "prop": "property", "rel": "is"|"not"|"min"|"max"|"any_of"|"has", "value": "value"}]}
"confused": the shopper did not follow the question. "too_many": the shopper asks to take one matter at a time or finds
the message too much at once. A question back that states a value ("Is it clear that X means Y for me, nothing less?")
is a wish, not confusion. Words of insistence about a value (must, only, firm, required, will
not budge, not negotiable, no other way) mean the shopper REQUIRES that value: rel "is". Use rel "not" only when the
shopper rejects the value itself ("anything but X", "no X"). "prop" is the property the ANSWER names, which can differ
from the one asked about (asked about speed, answered "only wireless n will do for network" -> prop "network"); take the
asked property only when the answer names none. Words about the decision itself (closed, settled, fixed, final, locked)
are never part of the value: "that one is closed, with oak and nothing else" -> value "oak".
When the answer gives no value of its own but points at another listing (its department, who sells it, a few of its
properties), the value is whatever that listing has: give "value": "" and add to the wish
"ref": {"department": "...", "seller": "...", "with": {"property": "value"}, "same": true}. "same" is false when the
shopper rejects that listing's value ("anything other than", "anything but", "not what ... has"). Add "pair": true when
the shopper wants a value that goes with or works alongside that listing's value rather than the value itself."""

NOTICE_PROMPT = """New lines appeared while an assistant shops for a customer. What does each change?
Reply with JSON only: {"events": [{"type": "drop"|"add"|"budget"|"quantity"|"price"|"sold_out"|"refused"|"agreed"|"nothing",
 "prop": "property or null", "rel": "is"|"not"|"min"|"max" or null, "value": "value or null", "amount": number or null, "item": "product or null",
 "ids": ["cart ids it concerns"], "why": "short reason"}]}
drop: the shopper no longer needs a stated property. add: a new firm wish ("rel" "not" when a value is ruled out). budget: a new spending cap in the store's unit.
quantity: the shopper now wants a different number of units of one product ("amount" = the new number, "item" = which).
price: a listing's price changed. sold_out: something is no longer available ("your pick" means the cart item).
refused: the store blocked the last action. agreed: the shopper approved.
CONDITIONS NOW: {conds}
CART: {cart}
LINES:
{lines}"""

CHECK_PROMPT = """Before buying, check one product record against the shopper's conditions. A condition is met when ANY
field about that property (including more specific names for it, e.g. a field naming the product kind plus the
property) states the wanted value, a spelling of it, or plainly the same thing; one such field is enough even if another
field disagrees. With no field about the property, the title may show it. A related but different value does not count.
"not" conditions fail when the record shows the excluded value. A condition nothing addresses is "unclear", not failed.
Reply with JSON only: {"verdict": "fits" | "fails" | "unclear", "failed": ["condition that is clearly not met"]}
CONDITIONS: {conds}
RECORD: {rec}"""

LINK_PROMPT = """Does the shopper say the products must work with, fit, or match a product they already own or are buying
(a device, machine, vehicle or system)? Reply with JSON only: {"anchor": "that product as named (maker and model words)" or null,
"for_items": [0-based indexes of the wanted products that must fit it]}
WANTED PRODUCTS: {items}
MESSAGE: {q}"""

CODE_VALUE = re.compile(r"(?=(?:\D*\d){3})(?:[A-Z]{2,5}(?:\d{1,2}-)?[ -]?|[A-Za-z]{2,5}(?:\d{1,2})?-?)\d{2,6}(?:-\d{2,6})?")
def is_code(value: Any) -> bool:
    """A model or part code ("QRX4410", "KTV 5530", "ktv 5510", "MH3-20418"), whatever its case."""
    return bool(CODE_VALUE.fullmatch(str(value or "").strip().upper()))


CODE_PARTS = re.compile(r"([A-Za-z]{2,5})(?:(\d{1,2})-)?[ -]?(\d{2,6})(?:-(\d{2,6}))?")
NOT_FIT = re.compile(r"not compatible with|incompatible with|does not fit|doesn't fit|won't fit|will not fit|not for|"
                     r"except(?:ing)?|excluding|other than|but not", re.I)


def code_fit(code: str, text: Any) -> bool | None:
    """Whether a record's text says the item fits the model code: True (named exactly, by a wildcard family "55xx", a
    part series "series MH3", or a range "5500-5599"), False (named in an exclusion, or only other models of the same
    maker prefix are named), None (the text names no model of that prefix)."""
    m = CODE_PARTS.fullmatch(code.strip())
    if not m:
        return None
    pre, ser, num = m.group(1).lower(), m.group(2) or "", m.group(3)
    t = " ".join(str(text or "").lower().split())
    hits, others = [], 0
    pat = re.compile(r"(?<![a-z0-9])" + re.escape(pre) + r"(?:(\d{1,2})-)?[ -]?((?:\d|x){2,6})(?:\s*(?:-|–|to|through|thru)\s*(?:"
                     + re.escape(pre) + r"(?:\d{1,2}-)?[ -]?)?(\d{2,6}))?(?![a-z0-9])")
    for f in pat.finditer(t):
        s1, body, hi = f.group(1) or "", f.group(2), f.group(3)
        if ser and s1 and s1 != ser:
            others += 1
            continue
        ok = False
        if "x" in body:
            stem_ = body.rstrip("x")
            ok = len(body) == len(num) and "x" not in stem_ and num.startswith(stem_)
        elif hi and len(hi) == len(body) == len(num):
            ok = int(body) <= int(num) <= int(hi)
        else:
            ok = body == num or (ser and s1 == ser and not body)
        if ok and ("x" in body or hi):   # a family or range can carve models out right after it ("except 5599")
            cut = re.match(r"[\s,(]*(?:except(?:ing)?|excluding|but not|other than|save)\s+((?:(?:and|or|,|" + re.escape(pre)
                           + r"|\d{1,2}-|[ -])*\d{2,6})+)", t[f.end():f.end() + 80])
            if cut and num in re.findall(r"\d{2,6}", cut.group(1)):
                ok = False
        if ok:
            hits.append(f.start())
        else:
            others += 1
    if ser:   # "part series MH1, MH3": the series alone vouches for every part number in it
        for f in re.finditer(r"series[^.;]{0,60}", t):
            seg = f.group(0)
            if re.search(r"(?<![a-z0-9])" + re.escape(pre + ser) + r"(?![a-z0-9])", seg):
                hits.append(f.start())
            elif re.search(r"(?<![a-z0-9])" + re.escape(pre) + r"\d", seg):
                others += 1
    for h in hits:
        if NOT_FIT.search(t[max(0, h - 40):h]):
            return False
    if hits:
        return True
    return False if others else None
MODEL_CODE = re.compile(r"\b[A-Z]{2,5}\d{3,6}\b|\b[A-Z][a-z]{2,}\s\d{3}\b")


def model_codes(r: "Listing") -> set[str]:
    """Model numbers a listing's own text names (its number, names it is also sold under, models it works with); the
    title is left out, where a listing's unique stock code sits."""
    text = " ".join(str(v) for v in r.raw.values() if isinstance(v, str) and v != r.title)
    return {c.lower() for c in MODEL_CODE.findall(text)}


PROSE_PROMPT = """These product records state their properties in sentences rather than in a table. For each record list
every property its text states about the product itself, as a short property name and the value in the record's own
words (sizes and amounts with their units). Leave out shipping, store hours, offers, guarantees and anything the text
does not plainly state. Reply with JSON only: {"records": [{"id": "...", "facts": {"property": "value"}}]}
RECORDS: """
STATED = re.compile(r"\b(?:is|are|measures|weighs|comes in|made (?:of|from))\s+[^.;!?]{1,60}[.;!?]|\b[a-z][a-z ]{2,30}:\s*[^,;.]{1,40}", re.I)


def stated_count(r: "Listing | None") -> int:
    """Properties a record states in its text rather than as fields (a rough count of "its X is Y." sentences)."""
    return len(STATED.findall(r.text)) if r else 0


PAGE_FROM = re.compile(r"\b(?:page|listing) for (?:the |a |an )?([a-z][\w &,'-]{1,60}?) from ([A-Z][\w'-]+)")
REF_HOPS = 7   # listings a reference may hand on through before the one that states the value

HOP_PROMPT = """A shopper's wish for the {prop} of a product follows another listing; this is that listing's record.
Its notes may hand the {prop} on to a different product ("take it from", "look to", "refer to", "comes from another
product", "not from this item"): then that product is next, even when this record has a {prop} field of its own. A
version said only to exist elsewhere is not a hand-on. Otherwise give the {prop} this listing states (a field or the
notes). The notes may also state a rule pairing {prop} values ("pairs A with B"): give it as {"A": "B", ...}.
Reply with JSON only: {"next": {"seller": "...", "department": "...", "with": {"property": "value"}} or null,
"value": "the {prop} stated here" or null, "pairs": {...} or null}
RECORD: """

SERIES_CODE = re.compile(r"\bseries\s*(?:no\.?|number|code)?\s*[:#]?\s*([A-Za-z]{1,4}\d{1,4})\b", re.I)
INSIST = re.compile(r"nothing less|full stop|not flexible|only .{1,60} will|firm on|settled|must stay|cannot be skipped|"
                    r"stays firmly|will not give up|keep .{1,60} fixed|filter strictly|meant to be|dead set|non-negotiable", re.I)
REF_WORDS = re.compile(r"listing|sold by|seller|what'?s (?:given|listed|shown|on)|on the page|alongside|works with", re.I)
VALUE_AFTER = re.compile(r"\b(?:match|matching|go past|sitting at|except|stick(?:ing)? to|be)\s+([a-z0-9][a-z0-9 .\-/+&']{0,28}?)(?=,| or | as | for | no | even | when | in the |[.!?]|$)", re.I)
GONE_WORDS = re.compile(r"sold out|out of stock|none (?:are )?left|zero left|no longer|unavailable|not available|gone|"
                        r"can.?t be (?:ordered|bought)|last (?:piece|one|unit)|been taken|(?:shelf|listing).{0,40}empty|r[au]n out", re.I)
DECLINE_TEXT = "I'm sorry, I could not find anything that meets all of your conditions, so I am not placing an order."


# ======================================================================================================== the shopper
class Sentinel:
    def __init__(self, problem: dict[str, Any]):
        self.gw = Gateway(problem["environment"])
        self.roles = Roles(self.gw.schemas)
        self.wants: list[Want] = []
        self.plan: dict[str, Any] = {}
        self.listings: dict[str, Listing] = {}
        self.found_for: dict[str, set[int]] = {}
        self.rules: dict[int, tuple[int, list[Rule]]] = {}
        self.marks: dict[tuple[int, str], dict[str, Any]] = {}
        self.done_queries: set[str] = set()
        self.cursors: list[dict[str, Any]] = []
        self.asked: dict[int, list[str]] = {}
        self.ask_order: dict[int, list[str]] = {}
        self.unheard: dict[int, list[str]] = {}
        self.samples: dict[int, dict[str, list[str]]] = {}
        self.measures: dict[int, set[str]] = {}   # question properties whose values are figures with units
        self.one_each = False   # set when the shopper asks for one matter per message
        self.heard_meaning = ""
        self.msgs = 0
        self.msg_cap = 3
        # race behaviour in every world: race message limits now vary (5, then 2, then 3), so a limit is no race test
        self.race = RACE_ALWAYS
        self.budget: float | None = None
        self.cart: dict[str, int] = {}
        self.batch_cap: int | None = None
        self.last_total: int | None = None
        self.banded: set[tuple[int, float]] = set()
        self.vetted: set[tuple[int, str]] = set()
        self.rechecked: set[str] = set()   # cart lines read again just before the order
        self.peeked: set[str] = set()   # records also read through a separate short-view tool
        self.early_id: str | None = None   # the line held early to draw the store's news
        self.item_limit: int | None = None   # step count where the current item's exploring share ends
        self.matched: dict[str, set[str]] = {}   # record id -> words of the searches that listed it
        self.struck: dict[tuple[int, str], str] = {}
        self.heard_answers: list[tuple[str, str]] = []
        self.unreadable: set[str] = set()
        self.swapped = False
        self.probed = False
        self.probe_tries = 0
        self.store_cents: bool | None = None   # whether the budget is held in cents (learnt from the request)
        self.cost_cuts = 0
        self.prosed: set[str] = set()   # records whose sentences were read out into fields
        self.prose_missed: set[str] = set()
        self.ref_src: set[str] = set()   # listings read as the source of a wish's value: other products, not the pick
        self.cart_item: dict[str, int] = {}   # cart line id -> the wanted product it was put there for
        self.told: tuple | None = None
        self.viewed: tuple | None = None
        self.orders = 0
        self.fixes = 0   # refused calls written again by a model
        self.link: dict[str, Any] = {}
        self.log: list[str] = []

    # ---------------------------------------------------------------- basic moves
    def use(self, role: str, args: dict[str, Any] | None = None, by_purpose: dict[str, Any] | None = None) -> Any:
        e = self.roles.by_role.get(role)
        if not e:
            return None
        call = dict(e.get("fixed") or {})
        props = self.roles.props(e["tool"])
        for purpose, v in (by_purpose or {}).items():
            a = self.roles.arg(role, purpose)
            if not a:
                # the labels missed it: the one required argument of the matching type that is still unset
                want_type = "array" if isinstance(v, list) else ("integer" if purpose == "quantity" else "string")
                free = [x for x in self.roles.required(e["tool"]) if x not in call and not (props.get(x) or {}).get("enum")
                        and ((props.get(x) or {}).get("type") in (want_type, "number" if want_type == "integer" else want_type))]
                a = free[0] if len(free) == 1 else None
            if not a and purpose in NAMED_ARG:
                # an optional argument the labels skipped: its name says what it takes (an unlabelled search-words
                # argument would send every search empty)
                a = next((x for x, p in props.items() if x not in call and (p or {}).get("type") in ("string", None)
                          and not (p or {}).get("enum") and NAMED_ARG[purpose].search(x)), None)
            if not a and purpose == "remark":   # an optional note on the finishing call carries the reasons
                a = next((x for x, p in props.items() if x not in call and p.get("type") == "string" and not p.get("enum")
                          and re.search(r"note|explan|reason|remark|comment|summary|message|why", x, re.I)), None)
            if a:
                call[a] = v
        call.update(args or {})
        sent = self.roles.shape(e["tool"], call)
        obs = self.gw.tool(e["tool"], sent)
        if refused(obs) and role in ("cart_add", "cart_drop", "order", "give_up", "talk"):
            obs = self.repair(role, e["tool"], sent, obs)
        self.take(obs, role in ("inspect", "peek", "multi_inspect"))
        if role == "talk":
            # the shopper's own answer is not a store notice; only lines delivered beside it are
            said = set(direct_reply(obs))
            self.gw.unread = [x for x in self.gw.unread if x.strip() not in said]
        return obs

    def repair(self, role: str, tool: str, sent: dict[str, Any], obs: Any) -> Any:
        """A refused call that matters (cart, order, decline, message) is written again once by a model from the schema
        and the store's own complaint; an empty-basket fault is left to the order logic."""
        if self.fixes >= 4 or self.tight(CLOSE_S) or self.gw.steps_left() <= 1 or "NO_ITEMS" in dumps(obs, 400).upper():
            return obs
        self.fixes += 1
        res = self.gw.think(QUICK, FIX_PROMPT + "TOOL: " + dumps(self.roles.schemas.get(tool), 3000) + "\nSENT: " +
                            dumps(sent, 1500) + "\nREPLY: " + dumps(obs, 600) + f"\nMEANT: {role.replace('_', ' ')}", 700)
        args = res.get("args") if isinstance(res, dict) else None
        if not isinstance(args, dict):
            return obs
        self.log.append(f"repaired {tool}")
        again = self.gw.tool(tool, args)
        return again if again is not None else obs

    def take(self, obs: Any, full: bool, item: int | None = None) -> list[str]:
        ids = []
        for d in listing_dicts(obs):
            rec = to_listing(d)
            if not rec:
                continue
            old = self.listings.get(rec.id)
            if old:
                old.title = rec.title or old.title
                old.shelf = rec.shelf or old.shelf
                old.maker = rec.maker or old.maker
                # a short view (search row, several-ids summary) must not cut down the text a full view gave
                old.text = rec.text if len(rec.text) >= len(old.text) else old.text
                old.notes = rec.notes if len(rec.notes) >= len(old.notes) else old.notes
                if rec.price is not None:
                    old.price = rec.price
                if rec.stock is not None:
                    old.stock = rec.stock
                old.attrs.update(rec.attrs)
                old.raw.update(rec.raw)
                old.full = old.full or (full and bool(rec.attrs))
            else:
                rec.full = full and bool(rec.attrs)
                self.listings[rec.id] = rec
            ids.append(rec.id)
            if item is not None:
                self.found_for.setdefault(rec.id, set()).add(item)
        return ids

    # ---------------------------------------------------------------- set-up
    def start(self) -> None:
        q = self.gw.request + ("\n\n" + dumps(self.gw.opening) if self.gw.opening else "")
        labels, ledger = self.gw.think_all([(DEEP, LABEL_PROMPT + dumps(self.gw.schemas, 12000), 2600),
                                            (DEEP, LEDGER_PROMPT + q, 1500)])
        if not isinstance(ledger, dict) or not ledger.get("items"):
            ledger = self.gw.think(QUICK, LEDGER_PROMPT + q, 2600)
        if not isinstance(ledger, dict) or not ledger.get("items"):
            # a long multi-item message can outrun the reply cap or the clock: once more, shorter and with more time
            ledger = self.gw.think(QUICK[1:] + QUICK[:1], LEDGER_PROMPT.replace(', "quote": "the words it came from"', "") + q,
                                   3000, cap_s=70.0)
            self.log.append("ledger retry " + ("ok" if isinstance(ledger, dict) and ledger.get("items") else "failed"))
        self.roles.learn(labels if isinstance(labels, dict) else {})
        if not self.core_roles():   # labels missing or garbled: one more labelling pass on another model
            again = self.gw.think(QUICK, LABEL_PROMPT + dumps(self.gw.schemas, 12000), 2600)
            if isinstance(again, dict):
                self.roles = Roles(self.gw.schemas)
                self.roles.learn(again)
        self.plan = ledger if isinstance(ledger, dict) else {}
        first = (self.plan.get("items") or [None])[0]
        if isinstance(first, dict):   # outer keys that a misplaced bracket left inside the first item
            for k in ("budget", "budget_per_unit", "goal", "best_by", "in_stock", "message_limit", "step_limit",
                      "confirm_first", "rules", "silent_decline", "story"):
                if k in first and k not in self.plan:
                    self.plan[k] = first.pop(k)
        for it in self.plan.get("items") or []:
            if not isinstance(it, dict):
                continue
            kind = roots(str(it.get("what")) + " " + str(it.get("department")))
            needs = []
            for c in it.get("conditions") or []:
                ref = c.get("ref") if isinstance(c, dict) and isinstance(c.get("ref"), dict) and \
                    (str(c["ref"].get("seller") or "").strip() or c["ref"].get("with")) else None
                if not isinstance(c, dict) or not c.get("prop") or (c.get("value") in (None, "") and not ref):
                    continue
                prop = str(c["prop"])
                if re.search(r"\b(price|budget|cost|stock|availab|category|department|range)\b", prop.lower()):
                    continue
                if ref:   # the value is another listing's: looked up after the questions; until then it cannot block
                    need = Need(prop, "is", "")
                    ref["pair"] = ref.get("pair") is True
                    need.ref, need.off = ref, True
                    needs.append(need)
                    continue
                if prop.lower() in ("type", "kind", "product type") and roots(c["value"]) <= kind:
                    continue
                need = Need(prop, str(c.get("rel") or "is"), str(c["value"]))
                if is_code(need.value.strip()):
                    need.hidden = True   # a compatibility code must be named by the listing itself; silence is a miss
                needs.append(need)
            own = roots(str(it.get("what")) + " " + str(it.get("department"))) - FILLER
            # a search phrase carries the product's own words; "the same as ..." or a stray request fragment never does
            qs = [str(x) for x in (it.get("queries") or []) if tokens(x) and (not own or roots(x) & own)
                  and not re.search(r"\bsame as\b|\bcould you\b|\bwhen i\b|\bsend\b", str(x), re.I)]
            self.wants.append(Want(str(it.get("what") or "item"), str(it.get("department") or ""),
                                   qs or [" ".join(tokens(it.get("what"))[:3])], max(1, int(as_number(it.get("qty")) or 1)), needs))
        if not self.wants:
            self.wants = [Want("item", "", [" ".join(tokens(self.gw.request)[:3])], 1, [])]
        self.budget = as_number(self.plan.get("budget"))
        for amt, cents in money_said(self.gw.request):   # which unit the budget was read in: news is turned into it
            if self.budget and abs(amt - self.budget) < 1e-6:
                self.store_cents = cents
            elif self.budget and abs(amt / 100 - self.budget) < 1e-6 and cents:
                self.store_cents = False
            elif self.budget and abs(amt * 100 - self.budget) < 1e-6 and not cents:
                self.store_cents = True
        mc = as_number(self.plan.get("message_limit"))
        self.msg_cap = int(mc) if mc is not None else 3
        sl = as_number(self.plan.get("step_limit"))
        if sl and 3 <= sl < self.gw.step_cap:
            self.gw.step_cap = int(sl)

    def core_roles(self) -> bool:
        return all(self.roles.has(r) for r in ("order", "cart_add")) and \
            (self.roles.has("search") or self.roles.has("shelf")) and \
            (self.roles.has("inspect") or self.roles.has("multi_inspect"))

    @property
    def cheapest(self) -> bool:
        return self.plan.get("goal") == "cheapest"

    # ---------------------------------------------------------------- reserves
    def basket_msgs(self) -> int:
        # the race shopper answers a basket summary like any other message, so it spends a question; it is sent only
        # when the shopper asked to approve or hear the basket
        if not self.roles.has("talk") or not self.wants_basket():
            return 0
        return 2 if self.msg_cap >= 5 else 1

    def wants_basket(self) -> bool:
        rules = " ".join(str(r) for r in (self.plan.get("rules") or []))
        return bool(self.plan.get("confirm_first")) or bool(re.search(
            r"basket|cart|confirm|approv|before (?:you )?(?:order|buy|pay|check)|tell me|show me|let me (?:see|know)", rules, re.I))

    def close_steps(self) -> int:
        n = len(self.wants)
        # an add tool without a count takes one call per unit
        adds = n if self.roles.arg("cart_add", "quantity") else sum(max(1, int(as_number(w.qty) or 1)) for w in self.wants)
        if self.gw.step_cap < 14 * n:
            # few steps per item (leaders close a three-item order in ~5): the cart lines, one look at the cart, the
            # order and a spare; no room is held for re-reading every line
            return adds + 3 + (1 if self.roles.has("talk") else 0)
        return n + adds + 5 + (1 if self.roles.has("talk") else 0)   # one more for the cart view before each order

    def lean_close(self) -> int:
        """The fewest steps that still close an order: a call per cart line (or unit), a look at the cart, the order."""
        adds = len(self.wants) if self.roles.arg("cart_add", "quantity") else sum(max(1, int(as_number(w.qty) or 1)) for w in self.wants)
        return adds + 2

    def free_steps(self) -> int:
        questions = max(0, self.msg_cap - self.msgs - self.basket_msgs())
        return self.gw.steps_left() - self.close_steps() - questions

    def tight(self, s: float) -> bool:
        return self.gw.left() < s

    # ---------------------------------------------------------------- gathering
    def per_unit_cap(self, i: int) -> float | None:
        if not self.budget:
            return None
        if self.plan.get("budget_per_unit"):
            return self.budget
        rest = 0.0
        for j, w in enumerate(self.wants):
            if j != i:
                prices = [r.price for r in self.shelf_pool(j) if r.price is not None]
                rest += (min(prices) if prices else 0.0) * w.qty
        return max(0.0, self.budget - rest) / self.wants[i].qty

    @staticmethod
    def qkey(role: str, value: str, cap: float | None = None) -> str:
        """Identity of a listing query; word order never matters to a search."""
        return f"{role}:{' '.join(sorted(roots(value))) if role == 'search' else value.lower()}:{cap}"

    def gather(self, role: str, value: str, i: int, pages: int = 1, cap: float | None = None) -> list[str]:
        """One listing query (search words or a shelf name), following its cursor for up to `pages` pages."""
        key = self.qkey(role, value, cap)
        self.last_total = None
        if not value or key in self.done_queries or not self.roles.has(role):
            return []
        self.done_queries.add(key)
        purpose = {"search": "words", "shelf": "department", "maker": "brand"}[role]
        base: dict[str, Any] = {}
        if cap is not None and self.roles.arg(role, "price_max"):
            base[self.roles.arg(role, "price_max")] = int(cap) if float(cap).is_integer() else round(cap, 2)
        got: list[str] = []
        token = None
        page_arg = self.roles.arg(role, "cursor") or self.roles.arg(role, "page_number") or self.roles.arg(role, "offset")
        kind = self.roles.by_role[role]["args"].get(page_arg) if page_arg else None
        total = None
        if getattr(self, "light", False):
            pages = 1   # first look: one page per query until the shopper has been asked
        for p in range(pages):
            if self.free_steps() <= 0 or self.tight(70) or (self.item_limit and self.gw.steps >= self.item_limit):
                break
            args = dict(base)
            if p:
                if token is None or not page_arg:
                    break
                args[page_arg] = token
            obs = self.use(role, args, {purpose: value})
            ids = self.take(obs, False, i)
            if role == "search":
                for x in ids:
                    self.matched.setdefault(x, set()).update(roots(value))
            got += ids
            token, t, marked = cursor_state(obs)
            total = t if t is not None else total
            self.last_total = total
            prices = [self.listings[x].price for x in ids if x in self.listings]
            if p == 0 and role == "search" and (as_number(total) or 0) > max(60, 8 * len(ids)) and len(prices) >= 4 \
                    and None not in prices and any(a > b for a, b in zip(prices, prices[1:])):
                break   # a long list in no price order: later pages hold nothing a sharper query would not find first
            ptype = (self.roles.props(self.roles.by_role[role]["tool"]).get(page_arg) or {}).get("type") if page_arg else None
            if kind != "cursor" and ptype in ("integer", "number"):
                if isinstance(token, str) and not token.strip().isdigit():
                    token = None
                if token is None and not marked and total and ids and len(got) < total:
                    token = p + 2 if kind == "page_number" else len(got)
            if not ids or token is None:
                break
        if token is not None and page_arg:
            self.cursors.append({"role": role, "value": value, "cap": cap, "token": token, "item": i, "base": base,
                                 "purpose": purpose, "arg": page_arg, "total": total, "pages": 0})
        return got

    def more_pages(self, i: int, below: float | None) -> bool:
        """Read one more page of the item's most selective open listing (its words carry a wanted value, fewest hits)."""
        wanted = set()
        for n in self.wants[i].live() if 0 <= i < len(self.wants) else []:
            if n.rel != "not":
                wanted |= roots(n.value)
        open_ = [c for c in self.cursors if c["item"] == i and c["token"] is not None and c["pages"] < 3]
        open_.sort(key=lambda c: (0 if roots(c["value"]) & wanted else 1, c["total"] or 10 ** 9))
        for c in open_:
            if self.free_steps() <= 1 or self.tight(75):
                return False
            args = dict(c["base"])
            args[c["arg"]] = c["token"]
            obs = self.use(c["role"], args, {c["purpose"]: c["value"]})
            ids = self.take(obs, False, i)
            if c["role"] == "search":
                for x in ids:
                    self.matched.setdefault(x, set()).update(roots(c["value"]))
            c["pages"] += 1
            c["token"] = cursor_of(obs)[0] if ids else None
            if ids:
                prices = [self.listings[x].price for x in ids if self.listings[x].price is not None]
                if below is None or not prices or min(prices) < below:
                    return True
        return False

    def shelf_name(self, i: int) -> str | None:
        """The store's own department label for the item: the commonest shelf among its rows that the request names or
        that shares the product's words."""
        w = self.wants[i]
        named = roots(w.department)
        kind = roots(w.what) | named | {root(t) for q in w.queries for t in tokens(q)}
        tally: dict[str, int] = {}
        for rid, rec in self.listings.items():
            if i in self.found_for.get(rid, set()) and rec.shelf:
                tally[rec.shelf] = tally.get(rec.shelf, 0) + 1
        if not tally:
            return None

        def rank(s: str) -> tuple:
            r = roots(s)
            # between shelves sharing the product's words, the one adding fewer words of its own is the product's
            # ("mattresses", not "mattress pads, protectors & toppers")
            return (len(r & named), len(r & kind), -len(r - kind - FILLER), tally[s])

        best = max(tally, key=rank)
        # a shelf sharing no word with the product is where cheap look-alikes sit (shirts titled as sweaters), not its own
        return best if rank(best)[1] else None

    def shelf_pool(self, i: int) -> list[Listing]:
        w = self.wants[i]
        label = self.shelf_name(i)
        kind = roots(w.what) | roots(w.department) | {root(t) for q in w.queries for t in tokens(q)}
        out = []
        for rid, rec in self.listings.items():
            if i not in self.found_for.get(rid, set()) or rec.gone or rec.stock is False:
                continue
            if self.budget and rec.price is not None and rec.price * w.qty > self.budget + 1e-9:
                continue
            if label and rec.shelf and " ".join(tokens(rec.shelf)) != " ".join(tokens(label)):
                continue
            if not label and rec.shelf and not (roots(rec.shelf + " " + rec.title) & kind):
                continue
            out.append(rec)
        return out

    def explore(self, i: int) -> None:
        w = self.wants[i]
        cap = self.per_unit_cap(i)
        # each item gets its share of the exploring steps: the first item of a basket must not eat the others' searches
        left = len(self.wants) - i
        self.item_limit = self.gw.steps + max(4, self.free_steps() // left) if left > 1 else None
        try:
            self.explore_one(i, w, cap)
        finally:
            self.item_limit = None

    def explore_one(self, i: int, w: "Want", cap: float | None) -> None:
        if self.roles.has("search"):
            wanted = set()
            for n in w.live():
                if n.rel != "not":
                    wanted |= roots(n.value)
            # every word must match, so the product noun with every stated value (codes too) names a short list that
            # holds the fitting listings on its first page; the broad phrases follow only while that list is thin
            before = len(self.shelf_pool(i))
            self.sharp_search(i)
            sharp = len(self.shelf_pool(i)) - before >= 3
            for q in w.queries[:1 if sharp else 3]:   # a query that carries a wanted value is read to its last page
                self.gather("search", q, i, pages=4 if roots(q) & wanted else (2 if self.cheapest else 1))
                if len(self.shelf_pool(i)) >= 12 and not roots(q) & wanted:
                    break
            done = 0
            for n in w.live():   # each stated value with the store's own words for the product, read to the end
                if done >= 2 or n.rel not in ("is", "has", "any_of") or "," in n.value or not 0 < len(tokens(n.value)) <= 3:
                    continue
                kind = self.label_words(i)
                vt = [t for t in tokens(n.value) if root(t) not in {root(k) for k in kind}]
                if not names_something(vt):
                    continue
                done += 1
                if not self.gather("search", " ".join((vt + kind)[:6]), i, pages=4):
                    head = [self.noun(i)]
                    if not self.gather("search", " ".join(vt + head), i, pages=4) and len(vt) >= 2:
                        self.gather("search", " ".join(vt[:-1] + head), i, pages=4)   # every word must match: ease one
                cp = CODE_PARTS.fullmatch(n.value.strip()) if is_code(n.value.strip()) else None
                if cp and self.free_steps() > 4:
                    # a listing names its family or range ("55xx", "5500 to 5599", "series MH3") more often than the
                    # model itself: the maker prefix (with its series) and the product's words list those too
                    self.gather("search", " ".join([cp.group(1) + (cp.group(2) or "")] + (kind or [self.noun(i)])[:3]), i, pages=3)
            if len(self.shelf_pool(i)) < 4:
                for q in self.fallback_queries(i):
                    self.gather("search", q, i, pages=2)
                    if len(self.shelf_pool(i)) >= 6:
                        break
        if self.roles.has("shelf"):
            for name in [self.shelf_name(i), w.department, w.what]:
                if name and self.gather("shelf", name, i, pages=2 if self.cheapest else 1, cap=cap):
                    break
        if self.link.get("code") and i in self.link.get("items", []) and self.roles.has("search"):
            self.gather("search", f"{self.link['code']} {self.noun(i)}", i, pages=2)

    def sharp_search(self, i: int) -> None:
        """Every word must match, so the wanted values together with the product noun name a short list: revealed values
        first, then stated ones; a word is dropped from the end while the list comes back empty."""
        vals: list[str] = []
        for n in sorted(self.wants[i].live(), key=lambda x: not x.hidden):
            if n.rel in ("is", "has", "any_of") and "," not in n.value and 0 < len(tokens(n.value)) <= 3:
                vt = [t for t in tokens(n.value) if root(t) not in {root(k) for k in self.label_words(i)}]
                if names_something(vt):
                    vals += [t for t in vt if t not in vals]
        if not vals:
            return
        # the product's own words first ("knuckle bearing"), then its department label as the store files it (every
        # record carries its department, so the label words always match while they narrow the list)
        label = self.shelf_name(i)
        bases: list[list[str]] = []
        for src in (self.wants[i].what, label):
            b = [t for t in tokens(src) if t not in FILLER and len(t) > 1][-3:]
            if b and b not in bases:
                bases.append(b)
        bases = bases or [[self.noun(i)]]
        for base in bases:
            vv = [v for v in vals if root(v) not in {root(b) for b in base}][:4]
            words_ = vv + base
            while vv and len(words_) > len(base):
                key = self.qkey("search", " ".join(words_))
                if key in self.done_queries:
                    # the first look already ran it (one page): carry on down its list rather than easing a word
                    for c in self.cursors:
                        if c["role"] == "search" and self.qkey("search", c["value"], c["cap"]) == key:
                            while c["token"] is not None and c["pages"] < 2 and self.free_steps() > 2 and not self.tight(75):
                                args = dict(c["base"])
                                args[c["arg"]] = c["token"]
                                obs = self.use("search", args, {c["purpose"]: c["value"]})
                                ids = self.take(obs, False, i)
                                for x in ids:
                                    self.matched.setdefault(x, set()).update(roots(c["value"]))
                                c["pages"] += 1
                                c["token"] = cursor_of(obs)[0] if ids else None
                    return
                if self.gather("search", " ".join(words_), i, pages=3) or self.free_steps() <= 2:
                    return
                words_ = words_[:len(words_) - len(base) - 1] + base   # ease the last value, keep the product's name

    def label_words(self, i: int) -> list[str]:
        """The store's department label for the item as search words (the product noun until a label is known)."""
        label = self.shelf_name(i)
        words_ = [t for t in tokens(label) if t not in FILLER and len(t) > 1] if label else []
        return words_[:4] or [self.noun(i)]

    def noun(self, i: int) -> str:
        w = self.wants[i]
        for src in (w.queries[0] if w.queries else "", w.what, w.department):
            t = [x for x in tokens(src) if len(x) > 2]
            if t:
                return t[-1]
        return "item"

    def fallback_queries(self, i: int) -> list[str]:
        w = self.wants[i]
        out = [" ".join(t for t in tokens(w.department) if t not in FILLER and t != "range")[:60], self.noun(i)]
        for n in w.live():
            if n.rel in ("is", "has", "any_of") and 0 < len(tokens(n.value)) <= 2:
                out.insert(0, f"{n.value} {self.noun(i)}")
                break
        seen, res = set(), []
        for q in out:
            if q and q not in seen:
                seen.add(q)
                res.append(q)
        return res

    # ---------------------------------------------------------------- reading + scoring
    def batch_size(self) -> int:
        """Records per fact-sheet call: the cap learned from a refusal, else the schema's own list bound, else 3."""
        if self.batch_cap:
            return self.batch_cap
        e = self.roles.by_role.get("multi_inspect")
        bound = None
        for p in (self.roles.props(e["tool"]) if e else {}).values():
            if isinstance(p, dict) and p.get("type") == "array" and as_number(p.get("maxItems")):
                bound = int(as_number(p.get("maxItems")))
        return max(1, min(bound, 4)) if bound else 3   # stores refuse longer lists than the schema admits

    def read(self, ids: list[str], floor: int | None = None) -> int:
        """Fetch fact sheets; only records that actually come back count as read. Returns how many came back.
        `floor`: the steps that must stay (the closing reserve unless a leaner one is given)."""
        queue = [x for x in ids if x in self.listings and x not in self.unreadable]
        got = 0
        keep = self.close_steps() if floor is None else floor
        while queue:
            if self.gw.steps_left() <= keep or self.tight(CLOSE_S):
                break
            if not self.probed:
                before = self.gw.steps_left()
                self.probe_readers(queue[0])
                if self.gw.steps_left() < before:   # opened by the probe already, facts or not
                    got += int(self.listings[queue[0]].full)
                    queue = queue[1:]
                    continue
            if self.roles.has("multi_inspect"):
                chunk = queue[:self.batch_size()]
                obs = self.use("multi_inspect", None, {"product_ids": chunk})
            elif self.roles.has("inspect"):
                chunk = queue[:1]
                obs = self.use("inspect", None, {"product_id": chunk[0]})
            else:
                return got
            recs = [r for r in (to_listing(d) for d in listing_dicts(obs)) if r and r.id in chunk]
            back = {r.id for r in recs}
            if back and max(len(r.attrs) for r in recs) <= 3 and self.roles.has("peek") and not self.swapped \
                    and not self.roles.has("multi_inspect") and self.roles.by_role["peek"]["tool"] != self.roles.by_role["inspect"]["tool"]:
                # the "full record" tool answered with a short view: the labels may be crossed. One empty record is not
                # proof (a listing can lack attributes): the other tool must show more for the same listing first
                before = len(self.listings[chunk[0]].attrs)
                self.peeked.add(chunk[0])
                self.use("peek", None, {"product_id": chunk[0]})
                if len(self.listings[chunk[0]].attrs) > before + 3:
                    self.swapped = True
                    rb = self.roles.by_role
                    rb["inspect"], rb["peek"] = rb["peek"], rb["inspect"]
                    self.log.append("inspect/peek swapped")
                else:
                    self.swapped = True   # checked once: the labels stand
                    self.log.append("empty record kept " + chunk[0])
                    self.listings[chunk[0]].full = True
                    got += 1
                    queue = queue[1:]
                    continue
                continue
            if not back and len(chunk) > 1 and obs is not None:
                self.batch_cap = max(1, len(chunk) // 2)   # the store refused the list: send fewer at a time
                self.log.append(f"batch cap {self.batch_cap}")
                continue
            queue = queue[len(chunk):]
            for x in chunk:
                if x in back:   # a record that came back without attributes is still judged on what it shows
                    self.listings[x].full = True
                    got += 1
                else:
                    self.unreadable.add(x)
        self.prose_fill()
        return got

    def prose_fill(self) -> None:
        """A fully read record with (almost) no fields that states its properties in sentences: read the sentences
        out into fields (a few records per model call, the calls side by side), so questions, rules and checks see
        them. Records with a property table are left as they are."""
        todo = [r for r in self.listings.values() if r.full and r.id not in self.prosed and len(r.attrs) <= 3
                and stated_count(r) >= 2]
        if not todo or self.tight(CLOSE_S + 20):
            return
        chunks = [todo[k:k + 3] for k in range(0, len(todo), 3)][:5]
        jobs = [(QUICK, PROSE_PROMPT + dumps([{"id": r.id, "title": r.title, "text": r.text[:1200]} for r in c], 5000), 1600)
                for c in chunks]
        got = 0
        for c, res in zip(chunks, self.gw.think_all(jobs)):
            rows = res.get("records") if isinstance(res, dict) else None
            for r in c:   # an unread chunk is tried once more with the next read
                if isinstance(rows, list) or r.id in self.prose_missed:
                    self.prosed.add(r.id)
                self.prose_missed.add(r.id)
            if not isinstance(rows, list):
                continue
            for row in rows:
                r = self.listings.get(str(row.get("id"))) if isinstance(row, dict) else None
                facts = row.get("facts") if r else None
                if r and isinstance(facts, dict):
                    for k, v in facts.items():
                        if isinstance(v, (str, int, float)) and not isinstance(v, bool) and 0 < len(str(v)) <= 80 and tokens(k):
                            r.attrs.setdefault(" ".join(tokens(k)), str(v))
                            got += 1
        self.log.append(f"prose facts {got} from {sum(len(c) for c in chunks)}")

    def probe_readers(self, rid: str) -> None:
        """Once per episode, when the store has more than one way to read a record: read the same listing through each
        and keep the one that returns the most attributes as the record reader. The labels say what a tool claims; the
        records say what it gives (a several-ids view can be a short summary without the property table, and a "detail"
        view can hold stock and notes but no properties)."""
        self.probed = True
        rb = self.roles.by_role
        tools: dict[str, dict[str, Any]] = {}
        for role in ("inspect", "peek", "multi_inspect"):
            e = rb.get(role)
            if e and e["tool"] not in {x["tool"] for x in tools.values()}:
                tools[role] = e
        labelled = {e["tool"] for e in rb.values()}
        for name, s in self.roles.schemas.items():   # a viewer the labels left without a role
            req = self.roles.required(name)
            words_ = (str(name) + " " + str(s.get("description") or "")).lower()
            if name not in labelled and len(req) == 1 and re.search(r"id|code|ref|sku|item|product|listing", req[0].lower()) \
                    and READER_WORDS.search(words_) and not CART_WORDS.search(words_):
                rb["probe:" + name] = {"tool": name, "args": {req[0]: "product_id"}, "fixed": {}, "sets_line": False}
                tools["probe:" + name] = rb["probe:" + name]
        if len(tools) < 2:
            return
        size: dict[str, int] = {}
        said: dict[str, int] = {}
        for role in tools:
            if self.gw.steps_left() <= self.close_steps() + 1 or self.tight(CLOSE_S):
                break
            if size.get("inspect", -1) > 2 and role != "multi_inspect":
                continue   # the record reader gave a real table: other single views would only cost steps
            obs = self.use(role, None, {"product_ids": [rid]} if role == "multi_inspect" else {"product_id": rid})
            recs = [r for r in (to_listing(d) for d in listing_dicts(obs)) if r and r.id == rid]
            size[role] = max((len(r.attrs) for r in recs), default=-1)
            said[role] = max((stated_count(r) for r in recs), default=0)
        if len(size) < 2:
            if size.get("inspect", -1) > 2:
                self.listings[rid].full = True
                for r in [r for r in tools if r.startswith("probe:")]:
                    rb.pop(r, None)
            return
        best = max(size.values())
        self.probe_tries += 1
        if best <= 2 and self.probe_tries < 3:   # a listing without a property table tells the tools apart by nothing
            self.probed = False
            for r in [r for r in size if r.startswith("probe:")]:
                rb.pop(r, None)
            self.log.append(f"readers inconclusive on {rid}")
            if best > 0:
                self.listings[rid].full = True
            return
        self.swapped = True   # the reader is settled by what came back, not by the labels
        self.log.append("readers " + ", ".join(f"{r}={tools[r]['tool']}:{n}/{said.get(r, 0)}" for r, n in size.items()))
        if best <= 2 and "multi_inspect" in size and len(size) > 1 and \
                said.get("multi_inspect", 0) <= max(v for r, v in said.items() if r != "multi_inspect"):
            # no tool gave a table on three listings: a several-ids view stating no more than one listing's own view is a
            # summary (the properties may sit in sentences there, read out by prose_fill)
            rb.pop("multi_inspect")
            size.pop("multi_inspect")
            self.log.append("several-ids view dropped: no more than the single view")
        if best > 0:
            self.listings[rid].full = True
        if "multi_inspect" in size and size["multi_inspect"] < best - 2:
            rb.pop("multi_inspect")
            self.log.append("several-ids view dropped: short records")
        if "peek" in size and size["peek"] > size.get("inspect", -1) + 2:
            if "inspect" in rb:
                rb["inspect"], rb["peek"] = rb["peek"], rb["inspect"]
            else:
                rb["inspect"] = rb["peek"]
            size["inspect"], size["peek"] = size["peek"], size.get("inspect", -1)
        extra = [r for r in size if r.startswith("probe:")]
        top = max(extra, key=lambda r: size[r]) if extra else None
        if top and size[top] > max(size.get("inspect", -1), size.get("peek", -1)) + 2:
            if "inspect" in rb and "peek" not in rb:
                rb["peek"] = rb["inspect"]   # the old reader still shows what the new one leaves out
            rb["inspect"] = dict(rb[top])
        for r in extra:
            rb.pop(r, None)
        if "inspect" in rb:
            self.log.append("record reader = " + rb["inspect"]["tool"])

    def look_closer(self, ids: list[str], closing: bool = False) -> int:
        """Read records through the short-view tool as well, when that is a separate tool: some stores keep stock and a
        listing's own notes (where a reference hands a property on) only there. Returns how many were read."""
        p, q = self.roles.by_role.get("peek"), self.roles.by_role.get("inspect")
        if not p or (q and p.get("tool") == q.get("tool")):
            return 0
        got = 0
        for rid in ids:
            # while closing, the reserve is what pays for this look; otherwise the reserve is kept whole
            if rid in self.peeked or rid not in self.listings or self.tight(CLOSE_S) or \
                    self.gw.steps_left() <= (3 if closing else self.close_steps() + 1):
                continue
            self.peeked.add(rid)
            self.use("peek", None, {"product_id": rid})
            got += 1
        return got

    def search_matches_all(self) -> bool:
        """Whether a listed record holds every word of the search that listed it (an all-words store matches the full
        record, so the words stand in for values a short row does not show); judged on the records read so far."""
        agree = differ = 0
        for rid, ws in self.matched.items():
            r = self.listings.get(rid)
            if not r or not r.full or not ws:
                continue
            have = roots(r.blob())
            if len(ws & have) >= 0.8 * len(ws):
                agree += 1
            else:
                differ += 1
        return agree + 1 >= 2 * differ

    def order_to_read(self, i: int) -> list[Listing]:
        wanted = set()
        groups = []
        for n in self.wants[i].live():
            if n.rel != "not":
                wanted |= roots(n.value)
                if roots(n.value):
                    groups.append(roots(n.value))
        code = self.link.get("code") if i in self.link.get("items", []) else None
        trust = self.search_matches_all()

        def key(r: Listing) -> tuple:
            text = roots(r.title + " " + r.text + " " + " ".join(r.attrs.values()))
            if trust:
                text |= self.matched.get(r.id, set())
            met = sum(1 for g in groups if g <= text)
            hit = len(wanted & text) + (3 if code and code.lower() in r.blob().lower() else 0)
            price = r.price if r.price is not None else float("inf")
            # rows that meet the most wishes (shown, or matched by a search for them) go first, cheapest first; the
            # price ceiling in verify() then sweeps cheaper rows
            # a row naming the item's own kind ("valve seal" on a seals shelf) is read before one sharing only its shelf
            kind = 0 if self.kind_named(i, r) else 1
            return (kind, -met, 0 if hit else 1, price, -hit) if self.cheapest else (kind, -met, -hit, price)

        return sorted([r for r in self.shelf_pool(i) if not r.full and r.id not in self.unreadable
                       and (i, r.id) not in self.marks], key=key)

    def rules_for(self, i: int) -> list[Rule]:
        needs = self.wants[i].live()
        recs = [r for r in self.shelf_pool(i) if r.full and r.attrs]
        samples: dict[str, set[str]] = {}
        for r in recs:
            for k, v in r.attrs.items():
                if v not in ("", None):
                    samples.setdefault(k, set()).add(str(v)[:60])
        size = sum(len(v) for v in samples.values())
        old = self.rules.get(i)
        if old and len(old[1]) == len(needs) and all(a.need is b for a, b in zip(old[1], needs)) and \
                (old[0] >= size * 0.7 or self.tight(CLOSE_S + 25)):
            return old[1]
        if not needs:
            self.rules[i] = (size, [])
            return []
        if self.tight(CLOSE_S + 25) or not samples:
            made = [Rule(n, [], [], []) for n in needs]
            if not samples:
                return made
            self.rules[i] = (0, made)
            return made
        rel_words = set().union(*(roots(n.prop) for n in needs))
        keys = sorted(samples, key=lambda k: (-len(roots(k) & rel_words), -len(samples[k])))[:40]
        table = {k: sorted(samples[k])[:8] for k in keys}
        conds = [{"n": k + 1, "prop": n.prop, "rel": n.rel, "value": n.value} for k, n in enumerate(needs)]
        res = self.gw.think(QUICK, MAP_PROMPT + dumps(conds, 1200) + "\nFIELDS:\n" + dumps(table, 6000), 1800)
        maps = {int(as_number(m.get("n")) or 0): m for m in (res.get("map") or []) if isinstance(m, dict)} \
            if isinstance(res, dict) else {}
        made = []
        for k, n in enumerate(needs):
            m = maps.get(k + 1) or {}
            ru = Rule(n, [str(x) for x in m.get("fields") or []], [str(x) for x in m.get("yes") or []],
                      [str(x) for x in m.get("no") or []])
            if n.rel in ("is", "has"):
                # the store spells the wanted value itself: only that spelling counts, not a neighbouring value
                want = roots(n.value)
                seen = [roots(v) for r in recs for v in ru.field_hits(r)]
                if want and any(v == want for v in seen):
                    ru.yes = [want]
                else:
                    # words naming the product kind itself are met by the shelf the listing sits on
                    w = self.wants[i]
                    rest = want - (roots(w.what) | roots(w.department) | roots(self.shelf_name(i) or ""))
                    if rest and rest != want:
                        ru.yes.append(rest)
            made.append(ru)
        if not maps:   # the mapping call failed: judge with these for now, but map again next time
            return made
        self.rules[i] = (size, made)
        return made

    def score(self, i: int) -> None:
        rules = self.rules_for(i)
        code = self.link.get("code") if i in self.link.get("items", []) else None
        kind = roots(self.wants[i].what) | roots(self.wants[i].department)

        def fielded(r: Listing, ru: Rule) -> bool:
            p = roots(ru.need.prop) - kind - BROAD or roots(ru.need.prop)
            return any(p and p <= roots(plain_key(k)) for k in r.attrs)

        full = [r for r in self.shelf_pool(i) if r.full]
        carried = [sum(1 for r in full if fielded(r, ru)) for ru in rules]
        for r in self.shelf_pool(i):
            if not r.full:
                continue
            res = [ru.check(r) for ru in rules]
            # the store grades a value by the field named for it: where fellow listings carry that field, a value seen
            # only in the title or text is not shown, and the record waits behind those that state it
            res = ["maybe" if x == "yes" and ru.need.rel in ("is", "has", "any_of") and carried[k] >= 2 and not fielded(r, ru)
                   else x for k, (ru, x) in enumerate(zip(rules, res))]
            res = ["no" if (x == "unknown" and self.silent(i, r, ru)) or (x != "no" and self.contradicts(r, ru))
                   or (x != "no" and self.variant_clash(r, ru)) else x for ru, x in zip(rules, res)]
            failed = [ru.need.prop for ru, x in zip(rules, res) if x == "no"]
            if code and r.id != self.link.get("anchor_id") and not re.search(
                    r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])", r.blob(), re.I):
                failed.append("series " + code)
            if (i, r.id) in self.struck:
                failed.append(self.struck[(i, r.id)])
            state = "no" if failed else ("yes" if all(x == "yes" for x in res) else "maybe")
            self.marks[(i, r.id)] = {"state": state, "failed": failed, "plain": self.plainly(i, r, rules),
                                     "open": sum(1 for x in res if x != "yes"), "exact": self.exactly(r, rules)}

    def silent(self, i: int, r: Listing, ru: Rule) -> bool:
        """A record read with its fields, none named for the property, and the wanted value written nowhere in it: the
        store does not count it as having that value."""
        if ru.need.rel not in ("is", "has", "any_of", "min", "max") or not r.attrs:
            return False
        kind = roots(self.wants[i].what) | roots(self.wants[i].department)
        p = roots(ru.need.prop) - kind - BROAD or roots(ru.need.prop)
        if any(p == (roots(plain_key(k)) - kind - BROAD) for k in r.attrs):
            return False
        said = " ".join([r.title, r.text, " ".join(str(v) for v in r.attrs.values())])
        have = roots(said) | split_figures(said)
        if ru.need.rel in ("min", "max"):   # a bound the record gives no figure for, under any name
            want = as_number(ru.need.value)
            return want is not None and not any(as_number(t) == want for t in re.findall(r"\d+(?:\.\d+)?", " ".join(have)))
        options = self.options(ru)
        return bool(options) and not any(o <= have or split_figures(" ".join(sorted(o))) <= have for o in options)

    @staticmethod
    def options(ru: Rule) -> list[set[str]]:
        return [roots(o) for o in re.split(r",|/|;|\bor\b", ru.need.value) if roots(o)]

    def contradicts(self, r: Listing, ru: Rule) -> bool:
        """A field named exactly for the property states something else, and no field about it (the exact name or a more
        specific one, e.g. "skirt style" for style) agrees."""
        if ru.need.rel not in ("is", "any_of"):
            return False
        p = roots(ru.need.prop)
        options = self.options(ru)
        if not p or not options or not any(roots(plain_key(k)) == p for k in r.attrs):
            return False
        about = [str(v) for k, v in r.attrs.items() if p <= roots(plain_key(k)) and not roots(plain_key(k)) & VARIANT_WORDS]
        # "single" agrees with "single can"; "4g" does not agree with "wireless n"; "250ml" agrees with "250 ml"
        return not any(sv and (so <= sv or sv <= so) for v in about for o in options
                       for sv, so in ((roots(v), o), (split_figures(v), split_figures(" ".join(sorted(o))))))

    @staticmethod
    def variant_clash(r: Listing, ru: Rule) -> bool:
        """The listing's own menu of orderable variants states another count of the wanted thing ("toyota hiace /
        2pcs/1set" against a wanted "1 piece"): the order would be that variant, whatever another field says."""
        if ru.need.rel not in ("is", "any_of", "has"):
            return False
        want = counts(ru.need.value)
        if len(want) != 1:
            return False
        n, w = want[0]
        menu = [c for k, v in r.attrs.items() if roots(plain_key(k)) & VARIANT_WORDS for c in counts(v)]
        alike = [m for m, u in menu if same_word(u, w)]
        return bool(alike) and n not in alike

    def kind_named(self, i: int, r: Listing) -> bool:
        """The record names the item's own words beyond its department's name ("valve seal" on a bearings, bushings
        and seals shelf); with no such words every record does."""
        own = roots(self.wants[i].what) - roots(r.shelf) - roots(self.wants[i].department) - roots(" ".join(FILLER | COUNT_WORDS))
        return not own or own <= roots(" ".join([r.title, r.text, " ".join(f"{k} {v}" for k, v in r.attrs.items())]))

    def exactly(self, r: Listing, rules: list[Rule]) -> int:
        """How many wanted values sit under a field named exactly for their property, worded no broader than wanted
        ("sweet" or "single" count for "sweet" / "single can"; "sweet and fresh" does not count for "sweet")."""
        n = 0
        for ru in rules:
            p = roots(ru.need.prop)
            if ru.need.rel in ("is", "has", "any_of") and any(
                    p and roots(plain_key(k)) == p and roots(v) and any(roots(v) <= o for o in self.options(ru))
                    for k, v in r.attrs.items()):
                n += 1
        return n

    def plainly(self, i: int, r: Listing, rules: list[Rule]) -> bool:
        """Every wanted value is stated under a field named for its property (the product's own words aside)."""
        kind = roots(self.wants[i].what) | roots(self.wants[i].department)
        for ru in rules:
            if ru.need.rel not in ("is", "has", "any_of"):
                continue
            head = re.split(r"\bof\b", ru.need.prop, maxsplit=1)[0]   # "type of top" is about the type
            p = roots(head) - kind or roots(ru.need.prop)
            if not any(p & roots(k) and ru.says(v, ru.yes) for k, v in r.attrs.items()):
                return False
        return True

    def ranked(self, i: int) -> list[tuple[int, Listing]]:
        out = []
        for r in self.shelf_pool(i):
            m = self.marks.get((i, r.id))
            if not m or m["state"] == "no":
                continue
            # the listing a wish takes its value from ("go by the one from seller X") is another product: a fallback only
            out.append((0 if m["state"] == "yes" and r.id not in self.ref_src else 1, r))
        named = {r.id: self.kind_named(i, r) for _, r in out}
        # among fits, cheapest first when asked; otherwise a record that states each value under its own name first
        # a "maybe" that confirms more of the conditions comes before one that confirms fewer; without a cheapest rule,
        # values stated under fields named exactly for their properties come first
        # a fit naming the item's own kind comes before one that only shares its department
        out.sort(key=lambda x: (x[0], not named[x[1].id], self.marks[(i, x[1].id)].get("open", 0) if x[0] else 0,
                                0 if self.cheapest else -self.marks[(i, x[1].id)].get("exact", 0),
                                x[1].price if (self.cheapest and x[1].price is not None) else (float("inf") if self.cheapest else
                                (0 if self.marks[(i, x[1].id)].get("plain") else 1))))
        return out

    def verify(self, i: int, want: int, cap_reads: int | None = None, plain: bool = False) -> None:
        reads = 0
        while not self.tight(70):
            self.score(i)
            fits = [r for c, r in self.ranked(i) if c == 0 and (not plain or self.marks[(i, r.id)].get("plain"))]
            if any(self.kind_named(i, r) for r in self.shelf_pool(i)):
                fits = [r for r in fits if self.kind_named(i, r)]   # the shelf shows the item's own kind: others do not count
            todo = self.order_to_read(i)
            if len(fits) >= want:
                # enough fits; for "cheapest" keep sweeping unread rows priced under the best one
                best = min((r.price for r in fits if r.price is not None), default=None)
                strong = [r for r in todo if self.shows_all(i, r)]
                todo = [r for r in todo if self.cheapest and best is not None and r.price is not None and r.price < best]
                told = any(x.hidden or x.ref for x in self.wants[i].needs)
                # (five free steps stay for the news a cart add brings and a swap: a held line can sell out)
                if not todo and self.race and told and strong and reads < self.read_share(i) and self.free_steps() > 5:
                    # race worlds: one of several fits is the right one, and the shopper's firm wish (or a code or a
                    # reference) is what tells them apart: every unread row showing all the stated values is read
                    # against it while this item's share of the steps lasts. Without such a wish more reads only add
                    # look-alike fits, and the first fit found is as good a pick as any
                    todo = strong
                if not todo and self.cheapest and best is not None and self.band(i, best):
                    continue   # the department listed under that price showed new cheaper rows
                if not todo or self.free_steps() <= 2:
                    return
            elif self.cheapest and len(fits) >= 1:
                ceiling = fits[min(len(fits), want) - 1].price
                todo = [r for r in todo if ceiling is None or (r.price or 0) <= ceiling]
            free = self.free_steps() - (5 if self.race and fits else 0)   # race: steps for the news after an add
            room = free if cap_reads is None else min(free, cap_reads - reads)
            if room <= 0:
                break
            if not todo:
                if self.cheapest and fits and self.more_pages(i, fits[0].price):
                    reads += 1   # a page costs a step like a read
                    continue
                if not self.widen(i):
                    break
                continue
            per = self.batch_size() if self.roles.has("multi_inspect") else min(3, room)
            batch = todo[:max(1, min(per, room))]
            self.read([r.id for r in batch])
            reads += len(batch)
        self.score(i)

    def shows_all(self, i: int, r: Listing) -> bool:
        """The row shows (or a search for them matched) every stated value of the item."""
        groups = [roots(n.value) for n in self.wants[i].live() if n.rel != "not" and roots(n.value)]
        if not groups:
            return False
        text = roots(r.title + " " + r.text + " " + " ".join(r.attrs.values()))
        if self.search_matches_all():
            text |= self.matched.get(r.id, set())
        return all(g <= text for g in groups)

    def read_share(self, i: int) -> int:
        """Reads this item may still take: its share of the free steps among the items not yet settled."""
        return max(2, self.free_steps() // max(1, len(self.wants) - i))

    def band(self, i: int, below: float) -> bool:
        """List the item's department under a price: everything cheaper than the best fit, narrowing the cap while the
        listing reports more rows than it shows. True when new rows arrived."""
        label = self.shelf_name(i)
        if not label or not self.roles.has("shelf") or not self.roles.arg("shelf", "price_max") or \
                (i, below) in self.banded or self.free_steps() <= 5 or self.tight(90):
            return False
        self.banded.add((i, below))
        before = {r.id for r in self.shelf_pool(i)}
        cap = below - 0.01
        for _ in range(3):
            got = self.gather("shelf", label, i, pages=3, cap=round(cap, 2))
            total = self.last_total
            if not got or total is None or total <= len(got):
                break
            prices = sorted(self.listings[x].price for x in got if self.listings[x].price is not None)
            if not prices or self.free_steps() <= 4:
                break
            # the rows shown are a sample of all listed under the cap: the price at the shown/listed share of that
            # sample is where a cap would list about as many rows as can be shown
            k = int(len(prices) * len(got) / max(total, 1)) + 1
            new = prices[max(0, min(k, len(prices) - 1))]
            if new >= cap:
                break
            cap = new
        return bool({r.id for r in self.shelf_pool(i)} - before)

    def widen(self, i: int) -> bool:
        """New candidate rows from value searches or a deeper shelf read; True if any arrived."""
        if self.free_steps() <= 1 or self.tight(75):
            return False
        before = {r.id for r in self.shelf_pool(i)}
        tries = 0
        if self.roles.has("search"):
            for n in self.wants[i].live():
                if n.rel not in ("is", "has", "any_of"):
                    continue
                vw = [t for t in tokens(n.value) if len(t) > 1][:3]
                if not names_something(vw):
                    continue   # a yes/no value names nothing to search for
                for q in (" ".join(vw + self.label_words(i)), " ".join(vw + [self.noun(i)]), " ".join(vw)):
                    if q and self.qkey("search", q) not in self.done_queries:
                        self.gather("search", q, i, pages=4)
                        tries += 1
                if {r.id for r in self.shelf_pool(i)} - before or tries >= 2:
                    break
        if not ({r.id for r in self.shelf_pool(i)} - before):
            for q in self.fallback_queries(i):
                if self.qkey("search", q) not in self.done_queries:
                    self.gather("search", q, i, pages=1)
                    break
        if not ({r.id for r in self.shelf_pool(i)} - before) and self.roles.has("shelf") and self.shelf_name(i):
            self.gather("shelf", self.shelf_name(i), i, pages=3)
        if not ({r.id for r in self.shelf_pool(i)} - before):
            self.more_pages(i, None)
        return bool({r.id for r in self.shelf_pool(i)} - before)

    # ---------------------------------------------------------------- interview
    def properties(self, i: int) -> list[str]:
        """Properties worth asking about: carried by most read records of the shelf, with a few repeating values."""
        stated = {" ".join(tokens(n.prop)) for n in self.wants[i].needs}
        recs = [r for r in self.shelf_pool(i) if r.full and r.attrs]
        stats: dict[str, dict[str, Any]] = {}
        for r in recs:
            for k, v in r.attrs.items():
                nk = plain_key(k)
                if not nk or re.search(r"\b(id|sku|code|price|stock|title|name|brand|seller|shop|url|image|rating|review)\b", nk):
                    continue
                if any(nk == s or nk in s or s in nk for s in stated if s) or roots(nk) & (VARIANT_WORDS | CATCH_ALL):
                    continue   # a menu of orderable variants or an umbrella name is not a property a shopper holds a wish about
                st = stats.setdefault(nk, {"n": 0, "vals": {}, "shown": k, "measured": 0})
                st["n"] += 1
                st["measured"] += bool(MEASURE.match(str(v)))
                vv = " ".join(tokens(v))
                st["vals"][vv] = st["vals"].get(vv, 0) + 1
        keep = []
        for nk, st in stats.items():
            vals = st["vals"]
            avg = sum(len(v) for v in vals) / max(1, len(vals))
            coded = sum(c for v, c in vals.items() if re.search(r"\d", v)) / max(1, st["n"])
            measure = st["measured"] * 2 >= st["n"]   # a figure with its unit (a depth, a capacity), not a model code
            if avg > 40 or (not measure and ((st["n"] >= 4 and max(vals.values()) == 1) or coded >= 0.5)):
                continue
            if any(re.search(r"\s/\s|\w:\w", v) for v in vals) and sum(1 for v in vals if re.search(r"\s/\s|\w:\w", v)) * 2 >= len(vals):
                continue
            if st["n"] < max(1, len(recs) // 4):
                continue
            # a measure is a wish less often than a look or a kind: it waits behind them unless clearly more common
            keep.append(((st["n"] / max(1, len(recs)) * 2 + min(0.6, 0.12 * (len(vals) - 1))) * (0.8 if measure else 1.0), nk, measure))
        keep.sort(reverse=True)
        # one property often goes by two names across sellers ("finish" / "type of finish", "wrapping type" /
        # "wrapping"); the shopper knows it by one of them, so both are named together (one subject, no complaint)
        kind = roots(self.wants[i].what) | roots(self.wants[i].department)
        def core(nk: str) -> frozenset[str]:
            c = roots(nk) - BROAD - {"filter"}
            return frozenset(c - kind or c or roots(nk))
        names: dict[frozenset[str], list[str]] = {}
        for _, nk, _m in keep:
            names.setdefault(core(nk), []).append(nk)
        groups: dict[frozenset[str], float] = {}
        for sc, nk, _m in keep:   # keep is sorted: the first name of a group carries its best score
            groups.setdefault(core(nk), sc)
        out: list[str] = []
        measured = 0
        for c in sorted(groups, key=lambda g: -groups[g]):
            # a name repeating the product's own word ("type of lamp shade" for lamps) is said last
            nks = sorted(names[c], key=lambda nk: (bool(roots(nk) & kind), -stats[nk]["n"]))
            slot = " / ".join(nks[:2])
            if any(nk in o.split(" / ") for nk in nks for o in out):
                continue
            if stats[nks[0]]["measured"] * 2 >= stats[nks[0]]["n"]:
                if measured >= 3:   # three measures at most: a shelf lists many, and they would crowd out the looks and kinds
                    continue
                measured += 1
            out.append(slot)
            if stats[nks[0]]["measured"] * 2 >= stats[nks[0]]["n"]:
                self.measures.setdefault(i, set()).add(slot)
        self.samples[i] = {slot: [v for v, _ in sorted(stats[slot.split(" / ")[0]]["vals"].items(), key=lambda t: -t[1])[:3]]
                           for slot in out}
        asked = self.asked.get(i, [])
        return [p for p in out if not any(n == a or n in a or a in n for n in p.split(" / ") for a in asked)]

    def question_order(self, i: int) -> list[str]:
        props = self.properties(i)
        if len(props) <= 1 or self.tight(120):
            lead = [p for p in self.unheard.get(i, []) if p in props]
            return lead + [p for p in props if p not in lead]
        if i not in self.ask_order:
            story = [str(s) for s in (self.plan.get("story") or [])][:6]
            prompt = (PICK_PROMPT + self.gw.request[:2500] + "\nPRODUCT: " + self.wants[i].what +
                      "\nALREADY ASKED (answer): " + dumps(self.heard_answers[-4:], 700) + "\nALREADY STATED: " +
                      dumps([n.prop for n in self.wants[i].needs], 300) + "\nSTORY HINTS: " + dumps(story, 300) +
                      "\nPROPERTIES (a few values each): " +
                      dumps({p: [v[:30] for v in self.samples.get(i, {}).get(p, [])] for p in props[:10]}, 1200))
            res = self.gw.think(DEEP, prompt, 900)
            self.ask_order[i] = [str(p) for p in (res.get("ask") or [])] if isinstance(res, dict) else []
        lead = [p for p in self.unheard.get(i, []) if p in props]
        first = lead + [p for p in self.ask_order[i] if p in props and p not in lead]
        order = first + [p for p in props if p not in first]
        # one measure among the early questions: firm wishes are mostly looks and kinds, and messages are few
        spent = any(p in self.measures.get(i, set()) for p in self.asked.get(i, []))
        late = [p for p in order if p in self.measures.get(i, set())][0 if spent else 1:]
        return [p for p in order if p not in late] + late

    def interview(self) -> None:
        allowance = self.msg_cap - self.basket_msgs()
        n = len(self.wants)
        turn = 0
        while self.msgs < allowance and turn < allowance + 2 and not self.tight(100) and self.roles.has("talk"):
            turn += 1
            if n == 1 and any(x.ref and x.hidden for x in self.wants[0].needs):
                break   # the item's firm wish points at a trail: its steps go to walking it, not to more questions
            # an item whose firm wish is already heard rarely holds a second: the questions go to the others first
            # (a compatibility code from the request is held as strictly as a heard wish, but nobody said it in reply)
            open_ = [j for j in range(n) if not any(x.hidden and not is_code(x.value.strip())
                                                    for x in self.wants[j].needs)] or list(range(n))
            i = min(open_, key=lambda j: (len(self.asked.get(j, [])), j)) if n > 1 else 0
            props = self.question_order(i)
            if not props:
                others = [j for j in range(n) if j != i and self.question_order(j)]
                if not others:
                    break
                i = others[0]
                props = self.question_order(i)
            # one question names several properties; look-alike names share one slot, so neither pushes the next
            # property back while the shopper's own word for it is still said.
            # The shopper speaks only to the first named subject on their list: a firm answer clears the properties up
            # to the one it names; a shrug means that subject was a decoy, so only the first property is surely
            # settled and the rest go first next time; a confused reply means none of them was on the list
            own = roots(self.wants[i].what) | KIN_IGNORE
            group, kin = [], {}
            width = 1 if self.one_each else ASK_GROUP
            for p in props:
                twin = width > 1 and next((g for g in group if alike(roots(p) - own, roots(g) - own)), None)
                if twin:
                    kin.setdefault(twin, []).append(p)
                    continue
                group.append(p)
                if len(group) == width:
                    break
            name = re.sub(r"^(?:a|an|the|some|one|any)\s+", "", self.wants[i].what.strip(), flags=re.I) or "item"
            slots = ["/".join([p] + kin.get(p, [])[:1]) for p in group]
            listed = slots[0] if len(slots) == 1 else ", ".join(slots[:-1]) + " or " + slots[-1]
            text = f"Do you have a preference for the {listed} of the {name}?"
            self.gw.drain()
            obs = self.use("talk", None, {"message": text})
            self.msgs += 1
            self.gw.drain()
            before = len(self.wants[i].needs)
            reply = " ".join(direct_reply(obs))
            cleared = group
            self.unheard.pop(i, None)
            if self.hear(text, obs, i):
                said = roots(reply) | {r for n in self.wants[i].needs[before:] for r in roots(n.prop)}
                hit = next((k for k, p in enumerate(slots) if roots(p) & said), None)
                if hit is not None:
                    cleared = group[:hit + 1]
            elif self.heard_meaning == "too_many" and len(group) > 1:
                # the message touched two or more of the shopper's subjects: nothing was answered, ask them singly
                # (a lone property that draws this is a broad name like "features" spanning several: let it go)
                self.one_each = True
                cleared = []
                self.unheard[i] = group
            elif self.heard_meaning == "indifferent" and len(group) > 1:
                cleared = group[:1]
                self.unheard[i] = group[1:]
            self.asked.setdefault(i, []).extend(cleared + [t for p in cleared for t in kin.get(p, [])])
            self.heard_answers.append((listed, reply[:160]))
            self.ask_order.pop(i, None)   # plan the next question knowing this answer

    def hear(self, text: str, obs: Any, i: int) -> bool:
        """Read the shopper's answer to our message; a firm wish becomes a condition. True when one was added."""
        self.heard_meaning = ""
        reply = " ".join(direct_reply(obs)) or " ".join(voice_lines(obs)) or dumps(obs, 500)
        if not reply.strip() or self.tight(CLOSE_S + 15):
            return False
        prompt = ANSWER_PROMPT.replace("{q}", text[:600]).replace("{a}", reply[:600])
        res = self.gw.think(QUICK, prompt, 400)
        asked = roots(text) - roots(self.wants[i].what) - roots("do you have a preference for the of")
        if (not isinstance(res, dict) or res.get("meaning") in ("confused", "other")) and \
                (INSIST.search(reply) or asked & roots(reply)) and not self.tight(CLOSE_S + 25):
            # an answer naming the asked property (or insisting) read as confusion: a second reader decides
            again = self.gw.think(QUICK[1:] + QUICK[:1], prompt, 400)
            if isinstance(again, dict) and again.get("meaning") == "wish":
                res = again
        self.heard_meaning = str(res.get("meaning") or "") if isinstance(res, dict) else ""
        if not isinstance(res, dict) or res.get("meaning") != "wish":
            return False
        added = False
        for wsh in res.get("wishes") or []:
            ref = wsh.get("ref") if isinstance(wsh, dict) and isinstance(wsh.get("ref"), dict) else None
            if ref and not REF_WORDS.search(reply):
                # "has to match cassette" names its value; a pointer always names another listing or its seller
                ref = None
                m = VALUE_AFTER.search(reply)
                if isinstance(wsh, dict) and not wsh.get("value") and m:
                    wsh["value"] = m.group(1).strip(" .'\"")
            if isinstance(wsh, dict) and wsh.get("prop") and (wsh.get("value") not in (None, "") or ref):
                j = self.named_item(reply, self.which(wsh.get("item"), i))
                need = Need(self.named_prop(str(wsh["prop"]), reply), str(wsh.get("rel") or "is"),
                            str(wsh.get("value") or ""), hidden=True)
                if ref:   # looked up once the questions are done; until then it cannot be checked, so it cannot block
                    ref["pair"] = ref.get("pair") is True or bool(re.search(r"\balongside\b|\bpairs? with\b|\bgoes with\b", reply, re.I))
                    m = PAGE_FROM.search(reply)
                    if m and not str(ref.get("seller") or "").strip():
                        ref["seller"] = m.group(2)
                        ref["department"] = ref.get("department") or m.group(1)
                    need.ref = ref
                    need.off = not need.value
                if any(roots(x.prop) == roots(need.prop) and (x.value == need.value or (x.ref and ref))
                       for x in self.wants[j].needs if x.hidden):
                    continue   # the same wish said again (a related question draws the same answer)
                self.wants[j].needs.append(need)
                self.forget(j)
                added = True
        return added

    def seller_of(self, r: Listing) -> str:
        """Every seller-like field, joined: a record can carry both its maker ("vendor") and the shop selling it."""
        return " ".join(str(v) for k, v in list(r.attrs.items()) + list(r.raw.items())
                        if SELLER_KEY.search(str(k)) and isinstance(v, str))

    def resolve_ref(self, need: Need) -> None:
        """A wish whose value is another listing's: find that listing by its seller and the properties named; when its
        notes hand the property on to another product, follow them (the store checks that the trail was walked to the
        listing that states the value), then copy, exclude or pair that value."""
        ref = dict(need.ref or {})
        pw = roots(need.prop) - KIN_IGNORE
        core = pw - BROAD or pw   # the words that make the name specific ("hoodie" in "hoodie style type")

        def key_of(r: Listing) -> str | None:
            keys = [k for k in r.attrs if core <= roots(plain_key(k)) or (roots(plain_key(k)) - BROAD and roots(plain_key(k)) <= pw)]
            return max(keys, key=lambda k: len(roots(plain_key(k)) & pw), default=None)

        seen: set[str] = set()
        pairs: dict[str, str] = {}
        value, best = None, None
        dead = 0
        for hop in range(REF_HOPS + 2):
            best = self.ref_page(ref, seen)
            if best is None:
                break
            seen.add(best.id)
            rec = {k: v for k, v in best.raw.items() if isinstance(v, (str, int, float))}
            rec["facts"] = best.attrs
            res = self.gw.think(QUICK, HOP_PROMPT.replace("{prop}", need.prop) + dumps(rec, 2500), 300)
            res = res if isinstance(res, dict) else {}
            if isinstance(res.get("pairs"), dict):
                pairs.update({" ".join(tokens(str(a))): str(b) for a, b in res["pairs"].items() if a and b})
            nxt = res.get("next")
            if isinstance(nxt, dict) and str(nxt.get("seller") or "").strip() and hop < REF_HOPS - 1:
                ref = {"seller": str(nxt["seller"]), "department": nxt.get("department"),
                       "with": nxt.get("with") if isinstance(nxt.get("with"), dict) else {}}
                self.log.append(f"ref {need.prop} {best.id} -> {ref['seller']}"[:120])
                continue
            k = key_of(best)
            value = str(res.get("value") or "").strip() or (str(best.attrs[k]) if k else None)
            if not value and dead < 2 and hop < REF_HOPS:
                # this page of the seller neither states the value nor hands it on: another of their listings
                # matching the details may be the one meant
                dead += 1
                self.log.append(f"ref {need.prop} {best.id} silent: next listing of the seller")
                continue
            break
        self.ref_src |= seen
        if value and need.ref and need.ref.get("pair"):
            paired = pairs.get(" ".join(tokens(value)))
            if not paired:
                self.log.append(f"ref {need.prop} no pairing for {value}")
                return
            value = paired
        elif value and pairs.get(" ".join(tokens(value))) and (need.ref or {}).get("same", True) is not False:
            # "go by" that listing, and its own notes pair the value with another ("pairs it with"): that one
            value = pairs[" ".join(tokens(value))]
            self.log.append(f"ref {need.prop} paired by the source's notes")
        if value:
            need.value = str(value)
            need.rel = "is" if (need.ref or {}).get("same", True) is not False or (need.ref or {}).get("pair") else "not"
            need.off = False
            self.log.append(f"ref {need.prop} <- {best.id if best else '?'}: {need.rel} {value}"[:160])
        else:
            self.log.append(f"ref {need.prop} unresolved ({len(seen)} pages)")

    def ref_page(self, ref: dict[str, Any], seen: set[str]) -> Listing | None:
        """The listing a reference names (its seller, department and named properties), read in full."""
        seller = str(ref.get("seller") or "").strip()
        given = {str(k): str(v) for k, v in ref.get("with").items() if v not in (None, "")} \
            if isinstance(ref.get("with"), dict) else {}
        if not seller or not self.roles.has("search") or self.free_steps() <= 2 or self.tight(110):
            self.log.append(f"ref page skipped ({seller})")
            return None
        sw = roots(seller)
        dw = roots(ref.get("department")) - KIN_IGNORE

        def theirs() -> list[Listing]:
            return [r for r in self.listings.values() if sw and sw <= roots(self.seller_of(r)) and r.id not in seen]

        def fit(r: Listing) -> int:
            got = 0
            for k, v in given.items():
                vw = roots(v)
                vals = [val for key, val in r.attrs.items() if roots(plain_key(key)) & roots(k)] or [r.blob()]
                got += 2 * any(vw and (vw <= roots(val) or split_figures(v) <= split_figures(val)) for val in vals)
            return got + bool(dw & roots(r.shelf))

        probe = -2   # pseudo item slot for the reference search; never a wanted index
        words = [t for v in given.values() for t in tokens(v) if len(t) > 1][:3]
        dept = [t for t in tokens(ref.get("department")) if len(t) > 2][:2]
        # a trail can run five listings deep in ~35 steps: one search a hop when it shows the seller, reading decides;
        # the seller with its department, then the seller alone, only when no read listing fits
        queries = [" ".join([seller] + words)] if words else []
        queries += [" ".join([seller] + dept)] if dept else []
        queries += [seller]
        need_fit = max(1, 2 * len(given) - 2)   # most of the named properties agree

        def ranked() -> list[Listing]:
            return sorted((r for r in theirs() if r.full and fit(r) >= need_fit), key=lambda r: -fit(r))

        for qn, q in enumerate(queries):
            if qn and (ranked() or self.free_steps() <= 2):
                break
            self.gather("search", q, probe, pages=2 if q == seller else 1)
            if not theirs():
                continue
            for batch in (1, 2):   # the likeliest page first: a long trail cannot afford three reads a hop
                unread = [r.id for r in sorted(theirs(), key=lambda r: -fit(r)) if not r.full][:batch]
                if ranked() and fit(ranked()[0]) >= 2 * len(given) or not unread or self.free_steps() <= 1:
                    break
                self.read(unread)
        rk = ranked()
        if rk:
            self.look_closer([rk[0].id])   # the notes that may hand the property on can sit in the short view only
        return rk[0] if rk else None

    def named_prop(self, prop: str, reply: str) -> str:
        """The property a reply names: when the model's choice is not in the reply but a record field name is, the field."""
        said = roots(reply)
        if roots(prop) & said:
            return prop
        keys = {plain_key(k) for r in self.listings.values() for k in r.attrs}
        hits = [k for k in keys if roots(k) and roots(k) <= said]
        return max(hits, key=lambda k: len(roots(k))) if hits else prop

    def notice_item(self, e: dict[str, Any]) -> int:
        """The wanted product a notice concerns: the one it names; else the one whose cart line it points at; else the
        one in the cart (a notice comes once a line is in)."""
        j = self.which(e.get("item"), -1)
        if j >= 0:
            return j
        for x in e.get("ids") or []:
            if x in self.cart_item:
                return self.cart_item[x]
        held = [self.cart_item[x] for x in self.cart if x in self.cart_item]
        return held[0] if held else 0

    def named_item(self, reply: str, j: int) -> int:
        """An answer can concern another wanted product than the one asked about (the length of the second product when
        asked about the first): the one product whose own words the answer names, when the chosen one's words are absent."""
        if len(self.wants) < 2 or not 0 <= j < len(self.wants):
            return j
        said = roots(reply)
        common = set.intersection(*[roots(w.what) for w in self.wants])
        score = [len((roots(w.what) - common - FILLER) & said) for w in self.wants]
        top = max(score)
        if score[j] == 0 and top > 0 and score.count(top) == 1:
            self.log.append(f"answer moved to item {score.index(top)}")
            return score.index(top)
        return j

    def which(self, name: Any, default: int) -> int:
        best, top = default, 0
        for j, w in enumerate(self.wants):
            s = len(roots(name) & (roots(w.what) | roots(w.department)))
            if s > top:
                best, top = j, s
        return best

    def forget(self, i: int) -> None:
        """Conditions changed: every verdict for the item, including checks, is redone."""
        for k in [k for k in self.marks if k[0] == i]:
            del self.marks[k]
        for k in [k for k in self.struck if k[0] == i]:
            del self.struck[k]
        self.vetted = {k for k in self.vetted if k[0] != i}
        self.rules.pop(i, None)

    # ---------------------------------------------------------------- notices
    def notices(self, after_order: bool = False) -> list[dict[str, Any]]:
        lines = self.gw.drain()
        if not lines:
            return []
        events: list[dict[str, Any]] = []
        if not (self.tight(CLOSE_S + 12) or self.gw.closing):
            conds = [{"item": w.what, "conditions": [(n.prop, n.rel, n.value) for n in w.live()]} for w in self.wants]
            cart = {rid: (self.listings[rid].title if rid in self.listings else "")[:80] for rid in self.cart}
            res = self.gw.think(QUICK, NOTICE_PROMPT.replace("{conds}", dumps(conds, 1500)).replace("{cart}", dumps(cart, 1000))
                                .replace("{lines}", "\n".join(lines[-8:])), 600)
            events = [e for e in (res.get("events") or []) if isinstance(e, dict)] if isinstance(res, dict) else []
        if not any(e.get("type") == "sold_out" for e in events) and any(GONE_WORDS.search(l) for l in lines):
            events.append({"type": "sold_out"})   # the wording says it is gone, whatever the reader made of it
        kept = []
        for e in events:
            t = e.get("type")
            if t == "drop" and e.get("prop"):
                pw = roots(e["prop"])
                if not pw & roots(" ".join(lines)):
                    continue   # a drop the lines never name is a misreading; stated wishes stay
                for j, w in enumerate(self.wants):
                    for n in w.needs:
                        if roots(n.prop) and (roots(n.prop) <= pw or pw <= roots(n.prop)):
                            n.off = True
                            self.forget(j)
                kept.append(e)
            elif t == "add" and e.get("prop") and e.get("value"):
                j = self.notice_item(e)
                rel = str(e.get("rel") or "is") if e.get("rel") in ("is", "not", "min", "max") else "is"
                self.wants[j].needs.append(Need(str(e["prop"]), rel, str(e["value"]), hidden=True))
                self.forget(j)
                kept.append(e)
            elif t == "quantity" and as_number(e.get("amount")) and 0 < as_number(e.get("amount")) < 1000:
                j = self.notice_item(e)
                self.wants[j].qty = int(as_number(e.get("amount")))
                kept.append(e)
            elif t == "budget" and as_number(e.get("amount")):
                self.budget = self.news_budget(as_number(e.get("amount")), lines)
                self.plan["budget_per_unit"] = False
                kept.append(e)
            elif t == "sold_out":
                ids = [x for x in (e.get("ids") or []) if x in self.cart] or list(self.cart)[:1]
                for x in ids:
                    if x in self.listings:
                        self.listings[x].gone = True
                kept.append(e)
            elif t in ("price", "refused"):
                kept.append(e)
        if after_order and not kept:
            kept = [{"type": "refused"}]
        self.log += [dumps(e, 140) for e in kept]
        return kept

    def news_budget(self, amount: float, lines: list[str]) -> float:
        """A new budget in the unit the first was held in: the sum the news line names, converted (a reader can turn
        "N cents" into N hundredths of the main unit while the budget is kept in cents)."""
        said = money_said(" ".join(lines))
        if not said or self.store_cents is None:
            return amount
        conv = [a if c == self.store_cents else (a / 100 if c else a * 100) for a, c in said]
        best = min(conv, key=lambda v: abs(v - amount) / max(v, amount, 1e-9))
        if abs(best - amount) > 1e-6:
            self.log.append(f"budget news read as {best:g} (reader said {amount:g})")
        return best

    def react(self, events: list[dict[str, Any]]) -> None:
        kinds = {e.get("type") for e in events}
        if kinds & {"price", "refused"} and not kinds & {"sold_out", "budget", "drop"}:
            look = "peek" if self.roles.has("peek") else "inspect"
            for rid in list(self.cart)[:2]:
                if self.gw.steps_left() > 2:
                    self.use(look, None, {"product_id": rid})
                    if self.listings.get(rid) and self.listings[rid].stock is False:
                        self.listings[rid].gone = True
        for i in range(len(self.wants)):
            self.score(i)
            need = 2 if self.cheapest and "drop" in kinds else 1
            if sum(1 for c, _ in self.ranked(i) if c == 0) < need:
                self.verify(i, want=need)

    # ---------------------------------------------------------------- linked products
    def find_link(self) -> None:
        if self.tight(150) or not re.search(r"compatib|\bfits?\b|work with|works with|goes with|for my|\bmatch", self.gw.request.lower()):
            return
        res = self.gw.think(QUICK, LINK_PROMPT.replace("{items}", dumps([w.what for w in self.wants], 600))
                            .replace("{q}", self.gw.request[:2500]), 300)
        if not isinstance(res, dict) or not res.get("anchor"):
            return
        items = [int(x) for x in (res.get("for_items") or []) if isinstance(x, (int, float)) and 0 <= int(x) < len(self.wants)]
        self.link = {"anchor": str(res["anchor"]), "items": items or list(range(len(self.wants))), "code": None}
        if not self.roles.has("search"):
            return
        probe = -1   # pseudo item slot for the anchor search; never a wanted index
        self.gather("search", " ".join(tokens(res["anchor"])[:5]), probe)
        aw = roots(res["anchor"])
        near = sorted((r for rid, r in self.listings.items() if probe in self.found_for.get(rid, set())),
                      key=lambda r: -len(aw & roots(r.title)))[:2]
        for r in near:
            codes = [c.upper() for c in SERIES_CODE.findall(r.blob()) if re.search(r"\d", c)]
            if not codes and not r.full and self.gw.steps_left() > self.close_steps() + 3:
                self.read([r.id])
                codes = [c.upper() for c in SERIES_CODE.findall(r.blob()) if re.search(r"\d", c)]
            if codes:
                self.link.update(code=codes[0], anchor_id=r.id)
                self.log.append(f"link {res['anchor']} -> series {codes[0]}")
                break
        for rid in list(self.found_for):
            self.found_for[rid].discard(probe)

    # ---------------------------------------------------------------- basket + close
    def picks(self, fresh_reads: bool = True) -> dict[int, Listing] | None:
        chosen: dict[int, Listing] = {}
        for i in range(len(self.wants)):
            # each wanted product is its own cart line: a listing already picked for another is not this one's pick
            used = {r.id for r in chosen.values()}
            rk = [x for x in self.ranked(i) if x[1].id not in used]
            if fresh_reads and (not rk or rk[0][0] > 0) and self.free_steps() > 1 and not self.tight(80):
                self.verify(i, want=1)
                rk = [x for x in self.ranked(i) if x[1].id not in used]
            elif fresh_reads and rk and not self.marks[(i, rk[0][1].id)].get("plain") and self.free_steps() > 3 \
                    and not self.tight(90):
                self.verify(i, want=1, plain=True)   # the best fit only shows the value under another name: read on
                rk = [x for x in self.ranked(i) if x[1].id not in used]
            if not rk:
                near = None if fresh_reads else self.nearest(i, used)
                if not near:
                    return None
                chosen[i] = near
                continue
            chosen[i] = rk[0][1]
            if not self.cheapest:   # a cart pick that still fits stays (news may have changed the order of fits);
                # the line held early only to draw the store's news was picked before the last searches: it competes
                keep = [r for c, r in rk if c == 0 and r.id in self.cart and r.id != self.early_id]
                if keep and rk[0][0] == 0:
                    chosen[i] = keep[0]
        self.pair_up(chosen, fresh_reads)
        if self.budget and self.total(chosen) > self.budget:   # over budget: the cheapest fit of each line
            for i in chosen:
                taken = {r.id for j, r in chosen.items() if j != i}
                alts = sorted([r for c, r in self.ranked(i) if c == 0 and r.price is not None and r.id not in taken],
                              key=lambda r: r.price)
                if alts:
                    chosen[i] = alts[0]
        return chosen

    def pair_up(self, chosen: dict[int, Listing], fresh: bool) -> None:
        """Products bought together for one machine: a listing naming another chosen listing's model code (its number or
        a name it is also sold under) is the one that works with it, so among fits the linked ones are taken. The store's
        search reads the whole record, so searching a code finds the listings that name it."""
        searched = 0
        for i in list(chosen):
            others = set().union(*[model_codes(r) for j, r in chosen.items() if j != i])
            if not others or model_codes(chosen[i]) & others:
                continue
            linked = [r for c, r in self.ranked(i) if c == 0 and model_codes(r) & others]
            if not linked and fresh and searched < 4 and self.roles.has("search") and self.free_steps() > 4 \
                    and not self.tight(120):
                for code in sorted(others)[:2]:
                    self.gather("search", code, i)
                    searched += 1
                self.score(i)
                todo = [r.id for r in self.shelf_pool(i) if not r.full and model_codes(r) & others][:2]
                if todo:
                    self.read(todo)
                    self.score(i)
                linked = [r for c, r in self.ranked(i) if c == 0 and model_codes(r) & others]
            if linked:
                chosen[i] = linked[0]
                self.log.append(f"pair {i} -> {linked[0].id}")

    def nearest(self, i: int, used: set[str] | None = None) -> Listing | None:
        """No record shows every condition. An order of a record never read is graded a miss (race 10: 0 of 34), so the
        likeliest unread rows are read first while the clock allows, then the read record breaking the fewest
        conditions is taken. Only when many read records each break one is the answer that nothing fits."""
        todo = self.order_to_read(i)
        # race worlds: a close miss would be bought, so unread rows (a search for the shopper's wish may have just
        # listed them) are read while a lean finish still fits: the cart lines, one look and the order
        floor = self.lean_close() if self.race else self.close_steps()
        if todo and self.gw.steps_left() > floor + 1 and not self.tight(CLOSE_S + 10):
            self.read([r.id for r in todo[:max(1, min(2 if self.race else 3, self.gw.steps_left() - floor - 1))]], floor=floor)
            self.score(i)
            rk = [x for x in self.ranked(i) if x[1].id not in (used or set())]
            if rk:
                return rk[0][1]
        if self.plan.get("silent_decline"):
            return None   # the shopper asked to hear that nothing fits rather than be sold a near miss
        pool = [r for r in self.shelf_pool(i) if r.id not in (used or set())]
        no = [r for r in pool if self.marks.get((i, r.id), {}).get("state") == "no" and not r.gone and r.stock is not False]
        race = self.race   # race worlds: nothing-fits tasks were 2 of 90 (races 10, 11), so a close miss is taken
        if race and len(no) >= SURE_NONE:
            no = [r for r in no if len(self.marks[(i, r.id)]["failed"]) == 1]   # only a listing missing one condition
        elif not no or len(no) >= SURE_NONE:
            return None
        if not no:
            return None
        return min(no, key=lambda r: (len(self.marks[(i, r.id)]["failed"]), r.price if r.price is not None else float("inf")))

    def vet(self, chosen: dict[int, Listing]) -> dict[int, Listing] | None:
        """An independent look at each pick's full record before it goes in the cart; a pick it finds failing is
        struck and the next fit takes its place (each record is looked at once)."""
        for _ in range(3):
            struck = False
            for i, r in list(chosen.items()):
                key = (i, r.id)
                if key in self.vetted or not r.full or self.tight(CLOSE_S + 30):
                    continue
                self.vetted.add(key)
                # a compatibility code is checked on the whole record by code (Rule.check); the model would look for it
                # in the fields alone, while listings name it in their text ("Compatible with: QRX4410")
                conds = [{"property": n.prop, "rel": n.rel, "value": n.value} for n in self.wants[i].live()
                         if not is_code(n.value.strip())]
                if not conds:
                    continue
                notes = {k: v for k, v in r.raw.items() if isinstance(v, str) and k not in ("title",) and len(v) > 30}
                rec = {"title": r.title, "fields": r.attrs, "price": r.price, "text": dumps(notes, 900)}
                res = self.gw.think(QUICK, CHECK_PROMPT.replace("{conds}", dumps(conds, 1200)).replace("{rec}", dumps(rec, 3000)), 300)
                if isinstance(res, dict) and res.get("verdict") == "fails" and res.get("failed"):
                    self.struck[key] = "check: " + dumps(res.get("failed"), 80)
                    self.marks[key] = {"state": "no", "failed": [self.struck[key]]}
                    self.log.append(f"check struck {r.id}")
                    struck = True
            if not struck:
                return chosen
            chosen = self.picks()
            if chosen is None:
                return None
        return chosen

    def early_hold(self) -> None:
        """Race stores deliver their news (a new budget, the held item selling out, a price move) on the first call after
        the first cart add. Holding the best candidate early and looking at the cart brings that news while there are
        still steps to act on it; the record is then viewed again, since a sell-out can be silent."""
        if self.cart or not self.roles.has("cart_add") or self.free_steps() < 6 or self.tight(150) or not self.race:
            return   # race worlds (four or more messages) bring news after the first add; others at the order itself
        if any(x.ref and not x.value for w in self.wants for x in w.needs):
            return   # a reference trail still to walk needs those steps more (a trail runs up to five listings deep)
        for i in range(len(self.wants)):
            self.score(i)
        # only a listing that meets every condition so far: a guess held early can end up bought where nothing fits
        fits = [(i, rk[0][1]) for i in range(len(self.wants)) for rk in [self.ranked(i)] if rk and rk[0][0] == 0]
        if not fits:
            return
        i, r = fits[0]
        if self.budget and (r.price or 0) * self.wants[i].qty > self.budget + 1e-6:
            return
        self.sync_cart({i: r})
        if not self.cart:
            return
        self.early_id = r.id
        if self.roles.has("cart_show"):
            self.check_cart()
        ev = [e for e in self.notices() if e.get("type") not in ("agreed", "nothing")]
        self.peeked.discard(r.id)
        if self.look_closer([r.id]):
            self.rechecked.add(r.id)   # its state after the news is known
        if r.stock is False:
            r.gone = True
            ev.append({"type": "sold_out"})
        if ev:
            self.react(ev)
        self.log.append(f"early hold {r.id} news {[e.get('type') for e in ev]}")

    def total(self, chosen: dict[int, Listing]) -> float:
        return sum((r.price or 0) * self.wants[i].qty for i, r in chosen.items())

    def sync_cart(self, chosen: dict[int, Listing]) -> None:
        want = {r.id: self.wants[i].qty for i, r in chosen.items()}
        self.cart_item.update({r.id: i for i, r in chosen.items()})
        add = self.roles.by_role.get("cart_add")
        drop = self.roles.by_role.get("cart_drop")
        if add and add.get("lines"):
            if want != self.cart:
                arr, idk, qk = add["lines"]
                obs = self.use("cart_add", {arr: [{idk: rid, qk: q} for rid, q in want.items()]})
                if obs is not None and not refused(obs):
                    self.cart = dict(want)
            return
        for rid, q in list(self.cart.items()):
            if want.get(rid) == q or (want.get(rid, 0) > q and not self.roles.arg("cart_add", "quantity")):
                continue   # right count, or short of it where each add call brings one more unit
            qarg = self.roles.arg("cart_drop", "quantity")
            fixed = {k: v for k, v in ((drop or {}).get("fixed") or {}).items() if k != qarg}
            # one tool that only sets a line's count: no switch tells a removal from an addition, so a count of zero removes
            same_tool = bool(drop and add and drop.get("tool") == add.get("tool") and fixed == (add.get("fixed") or {}))
            if drop and not same_tool:
                extra = {"product_id": rid}
                if qarg and qarg not in (drop.get("fixed") or {}):
                    extra["quantity"] = q   # units to take out
                    self.use("cart_drop", None, extra)
                else:
                    # no count argument: each call takes out one unit; keep the units still wanted
                    for _ in range(max(1, q - (want.get(rid) or 0))):
                        self.use("cart_drop", None, extra)
                    if want.get(rid) and want[rid] < q:
                        self.cart[rid] = want[rid]
                        continue
            elif add and (add.get("sets_line") or same_tool) and self.roles.arg("cart_add", "quantity"):
                self.use("cart_add", None, {"product_id": rid, "quantity": 0})   # a count of zero removes the line
            self.cart.pop(rid, None)
        for rid, q in want.items():
            if self.cart.get(rid) == q or not add:
                continue
            extra: dict[str, Any] = {"product_id": rid}
            if self.roles.arg("cart_add", "quantity"):
                extra["quantity"] = q
                obs = self.use("cart_add", None, extra)
                if obs is not None and not refused(obs):
                    self.cart[rid] = q
                continue
            # no count argument: each call adds one unit (race 10 counted one per call), so call once per unit missing
            have = self.cart.get(rid, 0)
            for _ in range(max(0, q - have)):
                obs = self.use("cart_add", None, extra)
                if obs is None or refused(obs):
                    break
                have += 1
            self.cart[rid] = have

    def recheck(self, chosen: dict[int, Listing]) -> bool:
        """Stock and prices move while the shopper is asked and the cart is filled: each line's record is read again just
        before the order, once. True when a line sold out or changed price (the picks are then made again)."""
        fresh = [r.id for r in chosen.values() if r.id not in self.rechecked]
        race = self.race and self.roles.has("peek")
        if not fresh or not (self.roles.has("inspect") or self.roles.has("multi_inspect")) or self.tight(CLOSE_S):
            return False
        if race:
            # race worlds sell the held line out silently and only the short view carries stock and price: one look a
            # line, made while that look, the order and one swap still fit
            if self.gw.steps_left() < len(fresh) + 4:
                return False
        elif self.gw.steps_left() <= self.close_steps() + (1 if self.roles.has("multi_inspect") else len(fresh)):
            return False
        self.rechecked.update(fresh)
        price = {rid: self.listings[rid].price for rid in fresh}
        if race:
            self.peeked -= set(fresh)
            self.look_closer(fresh, closing=True)
        else:
            self.read(fresh)
        moved = [rid for rid in fresh if self.listings[rid].stock is False or self.listings[rid].gone or
                 (price[rid] is not None and self.listings[rid].price is not None and abs(self.listings[rid].price - price[rid]) > 1e-9)]
        if moved:
            self.log.append(f"recheck moved {moved}")
            for i in range(len(self.wants)):
                self.score(i)
        return bool(moved)

    def check_cart(self) -> None:
        obs = self.use("cart_show")
        if obs is None:
            return
        seen: dict[str, int] = {}
        for d in listing_dicts(obs):
            rec = to_listing(d)
            if not rec:
                continue
            q = next((int(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                      and re.search(r"qty|quant|count|units|pieces|number", str(k), re.I)
                      and not re.search(r"price|cost|total", str(k), re.I)), None)
            seen[rec.id] = q if q is not None else self.cart.get(rec.id, 1)
        if seen or re.search(r"empty|no items|nothing", dumps(obs, 300).lower()):
            self.cart = seen
        self.viewed = tuple(sorted(self.cart.items()))

    def tell_basket(self, chosen: dict[int, Listing]) -> bool:
        """The shopper hears the exact basket before the order when they asked to hear or approve it."""
        lines = tuple(sorted(self.cart.items()))
        if not self.roles.has("talk") or not self.wants_basket() or self.msgs >= self.msg_cap or \
                self.gw.steps_left() < 2 or lines == self.told:
            return False
        parts = [f"{self.wants[i].qty} x {(r.title or 'item')[:80]} ({r.id}) at {r.price if r.price is not None else '?'}"
                 for i, r in chosen.items()]
        money = f", within your budget of {self.budget:g}" if self.budget else ""
        ending = "May I place the order?" if self.plan.get("confirm_first") else "I am placing the order now."
        text = f"Here is your basket: {'; '.join(parts)}. Total {self.total(chosen):g}{money}. {ending}"[:780]
        obs = self.use("talk", None, {"message": text})
        self.msgs += 1
        self.told = lines
        if self.hear(text, obs, 0):
            for i in range(len(self.wants)):
                self.score(i)
        return True

    def place(self) -> bool:
        if not self.roles.has("order"):
            return False
        self.orders += 1
        obs = self.use("order", None, {"remark": self.order_note()})
        if not self.gw.done and refused(obs):
            self.cart = {}   # the store holds none of it: the lines go in again before the next order
        return self.gw.done

    def order_note(self) -> str:
        """What is being bought and why it fits, for the order's own note."""
        lines, total = [], 0.0
        for rid, q in self.cart.items():
            r = self.listings.get(rid)
            pr = r.price if r and r.price is not None else None
            total += (pr or 0) * q
            lines.append(f"{q} x {(r.title if r and r.title else rid)[:70]} ({rid})" + (f" at {pr:g} each" if pr is not None else ""))
        word = {"min": "at least ", "max": "at most ", "not": "not ", "has": "with "}
        conds = [f"{n.prop} {word.get(n.rel, '')}{n.value}" for w in self.wants for n in w.needs if not n.off][:8]
        money = f", within the budget of {self.budget:g}" if self.budget else ""
        return (f"Ordered {'; '.join(lines)}. Total {total:g}{money}" + (f"; it meets {', '.join(conds)}" if conds else "") + ".")[:700]

    def cut_cost(self, chosen: dict[int, Listing]) -> bool:
        """The picks cost more than the budget: for the costliest lines first, read the cheaper rows not yet read (one
        more search when none is listed), so a cheaper fit can take their place. True when a record came back."""
        self.cost_cuts += 1
        got = 0
        for i in sorted(chosen, key=lambda j: -(chosen[j].price or 0) * self.wants[j].qty):
            # a decline is the other way out, so the closing reserve may be spent down to the cart lines and the order
            if self.gw.steps_left() <= len(chosen) + 3 or self.tight(CLOSE_S + 20):
                break
            cap = chosen[i].price if chosen[i].price is not None else float("inf")

            def cheaper() -> list[Listing]:
                # the usual reading order (rows showing the wanted values first), cut to rows not dearer than the pick;
                # rows showing no price may be cheaper too
                return [r for r in self.order_to_read(i) if r.price is None or r.price < cap]
            todo = cheaper()
            if not todo and self.widen(i):
                todo = cheaper()
            if todo:
                got += self.read([r.id for r in todo[:3]], floor=len(chosen) + 2)
                self.score(i)
        self.log.append(f"over budget: {got} cheaper rows read")
        return got > 0

    def give_up(self) -> bool:
        if not self.roles.has("give_up"):
            return False
        self.use("give_up", None, {"remark": DECLINE_TEXT})
        return self.gw.done

    def last_call(self) -> bool:
        """Too few steps for another round: order what fits now (or the cart as it stands)."""
        if self.gw.steps_left() > 2 * len(self.wants) + 1:   # a swap costs a drop and an add per item, then the order
            return False
        chosen = self.picks(False)
        if chosen and (not self.budget or self.total(chosen) <= self.budget + 1e-6) and \
                self.gw.steps_left() > sum(1 for i, r in chosen.items() if self.cart.get(r.id) != self.wants[i].qty):
            self.sync_cart(chosen)
        self.close_out()
        return True

    def settle(self) -> None:
        for attempt in range(6):
            if self.gw.done or self.gw.steps_left() <= 0:
                return
            if self.last_call():
                return
            chosen = self.picks()
            if chosen is None:
                for i in range(len(self.wants)):
                    if not self.ranked(i) and self.free_steps() > 2 and not self.tight(70):
                        self.widen(i)
                        self.verify(i, want=1)
                chosen = self.picks()
            if chosen is not None:
                chosen = self.vet(chosen)
            if chosen is None:
                chosen = self.picks(False)   # nothing confirmed: the nearest records, not a decline
            if chosen is not None and self.budget and self.total(chosen) > self.budget + 1e-6 and self.cost_cuts < 2 \
                    and self.cut_cost(chosen):
                continue   # cheaper fits were read for the costliest picks: choose again
            if chosen is None or (self.budget and self.total(chosen) > self.budget + 1e-6):
                self.log.append("no candidate: give up" if chosen is None else f"over budget {self.total(chosen)}")
                if self.give_up() or not self.roles.has("give_up"):
                    return
                # the decline was blocked by news (a dropped wish, a new budget): take it in and look again
                self.react(self.notices(after_order=True))
                continue
            self.sync_cart(chosen)
            if self.recheck(chosen):
                continue   # a cart line sold out or changed price since it was read: pick again from fresh records
            ev = self.notices()
            if self.roles.has("cart_show") and self.gw.steps_left() > self.close_steps() and \
                    (attempt == 0 or ev or len(chosen) > 1):
                self.check_cart()
                ev += self.notices()
                if set(self.cart) != {r.id for r in chosen.values()} and self.gw.steps_left() > 3:
                    self.sync_cart(chosen)
            if ev:
                self.react(ev)
                if self.picks(False) != chosen:
                    continue
            if self.tell_basket(chosen):
                ev = [e for e in self.notices() if e.get("type") not in ("agreed", "nothing")]
                if ev:
                    self.react(ev)
                if self.picks(False) != chosen:
                    continue
            if self.roles.has("cart_show") and tuple(sorted(self.cart.items())) != self.viewed and \
                    self.gw.steps_left() > 2:
                self.check_cart()   # the cart is looked over after its last change (a swap after a refusal too)
                ev = [e for e in self.notices() if e.get("type") not in ("agreed", "nothing")]
                if ev:   # news can ride on this very look: take it in before ordering
                    self.react(ev)
                    if self.picks(False) != chosen:
                        continue
            if self.place():
                return
            self.react(self.notices(after_order=True))
        self.close_out()

    def close_out(self) -> None:
        self.gw.closing = True
        if self.gw.done or self.gw.steps_left() <= 0:
            return
        if self.cart and self.roles.has("order") and self.orders < 6:
            self.place()
        for _ in range(2):   # a decline can be blocked once by news; the second attempt closes
            if self.gw.done or self.gw.steps_left() <= 0:
                return
            self.give_up()

    def hurry(self) -> None:
        """Close the episode from what is known: no new reads, no model calls beyond the close reserve."""
        self.gw.closing = True
        if self.gw.done or self.gw.steps_left() <= 0 or not self.roles.has("order"):
            self.close_out()
            return
        for i in range(len(self.wants)):
            self.score(i)
        chosen = self.picks(False)
        if chosen and (not self.budget or self.total(chosen) <= self.budget + 1e-6):
            self.sync_cart(chosen)
            if self.gw.left() > 14 and self.gw.steps_left() >= 2:
                self.tell_basket(chosen)
            if self.place():
                return
        self.close_out()

    # ---------------------------------------------------------------- episode
    def run(self) -> list[dict[str, Any]]:
        try:
            self.start()
            if not self.core_roles():
                self.log.append("tool loop: roles " + ",".join(self.roles.by_role))
                self.gw.closing = True
                return ToolLoop(self).run()
            self.find_link()
            n = len(self.wants)
            # with four or more messages (race worlds) the shopper is asked early, from a first look: what they reveal
            # then steers the searches, instead of the steps going into deep pages of a list read before knowing it
            self.light = self.race
            for i in range(n):
                self.explore(i)
                self.notices()
            if not self.light:
                share = max(3, self.free_steps() // (2 * n))
                for i in range(n):
                    self.verify(i, want=2, cap_reads=share)
            self.notices()
            share = max(1, min(3, self.free_steps() // max(1, n)))   # every item gets its turn, the last one too
            for i in range(n):   # questions are chosen from read records: have a few with properties before asking
                have = sum(1 for r in self.shelf_pool(i) if r.full and len(r.attrs) > 2)
                if have < share and self.free_steps() > 0:
                    self.read([r.id for r in self.order_to_read(i)[:share - have]])
            self.interview()
            self.notices()
            self.early_hold()
            for j, w in enumerate(self.wants):   # a wish given as another listing's value: look that listing up
                for x in w.needs:
                    if x.ref and not x.value:
                        self.resolve_ref(x)
                        self.forget(j)
            self.light = False

            def heard(i: int) -> list[Need]:   # wishes the shopper revealed (a request's code is not one); never "not"
                return [x for x in self.wants[i].needs if x.hidden and not x.off and x.rel != "not"
                        and not is_code(x.value.strip())]

            # an item whose revealed wish nothing read meets goes first, and each item gets its share of the steps:
            # one item's deep pages must not leave another's wish unsearched
            order = sorted(range(n), key=lambda i: (0 if heard(i) and not any(c == 0 for c, _ in self.ranked(i)) else 1, i))
            for k, i in enumerate(order):
                share = max(3, self.free_steps() // max(1, n - k)) if n > 1 else None
                self.item_limit = self.gw.steps + share if share else None
                try:
                    if self.race and self.roles.has("search"):
                        self.sharp_search(i)
                    late = heard(i)
                    if late and self.roles.has("search"):   # a revealed wish gets its own value search, read deep
                        vt = [t for t in tokens(late[-1].value) if root(t) not in {root(k) for k in self.label_words(i)}]
                        # the product's own noun first (a department label is broader than the item), then the label
                        base = [t for t in tokens(self.wants[i].queries[0]) if len(t) > 1][:3] if self.wants[i].queries else []
                        if names_something(vt) and not self.gather("search", " ".join(vt + (base or [self.noun(i)])), i, pages=4):
                            self.gather("search", " ".join((vt + self.label_words(i))[:6]), i, pages=4)
                    self.verify(i, want=3, cap_reads=max(2, self.item_limit - self.gw.steps) if share else None)
                finally:
                    self.item_limit = None
            self.settle()
        except Hurry:
            self.log.append(f"hurry at step {self.gw.steps}, {self.gw.left():.0f}s left")
            self.hurry()
        return self.gw.transcript

    def report(self) -> dict[str, Any]:
        return {"role": "assistant", "content": "sentinel2 " + dumps({
            "roles": {k: v["tool"] for k, v in self.roles.by_role.items()},
            "wants": [{"what": w.what, "qty": w.qty, "needs": [(n.prop, n.rel, n.value, n.hidden, n.off) for n in w.needs]}
                      for w in self.wants],
            "budget": self.budget, "goal": self.plan.get("goal"), "msgs": f"{self.msgs}/{self.msg_cap}",
            "link": self.link, "log": self.log[-12:], "cart": self.cart,
            "marks": {f"{k[0]}:{k[1]}": (v["state"], v["failed"]) for k, v in list(self.marks.items())[:30]},
            "cost": round(self.gw.cost, 4), "llm": self.gw.llm_calls, "slow": self.gw.slow_hits,
            "t": round(TOTAL_S - self.gw.left(), 1), "steps": self.gw.steps}, 6000)}


# ======================================================================================================== tool loop
class ToolLoop:
    """When the store's tools cannot be labelled: a model drives the tools directly, with a compact history."""

    def __init__(self, agent: Sentinel):
        self.a = agent

    def run(self) -> list[dict[str, Any]]:
        gw = self.a.gw
        guide = ("You are a store assistant buying for a customer with the given tools; each call is one step. Honour every "
                 "condition in the request (quantities, budget, cheapest rule, message limit, procedures). Ask the shopper "
                 "one property per message. Confirm attributes on a product's own record before buying. Tell the shopper "
                 "the basket before ordering. Decline only if nothing fits, without naming products.")
        msgs = [{"role": "system", "content": guide},
                {"role": "user", "content": gw.request + ("\n\n" + dumps(gw.opening) if gw.opening else "")}]
        tools = [{"type": "function", "function": s} for s in gw.schemas]
        idle = 0
        while not gw.done and gw.steps_left() > 0 and gw.left() > 15:
            res = gw.think(DEEP, "", 600, messages=msgs[:2] + msgs[2:][-24:], tools=tools, cap_s=35)
            calls = (res or {}).get("tool_calls") or []
            if not calls:
                idle += 1
                if idle > 2:
                    break
                msgs.append({"role": "user", "content": "Use one of the tools now."})
                continue
            idle = 0
            c = calls[0]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            obs = gw.tool(c["function"]["name"], args if isinstance(args, dict) else {})
            msgs.append({"role": "assistant", "content": res.get("content") or "", "tool_calls": [c]})
            msgs.append({"role": "tool", "tool_call_id": c.get("id"), "content": dumps(obs, 5000) + f"\n[steps left {gw.steps_left()}]"})
        if not gw.done and gw.steps_left() > 0 and self.a.roles.has("order"):
            gw.tool(self.a.roles.by_role["order"]["tool"], {})   # never leave the episode open
        if not gw.done and gw.steps_left() > 0:
            self.a.give_up()
        return gw.transcript


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    agent = Sentinel(problem_data)
    try:
        out = agent.run()
        out.append(agent.report())
        return out
    except Exception as exc:  # never leave an episode open while it can still be closed
        gw = agent.gw
        gw.closing = True
        if not gw.done and gw.steps_left() > 0:
            try:
                if agent.cart and agent.roles.has("order"):
                    agent.use("order")
                elif agent.roles.has("give_up"):
                    agent.give_up()
            except Exception:
                pass
        gw.transcript.append({"role": "assistant", "content": f"error: {type(exc).__name__}"})
        return gw.transcript


__all__ = ["agent_main"]
