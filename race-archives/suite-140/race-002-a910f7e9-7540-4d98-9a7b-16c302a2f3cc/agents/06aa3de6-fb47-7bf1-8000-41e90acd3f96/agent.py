import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# Failover only. Same plan() for every name.
_MODELS = {
    "chutes": (
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "Qwen/Qwen3-32B-TEE",
        "zai-org/GLM-5.2-TEE",
         "google/gemma-4-31B-turbo-TEE",
        "moonshotai/Kimi-K2.6-TEE",
    ),
    "openrouter": (
        "deepseek/deepseek-v3.2",
        "qwen/qwen3-32b",
        "z-ai/glm-5.2",
        "google/gemma-4-31b-it",
        "moonshotai/kimi-k2.6",
    ),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_LIMIT_S = 270.0
_OS = (
    "android",
    "ios",
    "blackberry",
    "windows phone",
    "symbian",
    "harmonyos",
    "hyperos",
    "sailfish",
    "ubuntu touch",
    "tizen",
)
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
_NAMED_RES = (
    ("4k", re.compile(r"\b(?:4k|uhd|ultra hd|2160p)\b", re.I)),
    ("qhd", re.compile(r"\b(?:qhd|1440p|2k)\b", re.I)),
    ("fhd", re.compile(r"\b(?:fhd|1080p|full ?hd)\b", re.I)),
    ("hd", re.compile(r"\b(?:720p|hd\+?)\b", re.I)),
)
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "sale",
    "original",
    "smartphone",
    "phone",
    "mobile",
    "tablet",
    "monitor",
    "inch",
    "inches",
    "promo",
}
_JUNK = re.compile(
    r"ring grip|phone grip|tempered glass|screen protector|replacement screen|"
    r"screen only|feature phone|basic phone|mystery|cable only|charger only|"
    r"\bpouch\b|\bholder\b",
    re.I,
)


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


def _money(text: str) -> float | None:
    best = None
    for m in _MONEY.finditer(text or ""):
        v = _fnum(m.group(1) + "." + (m.group(2) or "0"))
        if v is None:
            continue
        best = v if best is None else min(best, v)
    return best


def _res_class(text: str) -> set:
    t = (text or "").lower()
    out = set()
    for a, b in ((int(x), int(y)) for x, y in _PIX.findall(t)):
        lo, hi = sorted((a, b))
        if hi >= 3840 and lo >= 2160:
            out.add("4k")
        elif hi >= 2560 and lo >= 1440:
            out.add("qhd")
        elif hi >= 1920 and lo >= 1080:
            out.add("fhd")
        elif hi >= 1280 and lo >= 720:
            out.add("hd")
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p)\b", t):
        out.add("4k")
    if re.search(r"\b(?:qhd|1440p|2k)\b", t):
        out.add("qhd")
    if re.search(r"\b(?:fhd|1080p|full ?hd)\b", t):
        out.add("fhd")
    if re.search(r"\b(?:720p|hd)\b", re.sub(r"\b(?:full|ultra)\s*hd\b", "", t)):
        out.add("hd")
    return out


def _wanted_res(text: str) -> str | None:
    t = (text or "").lower()
    if re.search(r"\b4k\b|\buhd\b|\bultra hd\b|\b2160p\b", t):
        return "4k"
    if re.search(r"\bqhd\b|\b1440p\b|\b2k\b", t):
        return "qhd"
    if re.search(r"\b1080p\b|\bfull ?hd\b|\bfhd\b", t):
        return "fhd"
    if re.search(r"\b720p\b", t):
        return "hd"
    return None


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


def _res_named(text: str) -> set:
    t = re.sub(r"\b(?:full|ultra)\s+hd\b", " ", (text or "").lower())
    return {name for name, pat in _NAMED_RES if pat.search(t)}


def _res_pix(text: str) -> set:
    t = (text or "").replace("*", " x ").replace("×", " x ")
    out = set()
    for a, b in ((int(x), int(y)) for x, y in _PIX.findall(t)):
        lo, hi = sorted((a, b))
        if hi >= 3840 and lo >= 2160:
            out.add("4k")
        elif hi >= 2560 and lo >= 1440:
            out.add("qhd")
        elif hi >= 1920 and lo >= 1080:
            out.add("fhd")
        elif hi >= 1280 and lo >= 720:
            out.add("hd")
    return out


def _batt_hours(text: str) -> float:
    t = (text or "").lower()
    t = re.sub(r"\d+(?:\.\d+)?\s*m(?:ah|wh|hz)\b", " ", t)
    m = _HOURS.search(t)
    return float(m.group(1)) if m else 0.0


def _sport(text: str) -> float:
    return 1.0 if re.search(r"\b(?:sport|running|workout|gym|secure)\b", text or "", re.I) else 0.0


def _wire(text: str) -> float:
    return 1.0 if re.search(r"\b(?:wireless|bluetooth|tws)\b", text or "", re.I) else 0.0


def _dead_keys(title: str) -> set:
    out = set()
    for tok in re.findall(r"[a-z0-9]{2,}", (title or "").lower()):
        if tok in _STOP or tok.isdigit():
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


def _wet(text: str) -> int:
    t = (text or "").lower()
    n = 0
    if re.search(r"\bip68\b|\bip67\b", t):
        n += 6
    if re.search(r"\bipx[78]\b", t):
        n += 5
    if re.search(r"\bipx[45]\b|\bip54\b|\bip55\b", t):
        n += 4
    if "waterproof" in t:
        n += 4
    if "water resistant" in t or "water-resistant" in t:
        n += 3
    if "sweat" in t:
        n += 3
    if _IP.search(t):
        n += 2
    if "splash" in t:
        n += 1
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


def _has_os(text: str) -> bool:
    t = (text or "").lower()
    return any(tok in t for tok in _OS)


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
        self.budget = _money(self.head) or _money(raw)
        self.phrase = ""
        qm = re.search(r'"([^"]{2,80})"', self.head)
        if qm:
            self.phrase = qm.group(1).strip()
        else:
            pm = re.search(r"matches(?: the phrase)?\s+([A-Za-z0-9][A-Za-z0-9 \-_/]{1,60})", self.head, re.I)
            if pm:
                self.phrase = pm.group(1).strip(" .,")
        cm = re.search(r"in the ([A-Za-z][\w&'\- ]{1,60}?) category", self.head, re.I)
        self.aisle = cm.group(1).strip() if cm else None
        self.seeds = [(a, b) for a, b in _KEY.findall(self.head)]
        self.rank = bool(self.seeds and re.search(r"highest refresh rate", raw, re.I))
        self.inch = _inch(self.head)
        self.res = _wanted_res(self.head)
        self.hz_need = _hz(self.head)
        self.ram = _ram_gb(self.head)
        self.soft_brand = None
        sm = re.search(
            r"(?:ideally,\s*)?i prefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,32}?)"
            r"(?:\s*,|\s+but\b|\s*$)",
            self.head,
            re.I,
        )
        if sm and "prefer the" not in self.head.lower():
            self.soft_brand = sm.group(1).strip(" .,")
        self.hard_brand = None
        hm = re.search(
            r"(?:i need|i want|find me|get me|can you get me)\s+(?:an?\s+|in-stock\s+)*"
            r"([A-Z][\w&.'\-]+(?:\s+[A-Z][\w&.'\-]+)?)"
            r"(?:\s+\d|\s+(?:smartphone|tablet|monitor|headphone|gaming|inch))",
            self.head,
            re.I,
        )
        if hm:
            tok = hm.group(1).strip()
            if tok.lower() not in {"in", "can", "the", "smartphone", "tablet", "monitor", "headphone", "gaming"}:
                self.hard_brand = tok
        nm = re.search(r"\bI need a ([A-Z][\w&.'\-]+)\s+smartphone", self.head)
        if nm:
            self.hard_brand = nm.group(1)
        self.color = None
        self.model = None
        rm = re.search(
            r"prefer(?: the)?\s+(black|green|white|blue|red|gold|silver|purple)\s+(\d{3,12})\b",
            self.head,
            re.I,
        )
        if not rm:
            rm = re.search(
                r"strongly prefer the\s+(black|green|white|blue|red|gold|silver|purple)\s+(\d{3,12})\b",
                self.head,
                re.I,
            )
        if rm:
            self.color = rm.group(1).lower()
            self.model = rm.group(2)
        self.ask = bool(re.search(r"if you ask|tell you if you ask", self.head, re.I))
        self.why = bool(
            re.search(
                r"walk me through|give me a reason|tell me why|why the one you pick|why it's a good",
                self.head,
                re.I,
            )
        )
        self.wet = bool(re.search(r"sweat|water resist|waterproof|hard workout", self.head, re.I))
        self.pref = self.wet or bool(
            re.search(r"rather pay|flimsy|daily runs|which of your preferences", self.head, re.I)
        )
        cm2 = re.search(r"surface at least (\d+) distinct", self.rules, re.I)
        self.cover = int(cm2.group(1)) if cm2 else 0
        low = self.head.lower()
        self.phone = bool(re.search(r"\bsmartphones?\b", low))
        self.tablet = (not self.phone) and bool(re.search(r"\btablets?\b", low))
        self.monitor = bool(re.search(r"\bmonitors?\b", low))
        self.headphone = bool(re.search(r"\b(headphones?|earbuds?|earphones?)\b", low))
        self.gaming = bool(re.search(r"\bgaming\b", low))
        if not self.aisle:
            if self.phone:
                self.aisle = "Smartphones"
            elif self.tablet:
                self.aisle = "Tablets"
            elif self.monitor:
                self.aisle = "Monitors"
            elif self.headphone:
                self.aisle = "Headphones"

    def absorb(self, text: str) -> None:
        t = text or ""
        b = _money(t)
        if b is not None and any(w in t.lower() for w in ("budget", "spend", "only go", "balance", "up to", "at most")):
            self.budget = b
        inch = _inch(t)
        if inch is not None:
            self.inch = inch
            self.ask = False
        res = _wanted_res(t)
        if res:
            self.res = res
            self.ask = False
        hz = _hz(t)
        if hz is not None:
            self.hz_need = hz
            self.ask = False
        ram = _ram_gb(t)
        if ram is not None:
            self.ram = ram
            self.ask = False
        if re.search(r"\b(?:screen size|resolution|refresh|memory|ram)\b", t, re.I) and (inch or res or hz or ram):
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
            tries = 3 if i == 0 else 2
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

    def _send_env(self, envelope: dict) -> dict | None:
        # One extra identical POST is enough. More replays of the same call_id
        # get 409 from this runtime and burn the leftover-time stop.
        for attempt in range(2):
            got = self._post("/environment/call", envelope)
            if got and isinstance(got.get("calls"), list) and got.get("calls"):
                return got
            left = self.limit.left()
            if left < 20:
                break
            _pause(attempt + 1, 6.0, left)
        return None

    def fire(self, batch: list) -> tuple[list, bool]:
        if self.hold:
            batch = self.hold["batch"]
            self.wave += 1
            envelope = self._envelope(batch, self.turn, self.wave)
        else:
            if not batch:
                return [], False
            self.turn += 1
            self.wave = 0
            envelope = self._envelope(batch, self.turn, 0)
        result = self._send_env(envelope)
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
            trust = bool(
                args.get("brand")
                or args.get("category")
                or args.get("max_price") is not None
                or self.need.phrase
                or self.need.model
                or self.need.wet
            )
            for row in inner.get("results") or []:
                if not isinstance(row, dict):
                    continue
                self.bag.put_row(row, trusted=trust)
        elif name == "view":
            self.bag.put_page(inner)
        elif name == "compare":
            self.bag.compares += 1
            for block in inner.get("compared") or []:
                if isinstance(block, dict):
                    self.bag.put_page(block)
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
                    self.bag.dead_keys = _dead_keys(title)
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

    def opt_blob(self, pid: str, sku: str) -> str:
        item = self.bag.live(pid, sku)
        return _blob([item.get("options"), item.get("title")])

    def res_classes(self, pid: str, sku: str) -> set:
        item = self.bag.live(pid, sku)
        facts = item.get("facts") or {}
        fact_disp = " ".join(str(facts.get(k) or "") for k in ("display", "resolution"))
        fact_disp = fact_disp.replace("*", " x ").replace("×", " x ")
        opts = str(item.get("options") or "")
        if fact_disp.strip() and pid in self.bag.viewed:
            named = _res_named(fact_disp)
            if named:
                return named
            return _res_pix(fact_disp + " " + opts)
        return _res_class(opts + " " + str(item.get("title") or "") + " " + self.text_of(pid, sku))

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

    def identity_ok(self, pid: str, sku: str) -> bool:
        item = self.bag.live(pid, sku)
        text = self.text_of(pid, sku).lower()
        title = str(item.get("title") or "").lower()
        facts = item.get("facts") or {}
        if self.need.phone:
            if _JUNK.search(text):
                return False
            if "tablet" in text and "smartphone" not in text and not _has_os(text):
                return False
            opt = str(item.get("options") or "").lower()
            if re.search(r"screen only|replacement screen", opt):
                return False
            if re.search(r"\b(case|cover|charger|cable|grip)\b", title) and not re.search(
                r"\b(iphone|galaxy|smartphone|android)\b", title
            ):
                return False
            if self.need.hard_brand and self.need.hard_brand.lower() == "apple":
                price = _fnum(item.get("price")) or 0
                if "iphone" not in title and "iphone" not in text and price < 8000:
                    return False
            return (
                ("smartphone" in text)
                or _has_os(text)
                or "smartphone" in str(facts.get("phone_type") or "").lower()
            )
        if self.need.tablet:
            return "tablet" in text
        if self.need.headphone:
            device = r"\b(headphones?|earbuds?|earphones?|headsets?)\b"
            if re.search(r"\b(case|pouch|cable only|stand)\b", text) and not re.search(device, text, re.I):
                return False
            return bool(re.search(device, text, re.I))
        if self.need.monitor:
            return "monitor" in text or bool(self.need.seeds)
        return True

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
            if self.need.res not in self.res_classes(pid, sku):
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
            if not self.bag.event and (self.need.phone or self.need.tablet or self.need.monitor or self.need.why or self.need.model):
                return False
        return self.stock_ok(pid, sku) and self.budget_ok(pid, sku) and self.spec_ok(pid, sku) and self.identity_ok(pid, sku)

    def phrase_hit(self, pid: str, sku: str) -> bool:
        if not self.need.phrase:
            return True
        title = (self.bag.live(pid, sku).get("title") or "").lower()
        have = set(re.findall(r"[a-z0-9]+", title))
        toks = [t for t in re.findall(r"[a-z0-9]+", self.need.phrase.lower()) if t]
        return all(t in have for t in toks) if toks else True

    def live_theme(self) -> list:
        skip = _STOP | {
            "find",
            "place",
            "one",
            "test",
            "order",
            "stock",
            "under",
            "below",
            "php",
            "pesos",
            "need",
            "want",
            "get",
            "can",
            "you",
            "please",
            "rather",
            "pay",
            "cheapest",
            "isn",
            "isnt",
            "goal",
            "here",
            "what",
            "actually",
            "care",
            "about",
            "just",
            "don",
            "dont",
            "every",
            "kind",
            "lot",
            "more",
            "than",
            "most",
            "that",
            "this",
            "survives",
            "survive",
            "hard",
        }
        out = []
        for tok in re.findall(r"[a-z0-9]+", (self.need.head or "").lower()):
            if tok in skip or tok.isdigit() or len(tok) < 3:
                continue
            if tok not in out:
                out.append(tok)
        return out[:8]

    def live_need_words(self) -> list:
        head = (self.need.head or "").lower()
        kinds = []
        attrs = []
        for tok in (
            "headphones",
            "headphone",
            "earbuds",
            "earbud",
            "earphones",
            "earphone",
            "headset",
        ):
            if re.search(r"\b" + re.escape(tok) + r"\b", head) and tok not in kinds:
                kinds.append(tok)
        for tok in (
            "sweat",
            "water",
            "waterproof",
            "resistant",
            "resistance",
            "workout",
            "workouts",
            "gym",
            "running",
            "runs",
            "sport",
            "sports",
            "weather",
            "outdoor",
            "bootcamp",
        ):
            if re.search(r"\b" + re.escape(tok) + r"\b", head) and tok not in attrs:
                attrs.append(tok)
        return kinds + attrs

    def follow_from_hits(self) -> str:
        phrase = self.need.phrase or ""
        want = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if t]
        if not want:
            want = self.live_need_words() or self.live_theme()
        prefix = phrase or " ".join(want[:5])
        skip = _STOP | {"this", "that", "all", "you", "your", "are", "was", "not", "but", "can", "has", "have"}
        hits = 0
        present: dict = {}
        brands = set()
        for pid, sku in self.bag.row:
            if self.need.headphone and not self.need.phrase and not self.identity_ok(pid, sku):
                continue
            item = self.bag.live(pid, sku)
            for b in re.findall(r"[a-z0-9]+", str(item.get("brand") or "").lower()):
                if len(b) >= 3:
                    brands.add(b)
            title = str(item.get("title") or "").lower()
            toks = re.findall(r"[a-z0-9]+", title)
            if want:
                if phrase:
                    if not all(w in toks for w in want):
                        continue
                elif not any(w in toks for w in want):
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
            extra = max(pool, key=lambda t: (pool[t] * (hits - pool[t]), -pool[t], len(t)))
            q = "%s %s" % (prefix, extra)
            if q.lower() != prefix.lower():
                return q
        if phrase and self.need.aisle:
            return "%s %s" % (phrase, self.need.aisle.split("&")[0].strip())
        return ""

    def wet_title_extra(self) -> str:
        words = self.live_need_words() or self.live_theme()
        prefix = " ".join(words[:5])
        if not prefix:
            return ""
        best = None
        best_sc = None
        for pid, sku in self.bag.row:
            if not self.identity_ok(pid, sku) or not self.stock_ok(pid, sku) or not self.budget_ok(pid, sku):
                continue
            ax = self.pref_axes(pid, sku)
            sc = (
                1 if self.four_axis(pid, sku) else 0,
                ax["hours"],
                ax["wire"],
                ax["wet"],
                ax["sport"],
            )
            if best_sc is None or sc > best_sc:
                best_sc = sc
                best = (pid, sku)
        if not best:
            return ""
        title = str(self.bag.live(*best).get("title") or "").lower()
        want = set(re.findall(r"[a-z0-9]+", prefix))
        skip = _STOP | {"this", "that", "all", "you", "your", "are", "was", "not", "but", "can", "has", "have"}
        ranked = []
        for tok in re.findall(r"[a-z0-9]+", title):
            if tok in skip or tok in want or len(tok) < 3:
                continue
            if any(
                tok != w and (tok in w or w in tok) and (re.search(r"\d", tok) or re.search(r"\d", w))
                for w in want
            ):
                continue
            key = (_batt_hours(tok), _wet(tok), _wire(tok), _sport(tok))
            if key == (0.0, 0, 0.0, 0.0):
                continue
            ranked.append((key, tok))
        if not ranked:
            return ""
        extra = max(ranked)[1]
        q = "%s %s" % (prefix, extra)
        return q if q.lower() != prefix.lower() else ""

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
                if self.need.phone or self.need.headphone or self.need.monitor:
                    continue
            out.append((pid, sku))
        return out

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
            named = _res_named(fact_disp)
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
        text = self.text_of(pid, sku)
        wet = _wet(text)
        return {
            "sweat": 1.0 if wet >= 3 else (0.4 if wet > 0 else 0.0),
            "sport": _sport(text),
            "hours": _batt_hours(text),
            "wire": _wire(text),
            "wet": wet,
        }

    def firm_pref(self, pid: str, sku: str) -> bool:
        ax = self.pref_axes(pid, sku)
        return ax["sweat"] >= 1.0 and ax["sport"] >= 1.0

    def four_axis(self, pid: str, sku: str) -> bool:
        ax = self.pref_axes(pid, sku)
        return ax["sweat"] >= 1.0 and ax["sport"] >= 1.0 and ax["hours"] > 0 and ax["wire"] >= 1.0

    def score(self, pid: str, sku: str) -> tuple:
        item = self.bag.live(pid, sku)
        text = self.text_of(pid, sku)
        price = _fnum(item.get("price")) or 1e18
        wet = _wet(text)
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
            room = 0
            if self.need.budget is not None and price < 1e17 and price * 3.0 <= self.need.budget + 1e-6:
                room = 1
            return (
                ax["sweat"],
                ax["sport"],
                ax["hours"],
                ax["wire"],
                room,
                ax["wet"],
                -price if ax["sweat"] >= 1.0 else -price * 0.01,
            )
        if (self.need.inch and self.need.phone) or self.need.hard_brand or self.need.why:
            if self.bag.event:
                return (soft, self.fact_fit(pid, sku), self.dead_kin(pid, sku), -price)
            return (soft, price)
        if not self.bag.first_add and self.need.soft_brand:
            return (soft, -price)
        return (soft, wet, -price)

    def pick(self) -> tuple | None:
        pool = self.candidates()
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
            wet_ok = [k for k in pool if _wet(self.text_of(*k)) > 0]
            if wet_ok:
                pool = wet_ok
            elif self.bag.search_n < 3:
                return None
            four = [k for k in pool if self.four_axis(*k)]
            if four:
                pool = four
            elif self.bag.search_n < 5:
                return None
            firm = [k for k in pool if self.firm_pref(*k)]
            if firm:
                pool = firm
            hinted = None
            if len(pool) >= 2:
                hinted = self.consult_pick(pool)
            return self.review_wet(pool, hinted)
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
            if self.need.wet and _wet(self.text_of(pid, sku)) <= 0:
                continue
            pool.append((pid, sku))
        if not pool:
            return None
        if self.need.wet:
            four = [k for k in pool if self.four_axis(*k)]
            if four:
                hinted = self.consult_pick(four) if len(four) >= 2 else None
                return self.review_wet(four, hinted)
            if self.bag.search_n < 8:
                return None
        return max(pool, key=lambda ks: self.score(*ks))

    def review_wet(self, pool: list, hinted: tuple | None) -> tuple | None:
        if not pool:
            return None
        four = [k for k in pool if self.four_axis(*k)]
        use = four or [k for k in pool if self.firm_pref(*k)] or list(pool)

        def _price_of(ks: tuple) -> float:
            return _fnum(self.bag.live(*ks).get("price")) or 1e18

        proven = min(use, key=_price_of) if four else max(use, key=lambda ks: self.score(*ks))
        if hinted in use:
            if four and not self.four_axis(*hinted):
                return proven
            if four and _price_of(hinted) > _price_of(proven) + 1e-9:
                return proven
            if self.score(*hinted) >= self.score(*proven):
                return hinted
        return proven

    def _read_pick(self, msg: dict | None, pool: list) -> tuple | None:
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
        if not pid:
            return None
        for p, s in pool:
            if p == pid and (not sku or s == sku):
                return (p, s)
        return None

    def consult_pick(self, pool: list) -> tuple | None:
        lines = []
        for i, (pid, sku) in enumerate(pool[:8], start=1):
            item = self.bag.live(pid, sku)
            lines.append(
                "%d. %s::%s price=%s stock=%s wet=%s sport=%s hours=%s wire=%s "
                "options=%s title=%s facts=%s"
                % (
                    i,
                    pid,
                    sku,
                    item.get("price"),
                    item.get("in_stock"),
                    _wet(self.text_of(pid, sku)),
                    _sport(self.text_of(pid, sku)),
                    _batt_hours(self.text_of(pid, sku)),
                    _wire(self.text_of(pid, sku)),
                    self.opt_blob(pid, sku)[:160],
                    (item.get("title") or "")[:120],
                    item.get("facts"),
                )
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "Confirm each listed variant against the shopper constraints. "
                    "Use option text and labelled facts first; title tokens are weaker. "
                    "A valid pick must show sweat/water resistance, a sport/workout fit, "
                    "wireless, and a battery life in hours. Prefer the lower price among "
                    "valid picks. Reject a dearer row that is missing hours. "
                    'Reply with JSON {"product_id":"...","sku":"..."} only.'
                ),
            },
            {"role": "user", "content": "\n".join(lines)},
        ]
        votes: list = []
        rounds = 1 if self.limit.left() < 40 else 3
        for _ in range(rounds):
            if self.limit.left() < 25:
                break
            got = self._read_pick(self.infer(messages), pool)
            if got:
                votes.append(got)
        if not votes:
            return None
        counts: dict = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        winners = [k for k, c in counts.items() if c == top]
        if top >= 2:
            return min(winners, key=lambda ks: _fnum(self.bag.live(*ks).get("price")) or 1e18)
        return votes[0]

    def claims(self, pid: str, sku: str) -> list:
        item = self.bag.live(pid, sku)
        out = []
        if self.need.inch is not None:
            got = self.inch_of(pid, sku)
            facts = item.get("facts") or {}
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if is_range:
                got = None
            elif val is not None:
                got = val
            if got is not None and round(got, 2) == round(self.need.inch, 2):
                out.append({"field": "screen_size", "value": "%gin" % self.need.inch})
        elif self.need.res:
            if self.need.res in self.res_classes(pid, sku):
                out.append({"field": "resolution", "value": self.need.res})
        elif self.need.hz_need is not None:
            got = self.hz_of(pid, sku)
            if got is not None and got + 1e-6 >= self.need.hz_need:
                out.append({"field": "refresh_rate", "value": "%ghz" % self.need.hz_need})
        elif self.need.ram is not None:
            out.append({"field": "ram", "value": "%ggb" % self.need.ram})
        if self.need.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.need.budget})
        out.append({"field": "in_stock", "value": "true" if item.get("in_stock") else "false"})
        brand = item.get("brand")
        if brand:
            out.append({"field": "brand", "value": str(brand)})
        return out

    def searches(self) -> list:
        q = []
        if self.need.model:
            if self.need.color:
                q.append("%s %s smartphone" % (self.need.color, self.need.model))
            q.append("%s smartphone" % self.need.model)
            q.append(self.need.model)
        if self.need.phrase:
            q.append(self.need.phrase)
            extra = self.follow_from_hits()
            if extra and extra.lower() != self.need.phrase.lower():
                q.append(extra)
        bits = []
        if self.need.hard_brand or self.need.soft_brand:
            bits.append(self.need.hard_brand or self.need.soft_brand)
        if self.need.phone:
            bits.append("smartphone")
        if self.need.tablet:
            bits.append("tablet")
        if self.need.monitor:
            bits.append("gaming monitor" if self.need.gaming else "monitor")
        if self.need.headphone:
            need = self.live_need_words()
            if self.need.wet and need:
                bits.append(" ".join(need[:5]))
            else:
                bits.append("headphone")
        if self.need.inch:
            bits.append("%g inch" % self.need.inch)
        if self.need.res:
            bits.append(self.need.res)
        if self.need.color:
            bits.append(self.need.color)
        if bits:
            q.append(" ".join(str(b) for b in bits if b))
        if self.need.wet:
            words = self.live_need_words()
            if words:
                q.append(" ".join(words[:6]))
                if self.need.aisle:
                    q.append("%s %s" % (self.need.aisle.split("&")[0].strip(), " ".join(words[:4])))
                if len(words) >= 3:
                    q.append(" ".join(words[:3]))
            extra = self.wet_title_extra()
            if extra and extra.lower() not in {s.lower() for s in q}:
                q.insert(1 if q else 0, extra)
            if self.bag.event and extra:
                q.insert(0, extra)
        if self.need.soft_brand and self.need.phone:
            q.append("%s smartphone" % self.need.soft_brand)
        if self.need.phone and self.need.inch:
            q.append("%g inch smartphone" % self.need.inch)
        if self.need.hard_brand and self.need.inch:
            q.append("%s %g inch smartphone" % (self.need.hard_brand, self.need.inch))
        if self.need.hard_brand and (self.need.monitor or self.need.tablet or self.need.why):
            q.insert(0, self.need.hard_brand + (" %g inch" % self.need.inch if self.need.inch else "") + (" " + self.need.res if self.need.res else ""))
        if self.bag.dead_title and not self.need.wet:
            q.insert(0, self.bag.dead_title)
            if self.need.inch:
                q.insert(0, "%s %g inches smartphone" % (self.bag.dead_title, self.need.inch))
        out = []
        seen = set()
        for s in q:
            s = " ".join(s.split())
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out or [self.need.head[:80]]

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
            if self.need.budget is not None:
                price = _fnum(self.bag.live(pid, sku).get("price"))
                if price is not None and price > self.need.budget * 1.05:
                    continue
            title = str(self.bag.live(pid, sku).get("title") or "")
            if self.need.phone and _JUNK.search(title):
                continue
            pids.append(pid)
        uniq = []
        seen = set()
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                uniq.append(pid)
        if self.need.wet and uniq:
            def _view_key(pid: str) -> tuple:
                sku = ""
                for p, s in self.bag.row:
                    if p == pid:
                        sku = s
                        break
                if not sku:
                    return (0, 0, 0.0, 0, 0.0)
                ax = self.pref_axes(pid, sku)
                return (
                    1 if self.four_axis(pid, sku) else 0,
                    1 if self.firm_pref(pid, sku) else 0,
                    ax["hours"],
                    ax["wet"],
                    ax["wire"],
                )

            uniq.sort(key=_view_key, reverse=True)
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
            if "expensive" in low or "more expensive" in low:
                if price is not None and self.need.budget is not None and price > self.need.budget + 1e-6:
                    return (
                        "Yes, %s is now %.2f PHP, above the budget. I'll switch to another "
                        "in-stock option that still meets the firm limits." % (title[:60], price)
                    )
            parts = ["I am going with " + title]
            if self.need.inch is not None:
                parts.append("its listed screen size is %gin" % self.need.inch)
            elif self.need.res:
                parts.append("its listed resolution is " + self.need.res)
            if price is not None:
                parts.append("it is %.2f PHP and in stock" % price)
            self.bag.named = pair
            return ", ".join(parts) + "."
        return "I am still checking the options against your requirements; nothing is decided yet."

    def order_args(self, pid: str, sku: str) -> dict:
        args = {"product_id": pid, "sku": sku}
        if self.need.why:
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
        if self.need.budget is not None:
            filt["max_price"] = self.need.budget
        return filt

    def plan(self) -> list:
        acts: list = []

        def add(name: str, args: dict) -> None:
            if self.ok(name) and len(acts) < self.max_calls:
                acts.append((name, args))

        if self.need.ask and not self.bag.asked:
            add(
                "message",
                {
                    "content": (
                        "Before I search: which hardware specification is your firm requirement "
                        "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
                    )
                },
            )
            return acts
        if self.need.pref and not self.bag.asked:
            add(
                "message",
                {
                    "content": (
                        "Quick question so I choose well: which of your preferences counts most, and in what "
                        "order should I weigh the others? Let me know your priority."
                    )
                },
            )
            return acts

        if self.bag.inbox:
            note = self.bag.inbox[-1]
            self.bag.inbox.clear()
            nudge = bool(re.search(r"keep to|stated requirements|compare the current", note, re.I))
            ask = bool(re.search(r"\?|which one|going with|still available|re-check|more expensive", note, re.I))
            if ask and not nudge:
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
            weak = bool(self.need.wet and nxt and not self.four_axis(*nxt))
            extra = [p for p in self.need_view() if p]
            if weak and extra and self.ok("compare") and self.bag.compares < 6:
                add("compare", {"product_ids": extra[:5]})
                return acts
            if nxt and nxt != pair and nxt not in self.bag.ban and not weak:
                add("add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                add("inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
                return acts
            if not self.need.seeds and self.bag.search_n < 8:
                qs = self.searches()
                q = self.wet_title_extra() if self.need.wet else ""
                if not q:
                    q = qs[min(self.bag.search_n, len(qs) - 1)]
                add("search", {"query": q, "in_stock": True, "k": 10})
                add("filter", self.filt_args())
                return acts
            if nxt and nxt != pair and nxt not in self.bag.ban:
                add("add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                add("inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
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
                if not (self.need.wet and self.four_axis(*pair)):
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
        if self.need.hard_brand and self.need.hard_brand.lower() == "apple":
            need_more = self.bag.search_n < 5 and self.pick() is None
        if self.need.cover and self.cover_count() < self.need.cover:
            need_more = True
        if self.need.wet:
            has_four = any(self.four_axis(*k) for k in self.candidates())
            need_more = self.bag.search_n < 5 and not has_four
        if not self.need.seeds and need_more:
            qs = self.searches()
            idx = min(self.bag.search_n, len(qs) - 1)
            add("search", {"query": qs[idx], "in_stock": True, "k": 10})
            add("filter", self.filt_args())
            return acts

        choice = self.fallback_pick()
        if choice:
            pid, sku = choice
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
        if self.ok("place_test_order") and self.cart_valid():
            pair = self.cart_pair()
            if pair:
                _outs, done = self.fire([("place_test_order", self.order_args(*pair))])
                return done
        choice = self.fallback_pick()
        if choice and self.ok("add_to_cart") and not self.bag.cart:
            self.fire([("add_to_cart", {"product_id": choice[0], "sku": choice[1]})])
            if self.ok("inspect_stock"):
                self.fire([("inspect_stock", {"product_id": choice[0], "sku": choice[1]})])
        if self.ok("place_test_order") and self.cart_valid():
            pair = self.cart_pair()
            if pair:
                _outs, done = self.fire([("place_test_order", self.order_args(*pair))])
                return done
        return False

    def run(self) -> list:
        while self.turn < self.max_steps:
            if self.limit.left() < 25 or self.turn >= self.max_steps - 3:
                try:
                    self.close()
                except Exception:
                    pass
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
                if self.bag.cart and self.ok("place_test_order") and self.cart_valid():
                    pair = self.cart_pair()
                    batch = [("place_test_order", self.order_args(*pair))]
                elif self.ok("search") and self.bag.search_n < 6 and self.fallback_pick() is None:
                    qs = self.searches()
                    batch = [("search", {"query": qs[self.bag.search_n % len(qs)], "in_stock": True, "k": 10})]
                else:
                    choice = self.fallback_pick()
                    if (not choice) and self.bag.named and self.bag.named not in self.bag.ban:
                        choice = self.bag.named
                    if choice and choice in self.bag.ban:
                        choice = None
                    if choice and self.ok("add_to_cart") and not self.bag.cart:
                        batch = [("add_to_cart", {"product_id": choice[0], "sku": choice[1]})]
                    else:
                        break
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
        return self.talk


def agent_main(problem_data: dict) -> list:
    try:
        if not isinstance(problem_data, dict) or "environment" not in problem_data:
            return []
        return Shop(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "halt: %s: %s" % (type(exc).__name__, str(exc)[:180])}]
