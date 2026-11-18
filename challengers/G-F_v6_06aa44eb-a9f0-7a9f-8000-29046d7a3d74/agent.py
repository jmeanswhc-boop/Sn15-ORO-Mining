import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# Failover only. Same decide() stages for every model name.
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
_LIMIT_S = 270.0
_WRITE = frozenset({"add_to_cart", "remove_from_cart", "place_test_order"})
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


def _sport(text: str) -> float:
    return 1.0 if re.search(r"\b(?:sports?|running|workouts?|gym|secure)\b", text or "", re.I) else 0.0


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


class Goal:
    def __init__(self, query: str) -> None:
        raw = query or ""
        head, _, rules = raw.partition("\n\nTask rules:")
        if not rules:
            head, _, rules = raw.partition("Task rules:")
        self.head = head.strip()
        self.rules = rules
        self.budget = _money(self.head) or _money(raw)
        self.budget_cut = False
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
            if self.budget is not None and b + 1e-6 < self.budget:
                self.budget_cut = True
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


class Catalog:
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
        self.found: list = []
        self.notes: list = []
        self.confirm_n = 0
        self.final_look = False

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


class SoftTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not SoftTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not SoftTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not SoftTap.LIVE:
            return None
        return self.peer
class HardTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not HardTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not HardTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not HardTap.LIVE:
            return None
        return self.peer
class BudgetTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not BudgetTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not BudgetTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not BudgetTap.LIVE:
            return None
        return self.peer
class PhraseTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not PhraseTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not PhraseTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not PhraseTap.LIVE:
            return None
        return self.peer
class AisleTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not AisleTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not AisleTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not AisleTap.LIVE:
            return None
        return self.peer
class BrandTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not BrandTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not BrandTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not BrandTap.LIVE:
            return None
        return self.peer
class ColorTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not ColorTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not ColorTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not ColorTap.LIVE:
            return None
        return self.peer
class ModelTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not ModelTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not ModelTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not ModelTap.LIVE:
            return None
        return self.peer
class HzTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not HzTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not HzTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not HzTap.LIVE:
            return None
        return self.peer
class InchTap:
    LIVE = False
    def __init__(self) -> None:
        self.peer = None
        self.host = None
    def bind(self, host) -> None:
        if not InchTap.LIVE:
            return
        self.host = host
    def handoff(self, *nodes):
        if not InchTap.LIVE:
            return None
        node = nodes[0] if nodes else self.peer
        if node is None:
            return None
        return node.accept(self)
    def accept(self, _node):
        if not InchTap.LIVE:
            return None
        return self.peer
class Episode:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.bind = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = {(t.get("function") or {}).get("name") for t in (self.policy.get("tools") or []) if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.pid = str(self.bind.get("session_id") or problem.get("problem_id") or problem.get("id") or "p")
        self.goal = Goal(str(self.policy.get("query") or ""))
        self.mem = Catalog()
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
        for a, b in self.goal.seeds:
            self.mem.put_row({"product_id": a, "sku": b, "in_stock": True}, trusted=True)
        self.extra_taps = [SoftTap(), HardTap(), BudgetTap(), PhraseTap(), AisleTap(), BrandTap(), ColorTap(), ModelTap(), HzTap(), InchTap()]
        for _tap in self.extra_taps:
            _tap.bind(self)
        for _a, _b in zip(self.extra_taps, self.extra_taps[1:] + [self.extra_taps[0]]):
            _a.peer = _b
        self.extra_taps[0].handoff(*self.extra_taps)

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
        return any(n in _WRITE for n, _a in batch)

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

    def _accept(self, batch: list, result: dict) -> tuple[list, bool]:
        self.hold = None
        self.hold_n = 0
        self.talk.append({"role": "assistant", "batch": batch, "environment_result": result})
        um = result.get("user_message")
        if isinstance(um, dict) and um.get("content"):
            try:
                self.goal.absorb(str(um["content"]))
                self.mem.inbox.append(str(um["content"]))
                note = str(um["content"]).strip()
                if note:
                    self.mem.notes.append(note)
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
                self.mem.search_n += 1
                self.mem.hits += 1
                q = str((args or {}).get("query") or "").strip()
                if q:
                    self.mem.sent_q.append(q)
            elif name == "filter":
                self.mem.hits += 1
            elif name == "add_to_cart":
                pid = str((args or {}).get("product_id") or "")
                sku = str((args or {}).get("sku") or "")
                if pid and sku:
                    if not self.mem.cart:
                        self.mem.cart = [{"product_id": pid, "sku": sku}]
                    self.mem.first_add = True
            elif name == "remove_from_cart":
                pid = str((args or {}).get("product_id") or "")
                sku = str((args or {}).get("sku") or "")
                if pid and sku:
                    self.mem.ban.add((pid, sku))
                    self.mem.cart = [
                        c
                        for c in self.mem.cart
                        if not (str(c.get("product_id")) == pid and str(c.get("sku")) == sku)
                    ]
            elif name == "compare":
                self.mem.compares += 1
            elif name == "message":
                self.mem.asked = True
            elif name == "view":
                pid = str((args or {}).get("product_id") or "")
                if pid:
                    self.mem.viewed.add(pid)
            elif name == "inspect_stock":
                pid = str((args or {}).get("product_id") or "")
                sku = str((args or {}).get("sku") or "")
                if pid and sku and (pid, sku) not in self.mem.tick:
                    self.mem.tick[(pid, sku)] = {}
            elif name == "inspect_cart":
                self.mem.rechecked = True
        self.hold = None
        self.hold_n = 0
        self.talk.append({"role": "assistant", "content": "env committed", "batch": batch})
        return done

    def fire(self, batch: list) -> tuple[list, bool]:
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
            self.mem.hits += 1
            self.mem.search_n += 1
            q = str(args.get("query") or "").strip()
            if q:
                self.mem.sent_q.append(q)
            page = []
            for row in inner.get("results") or []:
                if isinstance(row, dict):
                    self.mem.put_row(row, trusted=True)
                    page.append(
                        (
                            str(row.get("product_id") or ""),
                            str(row.get("sku") or ""),
                            str(row.get("title") or ""),
                            str(row.get("brand") or ""),
                        )
                    )
            if self.mem.search_n == 1:
                self.mem.found = page
        elif name == "filter":
            self.mem.hits += 1
            trust = bool(
                args.get("brand")
                or args.get("category")
                or args.get("max_price") is not None
                or self.goal.phrase
                or self.goal.model
                or self.goal.wet
            )
            for row in inner.get("results") or []:
                if not isinstance(row, dict):
                    continue
                self.mem.put_row(row, trusted=trust)
        elif name == "view":
            self.mem.put_page(inner)
        elif name == "compare":
            self.mem.compares += 1
            for block in inner.get("compared") or []:
                if isinstance(block, dict):
                    self.mem.put_page(block)
        elif name == "inspect_stock":
            pid = str(args.get("product_id") or inner.get("product_id") or "")
            sku = str(args.get("sku") or inner.get("sku") or "")
            if pid and sku:
                self.mem.tick[(pid, sku)] = {
                    "price": inner.get("price"),
                    "in_stock": inner.get("in_stock"),
                }
        elif name == "add_to_cart":
            added = inner.get("added") if isinstance(inner.get("added"), dict) else {}
            cart = inner.get("cart") if isinstance(inner.get("cart"), list) else []
            self.mem.cart = cart
            self.mem.first_add = True
            self.mem.rechecked = False
            ev = inner.get("market_event")
            if isinstance(ev, dict):
                self.mem.event = True
                self.mem.need_rescan = True
                tgt = ev.get("target") if isinstance(ev.get("target"), dict) else {}
                pid = str(tgt.get("product_id") or added.get("product_id") or args.get("product_id") or "")
                sku = str(tgt.get("sku") or added.get("sku") or args.get("sku") or "")
                if pid and sku:
                    self.mem.dead_stem = _sku_stem(sku)
                    title = str((self.mem.live(pid, sku) or {}).get("title") or "")
                    self.mem.dead_title = " ".join(title.split()[:8])
                    self.mem.dead_keys = _dead_keys(title)
                    self.mem.tick[(pid, sku)] = {
                        "price": ev.get("new_price", added.get("price")),
                        "in_stock": ev.get("in_stock_after", added.get("in_stock")),
                    }
            elif added:
                pid = str(added.get("product_id") or args.get("product_id") or "")
                sku = str(added.get("sku") or args.get("sku") or "")
                if pid and sku:
                    self.mem.tick[(pid, sku)] = {
                        "price": added.get("price"),
                        "in_stock": added.get("in_stock"),
                    }
        elif name == "inspect_cart":
            cart = inner.get("cart")
            if isinstance(cart, list):
                self.mem.cart = cart
            elif isinstance(inner.get("items"), list):
                self.mem.cart = inner["items"]
            self.mem.rechecked = True
        elif name == "remove_from_cart":
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and sku:
                self.mem.ban.add((pid, sku))
                if self.mem.named == (pid, sku):
                    self.mem.named = None
                self.mem.cart = [
                    c
                    for c in self.mem.cart
                    if not (str(c.get("product_id")) == pid and str(c.get("sku")) == sku)
                ]
        elif name == "message":
            self.mem.asked = True
        for d in _walk(inner):
            if d.get("product_id") and (d.get("sku") or d.get("title") or d.get("variants")):
                if "variants" in d:
                    self.mem.put_page(d)
                elif d.get("sku") and name in ("search", "view", "compare", "inspect_stock"):
                    self.mem.put_row(d, trusted=True)

    def text_of(self, pid: str, sku: str) -> str:
        item = self.mem.live(pid, sku)
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
        item = self.mem.live(pid, sku)
        return _blob([item.get("options"), item.get("title")])

    def res_classes(self, pid: str, sku: str) -> set:
        item = self.mem.live(pid, sku)
        facts = item.get("facts") or {}
        fact_disp = " ".join(str(facts.get(k) or "") for k in ("display", "resolution"))
        fact_disp = fact_disp.replace("*", " x ").replace("×", " x ")
        opts = str(item.get("options") or "")
        if fact_disp.strip() and pid in self.mem.viewed:
            named = _res_named(fact_disp)
            if named:
                return named
            return _res_pix(fact_disp + " " + opts)
        return _res_class(opts + " " + str(item.get("title") or "") + " " + self.text_of(pid, sku))

    def inch_of(self, pid: str, sku: str) -> float | None:
        item = self.mem.live(pid, sku)
        facts = item.get("facts") or {}
        from_opt = _inch(str(item.get("options") or "")) or _inch(str(item.get("title") or ""))
        if self.goal.rank:
            return from_opt or _inch(str(facts.get("size") or ""))
        val, is_range = _size_fact(str(facts.get("size") or ""))
        if is_range:
            return None
        if val is not None:
            return val
        return from_opt

    def hz_of(self, pid: str, sku: str) -> float | None:
        item = self.mem.live(pid, sku)
        facts = item.get("facts") or {}
        return _hz(self.opt_blob(pid, sku)) or _hz(str(facts.get("refresh_rate") or facts.get("display") or ""))

    def identity_ok(self, pid: str, sku: str) -> bool:
        item = self.mem.live(pid, sku)
        text = self.text_of(pid, sku).lower()
        title = str(item.get("title") or "").lower()
        facts = item.get("facts") or {}
        if self.goal.phone:
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
            if self.goal.hard_brand and self.goal.hard_brand.lower() == "apple":
                price = _fnum(item.get("price")) or 0
                if "iphone" not in title and "iphone" not in text and price < 8000:
                    return False
            return (
                ("smartphone" in text)
                or _has_os(text)
                or "smartphone" in str(facts.get("phone_type") or "").lower()
            )
        if self.goal.tablet:
            return "tablet" in text
        if self.goal.headphone:
            device = r"\b(headphones?|earbuds?|earphones?|headsets?)\b"
            if re.search(r"\b(case|pouch|cable only|stand|anti-lost|lanyard|waist bag)\b", text) and not re.search(
                device, text, re.I
            ):
                return False
            return bool(re.search(device, text, re.I))
        if self.goal.monitor:
            return "monitor" in text or bool(self.goal.seeds)
        return True

    def spec_ok(self, pid: str, sku: str) -> bool:
        item = self.mem.live(pid, sku)
        text = self.text_of(pid, sku)
        if self.goal.inch is not None:
            facts = item.get("facts") or {}
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if pid in self.mem.viewed and str(facts.get("size") or "").strip():
                if is_range or val is None or round(val, 2) != round(self.goal.inch, 2):
                    return False
            else:
                sizes = [
                    _inch(str(item.get("options") or "")),
                    val,
                    _inch(str(item.get("title") or "")),
                ]
                if not any(s is not None and round(s, 2) == round(self.goal.inch, 2) for s in sizes):
                    return False
        if self.goal.res:
            if self.goal.res not in self.res_classes(pid, sku):
                return False
        if self.goal.hz_need is not None:
            got = self.hz_of(pid, sku)
            if got is None or got + 1e-6 < self.goal.hz_need:
                return False
        if self.goal.ram is not None:
            got = _ram_gb(str(item.get("options") or "") + " " + text)
            if got is None or abs(got - self.goal.ram) > 0.2:
                return False
        if self.goal.hard_brand:
            title = str(item.get("title") or "").lower()
            brand = str(item.get("brand") or "").lower()
            toks = self.goal.hard_brand.lower().split()
            if not (all(t in brand for t in toks) or all(t in title for t in toks)):
                return False
        return True

    def budget_ok(self, pid: str, sku: str) -> bool:
        item = self.mem.live(pid, sku)
        price = _fnum(item.get("price"))
        if self.goal.budget is None or price is None:
            return price is not None
        return price <= self.goal.budget + 1e-6

    def stock_ok(self, pid: str, sku: str) -> bool:
        item = self.mem.live(pid, sku)
        return bool(item.get("in_stock"))

    def hard_ok(self, pid: str, sku: str) -> bool:
        if (pid, sku) in self.mem.ban:
            return False
        if not sku:
            return False
        if self.goal.rank:
            if (pid, sku) not in self.goal.seeds:
                return False
            if self.hz_of(pid, sku) is None or self.inch_of(pid, sku) is None:
                return False
            return self.stock_ok(pid, sku) and self.budget_ok(pid, sku)
        if self.goal.model and self.mem.dead_stem and _sku_stem(sku) == self.mem.dead_stem:
            return self.stock_ok(pid, sku) and self.budget_ok(pid, sku)
        if pid not in self.mem.viewed and not self.goal.phrase:
            if not self.mem.event and (self.goal.phone or self.goal.tablet or self.goal.monitor or self.goal.why or self.goal.model):
                return False
        return self.stock_ok(pid, sku) and self.budget_ok(pid, sku) and self.spec_ok(pid, sku) and self.identity_ok(pid, sku)

    def phrase_hit(self, pid: str, sku: str) -> bool:
        if not self.goal.phrase:
            return True
        title = (self.mem.live(pid, sku).get("title") or "").lower()
        have = set(re.findall(r"[a-z0-9]+", title))
        toks = [t for t in re.findall(r"[a-z0-9]+", self.goal.phrase.lower()) if t]
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
        for tok in re.findall(r"[a-z0-9]+", (self.goal.head or "").lower()):
            if tok in skip or tok.isdigit() or len(tok) < 3:
                continue
            if tok not in out:
                out.append(tok)
        return out[:8]

    def live_need_words(self) -> list:
        head = (self.goal.head or "").lower()
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
        phrase = self.goal.phrase or ""
        want = [t for t in re.findall(r"[a-z0-9]+", phrase.lower()) if t]
        if not want:
            return ""
        skip = _STOP | {"this", "that", "all", "you", "your", "are", "was", "not", "but", "can", "has", "have"}
        rows = list(self.mem.found) if self.mem.found else []
        if not rows:
            for pid, sku in self.mem.row:
                item = self.mem.live(pid, sku)
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
            if (not any(re.search(r"\d", t) for t in want)) and self.goal.aisle:
                return "%s %s" % (phrase, self.goal.aisle.split("&")[0].strip())
            return ""

        if not hits:
            return _aisle()

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
            return _aisle()
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
        prefix = " ".join(words[:2] if self.goal.wet else words[:5])
        if not prefix:
            return ""
        want = set(re.findall(r"[a-z0-9]+", prefix))
        skip = _STOP | {"this", "that", "all", "you", "your", "are", "was", "not", "but", "can", "has", "have"}
        ranked = []
        for pid, sku in self.mem.row:
            if not self.identity_ok(pid, sku) or not self.stock_ok(pid, sku):
                continue
            blob = (
                str(self.mem.live(pid, sku).get("title") or "")
                + " "
                + self.opt_blob(pid, sku)
                + " "
                + self.text_of(pid, sku)
            ).lower()
            whole = (_batt_hours(blob), _wet(blob), _wire(blob), _sport(blob))
            if whole == (0.0, 0, 0.0, 0.0):
                continue
            picked = _hours_tok(blob)
            compact = 1 if picked and not re.search(r"\s", picked) else 0
            if not picked:
                seen = set()
                for tok in re.findall(r"[a-z0-9]+", blob):
                    if tok in skip or tok in want or tok in seen or len(tok) < 3:
                        continue
                    if _wire(tok) or _wet(tok) or _sport(tok):
                        picked = tok
                        break
                    seen.add(tok)
            if not picked:
                continue
            price = _fnum(self.mem.live(pid, sku).get("price")) or 1e18
            hours = whole[0] if whole[0] > 0 else 1e9
            ranked.append(((-compact, hours, price), picked))
        if not ranked:
            return ""
        extra = min(ranked)[1]
        q = "%s %s" % (prefix, extra)
        return q if q.lower() != prefix.lower() else ""

    def next_query(self) -> str:
        sent = {str(q).lower() for q in self.mem.sent_q}
        qs = self.searches()
        extra = ""
        if self.goal.wet:
            extra = self.wet_title_extra()
        elif self.goal.phrase:
            extra = self.follow_from_hits()
        ordered = []
        if qs:
            ordered.append(qs[0])
        if extra and extra.lower() not in {q.lower() for q in ordered}:
            ordered.append(extra)
            tok = extra.split()[-1] if extra.split() else ""
            if self.goal.wet and tok and tok.lower() not in {x.lower() for x in ordered}:
                ordered.append(tok)
        for q in qs[1:]:
            if q.lower() not in {x.lower() for x in ordered}:
                ordered.append(q)
        if self.goal.wet:
            leftover = [t for t in (self.live_need_words() or []) if t.lower() not in " ".join(sent)]
            if leftover:
                span = " ".join((self.live_need_words() or self.live_theme())[:4])
                more = "%s %s" % (span, leftover[0])
                if more.lower() not in {x.lower() for x in ordered}:
                    ordered.append(more)
        for q in ordered:
            if q and q.lower() not in sent:
                return q
        if not qs:
            return (self.goal.head or "")[:80]
        if self.goal.phrase:
            return ""
        return qs[min(self.mem.search_n, len(qs) - 1)]

    def cover_count(self) -> int:
        ids = set()
        for pid, sku in self.mem.row:
            if self.phrase_hit(pid, sku) and self.stock_ok(pid, sku) and self.budget_ok(pid, sku):
                ids.add(pid)
        return len(ids)

    def relax(self, pid: str, sku: str) -> int:
        if not self.goal.model:
            return 0
        text = self.text_of(pid, sku).lower()
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        same_model = bool(
            re.search(r"(?<!\d)" + re.escape(self.goal.model) + r"(?!\d)", text)
            or squash(self.goal.model) in squash(text)
        )
        if self.mem.dead_stem and _sku_stem(sku) == self.mem.dead_stem:
            same_model = True
        shade = self.opt_blob(pid, sku).lower() + " " + str((self.mem.live(pid, sku).get("facts") or {}).get("color") or "")
        same_color = bool(self.goal.color) and re.search(
            r"(?<![a-z])" + re.escape(self.goal.color) + r"(?![a-z])", shade + " " + text
        )
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def candidates(self) -> list:
        out = []
        keys = list(self.mem.row)
        if self.goal.seeds:
            keys = list(self.goal.seeds) + [k for k in keys if k not in self.goal.seeds]
        for pid, sku in keys:
            if not self.hard_ok(pid, sku):
                continue
            if self.goal.phrase and not self.phrase_hit(pid, sku):
                continue
            if not self.goal.seeds and (pid, sku) not in self.mem.trusted and not self.goal.phrase:
                if self.goal.phone or self.goal.headphone or self.goal.monitor:
                    continue
            out.append((pid, sku))
        return out

    def fact_fit(self, pid: str, sku: str) -> int:
        item = self.mem.live(pid, sku)
        facts = item.get("facts") or {}
        n = 0
        if self.goal.inch is not None:
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if (not is_range) and val is not None and round(val, 2) == round(self.goal.inch, 2):
                n += 2
        if self.goal.res:
            fact_disp = " ".join(str(facts.get(k) or "") for k in ("display", "resolution"))
            named = _res_named(fact_disp)
            if self.goal.res in named and len(named) == 1:
                n += 2
            elif self.goal.res in named:
                n += 1
        return n

    def dead_kin(self, pid: str, sku: str) -> int:
        if not self.mem.dead_keys:
            return 0
        text = self.text_of(pid, sku).lower()
        return sum(1 for t in self.mem.dead_keys if t in text)

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

    def paid_four(self) -> bool:
        for pid, sku in self.mem.row:
            if (
                self.budget_ok(pid, sku)
                and self.stock_ok(pid, sku)
                and self.identity_ok(pid, sku)
                and self.four_axis(pid, sku)
            ):
                return True
        return False

    def score(self, pid: str, sku: str) -> tuple:
        item = self.mem.live(pid, sku)
        text = self.text_of(pid, sku)
        price = _fnum(item.get("price")) or 1e18
        wet = _wet(text)
        brand = str(item.get("brand") or "")
        soft = 0
        if self.goal.soft_brand and brand.lower() == self.goal.soft_brand.lower():
            soft = 1
        if self.goal.rank:
            return (self.hz_of(pid, sku) or 0, self.inch_of(pid, sku) or 0, -price)
        if self.goal.model:
            return (-self.relax(pid, sku), soft, -price)
        if self.goal.wet:
            ax = self.pref_axes(pid, sku)
            room = 0
            if self.goal.budget is not None and price < 1e17 and price * 3.0 <= self.goal.budget + 1e-6:
                room = 1
            return (
                ax["sweat"],
                ax["sport"],
                ax["hours"],
                ax["wire"],
                room,
                ax["wet"],
                self.note_hit(pid, sku),
                -price if ax["sweat"] >= 1.0 else -price * 0.01,
            )
        if (self.goal.inch and self.goal.phone) or self.goal.hard_brand or self.goal.why:
            if self.mem.event:
                return (soft, self.fact_fit(pid, sku), self.dead_kin(pid, sku), self.note_hit(pid, sku), -price)
            return (soft, price)
        if not self.mem.first_add and self.goal.soft_brand:
            return (soft, self.note_hit(pid, sku), -price)
        return (soft, wet, self.note_hit(pid, sku), -price)

    def note_hit(self, pid: str, sku: str) -> int:
        blob = self.text_of(pid, sku).lower()
        skip = _STOP | {
            "budget",
            "stock",
            "under",
            "below",
            "php",
            "inch",
            "inches",
            "please",
            "still",
            "need",
            "want",
            "only",
            "sorry",
            "hold",
            "checked",
            "balance",
            "requirements",
            "negotiable",
            "available",
            "going",
            "which",
        }
        n = 0
        seen: set = set()
        for note in self.mem.notes:
            for tok in re.findall(r"[a-z0-9]{4,}", (note or "").lower()):
                if tok in skip or tok in seen or tok.isdigit():
                    continue
                if re.search(r"\b%s\b" % re.escape(tok), blob):
                    seen.add(tok)
                    n += 1
        return n

    def pick(self, vote: bool = True) -> tuple | None:
        pool = self.candidates()
        if not pool:
            return None
        if self.goal.model:
            if self.mem.dead_stem:
                sib = [k for k in pool if _sku_stem(k[1]) == self.mem.dead_stem]
                if sib:
                    pool = sib
            close = [k for k in pool if self.relax(*k) <= (1 if self.mem.first_add else 0)]
            if close:
                pool = close
            elif self.mem.search_n < 3:
                return None
        if self.goal.wet:
            wet_ok = [k for k in pool if _wet(self.text_of(*k)) > 0]
            if wet_ok:
                pool = wet_ok
            elif self.mem.search_n < 3:
                return None
            four = [k for k in pool if self.four_axis(*k)]
            if four:
                pool = four
            elif self.mem.search_n < 3:
                return None
            firm = [k for k in pool if self.firm_pref(*k)]
            if firm:
                pool = firm
            hinted = None
            if vote and len(pool) >= 2:
                hinted = self.consult_pick(pool)
            return self.review_wet(pool, hinted)
        proven = max(pool, key=lambda ks: self.score(*ks))
        if vote and len(pool) >= 2:
            return self.review_need(pool, self.consult_pick(pool))
        return proven

    def fallback_pick(self) -> tuple | None:
        choice = self.pick()
        if choice or not self.mem.event:
            return choice
        pool = []
        for pid, sku in self.mem.row:
            if (pid, sku) in self.mem.ban or not sku:
                continue
            if not self.stock_ok(pid, sku) or not self.budget_ok(pid, sku):
                continue
            if not self.spec_ok(pid, sku) or not self.identity_ok(pid, sku):
                continue
            if self.goal.wet and _wet(self.text_of(pid, sku)) <= 0:
                continue
            pool.append((pid, sku))
        if not pool:
            return None
        if self.goal.wet:
            four = [k for k in pool if self.four_axis(*k)]
            if four:
                hinted = self.consult_pick(four) if len(four) >= 2 else None
                return self.review_wet(four, hinted)
            if self.mem.search_n < 2:
                return None
        return max(pool, key=lambda ks: self.score(*ks))

    def review_wet(self, pool: list, hinted: tuple | None) -> tuple | None:
        if not pool:
            return None
        four = [k for k in pool if self.four_axis(*k)]
        use = four or [k for k in pool if self.firm_pref(*k)] or list(pool)

        def _price_of(ks: tuple) -> float:
            return _fnum(self.mem.live(*ks).get("price")) or 1e18

        proven = min(use, key=_price_of) if four else max(use, key=lambda ks: self.score(*ks))
        if hinted in use:
            if four and not self.four_axis(*hinted):
                return proven
            if four and _price_of(hinted) > _price_of(proven) + 1e-9:
                return proven
            if self.score(*hinted) >= self.score(*proven):
                return hinted
        return proven

    def review_need(self, pool: list, hinted: tuple | None) -> tuple | None:
        if not pool:
            return None
        use = list(pool)
        if (not self.mem.event) and self.goal.soft_brand:
            branded = [
                k
                for k in use
                if str(self.mem.live(*k).get("brand") or "").lower() == self.goal.soft_brand.lower()
                or self.goal.soft_brand.lower() in str(self.mem.live(*k).get("title") or "").lower()
            ]
            if branded:
                use = branded
        proven = max(use, key=lambda ks: self.score(*ks))
        if hinted in use:
            if self.goal.phrase and not self.phrase_hit(*hinted):
                return proven
            if self.score(*hinted) >= self.score(*proven):
                return hinted
        return proven

    def _need_brief(self) -> str:
        bits = [
            "Confirm each listed variant against the shopper constraints.",
            "Use option text and labelled facts first; title tokens are weaker.",
        ]
        if self.goal.phrase:
            bits.append("Every token of the requested phrase must appear in the title.")
        if self.goal.inch is not None:
            bits.append("The listed screen size must be %g inches." % self.goal.inch)
        if self.goal.res:
            bits.append("The listed resolution class must be %s." % self.goal.res)
        if self.goal.model:
            bits.append("Stay on the named model and color when a passing row exists.")
        if self.goal.rank:
            bits.append("Prefer the highest listed refresh rate among the named variants.")
        if self.goal.soft_brand and not self.mem.event:
            bits.append(
                "For the first cart, honor the named preferred brand when a fully "
                "compliant row of that brand is listed."
            )
        if self.goal.wet:
            bits.append(
                "A valid pick must show sweat/water resistance, a sport/workout fit, "
                "wireless, and a battery life in hours. Reject a dearer row that is missing hours."
            )
        bits.append("Prefer the lower price among valid picks.")
        bits.append('Reply with JSON {"product_id":"...","sku":"..."} only.')
        return " ".join(bits)

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
            item = self.mem.live(pid, sku)
            brand = str(item.get("brand") or "")
            want = (self.goal.soft_brand or self.goal.hard_brand or "").lower()
            lines.append(
                "%d. %s::%s price=%s stock=%s phrase=%s inch=%s spec=%s brand=%s "
                "wet=%s sport=%s hours=%s wire=%s hz=%s options=%s title=%s facts=%s"
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
                    _wet(self.text_of(pid, sku)),
                    _sport(self.text_of(pid, sku)),
                    _batt_hours(self.text_of(pid, sku)),
                    _wire(self.text_of(pid, sku)),
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
        votes: list = []
        rounds = 1 if self.limit.left() < 40 else (4 if len(pool) >= 2 else 3)
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
            return min(winners, key=lambda ks: _fnum(self.mem.live(*ks).get("price")) or 1e18)
        return votes[0]

    def claims(self, pid: str, sku: str) -> list:
        item = self.mem.live(pid, sku)
        out = []
        if self.goal.inch is not None:
            got = self.inch_of(pid, sku)
            facts = item.get("facts") or {}
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if is_range:
                got = None
            elif val is not None:
                got = val
            if got is not None and round(got, 2) == round(self.goal.inch, 2):
                out.append({"field": "screen_size", "value": "%gin" % self.goal.inch})
        elif self.goal.res:
            if self.goal.res in self.res_classes(pid, sku):
                out.append({"field": "resolution", "value": self.goal.res})
        elif self.goal.hz_need is not None:
            got = self.hz_of(pid, sku)
            if got is not None and got + 1e-6 >= self.goal.hz_need:
                out.append({"field": "refresh_rate", "value": "%ghz" % self.goal.hz_need})
        elif self.goal.ram is not None:
            out.append({"field": "ram", "value": "%ggb" % self.goal.ram})
        if self.goal.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.goal.budget})
        out.append({"field": "in_stock", "value": "true" if item.get("in_stock") else "false"})
        brand = item.get("brand")
        if brand:
            out.append({"field": "brand", "value": str(brand)})
        return out

    def searches(self) -> list:
        q = []
        if self.goal.model:
            if self.goal.color:
                q.append("%s %s smartphone" % (self.goal.color, self.goal.model))
            q.append("%s smartphone" % self.goal.model)
            q.append(self.goal.model)
        if self.goal.phrase:
            q.append(self.goal.phrase)
            extra = self.follow_from_hits()
            if extra and extra.lower() != self.goal.phrase.lower():
                q.append(extra)
        bits = []
        if self.goal.hard_brand or self.goal.soft_brand:
            bits.append(self.goal.hard_brand or self.goal.soft_brand)
        if self.goal.phone:
            bits.append("smartphone")
        if self.goal.tablet:
            bits.append("tablet")
        if self.goal.monitor:
            bits.append("gaming monitor" if self.goal.gaming else "monitor")
        if self.goal.headphone:
            need = self.live_need_words()
            if self.goal.wet and need:
                bits.append(" ".join(need[:5]))
            else:
                bits.append("headphone")
        if self.goal.inch:
            bits.append("%g inch" % self.goal.inch)
        if self.goal.res:
            bits.append(self.goal.res)
        if self.goal.color:
            bits.append(self.goal.color)
        if bits:
            q.append(" ".join(str(b) for b in bits if b))
        if self.goal.wet:
            words = self.live_need_words()
            if words:
                q.append(" ".join(words[:6]))
                if self.goal.aisle:
                    q.append("%s %s" % (self.goal.aisle.split("&")[0].strip(), " ".join(words[:4])))
                if len(words) >= 3:
                    q.append(" ".join(words[:3]))
            extra = self.wet_title_extra()
            if extra and extra.lower() not in {s.lower() for s in q}:
                q.insert(1 if q else 0, extra)
            if self.mem.event and extra:
                q.insert(0, extra)
        if self.goal.soft_brand and self.goal.phone:
            q.append("%s smartphone" % self.goal.soft_brand)
        if self.goal.phone and self.goal.inch:
            q.append("%g inch smartphone" % self.goal.inch)
        if self.goal.hard_brand and self.goal.inch:
            q.append("%s %g inch smartphone" % (self.goal.hard_brand, self.goal.inch))
        if self.goal.hard_brand and (self.goal.monitor or self.goal.tablet or self.goal.why):
            q.insert(0, self.goal.hard_brand + (" %g inch" % self.goal.inch if self.goal.inch else "") + (" " + self.goal.res if self.goal.res else ""))
        if self.mem.dead_title and not self.goal.wet:
            q.insert(0, self.mem.dead_title)
            if self.goal.inch:
                q.insert(0, "%s %g inches smartphone" % (self.mem.dead_title, self.goal.inch))
        out = []
        seen = set()
        for s in q:
            s = " ".join(s.split())
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out or [self.goal.head[:80]]

    def need_view(self) -> list:
        pids = []
        if self.goal.seeds:
            for pid, _sku in self.goal.seeds:
                if pid not in self.mem.viewed:
                    pids.append(pid)
            return pids
        for pid, sku in self.mem.row:
            if pid in self.mem.viewed:
                continue
            if (pid, sku) not in self.mem.trusted and not self.goal.phrase:
                continue
            if self.goal.phrase and not self.phrase_hit(pid, sku):
                continue
            if self.goal.budget is not None:
                price = _fnum(self.mem.live(pid, sku).get("price"))
                if price is not None and price > self.goal.budget * 1.05:
                    continue
            title = str(self.mem.live(pid, sku).get("title") or "")
            if self.goal.phone and _JUNK.search(title):
                continue
            pids.append(pid)
        uniq = []
        seen = set()
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                uniq.append(pid)
        if self.goal.wet and uniq:
            def _view_key(pid: str) -> tuple:
                sku = ""
                for p, s in self.mem.row:
                    if p == pid:
                        sku = s
                        break
                if not sku:
                    return (0, 0, 0.0, 0, 0.0, 0.0)
                ax = self.pref_axes(pid, sku)
                price = _fnum(self.mem.live(pid, sku).get("price")) or 1e18
                return (
                    1 if self.four_axis(pid, sku) else 0,
                    1 if self.firm_pref(pid, sku) else 0,
                    ax["hours"],
                    ax["wet"],
                    ax["wire"],
                    -price,
                )

            uniq.sort(key=_view_key, reverse=True)
        return uniq[:8]

    def cart_pair(self) -> tuple | None:
        if not self.mem.cart:
            return None
        row = self.mem.cart[0]
        return (str(row.get("product_id") or ""), str(row.get("sku") or ""))

    def cart_valid(self) -> bool:
        pair = self.cart_pair()
        if not pair or not pair[0] or not pair[1]:
            return False
        return self.hard_ok(*pair)

    def reply(self, note: str, pair: tuple | None) -> str:
        low = (note or "").lower()
        if pair:
            item = self.mem.live(*pair)
            title = (item.get("title") or pair[0])[:90]
            price = _fnum(item.get("price"))
            if "expensive" in low or "more expensive" in low:
                if price is not None and self.goal.budget is not None and price > self.goal.budget + 1e-6:
                    return (
                        "Yes, %s is now %.2f PHP, above the budget. I'll switch to another "
                        "in-stock option that still meets the firm limits." % (title[:60], price)
                    )
            parts = ["I am going with " + title]
            if self.goal.inch is not None:
                parts.append("its listed screen size is %gin" % self.goal.inch)
            elif self.goal.res:
                parts.append("its listed resolution is " + self.goal.res)
            if price is not None:
                parts.append("it is %.2f PHP and in stock" % price)
            self.mem.named = pair
            return ", ".join(parts) + "."
        return "I am still checking the options against your requirements; nothing is decided yet."

    def order_args(self, pid: str, sku: str) -> dict:
        args = {"product_id": pid, "sku": sku}
        if self.goal.why:
            args["justification"] = {"claims": self.claims(pid, sku)}
        return args

    def filt_args(self) -> dict:
        filt: dict = {"in_stock": True, "k": 10}
        if self.goal.aisle:
            filt["category"] = self.goal.aisle
        if self.goal.hard_brand:
            filt["brand"] = self.goal.hard_brand
        elif self.goal.soft_brand:
            keep = True
            if self.mem.event or self.goal.budget_cut:
                keep = any(
                    str(self.mem.live(pid, sku).get("brand") or "").lower() == self.goal.soft_brand.lower()
                    and self.budget_ok(pid, sku)
                    and self.stock_ok(pid, sku)
                    and self.spec_ok(pid, sku)
                    for pid, sku in self.mem.row
                )
            if keep:
                filt["brand"] = self.goal.soft_brand
        if self.goal.budget is not None and not self.goal.wet:
            filt["max_price"] = self.goal.budget
        return filt

    def _push(self, acts: list, name: str, args: dict) -> None:
        if self.ok(name) and len(acts) < self.max_calls:
            acts.append((name, args))

    def _page(self, acts: list) -> list:
        self._push(acts, "search", {"query": self.next_query(), "in_stock": True, "k": 10})
        self._push(acts, "filter", self.filt_args())
        return acts

    def _ask(self) -> list:
        acts: list = []
        if self.goal.ask and not self.mem.asked:
            self._push(
                acts,
                "message",
                {
                    "content": (
                        "Before I search: which hardware specification is your firm requirement "
                        "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
                    )
                },
            )
            return acts
        if self.goal.pref and not self.mem.asked:
            self._push(
                acts,
                "message",
                {
                    "content": (
                        "Quick question so I choose well: which of your preferences counts most, and in what "
                        "order should I weigh the others? Let me know your priority."
                    )
                },
            )
        return acts

    def _talk(self) -> list:
        acts: list = []
        if not self.mem.inbox:
            return acts
        note = self.mem.inbox[-1]
        self.mem.inbox.clear()
        nudge = bool(re.search(r"keep to|stated requirements|compare the current", note, re.I))
        ask = bool(re.search(r"\?|which one|going with|still available|re-check|more expensive", note, re.I))
        if ask and not nudge:
            if self.mem.event and (self.goal.rank or self.goal.model or not self.cart_valid()):
                pair = self.fallback_pick() or self.cart_pair()
            else:
                pair = self.cart_pair() or self.fallback_pick()
            self._push(acts, "message", {"content": self.reply(note, pair)})
            if pair:
                self._push(acts, "inspect_stock", {"product_id": pair[0], "sku": pair[1]})
            if self.cart_pair():
                self._push(acts, "inspect_cart", {})
        return acts

    def _search_budget(self) -> int:
        if self.goal.phrase:
            return 2
        cap = 2 if self.goal.cover or self.goal.pref or self.goal.model else 1
        if self.goal.hard_brand and str(self.goal.hard_brand).lower() == "apple":
            cap = 5 if self.pick(vote=False) is None else self.mem.search_n
        if self.goal.cover and self.cover_count() < self.goal.cover:
            cap = max(cap, self.mem.search_n + 1)
        if self.goal.wet:
            cap = self.mem.search_n if self.paid_four() else 3
        if (self.mem.event or self.goal.budget_cut) and not self.goal.phrase and not self.goal.wet:
            if self.pick(vote=False) is None:
                cap = max(cap, 8)
        return cap

    def _retrieve(self) -> list:
        acts: list = []
        if self.goal.seeds:
            return acts
        if self.mem.search_n < self._search_budget():
            q = self.next_query()
            if not q or q.lower() in {str(s).lower() for s in self.mem.sent_q}:
                return acts
            return self._page(acts)
        return acts

    def _expand(self) -> list:
        acts: list = []
        unseen = self.need_view()
        view_cap = 8 if self.goal.wet else 6
        if unseen and self.ok("compare") and len(unseen) >= 2 and self.mem.compares < view_cap:
            self._push(acts, "compare", {"product_ids": unseen[:5]})
            return acts
        if unseen and self.ok("view"):
            for pid in unseen[: min(5, self.max_calls)]:
                self._push(acts, "view", {"product_id": pid})
        return acts

    def _recover(self) -> list:
        acts: list = []
        pair = self.cart_pair()
        if self.goal.rank and self.mem.event and self.goal.seeds:
            missing = [ks for ks in self.goal.seeds if ks not in self.mem.tick or self.mem.need_rescan]
            if self.mem.need_rescan:
                for pid, sku in self.goal.seeds:
                    self._push(acts, "inspect_stock", {"product_id": pid, "sku": sku})
                self.mem.need_rescan = False
                return acts
            top = self.pick()
            if pair and top and top != pair:
                self._push(acts, "remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                self._push(acts, "add_to_cart", {"product_id": top[0], "sku": top[1]})
                self._push(acts, "inspect_stock", {"product_id": top[0], "sku": top[1]})
                return acts
        if pair and not self.cart_valid():
            self._push(acts, "remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
            if pair[0] not in self.mem.viewed:
                self._push(acts, "view", {"product_id": pair[0]})
            nxt = self.fallback_pick()
            if (not nxt) and self.mem.named and self.mem.named not in self.mem.ban and self.mem.named != pair:
                nxt = self.mem.named
            weak = bool(self.goal.wet and nxt and not self.four_axis(*nxt) and not self.firm_pref(*nxt))
            extra = [p for p in self.need_view() if p]
            if weak and extra and self.ok("compare") and self.mem.compares < 8:
                self._push(acts, "compare", {"product_ids": extra[:5]})
                return acts
            if nxt and nxt != pair and nxt not in self.mem.ban and not weak:
                self._push(acts, "add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                self._push(acts, "inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
                return acts
            if self.goal.wet and nxt and nxt != pair and nxt not in self.mem.ban and self.firm_pref(*nxt):
                self._push(acts, "add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                self._push(acts, "inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
                return acts
            if not self.goal.seeds and self.mem.search_n < 8:
                return self._page(acts)
            if nxt and nxt != pair and nxt not in self.mem.ban:
                self._push(acts, "add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                self._push(acts, "inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
            return acts
        if pair and self.cart_valid() and (self.goal.pref or self.goal.wet) and self.mem.event:
            if not (self.goal.wet and self.four_axis(*pair)):
                top = self.fallback_pick()
                if top and top != pair:
                    better = self.score(*top) > self.score(*pair)
                    if self.goal.wet:
                        better = self.score(*top)[:4] > self.score(*pair)[:4]
                    if better:
                        self._push(acts, "remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                        self._push(acts, "add_to_cart", {"product_id": top[0], "sku": top[1]})
                        self._push(acts, "inspect_stock", {"product_id": top[0], "sku": top[1]})
                        return acts
            extra = [p for p in self.need_view() if p]
            four = self.paid_four()
            if (not four) and extra and self.ok("compare") and self.mem.compares < 8:
                self._push(acts, "compare", {"product_ids": extra[:5]})
                return acts
            if (not four) and self.mem.search_n < (3 if self.goal.wet else 8):
                return self._page(acts)
        if (
            self.mem.event
            and not pair
            and self.fallback_pick() is None
            and self.ok("search")
            and not self.goal.phrase
            and self.mem.search_n < 8
        ):
            return self._page(acts)
        return acts

    def _choose(self) -> list:
        acts: list = []
        if self.goal.seeds and self.goal.rank:
            missing = [ks for ks in self.goal.seeds if ks not in self.mem.tick]
            if missing:
                for pid, sku in missing[: self.max_calls]:
                    self._push(acts, "inspect_stock", {"product_id": pid, "sku": sku})
                return acts
        choice = self.fallback_pick()
        if choice:
            pid, sku = choice
            unread = [p for p in self.need_view() if p]
            if (
                self.goal.wet
                and not self.mem.cart
                and not self.four_axis(pid, sku)
                and unread
                and self.ok("compare")
                and self.mem.compares < 8
            ):
                self._push(acts, "compare", {"product_ids": unread[:5]})
                return acts
            if (
                self.goal.wet
                and not self.mem.cart
                and not self.paid_four()
                and self.mem.search_n < 3
            ):
                return self._page(acts)
            if (pid, sku) not in self.mem.tick:
                self._push(acts, "inspect_stock", {"product_id": pid, "sku": sku})
                alt = next((k for k in self.candidates() if k != (pid, sku) and k not in self.mem.tick), None)
                if alt:
                    self._push(acts, "inspect_stock", {"product_id": alt[0], "sku": alt[1]})
                self.mem.confirm_n = max(self.mem.confirm_n, 1)
                return acts
            if not self.mem.cart:
                self.mem.confirm_n = max(self.mem.confirm_n, 2)
                self._push(acts, "add_to_cart", {"product_id": pid, "sku": sku})
                self._push(acts, "inspect_stock", {"product_id": pid, "sku": sku})
                self._push(acts, "inspect_cart", {})
                return acts
        if (
            not self.goal.seeds
            and not self.goal.phrase
            and self.mem.search_n < (3 if self.goal.wet else 8)
            and self.fallback_pick() is None
        ):
            return self._page(acts)
        if self.mem.named and self.mem.named not in self.mem.ban and not self.mem.cart:
            pid, sku = self.mem.named
            self._push(acts, "add_to_cart", {"product_id": pid, "sku": sku})
            self._push(acts, "inspect_stock", {"product_id": pid, "sku": sku})
        return acts

    def _verify(self) -> list:
        acts: list = []
        pair = self.cart_pair()
        if pair and not self.mem.rechecked:
            self._push(acts, "inspect_stock", {"product_id": pair[0], "sku": pair[1]})
            self._push(acts, "inspect_cart", {})
        return acts

    def _order(self) -> list:
        acts: list = []
        pair = self.cart_pair()
        if pair and self.cart_valid() and self.ok("place_test_order"):
            if self.goal.rank and self.mem.event:
                top = self.pick()
                if top and top != pair:
                    self._push(acts, "remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                    self._push(acts, "add_to_cart", {"product_id": top[0], "sku": top[1]})
                    self._push(acts, "inspect_stock", {"product_id": top[0], "sku": top[1]})
                    return acts
            if self.limit.left() > 40 and not self.mem.final_look:
                self.mem.final_look = True
                self._push(acts, "inspect_stock", {"product_id": pair[0], "sku": pair[1]})
                self._push(acts, "inspect_cart", {})
                return acts
            top = self.fallback_pick()
            if top and top != pair:
                better = self.score(*top) > self.score(*pair)
                if self.goal.wet:
                    better = self.score(*top)[:4] > self.score(*pair)[:4]
                if better:
                    self._push(acts, "remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                    self._push(acts, "add_to_cart", {"product_id": top[0], "sku": top[1]})
                    self._push(acts, "inspect_stock", {"product_id": top[0], "sku": top[1]})
                    return acts
            self._push(acts, "place_test_order", self.order_args(*pair))
        return acts

    def decide(self) -> list:
        acts: list = []
        stages = (
            self._ask,
            self._talk,
            self._verify,
            self._recover,
            self._order,
            self._retrieve,
            self._expand,
            self._choose,
        )
        if not self.goal.phrase:
            stages = (
                self._ask,
                self._talk,
                self._verify,
                self._recover,
                self._order,
                self._expand,
                self._retrieve,
                self._choose,
            )
        for step in stages:
            try:
                more = step()
            except Exception as exc:
                self.talk.append(
                    {"role": "assistant", "content": "stage: %s: %s" % (type(exc).__name__, str(exc)[:120])}
                )
                more = []
            if not more:
                continue
            for item in more:
                if len(acts) < self.max_calls:
                    acts.append(item)
            if step is self._talk and not (self.cart_valid() and self.mem.rechecked):
                continue
            return acts
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
        if self.ok("place_test_order") and self.cart_valid():
            pair = self.cart_pair()
            if pair:
                _outs, done = self.fire([("place_test_order", self.order_args(*pair))])
                return done
        choice = self.fallback_pick()
        if choice and self.ok("add_to_cart") and not self.mem.cart:
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
                batch = self.decide()
            except Exception as exc:
                self.talk.append(
                    {"role": "assistant", "content": "decide: %s: %s" % (type(exc).__name__, str(exc)[:120])}
                )
                batch = []
            if not batch:
                if self.mem.cart and self.ok("place_test_order") and self.cart_valid():
                    pair = self.cart_pair()
                    batch = [("place_test_order", self.order_args(*pair))]
                elif (
                    self.ok("search")
                    and not self.goal.phrase
                    and self.mem.search_n < max(self._search_budget(), 8 if (self.mem.event or self.goal.budget_cut) else 0)
                    and self.fallback_pick() is None
                ):
                    q = self.next_query()
                    if q:
                        batch = [("search", {"query": q, "in_stock": True, "k": 10})]
                if not batch:
                    choice = self.fallback_pick()
                    if (not choice) and self.mem.named and self.mem.named not in self.mem.ban:
                        choice = self.mem.named
                    if choice and choice in self.mem.ban:
                        choice = None
                    if choice and self.ok("add_to_cart") and not self.mem.cart:
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
        return Episode(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "halt: %s: %s" % (type(exc).__name__, str(exc)[:180])}]
