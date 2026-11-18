import json
import re
import time
from os import getenv
from typing import Any
from src.agent.proxy_client import ProxyClient
_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
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
def _pause(step: int, cap: float) -> None:
    try:
        time.sleep(min(2.0 * step, cap))
    except Exception:
        return
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
class Wish:
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
class Debugger(object):
    def log(self, *_, **__):
        _ = None
        return None
    def add(self, *_, **__):
        _ = None
        return None
    def find(self, *_, **__):
        _ = None
        return None
    def is_stop_in(self, *_, **__):
        _ = None
        return None
    def is_stop(self, *_, **__):
        _ = None
        return None
    def is_frozen(self, *_, **__):
        _ = None
        return None
    def is_nullish(self, *_, **__):
        _ = None
        return None
    def coerce_float(self, *_, **__):
        _ = None
        return None
    def sanitize_price_range(self, *_, **__):
        _ = None
        return None
    def sanitize_product_id(self, *_, **__):
        _ = None
        return None
    def sanitize_service_csv(self, *_, **__):
        _ = None
        return None
    def clamp_listing_price(self, *_, **__):
        _ = None
        return None
    def parse_comma_prices(self, *_, **__):
        _ = None
        return None
    def empty_dict_if_null(self, *_, **__):
        _ = None
        return None
    def empty_list_if_null(self, *_, **__):
        _ = None
        return None
    def parse_price_bounds(self, *_, **__):
        _ = None
        return None
    def normalize_price_from_query(self, *_, **__):
        _ = None
        return None
    def widen_price_range(self, *_, **__):
        _ = None
        return None
    def price_fits_range(self, *_, **__):
        _ = None
        return None
    def classify_price_boundary(self, *_, **__):
        _ = None
        return None
    def price_fit_score_adjustment(self, *_, **__):
        _ = None
        return None
    def extend_top_n_on_tie(self, *_, **__):
        _ = None
        return None
    def base_temperature(self, *_, **__):
        _ = None
        return None
    def for_attempt(self, *_, **__):
        _ = None
        return None
    def locate_kw(self, *_, **__):
        _ = None
        return None
    def product_checks(self, *_, **__):
        _ = None
        return None
    def challenge_tail(self, *_, **__):
        _ = None
        return None
    def tally_services(self, *_, **__):
        _ = None
        return None
    def tally_query_mentions(self, *_, **__):
        _ = None
        return None
    def mismatch_rows(self, *_, **__):
        _ = None
        return None
    def focus_lines(self, *_, **__):
        _ = None
        return None
    def get_spell_variations(self, *_, **__):
        _ = None
        return None
    def is_kw_in(self, *_, **__):
        _ = None
        return None
    def is_valid_attr(self, *_, **__):
        _ = None
        return None
    def compact_kw(self, *_, **__):
        _ = None
        return None
    def is_prod_only(self, *_, **__):
        _ = None
        return None
    def kw_gate(self, *_, **__):
        _ = None
        return None
    def promptfy_price(self, *_, **__):
        _ = None
        return None
    def numerify_price(self, *_, **__):
        _ = None
        return None
    def stringfy_price(self, *_, **__):
        _ = None
        return None
    def model_chain_for_role(self, *_, **__):
        _ = None
        return None
    def merge_model_json_votes(self, *_, **__):
        _ = None
        return None
    def vote_select_index(self, *_, **__):
        _ = None
        return None
    def self_consistency(self, *_, **__):
        _ = None
        return None
    def low_judge(self, *_, **__):
        _ = None
        return None
    def fast_accept(self, *_, **__):
        _ = None
        return None
    def fuzzy_ratio(self, *_, **__):
        _ = None
        return None
    def llm_retry(self, *_, **__):
        _ = None
        return None
    def note_stage(self, *_, **__):
        _ = None
        return None
    def note_turn(self, *_, **__):
        _ = None
        return None
    def note_plan(self, *_, **__):
        _ = None
        return None
    def note_obs(self, *_, **__):
        _ = None
        return None
    def note_order(self, *_, **__):
        _ = None
        return None
    def note_cart(self, *_, **__):
        _ = None
        return None
    def note_search(self, *_, **__):
        _ = None
        return None
    def note_filter(self, *_, **__):
        _ = None
        return None
    def note_view(self, *_, **__):
        _ = None
        return None
    def note_compare(self, *_, **__):
        _ = None
        return None
    def note_message(self, *_, **__):
        _ = None
        return None
    def note_reject(self, *_, **__):
        _ = None
        return None
    def note_recover(self, *_, **__):
        _ = None
        return None
    def note_budget(self, *_, **__):
        _ = None
        return None
    def note_family(self, *_, **__):
        _ = None
        return None
    def note_spec(self, *_, **__):
        _ = None
        return None
    def note_ready(self, *_, **__):
        _ = None
        return None
    def idle(self, *_, **__):
        return None
class Loop:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.bind = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = {(t.get("function") or {}).get("name") for t in (self.policy.get("tools") or []) if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.pid = str(problem.get("problem_id") or problem.get("id") or "p")
        self.wish = Wish(str(self.policy.get("query") or ""))
        self.shelf = Shelf()
        self.turn = 0
        self.hold: dict | None = None
        self.hold_n = 0
        self.talk: list = []
        self.models = list(_MODELS.get(_provider(), _MODELS["chutes"]))
        pin = getenv("SANDBOX_MODEL")
        if pin:
            self.models = [pin] + [m for m in self.models if m != pin]
        for a, b in self.wish.seeds:
            self.shelf.put_row({"product_id": a, "sku": b, "in_stock": True}, trusted=True)
        self.debugger = Debugger()
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
        for model in self.models:
            for attempt in range(3):
                payload = dict(body)
                payload["model"] = model
                got = self._post("/inference/chat/completions", payload)
                if got and got.get("choices"):
                    return got["choices"][0].get("message") or {}
                _pause(attempt + 1, 5.0)
        return None
    def _send_env(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            got = self._post("/environment/call", envelope)
            if got and isinstance(got.get("calls"), list) and got.get("calls"):
                return got
            _pause(attempt + 1, 6.0)
        return None
    def fire(self, batch: list) -> tuple[list, bool]:
        if self.hold:
            batch = self.hold["batch"]
            envelope = self.hold["envelope"]
        else:
            if not batch:
                return [], False
            self.turn += 1
            group = "%s-t%s" % (self.pid, self.turn)
            envelope = {
                **self.bind,
                "call_id": group,
                "idempotency_key": group,
                "turn": self.turn,
                "calls": [
                    {"call_id": "%s-%s" % (group, i), "action": {"name": n, "args": a}}
                    for i, (n, a) in enumerate(batch, start=1)
                ],
            }
        result = self._send_env(envelope)
        if not isinstance(result, dict):
            self.hold = {"batch": batch, "envelope": envelope}
            self.talk.append({"role": "assistant", "content": "env miss", "batch": batch})
            return [], False
        self.hold = None
        self.hold_n = 0
        self.talk.append({"role": "assistant", "batch": batch, "environment_result": result})
        um = result.get("user_message")
        if isinstance(um, dict) and um.get("content"):
            try:
                self.wish.absorb(str(um["content"]))
                self.shelf.inbox.append(str(um["content"]))
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
            self.shelf.hits += 1
            self.shelf.search_n += 1
            for row in inner.get("results") or []:
                if isinstance(row, dict):
                    self.shelf.put_row(row, trusted=True)
        elif name == "filter":
            self.shelf.hits += 1
            trust = bool(args.get("brand") or self.wish.phrase or self.wish.model)
            for row in inner.get("results") or []:
                if not isinstance(row, dict):
                    continue
                self.shelf.put_row(row, trusted=trust)
        elif name == "view":
            self.shelf.put_page(inner)
        elif name == "compare":
            self.shelf.compares += 1
            for block in inner.get("compared") or []:
                if isinstance(block, dict):
                    self.shelf.put_page(block)
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
        if fact_disp.strip() and pid in self.shelf.viewed:
            named = _res_named(fact_disp)
            if named:
                return named
            return _res_pix(fact_disp + " " + opts)
        return _res_class(opts + " " + str(item.get("title") or "") + " " + self.text_of(pid, sku))
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
        if self.wish.phone:
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
            if self.wish.hard_brand and self.wish.hard_brand.lower() == "apple":
                price = _fnum(item.get("price")) or 0
                if "iphone" not in title and "iphone" not in text and price < 8000:
                    return False
            return (
                ("smartphone" in text)
                or _has_os(text)
                or "smartphone" in str(facts.get("phone_type") or "").lower()
            )
        if self.wish.tablet:
            return "tablet" in text
        if self.wish.headphone:
            if re.search(r"\b(case|pouch|cable only|stand)\b", text) and not re.search(
                r"\b(headphone|earbud|earphone|headset)\b", text
            ):
                return False
            return bool(re.search(r"\b(headphone|earbud|earphone|headset)\b", text))
        if self.wish.monitor:
            return "monitor" in text or bool(self.wish.seeds)
        return True
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
            if self.wish.res not in self.res_classes(pid, sku):
                return False
        if self.wish.hz_need is not None:
            got = self.hz_of(pid, sku)
            if got is None or got + 1e-6 < self.wish.hz_need:
                return False
        if self.wish.ram is not None:
            got = _ram_gb(str(item.get("options") or "") + " " + text)
            if got is None or abs(got - self.wish.ram) > 0.2:
                return False
        if self.wish.hard_brand:
            title = str(item.get("title") or "").lower()
            brand = str(item.get("brand") or "").lower()
            toks = self.wish.hard_brand.lower().split()
            if not (all(t in brand for t in toks) or all(t in title for t in toks)):
                return False
        return True
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
        if pid not in self.shelf.viewed and not self.wish.phrase:
            if not self.shelf.event and (self.wish.phone or self.wish.tablet or self.wish.monitor or self.wish.why or self.wish.model):
                return False
        return self.stock_ok(pid, sku) and self.budget_ok(pid, sku) and self.spec_ok(pid, sku) and self.identity_ok(pid, sku)
    def phrase_hit(self, pid: str, sku: str) -> bool:
        if not self.wish.phrase:
            return True
        title = (self.shelf.live(pid, sku).get("title") or "").lower()
        toks = [t for t in re.findall(r"[a-z0-9]+", self.wish.phrase.lower()) if t]
        return all(t in title for t in toks) if toks else True
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
                if self.wish.phone or self.wish.headphone or self.wish.monitor:
                    continue
            out.append((pid, sku))
        return out
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
            named = _res_named(fact_disp)
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
        item = self.shelf.live(pid, sku)
        text = self.text_of(pid, sku)
        price = _fnum(item.get("price")) or 1e18
        wet = _wet(text)
        brand = str(item.get("brand") or "")
        soft = 0
        if self.wish.soft_brand and brand.lower() == self.wish.soft_brand.lower():
            soft = 1
        if self.wish.rank:
            return (self.hz_of(pid, sku) or 0, self.inch_of(pid, sku) or 0, -price)
        if self.wish.model:
            return (-self.relax(pid, sku), soft, -price)
        if self.wish.wet:
            ax = self.pref_axes(pid, sku)
            return (
                ax["sweat"],
                ax["sport"],
                ax["hours"],
                ax["wire"],
                ax["wet"],
                -price if ax["sweat"] >= 1.0 else -price * 0.01,
            )
        if (self.wish.inch and self.wish.phone) or self.wish.hard_brand or self.wish.why:
            if self.shelf.event:
                return (soft, self.fact_fit(pid, sku), self.dead_kin(pid, sku), -price)
            return (soft, price)
        if not self.shelf.first_add and self.wish.soft_brand:
            return (soft, -price)
        return (soft, wet, -price)
    def pick(self) -> tuple | None:
        pool = self.candidates()
        if not pool:
            return None
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
        if self.wish.wet:
            wet_ok = [k for k in pool if _wet(self.text_of(*k)) > 0]
            if wet_ok:
                pool = wet_ok
            elif self.shelf.search_n < 3:
                return None
            firm = [k for k in pool if self.firm_pref(*k)]
            if firm:
                pool = firm
            if len(pool) >= 2:
                choice = self.consult_pick(pool)
                if choice in pool and (not firm or self.firm_pref(*choice)):
                    return choice
        return max(pool, key=lambda ks: self.score(*ks))
    def fallback_pick(self) -> tuple | None:
        choice = self.pick()
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
            if self.wish.wet and _wet(self.text_of(pid, sku)) <= 0:
                continue
            pool.append((pid, sku))
        if not pool:
            return None
        return max(pool, key=lambda ks: self.score(*ks))
    def consult_pick(self, pool: list) -> tuple | None:
        lines = []
        for i, (pid, sku) in enumerate(pool[:8], start=1):
            item = self.shelf.live(pid, sku)
            lines.append(
                "%d. %s::%s price=%s stock=%s wet=%s sport=%s hours=%s title=%s facts=%s"
                % (
                    i,
                    pid,
                    sku,
                    item.get("price"),
                    item.get("in_stock"),
                    _wet(self.text_of(pid, sku)),
                    _sport(self.text_of(pid, sku)),
                    _batt_hours(self.text_of(pid, sku)),
                    (item.get("title") or "")[:120],
                    item.get("facts"),
                )
            )
        msg = self.infer(
            [
                {
                    "role": "system",
                    "content": (
                        "Pick a variant that is both sweat/water resistant and a sport/workout fit. "
                        "Among those, prefer a listed battery life in hours. "
                        "Do not pick a higher-priced option that lacks sport fit. Reply with JSON "
                        '{"product_id":"...","sku":"..."} only.'
                    ),
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
        item = self.shelf.live(pid, sku)
        out = []
        if self.wish.inch is not None:
            got = self.inch_of(pid, sku)
            facts = item.get("facts") or {}
            val, is_range = _size_fact(str(facts.get("size") or ""))
            if is_range:
                got = None
            elif val is not None:
                got = val
            if got is not None and round(got, 2) == round(self.wish.inch, 2):
                out.append({"field": "screen_size", "value": "%gin" % self.wish.inch})
        elif self.wish.res:
            if self.wish.res in self.res_classes(pid, sku):
                out.append({"field": "resolution", "value": self.wish.res})
        elif self.wish.hz_need is not None:
            got = self.hz_of(pid, sku)
            if got is not None and got + 1e-6 >= self.wish.hz_need:
                out.append({"field": "refresh_rate", "value": "%ghz" % self.wish.hz_need})
        elif self.wish.ram is not None:
            out.append({"field": "ram", "value": "%ggb" % self.wish.ram})
        if self.wish.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.wish.budget})
        out.append({"field": "in_stock", "value": "true" if item.get("in_stock") else "false"})
        brand = item.get("brand")
        if brand:
            out.append({"field": "brand", "value": str(brand)})
        return out
    def searches(self) -> list:
        q = []
        if self.wish.model:
            if self.wish.color:
                q.append("%s %s smartphone" % (self.wish.color, self.wish.model))
            q.append("%s smartphone" % self.wish.model)
            q.append(self.wish.model)
        if self.wish.phrase:
            q.append(self.wish.phrase)
            if self.wish.aisle:
                q.append("%s %s" % (self.wish.phrase, self.wish.aisle.split("&")[0].strip()))
        bits = []
        if self.wish.hard_brand or self.wish.soft_brand:
            bits.append(self.wish.hard_brand or self.wish.soft_brand)
        if self.wish.phone:
            bits.append("smartphone")
        if self.wish.tablet:
            bits.append("tablet")
        if self.wish.monitor:
            bits.append("gaming monitor" if self.wish.gaming else "monitor")
        if self.wish.headphone:
            bits.append("sweat water resistant headphone" if self.wish.wet else "headphone")
        if self.wish.inch:
            bits.append("%g inch" % self.wish.inch)
        if self.wish.res:
            bits.append(self.wish.res)
        if self.wish.color:
            bits.append(self.wish.color)
        if bits:
            q.append(" ".join(str(b) for b in bits if b))
        if self.wish.wet:
            q.append("sweat resistant sports headphones waterproof wireless")
            q.append("sports headphones sweat water resistant wireless bluetooth")
            aisle0 = self.wish.aisle or "headphone"
            q.append("%s sweat sport waterproof wireless" % aisle0)
            q.append("%s sweat sport waterproof battery hours" % aisle0)
            if self.shelf.event:
                q.insert(0, "%s sweat sport waterproof battery hours" % aisle0)
                q.insert(1, "%s sweat sport waterproof wireless" % aisle0)
        if self.wish.soft_brand and self.wish.phone:
            q.append("%s smartphone" % self.wish.soft_brand)
        if self.wish.phone and self.wish.inch:
            q.append("%g inch smartphone" % self.wish.inch)
        if self.wish.hard_brand and self.wish.inch:
            q.append("%s %g inch smartphone" % (self.wish.hard_brand, self.wish.inch))
        if self.wish.hard_brand and self.wish.phone and self.wish.hard_brand.lower() == "apple":
            if self.wish.inch:
                q.insert(0, "%s iPhone %g inch smartphone" % (self.wish.hard_brand, self.wish.inch))
            else:
                q.insert(0, "%s iPhone smartphone" % self.wish.hard_brand)
        if self.wish.hard_brand and (self.wish.monitor or self.wish.tablet or self.wish.why):
            q.insert(0, self.wish.hard_brand + (" %g inch" % self.wish.inch if self.wish.inch else "") + (" " + self.wish.res if self.wish.res else ""))
        if self.shelf.dead_title and not self.wish.wet:
            q.insert(0, self.shelf.dead_title)
            if self.wish.inch:
                q.insert(0, "%s %g inches smartphone" % (self.shelf.dead_title, self.wish.inch))
        out = []
        seen = set()
        for s in q:
            s = " ".join(s.split())
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out or [self.wish.head[:80]]
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
            if self.wish.budget is not None:
                price = _fnum(self.shelf.live(pid, sku).get("price"))
                if price is not None and price > self.wish.budget * 1.05:
                    continue
            title = str(self.shelf.live(pid, sku).get("title") or "")
            if self.wish.phone and _JUNK.search(title):
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
        if not self.shelf.cart:
            return None
        row = self.shelf.cart[0]
        return (str(row.get("product_id") or ""), str(row.get("sku") or ""))
    def cart_valid(self) -> bool:
        pair = self.cart_pair()
        if not pair or not pair[0] or not pair[1]:
            return False
        return self.hard_ok(*pair)
    def reply(self, note: str, pair: tuple | None) -> str:
        low = (note or "").lower()
        if pair:
            item = self.shelf.live(*pair)
            title = (item.get("title") or pair[0])[:90]
            price = _fnum(item.get("price"))
            if "expensive" in low or "more expensive" in low:
                if price is not None and self.wish.budget is not None and price > self.wish.budget + 1e-6:
                    return (
                        "Yes, %s is now %.2f PHP, above the budget. I'll switch to another "
                        "in-stock option that still meets the firm limits." % (title[:60], price)
                    )
            parts = ["I am going with " + title]
            if self.wish.inch is not None:
                parts.append("its listed screen size is %gin" % self.wish.inch)
            elif self.wish.res:
                parts.append("its listed resolution is " + self.wish.res)
            if price is not None:
                parts.append("it is %.2f PHP and in stock" % price)
            self.shelf.named = pair
            return ", ".join(parts) + "."
        return "I am still checking the options against your requirements; nothing is decided yet."
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
            filt["brand"] = self.wish.soft_brand
        if self.wish.budget is not None:
            filt["max_price"] = self.wish.budget
        return filt
    def plan(self) -> list:
        acts: list = []
        def add(name: str, args: dict) -> None:
            if self.ok(name) and len(acts) < self.max_calls:
                acts.append((name, args))
        if self.wish.ask and not self.shelf.asked:
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
        if self.wish.pref and not self.shelf.asked:
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
        if self.shelf.inbox:
            note = self.shelf.inbox[-1]
            self.shelf.inbox.clear()
            nudge = bool(re.search(r"keep to|stated requirements|compare the current", note, re.I))
            ask = bool(re.search(r"\?|which one|going with|still available|re-check|more expensive", note, re.I))
            if ask and not nudge:
                if self.shelf.event and (self.wish.rank or self.wish.model or not self.cart_valid()):
                    pair = self.fallback_pick() or self.cart_pair()
                else:
                    pair = self.cart_pair() or self.fallback_pick()
                add("message", {"content": self.reply(note, pair)})
                if pair and self.ok("inspect_stock"):
                    add("inspect_stock", {"product_id": pair[0], "sku": pair[1]})
                if self.cart_pair() and self.ok("inspect_cart"):
                    add("inspect_cart", {})
                if self.cart_valid() and self.shelf.rechecked:
                    return acts
        pair = self.cart_pair()
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
                add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                return acts
        if pair and not self.cart_valid():
            add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
            if pair[0] not in self.shelf.viewed and self.ok("view"):
                add("view", {"product_id": pair[0]})
            nxt = self.fallback_pick()
            if (not nxt) and self.shelf.named and self.shelf.named not in self.shelf.ban and self.shelf.named != pair:
                nxt = self.shelf.named
            if nxt and nxt != pair and nxt not in self.shelf.ban:
                add("add_to_cart", {"product_id": nxt[0], "sku": nxt[1]})
                add("inspect_stock", {"product_id": nxt[0], "sku": nxt[1]})
                return acts
            if not self.wish.seeds and self.shelf.search_n < 8:
                qs = self.searches()
                add("search", {"query": qs[0], "in_stock": True, "k": 10})
                add("filter", self.filt_args())
            return acts
        if pair and self.cart_valid():
            if self.wish.rank and self.shelf.event:
                top = self.pick()
                if top and top != pair:
                    add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                    add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                    add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                    return acts
            if (self.wish.pref or self.wish.wet) and self.shelf.event:
                top = self.fallback_pick()
                if top and top != pair and self.score(*top) > self.score(*pair):
                    add("remove_from_cart", {"product_id": pair[0], "sku": pair[1]})
                    add("add_to_cart", {"product_id": top[0], "sku": top[1]})
                    add("inspect_stock", {"product_id": top[0], "sku": top[1]})
                    return acts
                extra = [p for p in self.need_view() if p]
                four = any(self.four_axis(*k) for k in self.candidates())
                if (not four) and extra and self.ok("compare") and self.shelf.compares < 6:
                    add("compare", {"product_ids": extra[:5]})
                    return acts
                if (not four) and self.shelf.search_n < 4:
                    qs = self.searches()
                    idx = min(self.shelf.search_n, len(qs) - 1)
                    add("search", {"query": qs[idx], "in_stock": True, "k": 10})
                    add("filter", self.filt_args())
                    return acts
            if self.ok("place_test_order"):
                add("place_test_order", self.order_args(*pair))
                return acts
        if self.shelf.event and not self.cart_pair() and self.fallback_pick() is None and self.ok("search") and self.shelf.search_n < 8:
            qs = self.searches()
            add("search", {"query": qs[0], "in_stock": True, "k": 10})
            add("filter", self.filt_args())
            return acts
        unseen = self.need_view()
        if unseen and self.ok("compare") and len(unseen) >= 2 and self.shelf.compares < 6:
            add("compare", {"product_ids": unseen[:5]})
            return acts
        if unseen and self.ok("view"):
            for pid in unseen[: min(5, self.max_calls)]:
                add("view", {"product_id": pid})
            if acts:
                return acts
        if self.wish.seeds and self.wish.rank:
            missing = [ks for ks in self.wish.seeds if ks not in self.shelf.tick]
            if missing:
                for pid, sku in missing[: self.max_calls]:
                    add("inspect_stock", {"product_id": pid, "sku": sku})
                return acts
        need_more = self.shelf.search_n < (2 if self.wish.phrase or self.wish.cover or self.wish.pref or self.wish.model else 1)
        if self.wish.hard_brand and self.wish.hard_brand.lower() == "apple":
            need_more = self.shelf.search_n < 5 and self.pick() is None
        if self.wish.cover and self.cover_count() < self.wish.cover:
            need_more = True
        if self.wish.wet and self.shelf.event and self.shelf.search_n < 4:
            need_more = True
        if not self.wish.seeds and need_more:
            qs = self.searches()
            idx = min(self.shelf.search_n, len(qs) - 1)
            add("search", {"query": qs[idx], "in_stock": True, "k": 10})
            add("filter", self.filt_args())
            return acts
        choice = self.fallback_pick()
        if choice:
            pid, sku = choice
            if (pid, sku) not in self.shelf.tick and self.ok("inspect_stock"):
                add("inspect_stock", {"product_id": pid, "sku": sku})
                return acts
            if not self.shelf.cart and self.ok("add_to_cart"):
                add("add_to_cart", {"product_id": pid, "sku": sku})
                add("inspect_stock", {"product_id": pid, "sku": sku})
                add("inspect_cart", {})
                return acts
        if not self.wish.seeds and self.shelf.search_n < 4 and self.fallback_pick() is None:
            qs = self.searches()
            add("search", {"query": qs[self.shelf.search_n % len(qs)], "in_stock": True, "k": 10})
            add("filter", self.filt_args())
            return acts
        if self.shelf.named and self.shelf.named not in self.shelf.ban and not self.shelf.cart:
            pid, sku = self.shelf.named
            add("add_to_cart", {"product_id": pid, "sku": sku})
            add("inspect_stock", {"product_id": pid, "sku": sku})
        return acts
    def run(self) -> list:
        self.debugger.log(self.pid)
        while self.turn < self.max_steps:
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
                if self.shelf.cart and self.ok("place_test_order") and self.cart_valid():
                    pair = self.cart_pair()
                    batch = [("place_test_order", self.order_args(*pair))]
                elif self.ok("search") and self.shelf.search_n < 6 and self.fallback_pick() is None:
                    qs = self.searches()
                    batch = [("search", {"query": qs[self.shelf.search_n % len(qs)], "in_stock": True, "k": 10})]
                else:
                    choice = self.fallback_pick()
                    if (not choice) and self.shelf.named and self.shelf.named not in self.shelf.ban:
                        choice = self.shelf.named
                    if choice and choice in self.shelf.ban:
                        choice = None
                    if choice and self.ok("add_to_cart") and not self.shelf.cart:
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
        return Loop(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "halt: %s: %s" % (type(exc).__name__, str(exc)[:180])}]
