"""ORO-Bench shopping agent: query genome mutator.

Search is a mutation engine. Genome sows quoted phrase, model code,
brand+spec+shelf, and stemmed synonyms. Each Chromosome is a search/filter;
after results it breeds drop-rarest, synonym-swap, reorder, category-only
filter, and in_stock/k=10 mutants, rejecting duplicate (name, canonical args).
CoverageBook counts titles that carry every theme token as a whole word and
blocks recall orders below N (N+2 early; margin lapses after half the turns
or eight searches). Then unread leads are read; the closer inspects, carts,
rechecks, and orders.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_ROSTER = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_LINK = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_SPAN_S = 270.0
_CLIP = 7000
_NUDGE_CAP = 2
_TOL = 1e-9

_DIGITS = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_CAP_RX = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _DIGITS
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_PAIR_RX = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTE_RX = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RX = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURF_RX = re.compile(r"surface at least (\d+) distinct", re.I)
_PREF_RX = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_SOFT_RX = re.compile(r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)")
_SHELF_RX = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SPECISH_RX = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_WHY_RX = re.compile(r"\b(?:why|reason|explain|justif)", re.I)
_ASK_RX = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
# Evidence: labelled facts, then variant options, then title.
_INCH_RX = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HERTZ_RX = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_PIXEL_RX = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_MEM_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_COMBO_RX = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_COMBO_HINT_RX = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_RAM_RX = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QHD_RX = re.compile(r"\bqHD\b")

_LABEL = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}


def _gather_inches(blob: str) -> set:
    found = set()
    for hit in _INCH_RX.finditer(blob or ""):
        found.add(("%g" % float(hit.group(1))) + "in")
    return found


def _gather_hertz(blob: str) -> set:
    found = set()
    for hit in _HERTZ_RX.finditer(blob or ""):
        found.add(("%g" % float(hit.group(1))) + "hz")
    return found


def _gather_gigs(blob: str) -> set:
    found = set()
    for hit in _MEM_RX.finditer(blob or ""):
        n = float(hit.group(1))
        if hit.group(2).lower() == "tb":
            n *= 1024
        found.add(("%g" % n) + "gb")
    return found


def _panel_class(wide: int, tall: int) -> str | None:
    if wide >= 3840 and tall >= 2160:
        return "4k"
    if wide >= 2560 and tall >= 1440:
        return "qhd"
    if wide >= 1920 and tall >= 1080:
        return "fhd"
    if wide >= 1280 and tall >= 720:
        return "hd"
    return None


def _gather_res(blob: str) -> set:
    blob = blob or ""
    if _QHD_RX.search(blob):
        return set()
    low = blob.lower()
    found = set()
    for hit in _PIXEL_RX.finditer(low):
        sides = sorted((int(hit.group(1)), int(hit.group(2))), reverse=True)
        kind = _panel_class(sides[0], sides[1])
        if kind:
            found.add(kind)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        found.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        found.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        found.add("fhd")
    if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
        found.add("hd")
    return found


_PULL = {"screen_size": _gather_inches, "refresh_rate": _gather_hertz, "resolution": _gather_res, "ram": _gather_gigs}


def _adj(field: str, blob: str) -> set:
    blob = blob or ""
    if field != "ram":
        return _PULL[field](blob)
    combos = {"%ggb" % float(a) for a, _b in _COMBO_RX.findall(blob)}
    if combos:
        return combos
    return {"%ggb" % float(n) for n in _RAM_RX.findall(blob)}


def _talks_about(field: str, blob: str) -> bool:
    blob = blob or ""
    if field == "ram":
        return bool(_COMBO_HINT_RX.search(blob) or _RAM_RX.search(blob))
    if field == "screen_size":
        return bool(_INCH_RX.search(blob))
    if field == "refresh_rate":
        return bool(_HERTZ_RX.search(blob))
    return bool(
        _PIXEL_RX.search(blob)
        or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", blob, re.I)
    )


def _lone(values: set) -> str | None:
    if len(values) != 1:
        return None
    return next(iter(values))


def _asked_value(field: str | None, blob: str) -> str | None:
    if not field:
        return None
    return _lone(_PULL[field](blob or ""))


def _wordbag(blob: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (blob or "").lower()))


def _noun_eq(left: str, right: str) -> bool:
    a = re.sub(r"[^a-z0-9]+", "", (left or "").lower()).rstrip("s")
    b = re.sub(r"[^a-z0-9]+", "", (right or "").lower()).rstrip("s")
    return bool(a) and a == b


def _whole_word(token: str, blob: str) -> bool:
    if not token:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", (blob or "").lower()) is not None


def _theme_covered(title: str, theme: list) -> bool:
    if not theme:
        return True
    return all(_whole_word(tok, title) for tok in theme)


def _money(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


def _parse_args(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


_OS_RX = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|"
    r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_IS_PHONE_RX = re.compile(r"\bsmart\s*phones?\b", re.I)
_NOT_PHONE_RX = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
    re.I,
)
_PART_RX = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_NEG_RX = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
    re.I,
)
_BUNDLE_RX = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.I,
)


def _scrub_identity(blob: str, title: bool = False) -> str:
    blob = _NEG_RX.sub("", blob or "")
    return _BUNDLE_RX.split(blob, maxsplit=1)[0] if title else blob


def _handset_verdict(facts: dict, title: str, options: str) -> str:
    if _PART_RX.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _scrub_identity(raw_title, title=True)
    kind = " ".join(_scrub_identity(str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
    blob = head + " " + kind
    if _NOT_PHONE_RX.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _IS_PHONE_RX.search(blob):
        return "yes"
    os_blob = " ".join(
        _scrub_identity(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS_RX.search(os_blob):
        return "yes"
    if _OS_RX.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
        return "yes"
    return "unclear"


_LANE_SIGNS = (
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("justification", "trustworthy reason"),
    ("intent", "nice-to-haves"),
)

_LANE_COPY = {
    "intent": (
        "Hard gates are the shelf, the named hardware figure, the purse and stock. A named "
        "maker is a preference: take it when a fully legal unit of that maker exists, otherwise "
        "any legal maker. Filter the shelf, then read pages so the closer can verify a variant."
    ),
    "recall": (
        "The genome mutates search wording until enough distinct titles carry every requested "
        "word as a whole word. After coverage is met, read the strongest leads so one can be bought."
    ),
    "constraint": (
        "One hardware gate is hidden until you ask. Obey the revealed value on the exact variant, "
        "reading that variant's own option text first. If the carted unit dies, move to a different product."
    ),
    "preference": (
        "You already asked which preference ranks first. Search or filter the shelf inside the purse, "
        "then compare products that live on different listing_id values. The first cart line is graded."
    ),
    "ranking": (
        "Only the named id::sku pairs count. Compare them so their facts are on record. Highest "
        "advertised max refresh wins, then largest panel, then lowest price, reading the selected "
        "variant before a product-wide figure. After a shock, the survivors are re-ranked."
    ),
    "recovery": (
        "Hunt the preferred model and color first: search the model code, then read every color. "
        "The first cart line must be that exact pairing. If it vanishes, relax as little as possible "
        "— same model, other color, before any other model. Never relax shelf, purse or stock."
    ),
    "justification": (
        "Find a unit whose own labelled facts state the requested spec, on the requested shelf and "
        "maker, in stock and inside the purse. Read it, then the closer inspects, carts and orders "
        "with claims taken only from that evidence."
    ),
    "generic": (
        "Learn the hard gates, let the genome probe the catalog, read pages, and let the closer "
        "verify and buy the best surviving unit."
    ),
}

_OPERATOR_COPY = (
    "You are a shopping operator on a live catalog. One tool call per turn, at most thirty turns. "
    "A query genome already owns search wording and a closer already owns cart and order once a "
    "unit is fully verified. Your useful moves are: view or compare a lead the genome surfaced, "
    "answer the shopper, or propose a fresh look only when the genome is stuck. Never cart a "
    "placeholder — the first add is graded. Never ask the shopper to raise the purse. Never send "
    "status chatter. Do not fire an identical tool call twice. One compare of up to five ids beats "
    "several views."
)


class Hourglass:
    def __init__(self, span: float) -> None:
        self.t0 = time.monotonic()
        self.span = span

    def left(self) -> float:
        return self.span - (time.monotonic() - self.t0)


class Request:
    def __init__(self, query: str) -> None:
        self.raw = query or ""
        aim, _, rest = self.raw.partition("\n\nTask rules:")
        self.aim = aim.strip()
        self.rule_blob = rest.lower()
        self.purse = self._read_purse(self.aim)
        self.currency = "PHP"
        iso = re.search(r"\b([A-Z]{3})\b", self.aim)
        if iso and iso.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = iso.group(1)
        self.named_pairs = [(pid, sku.rstrip(".,;:")) for pid, sku in _PAIR_RX.findall(self.aim)]
        phrase = _PHRASE_RX.search(self.aim)
        quoted = _QUOTE_RX.findall(self.aim)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        surf = _SURF_RX.search(self.rule_blob)
        self.need_ids = int(surf.group(1)) if surf else 0
        pref = _PREF_RX.search(self.aim)
        self.want_color = pref.group(1).lower() if pref else None
        self.want_model = pref.group(2) if pref else None
        self.shelf = self._read_shelf(self.aim)
        self.maker = self._read_maker(self.aim, self.shelf)
        self.soft_brand = None if pref else self._read_soft(self.aim)
        self.shape = self._read_shape(self.aim)
        self.figures = self._read_figures(self.aim)
        self.gate_field, self.gate_value = self._first_gate()
        self.hidden = self.gate_field is None and "firm requirement" in self.aim.lower()
        self.lane = self._classify()

    def _classify(self) -> str:
        if self.named_pairs:
            return "ranking"
        if self.need_ids:
            return "recall"
        if self.want_model:
            return "recovery"
        for name, sign in _LANE_SIGNS:
            if sign in self.rule_blob:
                return name
        if self.theme:
            return "recall"
        if _WHY_RX.search(self.aim):
            return "justification"
        if self.soft_brand:
            return "intent"
        if self.hidden or self.gate_field or self.maker:
            return "constraint"
        return "preference"

    @staticmethod
    def _read_purse(blob: str) -> float | None:
        best = None
        for hit in _CAP_RX.finditer(blob or ""):
            whole = hit.group(1).replace(",", "")
            frac = hit.group(2) or "0"
            try:
                val = float(whole + "." + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best

    @staticmethod
    def _read_shelf(aim: str) -> str | None:
        hit = _SHELF_RX.search(aim or "")
        if hit:
            return hit.group(1).strip()
        hit = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            aim or "",
            re.I,
        )
        return hit.group(1) if hit else None

    @staticmethod
    def _read_maker(aim: str, shelf: str | None) -> str | None:
        if not shelf:
            return None
        hit = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(shelf), aim or "", re.I)
        if not hit:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPECISH_RX.sub(" ", hit.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _gather_res(w)]
        return " ".join(keep) or None

    @staticmethod
    def _read_soft(aim: str) -> str | None:
        hit = _SOFT_RX.search(aim or "")
        if not hit:
            return None
        name = hit.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def _read_shape(aim: str) -> str | None:
        if re.search(r"\bgaming\s+monitors?\b", aim or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", aim or "", re.I):
            return "over ear"
        return None

    @staticmethod
    def _read_figures(aim: str) -> dict:
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = _asked_value(field, aim)
            if value:
                out[field] = value
        return out

    def _first_gate(self) -> tuple:
        for field, value in self.figures.items():
            return field, value
        return None, None

    def absorb_gate(self, blob: str) -> str | None:
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = _asked_value(field, blob)
            if value:
                self.gate_field, self.gate_value = field, value
                self.figures[field] = value
                self.hidden = False
                return field + " = " + value
        return None


class CoverageBook:
    def __init__(self) -> None:
        self.basket: list = []
        self.ticks: dict = {}
        self.tick_at: dict = {}
        self.probed: set = set()
        self.pages: dict = {}
        self.rows: dict = {}
        self.opts: dict = {}
        self.listings: set = set()
        self.listing_of: dict = {}
        self.hits: set = set()
        self.tick = 0
        self.first_add_at: int | None = None
        self.last_add_at: int | None = None
        self.adds = 0
        self.shock_key: str | None = None
        self.dead: set = set()
        self.dead_pids: set = set()
        self.fingerprints: set = set()

    @staticmethod
    def vid(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def cover_need(self, probes: int, step: int, cap: int, required: int) -> int:
        if not required:
            return 0
        early = probes < 8 and step < cap * 0.5
        return required + (2 if early else 0)

    def starved(self, probes: int, step: int, cap: int, required: int) -> bool:
        if not required:
            return False
        return len(self.hits) < self.cover_need(probes, step, cap, required)

    def mark_title(self, pid: Any, title: str, theme: list) -> None:
        if not pid:
            return
        if theme:
            if _theme_covered(title, theme):
                self.hits.add(str(pid))
        else:
            self.hits.add(str(pid))

    def ingest(self, name: str, args: dict, obs: dict, ask: Request) -> None:
        self.tick += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._row(row, ask)
        elif name == "view":
            self._page(obs, ask)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._page(prod, ask)
        elif name == "inspect_stock":
            k = self.vid(obs.get("product_id"), obs.get("sku"))
            self.ticks[k] = {"in_stock": bool(obs.get("in_stock")), "price": _money(obs.get("price"))}
            self.tick_at[k] = self.tick
            self.probed.add(k)
        elif name == "add_to_cart":
            added = obs.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                pair = (pid, sku)
                if pair not in self.basket:
                    self.basket.append(pair)
                if self.first_add_at is None:
                    self.first_add_at = self.tick
                self.last_add_at = self.tick
            self._sync(obs)
        elif name == "remove_from_cart":
            self._sync(obs)
            removed = obs.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.basket:
                self.basket.remove(pair)
        elif name == "inspect_cart":
            self._sync(obs)

    def _sync(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.basket = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.vid(r.get("product_id"), r.get("sku"))
            price = _money(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.ticks.get(k, {})
            self.ticks[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.tick_at[k] = self.tick

    def _row(self, row: dict, ask: Request) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        self.mark_title(pid, str(row.get("title") or ""), ask.theme)
        if not row.get("sku"):
            return
        k = self.vid(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.ticks.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _money(row.get("price"))})

    def _page(self, prod: dict, ask: Request) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.listings.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        self.mark_title(pid, str(prod.get("title") or ""), ask.theme)
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.vid(pid, v.get("sku"))
            self.ticks[k] = {"in_stock": bool(v.get("in_stock")), "price": _money(v.get("price"))}
            self.tick_at[k] = self.tick
            self.opts[k] = str(v.get("options") or "")

    def status(self, pid: str, sku: str) -> dict | None:
        return self.ticks.get(self.vid(pid, sku))


_KIN = {
    "monitor": ("display", "screen"),
    "monitors": ("displays", "screens"),
    "display": ("monitor", "screen"),
    "displays": ("monitors", "screens"),
    "screen": ("monitor", "display"),
    "screens": ("monitors", "displays"),
    "phone": ("smartphone", "cellphone", "mobile"),
    "phones": ("smartphones", "cellphones", "mobiles"),
    "smartphone": ("phone", "cellphone", "mobile"),
    "smartphones": ("phones", "cellphones", "mobiles"),
    "cellphone": ("phone", "smartphone", "mobile"),
    "cellphones": ("phones", "smartphones", "mobiles"),
    "mobile": ("phone", "smartphone", "cellphone"),
    "mobiles": ("phones", "smartphones", "cellphones"),
    "headphone": ("headset", "earphone"),
    "headphones": ("headsets", "earphones"),
    "headset": ("headphone", "earphone"),
    "headsets": ("headphones", "earphones"),
    "earphone": ("headphone", "headset"),
    "earphones": ("headphones", "headsets"),
    "earbud": ("earphone", "headphone"),
    "earbuds": ("earphones", "headphones"),
    "laptop": ("notebook", "computer"),
    "laptops": ("notebooks", "computers"),
    "notebook": ("laptop",),
    "notebooks": ("laptops",),
    "tv": ("television",),
    "television": ("tv",),
    "tablet": ("pad",),
    "keyboard": ("keypad",),
}


class Chromosome:
    def __init__(self, verb: str, payload: dict, query: str = "") -> None:
        self.verb = verb
        self.payload = dict(payload)
        self.query = query or str(payload.get("query") or payload.get("category") or "")

    def stamp(self) -> str:
        body = {}
        for key in sorted(self.payload):
            val = self.payload[key]
            if isinstance(val, float):
                body[key] = round(val, 4)
            else:
                body[key] = val
        return self.verb + "|" + json.dumps(body, sort_keys=True, default=str)

    def with_stock_cap(self) -> Chromosome | None:
        payload = dict(self.payload)
        changed = False
        if payload.get("in_stock") is not True:
            payload["in_stock"] = True
            changed = True
        if payload.get("k") != 10:
            payload["k"] = 10
            changed = True
        if not changed:
            return None
        return Chromosome(self.verb, payload, self.query)

    def without_rarest(self, freq: dict) -> Chromosome | None:
        if self.verb != "search":
            return None
        words = [w for w in self.query.split() if w]
        if len(words) < 2:
            return None
        rarest = min(range(len(words)), key=lambda i: (freq.get(words[i].lower(), 0), i))
        kept = [w for i, w in enumerate(words) if i != rarest]
        if not kept:
            return None
        payload = dict(self.payload)
        payload["query"] = " ".join(kept)
        payload["in_stock"] = True
        payload["k"] = 10
        return Chromosome("search", payload)

    def synonym_swap(self) -> Chromosome | None:
        if self.verb != "search":
            return None
        words = [w for w in self.query.split() if w]
        have = {w.lower() for w in words}
        for i, word in enumerate(words):
            alts = _KIN.get(word.lower()) or _KIN.get(word.lower().rstrip("s"))
            if not alts:
                continue
            for alt in alts:
                if alt.lower() in have:
                    continue
                nxt = list(words)
                nxt[i] = alt
                payload = dict(self.payload)
                payload["query"] = " ".join(nxt)
                payload["in_stock"] = True
                payload["k"] = 10
                return Chromosome("search", payload)
        return None

    def reshuffled(self) -> Chromosome | None:
        if self.verb != "search":
            return None
        words = [w for w in self.query.split() if w]
        if len(words) < 2:
            return None
        rotated = words[1:] + words[:1]
        if rotated == words:
            return None
        payload = dict(self.payload)
        payload["query"] = " ".join(rotated)
        payload["in_stock"] = True
        payload["k"] = 10
        return Chromosome("search", payload)

    def as_shelf_filter(self, ask: Request) -> Chromosome | None:
        if self.verb == "filter" or not ask.shelf:
            return None
        args: dict[str, Any] = {"category": ask.shelf, "in_stock": True, "k": 10}
        if ask.purse is not None:
            args["max_price"] = ask.purse
        if ask.maker:
            args["brand"] = ask.maker
        return Chromosome("filter", args)


class Genome:
    def __init__(self, ask: Request) -> None:
        self.ask = ask
        self.used: set = set()
        self.queue: list = []
        self.freq: dict = {}
        self.last: Chromosome | None = None
        self.hatched = 0
        self.twin_done = False
        self.sib_done = False
        self._sow()

    def has_pending(self) -> bool:
        return bool(self.queue)

    def _enqueue(self, chromo: Chromosome | None) -> bool:
        if chromo is None:
            return False
        stamp = chromo.stamp()
        if stamp in self.used:
            return False
        for pending in self.queue:
            if pending.stamp() == stamp:
                return False
        self.queue.append(chromo)
        return True

    def remember(self, chromo: Chromosome) -> bool:
        stamp = chromo.stamp()
        if stamp in self.used:
            return False
        self.used.add(stamp)
        self.last = chromo
        self.hatched += 1
        self.queue = [c for c in self.queue if c.stamp() != stamp]
        return True

    def offer(self) -> Chromosome | None:
        while self.queue:
            chromo = self.queue.pop(0)
            if chromo.stamp() in self.used:
                continue
            return chromo
        return None

    def _sow(self) -> None:
        ask = self.ask
        if ask.theme:
            self._enqueue(Chromosome("search", {"query": " ".join(ask.theme), "in_stock": True, "k": 10}))
            swapped = self._swap_seq(ask.theme)
            if swapped != list(ask.theme):
                self._enqueue(Chromosome("search", {"query": " ".join(swapped), "in_stock": True, "k": 10}))
        if ask.want_model:
            self._enqueue(Chromosome("search", {"query": ask.want_model, "in_stock": True, "k": 10}))
            if ask.want_color:
                self._enqueue(
                    Chromosome("search", {"query": ask.want_model + " " + ask.want_color, "in_stock": True, "k": 10})
                )
        bits = [str(x) for x in (ask.maker, ask.gate_value, ask.shelf) if x]
        if bits:
            self._enqueue(Chromosome("search", {"query": " ".join(bits), "in_stock": True, "k": 10}))
        if ask.shelf:
            filt: dict[str, Any] = {"category": ask.shelf, "in_stock": True, "k": 10}
            if ask.purse is not None:
                filt["max_price"] = ask.purse
            if ask.maker:
                filt["brand"] = ask.maker
            self._enqueue(Chromosome("filter", filt))
            syn = self._first_syn(ask.shelf)
            if syn:
                self._enqueue(Chromosome("search", {"query": syn, "in_stock": True, "k": 10}))
        if ask.soft_brand and ask.shelf:
            self._enqueue(
                Chromosome("search", {"query": ask.soft_brand + " " + ask.shelf, "in_stock": True, "k": 10})
            )

    @staticmethod
    def _swap_seq(words: list) -> list:
        out = list(words)
        have = {w.lower() for w in out}
        for i, word in enumerate(out):
            alts = _KIN.get(word.lower()) or _KIN.get(word.lower().rstrip("s"))
            if not alts:
                continue
            for alt in alts:
                if alt.lower() not in have:
                    out[i] = alt
                    return out
        return out

    @staticmethod
    def _first_syn(text: str) -> str | None:
        for word in re.findall(r"[a-z0-9]+", (text or "").lower()):
            alts = _KIN.get(word) or _KIN.get(word.rstrip("s"))
            if alts:
                return (text or "").lower().replace(word, alts[0], 1)
        return None

    def _count_titles(self, titles: list) -> None:
        for title in titles:
            for tok in re.findall(r"[a-z0-9]+", str(title or "").lower()):
                self.freq[tok] = self.freq.get(tok, 0) + 1

    def breed(self, parent: Chromosome, titles: list) -> None:
        self._count_titles(titles)
        kids = (
            parent.with_stock_cap(),
            parent.without_rarest(self.freq),
            parent.synonym_swap(),
            parent.reshuffled(),
            parent.as_shelf_filter(self.ask),
        )
        for kid in kids:
            self._enqueue(kid)

    def graft_obituary(self, title: str) -> Chromosome | None:
        words = [w for w in str(title or "").split() if w][:8]
        if not words or self.twin_done:
            return None
        self.twin_done = True
        chromo = Chromosome("search", {"query": " ".join(words), "k": 10, "in_stock": True})
        self._enqueue(chromo)
        self.breed(chromo, [title])
        return chromo if chromo.stamp() not in self.used else self.offer()

    def graft_sibling(self, title: str) -> Chromosome | None:
        words = [w for w in str(title or "").split() if w][:8]
        if not words or self.sib_done:
            return None
        self.sib_done = True
        chromo = Chromosome("search", {"query": " ".join(words), "k": 10, "in_stock": True})
        self._enqueue(chromo)
        return chromo if chromo.stamp() not in self.used else self.offer()


class MutantShop:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.cap = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.ask = Request(str(self.policy.get("query") or ""))
        self.purse = self.ask.purse
        self.book = CoverageBook()
        self.genome = Genome(self.ask)
        self.hour = Hourglass(_SPAN_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self._briefing()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.step = 0
        self.asked = False
        self.owe_reply = False
        self.shopper_moved = False
        self.nays = 0
        self.held: dict | None = None
        self.probes = 0
        self.cart_looks = 0
        self.recheck: tuple | None = None
        self.forced_reads = 0
        self.hinted_exact = False
        self.hinted_brand = False
        self.q_count = 0
        self.q_waits = 0
        self._wants: set | None = None
        self.pages_done = False
        self.shock_at: int | None = None
        self.opaque: set = set()
        self.finished = False
        self.slot = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.bench = _ROSTER.get(provider) or _ROSTER["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.bench = (pinned,) + tuple(m for m in self.bench if m != pinned)

    def _briefing(self) -> str:
        lane = _LANE_COPY.get(self.ask.lane, _LANE_COPY["generic"])
        extra = ""
        if self.ask.lane == "recall" and self.ask.need_ids:
            extra = (
                " CoverageBook will not allow an order until at least %d distinct matching "
                "titles have been surfaced; it reports the running count after each probe."
                % self.ask.need_ids
            )
        return _OPERATOR_COPY + "\n\nLane: " + lane + extra

    def _first_ask(self) -> str | None:
        if self.ask.lane == "constraint" or self.ask.hidden:
            return (
                "Before I probe the catalog: which hardware figure is the firm gate "
                "(panel size, resolution class, refresh, or memory), and what value must it be?"
            )
        if self.ask.lane == "preference":
            return (
                "So I rank well: which of your preferences should I treat as first, and how "
                "should I break ties among the rest?"
            )
        return None

    def _whisper(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _page(self, pid: str) -> dict:
        return self.book.pages.get(str(pid)) or {}

    def _raw_facts(self, pid: str) -> dict:
        blob = self._page(pid).get("facts")
        return blob if isinstance(blob, dict) else {}

    def _heading(self, pid: str) -> str:
        title = str(self._page(pid).get("title") or "")
        if title:
            return title
        prefix = str(pid) + "::"
        for key, row in self.book.rows.items():
            if key.startswith(prefix):
                return str(row.get("title") or "")
        return ""

    def _maker_of(self, pid: str) -> str:
        brand = self._page(pid).get("brand")
        if brand:
            return str(brand)
        prefix = str(pid) + "::"
        for key, row in self.book.rows.items():
            if key.startswith(prefix) and row.get("brand"):
                return str(row["brand"])
        return ""

    def _shelf_path(self, pid: str) -> list:
        path = self._page(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def _opt_text(self, pid: str, sku: str) -> str:
        return self.book.opts.get(CoverageBook.vid(pid, sku), "")

    def _drop_codes(self, pid: str, title: str) -> str:
        for token in re.split(r"[;,]", str(self._raw_facts(pid).get("model") or "")):
            token = token.strip()
            stem = re.split(r"[-_]", token)[0] if token else ""
            if (
                token
                and re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", token, re.I)
                and re.search(r"[a-z]", stem, re.I)
                and re.search(r"\d", stem)
                and not re.fullmatch(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I)
            ):
                title = re.sub(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", "", title, flags=re.I)
        return title

    def _label_read(self, pid: str, field: str) -> str | None:
        raw = self._raw_facts(pid).get(_LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _PULL[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return ("%g" % float(text)) + unit
        return _lone(found)

    def _page_read(self, pid: str, field: str) -> str | None:
        pile: set = set()
        clash = False
        raw = self._raw_facts(pid).get(_LABEL[field])
        if raw:
            got = _PULL[field](str(raw))
            clash = clash or len(got) > 1
            if field == "resolution" and _QHD_RX.search(str(raw)):
                clash = True
            pile |= got
        heading = self._heading(pid)
        if field == "resolution":
            heading = self._drop_codes(pid, heading)
            if _QHD_RX.search(heading):
                clash = True
        pile |= _adj(field, heading)
        return None if clash else _lone(pile)

    def _sku_read(self, pid: str, sku: str, field: str) -> str | None:
        opt = self._opt_text(pid, sku)
        nearby = _adj(field, opt)
        if len(nearby) > 1:
            return None
        if nearby:
            return next(iter(nearby))
        if _talks_about(field, opt):
            return None
        return self._page_read(pid, field)

    def _qty(self, pid: str, sku: str, field: str) -> float | None:
        value = self._sku_read(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @staticmethod
    def _roots(blob: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (blob or "").lower())}

    def _shelf_fits(self, pid: str) -> bool:
        want = self.ask.shelf
        path = self._shelf_path(pid)
        if not want or not path:
            return True
        wanted = self._roots(want)
        labels = set()
        for label in path:
            labels |= self._roots(str(label))
        placed = bool(wanted & labels) or any(
            self._roots("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        if not placed and "smartphone" in wanted:
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                for label in path
            )
        if not placed:
            return False
        shape = self.ask.shape
        if not shape:
            return True
        title = self._heading(pid)
        words = self._roots(title) | _wordbag(title)
        if shape == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in _gather_hertz(title))
            return "monitor" in words and ("gaming" in words or fast)
        return "over" in words and "ear" in words or "overear" in words or "circumaural" in words

    def _maker_fits(self, pid: str) -> bool:
        want = self.ask.maker
        if not want:
            return True
        wanted = _wordbag(want)
        return bool(wanted <= _wordbag(self._maker_of(pid)) or wanted <= _wordbag(self._heading(pid)))

    def _gadget_fits(self, pid: str, sku: str) -> bool:
        if not _noun_eq(self.ask.shelf or "", "smartphone"):
            return True
        verdict = _handset_verdict(self._raw_facts(pid), self._heading(pid), self._opt_text(pid, sku))
        if self.ask.lane in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def _used_lot(self, pid: str) -> bool:
        return bool(
            re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self._heading(pid), re.I)
        )

    def _distance(self, pid: str, sku: str) -> int:
        model, color = self.ask.want_model, self.ask.want_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self._heading(pid) + " " + " ".join(str(v) for v in self._raw_facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self._opt_text(pid, sku) + " " + str(self._raw_facts(pid).get("color") or "") + " " + self._heading(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _grade(self, pid: str, sku: str) -> dict:
        lane = self.ask.lane
        key = CoverageBook.vid(pid, sku)
        st = self.book.ticks.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.book.dead or (str(pid) in self.book.dead_pids and lane != "recovery")
        live = bool(stock) and price is not None and (self.purse is None or price <= self.purse + _TOL) and not dead
        viewed = str(pid) in self.book.pages
        field, want = self.ask.gate_field, self.ask.gate_value
        spec = 2
        if field and want:
            got = self._sku_read(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self._maker_fits(pid) and self._gadget_fits(pid, sku))
        cat_ok = self._shelf_fits(pid)
        ok = ok and cat_ok
        if lane == "justification":
            ok = ok and not self._used_lot(pid) and bool(self._maker_of(pid))
            if field and want:
                ok = ok and self._label_read(pid, field) == want and self._page_read(pid, field) == want
        if lane == "preference" and self.book.shock_key:
            dead_listing = self.book.listing_of.get(self.book.shock_key.split("::", 1)[0])
            if dead_listing and self.book.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if lane == "recall" and self.ask.theme:
            ok = ok and _theme_covered(self._heading(pid), self.ask.theme)
        if lane == "ranking":
            ok = ok and (str(pid), str(sku)) in self.ask.named_pairs
            ok = ok and self._qty(pid, sku, "refresh_rate") is not None and self._qty(pid, sku, "screen_size") is not None
        return {
            "pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live,
            "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok,
            "relax": self._distance(pid, sku) if lane == "recovery" else 0,
        }

    def _sort_key(self, c: dict, first: bool) -> tuple:
        lane = self.ask.lane
        price = c["price"] if c["price"] is not None else 1e12
        if lane == "ranking":
            hz = self._qty(c["pid"], c["sku"], "refresh_rate") or 0.0
            inch = self._qty(c["pid"], c["sku"], "screen_size") or 0.0
            return (-hz, -inch, price, c["key"])
        if lane == "recovery":
            return (c["relax"], price, c["key"])
        extra = sum(
            1 for f, v in self.ask.figures.items()
            if f != self.ask.gate_field and self._sku_read(c["pid"], c["sku"], f) != v
        )
        proven = 0 if _handset_verdict(self._raw_facts(c["pid"]), self._heading(c["pid"]), self._opt_text(c["pid"], c["sku"])) == "yes" else 1
        kin = 0
        sku_kin = 1
        if self.book.shock_key and not first:
            dead_pid, _, dead_sku = self.book.shock_key.partition("::")
            dead_title = self._heading(dead_pid)
            kin = -len(_wordbag(self._heading(c["pid"])) & _wordbag(dead_title))
            stem = str(dead_sku).rsplit("-", 1)[0]
            sku_kin = 0 if str(c["sku"]).rsplit("-", 1)[0] == stem else 1
        if lane == "intent" and first and self.ask.soft_brand:
            wanted = _wordbag(self.ask.soft_brand)
            near = 0 if (wanted <= _wordbag(self._maker_of(c["pid"])) or wanted <= _wordbag(self._heading(c["pid"]))) else 1
            return (near, extra, proven, price, c["key"])
        return (sku_kin, kin, extra, proven, price, c["key"])

    def _pool(self) -> list:
        seen = set()
        out = []
        for key in list(self.book.ticks):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._grade(pid, sku))
        return out

    def _wish_terms(self) -> set:
        if self._wants is None:
            said = self.ask.aim + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _wordbag(self.ask.shelf or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._wants = {w for w in _wordbag(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wants

    def _pref_hits(self, pid: str, sku: str) -> int:
        wants = self._wish_terms()
        if not wants:
            return 0
        have = _wordbag(
            " ".join(str(v) for v in self._raw_facts(pid).values()) + " " + self._heading(pid) + " " + self._opt_text(pid, sku)
        )
        return len(wants & have)

    def _pick(self) -> dict | None:
        first = self.book.adds == 0
        lane = self.ask.lane
        good = [c for c in self._pool() if c["ok"]]
        if lane == "preference":
            carted = [c for c in good if (c["pid"], c["sku"]) in self.book.basket]
            if carted:
                return carted[0]
            if good and self.step >= self.cap - 8:
                good.sort(key=lambda c: (-self._pref_hits(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
                return good[0]
            return None
        if lane == "intent" and first and self.ask.soft_brand:
            wanted = _wordbag(self.ask.soft_brand)
            named = [c for c in good if wanted <= _wordbag(self._maker_of(c["pid"])) or wanted <= _wordbag(self._heading(c["pid"]))]
            if named:
                good = named
            elif self.step < self.cap * 0.6:
                self._brand_nudge()
                return None
        if lane == "recovery" and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.step < self.cap * 0.6:
                self._exact_nudge()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._sort_key(c, first))
        return good[0]

    def _reply(self) -> str:
        target = self._pick() or (self._pool() or [None])[0]
        if not target:
            return "I am still matching the catalog to your gates; nothing is committed yet."
        pid, sku = target["pid"], target["sku"]
        bits = ["I am going with " + (self._heading(pid) or pid)[:90]]
        field, want = self.ask.gate_field, self.ask.gate_value
        if field and want and self._sku_read(pid, sku, field) == want:
            bits.append("its listed " + field.replace("_", " ") + " is " + want)
        if target.get("price") is not None:
            bits.append("it is %.2f %s and in stock" % (target["price"], self.ask.currency))
        return ", ".join(bits) + "."

    def _unread(self) -> list:
        leads = []
        for key, row in self.book.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.book.pages or str(pid) in self.book.dead_pids:
                continue
            st = self.book.ticks.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.purse is not None and price > self.purse + _TOL):
                continue
            title = str(row.get("title") or "")
            if self.ask.theme and not _theme_covered(title, self.ask.theme):
                continue
            if self.ask.want_model and self.book.adds == 0 and not self.book.shock_key:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.ask.want_model.lower()) not in squashed:
                    continue
            if self.ask.maker:
                wanted = _wordbag(self.ask.maker)
                if not (wanted <= _wordbag(str(row.get("brand") or "")) or wanted <= _wordbag(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _brand_nudge(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._whisper(
            "No %s unit that clears every hard gate has been read yet. Filter brand=%s inside the "
            "purse and read those pages before any other maker." % (self.ask.soft_brand, self.ask.soft_brand)
        )

    def _exact_nudge(self) -> None:
        if self.hinted_exact or not self.ask.want_model:
            return
        self.hinted_exact = True
        extra = ", " + self.ask.want_color if self.ask.want_color else ""
        self._whisper(
            "The exact pairing (%s%s) is still missing. The genome will search the model code; "
            "read every color before any first cart line." % (self.ask.want_model, extra)
        )

    def _shortlist(self) -> None:
        cands = self._pool()
        if self.ask.lane == "preference" and not self.book.basket:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: (-self._pref_hits(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
            if good:
                self._whisper(
                    "Units that clear every hard gate so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". Pick the one that best fits the stated priorities, inspect it, then add it."
                )
                return
        best = self._pick()
        if best:
            self._whisper(
                "Verified unit: %s::%s at %.2f. The closer will inspect, cart and order it; "
                "read another only if it could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        field, want = self.ask.gate_field, self.ask.gate_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self._whisper(
                "None of the pages read so far verify. Evidence for %s must resolve to exactly %s "
                "from the variant option text or labelled facts: %s. Look at different products."
                % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self._whisper(
                "The pages read so far fail a hard gate (shelf, maker, stock, purse or device kind). "
                "Let the genome mutate toward different wording."
            )

    def _cut_purse(self, blob: str) -> None:
        new = Request._read_purse(blob)
        if new is None or (self.purse is not None and new >= self.purse):
            return
        self.purse = new
        self.ask.purse = new
        self.shopper_moved = True
        self._whisper("Purse cut by the shopper: at most %.2f %s. Every later pick must fit." % (new, self.ask.currency))
        for pid, sku in list(self.book.basket):
            st = self.book.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _TOL:
                self.book.dead.add(CoverageBook.vid(pid, sku))
                self._whisper(
                    "Carted %s costs %.2f, above the new purse; the closer will replace it."
                    % (CoverageBook.vid(pid, sku), price)
                )

    def _claims(self, pid: str, sku: str) -> list:
        out = []
        field, want = self.ask.gate_field, self.ask.gate_value
        if not field and self.ask.lane == "justification":
            field, want = "resolution", self._label_read(pid, "resolution")
        if not field or not want:
            return []
        stated = dict(self.ask.figures) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self._label_read(pid, name) == value and self._page_read(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = self.book.status(pid, sku) or {}
        if self.purse is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.purse})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self._maker_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def _order_payload(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.ask.lane == "justification":
            claims = self._claims(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _legal_now(self, pid: str, sku: str) -> tuple:
        st = self.book.status(pid, sku)
        k = CoverageBook.vid(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.book.basket:
            return False, "add that exact variant to the cart first"
        if self.book.last_add_at is not None and self.book.tick_at.get(k, 0) <= self.book.last_add_at:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.purse is not None and price is not None and price > self.purse + _TOL:
            return False, "that variant costs %.2f, above the purse of %.2f" % (price, self.purse)
        if k in self.book.dead:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    def _can_commit(self) -> bool:
        if self.book.adds:
            return True
        if self.ask.hidden:
            if self.step < self.cap * 0.6:
                return False
            self.ask.hidden = False
            self._whisper("The hidden gate was never stated, so the remaining gates decide.")
        if self.ask.lane in ("intent", "constraint") and self.step < 4 and not self.shopper_moved:
            return False
        if self.ask.lane == "preference" and len(self.book.listings) < 2:
            return False
        roomy = self.step < self.cap * 0.45 and self.hour.left() > 140
        if self.ask.lane == "recall" and roomy:
            if self.book.starved(self.probes, self.step, self.cap, self.ask.need_ids) or self.probes < 2:
                return False
        if roomy and not self.pages_done and len({c["pid"] for c in self._pool() if c["ok"]}) < 2:
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = CoverageBook.vid(pid, sku)
        if k not in self.book.probed:
            return False
        return self.book.last_add_at is None or self.book.tick_at.get(k, 0) > self.book.last_add_at

    def _should_mutate(self) -> bool:
        if self.ask.lane == "ranking" and not self.book.shock_key:
            return False
        if self.ask.lane == "recall" and self.book.starved(self.probes, self.step, self.cap, self.ask.need_ids):
            return True
        if self._unread():
            return False
        if self.step >= self.cap - 5 and self.book.adds:
            return False
        if self._pick() and self._can_commit():
            return False
        return self.genome.has_pending()

    def _maybe_mutate(self) -> tuple | None:
        if self.book.shock_key and not self.genome.twin_done and "search" in self.tool_names:
            title = self._heading(self.book.shock_key.split("::", 1)[0])
            chromo = self.genome.graft_obituary(title)
            if chromo is not None and self.genome.remember(chromo):
                self._whisper(
                    "The dead listing shares a title with siblings; the genome grafted its first "
                    "eight tokens and will mutate from there."
                )
                return (chromo.verb, chromo.payload)
        if self.ask.lane == "recovery" and not self.book.adds and self.ask.want_model and "search" in self.tool_names:
            rows = [c for c in self._pool() if c["live"] and c["viewed"] and c["relax"] == 0]
            near = [c for c in self._pool() if c["viewed"] and c["relax"] == 1]
            if not rows and near:
                title = self._heading(near[0]["pid"])
                chromo = self.genome.graft_sibling(title)
                if chromo is not None and self.genome.remember(chromo):
                    return (chromo.verb, chromo.payload)
        if not self._should_mutate():
            return None
        chromo = self.genome.offer()
        if chromo is None:
            return None
        if chromo.verb not in self.tool_names:
            return None
        if not self.genome.remember(chromo):
            return None
        return (chromo.verb, chromo.payload)

    def _maybe_read(self) -> tuple | None:
        if self.ask.lane == "recall" and self.book.starved(self.probes, self.step, self.cap, self.ask.need_ids):
            return None
        if self.step >= self.cap - 5 and self.book.adds:
            return None
        if self.step >= self.cap - 5 and self._pick() and self._can_commit():
            return None
        leads = self._unread()
        if leads and self.forced_reads < 8 and self.step >= 2:
            self.forced_reads += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        if not leads and not self.genome.has_pending():
            self.pages_done = True
        return None

    def _maybe_close(self) -> tuple | None:
        if self.finished:
            return None
        if self.ask.lane == "ranking" and "compare" in self.tool_names:
            missing = [p for p, _s in self.ask.named_pairs if p not in self.book.pages and p not in self.opaque]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        target = self._pick()
        if target is not None and (target["pid"], target["sku"]) in self.book.basket:
            pid, sku = target["pid"], target["sku"]
            if not self._fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self._order_payload(pid, sku))
        if self.book.shock_key and not self.genome.twin_done:
            title = self._heading(self.book.shock_key.split("::", 1)[0])
            if title:
                return None
            self.genome.twin_done = True
        if self.book.shock_key and self._unread() and self.step < self.cap - 4:
            return None
        if target is None or not self._can_commit():
            return None
        pid, sku = target["pid"], target["sku"]
        if (pid, sku) in self.book.basket:
            if not self._fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self._order_payload(pid, sku))
        for cpid, csku in self.book.basket:
            if CoverageBook.vid(cpid, csku) in self.book.dead and "remove_from_cart" in self.tool_names:
                if self.step < self.cap - 4:
                    return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        late = self.step >= self.cap - 3
        if not self._fresh(pid, sku) and not late:
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    def _consult(self) -> dict | None:
        for attempt in range(3):
            model = self.bench[min(self.slot, len(self.bench) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _LINK.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._whisper("No tool call arrived. Fire exactly one of the published tools now.")
                continue
            if self.hour.left() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.slot = min(self.slot + 1, len(self.bench) - 1)
        return None

    def _post_env(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            result = _LINK.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.hour.left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _fire(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.held is not None:
            pend = self.held
            replay = self._post_env(pend["envelope"])
            if replay is None:
                self._whisper("The environment is silent; the held action will be retried.")
                return None
            self.held = None
            self.step = pend["turn"]
            self._ingest_env(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.finished:
                return None
        turn = self.step + 1
        group = self.problem_id + "-turn-" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
        }
        result = self._post_env(envelope)
        if result is None:
            self.held = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.step = turn
        return self._ingest_env(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _ingest_env(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.book.fingerprints.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps({"error": "environment call failed; it will be retried automatically"})}
            )
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.finished = True
        self.book.ingest(name, args, obs, self.ask)
        if name in ("search", "filter") and isinstance(obs, dict) and not obs.get("error"):
            titles = [str(r.get("title") or "") for r in (obs.get("results") or []) if isinstance(r, dict)]
            if self.genome.last is not None:
                self.genome.breed(self.genome.last, titles)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _clip(json.dumps(obs), _CLIP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._cut_purse(text)
            if self.ask.lane in ("constraint", "intent") and not self.ask.gate_value:
                stated = self.ask.absorb_gate(text)
                if stated:
                    self.shopper_moved = True
                    self._whisper("Hard gate recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            if _ASK_RX.search(text):
                self.owe_reply = True
                self._whisper("The shopper asked a question: answer with message next, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.finished = True
        return obs

    def _after_add(self) -> None:
        pid, sku = self.recheck
        self.recheck = None
        obs = self._fire("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _money(obs.get("price"))
        over = self.purse is not None and price is not None and price > self.purse + _TOL
        if stock and not over:
            self._whisper(
                "Post-add recheck: %s is still in stock at %s. The closer will order it."
                % (CoverageBook.vid(pid, sku), price)
            )
            return
        self.book.shock_key = CoverageBook.vid(pid, sku)
        self.shock_at = self.step
        self.book.dead.add(CoverageBook.vid(pid, sku))
        self.book.dead_pids.add(str(pid))
        self.forced_reads = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the purse" % (price or 0.0)
        guidance = {
            "recovery": "Recover with the smallest change: same model, other color, before any other model.",
            "ranking": "The remaining named pairs are re-ranked; the closer takes the new best.",
            "preference": "Recover to a product with a different listing_id that still fits the priorities.",
            "constraint": "Choose a different product that still meets every gate, not another variant of this one.",
        }.get(self.ask.lane, "Choose a different product that meets every hard gate.")
        self._whisper("Market shock: carted %s %s. " % (CoverageBook.vid(pid, sku), reason) + guidance)
        self._shortlist()

    def _panic_one(self, pid: str, sku: str) -> bool:
        remaining = self.cap - self.step
        if not self._fresh(pid, sku) and remaining > 3:
            obs = self._fire("inspect_stock", {"product_id": pid, "sku": sku})
            if self.finished or not isinstance(obs, dict) or not obs.get("in_stock"):
                return False
            price = _money(obs.get("price"))
            if self.purse is not None and price is not None and price > self.purse + _TOL:
                return False
        if (pid, sku) not in self.book.basket:
            obs = self._fire("add_to_cart", {"product_id": pid, "sku": sku})
            if self.finished or not isinstance(obs, dict) or obs.get("error"):
                return False
            obs = self._fire("inspect_stock", {"product_id": pid, "sku": sku})
            if self.finished or not isinstance(obs, dict) or not obs.get("in_stock"):
                return False
            price = _money(obs.get("price"))
            if self.purse is not None and price is not None and price > self.purse + _TOL:
                return False
        ok, _why = self._legal_now(pid, sku)
        if not ok:
            return False
        self._fire("place_test_order", self._order_payload(pid, sku))
        return self.finished

    def _panic_buy(self) -> None:
        for pid, sku in reversed(self.book.basket):
            ok, _why = self._legal_now(pid, sku)
            if ok and not self.finished:
                self._fire("place_test_order", self._order_payload(pid, sku))
                return
        if self.finished:
            return
        if self.step >= self.cap - 1:
            return
        good = [c for c in self._pool() if c["ok"]]
        if not good and self.ask.lane != "justification":
            good = [
                c for c in self._pool()
                if c["live"] and c["spec"] == 2 and self._maker_fits(c["pid"]) and c["cat_ok"]
                and (not self.ask.named_pairs or (c["pid"], c["sku"]) in self.ask.named_pairs)
            ]
        good.sort(key=lambda c: self._sort_key(c, self.book.adds == 0))
        for row in good[:4]:
            if self.finished:
                return
            if self._panic_one(row["pid"], row["sku"]):
                return

    def _mark_carted(self, obs: dict | None) -> None:
        if not isinstance(obs, dict) or obs.get("error"):
            return
        added = obs.get("added") or {}
        if added.get("product_id"):
            self.recheck = (str(added["product_id"]), str(added["sku"]))

    def run(self) -> list:
        question = self._first_ask()
        last_forced, repeats = None, 0
        while not self.finished and self.step < self.cap:
            if self.hour.left() < 25 or self.step >= self.cap - 3:
                self._panic_buy()
                if self.finished:
                    break
            if question and not self.asked and "message" in self.tool_names:
                self.asked = True
                self._fire("message", {"content": question})
                continue
            if self.recheck and "inspect_stock" in self.tool_names:
                self._after_add()
                continue
            if self.owe_reply:
                if self.q_waits < 2 and "message" in self.tool_names:
                    self.q_waits += 1
                    proposal = self._consult()
                    if proposal is None:
                        self._fire("message", {"content": self._reply()})
                        self.owe_reply = False
                        self.q_waits = 0
                        continue
                    call = proposal["tool_calls"][0]
                    name = (call.get("function") or {}).get("name") or ""
                    args = _parse_args(call)
                    if name == "message":
                        self._fire(name, args, proposal.get("content") or "", call.get("id"))
                        self.owe_reply = False
                        self.q_waits = 0
                        continue
                    self._fire("message", {"content": self._reply()})
                    self.owe_reply = False
                    self.q_waits = 0
                    continue
                if "message" in self.tool_names:
                    self._fire("message", {"content": self._reply()})
                self.owe_reply = False
                self.q_waits = 0
                continue
            forced = self._maybe_mutate() or self._maybe_read() or self._maybe_close()
            if forced is not None:
                name, args = forced
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.book.dead.add(CoverageBook.vid(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.opaque.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.owe_reply = False
                    last_forced, repeats = None, 0
                    self._whisper("That step changed nothing, so the closer moved on.")
                    continue
                obs = self._fire(name, args)
                if name == "message":
                    self.owe_reply = False
                    self.q_waits = 0
                if name == "add_to_cart":
                    self._mark_carted(obs)
                if name in ("search", "filter"):
                    self.probes += 1
                    if self.ask.lane == "recall" and self.ask.need_ids:
                        have = len(self.book.hits)
                        need = self.book.cover_need(self.probes, self.step, self.cap, self.ask.need_ids)
                        self._whisper(
                            "CoverageBook identities: %d of the required %d." % (have, need)
                            + ("" if have >= need else " The genome will keep mutating wording.")
                        )
                if name in ("view", "compare"):
                    self._shortlist()
                continue
            last_forced, repeats = None, 0
            proposal = self._consult()
            if proposal is None:
                self._panic_buy()
                if self.finished:
                    break
                continue
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _parse_args(call)
            content = proposal.get("content") or ""
            nudges = 0
            while True:
                veto = self._veto(name, args)
                if veto is None:
                    break
                if nudges >= _NUDGE_CAP:
                    self.nays += 1
                    self._whisper("That call was refused: " + veto + ". Choose a different valid action.")
                    if self.nays >= 3:
                        self._shortlist()
                        self.nays = 0
                    veto = "skip"
                    break
                nudges += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": veto})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + veto[:200]})
                again = self._consult()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _parse_args(call)
                content = again.get("content") or ""
            if veto == "skip":
                continue
            self.nays = 0
            if name not in self.tool_names:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.book.basket:
                    pid, sku = self.book.basket[-1]
                args = self._order_payload(pid, sku)
            if name == "message":
                if not self.owe_reply:
                    self.q_count += 1
                self.owe_reply = False
                self.q_waits = 0
            if name in ("search", "filter"):
                chromo = Chromosome(name, args)
                if not self.genome.remember(chromo):
                    self._whisper("Genome already fired that (name, args) pair; pick a different probe.")
                    continue
            obs = self._fire(name, args, content, call.get("id"))
            if name == "add_to_cart":
                self._mark_carted(obs)
            if name in ("search", "filter"):
                self.probes += 1
                if self.ask.lane == "recall" and self.ask.need_ids:
                    have = len(self.book.hits)
                    need = self.book.cover_need(self.probes, self.step, self.cap, self.ask.need_ids)
                    self._whisper(
                        "CoverageBook identities: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep mutating the query.")
                    )
            if name == "inspect_cart":
                self.cart_looks += 1
            if name in ("view", "compare"):
                self._shortlist()
        return self.dialogue

    def _veto(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.book.basket:
                pid, sku = self.book.basket[-1]
            ok, why = self._legal_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.ask.lane == "recall" and self.book.starved(self.probes, self.step, self.cap, self.ask.need_ids):
                need = self.book.cover_need(self.probes, self.step, self.cap, self.ask.need_ids)
                return (
                    "order refused: only %d distinct matching titles have been surfaced; mutate the query "
                    "until at least %d appear" % (len(self.book.hits), need)
                )
            if self.ask.lane == "preference" and len(self.book.listings) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self._grade(pid, sku)
            key = CoverageBook.vid(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over purse, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.book.probed:
                return "inspect_stock that exact variant first: current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.ask.gate_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.ask.gate_field, self.ask.gate_value)
                )
            if not c["ok"]:
                return "that product fails a hard gate (shelf, maker, or the kind of device requested)"
            if self.ask.lane == "preference" and len(self.book.listings) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.ask.lane == "recovery" and not self.book.adds and self.ask.want_model and self._distance(pid, sku):
                return (
                    "the first cart line must be the exact model and color; other colors share the title, "
                    "so let the genome search that title and read them first"
                )
            if not self._can_commit():
                return (
                    "hold: read one more product that also meets every gate first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.owe_reply and "?" not in text:
                return "do not send status updates; use message only to answer the shopper or to ask one question"
            if not self.owe_reply and self.q_count >= 2:
                return "the shopper has already answered; decide from the evidence rather than asking again"
        if name == "inspect_cart" and self.cart_looks >= 1:
            return "the cart contents are already known from earlier observations"
        if name in ("search", "filter"):
            stamp = Chromosome(name, args).stamp()
            if stamp in self.genome.used:
                return "genome already fired that (name, canonical args) pair; mutate the wording"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.book.fingerprints and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.book.fingerprints.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return MutantShop(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
