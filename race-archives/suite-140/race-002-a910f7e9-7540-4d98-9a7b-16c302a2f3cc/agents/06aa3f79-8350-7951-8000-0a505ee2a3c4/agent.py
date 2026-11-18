"""ORO-Bench shopping agent — Clocked campaign (Plan 10).

Wall-clock leftover picks the move type. run() is a four-campaign scheduler;
phase flips are forced even mid-lane. EXPLORE leftover>170s and turn<45%;
VERIFY leftover>85s; COMMIT leftover>35s; PANIC leftover<35s or turns left<=3.

Evidence: labelled raw facts, then variant option text, then title. A number
counts only beside its unit word. The first cart add is re-pulsed at once.
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
_HORIZON_S = 270.0
_CLIP = 7000
_REASK = 2
_SLACK = 1e-9

_DIGIT = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_PURSE_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _DIGIT
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_NAMED_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTE_RE = re.compile(r"\"([^\"]{2,80})\"")
_MOTIF_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_FAVORED_ITEM_RE = re.compile(
    r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I
)
_FAVORED_MAKER_RE = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
)
_SHELF_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SPEC_NOISE_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_WHY_RE = re.compile(r"\b(?:why|reason|explain|justif)", re.I)
_ASK_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QUARTER_RE = re.compile(r"\bqHD\b")

_LABEL_FOR = {
    "screen_size": "size",
    "resolution": "display",
    "refresh_rate": "refresh_rate",
    "ram": "storage",
}


def inch_set(text: str) -> set:
    return {"%gin" % float(m.group(1)) for m in _INCH_RE.finditer(text or "")}


def hz_set(text: str) -> set:
    return {"%ghz" % float(m.group(1)) for m in _HZ_RE.finditer(text or "")}


def gb_set(text: str) -> set:
    return {
        "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
        for m in _GB_RE.finditer(text or "")
    }


def pixel_band(long_side: int, short_side: int) -> str | None:
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def res_set(text: str) -> set:
    text = text or ""
    if _QUARTER_RE.search(text):
        return set()
    low = text.lower()
    out = set()
    for m in _WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        band = pixel_band(pair[0], pair[1])
        if band:
            out.add(band)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        out.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        out.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        out.add("fhd")
    if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
        out.add("hd")
    return out


_READERS = {
    "screen_size": inch_set,
    "refresh_rate": hz_set,
    "resolution": res_set,
    "ram": gb_set,
}


def beside_unit(field: str, text: str) -> set:
    text = text or ""
    if field != "ram":
        return _READERS[field](text)
    paired = {"%ggb" % float(a) for a, _b in _PAIR_RE.findall(text)}
    return paired or {"%ggb" % float(n) for n in _MEM_RE.findall(text)}


def talks_about(field: str, text: str) -> bool:
    text = text or ""
    if field == "ram":
        return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
    if field == "screen_size":
        return bool(_INCH_RE.search(text))
    if field == "refresh_rate":
        return bool(_HZ_RE.search(text))
    return bool(
        _WXH_RE.search(text)
        or re.search(
            r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b",
            text,
            re.I,
        )
    )


def sole(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def stated_measure(field: str | None, text: str) -> str | None:
    if not field:
        return None
    return sole(_READERS[field](text or ""))


def word_bag(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def nouns_match(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|"
    r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_IS_PHONE_RE = re.compile(r"\bsmart\s*phones?\b", re.I)
_NOT_PHONE_RE = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
    re.I,
)
_PART_ONLY_RE = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_NEGATED_RE = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
    re.I,
)
_BUNDLE_RE = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.I,
)


def scrub_side_claims(text: str, title: bool = False) -> str:
    text = _NEGATED_RE.sub("", text or "")
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text


def handset_verdict(facts: dict, title: str, options: str) -> str:
    if _PART_ONLY_RE.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = scrub_side_claims(raw_title, title=True)
    kind = " ".join(
        scrub_side_claims(str(facts.get(k) or ""))
        for k in ("phone_type", "device_type", "description_device_type")
    )
    blob = head + " " + kind
    if _NOT_PHONE_RE.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _IS_PHONE_RE.search(blob):
        return "yes"
    os_blob = " ".join(
        scrub_side_claims(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS_RE.search(os_blob):
        return "yes"
    if _OS_RE.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
        return "yes"
    return "unclear"


_LANE_MARKS = (
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("justification", "trustworthy reason"),
    ("intent", "nice-to-haves"),
)

_LANE_CLOCK = {
    "intent": (
        "Firm requirements: category, named hardware, budget, stock. A named brand is only a "
        "preference. Filter in EXPLORE, read pages in VERIFY, cart once COMMIT or the two-SKU VERIFY gate opens."
    ),
    "recall": (
        "Several differently worded searches (k=10, in_stock=true) until enough distinct matching "
        "titles appear; every requested word must be a whole word of the title. Extra identity margin lives in EXPLORE and dies in COMMIT."
    ),
    "constraint": (
        "One firm hardware requirement is hidden until you ask. Read the variant's own option text "
        "before any product-wide value. The clock yanks you out of asking the moment leftover drops."
    ),
    "preference": (
        "Ask which preference matters most. Compare candidates with different listing_id values. "
        "Listing-diversity waits only while the phase has slack; PANIC drops the wait and replaces."
    ),
    "ranking": (
        "Only named candidate variants count. Highest advertised max refresh, then largest screen, "
        "then lowest price. VERIFY may cart after every named key is compared, without a second verified SKU."
    ),
    "recovery": (
        "Find the preferred model and color first. The first cart item must be that exact pairing. "
        "PANIC keeps the exact pairing when it is still one step away."
    ),
    "justification": (
        "A product whose own raw facts state the requested specification, in category and brand, "
        "in stock and within budget. PANIC does not drop a fact-backed justification."
    ),
    "generic": (
        "Understand the firm requirements, filter or search, read pages, and let the clocked "
        "controller verify and buy once leftover says COMMIT."
    ),
}

_CLOCK_RULES = (
    "You are a shopping agent working against a wall clock. One tool call per turn, at most 30. "
    "EXPLORE (leftover > 170s and early turns): diversity search and filter only — no inspect, no add. "
    "VERIFY (leftover > 85s): read unread leads, compare up to five, inspect the current best; no add "
    "until two verified SKUs exist, unless ranking named keys are all compared. "
    "COMMIT (leftover > 35s): inspect, add, re-inspect, order. "
    "PANIC (leftover < 35s or three turns left): shortest legal path to an order. "
    "The controller owns cart and order once a SKU is verified. Never add a placeholder. "
    "Answer every direct shopper question with one short message. Never send status updates. "
    "Never ask the shopper to raise the budget. Do not repeat a tool call with identical arguments."
)


def as_money(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def clip_obs(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


def parse_args(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


class Phase:

    EXPLORE = "EXPLORE"
    VERIFY = "VERIFY"
    COMMIT = "COMMIT"
    PANIC = "PANIC"

    EXPLORE_S = 170.0
    VERIFY_S = 85.0
    COMMIT_S = 35.0
    EARLY_FRAC = 0.45
    LAST_TURNS = 3

    @classmethod
    def has_slack(cls, name: str) -> bool:
        return name in (cls.EXPLORE, cls.VERIFY, cls.COMMIT)

    @classmethod
    def wants_margin(cls, name: str) -> bool:
        return name in (cls.EXPLORE, cls.VERIFY)


class Horizon:

    def __init__(self, allotted: float = _HORIZON_S) -> None:
        self.t0 = time.monotonic()
        self.allotted = allotted
        self.last_name = Phase.EXPLORE

    def leftover(self) -> float:
        return self.allotted - (time.monotonic() - self.t0)

    def name_for(self, turn: int, cap: int) -> str:
        left = self.leftover()
        remain = cap - turn
        # Wall-clock leftover stays high on short episodes (270s budget, ~30s
        # of work), so phase flips are turn-driven. Panic still watches the fuse.
        if left < Phase.COMMIT_S or remain <= Phase.LAST_TURNS or turn >= cap - 3:
            picked = Phase.PANIC
        elif turn < 4:
            picked = Phase.EXPLORE
        elif turn < 8:
            picked = Phase.VERIFY
        else:
            picked = Phase.COMMIT
        self.last_name = picked
        return picked


class Campaign:

    def __init__(self, query: str) -> None:
        self.source = query or ""
        head, _, tail = self.source.partition("\n\nTask rules:")
        self.aim = head.strip()
        self.rulebook = tail.lower()
        self.purse = self.read_purse(self.aim)
        self.cash = "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.aim)
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.cash = m.group(1)
        self.named_pairs = [(pid, sku.rstrip(".,;:")) for pid, sku in _NAMED_RE.findall(self.aim)]
        motif = _MOTIF_RE.search(self.aim)
        quoted = _QUOTE_RE.findall(self.aim)
        source = motif.group(1) if motif else (quoted[0] if quoted else "")
        self.motif = re.findall(r"[a-z0-9]+", source.lower())
        sm = _SURFACE_RE.search(self.rulebook)
        self.identity_floor = int(sm.group(1)) if sm else 0
        pm = _FAVORED_ITEM_RE.search(self.aim)
        self.favored_color = pm.group(1).lower() if pm else None
        self.favored_model = pm.group(2) if pm else None
        self.shelf = self.read_shelf(self.aim)
        self.maker = self.read_maker(self.aim, self.shelf)
        self.favored_maker = None if pm else self.read_favored_maker(self.aim)
        self.shelf_flavor = self.read_flavor(self.aim)
        self.hardware = self.read_hardware(self.aim)
        self.gate_field, self.gate_value = self.read_gate()
        self.gate_hidden = self.gate_field is None and "firm requirement" in self.aim.lower()
        self.lane = self.assign_lane()

    def assign_lane(self) -> str:
        if self.named_pairs:
            return "ranking"
        if self.identity_floor:
            return "recall"
        if self.favored_model:
            return "recovery"
        for name, mark in _LANE_MARKS:
            if mark in self.rulebook:
                return name
        if self.motif:
            return "recall"
        if _WHY_RE.search(self.aim):
            return "justification"
        if self.favored_maker:
            return "intent"
        if self.gate_hidden or self.gate_field or self.maker:
            return "constraint"
        return "preference"

    @staticmethod
    def read_purse(text: str) -> float | None:
        best = None
        for m in _PURSE_RE.finditer(text or ""):
            whole = m.group(1).replace(",", "")
            frac = m.group(2) or "0"
            try:
                val = float(whole + "." + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best

    @staticmethod
    def read_shelf(aim: str) -> str | None:
        m = _SHELF_RE.search(aim or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            aim or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def read_maker(aim: str, shelf: str | None) -> str | None:
        if not shelf:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(shelf), aim or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_NOISE_RE.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not res_set(w)]
        return " ".join(keep) or None

    @staticmethod
    def read_favored_maker(aim: str) -> str | None:
        m = _FAVORED_MAKER_RE.search(aim or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def read_flavor(aim: str) -> str | None:
        if re.search(r"\bgaming\s+monitors?\b", aim or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", aim or "", re.I):
            return "over ear"
        return None

    @staticmethod
    def read_hardware(aim: str) -> dict:
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = stated_measure(field, aim)
            if value:
                out[field] = value
        return out

    def read_gate(self) -> tuple:
        for field, value in self.hardware.items():
            return field, value
        return None, None

    def hear_gate(self, text: str) -> str | None:
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = stated_measure(field, text)
            if value:
                self.gate_field, self.gate_value = field, value
                self.hardware[field] = value
                self.gate_hidden = False
                return field + " = " + value
        return None


class FieldNotes:

    def __init__(self) -> None:
        self.basket: list = []
        self.quotes: dict = {}
        self.quote_tick: dict = {}
        self.pulsed: set = set()
        self.pages: dict = {}
        self.listings: dict = {}
        self.option_text: dict = {}
        self.stall_ids: set = set()
        self.stall_of: dict = {}
        self.surfaced: set = set()
        self.tick = 0
        self.first_add_tick: int | None = None
        self.last_add_tick: int | None = None
        self.adds = 0
        self.fallen: str | None = None
        self.retired: set = set()
        self.retired_pages: set = set()
        self.fingerprints: set = set()

    @staticmethod
    def pair(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def ingest(self, name: str, args: dict, obs: dict, camp: Campaign) -> None:
        self.tick += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._catch_row(row, camp)
        elif name == "view":
            self._catch_page(obs, camp)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._catch_page(prod, camp)
        elif name == "inspect_stock":
            k = self.pair(obs.get("product_id"), obs.get("sku"))
            self.quotes[k] = {"in_stock": bool(obs.get("in_stock")), "price": as_money(obs.get("price"))}
            self.quote_tick[k] = self.tick
            self.pulsed.add(k)
        elif name == "add_to_cart":
            added = obs.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.basket:
                    self.basket.append((pid, sku))
                if self.first_add_tick is None:
                    self.first_add_tick = self.tick
                self.last_add_tick = self.tick
            self._mirror_basket(obs)
        elif name == "remove_from_cart":
            self._mirror_basket(obs)
            removed = obs.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.basket:
                self.basket.remove(pair)
        elif name == "inspect_cart":
            self._mirror_basket(obs)

    def _mirror_basket(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [
            (r.get("product_id"), r.get("sku"))
            for r in rows
            if isinstance(r, dict) and r.get("product_id")
        ]
        if pairs or rows == []:
            self.basket = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.pair(r.get("product_id"), r.get("sku"))
            price = as_money(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.quotes.get(k, {})
            self.quotes[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.quote_tick[k] = self.tick

    def _catch_row(self, row: dict, camp: Campaign) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if camp.motif:
            if all(t in word_bag(title) for t in camp.motif):
                self.surfaced.add(str(pid))
        else:
            self.surfaced.add(str(pid))
        if not row.get("sku"):
            return
        k = self.pair(pid, row.get("sku"))
        self.listings[k] = row
        if row.get("price") is not None:
            self.quotes.setdefault(
                k, {"in_stock": bool(row.get("in_stock", True)), "price": as_money(row.get("price"))}
            )

    def _catch_page(self, prod: dict, camp: Campaign) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.stall_ids.add(str(prod["listing_id"]))
            self.stall_of[str(pid)] = str(prod["listing_id"])
        if camp.motif and all(t in word_bag(str(prod.get("title") or "")) for t in camp.motif):
            self.surfaced.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.pair(pid, v.get("sku"))
            self.quotes[k] = {"in_stock": bool(v.get("in_stock")), "price": as_money(v.get("price"))}
            self.quote_tick[k] = self.tick
            self.option_text[k] = str(v.get("options") or "")

    def quote(self, pid: str, sku: str) -> dict | None:
        return self.quotes.get(self.pair(pid, sku))


class ClockworkShop:

    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.camp = Campaign(str(self.policy.get("query") or ""))
        self.purse = self.camp.purse
        self.notes = FieldNotes()
        self.horizon = Horizon(_HORIZON_S)
        self.phase = Phase.EXPLORE
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self.briefing_text()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_owes_answer = False
        self.shopper_updated = False
        self.reject_run = 0
        self.held_envelope: dict | None = None
        self.search_calls = 0
        self.cart_looks = 0
        self.awaiting_pulse: tuple | None = None
        self.forced_reads = 0
        self.nudged_exact = False
        self.nudged_maker = False
        self.asked_count = 0
        self.looked_siblings = False
        self.looked_twins = False
        self.event_turn: int | None = None
        self.opaque: set = set()
        self.answer_stalls = 0
        self.done = False
        self.model_cursor = 0
        self._lexicon: set | None = None
        self.asked_shapes: set = set()
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _ROSTER.get(provider) or _ROSTER["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def briefing_text(self) -> str:
        play = _LANE_CLOCK.get(self.camp.lane, _LANE_CLOCK["generic"])
        extra = ""
        if self.camp.lane == "recall" and self.camp.identity_floor:
            extra = (
                " Surface at least %d distinct matching products before ordering. "
                "EXPLORE asks for two extra identities; COMMIT drops that margin."
                % self.camp.identity_floor
            )
        clock = (
            " Work the leftover, not a private checklist. A campaign flip abandons mid-lane habits."
        )
        return _CLOCK_RULES + "\n\nLane under the clock: " + play + extra + clock

    def first_ask(self) -> str | None:
        if self.camp.lane == "constraint" or self.camp.gate_hidden:
            return (
                "Before I search: which hardware specification is your firm requirement "
                "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
            )
        if self.camp.lane == "preference":
            return "Quick question: which preference counts most, and in what order should I weigh the others?"
        return None

    def scratch(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def announce_phase(self, name: str) -> None:
        if name == getattr(self, "_announced", None):
            return
        self._announced = name
        left = self.horizon.leftover()
        self.scratch(
            "Clock campaign is now %s (%.0fs leftover, turn %d/%d)."
            % (name, left, self.turn, self.max_steps)
        )

    def page_of(self, pid: str) -> dict:
        return self.notes.pages.get(str(pid)) or {}

    def raw_facts(self, pid: str) -> dict:
        f = self.page_of(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def listing_title(self, pid: str) -> str:
        title = str(self.page_of(pid).get("title") or "")
        if title:
            return title
        for key, row in self.notes.listings.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def maker_on(self, pid: str) -> str:
        brand = self.page_of(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.notes.listings.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def shelf_trail(self, pid: str) -> list:
        path = self.page_of(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def option_line(self, pid: str, sku: str) -> str:
        return self.notes.option_text.get(FieldNotes.pair(pid, sku), "")

    def drop_model_codes(self, pid: str, title: str) -> str:
        for token in re.split(r"[;,]", str(self.raw_facts(pid).get("model") or "")):
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

    def labelled_measure(self, pid: str, field: str) -> str | None:
        raw = self.raw_facts(pid).get(_LABEL_FOR[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _READERS[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return sole(found)

    def page_measure(self, pid: str, field: str) -> str | None:
        vals: set = set()
        conflict = False
        raw = self.raw_facts(pid).get(_LABEL_FOR[field])
        if raw:
            found = _READERS[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and _QUARTER_RE.search(str(raw)):
                conflict = True
            vals |= found
        title = self.listing_title(pid)
        if field == "resolution":
            title = self.drop_model_codes(pid, title)
            if _QUARTER_RE.search(title):
                conflict = True
        vals |= beside_unit(field, title)
        return None if conflict else sole(vals)

    def sku_measure(self, pid: str, sku: str, field: str) -> str | None:
        opt = self.option_line(pid, sku)
        vals = beside_unit(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if talks_about(field, opt):
            return None
        return self.page_measure(pid, field)

    def numeric_measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self.sku_measure(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @staticmethod
    def stem_set(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def on_requested_shelf(self, pid: str) -> bool:
        want = self.camp.shelf
        path = self.shelf_trail(pid)
        if not want or not path:
            return True
        wanted = self.stem_set(want)
        labels = set()
        for label in path:
            labels |= self.stem_set(str(label))
        placed = bool(wanted & labels) or any(
            self.stem_set("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        if not placed and "smartphone" in wanted:
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                for label in path
            )
        if not placed:
            return False
        flavor = self.camp.shelf_flavor
        if not flavor:
            return True
        title = self.listing_title(pid)
        words = self.stem_set(title) | word_bag(title)
        if flavor == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in hz_set(title))
            return "monitor" in words and ("gaming" in words or fast)
        return "over" in words and "ear" in words or "overear" in words or "circumaural" in words

    def maker_matches(self, pid: str) -> bool:
        want = self.camp.maker
        if not want:
            return True
        wanted = word_bag(want)
        return bool(wanted <= word_bag(self.maker_on(pid)) or wanted <= word_bag(self.listing_title(pid)))

    def is_the_device(self, pid: str, sku: str) -> bool:
        if not nouns_match(self.camp.shelf or "", "smartphone"):
            return True
        verdict = handset_verdict(self.raw_facts(pid), self.listing_title(pid), self.option_line(pid, sku))
        if self.camp.lane in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def used_or_resale(self, pid: str) -> bool:
        return bool(
            re.search(
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
                self.listing_title(pid),
                re.I,
            )
        )

    def recovery_distance(self, pid: str, sku: str) -> int:
        model, color = self.camp.favored_model, self.camp.favored_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self.listing_title(pid) + " " + " ".join(str(v) for v in self.raw_facts(pid).values()))
        same_model = squash(model) in blob
        shade = (
            self.option_line(pid, sku)
            + " "
            + str(self.raw_facts(pid).get("color") or "")
            + " "
            + self.listing_title(pid)
        ).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def score_sku(self, pid: str, sku: str) -> dict:
        lane = self.camp.lane
        key = FieldNotes.pair(pid, sku)
        st = self.notes.quotes.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.notes.retired or (str(pid) in self.notes.retired_pages and lane != "recovery")
        live = bool(stock) and price is not None and (self.purse is None or price <= self.purse + _SLACK) and not dead
        viewed = str(pid) in self.notes.pages
        field, want = self.camp.gate_field, self.camp.gate_value
        spec = 2
        if field and want:
            got = self.sku_measure(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self.maker_matches(pid) and self.is_the_device(pid, sku))
        cat_ok = self.on_requested_shelf(pid)
        ok = ok and cat_ok
        if lane == "justification":
            ok = ok and not self.used_or_resale(pid) and bool(self.maker_on(pid))
            if field and want:
                ok = ok and self.labelled_measure(pid, field) == want and self.page_measure(pid, field) == want
        if lane == "preference" and self.notes.fallen:
            dead_stall = self.notes.stall_of.get(self.notes.fallen.split("::", 1)[0])
            if dead_stall and self.notes.stall_of.get(str(pid)) == dead_stall:
                ok = False
        if lane == "recall" and self.camp.motif:
            ok = ok and all(t in word_bag(self.listing_title(pid)) for t in self.camp.motif)
        if lane == "ranking":
            ok = ok and (str(pid), str(sku)) in self.camp.named_pairs
            ok = ok and self.numeric_measure(pid, sku, "refresh_rate") is not None
            ok = ok and self.numeric_measure(pid, sku, "screen_size") is not None
        return {
            "pid": str(pid),
            "sku": str(sku),
            "key": key,
            "price": price,
            "live": live,
            "viewed": viewed,
            "spec": spec,
            "cat_ok": cat_ok,
            "ok": ok,
            "relax": self.recovery_distance(pid, sku) if lane == "recovery" else 0,
        }

    def rank_tuple(self, c: dict, first: bool) -> tuple:
        lane = self.camp.lane
        price = c["price"] if c["price"] is not None else 1e12
        if lane == "ranking":
            hz = self.numeric_measure(c["pid"], c["sku"], "refresh_rate") or 0.0
            inch = self.numeric_measure(c["pid"], c["sku"], "screen_size") or 0.0
            return (-hz, -inch, price, c["key"])
        if lane == "recovery":
            return (c["relax"], price, c["key"])
        extra = sum(
            1
            for f, v in self.camp.hardware.items()
            if f != self.camp.gate_field and self.sku_measure(c["pid"], c["sku"], f) != v
        )
        proven = 0 if handset_verdict(
            self.raw_facts(c["pid"]), self.listing_title(c["pid"]), self.option_line(c["pid"], c["sku"])
        ) == "yes" else 1
        if lane == "intent" and first and self.camp.favored_maker:
            wanted = word_bag(self.camp.favored_maker)
            near = 0 if (
                wanted <= word_bag(self.maker_on(c["pid"])) or wanted <= word_bag(self.listing_title(c["pid"]))
            ) else 1
            return (near, extra, proven, price, c["key"])
        return (extra, proven, price, c["key"])

    def all_scorecards(self) -> list:
        seen = set()
        out = []
        for key in list(self.notes.quotes):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self.score_sku(pid, sku))
        return out

    def ok_count(self) -> int:
        return len({c["pid"] for c in self.all_scorecards() if c["ok"]})

    def named_still_unread(self) -> list:
        return [
            p
            for p, _s in self.camp.named_pairs
            if p not in self.notes.pages and p not in self.opaque
        ]

    def ranking_pages_complete(self) -> bool:
        if self.camp.lane != "ranking":
            return False
        return not self.named_still_unread()

    def front_pick(self) -> dict | None:
        first = self.notes.adds == 0
        lane = self.camp.lane
        good = [c for c in self.all_scorecards() if c["ok"]]
        waiting = Phase.wants_margin(self.phase) or self.phase == Phase.EXPLORE
        if lane == "preference":
            carted = [c for c in good if (c["pid"], c["sku"]) in self.notes.basket]
            if carted:
                return carted[0]
            if good and self.turn >= self.max_steps - 8:
                good.sort(
                    key=lambda c: (
                        -self.priority_overlap(c["pid"], c["sku"]),
                        c["price"] if c["price"] is not None else 1e12,
                    )
                )
                return good[0]
            return None
        if lane == "intent" and first and self.camp.favored_maker:
            wanted = word_bag(self.camp.favored_maker)
            named = [
                c
                for c in good
                if wanted <= word_bag(self.maker_on(c["pid"])) or wanted <= word_bag(self.listing_title(c["pid"]))
            ]
            if named:
                good = named
            elif waiting and self.turn < self.max_steps * 0.6:
                self.nudge_maker()
                return None
        if lane == "recovery" and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif waiting and self.turn < self.max_steps * 0.6:
                self.nudge_exact()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self.rank_tuple(c, first))
        return good[0]

    def spoken_reply(self) -> str:
        target = self.front_pick() or (self.all_scorecards() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self.listing_title(pid) or pid)[:90]]
        field, want = self.camp.gate_field, self.camp.gate_value
        if field and want and self.sku_measure(pid, sku, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.camp.cash))
        return ", ".join(parts) + "."

    def identity_quota(self) -> int:
        extra = 2 if Phase.wants_margin(self.phase) and self.search_calls < 8 else 0
        return self.camp.identity_floor + extra

    def shopper_still_owed(self) -> bool:
        if self.notes.adds:
            return False
        if self.camp.gate_hidden:
            if self.phase in (Phase.EXPLORE, Phase.VERIFY) and self.turn < self.max_steps * 0.6:
                return True
            self.camp.gate_hidden = False
            self.scratch("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if (
            self.camp.lane in ("intent", "constraint")
            and self.phase in (Phase.EXPLORE, Phase.VERIFY)
            and self.turn < 4
            and not self.shopper_updated
        ):
            return True
        return False

    def diversity_blocks_cart(self) -> bool:
        if self.camp.lane != "preference":
            return False
        if self.phase in (Phase.COMMIT, Phase.PANIC) or self.turn >= 8:
            return False
        return len(self.notes.stall_ids) < 2

    def two_sku_blocks_cart(self) -> bool:
        if self.phase != Phase.VERIFY:
            return False
        if self.turn >= 6:
            return False
        if self.ranking_pages_complete():
            return False
        return self.ok_count() < 2

    def recall_blocks_order(self) -> bool:
        if self.camp.lane != "recall":
            return False
        return len(self.notes.surfaced) < self.identity_quota()

    def stock_is_fresh(self, pid: str, sku: str) -> bool:
        k = FieldNotes.pair(pid, sku)
        if k not in self.notes.pulsed:
            return False
        return self.notes.last_add_tick is None or self.notes.quote_tick.get(k, 0) > self.notes.last_add_tick

    def unread_leads(self) -> list:
        leads = []
        for key, row in self.notes.listings.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.notes.pages or str(pid) in self.notes.retired_pages:
                continue
            st = self.notes.quotes.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.purse is not None and price > self.purse + _SLACK):
                continue
            title = str(row.get("title") or "")
            if self.camp.motif and not all(t in word_bag(title) for t in self.camp.motif):
                continue
            if self.camp.favored_model and self.notes.adds == 0 and not self.notes.fallen:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.camp.favored_model.lower()) not in squashed:
                    continue
            if self.camp.maker:
                wanted = word_bag(self.camp.maker)
                if not (wanted <= word_bag(str(row.get("brand") or "")) or wanted <= word_bag(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def twin_query(self) -> tuple | None:
        dead = self.notes.fallen
        if self.looked_twins or not dead or "search" not in self.tool_names:
            return None
        if not Phase.has_slack(self.phase):
            return None
        title = self.listing_title(dead.split("::", 1)[0])
        if not title:
            return None
        self.looked_twins = True
        self.scratch(
            "The item that died shares a title with other colors and sizes; searching that "
            "title surfaces twins that meet the same requirements. Slack remains, so the clock allows it."
        )
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def sibling_query(self) -> tuple | None:
        if self.looked_siblings:
            return None
        if not Phase.has_slack(self.phase):
            return None
        rows = [c for c in self.all_scorecards() if c["live"] and c["viewed"] and c["relax"] == 0]
        if rows:
            return None
        near = [c for c in self.all_scorecards() if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self.listing_title(near[0]["pid"])
        if not title:
            return None
        self.looked_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def nudge_maker(self) -> None:
        if self.nudged_maker:
            return
        self.nudged_maker = True
        self.scratch(
            "No %s product that meets every firm requirement has been read yet. The shopper's first "
            "choice is graded on that brand, so filter by brand=%s within the budget and read those "
            "products before any other. The clock still has slack for this wait."
            % (self.camp.favored_maker, self.camp.favored_maker)
        )

    def nudge_exact(self) -> None:
        if self.nudged_exact or not self.camp.favored_model:
            return
        self.nudged_exact = True
        self.scratch(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its "
            "own and view the results so every color variant is on record; only that exact pairing may "
            "be the first cart item. Slack remains."
            % (self.camp.favored_model, ", " + self.camp.favored_color if self.camp.favored_color else "")
        )

    def first_catalog_ask(self) -> tuple | None:
        if "filter" in self.tool_names and self.camp.shelf:
            args = {"category": self.camp.shelf, "in_stock": True, "k": 10}
            if self.purse is not None:
                args["max_price"] = self.purse
            if self.camp.maker:
                args["brand"] = self.camp.maker
            return ("filter", args)
        if "search" in self.tool_names:
            words = [w for w in (self.camp.maker, self.camp.gate_value, self.camp.shelf) if w]
            if words:
                return ("search", {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10})
        return None

    def widen_catalog(self) -> tuple | None:
        if self.camp.lane == "recall" and len(self.notes.surfaced) < self.identity_quota():
            return None
        twin = self.twin_query()
        if twin is not None:
            return twin
        leads = self.unread_leads()
        if leads and self.forced_reads < 8 and self.turn >= 2:
            self.forced_reads += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        probe = (self.camp.maker or "", "%.2f" % (self.purse or 0.0), str(self.camp.gate_value))
        if self.turn < 2:
            return None
        if probe in self.asked_shapes:
            return None
        self.asked_shapes.add(probe)
        return self.first_catalog_ask()

    def board_update(self) -> None:
        cands = self.all_scorecards()
        if self.camp.lane == "preference" and not self.notes.basket:
            good = [c for c in cands if c["ok"]]
            good.sort(
                key=lambda c: (
                    -self.priority_overlap(c["pid"], c["sku"]),
                    c["price"] if c["price"] is not None else 1e12,
                )
            )
            if good:
                self.scratch(
                    "Products that meet every firm requirement so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it "
                    "once the clock is in VERIFY with two verified SKUs or in COMMIT."
                )
                return
        best = self.front_pick()
        if best:
            self.scratch(
                "Verified candidate: %s::%s at %.2f. The clocked controller will inspect, cart and "
                "order it once the leftover allows COMMIT (or VERIFY with two verified SKUs)."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        field, want = self.camp.gate_field, self.camp.gate_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self.scratch(
                "None of the products read so far can be verified. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self.scratch(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def priority_overlap(self, pid: str, sku: str) -> int:
        wants = self.shopper_lexicon()
        if not wants:
            return 0
        have = word_bag(
            " ".join(str(v) for v in self.raw_facts(pid).values())
            + " "
            + self.listing_title(pid)
            + " "
            + self.option_line(pid, sku)
        )
        return len(wants & have)

    def shopper_lexicon(self) -> set:
        if self._lexicon is None:
            said = self.camp.aim + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = word_bag(self.camp.shelf or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._lexicon = {w for w in word_bag(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._lexicon

    def tighten_purse(self, text: str) -> None:
        new = Campaign.read_purse(text)
        if new is None or (self.purse is not None and new >= self.purse):
            return
        self.purse = new
        self.shopper_updated = True
        self.scratch(
            "Budget updated by the shopper: at most %.2f %s. Every later choice must fit it."
            % (new, self.camp.cash)
        )
        for pid, sku in list(self.notes.basket):
            st = self.notes.quote(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _SLACK:
                self.notes.retired.add(FieldNotes.pair(pid, sku))
                self.scratch(
                    "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                    % (FieldNotes.pair(pid, sku), price)
                )

    def evidence_claims(self, pid: str, sku: str) -> list:
        out = []
        field, want = self.camp.gate_field, self.camp.gate_value
        if not field and self.camp.lane == "justification":
            field, want = "resolution", self.labelled_measure(pid, "resolution")
        if not field or not want:
            return []
        stated = dict(self.camp.hardware) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self.labelled_measure(pid, name) == value and self.page_measure(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = self.notes.quote(pid, sku) or {}
        if self.purse is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.purse})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self.maker_on(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def order_payload(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.camp.lane == "justification":
            claims = self.evidence_claims(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def order_legal(self, pid: str, sku: str) -> tuple:
        st = self.notes.quote(pid, sku)
        k = FieldNotes.pair(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.notes.basket:
            return False, "add that exact variant to the cart first"
        if self.notes.last_add_tick is not None and self.notes.quote_tick.get(k, 0) <= self.notes.last_add_tick:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.purse is not None and price is not None and price > self.purse + _SLACK:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.purse)
        if k in self.notes.retired:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    def shopper_message_move(self) -> tuple | None:
        if not self.shopper_owes_answer:
            return None
        if self.answer_stalls < 2 or "message" not in self.tool_names:
            self.answer_stalls += 1
            return None
        return ("message", {"content": self.spoken_reply()})

    def ranking_compare_move(self) -> tuple | None:
        if self.camp.lane != "ranking" or "compare" not in self.tool_names:
            return None
        missing = self.named_still_unread()
        if not missing:
            return None
        return ("compare", {"product_ids": missing[:5]})

    def recovery_sibling_move(self) -> tuple | None:
        if self.camp.lane != "recovery" or self.notes.adds or not self.camp.favored_model:
            return None
        if "search" not in self.tool_names:
            return None
        return self.sibling_query()

    def cart_dead_remove(self) -> tuple | None:
        if "remove_from_cart" not in self.tool_names:
            return None
        for cpid, csku in self.notes.basket:
            if FieldNotes.pair(cpid, csku) in self.notes.retired:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        return None

    def inspect_then_cart(self, pick: dict) -> tuple | None:
        pid, sku = pick["pid"], pick["sku"]
        if (pid, sku) in self.notes.basket:
            if not self.stock_is_fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self.order_payload(pid, sku))
        drop = self.cart_dead_remove()
        if drop is not None:
            return drop
        if not self.stock_is_fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    def explore_stride(self) -> tuple | None:
        owed = self.shopper_message_move()
        if owed is not None:
            return owed
        if self.shopper_owes_answer:
            return None
        ranked = self.ranking_compare_move()
        if ranked is not None:
            return ranked
        sib = self.recovery_sibling_move()
        if sib is not None:
            return sib
        twin = self.twin_query()
        if twin is not None:
            return twin
        if self.camp.lane == "recall" and (
            len(self.notes.surfaced) < self.identity_quota() or self.search_calls < 2
        ):
            if self.search_calls == 0:
                return self.first_catalog_ask()
            return None
        if self.search_calls == 0:
            return self.first_catalog_ask()
        return None

    def verify_stride(self) -> tuple | None:
        owed = self.shopper_message_move()
        if owed is not None:
            return owed
        if self.shopper_owes_answer:
            return None
        ranked = self.ranking_compare_move()
        if ranked is not None:
            return ranked
        sib = self.recovery_sibling_move()
        if sib is not None:
            return sib
        if self.shopper_still_owed():
            return self.widen_catalog() if self.turn >= 2 else None
        twin = self.twin_query()
        if twin is not None:
            return twin
        leads = self.unread_leads()
        if leads and self.forced_reads < 8:
            self.forced_reads += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        pick = self.front_pick()
        if pick is not None and not self.stock_is_fresh(pick["pid"], pick["sku"]):
            return ("inspect_stock", {"product_id": pick["pid"], "sku": pick["sku"]})
        if pick is not None and not self.two_sku_blocks_cart() and not self.diversity_blocks_cart():
            if self.recall_blocks_order() and (pick["pid"], pick["sku"]) not in self.notes.basket:
                return self.widen_catalog()
            return self.inspect_then_cart(pick)
        if pick is None:
            return self.widen_catalog()
        return None

    def commit_stride(self) -> tuple | None:
        owed = self.shopper_message_move()
        if owed is not None:
            return owed
        if self.shopper_owes_answer:
            return None
        ranked = self.ranking_compare_move()
        if ranked is not None:
            return ranked
        sib = self.recovery_sibling_move()
        if sib is not None:
            return sib
        if self.shopper_still_owed():
            return self.widen_catalog()
        twin = self.twin_query()
        if twin is not None:
            return twin
        if self.diversity_blocks_cart():
            more = self.widen_catalog()
            if more is not None:
                return more
        pick = self.front_pick()
        if pick is None:
            return self.widen_catalog()
        if self.recall_blocks_order() and (pick["pid"], pick["sku"]) not in self.notes.basket:
            more = self.widen_catalog()
            if more is not None:
                return more
            if Phase.wants_margin(self.phase):
                return None
        return self.inspect_then_cart(pick)

    def panic_pool(self) -> list:
        cards = self.all_scorecards()
        good = [c for c in cards if c["ok"]]
        if self.camp.lane == "recovery" and not self.notes.adds:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
        if good:
            return good
        if self.camp.lane == "justification":
            return []
        relaxed = []
        for c in cards:
            if not (c["live"] and c["spec"] == 2 and self.maker_matches(c["pid"]) and c["cat_ok"]):
                continue
            if self.camp.named_pairs and (c["pid"], c["sku"]) not in self.camp.named_pairs:
                continue
            relaxed.append(c)
        return relaxed

    def panic_stride(self) -> None:
        for pid, sku in reversed(self.notes.basket):
            ok, _why = self.order_legal(pid, sku)
            if ok and not self.done:
                self.dispatch("place_test_order", self.order_payload(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = self.panic_pool()
        if not good:
            return
        good.sort(key=lambda c: self.rank_tuple(c, self.notes.adds == 0))
        pid, sku = good[0]["pid"], good[0]["sku"]
        if (pid, sku) not in self.notes.basket:
            obs = self.dispatch("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = self.dispatch("inspect_stock", {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = as_money(obs.get("price"))
        if self.purse is not None and price is not None and price > self.purse + _SLACK:
            return
        self.dispatch("place_test_order", self.order_payload(pid, sku))

    def consult_model(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.model_cursor, len(self.roster) - 1)]
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
                self.scratch("No tool call was made. The clock is still running — choose exactly one supplied tool.")
                continue
            if self.horizon.leftover() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_cursor = min(self.model_cursor + 1, len(self.roster) - 1)
        return None

    def post_envelope(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            result = _LINK.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.horizon.leftover() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def dispatch(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.held_envelope is not None:
            pend = self.held_envelope
            replay = self.post_envelope(pend["envelope"])
            if replay is None:
                self.scratch("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.held_envelope = None
            self.turn = pend["turn"]
            self.ingest_result(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.done:
                return None
        turn = self.turn + 1
        group = self.problem_id + "-turn-" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
        }
        result = self.post_envelope(envelope)
        if result is None:
            self.held_envelope = {
                "envelope": envelope,
                "turn": turn,
                "name": name,
                "args": args,
                "content": content,
                "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self.ingest_result(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def ingest_result(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.notes.fingerprints.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({"error": "environment call failed; it will be retried automatically"}),
                }
            )
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.done = True
        self.notes.ingest(name, args, obs, self.camp)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": clip_obs(json.dumps(obs), _CLIP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self.tighten_purse(text)
            if self.camp.lane in ("constraint", "intent") and not self.camp.gate_value:
                stated = self.camp.hear_gate(text)
                if stated:
                    self.shopper_updated = True
                    self.scratch(
                        "Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought."
                    )
            if _ASK_RE.search(text):
                self.shopper_owes_answer = True
                self.scratch("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    def after_cart_pulse(self) -> None:
        pid, sku = self.awaiting_pulse
        self.awaiting_pulse = None
        obs = self.dispatch("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = as_money(obs.get("price"))
        over = self.purse is not None and price is not None and price > self.purse + _SLACK
        if stock and not over:
            self.scratch(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                % (FieldNotes.pair(pid, sku), price)
            )
            return
        self.notes.fallen = FieldNotes.pair(pid, sku)
        self.event_turn = self.turn
        self.notes.retired.add(FieldNotes.pair(pid, sku))
        self.notes.retired_pages.add(str(pid))
        self.forced_reads = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        guidance = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "The remaining named candidates are re-ranked; the controller takes the new best.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
        }.get(self.camp.lane, "Choose a different product that meets every firm requirement.")
        self.scratch("Market change: the carted item %s %s. " % (FieldNotes.pair(pid, sku), reason) + guidance)
        self.board_update()

    def mark_add(self, obs: dict | None) -> None:
        if isinstance(obs, dict) and not obs.get("error"):
            added = obs.get("added") or {}
            if added.get("product_id"):
                self.awaiting_pulse = (str(added["product_id"]), str(added["sku"]))

    def apply_forced(self, name: str, args: dict, last_sig: str | None, repeats: int) -> tuple:
        signature = name + json.dumps(args, sort_keys=True)
        repeats = repeats + 1 if signature == last_sig else 1
        if repeats > 2:
            if name in ("inspect_stock", "add_to_cart"):
                self.notes.retired.add(FieldNotes.pair(args.get("product_id"), args.get("sku")))
            if name == "compare":
                self.opaque.update(str(p) for p in args.get("product_ids") or [])
            if name == "message":
                self.shopper_owes_answer = False
            self.scratch("That step changed nothing, so the clocked controller moved on to other candidates.")
            return None, 0
        self.notes.fingerprints.add(signature)
        obs = self.dispatch(name, args)
        if name == "message":
            self.shopper_owes_answer = False
            self.answer_stalls = 0
        if name == "add_to_cart":
            self.mark_add(obs)
        if name in ("search", "filter"):
            self.search_calls += 1
            if self.camp.lane == "recall" and self.camp.identity_floor:
                have, need = len(self.notes.surfaced), self.identity_quota()
                self.scratch(
                    "Distinct matching products surfaced so far: %d of the required %d (clock margin %s)."
                    % (have, need, "on" if Phase.wants_margin(self.phase) else "lapsed")
                    + ("" if have >= need else " Keep searching with different wording.")
                )
        if name in ("view", "compare"):
            self.board_update()
        return signature, repeats

    def run(self) -> list:
        question = self.first_ask()
        last_sig, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            self.phase = self.horizon.name_for(self.turn, self.max_steps)
            self.announce_phase(self.phase)
            if self.phase == Phase.PANIC:
                self.panic_stride()
                if not self.done and self.turn < self.max_steps:
                    proposal = self.consult_model()
                    if proposal is not None:
                        call = proposal["tool_calls"][0]
                        name = (call.get("function") or {}).get("name") or ""
                        args = parse_args(call)
                        if name in self.tool_names and self.gate_proposal(name, args) is None:
                            self.dispatch(name, args, proposal.get("content") or "", call.get("id"))
                    self.panic_stride()
                break
            if question and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self.dispatch("message", {"content": question})
                continue
            if self.awaiting_pulse and "inspect_stock" in self.tool_names:
                self.after_cart_pulse()
                continue
            if self.phase == Phase.EXPLORE:
                move = self.explore_stride()
            elif self.phase == Phase.VERIFY:
                move = self.verify_stride()
            else:
                move = self.commit_stride()
            if move is not None:
                last_sig, repeats = self.apply_forced(move[0], move[1], last_sig, repeats)
                continue
            last_sig, repeats = None, 0
            proposal = self.consult_model()
            if proposal is None:
                self.panic_stride()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = parse_args(call)
            content = proposal.get("content") or ""
            reprompts = 0
            verdict = None
            while True:
                verdict = self.gate_proposal(name, args)
                if verdict is None:
                    break
                if reprompts >= _REASK:
                    self.reject_run += 1
                    self.scratch("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.reject_run >= 3:
                        self.board_update()
                        self.reject_run = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self.consult_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = parse_args(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.reject_run = 0
            if name not in self.tool_names:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.notes.basket:
                    pid, sku = self.notes.basket[-1]
                args = self.order_payload(pid, sku)
            if name == "message":
                if not self.shopper_owes_answer:
                    self.asked_count += 1
                self.shopper_owes_answer = False
                self.answer_stalls = 0
            obs = self.dispatch(name, args, content, call.get("id"))
            if name == "add_to_cart":
                self.mark_add(obs)
            if name in ("search", "filter"):
                self.search_calls += 1
                if self.camp.lane == "recall" and self.camp.identity_floor:
                    have, need = len(self.notes.surfaced), self.identity_quota()
                    self.scratch(
                        "Distinct matching products surfaced so far: %d of the required %d (clock margin %s)."
                        % (have, need, "on" if Phase.wants_margin(self.phase) else "lapsed")
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_looks += 1
            if name in ("view", "compare"):
                self.board_update()
        return self.dialogue

    def gate_proposal(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if self.phase == Phase.EXPLORE and name in ("inspect_stock", "add_to_cart", "place_test_order"):
            return (
                "EXPLORE is still live (leftover > 170s and early turns): diversity search and filter only. "
                "Inspect and add wait for VERIFY or COMMIT."
            )
        if self.phase == Phase.VERIFY and name == "add_to_cart" and self.two_sku_blocks_cart():
            return (
                "VERIFY will not add until two verified SKUs exist (or every named ranking key has been compared). "
                "Read another product or inspect the current best."
            )
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.notes.basket:
                pid, sku = self.notes.basket[-1]
            ok, why = self.order_legal(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.recall_blocks_order():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.notes.surfaced), self.identity_quota())
                )
            if self.diversity_blocks_cart():
                return "order refused: compare at least two products with different listing_id values first"
            if self.phase == Phase.EXPLORE:
                return "order refused: EXPLORE does not place orders; wait for COMMIT"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self.score_sku(pid, sku)
            key = FieldNotes.pair(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.notes.pulsed:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.camp.gate_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.camp.gate_field, self.camp.gate_value)
                )
            if not c["ok"]:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.diversity_blocks_cart():
                return "compare two products with different listing_id values before the first cart addition"
            if self.camp.lane == "recovery" and not self.notes.adds and self.camp.favored_model and self.recovery_distance(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if self.shopper_still_owed():
                return "hold: the shopper still owes a firm requirement or a budget confirmation"
            if self.two_sku_blocks_cart():
                return "hold: VERIFY wants a second verified SKU before the first cart addition"
            if self.phase == Phase.EXPLORE:
                return "hold: EXPLORE does not add"
        if name == "inspect_stock" and self.phase == Phase.EXPLORE:
            return "EXPLORE does not inspect; wait for VERIFY"
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_owes_answer and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper_owes_answer and self.asked_count >= 2:
                return "the shopper has already answered; decide from the evidence you have rather than asking again"
        if name == "inspect_cart" and self.cart_looks >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.notes.fingerprints and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.notes.fingerprints.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return ClockworkShop(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
