"""ORO Bench shopping agent — Scout / Closer finite-state shop.

The episode is split into two roles that never share a toolbox. A Mandate
parses the public query (budget, currency, candidate keys, recall theme,
surface-N, preferred model/color, category, brand, pref brand, cat qualifier,
specs, sealed hidden spec, family). A CatalogPad holds rows, viewed pages,
stock ticks, identities, cart, dead keys and listing ids. A Lens class — not
a pile of module functions — reads specs the way the runtime grades them:
labelled facts first, then the selected variant's option text, then the title.
Conflicting evidence resolves to none. RAM counts only with an adjacent
ram/memory word or an N+M pair; never a bare GB figure.

run() is a state machine (ASK → SCOUT → CLOSE → RECOVER → ORDER), not a
while-plan loop. Scout may only search, filter, view, compare and message.
Closer may only inspect_stock, add_to_cart, remove_from_cart and
place_test_order. The closer commits when the pad holds two fully verified
SKUs (a primary plus a shock spare), or when the clock/turn panic fires
(remaining < 25s or turns left <= 3). A dead cart item snaps the machine to
RECOVER, which searches the dead title for twins, rebuilds a spare, then
hands off to ORDER.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient


_ROSTER = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_WIRE = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_SPAN_S = 270.0
_OBS_CAP = 7000
_REPROMPT_CAP = 2
_SLACK = 1e-9

_SCOUT_KIT = frozenset({"search", "filter", "view", "compare", "message"})
_CLOSER_KIT = frozenset({"inspect_stock", "add_to_cart", "remove_from_cart", "place_test_order"})

_AMOUNT = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RX = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _AMOUNT
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_KEY_RX = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RX = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RX = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RX = re.compile(r"surface at least (\d+) distinct", re.I)
_PREF_ITEM_RX = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_PREF_BRAND_RX = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
)
_CAT_PHRASE_RX = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SPEC_WORD_RX = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_REASON_RX = re.compile(r"\b(?:why|reason|explain|justif)", re.I)
_QUESTION_RX = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
_LANE_MARKS = (
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("justification", "trustworthy reason"),
    ("intent", "nice-to-haves"),
)

_LANE_SCRIPTS = {
    "intent": (
        "Scout the shelf with filter(category, brand, max_price, in_stock=true), then view or "
        "compare so labelled facts sit on the pad. Category, the named hardware figure, budget "
        "and stock are walls. A named brand is a soft pull: take it when a fully legal item of "
        "that brand exists, otherwise any legal brand. Answer the shopper with message. A later "
        "budget cut binds every later pick."
    ),
    "recall": (
        "Issue several differently worded searches (k=10, in_stock=true) from the requested "
        "words and close synonyms until enough distinct matching titles have appeared. Every "
        "requested word must sit as a whole word of the title. View the strongest hits so the "
        "closer can verify a pair and buy."
    ),
    "constraint": (
        "The shopper keeps one firm hardware gate hidden until asked; obey that answer exactly "
        "for the specific variant, reading the variant's own option text before any product-wide "
        "figure. Filter the category in-budget and in-stock, then view. If the carted variant "
        "dies, move to a different product that still clears every gate."
    ),
    "preference": (
        "You already asked which preference ranks first; use that answer. Search or filter the "
        "category inside budget, then compare items with different listing_id values, reading "
        "raw facts. The first cart line is graded, so make it the best genuine match. If it is "
        "repriced above budget, recover to a different listing_id."
    ),
    "ranking": (
        "Only the named candidate variants count. Compare all of them first so their facts sit "
        "on the pad. Highest advertised maximum refresh rate wins, then the largest screen, then "
        "the lowest price, reading the selected variant's own value before a product-wide one. "
        "After the market moves, the remaining named candidates are re-ranked."
    ),
    "recovery": (
        "Find the preferred model and color first: search the model code, then view it so every "
        "variant and color is on the pad. The first cart line must be exactly that model and "
        "color. If it vanishes, relax as little as possible — the same model in another color "
        "before any other model — and never relax category, budget or stock."
    ),
    "justification": (
        "Find a product whose own labelled facts state the requested specification, in the "
        "requested category and brand, in stock and inside budget. View or compare it so those "
        "facts are on the pad. Claims on the order may quote only that observed evidence."
    ),
    "generic": (
        "Read the firm walls, filter or search the catalog, view candidates, and let the closer "
        "verify a primary plus a spare before buying."
    ),
}

_DESK_BRIEF = (
    "You are the scout on a two-desk shopping floor. One tool call per turn, at most 30 turns. "
    "You may search, filter, view, compare or message — never touch the cart or the order; a "
    "separate closer desk inspects stock, carts a verified pair and places the test order. "
    "Surface candidates and get their raw facts, variant options and prices onto the pad. "
    "Never add a placeholder cart line. Prices and stock move while you shop. Answer every "
    "direct shopper question with one short message before anything else. Never send progress "
    "or status chatter. Never ask the shopper to raise the budget. Do not repeat a tool call "
    "with identical arguments; one compare of up to five products beats several view calls."
)


class Ink:
    """Tiny text/number helpers used by the desks and the pad."""

    @staticmethod
    def as_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def clip(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 40] + ' ... (truncated by agent)"}'

    @staticmethod
    def unpack(tool_call: dict) -> dict:
        raw = (tool_call.get("function") or {}).get("arguments", "{}")
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, ValueError):
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def bag(text: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def noun_eq(a: str, b: str) -> bool:
        x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
        y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
        return bool(x) and x == y


class Hourglass:
    """Wall-clock budget for one episode."""

    def __init__(self, span: float) -> None:
        self.t0 = time.monotonic()
        self.span = span

    def left(self) -> float:
        return self.span - (time.monotonic() - self.t0)


class Mandate:
    """Public query parsed into the gates the closer will enforce."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self.pull_budget(self.goal)
        self.currency = "PHP"
        iso = re.search(r"\b([A-Z]{3})\b", self.goal)
        if iso and iso.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = iso.group(1)
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RX.findall(self.goal)]
        phrase = _PHRASE_RX.search(self.goal)
        quoted = _QUOTED_RX.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        surfaced = _SURFACE_RX.search(self.rules)
        self.need_ids = int(surfaced.group(1)) if surfaced else 0
        pref = _PREF_ITEM_RX.search(self.goal)
        self.pref_color = pref.group(1).lower() if pref else None
        self.pref_model = pref.group(2) if pref else None
        self.category = self._pull_category(self.goal)
        self.brand = self._pull_brand(self.goal, self.category)
        self.pref_brand = None if pref else self._pull_pref_brand(self.goal)
        self.cat_qualifier = self._pull_qualifier(self.goal)
        self.specs = self._pull_specs(self.goal)
        self.spec_field, self.spec_value = self._first_gate(self.goal)
        self.spec_sealed = self.spec_field is None and "firm requirement" in self.goal.lower()
        self.lane = self._classify()

    def _classify(self) -> str:
        if self.candidate_keys:
            return "ranking"
        if self.need_ids:
            return "recall"
        if self.pref_model:
            return "recovery"
        for name, mark in _LANE_MARKS:
            if mark in self.rules:
                return name
        if self.theme:
            return "recall"
        if _REASON_RX.search(self.goal):
            return "justification"
        if self.pref_brand:
            return "intent"
        if self.spec_sealed or self.spec_field or self.brand:
            return "constraint"
        return "preference"

    @staticmethod
    def pull_budget(text: str) -> float | None:
        best = None
        for m in _BUDGET_RX.finditer(text or ""):
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
    def _pull_category(goal: str) -> str | None:
        m = _CAT_PHRASE_RX.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def _pull_brand(goal: str, category: str | None) -> str | None:
        if not category:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RX.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not Lens.res_pool(w)]
        return " ".join(keep) or None

    @staticmethod
    def _pull_pref_brand(goal: str) -> str | None:
        m = _PREF_BRAND_RX.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def _pull_qualifier(goal: str) -> str | None:
        if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
            return "over ear"
        return None

    @staticmethod
    def _pull_specs(goal: str) -> dict:
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = Lens.requested_value(field, goal)
            if value:
                out[field] = value
        return out

    def _first_gate(self, goal: str) -> tuple:
        for field, value in self.specs.items():
            return field, value
        return None, None

    def adopt_spec(self, text: str) -> str | None:
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = Lens.requested_value(field, text)
            if value:
                self.spec_field, self.spec_value = field, value
                self.specs[field] = value
                self.spec_sealed = False
                return field + " = " + value
        return None


class CatalogPad:
    """Everything the floor has seen, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.ticks: dict = {}
        self.tick_at: dict = {}
        self.inspected: set = set()
        self.pages: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.listing_ids: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.tick = 0
        self.first_add_tick: int | None = None
        self.last_add_tick: int | None = None
        self.add_count = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.seen_calls: set = set()

    @staticmethod
    def stamp(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def ingest(self, name: str, args: dict, obs: dict, mandate: Mandate) -> None:
        self.tick += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        sink = {
            "search": self._ingest_listing,
            "filter": self._ingest_listing,
            "view": self._ingest_page,
            "compare": self._ingest_compare,
            "inspect_stock": self._ingest_stock,
            "add_to_cart": self._ingest_add,
            "remove_from_cart": self._ingest_remove,
            "inspect_cart": self._ingest_basket,
        }.get(name)
        if sink:
            sink(obs, mandate)

    def _ingest_listing(self, obs: dict, mandate: Mandate) -> None:
        for row in obs.get("results") or []:
            if isinstance(row, dict):
                self._keep_row(row, mandate)

    def _ingest_page(self, obs: dict, mandate: Mandate) -> None:
        self._keep_page(obs, mandate)

    def _ingest_compare(self, obs: dict, mandate: Mandate) -> None:
        for prod in obs.get("compared") or []:
            if isinstance(prod, dict):
                self._keep_page(prod, mandate)

    def _ingest_stock(self, obs: dict, mandate: Mandate) -> None:
        k = self.stamp(obs.get("product_id"), obs.get("sku"))
        self.ticks[k] = {"in_stock": bool(obs.get("in_stock")), "price": Ink.as_float(obs.get("price"))}
        self.tick_at[k] = self.tick
        self.inspected.add(k)

    def _ingest_add(self, obs: dict, mandate: Mandate) -> None:
        added = obs.get("added") or {}
        pid, sku = added.get("product_id"), added.get("sku")
        if pid and sku:
            self.add_count += 1
            if (pid, sku) not in self.cart:
                self.cart.append((pid, sku))
            if self.first_add_tick is None:
                self.first_add_tick = self.tick
            self.last_add_tick = self.tick
        self._align_cart(obs)

    def _ingest_remove(self, obs: dict, mandate: Mandate) -> None:
        self._align_cart(obs)
        removed = obs.get("removed") or {}
        pair = (removed.get("product_id"), removed.get("sku"))
        if pair in self.cart:
            self.cart.remove(pair)

    def _ingest_basket(self, obs: dict, mandate: Mandate) -> None:
        self._align_cart(obs)

    def _align_cart(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.stamp(r.get("product_id"), r.get("sku"))
            price = Ink.as_float(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.ticks.get(k, {})
            self.ticks[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.tick_at[k] = self.tick

    def _keep_row(self, row: dict, mandate: Mandate) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if mandate.theme:
            if all(t in Ink.bag(title) for t in mandate.theme):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        k = self.stamp(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.ticks.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": Ink.as_float(row.get("price"))})

    def _keep_page(self, prod: dict, mandate: Mandate) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if mandate.theme and all(t in Ink.bag(str(prod.get("title") or "")) for t in mandate.theme):
            self.identities.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.stamp(pid, v.get("sku"))
            self.ticks[k] = {"in_stock": bool(v.get("in_stock")), "price": Ink.as_float(v.get("price"))}
            self.tick_at[k] = self.tick
            self.options[k] = str(v.get("options") or "")

    def latest(self, pid: str, sku: str) -> dict | None:
        return self.ticks.get(self.stamp(pid, sku))


class Lens:
    """Reads public evidence the way the runtime grades a variant.

    Labelled raw facts win first, then the selected variant's option text, then
    the listing title. A number counts only when its unit or concept word sits
    next to it. Evidence that disagrees with itself resolves to nothing. RAM
    never accepts a bare GB figure from title or option text.
    """

    INCH_RX = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
    HZ_RX = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
    WXH_RX = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
    GB_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
    PAIR_RX = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
    PAIR_HINT_RX = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
    MEM_RX = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
    QUARTER_RX = re.compile(r"\bqHD\b")
    LABEL = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}
    OS_RX = re.compile(
        r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|"
        r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
        re.I,
    )
    IS_PHONE_RX = re.compile(r"\bsmart\s*phones?\b", re.I)
    NOT_PHONE_RX = re.compile(
        r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
        r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
        r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
        re.I,
    )
    PART_ONLY_RX = re.compile(
        r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
        r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
        r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
        re.I,
    )
    NEGATED_RX = re.compile(
        r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
        re.I,
    )
    BUNDLE_RX = re.compile(
        r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
        r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
        re.I,
    )

    def __init__(self, pad: CatalogPad, mandate: Mandate) -> None:
        self.pad = pad
        self.mandate = mandate

    @classmethod
    def inch_pool(cls, text: str) -> set:
        return {"%gin" % float(m.group(1)) for m in cls.INCH_RX.finditer(text or "")}

    @classmethod
    def hz_pool(cls, text: str) -> set:
        return {"%ghz" % float(m.group(1)) for m in cls.HZ_RX.finditer(text or "")}

    @classmethod
    def gb_pool(cls, text: str) -> set:
        return {
            "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
            for m in cls.GB_RX.finditer(text or "")
        }

    @classmethod
    def pixel_band(cls, long_side: int, short_side: int) -> str | None:
        if long_side >= 3840 and short_side >= 2160:
            return "4k"
        if long_side >= 2560 and short_side >= 1440:
            return "qhd"
        if long_side >= 1920 and short_side >= 1080:
            return "fhd"
        if long_side >= 1280 and short_side >= 720:
            return "hd"
        return None

    @classmethod
    def res_pool(cls, text: str) -> set:
        text = text or ""
        if cls.QUARTER_RX.search(text):
            return set()
        low = text.lower()
        out = set()
        for m in cls.WXH_RX.finditer(low):
            pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
            band = cls.pixel_band(pair[0], pair[1])
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

    @classmethod
    def pool_for(cls, field: str) -> Callable[[str], set]:
        return {
            "screen_size": cls.inch_pool,
            "refresh_rate": cls.hz_pool,
            "resolution": cls.res_pool,
            "ram": cls.gb_pool,
        }[field]

    @classmethod
    def adjacent_pool(cls, field: str, text: str) -> set:
        text = text or ""
        if field != "ram":
            return cls.pool_for(field)(text)
        paired = {"%ggb" % float(a) for a, _b in cls.PAIR_RX.findall(text)}
        return paired or {"%ggb" % float(n) for n in cls.MEM_RX.findall(text)}

    @classmethod
    def talks_about(cls, field: str, text: str) -> bool:
        text = text or ""
        if field == "ram":
            return bool(cls.PAIR_HINT_RX.search(text) or cls.MEM_RX.search(text))
        if field == "screen_size":
            return bool(cls.INCH_RX.search(text))
        if field == "refresh_rate":
            return bool(cls.HZ_RX.search(text))
        return bool(
            cls.WXH_RX.search(text)
            or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I)
        )

    @staticmethod
    def only_one(values: set) -> str | None:
        return next(iter(values)) if len(values) == 1 else None

    @classmethod
    def requested_value(cls, field: str | None, text: str) -> str | None:
        if not field:
            return None
        return cls.only_one(cls.pool_for(field)(text or ""))

    def page_of(self, pid: str) -> dict:
        return self.pad.pages.get(str(pid)) or {}

    def facts_of(self, pid: str) -> dict:
        blob = self.page_of(pid).get("facts")
        return blob if isinstance(blob, dict) else {}

    def title_of(self, pid: str) -> str:
        title = str(self.page_of(pid).get("title") or "")
        if title:
            return title
        prefix = str(pid) + "::"
        for key, row in self.pad.rows.items():
            if key.startswith(prefix):
                return str(row.get("title") or "")
        return ""

    def brand_name(self, pid: str) -> str:
        brand = self.page_of(pid).get("brand")
        if brand:
            return str(brand)
        prefix = str(pid) + "::"
        for key, row in self.pad.rows.items():
            if key.startswith(prefix) and row.get("brand"):
                return str(row["brand"])
        return ""

    def cat_path(self, pid: str) -> list:
        path = self.page_of(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def option_text(self, pid: str, sku: str) -> str:
        return self.pad.options.get(CatalogPad.stamp(pid, sku), "")

    def drop_model_codes(self, pid: str, title: str) -> str:
        for token in re.split(r"[;,]", str(self.facts_of(pid).get("model") or "")):
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

    def labelled_fact(self, pid: str, field: str) -> str | None:
        raw = self.facts_of(pid).get(self.LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = self.pool_for(field)(text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return self.only_one(found)

    def shelf_value(self, pid: str, field: str) -> str | None:
        vals: set = set()
        conflict = False
        raw = self.facts_of(pid).get(self.LABEL[field])
        if raw:
            found = self.pool_for(field)(str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and self.QUARTER_RX.search(str(raw)):
                conflict = True
            vals |= found
        title = self.title_of(pid)
        if field == "resolution":
            title = self.drop_model_codes(pid, title)
            if self.QUARTER_RX.search(title):
                conflict = True
        vals |= self.adjacent_pool(field, title)
        return None if conflict else self.only_one(vals)

    def sku_value(self, pid: str, sku: str, field: str) -> str | None:
        opt = self.option_text(pid, sku)
        vals = self.adjacent_pool(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if self.talks_about(field, opt):
            return None
        return self.shelf_value(pid, field)

    def measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self.sku_value(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @staticmethod
    def stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def category_fits(self, pid: str) -> bool:
        want = self.mandate.category
        path = self.cat_path(pid)
        if not want or not path:
            return True
        wanted = self.stems(want)
        labels: set = set()
        for label in path:
            labels |= self.stems(str(label))
        placed = bool(wanted & labels) or any(
            self.stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        if not placed and "smartphone" in wanted:
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                for label in path
            )
        if not placed:
            return False
        qualifier = self.mandate.cat_qualifier
        if not qualifier:
            return True
        title = self.title_of(pid)
        words = self.stems(title) | Ink.bag(title)
        if qualifier == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in self.hz_pool(title))
            return "monitor" in words and ("gaming" in words or fast)
        return "over" in words and "ear" in words or "overear" in words or "circumaural" in words

    def brand_fits(self, pid: str) -> bool:
        want = self.mandate.brand
        if not want:
            return True
        wanted = Ink.bag(want)
        return bool(wanted <= Ink.bag(self.brand_name(pid)) or wanted <= Ink.bag(self.title_of(pid)))

    @classmethod
    def identity_clip(cls, text: str, title: bool = False) -> str:
        text = cls.NEGATED_RX.sub("", text or "")
        return cls.BUNDLE_RX.split(text, maxsplit=1)[0] if title else text

    @classmethod
    def phone_verdict(cls, facts: dict, title: str, options: str) -> str:
        if cls.PART_ONLY_RX.search(options or ""):
            return "no"
        facts = facts if isinstance(facts, dict) else {}
        raw_title = str(title or "")
        head = cls.identity_clip(raw_title, title=True)
        kind = " ".join(cls.identity_clip(str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
        blob = head + " " + kind
        if cls.NOT_PHONE_RX.search(blob):
            return "no"
        if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
            return "unclear"
        if cls.IS_PHONE_RX.search(blob):
            return "yes"
        os_blob = " ".join(
            cls.identity_clip(str(facts.get(k) or ""))
            for k in ("device_operating_system", "description_operating_system")
            if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
        )
        if cls.OS_RX.search(os_blob):
            return "yes"
        if cls.OS_RX.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
            return "yes"
        return "unclear"

    def device_fits(self, pid: str, sku: str) -> bool:
        if not Ink.noun_eq(self.mandate.category or "", "smartphone"):
            return True
        verdict = self.phone_verdict(self.facts_of(pid), self.title_of(pid), self.option_text(pid, sku))
        if self.mandate.lane in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def resale_listing(self, pid: str) -> bool:
        return bool(
            re.search(
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
                self.title_of(pid),
                re.I,
            )
        )

    def relax_distance(self, pid: str, sku: str) -> int:
        model, color = self.mandate.pref_model, self.mandate.pref_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self.title_of(pid) + " " + " ".join(str(v) for v in self.facts_of(pid).values()))
        same_model = squash(model) in blob
        shade = (self.option_text(pid, sku) + " " + str(self.facts_of(pid).get("color") or "") + " " + self.title_of(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def score_sku(self, pid: str, sku: str, ceiling: float | None) -> dict:
        lane = self.mandate.lane
        key = CatalogPad.stamp(pid, sku)
        st = self.pad.ticks.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.pad.dead_keys or (str(pid) in self.pad.dead_products and lane != "recovery")
        live = bool(stock) and price is not None and (ceiling is None or price <= ceiling + _SLACK) and not dead
        viewed = str(pid) in self.pad.pages
        field, want = self.mandate.spec_field, self.mandate.spec_value
        spec = 2
        if field and want:
            got = self.sku_value(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self.brand_fits(pid) and self.device_fits(pid, sku))
        cat_ok = self.category_fits(pid)
        ok = ok and cat_ok
        if lane == "justification":
            ok = ok and not self.resale_listing(pid) and bool(self.brand_name(pid))
            if field and want:
                ok = ok and self.labelled_fact(pid, field) == want and self.shelf_value(pid, field) == want
        if lane == "preference" and self.pad.event_target:
            dead_listing = self.pad.listing_of.get(self.pad.event_target.split("::", 1)[0])
            if dead_listing and self.pad.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if lane == "recall" and self.mandate.theme:
            ok = ok and all(t in Ink.bag(self.title_of(pid)) for t in self.mandate.theme)
        if lane == "ranking":
            ok = ok and (str(pid), str(sku)) in self.mandate.candidate_keys
            ok = ok and self.measure(pid, sku, "refresh_rate") is not None and self.measure(pid, sku, "screen_size") is not None
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
            "relax": self.relax_distance(pid, sku) if lane == "recovery" else 0,
        }

    def scoreboard(self, ceiling: float | None) -> list:
        seen = set()
        out = []
        for key in list(self.pad.ticks):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self.score_sku(pid, sku, ceiling))
        return out

    def rank_tuple(self, card: dict, first: bool) -> tuple:
        lane = self.mandate.lane
        price = card["price"] if card["price"] is not None else 1e12
        if lane == "ranking":
            hz = self.measure(card["pid"], card["sku"], "refresh_rate") or 0.0
            inch = self.measure(card["pid"], card["sku"], "screen_size") or 0.0
            return (-hz, -inch, price, card["key"])
        if lane == "recovery":
            return (card["relax"], price, card["key"])
        extra = sum(
            1
            for f, v in self.mandate.specs.items()
            if f != self.mandate.spec_field and self.sku_value(card["pid"], card["sku"], f) != v
        )
        proven = 0 if self.phone_verdict(self.facts_of(card["pid"]), self.title_of(card["pid"]), self.option_text(card["pid"], card["sku"])) == "yes" else 1
        if lane == "intent" and first and self.mandate.pref_brand:
            wanted = Ink.bag(self.mandate.pref_brand)
            near = 0 if (wanted <= Ink.bag(self.brand_name(card["pid"])) or wanted <= Ink.bag(self.title_of(card["pid"]))) else 1
            return (near, extra, proven, price, card["key"])
        return (extra, proven, price, card["key"])

    def is_fresh(self, pid: str, sku: str) -> bool:
        k = CatalogPad.stamp(pid, sku)
        if k not in self.pad.inspected:
            return False
        return self.pad.last_add_tick is None or self.pad.tick_at.get(k, 0) > self.pad.last_add_tick


class ScoutDesk:
    """Exploration desk: search, filter, view, compare, message. Never the cart."""

    def __init__(self, floor: "ShopFloor") -> None:
        self.floor = floor
        self.probes: set = set()
        self.reads_done = False
        self.forced_compares = 0
        self.siblings_queried = False
        self.hinted_exact = False
        self.hinted_brand = False

    def tick(self) -> None:
        floor = self.floor
        if floor.mandate.lane == "ranking" and "compare" in floor.toolset:
            missing = [
                p
                for p, _s in floor.mandate.candidate_keys
                if p not in floor.pad.pages and p not in floor.unreadable
            ]
            if missing:
                floor.enact("compare", {"product_ids": missing[:5]})
                return
        if floor.mandate.lane == "recovery" and not floor.pad.add_count and floor.mandate.pref_model:
            sib = self._sibling_hunt()
            if sib is not None:
                floor.enact(sib[0], sib[1])
                return
        if floor.may_commit() and floor.pick_primary() is not None:
            floor.phase = "CLOSE"
            return
        nxt = self.next_read()
        if nxt is not None:
            floor.enact(nxt[0], nxt[1])
            if nxt[0] in ("search", "filter"):
                floor.search_count += 1
                self._recall_progress()
            return
        self._consult_model()

    def _recall_progress(self) -> None:
        floor = self.floor
        if floor.mandate.lane != "recall" or not floor.mandate.need_ids:
            return
        have, need = len(floor.pad.identities), floor.recall_need()
        floor.whisper(
            "Distinct matching products on the pad: %d of the required %d." % (have, need)
            + ("" if have >= need else " Keep searching with different wording.")
        )

    def _sibling_hunt(self) -> tuple | None:
        if self.siblings_queried:
            return None
        rows = [c for c in self.floor.lens.scoreboard(self.floor.ceiling) if c["live"] and c["viewed"] and c["relax"] == 0]
        if rows:
            return None
        near = [c for c in self.floor.lens.scoreboard(self.floor.ceiling) if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self.floor.lens.title_of(near[0]["pid"])
        if not title:
            return None
        self.siblings_queried = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def next_read(self) -> tuple | None:
        floor = self.floor
        if floor.mandate.lane == "recall" and len(floor.pad.identities) < floor.recall_need():
            return None
        leads = floor.unread_leads()
        if leads and self.forced_compares < 8 and floor.turn >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in floor.toolset:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        probe = (floor.mandate.brand or "", "%.2f" % (floor.ceiling or 0.0), str(floor.mandate.spec_value))
        if floor.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
        if "filter" in floor.toolset and floor.mandate.category:
            args: dict = {"category": floor.mandate.category, "in_stock": True, "k": 10}
            if floor.ceiling is not None:
                args["max_price"] = floor.ceiling
            if floor.mandate.brand:
                args["brand"] = floor.mandate.brand
            return ("filter", args)
        if "search" in floor.toolset:
            words = [w for w in (floor.mandate.brand, floor.mandate.spec_value, floor.mandate.category) if w]
            if words:
                return ("search", {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def nudge_brand(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        brand = self.floor.mandate.pref_brand
        self.floor.whisper(
            "No %s product that clears every firm wall has been read yet. Filter by brand=%s "
            "inside the budget and read those pages before any other." % (brand, brand)
        )

    def nudge_exact(self) -> None:
        if self.hinted_exact or not self.floor.mandate.pref_model:
            return
        self.hinted_exact = True
        model = self.floor.mandate.pref_model
        color = self.floor.mandate.pref_color
        self.floor.whisper(
            "The shopper's exact preference (%s%s) is not on the pad yet. Search the model code "
            "alone and view the hits so every color variant is recorded; only that pairing may be "
            "the first cart line." % (model, ", " + color if color else "")
        )

    def _consult_model(self) -> None:
        floor = self.floor
        proposal = floor.consult_llm()
        if proposal is None:
            if floor.in_panic() or (floor.may_commit() and floor.pick_primary() is not None):
                floor.phase = "CLOSE"
            return
        call = proposal["tool_calls"][0]
        name = (call.get("function") or {}).get("name") or ""
        args = Ink.unpack(call)
        content = proposal.get("content") or ""
        verdict = None
        for _rep in range(_REPROMPT_CAP + 1):
            verdict = floor.veto_scout(name, args)
            if verdict is None:
                break
            if _rep >= _REPROMPT_CAP:
                floor.reject_streak += 1
                floor.whisper("Scout call blocked: " + verdict + ". Pick a different scout action.")
                if floor.reject_streak >= 3:
                    floor.announce_shortlist()
                    floor.reject_streak = 0
                return
            floor.chat.append({"role": "assistant", "content": content, "tool_calls": [call]})
            floor.chat.append(
                {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
            )
            floor.transcript.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
            again = floor.consult_llm()
            if again is None:
                return
            call = again["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = Ink.unpack(call)
            content = again.get("content") or ""
        if verdict is not None:
            return
        floor.reject_streak = 0
        if name not in floor.toolset:
            return
        if name == "message":
            if not floor.shopper_owes_answer:
                floor.questions_asked += 1
            floor.shopper_owes_answer = False
            floor.question_waits = 0
        obs = floor.enact(name, args, content, call.get("id"))
        if name in ("search", "filter"):
            floor.search_count += 1
            self._recall_progress()
        if name == "inspect_cart":
            floor.cart_peeks += 1
        if name in ("view", "compare") and isinstance(obs, dict):
            floor.announce_shortlist()


class CloserDesk:
    """Commit desk: inspect, cart, remove, order. Never search or message."""

    def __init__(self, floor: "ShopFloor") -> None:
        self.floor = floor

    def tick(self) -> None:
        floor = self.floor
        for pid, sku in list(floor.pad.cart):
            if CatalogPad.stamp(pid, sku) in floor.pad.dead_keys and "remove_from_cart" in floor.toolset:
                floor.enact("remove_from_cart", {"product_id": pid, "sku": sku})
                floor.phase = "RECOVER"
                return
        pick = floor.pick_primary()
        if pick is None:
            floor.phase = "ORDER" if floor.in_panic() else "SCOUT"
            return
        pid, sku = pick["pid"], pick["sku"]
        if (pid, sku) in floor.pad.cart:
            if not floor.lens.is_fresh(pid, sku):
                floor.enact("inspect_stock", {"product_id": pid, "sku": sku})
                return
            floor.phase = "ORDER"
            return
        if not floor.lens.is_fresh(pid, sku):
            floor.enact("inspect_stock", {"product_id": pid, "sku": sku})
            return
        obs = floor.enact("add_to_cart", {"product_id": pid, "sku": sku})
        if isinstance(obs, dict) and not obs.get("error"):
            added = obs.get("added") or {}
            if added.get("product_id"):
                floor.awaiting_recheck = (str(added["product_id"]), str(added["sku"]))

    def recheck_carted(self) -> None:
        floor = self.floor
        pid, sku = floor.awaiting_recheck
        floor.awaiting_recheck = None
        obs = floor.enact("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = Ink.as_float(obs.get("price"))
        over = floor.ceiling is not None and price is not None and price > floor.ceiling + _SLACK
        if stock and not over:
            floor.whisper(
                "Post-cart recheck: %s is in stock at %s. Handing the ticket to ORDER."
                % (CatalogPad.stamp(pid, sku), price)
            )
            floor.phase = "ORDER"
            return
        floor.pad.event_target = CatalogPad.stamp(pid, sku)
        floor.event_turn = floor.turn
        floor.pad.dead_keys.add(CatalogPad.stamp(pid, sku))
        floor.pad.dead_products.add(str(pid))
        floor.scout.forced_compares = 0
        floor.twins_queried = False
        floor.recover.extra_probe = False
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        tips = {
            "recovery": "Recover with the smallest change: same model, other color, before any other model.",
            "ranking": "The remaining named candidates are re-ranked.",
            "preference": "Recover to a product with a different listing_id that still fits the priorities.",
            "constraint": "Choose a different product that still meets every requirement.",
        }.get(floor.mandate.lane, "Choose a different product that meets every firm wall.")
        floor.whisper("Market move: carted item %s %s. " % (CatalogPad.stamp(pid, sku), reason) + tips)
        floor.announce_shortlist()
        floor.phase = "RECOVER"

    def rush_checkout(self) -> None:
        floor = self.floor
        for pid, sku in reversed(floor.pad.cart):
            ok, _why = floor.order_ready(pid, sku)
            if ok and not floor.stopped:
                floor.enact("place_test_order", floor.order_payload(pid, sku))
                return
        if floor.stopped or floor.turn > floor.step_cap - 3:
            return
        good = [c for c in floor.lens.scoreboard(floor.ceiling) if c["ok"]]
        if not good and floor.mandate.lane != "justification":
            good = [
                c
                for c in floor.lens.scoreboard(floor.ceiling)
                if c["live"]
                and c["spec"] == 2
                and floor.lens.brand_fits(c["pid"])
                and c["cat_ok"]
                and (not floor.mandate.candidate_keys or (c["pid"], c["sku"]) in floor.mandate.candidate_keys)
            ]
        if not good:
            return
        good.sort(key=lambda c: floor.lens.rank_tuple(c, floor.pad.add_count == 0))
        pid, sku = good[0]["pid"], good[0]["sku"]
        if (pid, sku) not in floor.pad.cart:
            obs = floor.enact("add_to_cart", {"product_id": pid, "sku": sku})
            if floor.stopped or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = floor.enact("inspect_stock", {"product_id": pid, "sku": sku})
        if floor.stopped or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = Ink.as_float(obs.get("price"))
        if floor.ceiling is not None and price is not None and price > floor.ceiling + _SLACK:
            return
        floor.enact("place_test_order", floor.order_payload(pid, sku))


class RecoverDesk:
    """After a dead cart line: hunt title-twins, rebuild a spare, then ORDER."""

    def __init__(self, floor: "ShopFloor") -> None:
        self.floor = floor
        self.extra_probe = False

    def tick(self) -> None:
        floor = self.floor
        if not floor.twins_queried and "search" in floor.toolset:
            dead = floor.pad.event_target
            title = floor.lens.title_of(dead.split("::", 1)[0]) if dead else ""
            floor.twins_queried = True
            if title:
                floor.whisper(
                    "The dead cart line shares a title with sibling listings; searching that "
                    "title surfaces other colors and sizes that still meet the same walls."
                )
                floor.enact("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})
                floor.search_count += 1
                return
        leads = floor.unread_leads()
        if leads:
            if len(leads) > 1 and "compare" in floor.toolset:
                floor.enact("compare", {"product_ids": leads[:5]})
            else:
                floor.enact("view", {"product_id": leads[0]})
            return
        spare = floor.pick_primary()
        if spare:
            floor.phase = "CLOSE"
            return
        if not self.extra_probe:
            self.extra_probe = True
            nxt = floor.scout.next_read()
            if nxt is not None and nxt[0] in ("search", "filter", "view", "compare"):
                floor.enact(nxt[0], nxt[1])
                if nxt[0] in ("search", "filter"):
                    floor.search_count += 1
                return
        floor.phase = "CLOSE" if floor.in_panic() else "SCOUT"


class ShopFloor:
    """Finite-state shop: ASK → SCOUT → CLOSE → RECOVER → ORDER."""

    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.toolset = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.step_cap = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.mandate = Mandate(str(self.policy.get("query") or ""))
        self.ceiling = self.mandate.budget
        self.pad = CatalogPad()
        self.lens = Lens(self.pad, self.mandate)
        self.hourglass = Hourglass(_SPAN_S)
        self.scout = ScoutDesk(self)
        self.closer = CloserDesk(self)
        self.recover = RecoverDesk(self)
        self.transcript: list = []
        self.chat: list = [
            {"role": "system", "content": self._briefing()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.phase = "ASK"
        self.asked_once = False
        self.shopper_owes_answer = False
        self.shopper_mutated = False
        self.reject_streak = 0
        self.pending_envelope: dict | None = None
        self.search_count = 0
        self.cart_peeks = 0
        self.awaiting_recheck: tuple | None = None
        self.questions_asked = 0
        self.idle = 0
        self.wish_cache: set | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.stopped = False
        self.twins_queried = False
        self.event_turn: int | None = None
        self.last_forced: str | None = None
        self.repeats = 0
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _ROSTER.get(provider) or _ROSTER["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _briefing(self) -> str:
        script = _LANE_SCRIPTS.get(self.mandate.lane, _LANE_SCRIPTS["generic"])
        extra = ""
        if self.mandate.lane == "recall" and self.mandate.need_ids:
            extra = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the floor reports the count after each search."
                % self.mandate.need_ids
            )
        return _DESK_BRIEF + "\n\nLane script: " + script + extra

    def opening_line(self) -> str | None:
        if self.mandate.lane == "constraint" or self.mandate.spec_sealed:
            return (
                "Before I open the catalog: which hardware specification is the firm requirement "
                "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
            )
        if self.mandate.lane == "preference":
            return (
                "One question so I rank well: which of your preferences counts most, and in what "
                "order should I weigh the others? Tell me your priority."
            )
        return None

    def whisper(self, text: str) -> None:
        self.chat.append({"role": "user", "content": "[note] " + text})
        self.transcript.append({"role": "assistant", "content": "[note] " + text[:300]})

    def in_panic(self) -> bool:
        return self.hourglass.left() < 25 or self.turn >= self.step_cap - 3

    def recall_need(self) -> int:
        early = self.search_count < 8 and self.turn < self.step_cap * 0.5
        return self.mandate.need_ids + (2 if early else 0)

    def may_commit(self) -> bool:
        if self.in_panic():
            return True
        if self.mandate.spec_sealed:
            if self.turn < 8:
                return False
            self.mandate.spec_sealed = False
            self.whisper("The firm requirement was never stated, so the remaining walls decide.")
        if self.mandate.lane in ("intent", "constraint") and self.turn < 4 and not self.shopper_mutated:
            return False
        if self.mandate.lane == "preference" and len(self.pad.listing_ids) < 2:
            return False
        roomy = self.turn < self.step_cap * 0.45 and self.hourglass.left() > 140
        if self.mandate.lane == "recall" and roomy:
            if len(self.pad.identities) < self.recall_need() or self.search_count < 2:
                return False
        if roomy and not self.scout.reads_done and len({c["pid"] for c in self.lens.scoreboard(self.ceiling) if c["ok"]}) < 2:
            return False
        return True

    def pick_primary(self) -> dict | None:
        first = self.pad.add_count == 0
        lane = self.mandate.lane
        good = [c for c in self.lens.scoreboard(self.ceiling) if c["ok"]]
        if lane == "preference":
            carted = [c for c in good if (c["pid"], c["sku"]) in self.pad.cart]
            if carted:
                return carted[0]
            if good:
                good.sort(
                    key=lambda c: (
                        -self.wish_hits(c["pid"], c["sku"]),
                        c["price"] if c["price"] is not None else 1e12,
                    )
                )
                return good[0]
            return None
        if lane == "intent" and first and self.mandate.pref_brand:
            wanted = Ink.bag(self.mandate.pref_brand)
            named = [
                c
                for c in good
                if wanted <= Ink.bag(self.lens.brand_name(c["pid"])) or wanted <= Ink.bag(self.lens.title_of(c["pid"]))
            ]
            if named:
                good = named
            elif self.turn < self.step_cap * 0.6 and not self.in_panic():
                self.scout.nudge_brand()
                return None
        if lane == "recovery" and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.step_cap * 0.6 and not self.in_panic():
                self.scout.nudge_exact()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self.lens.rank_tuple(c, first))
        return good[0]

    def wish_hits(self, pid: str, sku: str) -> int:
        wants = self._wish_words()
        if not wants:
            return 0
        have = Ink.bag(
            " ".join(str(v) for v in self.lens.facts_of(pid).values())
            + " "
            + self.lens.title_of(pid)
            + " "
            + self.lens.option_text(pid, sku)
        )
        return len(wants & have)

    def _wish_words(self) -> set:
        if self.wish_cache is None:
            said = self.mandate.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.chat if m.get("role") == "user"
            )
            drop = Ink.bag(self.mandate.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self.wish_cache = {w for w in Ink.bag(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self.wish_cache

    def unread_leads(self) -> list:
        leads = []
        for key, row in self.pad.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.pad.pages or str(pid) in self.pad.dead_products:
                continue
            st = self.pad.ticks.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.ceiling is not None and price > self.ceiling + _SLACK):
                continue
            title = str(row.get("title") or "")
            if self.mandate.theme and not all(t in Ink.bag(title) for t in self.mandate.theme):
                continue
            if self.mandate.pref_model and self.pad.add_count == 0 and not self.pad.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.mandate.pref_model.lower()) not in squashed:
                    continue
            if self.mandate.brand:
                wanted = Ink.bag(self.mandate.brand)
                if not (wanted <= Ink.bag(str(row.get("brand") or "")) or wanted <= Ink.bag(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def announce_shortlist(self) -> None:
        cards = self.lens.scoreboard(self.ceiling)
        if self.mandate.lane == "preference" and not self.pad.cart:
            good = [c for c in cards if c["ok"]]
            good.sort(
                key=lambda c: (
                    -self.wish_hits(c["pid"], c["sku"]),
                    c["price"] if c["price"] is not None else 1e12,
                )
            )
            if good:
                self.whisper(
                    "Products that clear every firm wall so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". The closer will take the best fit once a spare is also verified."
                )
                return
        best = self.pick_primary()
        if best:
            self.whisper(
                "Verified candidate: %s::%s at %.2f. The closer inspects, carts and orders; "
                "read others only if one could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        field, want = self.mandate.spec_field, self.mandate.spec_value
        blocked = [c for c in cards if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self.whisper(
                "None of the pages read so far can be verified. Evidence for %s must resolve to "
                "exactly %s from the variant option text or raw facts: %s. Look at different products."
                % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self.whisper(
                "Pages read so far fail a firm wall (category, brand, stock, budget or device kind). "
                "Search for different products."
            )

    def _apply_budget_cut(self, text: str) -> None:
        new = Mandate.pull_budget(text)
        if new is None or (self.ceiling is not None and new >= self.ceiling):
            return
        self.ceiling = new
        self.shopper_mutated = True
        self.whisper(
            "Budget updated by the shopper: at most %.2f %s. Every later pick must fit it."
            % (new, self.mandate.currency)
        )
        for pid, sku in list(self.pad.cart):
            st = self.pad.latest(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _SLACK:
                self.pad.dead_keys.add(CatalogPad.stamp(pid, sku))
                self.whisper(
                    "Carted item %s costs %.2f, above the new budget; the closer will replace it."
                    % (CatalogPad.stamp(pid, sku), price)
                )

    def _claims_for(self, pid: str, sku: str) -> list:
        out = []
        field, want = self.mandate.spec_field, self.mandate.spec_value
        if not field and self.mandate.lane == "justification":
            field, want = "resolution", self.lens.labelled_fact(pid, "resolution")
        if not field or not want:
            return []
        stated = dict(self.mandate.specs) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self.lens.labelled_fact(pid, name) == value and self.lens.shelf_value(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = self.pad.latest(pid, sku) or {}
        if self.ceiling is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.ceiling})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self.lens.brand_name(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def order_payload(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.mandate.lane == "justification":
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def order_ready(self, pid: str, sku: str) -> tuple:
        st = self.pad.latest(pid, sku)
        k = CatalogPad.stamp(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.pad.cart:
            return False, "add that exact variant to the cart first"
        if self.pad.last_add_tick is not None and self.pad.tick_at.get(k, 0) <= self.pad.last_add_tick:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.ceiling is not None and price is not None and price > self.ceiling + _SLACK:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.ceiling)
        if k in self.pad.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    def shopper_reply_text(self) -> str:
        target = self.pick_primary() or (self.lens.scoreboard(self.ceiling) or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self.lens.title_of(pid) or pid)[:90]]
        field, want = self.mandate.spec_field, self.mandate.spec_value
        if field and want and self.lens.sku_value(pid, sku, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.mandate.currency))
        return ", ".join(parts) + "."

    def _reply_shopper(self) -> None:
        if self.question_waits < 2 and "message" in self.toolset:
            self.question_waits += 1
            proposal = self.consult_llm()
            if proposal is not None:
                call = proposal["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = Ink.unpack(call)
                if name == "message" and str(args.get("content") or "").strip():
                    self.enact(name, args, proposal.get("content") or "", call.get("id"))
                    self.shopper_owes_answer = False
                    self.question_waits = 0
                    return
        if "message" in self.toolset:
            self.enact("message", {"content": self.shopper_reply_text()})
        self.shopper_owes_answer = False
        self.question_waits = 0

    def veto_scout(self, name: str, args: dict) -> str | None:
        if name not in self.toolset:
            return "unknown tool; choose one of " + str(sorted(n for n in self.toolset if n))
        if name in _CLOSER_KIT:
            return "the closer owns cart and order; keep scouting with search, filter, view, compare or message"
        if name == "place_test_order":
            return "the closer places the order after two verified SKUs sit on the pad"
        if name == "add_to_cart":
            return "do not cart from the scout desk; the closer commits a verified pair"
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_owes_answer and "?" not in text:
                return "do not send status updates; use message only to answer the shopper or to ask one question"
            if not self.shopper_owes_answer and self.questions_asked >= 2:
                return "the shopper has already answered; decide from the evidence rather than asking again"
        if name == "inspect_cart" and self.cart_peeks >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.pad.seen_calls and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        return None

    def consult_llm(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.chat,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _WIRE.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.chat.append({"role": "assistant", "content": msg.get("content") or ""})
                self.whisper("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            if self.hourglass.left() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    def _post_env(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            result = _WIRE.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.hourglass.left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def enact(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._post_env(pend["envelope"])
            if replay is None:
                self.whisper("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.stopped:
                return None
        if self._guard_repeat(name, args):
            return None
        if name in ("search", "filter", "view", "compare"):
            seen = name + json.dumps(args, sort_keys=True)
            if seen in self.pad.seen_calls:
                return None
            self.pad.seen_calls.add(seen)
        turn = self.turn + 1
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
            self.pending_envelope = {
                "envelope": envelope,
                "turn": turn,
                "name": name,
                "args": args,
                "content": content,
                "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.chat.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.transcript.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.pad.seen_calls.discard(name + json.dumps(args, sort_keys=True))
            self.chat.append(
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
            self.stopped = True
        self.pad.ingest(name, args, obs, self.mandate)
        self.chat.append({"role": "tool", "tool_call_id": call_id, "content": Ink.clip(json.dumps(obs), _OBS_CAP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.chat.append({"role": "user", "content": text})
            self._apply_budget_cut(text)
            if self.mandate.lane in ("constraint", "intent") and not self.mandate.spec_value:
                stated = self.mandate.adopt_spec(text)
                if stated:
                    self.shopper_mutated = True
                    self.whisper("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            if _QUESTION_RX.search(text):
                self.shopper_owes_answer = True
                self.whisper("The shopper asked a question: answer it with message on the next scout turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.stopped = True
        return obs

    def _guard_repeat(self, name: str, args: dict) -> bool:
        signature = name + json.dumps(args, sort_keys=True)
        self.repeats = self.repeats + 1 if signature == self.last_forced else 1
        self.last_forced = signature
        if self.repeats <= 2:
            return False
        if name in ("inspect_stock", "add_to_cart"):
            self.pad.dead_keys.add(CatalogPad.stamp(args.get("product_id"), args.get("sku")))
        if name == "compare":
            self.unreadable.update(str(p) for p in args.get("product_ids") or [])
        if name == "message":
            self.shopper_owes_answer = False
        self.last_forced, self.repeats = None, 0
        self.whisper("That step changed nothing, so the floor moved on to other candidates.")
        return True

    def _tick_ask(self) -> None:
        question = self.opening_line()
        if question and not self.asked_once and "message" in self.toolset:
            self.asked_once = True
            self.enact("message", {"content": question})
            self.phase = "SCOUT"
            return
        self.phase = "SCOUT"

    def _tick_order(self) -> None:
        self.closer.rush_checkout()
        if self.stopped:
            return
        # Never mark the episode done without a terminal environment order.
        self.phase = "CLOSE" if (self.pad.cart or self.pick_primary()) else "SCOUT"

    def run(self) -> list:
        desks: dict[str, Callable[[], None]] = {
            "ASK": self._tick_ask,
            "SCOUT": self.scout.tick,
            "CLOSE": self.closer.tick,
            "RECOVER": self.recover.tick,
            "ORDER": self._tick_order,
        }
        while not self.stopped and self.turn < self.step_cap:
            if self.awaiting_recheck and "inspect_stock" in self.toolset:
                self.closer.recheck_carted()
                continue
            if self.shopper_owes_answer and self.phase not in ("ORDER",):
                self._reply_shopper()
                continue
            if self.in_panic():
                self.phase = "ORDER"
            hop = desks.get(self.phase, self.scout.tick)
            before_turn = self.turn
            hop()
            if self.phase == "ORDER" and self.stopped:
                break
            if self.turn == before_turn:
                self.idle += 1
                if self.idle >= 3:
                    if self.pad.cart or self.pick_primary():
                        self.phase = "CLOSE"
                    elif self.in_panic():
                        self.phase = "ORDER"
                    continue
                if self.phase == "ASK":
                    self.phase = "SCOUT"
                elif self.phase == "SCOUT" and self.may_commit() and self.pick_primary() is not None:
                    self.phase = "CLOSE"
                elif self.phase == "CLOSE":
                    self.phase = "ORDER" if self.pad.cart else "SCOUT"
                elif self.phase == "RECOVER":
                    self.phase = "CLOSE"
            else:
                self.idle = 0
        return self.transcript


def agent_main(problem_data: dict) -> list:
    try:
        return ShopFloor(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
