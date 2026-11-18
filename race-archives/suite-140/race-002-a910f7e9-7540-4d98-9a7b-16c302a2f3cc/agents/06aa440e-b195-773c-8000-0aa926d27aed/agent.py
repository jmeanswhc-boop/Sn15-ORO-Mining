"""ORO-Bench shopping agent — Plan 09: shopper dialogue kernel.

The shopper is a co-pilot. Catalog work stays secondary until the wishlist is
a stable contract, or the turn budget forbids waiting. A language model may
propose one tool call per turn; a deterministic kernel owns dialogue debt and
a stockroom owns every graded commitment.

Evidence is read the way the environment grades it: labelled raw facts first,
then the selected variant's option text, then the title. A number counts only
when its unit or concept word sits next to it. Disagreeing evidence resolves
to nothing. The first cart addition is the public commitment.
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
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_FUSE_S = 270.0
_OBS_CAP = 7000
_REPROMPT_CAP = 2
_SLACK = 1e-9

_DIGIT = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_CAP_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _DIGIT
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_SKU_PAIR_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTE_RE = re.compile(r"\"([^\"]{2,80})\"")
_THEME_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_NEED_N_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_MODEL_COLOR_RE = re.compile(
    r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I
)
_SOFT_MAKER_RE = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*"
    r"(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
)
_AISLE_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SPECISH_RE = re.compile(
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
_RAISE_RE = re.compile(r"\b(?:raise|increase|higher)\b.*\bbudget\b|\bbudget\b.*\b(?:raise|increase|higher)\b", re.I)

# ---------------------------------------------------------------- evidence
# Labelled raw facts, then variant option text, then title. Adjacent units only.
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QHD_TRAP_RE = re.compile(r"\bqHD\b")

_LABEL_FOR = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}


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
    """Display classes asserted by a text: pixel pairs and the usual shorthands."""
    text = text or ""
    if _QHD_TRAP_RE.search(text):
        return set()
    low = text.lower()
    out = set()
    for m in _WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = pixel_band(pair[0], pair[1])
        if cls:
            out.add(cls)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        out.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        out.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        out.add("fhd")
    if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
        out.add("hd")
    return out


_SCRAPE = {"screen_size": inch_set, "refresh_rate": hz_set, "resolution": res_set, "ram": gb_set}


def nearby(field: str, text: str) -> set:
    """Values a free-form string asserts, requiring the number's own concept word."""
    text = text or ""
    if field != "ram":
        return _SCRAPE[field](text)
    paired = {"%ggb" % float(a) for a, _b in _PAIR_RE.findall(text)}
    return paired or {"%ggb" % float(n) for n in _MEM_RE.findall(text)}


def talks_about(field: str, text: str) -> bool:
    """Whether an option string talks about the concept at all (even ambiguously)."""
    text = text or ""
    if field == "ram":
        return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
    if field == "screen_size":
        return bool(_INCH_RE.search(text))
    if field == "refresh_rate":
        return bool(_HZ_RE.search(text))
    return bool(
        _WXH_RE.search(text)
        or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I)
    )


def only_one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def stated_value(field: str | None, text: str) -> str | None:
    """The single value a request states for a concept, or nothing when unclear."""
    if not field:
        return None
    return only_one(_SCRAPE[field](text or ""))


def word_bag(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def same_noun(a: str, b: str) -> bool:
    """Loose noun equality: ignores case, spacing and a trailing plural."""
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


def as_float(value: Any) -> float | None:
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


def tool_args(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


# ------------------------------------------------------------ device identity
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


def scrub_identity(text: str, title: bool = False) -> str:
    """Strip claims that describe what a listing is NOT, or what ships beside it."""
    text = _NEGATED_RE.sub("", text or "")
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text


def handset_verdict(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    if _PART_ONLY_RE.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = scrub_identity(raw_title, title=True)
    kind = " ".join(scrub_identity(str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
    blob = head + " " + kind
    if _NOT_PHONE_RE.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _IS_PHONE_RE.search(blob):
        return "yes"
    os_blob = " ".join(
        scrub_identity(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS_RE.search(os_blob):
        return "yes"
    if _OS_RE.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
        return "yes"
    return "unclear"


_KIND_MARKS = (
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("justification", "trustworthy reason"),
    ("intent", "nice-to-haves"),
)

_TALK_FIRST = (
    "You are a shopping co-pilot. Talk to the shopper first. The wishlist is the contract; "
    "the catalog is secondary until that contract is stable. Exactly one tool call per turn, "
    "at most 30 turns. A dialogue kernel owns every shopper question: answer it with message "
    "before any search, filter, view, inspect, or cart move. Never send progress or status "
    "updates. Never ask the shopper to raise the budget. At most two unsolicited questions. "
    "The first cart addition is graded, so do not park a placeholder. Prices and stock move "
    "the moment something is carted; the kernel re-checks and tells you when to recover. "
    "Do not repeat a tool call with identical arguments; one compare of up to five products "
    "beats several view calls."
)

_LANES = {
    "intent": (
        "Talk first: a later shopper line can still cut the budget, so do not rush the first "
        "add. Firm gates are category, the named hardware specification, budget and stock. A "
        "named brand is only a preference. After the wishlist settles, filter from the current "
        "hard gates and read raw facts before committing."
    ),
    "recall": (
        "Talk first if the shopper asks anything; otherwise search with differently worded "
        "strings compiled from the wishlist theme and gates (k=10, in_stock=true) until enough "
        "distinct matching titles have appeared. Every requested word must be a whole word of "
        "the title. Then view the strongest leads so a candidate can be verified and bought."
    ),
    "constraint": (
        "Talk first: the shopper hid one firm hardware gate. Ask which specification and "
        "value, then obey that exact variant reading — option text before any product-wide "
        "figure. Compile every filter from the current hard gates. If the carted variant "
        "sells out, move to a different product that still meets every gate."
    ),
    "preference": (
        "Talk first: ask which preference ranks first, then use the answer. Preference still "
        "needs two listing_id values before the first add. Search or filter from current hard "
        "gates, compare candidates, and let the first cart item be the best genuine match. "
        "If it is repriced above budget, recover to a different listing_id."
    ),
    "ranking": (
        "Talk first if the shopper asks; otherwise only the named candidate variants count. "
        "Compare those named keys so their facts are on the record. Highest advertised "
        "maximum refresh rate wins, then the largest screen, then the lowest price, reading "
        "the selected variant's own value before a product-wide one. After the market moves, "
        "the remaining named keys are re-ranked."
    ),
    "recovery": (
        "Talk first if the shopper asks; otherwise find the preferred model and color first. "
        "Compile the first search from that model code so every color is on the record. The "
        "first cart item must be exactly that pairing. If it disappears, the same model in "
        "another color before any other model — never relax category, budget or stock."
    ),
    "justification": (
        "Talk first if the shopper asks; otherwise find a product whose own raw facts state "
        "the requested specification, in the requested category and brand, in stock and "
        "within budget. View or compare it so its facts are on the record, then inspect, "
        "cart and order with claims taken only from that evidence."
    ),
    "generic": (
        "Talk to the shopper first. Understand the firm gates, compile search or filter from "
        "the current wishlist, view candidates, and buy the best verified one."
    ),
}


class Fuse:
    def __init__(self, budget: float) -> None:
        self.t0 = time.monotonic()
        self.budget = budget

    def left(self) -> float:
        return self.budget - (time.monotonic() - self.t0)


class Wishlist:
    """Hard gates, soft prefs, budget, currency, hidden spec, priority, category, brand, theme, model/color."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self.harvest_cap(self.goal)
        self.currency = "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.goal)
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = m.group(1)
        self.named_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _SKU_PAIR_RE.findall(self.goal)]
        phrase = _THEME_RE.search(self.goal)
        quoted = _QUOTE_RE.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = _NEED_N_RE.search(self.rules)
        self.need_n = int(sm.group(1)) if sm else 0
        pm = _MODEL_COLOR_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self.harvest_aisle(self.goal)
        self.brand = self.harvest_maker(self.goal, self.category)
        self.soft_brand = None if pm else self.harvest_soft_maker(self.goal)
        self.shape = self.harvest_shape(self.goal)
        self.gates = self.harvest_gates(self.goal)
        self.spec_field, self.spec_value = self.primary_gate()
        self.hidden = self.spec_field is None and "firm requirement" in self.goal.lower()
        self.priority = []
        self.kind = self.classify()

    def classify(self) -> str:
        if self.named_keys:
            return "ranking"
        if self.need_n:
            return "recall"
        if self.pref_model:
            return "recovery"
        for name, sign in _KIND_MARKS:
            if sign in self.rules:
                return name
        if self.theme:
            return "recall"
        if _WHY_RE.search(self.goal):
            return "justification"
        if self.soft_brand:
            return "intent"
        if self.hidden or self.spec_field or self.brand:
            return "constraint"
        return "preference"

    @staticmethod
    def harvest_cap(text: str) -> float | None:
        best = None
        for m in _CAP_RE.finditer(text or ""):
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
    def harvest_aisle(goal: str) -> str | None:
        m = _AISLE_RE.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def harvest_maker(goal: str, category: str | None) -> str | None:
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        if not category:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPECISH_RE.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not res_set(w)]
        return " ".join(keep) or None

    @staticmethod
    def harvest_soft_maker(goal: str) -> str | None:
        m = _SOFT_MAKER_RE.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def harvest_shape(goal: str) -> str | None:
        """A product shape that no shelf name captures, so the title must show it."""
        if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
            return "over ear"
        return None

    @staticmethod
    def harvest_gates(goal: str) -> dict:
        """Every hardware value the request states unambiguously."""
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = stated_value(field, goal)
            if value:
                out[field] = value
        return out

    def primary_gate(self) -> tuple:
        for field, value in self.gates.items():
            return field, value
        return None, None

    def absorb_gate(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = stated_value(field, text)
            if value:
                self.spec_field, self.spec_value = field, value
                self.gates[field] = value
                self.hidden = False
                return field + " = " + value
        return None

    def rank_prefs(self, text: str) -> None:
        """Soft priority order from a shopper reply about what ranks first."""
        bits = re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower())
        drop = {
            "the", "and", "for", "with", "that", "this", "you", "your", "please",
            "rank", "first", "most", "then", "prefer", "preference", "priority",
        }
        ordered = []
        for w in bits:
            if w not in drop and w not in ordered:
                ordered.append(w)
        if ordered:
            self.priority = ordered[:8]

    def compile_query(self) -> str:
        """Every search string is compiled from current hard gates."""
        bits: list[str] = []
        if self.brand:
            bits.append(self.brand)
        if self.spec_value:
            bits.append(str(self.spec_value))
        if self.category:
            bits.append(self.category)
        if self.theme:
            bits.extend(self.theme)
        if self.pref_model:
            bits.append(self.pref_model)
        return " ".join(bits)

    def compile_filter(self, cap: float | None) -> dict:
        """Every filter call is compiled from current hard gates."""
        args: dict[str, Any] = {"in_stock": True, "k": 10}
        if self.category:
            args["category"] = self.category
        if self.brand:
            args["brand"] = self.brand
        if cap is not None:
            args["max_price"] = cap
        return args

    def gate_tuple(self, cap: float | None) -> tuple:
        return (self.brand or "", "%.2f" % (cap or 0.0), str(self.spec_value), " ".join(self.theme))


class Utterance:
    """One shopper line: question, tighter budget, or a revealed hardware gate."""

    def __init__(self, text: str) -> None:
        self.text = text or ""

    def asks(self) -> bool:
        return bool(_ASK_RE.search(self.text))

    def wants_raise(self) -> bool:
        return bool(_RAISE_RE.search(self.text))

    def tighter_than(self, current: float | None) -> float | None:
        """Only a strictly tighter budget applies."""
        fresh = Wishlist.harvest_cap(self.text)
        if fresh is None:
            return None
        if current is not None and fresh >= current:
            return None
        return fresh

    def revealed_gate(self) -> tuple[str, str] | None:
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = stated_value(field, self.text)
            if value:
                return field, value
        return None

    def looks_like_priority(self) -> bool:
        return bool(
            re.search(
                r"\b(?:priority|prefer|most important|rank|first|then|weigh)\b",
                self.text,
                re.I,
            )
        )


class Stockroom:
    """What the agent has observed for itself, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}
        self.checked_at: dict = {}
        self.probed: set = set()
        self.pages: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.listing_ids: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.tick = 0
        self.first_commit: int | None = None
        self.last_commit: int | None = None
        self.commits = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.seen_calls: set = set()

    @staticmethod
    def stamp(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def ingest(self, name: str, args: dict, obs: dict, wish: Wishlist) -> None:
        self.tick += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        handler = {
            "search": self._from_listing,
            "filter": self._from_listing,
            "view": self._from_view,
            "compare": self._from_compare,
            "inspect_stock": self._from_probe,
            "add_to_cart": self._from_add,
            "remove_from_cart": self._from_remove,
            "inspect_cart": self._from_cart,
        }.get(name)
        if handler:
            handler(obs, wish)

    def _from_listing(self, obs: dict, wish: Wishlist) -> None:
        for row in obs.get("results") or []:
            if isinstance(row, dict):
                self._catch_row(row, wish)

    def _from_view(self, obs: dict, wish: Wishlist) -> None:
        self._catch_page(obs, wish)

    def _from_compare(self, obs: dict, wish: Wishlist) -> None:
        for prod in obs.get("compared") or []:
            if isinstance(prod, dict):
                self._catch_page(prod, wish)

    def _from_probe(self, obs: dict, wish: Wishlist) -> None:
        k = self.stamp(obs.get("product_id"), obs.get("sku"))
        self.checked[k] = {"in_stock": bool(obs.get("in_stock")), "price": as_float(obs.get("price"))}
        self.checked_at[k] = self.tick
        self.probed.add(k)

    def _from_add(self, obs: dict, wish: Wishlist) -> None:
        added = obs.get("added") or {}
        pid, sku = added.get("product_id"), added.get("sku")
        if pid and sku:
            self.commits += 1
            if (pid, sku) not in self.cart:
                self.cart.append((pid, sku))
            if self.first_commit is None:
                self.first_commit = self.tick
            self.last_commit = self.tick
        self._mirror_cart(obs)

    def _from_remove(self, obs: dict, wish: Wishlist) -> None:
        self._mirror_cart(obs)
        removed = obs.get("removed") or {}
        pair = (removed.get("product_id"), removed.get("sku"))
        if pair in self.cart:
            self.cart.remove(pair)

    def _from_cart(self, obs: dict, wish: Wishlist) -> None:
        self._mirror_cart(obs)

    def _mirror_cart(self, obs: dict) -> None:
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
            price = as_float(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.checked_at[k] = self.tick

    def _catch_row(self, row: dict, wish: Wishlist) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if wish.theme:
            if all(t in word_bag(title) for t in wish.theme):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        k = self.stamp(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.checked.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": as_float(row.get("price"))})

    def _catch_page(self, prod: dict, wish: Wishlist) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if wish.theme and all(t in word_bag(str(prod.get("title") or "")) for t in wish.theme):
            self.identities.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.stamp(pid, v.get("sku"))
            self.checked[k] = {"in_stock": bool(v.get("in_stock")), "price": as_float(v.get("price"))}
            self.checked_at[k] = self.tick
            self.options[k] = str(v.get("options") or "")

    def snapshot(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.stamp(pid, sku))

    def kill_over_cap(self, cap: float) -> list:
        """Mark carted SKUs above a tighter budget dead. Returns killed stamps."""
        killed = []
        for pid, sku in list(self.cart):
            st = self.snapshot(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > cap + _SLACK:
                k = self.stamp(pid, sku)
                self.dead_keys.add(k)
                killed.append((k, price))
        return killed

    def has_page(self, pid: str) -> bool:
        return str(pid) in self.pages


class TalkKernel:
    """Dialogue owns the turn. Shopper questions freeze shopping until answered."""

    def __init__(self, wish: Wishlist) -> None:
        self.wish = wish
        self.opened = False
        self.owed = False
        self.waits = 0
        self.unsolicited = 0
        self.revised = False

    def first_ask(self) -> str | None:
        if self.wish.kind == "constraint" or self.wish.hidden:
            return (
                "Before any catalog work I need the contract: which hardware gate is firm "
                "(screen size, resolution class, refresh rate, or memory), and what exact "
                "value should I lock? I will not search until that gate is on the wishlist."
            )
        if self.wish.kind == "preference":
            return (
                "Talk first, then shop: which of your preferences ranks first, and in what "
                "order should I weigh the others? I will hold the catalog until that priority "
                "is on the wishlist."
            )
        return None

    def may_open(self) -> bool:
        return not self.opened and self.first_ask() is not None

    def mark_opened(self) -> None:
        self.opened = True
        self.unsolicited += 1

    def preempt(self, shop: "TalkShop") -> tuple | None:
        """Freeze shopping while a shopper question is unpaid."""
        if not self.owed:
            return None
        if self.waits < 2 and "message" in shop.tool_names:
            self.waits += 1
            return ("hold", None)
        if "message" not in shop.tool_names:
            self.owed = False
            self.waits = 0
            return None
        return ("message", {"content": self.compose(shop)})

    def compose(self, shop: "TalkShop") -> str:
        """Auto-answer from observed evidence only: title, spec, price."""
        pick = shop.pick() or (shop.pool() or [None])[0]
        if not pick:
            return (
                "I am still matching the catalog to your wishlist; nothing is decided yet. "
                "I will answer from title, specification and price as soon as I have them."
            )
        pid, sku = pick["pid"], pick["sku"]
        parts = ["I am going with " + (shop.headline(pid) or pid)[:90]]
        field, want = self.wish.spec_field, self.wish.spec_value
        if field and want and shop.sku_value(pid, sku, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if pick.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (pick["price"], self.wish.currency))
        return ", ".join(parts) + "."

    def hear(self, text: str, shop: "TalkShop") -> None:
        """Parse a shopper reply into a tighter budget and/or an adopted spec."""
        line = Utterance(text)
        cut = line.tighter_than(shop.cap)
        if cut is not None:
            shop.cap = cut
            self.wish.budget = cut
            self.revised = True
            shop.whisper(
                "Wishlist budget cut by the shopper: at most %.2f %s. Every later choice must fit it."
                % (cut, self.wish.currency)
            )
            for stamp, price in shop.room.kill_over_cap(cut):
                shop.whisper(
                    "The carted item %s costs %.2f, above the new wishlist cap; it is marked dead."
                    % (stamp, price)
                )
        if self.wish.kind in ("constraint", "intent") and not self.wish.spec_value:
            taken = self.wish.absorb_gate(text)
            if taken:
                self.revised = True
                shop.whisper(
                    "Wishlist absorbed firm gate: " + taken + ". Only a variant that resolves to it may be bought."
                )
        elif line.revealed_gate() and self.wish.hidden:
            taken = self.wish.absorb_gate(text)
            if taken:
                self.revised = True
                shop.whisper("Wishlist absorbed the sealed specification: " + taken + ".")
        if self.wish.kind == "preference" and line.looks_like_priority():
            self.wish.rank_prefs(text)
            if self.wish.priority:
                shop.whisper("Wishlist priority order is now: " + ", ".join(self.wish.priority) + ".")
        if line.asks():
            self.owed = True
            shop.whisper(
                "The shopper asked a direct question. The talk kernel preempts every catalog "
                "action: answer with message next, then resume shopping."
            )

    def clear_debt(self) -> None:
        self.owed = False
        self.waits = 0

    def allow_unsolicited(self) -> bool:
        return self.unsolicited < 2

    def note_unsolicited(self) -> None:
        if not self.owed:
            self.unsolicited += 1


class TalkShop:
    """Co-pilot shop: talk kernel first, stockroom second, catalog last."""

    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.wish = Wishlist(str(self.policy.get("query") or ""))
        self.cap = self.wish.budget
        self.room = Stockroom()
        self.kernel = TalkKernel(self.wish)
        self.fuse = Fuse(_FUSE_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self._charter()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_calls = 0
        self.cart_looks = 0
        self.owed_recheck: tuple | None = None
        self.forced_reads = 0
        self.hinted_exact = False
        self.hinted_brand = False
        self._soft_cache: set | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.searched_siblings = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.done = False
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _ROSTER.get(provider) or _ROSTER["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _charter(self) -> str:
        lane = _LANES.get(self.wish.kind, _LANES["generic"])
        extra = ""
        if self.wish.kind == "recall" and self.wish.need_n:
            extra = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the stockroom reports the count after each search."
                % self.wish.need_n
            )
        return _TALK_FIRST + "\n\nTalk-first lane: " + lane + extra

    def whisper(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    # ------------------------------------------------------------------ evidence
    def page(self, pid: str) -> dict:
        return self.room.pages.get(str(pid)) or {}

    def raw_facts(self, pid: str) -> dict:
        f = self.page(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def headline(self, pid: str) -> str:
        title = str(self.page(pid).get("title") or "")
        if title:
            return title
        for key, row in self.room.rows.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def maker(self, pid: str) -> str:
        brand = self.page(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.room.rows.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def aisle(self, pid: str) -> list:
        path = self.page(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def option_line(self, pid: str, sku: str) -> str:
        return self.room.options.get(Stockroom.stamp(pid, sku), "")

    def drop_sku_codes(self, pid: str, title: str) -> str:
        """Drop model codes from a title so a product name is not read as a panel class."""
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

    def labelled(self, pid: str, field: str) -> str | None:
        """Labelled fact value; a bare number under a unit-bearing label inherits that unit."""
        raw = self.raw_facts(pid).get(_LABEL_FOR[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _SCRAPE[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return only_one(found)

    def shelf_value(self, pid: str, field: str) -> str | None:
        """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
        vals: set = set()
        conflict = False
        raw = self.raw_facts(pid).get(_LABEL_FOR[field])
        if raw:
            found = _SCRAPE[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and _QHD_TRAP_RE.search(str(raw)):
                conflict = True
            vals |= found
        title = self.headline(pid)
        if field == "resolution":
            title = self.drop_sku_codes(pid, title)
            if _QHD_TRAP_RE.search(title):
                conflict = True
        vals |= nearby(field, title)
        return None if conflict else only_one(vals)

    def sku_value(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        opt = self.option_line(pid, sku)
        vals = nearby(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if talks_about(field, opt):
            return None
        return self.shelf_value(pid, field)

    def numeric(self, pid: str, sku: str, field: str) -> float | None:
        value = self.sku_value(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @staticmethod
    def stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def aisle_fits(self, pid: str) -> bool:
        """Whether the listing sits on the shelf the shopper asked for."""
        want = self.wish.category
        path = self.aisle(pid)
        if not want or not path:
            return True
        wanted = self.stems(want)
        labels = set()
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
        qualifier = self.wish.shape
        if not qualifier:
            return True
        title = self.headline(pid)
        words = self.stems(title) | word_bag(title)
        if qualifier == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in hz_set(title))
            return "monitor" in words and ("gaming" in words or fast)
        return "over" in words and "ear" in words or "overear" in words or "circumaural" in words

    def maker_fits(self, pid: str) -> bool:
        want = self.wish.brand
        if not want:
            return True
        wanted = word_bag(want)
        return bool(wanted <= word_bag(self.maker(pid)) or wanted <= word_bag(self.headline(pid)))

    def kind_fits(self, pid: str, sku: str) -> bool:
        """Device-identity gate: positive evidence when the task turns on kind."""
        if not same_noun(self.wish.category or "", "smartphone"):
            return True
        verdict = handset_verdict(self.raw_facts(pid), self.headline(pid), self.option_line(pid, sku))
        if self.wish.kind in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def is_resale(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        return bool(
            re.search(
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
                self.headline(pid),
                re.I,
            )
        )

    def drift(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        model, color = self.wish.pref_model, self.wish.pref_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self.headline(pid) + " " + " ".join(str(v) for v in self.raw_facts(pid).values()))
        same_model = squash(model) in blob
        shade = (
            self.option_line(pid, sku)
            + " "
            + str(self.raw_facts(pid).get("color") or "")
            + " "
            + self.headline(pid)
        ).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def score(self, pid: str, sku: str) -> dict:
        """Score one variant against every gate the runtime will apply to it."""
        fam = self.wish.kind
        key = Stockroom.stamp(pid, sku)
        st = self.room.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.room.dead_keys or (str(pid) in self.room.dead_products and fam != "recovery")
        live = bool(stock) and price is not None and (self.cap is None or price <= self.cap + _SLACK) and not dead
        viewed = str(pid) in self.room.pages
        field, want = self.wish.spec_field, self.wish.spec_value
        spec = 2
        if field and want:
            got = self.sku_value(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self.maker_fits(pid) and self.kind_fits(pid, sku))
        cat_ok = self.aisle_fits(pid)
        ok = ok and cat_ok
        if fam == "justification":
            ok = ok and not self.is_resale(pid) and bool(self.maker(pid))
            if field and want:
                ok = ok and self.labelled(pid, field) == want and self.shelf_value(pid, field) == want
        if fam == "preference" and self.room.event_target:
            dead_listing = self.room.listing_of.get(self.room.event_target.split("::", 1)[0])
            if dead_listing and self.room.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == "recall" and self.wish.theme:
            ok = ok and all(t in word_bag(self.headline(pid)) for t in self.wish.theme)
        if fam == "ranking":
            ok = ok and (str(pid), str(sku)) in self.wish.named_keys
            ok = ok and self.numeric(pid, sku, "refresh_rate") is not None
            ok = ok and self.numeric(pid, sku, "screen_size") is not None
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
            "relax": self.drift(pid, sku) if fam == "recovery" else 0,
        }

    def sort_key(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates the way the task's own priority does."""
        fam = self.wish.kind
        price = c["price"] if c["price"] is not None else 1e12
        if fam == "ranking":
            hz = self.numeric(c["pid"], c["sku"], "refresh_rate") or 0.0
            inch = self.numeric(c["pid"], c["sku"], "screen_size") or 0.0
            return (-hz, -inch, price, c["key"])
        if fam == "recovery":
            return (c["relax"], price, c["key"])
        extra = sum(
            1
            for f, v in self.wish.gates.items()
            if f != self.wish.spec_field and self.sku_value(c["pid"], c["sku"], f) != v
        )
        proven = 0 if handset_verdict(self.raw_facts(c["pid"]), self.headline(c["pid"]), self.option_line(c["pid"], c["sku"])) == "yes" else 1
        if fam == "intent" and first and self.wish.soft_brand:
            wanted = word_bag(self.wish.soft_brand)
            near = 0 if (wanted <= word_bag(self.maker(c["pid"])) or wanted <= word_bag(self.headline(c["pid"]))) else 1
            return (near, extra, proven, price, c["key"])
        return (extra, proven, price, c["key"])

    def pool(self) -> list:
        seen = set()
        out = []
        for key in list(self.room.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self.score(pid, sku))
        return out

    def pick(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        first = self.room.commits == 0
        fam = self.wish.kind
        good = [c for c in self.pool() if c["ok"]]
        if fam == "preference":
            carted = [c for c in good if (c["pid"], c["sku"]) in self.room.cart]
            if carted:
                return carted[0]
            if good and self.turn >= self.max_steps - 8:
                good.sort(
                    key=lambda c: (
                        -self.pref_hits(c["pid"], c["sku"]),
                        c["price"] if c["price"] is not None else 1e12,
                    )
                )
                return good[0]
            return None
        if fam == "intent" and first and self.wish.soft_brand:
            wanted = word_bag(self.wish.soft_brand)
            named = [
                c
                for c in good
                if wanted <= word_bag(self.maker(c["pid"])) or wanted <= word_bag(self.headline(c["pid"]))
            ]
            if named:
                good = named
            elif self.turn < self.max_steps * 0.6:
                self._brand_nudge()
                return None
        if fam == "recovery" and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                self._exact_nudge()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self.sort_key(c, first))
        return good[0]

    def pref_hits(self, pid: str, sku: str) -> int:
        wants = self._soft_tokens()
        if not wants:
            return 0
        have = word_bag(
            " ".join(str(v) for v in self.raw_facts(pid).values())
            + " "
            + self.headline(pid)
            + " "
            + self.option_line(pid, sku)
        )
        if self.wish.priority:
            ranked = 0
            bag = have
            for i, token in enumerate(self.wish.priority):
                if token in bag:
                    ranked += len(self.wish.priority) - i
            return ranked + len(wants & have)
        return len(wants & have)

    def _soft_tokens(self) -> set:
        if self._soft_cache is None:
            said = self.wish.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = word_bag(self.wish.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._soft_cache = {w for w in word_bag(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._soft_cache

    def recall_need(self) -> int:
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return self.wish.need_n + (2 if early else 0)

    def stable(self) -> bool:
        """Whether the wishlist is a contract, or the turn budget forbids waiting."""
        fam = self.wish.kind
        if self.room.commits:
            return True
        if self.wish.hidden:
            if self.turn < self.max_steps * 0.6:
                return False
            self.wish.hidden = False
            self.whisper("The sealed specification was never stated, so the remaining gates decide the choice.")
        if fam in ("intent", "constraint") and self.turn < 4 and not self.kernel.revised:
            return False
        if fam == "preference" and len(self.room.listing_ids) < 2:
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.fuse.left() > 140
        if fam == "recall" and roomy:
            if len(self.room.identities) < self.recall_need() or self.search_calls < 2:
                return False
        if roomy and not self.reads_done and len({c["pid"] for c in self.pool() if c["ok"]}) < 2:
            return False
        return True

    def fresh(self, pid: str, sku: str) -> bool:
        k = Stockroom.stamp(pid, sku)
        if k not in self.room.probed:
            return False
        return self.room.last_commit is None or self.room.checked_at.get(k, 0) > self.room.last_commit

    def unread(self) -> list:
        """Promising variants whose product page has not been read yet."""
        leads = []
        for key, row in self.room.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.room.pages or str(pid) in self.room.dead_products:
                continue
            st = self.room.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.cap is not None and price > self.cap + _SLACK):
                continue
            title = str(row.get("title") or "")
            if self.wish.theme and not all(t in word_bag(title) for t in self.wish.theme):
                continue
            if self.wish.pref_model and self.room.commits == 0 and not self.room.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.wish.pref_model.lower()) not in squashed:
                    continue
            if self.wish.brand:
                wanted = word_bag(self.wish.brand)
                if not (wanted <= word_bag(str(row.get("brand") or "")) or wanted <= word_bag(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def hunt_twins(self) -> tuple | None:
        """After the market kills the carted item, its own twins are the nearest replacements."""
        dead = self.room.event_target
        if self.searched_twins or not dead or "search" not in self.tool_names:
            return None
        title = self.headline(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self.whisper(
            "The item that died is one of several listings sharing a title; searching that title "
            "surfaces its other colors and sizes, which meet the same wishlist gates."
        )
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def hunt_siblings(self) -> tuple | None:
        """A found model in the wrong color: its other colors share the same title."""
        if self.searched_siblings:
            return None
        rows = [c for c in self.pool() if c["live"] and c["viewed"] and c["relax"] == 0]
        if rows:
            return None
        near = [c for c in self.pool() if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self.headline(near[0]["pid"])
        if not title:
            return None
        self.searched_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _brand_nudge(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self.whisper(
            "No %s product that meets every firm wishlist gate has been read yet. The shopper's "
            "first choice is graded on that brand, so filter by brand=%s within the cap and read "
            "those products before any other." % (self.wish.soft_brand, self.wish.soft_brand)
        )

    def _exact_nudge(self) -> None:
        if self.hinted_exact or not self.wish.pref_model:
            return
        self.hinted_exact = True
        self.whisper(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code "
            "on its own and view the results so every color variant is on record; only that exact "
            "pairing may be the first cart item."
            % (self.wish.pref_model, ", " + self.wish.pref_color if self.wish.pref_color else "")
        )

    def browse(self) -> tuple | None:
        """Turn listing rows into read pages, then widen with compiled gates."""
        fam = self.wish.kind
        if fam == "recall" and len(self.room.identities) < self.recall_need():
            return None
        twin = self.hunt_twins()
        if twin is not None:
            return twin
        leads = self.unread()
        if leads and self.forced_reads < 8 and self.turn >= 2:
            self.forced_reads += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        probe = self.wish.gate_tuple(self.cap)
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
        if "filter" in self.tool_names and self.wish.category:
            return ("filter", self.wish.compile_filter(self.cap))
        if "search" in self.tool_names:
            q = self.wish.compile_query()
            if q:
                return ("search", {"query": q, "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def briefing(self) -> None:
        """Tell the model what the stockroom can and cannot yet verify."""
        cands = self.pool()
        if self.wish.kind == "preference" and not self.room.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(
                key=lambda c: (
                    -self.pref_hits(c["pid"], c["sku"]),
                    c["price"] if c["price"] is not None else 1e12,
                )
            )
            if good:
                self.whisper(
                    "Products that meet every firm wishlist gate so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it."
                )
                return
        best = self.pick()
        if best:
            self.whisper(
                "Verified candidate: %s::%s at %.2f. The stockroom will inspect, cart and order it; "
                "read other candidates only if one could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        field, want = self.wish.spec_field, self.wish.spec_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self.whisper(
                "None of the products read so far can be verified. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self.whisper(
                "The products read so far fail a firm wishlist gate (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def proofs(self, pid: str, sku: str) -> list:
        """Claims proven by what the agent has already observed about this exact item."""
        out = []
        field, want = self.wish.spec_field, self.wish.spec_value
        if not field and self.wish.kind == "justification":
            field, want = "resolution", self.labelled(pid, "resolution")
        if not field or not want:
            return []
        stated = dict(self.wish.gates) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self.labelled(pid, name) == value and self.shelf_value(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = self.room.snapshot(pid, sku) or {}
        if self.cap is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.cap})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self.maker(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def checkout(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.wish.kind == "justification":
            claims = self.proofs(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def orderable(self, pid: str, sku: str) -> tuple:
        st = self.room.snapshot(pid, sku)
        k = Stockroom.stamp(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.room.cart:
            return False, "add that exact variant to the cart first"
        if self.room.last_commit is not None and self.room.checked_at.get(k, 0) <= self.room.last_commit:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.cap is not None and price is not None and price > self.cap + _SLACK:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.cap)
        if k in self.room.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    # ------------------------------------------------------------------ steering
    def steer(self) -> tuple | None:
        """Controller move. Talk debt preempts every catalog action."""
        if self.done:
            return None
        held = self.kernel.preempt(self)
        if held is not None:
            if held[0] == "hold":
                return None
            return held
        fam = self.wish.kind
        if fam == "ranking" and "compare" in self.tool_names:
            missing = [
                p
                for p, _s in self.wish.named_keys
                if p not in self.room.pages and p not in self.unreadable
            ]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if fam == "recovery" and not self.room.commits and self.wish.pref_model and "search" in self.tool_names:
            forced = self.hunt_siblings()
            if forced is not None:
                return forced
        target = self.pick()
        if target is None or not self.stable():
            forced = self.browse()
            if forced is not None:
                return forced
            if target is None:
                return None
        pid, sku = target["pid"], target["sku"]
        if (pid, sku) in self.room.cart:
            if not self.fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self.checkout(pid, sku))
        if not self.stable():
            return None
        for cpid, csku in self.room.cart:
            if Stockroom.stamp(cpid, csku) in self.room.dead_keys and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if not self.fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    # ------------------------------------------------------------------ inference
    def consult(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _PROXY.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self.whisper("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            if self.fuse.left() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------------ environment
    def post_env(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; an identical envelope replays idempotently."""
        for attempt in range(4):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.fuse.left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def dispatch(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self.post_env(pend["envelope"])
            if replay is None:
                self.whisper("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self.ingest(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
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
        result = self.post_env(envelope)
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
        return self.ingest(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def ingest(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.room.seen_calls.discard(name + json.dumps(args, sort_keys=True))
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
        self.room.ingest(name, args, obs, self.wish)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": clip_obs(json.dumps(obs), _OBS_CAP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self.kernel.hear(text, self)
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    def after_commit(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        pid, sku = self.owed_recheck
        self.owed_recheck = None
        obs = self.dispatch("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = as_float(obs.get("price"))
        over = self.cap is not None and price is not None and price > self.cap + _SLACK
        if stock and not over:
            self.whisper(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The stockroom will order it."
                % (Stockroom.stamp(pid, sku), price)
            )
            return
        self.room.event_target = Stockroom.stamp(pid, sku)
        self.event_turn = self.turn
        self.room.dead_keys.add(Stockroom.stamp(pid, sku))
        self.room.dead_products.add(str(pid))
        self.forced_reads = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        guidance = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "The remaining named candidates are re-ranked; the stockroom takes the new best.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every wishlist gate, not another variant of the same one.",
        }.get(self.wish.kind, "Choose a different product that meets every firm wishlist gate.")
        self.whisper("Market change: the carted item %s %s. " % (Stockroom.stamp(pid, sku), reason) + guidance)
        self.briefing()

    def panic_buy(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for pid, sku in reversed(self.room.cart):
            ok, _why = self.orderable(pid, sku)
            if ok and not self.done:
                self.dispatch("place_test_order", self.checkout(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [c for c in self.pool() if c["ok"]]
        if not good and self.wish.kind != "justification":
            good = [
                c
                for c in self.pool()
                if c["live"]
                and c["spec"] == 2
                and self.maker_fits(c["pid"])
                and c["cat_ok"]
                and (not self.wish.named_keys or (c["pid"], c["sku"]) in self.wish.named_keys)
            ]
        if not good:
            return
        good.sort(key=lambda c: self.sort_key(c, self.room.commits == 0))
        pid, sku = good[0]["pid"], good[0]["sku"]
        if (pid, sku) not in self.room.cart:
            obs = self.dispatch("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = self.dispatch("inspect_stock", {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = as_float(obs.get("price"))
        if self.cap is not None and price is not None and price > self.cap + _SLACK:
            return
        self.dispatch("place_test_order", self.checkout(pid, sku))

    def run(self) -> list:
        opener = self.kernel.first_ask()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.fuse.left() < 25 or self.turn >= self.max_steps - 3:
                self.panic_buy()
                break
            if opener and self.kernel.may_open() and "message" in self.tool_names:
                self.kernel.mark_opened()
                self.dispatch("message", {"content": opener})
                opener = None
                continue
            if self.owed_recheck and "inspect_stock" in self.tool_names:
                self.after_commit()
                continue
            move = self.steer()
            if move is not None:
                name, args = move
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.room.dead_keys.add(Stockroom.stamp(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.kernel.clear_debt()
                    last_forced, repeats = None, 0
                    self.whisper("That step changed nothing, so the stockroom moved on to other candidates.")
                    continue
                obs = self.dispatch(name, args)
                if name == "message":
                    self.kernel.clear_debt()
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.owed_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            proposal = self.consult()
            if proposal is None:
                self.panic_buy()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = tool_args(call)
            content = proposal.get("content") or ""
            reprompts = 0
            while True:
                verdict = self.veto(name, args)
                if verdict is None:
                    break
                if reprompts >= _REPROMPT_CAP:
                    self.rejected_streak += 1
                    self.whisper("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self.briefing()
                        self.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self.consult()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = tool_args(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.room.cart:
                    pid, sku = self.room.cart[-1]
                args = self.checkout(pid, sku)
            if name == "message":
                if not self.kernel.owed:
                    self.kernel.note_unsolicited()
                self.kernel.clear_debt()
            obs = self.dispatch(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.owed_recheck = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                self.search_calls += 1
                if self.wish.kind == "recall" and self.wish.need_n:
                    have, need = len(self.room.identities), self.recall_need()
                    self.whisper(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording compiled from the wishlist.")
                    )
            if name == "inspect_cart":
                self.cart_looks += 1
            if name in ("view", "compare"):
                self.briefing()
        return self.dialogue

    def veto(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if self.kernel.owed and name != "message":
            return (
                "the shopper asked a direct question; the talk kernel preempts shopping — "
                "answer with message before any catalog action"
            )
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.room.cart:
                pid, sku = self.room.cart[-1]
            ok, why = self.orderable(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.wish.kind == "recall" and len(self.room.identities) < self.recall_need():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.room.identities), self.recall_need())
                )
            if self.wish.kind == "preference" and len(self.room.listing_ids) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self.score(pid, sku)
            key = Stockroom.stamp(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.room.probed:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.wish.spec_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.wish.spec_field, self.wish.spec_value)
                )
            if not c["ok"]:
                return "that product fails a firm wishlist gate (category, brand, or the kind of device requested)"
            if self.wish.kind == "preference" and len(self.room.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.wish.kind == "recovery" and not self.room.commits and self.wish.pref_model and self.drift(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if not self.stable():
                return (
                    "hold: the wishlist is not a stable contract yet, or one more compliant product should be "
                    "read first so there is a fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if Utterance(text).wants_raise():
                return "never ask the shopper to raise the budget; decide from the current wishlist cap"
            if not self.kernel.owed and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.kernel.owed and not self.kernel.allow_unsolicited():
                return (
                    "the shopper has already answered; decide from the evidence you have rather than asking again"
                )
        if name == "inspect_cart" and self.cart_looks >= 1:
            return "the cart contents are already known from earlier observations"
        if name in ("search", "filter"):
            compiled = self.wish.compile_query() if name == "search" else None
            if name == "search" and compiled and not str(args.get("query") or "").strip():
                return "search query is empty; compile it from the current wishlist hard gates"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.room.seen_calls and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.room.seen_calls.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return TalkShop(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
