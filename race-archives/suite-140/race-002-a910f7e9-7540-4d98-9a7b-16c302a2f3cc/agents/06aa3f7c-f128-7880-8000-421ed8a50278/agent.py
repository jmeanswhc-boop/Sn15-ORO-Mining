"""ORO-Bench shopping agent: reprice-headroom buyer (Plan 05).

The first cart addition is the public commitment, and the market often
reprices that exact variant the moment it lands. This controller therefore
hunts tickets that can absorb a bump: it sweeps the catalog in three price
bands, ranks legal tickets by leftover-budget headroom, and refuses to
panic-replace a carted item that is still in stock and still under the
ceiling. Evidence is still read the way the environment grades it, and the
environment envelope is unchanged.
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
_WIRE = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_FUSE_S = 270.0
_OBS_CAP = 7000
_RETRY_PROMPTS = 2
_EPS = 1e-9
_CUSHION_LO = 0.12
_CUSHION_HI = 0.40
_BAND_FRACS = (0.55, 0.80, 1.00)
_DEFAULT_SLACK = 0.25

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _NUM
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_PREF_ITEM_RE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_PREF_BRAND_RE = re.compile(r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)")
_CAT_PHRASE_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SPEC_WORD_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_REASON_RE = re.compile(r"\b(?:why|reason|explain|justif)", re.I)
_CLAIM_FIELDS = ("resolution", "refresh_rate", "screen_size", "ram", "price", "in_stock", "brand")
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

# ---------------------------------------------------------------- attributes
# The environment resolves a gated specification from public evidence only: the
# labelled raw facts a product page exposes, the selected variant's option text
# and the listing title. A number counts only when its unit or concept word sits
# next to it, and evidence that disagrees with itself resolves to nothing.
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QUARTER_RE = re.compile(r"\bqHD\b")
_FACT_LABEL = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}


def _inch_vals(text: str) -> set:
    return {"%gin" % float(m.group(1)) for m in _INCH_RE.finditer(text or "")}


def _hz_vals(text: str) -> set:
    return {"%ghz" % float(m.group(1)) for m in _HZ_RE.finditer(text or "")}


def _gb_vals(text: str) -> set:
    return {
        "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
        for m in _GB_RE.finditer(text or "")
    }


def _pixel_class(long_side: int, short_side: int) -> str | None:
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def _res_vals(text: str) -> set:
    """Display classes asserted by a text: pixel pairs and the usual shorthands."""
    text = text or ""
    if _QUARTER_RE.search(text):
        return set()
    low = text.lower()
    out = set()
    for m in _WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _pixel_class(pair[0], pair[1])
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


_VALUE_SETS = {"screen_size": _inch_vals, "refresh_rate": _hz_vals, "resolution": _res_vals, "ram": _gb_vals}


def _adjacent_vals(field: str, text: str) -> set:
    """Values a free-form string asserts, requiring the number's own concept word."""
    text = text or ""
    if field != "ram":
        return _VALUE_SETS[field](text)
    paired = {"%ggb" % float(a) for a, _b in _PAIR_RE.findall(text)}
    return paired or {"%ggb" % float(n) for n in _MEM_RE.findall(text)}


def _mentions(field: str, text: str) -> bool:
    """Whether an option string talks about the concept at all (even ambiguously)."""
    text = text or ""
    if field == "ram":
        return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
    if field == "screen_size":
        return bool(_INCH_RE.search(text))
    if field == "refresh_rate":
        return bool(_HZ_RE.search(text))
    return bool(_WXH_RE.search(text) or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I))


def _one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _wanted_value(field: str | None, text: str) -> str | None:
    if not field:
        return None
    return _one(_VALUE_SETS[field](text or ""))


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _same_word(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


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


def _call_args(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _headroom(price: float | None, budget: float | None) -> float:
    """Leftover-budget ratio. Unknown ceiling defaults to a mid cushion."""
    if budget:
        if price is None:
            return 0.0
        return (budget - price) / budget
    return _DEFAULT_SLACK


def _in_cushion(slack: float) -> bool:
    return _CUSHION_LO <= slack <= _CUSHION_HI


def _glued_to_ceiling(slack: float) -> bool:
    """True when the ticket sits on the budget lid and cannot absorb a bump."""
    return slack < _CUSHION_LO


def _band_caps(budget: float | None) -> list[float]:
    """Discovery ceilings: 55%, 80%, then the full purse, when a budget exists."""
    if not budget:
        return []
    return [frac * budget for frac in _BAND_FRACS]


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
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s+protector|charger|accessor(?:y|ies))\b))",
    re.I,
)


def _identity_text(text: str, title: bool = False) -> str:
    text = _NEGATED_RE.sub("", text or "")
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text


def _phone_identity(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    if _PART_ONLY_RE.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _identity_text(raw_title, title=True)
    kind = " ".join(_identity_text(str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
    blob = head + " " + kind
    if _NOT_PHONE_RE.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _IS_PHONE_RE.search(blob):
        return "yes"
    os_blob = " ".join(
        _identity_text(str(facts.get(k) or ""))
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

_SURVIVAL_LANES = {
    "intent": (
        "Firm gates: category, the named hardware spec, stock, and the live ceiling. A named brand is "
        "only a preference: if that brand has a headroom-legal ticket (in stock, under budget, spec "
        "resolved), take it; otherwise any surviving brand. Sweep filter/search in three price bands "
        "when a budget is known — 55 percent, 80 percent, then the full ceiling — with in_stock=true "
        "and k=10. Leave cushion so a first-add reprice does not eject the item. Answer the shopper "
        "with message; a later, lower budget applies to every later choice."
    ),
    "recall": (
        "Run several differently worded searches (k=10, in_stock=true) built from the requested words "
        "and close synonyms until enough distinct matching products have appeared; every requested word "
        "must appear as a whole word of the title. Then view the strongest candidates so a cushioned "
        "ticket can be verified and bought. Do not spend the whole purse on the first hit."
    ),
    "constraint": (
        "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for "
        "the specific variant, reading the variant's own option text before any product-wide value. "
        "Sweep the three price bands (55 / 80 / full) with filter(category, brand, max_price, "
        "in_stock=true) and view candidates that still have leftover budget. If the carted variant "
        "sells out or is pushed over the ceiling, move to a different product; a mere bump that stays "
        "legal is kept."
    ),
    "preference": (
        "You already asked which preference matters most; use the answer. Sweep the category through "
        "the three survival bands, then compare candidates with different listing_id values, reading "
        "raw facts. The first cart item is graded, so make it a genuine match that can absorb a "
        "reprice. Recover to a different listing_id only when the carted one is out of stock or now "
        "over budget."
    ),
    "ranking": (
        "Only the named candidate variants count. Headroom is ignored here: highest advertised maximum "
        "refresh rate wins, then the largest screen, then the lowest price, reading the selected "
        "variant's own value before a product-wide one. Compare all named keys first so their facts "
        "are on the record. After the market moves, the remaining named candidates are re-ranked the "
        "same way."
    ),
    "recovery": (
        "Find the preferred model and color first: search the model code, then view it so every variant "
        "and color is on the record. The first cart item must be exactly that model and color. If it "
        "disappears, relax as little as possible — the same model in another color before any other "
        "model — and never relax the category, budget or stock. A reprice that stays under the ceiling "
        "is not a disappearance."
    ),
    "justification": (
        "Find a product whose own raw facts state the requested specification, in the requested "
        "category and brand, in stock and within budget, preferably with 12–40 percent leftover so a "
        "first-add bump does not kill the claim set. View or compare it so its facts are on the "
        "record; the controller inspects, carts and orders with claims taken only from that evidence."
    ),
    "generic": (
        "Read the firm requirements, sweep the catalog in leftover-budget bands when a ceiling is "
        "known, view candidates, and let the controller verify and buy the ticket most likely to "
        "survive a first-add reprice."
    ),
}

_SURVIVAL_RULES = (
    "You are a shopping agent that buys for price-band survival. Exactly one tool call per turn, at "
    "most 30 turns. A deterministic controller owns the cart and the order once a ticket is fully "
    "verified; you surface and read candidates. The first cart addition is graded and is often "
    "repriced: prefer items that sit in the 12–40 percent leftover-budget band so they can absorb a "
    "bump. Sweep filter or search at 55 percent of budget, then 80 percent, then the full ceiling "
    "(in_stock=true, k=10); never resend an identical band query. Inspect stock before adding, and "
    "expect a recheck after the add. If the carted item is still in stock and still under budget, "
    "keep it — do not panic-replace a still-legal ticket. Never add a placeholder. Answer every "
    "direct shopper question with one short message before anything else; never send status chatter; "
    "never ask the shopper to raise the budget. One compare of up to five products beats several views."
)


class Purse:
    """Shopper constraints parsed from the public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self.pull_budget(self.goal)
        self.currency = "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.goal)
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = m.group(1)
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = _SURFACE_RE.search(self.rules)
        self.required_identities = int(sm.group(1)) if sm else 0
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self.pull_category(self.goal)
        self.brand = self.pull_brand(self.goal, self.category)
        self.pref_brand = None if pm else self.pull_pref_brand(self.goal)
        self.cat_qualifier = self.pull_qualifier(self.goal)
        self.specs = self.pull_specs(self.goal)
        self.spec_field, self.spec_value = self.lead_spec()
        self.spec_sealed = self.spec_field is None and "firm requirement" in self.goal.lower()
        self.family = self.classify()

    def classify(self) -> str:
        if self.candidate_keys:
            return "ranking"
        if self.required_identities:
            return "recall"
        if self.pref_model:
            return "recovery"
        for name, sign in _LANE_MARKS:
            if sign in self.rules:
                return name
        if self.theme:
            return "recall"
        if _REASON_RE.search(self.goal):
            return "justification"
        if self.pref_brand:
            return "intent"
        if self.spec_sealed or self.spec_field or self.brand:
            return "constraint"
        return "preference"

    @staticmethod
    def pull_budget(text: str) -> float | None:
        best = None
        for m in _BUDGET_RE.finditer(text or ""):
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
    def pull_category(goal: str) -> str | None:
        m = _CAT_PHRASE_RE.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def pull_brand(goal: str, category: str | None) -> str | None:
        if not category:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RE.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _res_vals(w)]
        return " ".join(keep) or None

    @staticmethod
    def pull_pref_brand(goal: str) -> str | None:
        m = _PREF_BRAND_RE.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def pull_qualifier(goal: str) -> str | None:
        if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
            return "over ear"
        return None

    @staticmethod
    def pull_specs(goal: str) -> dict:
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = _wanted_value(field, goal)
            if value:
                out[field] = value
        return out

    def lead_spec(self) -> tuple:
        for field, value in self.specs.items():
            return field, value
        return None, None

    def unseal(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = _wanted_value(field, text)
            if value:
                self.spec_field, self.spec_value = field, value
                self.specs[field] = value
                self.spec_sealed = False
                return field + " = " + value
        return None

    def tighten(self, text: str) -> float | None:
        """Adopt a lower ceiling from a later shopper message."""
        fresh = self.pull_budget(text)
        if fresh is None or (self.budget is not None and fresh >= self.budget):
            return None
        self.budget = fresh
        return fresh


class Ticket:
    """One product×sku with legality flags and leftover-budget slack."""

    def __init__(
        self,
        pid: str,
        sku: str,
        price: float | None,
        live: bool,
        viewed: bool,
        spec: int,
        cat_ok: bool,
        ok: bool,
        relax: int,
        budget: float | None,
    ) -> None:
        self.pid = str(pid)
        self.sku = str(sku)
        self.key = str(pid) + "::" + str(sku)
        self.price = price
        self.live = live
        self.viewed = viewed
        self.spec = spec
        self.cat_ok = cat_ok
        self.ok = ok
        self.relax = relax
        self.budget = budget

    @property
    def slack(self) -> float:
        return _headroom(self.price, self.budget)

    def in_cushion(self) -> bool:
        return _in_cushion(self.slack)

    def glued(self) -> bool:
        return _glued_to_ceiling(self.slack)

    def pair(self) -> tuple:
        return (self.pid, self.sku)


class Shelf:
    """What this run has observed, keyed by product and variant."""

    def __init__(self) -> None:
        self.basket: list = []
        self.quotes: dict = {}
        self.quote_tick: dict = {}
        self.verified: set = set()
        self.pages: dict = {}
        self.listings: dict = {}
        self.option_text: dict = {}
        self.listing_ids: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.tick = 0
        self.first_add_tick: int | None = None
        self.last_add_tick: int | None = None
        self.add_count = 0
        self.shock_key: str | None = None
        self.dead_keys: set = set()
        self.dead_pages: set = set()
        self.seen_calls: set = set()

    @staticmethod
    def pair(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def ingest(self, name: str, args: dict, obs: dict, purse: Purse) -> None:
        self.tick += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._catch_row(row, purse)
        elif name == "view":
            self._catch_page(obs, purse)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._catch_page(prod, purse)
        elif name == "inspect_stock":
            k = self.pair(obs.get("product_id"), obs.get("sku"))
            self.quotes[k] = {"in_stock": bool(obs.get("in_stock")), "price": _money(obs.get("price"))}
            self.quote_tick[k] = self.tick
            self.verified.add(k)
        elif name == "add_to_cart":
            added = obs.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.add_count += 1
                if (pid, sku) not in self.basket:
                    self.basket.append((pid, sku))
                if self.first_add_tick is None:
                    self.first_add_tick = self.tick
                self.last_add_tick = self.tick
            self._refresh_basket(obs)
        elif name == "remove_from_cart":
            self._refresh_basket(obs)
            removed = obs.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.basket:
                self.basket.remove(pair)
        elif name == "inspect_cart":
            self._refresh_basket(obs)

    def _refresh_basket(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.basket = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.pair(r.get("product_id"), r.get("sku"))
            price = _money(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.quotes.get(k, {})
            self.quotes[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.quote_tick[k] = self.tick

    def _catch_row(self, row: dict, purse: Purse) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if purse.theme:
            if all(t in _tokens(title) for t in purse.theme):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        k = self.pair(pid, row.get("sku"))
        self.listings[k] = row
        if row.get("price") is not None:
            self.quotes.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _money(row.get("price"))})

    def _catch_page(self, prod: dict, purse: Purse) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if purse.theme and all(t in _tokens(str(prod.get("title") or "")) for t in purse.theme):
            self.identities.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.pair(pid, v.get("sku"))
            self.quotes[k] = {"in_stock": bool(v.get("in_stock")), "price": _money(v.get("price"))}
            self.quote_tick[k] = self.tick
            self.option_text[k] = str(v.get("options") or "")

    def quote(self, pid: str, sku: str) -> dict | None:
        return self.quotes.get(self.pair(pid, sku))


class HeadroomRanker:
    """Orders legal tickets by leftover-budget survival, except where a lane forbids it."""

    def __init__(self, shop: "CushionShop") -> None:
        self.shop = shop

    def grade(self, pid: str, sku: str) -> Ticket:
        """Score one variant against every gate the runtime will apply to it."""
        purse = self.shop.purse
        shelf = self.shop.shelf
        fam = purse.family
        key = Shelf.pair(pid, sku)
        st = shelf.quotes.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in shelf.dead_keys or (str(pid) in shelf.dead_pages and fam != "recovery")
        live = bool(stock) and price is not None and (purse.budget is None or price <= purse.budget + _EPS) and not dead
        viewed = str(pid) in shelf.pages
        field, want = purse.spec_field, purse.spec_value
        spec = 2
        if field and want:
            got = self.shop.sku_value(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self.shop.maker_match(pid) and self.shop.device_match(pid, sku))
        cat_ok = self.shop.shelf_match(pid)
        ok = ok and cat_ok
        if fam == "justification":
            ok = ok and not self.shop.resale_title(pid) and bool(self.shop.maker_of(pid))
            if field and want:
                ok = ok and self.shop.labeled_fact(pid, field) == want and self.shop.page_value(pid, field) == want
        if fam == "preference" and shelf.shock_key:
            dead_listing = shelf.listing_of.get(shelf.shock_key.split("::", 1)[0])
            if dead_listing and shelf.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == "recall" and purse.theme:
            ok = ok and all(t in _tokens(self.shop.title_of(pid)) for t in purse.theme)
        if fam == "ranking":
            ok = ok and (str(pid), str(sku)) in purse.candidate_keys
            ok = ok and self.shop.numeric(pid, sku, "refresh_rate") is not None
            ok = ok and self.shop.numeric(pid, sku, "screen_size") is not None
        return Ticket(
            pid=str(pid),
            sku=str(sku),
            price=price,
            live=live,
            viewed=viewed,
            spec=spec,
            cat_ok=cat_ok,
            ok=ok,
            relax=self.shop.drift(pid, sku) if fam == "recovery" else 0,
            budget=purse.budget,
        )

    def pool(self) -> list[Ticket]:
        seen = set()
        out: list[Ticket] = []
        for key in list(self.shop.shelf.quotes):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self.grade(pid, sku))
        return out

    def extras(self, t: Ticket) -> int:
        purse = self.shop.purse
        return sum(
            1
            for f, v in purse.specs.items()
            if f != purse.spec_field and self.shop.sku_value(t.pid, t.sku, f) != v
        )

    def unproven(self, t: Ticket) -> int:
        verdict = _phone_identity(
            self.shop.facts_of(t.pid),
            self.shop.title_of(t.pid),
            self.shop.options_of(t.pid, t.sku),
        )
        return 0 if verdict == "yes" else 1

    def score(self, t: Ticket, first: bool) -> tuple:
        """Default: cushion band, then extras, device-proof, price. Ranking ignores slack."""
        purse = self.shop.purse
        fam = purse.family
        price = t.price if t.price is not None else 1e12
        if fam == "ranking":
            hz = self.shop.numeric(t.pid, t.sku, "refresh_rate") or 0.0
            inch = self.shop.numeric(t.pid, t.sku, "screen_size") or 0.0
            return (-hz, -inch, price, t.key)
        if fam == "recovery":
            return (t.relax, price, t.key)
        extra = self.extras(t)
        proven = self.unproven(t)
        slack = t.slack
        band = 0 if _in_cushion(slack) else 1
        glued = 1 if _glued_to_ceiling(slack) else 0
        if fam == "intent" and first and purse.pref_brand:
            wanted = _tokens(purse.pref_brand)
            near = 0 if (wanted <= _tokens(self.shop.maker_of(t.pid)) or wanted <= _tokens(self.shop.title_of(t.pid))) else 1
            return (near, band, extra, proven, glued, price, t.key)
        return (band, extra, proven, glued, price, t.key)

    def brand_hit(self, t: Ticket) -> bool:
        brand = self.shop.purse.pref_brand
        if not brand:
            return False
        wanted = _tokens(brand)
        return bool(wanted <= _tokens(self.shop.maker_of(t.pid)) or wanted <= _tokens(self.shop.title_of(t.pid)))

    def pick(self) -> Ticket | None:
        """Best fully verified ticket, or nothing while evidence is missing."""
        shop = self.shop
        first = shop.shelf.add_count == 0
        fam = shop.purse.family
        good = [t for t in self.pool() if t.ok]
        if fam == "preference":
            carted = [t for t in good if t.pair() in shop.shelf.basket]
            if carted:
                return carted[0]
            if good and shop.turn >= shop.max_steps - 8:
                good.sort(
                    key=lambda t: (
                        -shop.wish_hits(t.pid, t.sku),
                        t.price if t.price is not None else 1e12,
                    )
                )
                return good[0]
            return None
        if fam == "intent" and first and shop.purse.pref_brand:
            named = [t for t in good if self.brand_hit(t)]
            if named:
                good = named
            elif shop.turn < shop.max_steps * 0.6:
                shop.nudge_brand()
                return None
        if fam == "recovery" and first:
            exact = [t for t in good if t.relax == 0]
            if exact:
                good = exact
            elif shop.turn < shop.max_steps * 0.6:
                shop.nudge_exact()
                return None
        if not good:
            return None
        good.sort(key=lambda t: self.score(t, first))
        return good[0]


class CushionShop:
    """Controller that buys the ticket most likely to survive a first-add reprice."""

    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.purse = Purse(str(self.policy.get("query") or ""))
        self.shelf = Shelf()
        self.ranker = HeadroomRanker(self)
        self.started = time.monotonic()
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self.manifesto()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.shopper_updated = False
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_calls = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.hinted_exact = False
        self.questions_asked = 0
        self.searched_siblings = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self.swept_bands: set = set()
        self.catalog_exhausted = False
        self.searched_twins = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _ROSTER.get(provider) or _ROSTER["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def remaining(self) -> float:
        return _FUSE_S - (time.monotonic() - self.started)

    # ------------------------------------------------------------------ prompts
    def manifesto(self) -> str:
        play = _SURVIVAL_LANES.get(self.purse.family, _SURVIVAL_LANES["generic"])
        extra = ""
        if self.purse.family == "recall" and self.purse.required_identities:
            extra = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the controller reports the count after each search."
                % self.purse.required_identities
            )
        return _SURVIVAL_RULES + "\n\nSurvival lane: " + play + extra

    def first_ask(self) -> str | None:
        if self.purse.family == "constraint" or self.purse.spec_sealed:
            return (
                "Before I sweep price bands: which hardware specification is your firm requirement "
                "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
            )
        if self.purse.family == "preference":
            return (
                "Quick question so I choose a ticket that can survive a first-add reprice: which of "
                "your preferences counts most, and in what order should I weigh the others?"
            )
        return None

    def whisper(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    # ------------------------------------------------------------------ evidence
    def page_of(self, pid: str) -> dict:
        return self.shelf.pages.get(str(pid)) or {}

    def facts_of(self, pid: str) -> dict:
        f = self.page_of(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def title_of(self, pid: str) -> str:
        title = str(self.page_of(pid).get("title") or "")
        if title:
            return title
        for key, row in self.shelf.listings.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def maker_of(self, pid: str) -> str:
        brand = self.page_of(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.shelf.listings.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def path_of(self, pid: str) -> list:
        path = self.page_of(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def options_of(self, pid: str, sku: str) -> str:
        return self.shelf.option_text.get(Shelf.pair(pid, sku), "")

    def drop_codes(self, pid: str, title: str) -> str:
        """Drop model codes from a title so a product name is not read as a panel class."""
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

    def labeled_fact(self, pid: str, field: str) -> str | None:
        raw = self.facts_of(pid).get(_FACT_LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _VALUE_SETS[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return _one(found)

    def page_value(self, pid: str, field: str) -> str | None:
        """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
        vals: set = set()
        conflict = False
        raw = self.facts_of(pid).get(_FACT_LABEL[field])
        if raw:
            found = _VALUE_SETS[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and _QUARTER_RE.search(str(raw)):
                conflict = True
            vals |= found
        title = self.title_of(pid)
        if field == "resolution":
            title = self.drop_codes(pid, title)
            if _QUARTER_RE.search(title):
                conflict = True
        vals |= _adjacent_vals(field, title)
        return None if conflict else _one(vals)

    def sku_value(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        opt = self.options_of(pid, sku)
        vals = _adjacent_vals(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if _mentions(field, opt):
            return None
        return self.page_value(pid, field)

    def numeric(self, pid: str, sku: str, field: str) -> float | None:
        value = self.sku_value(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def shelf_match(self, pid: str) -> bool:
        want = self.purse.category
        path = self.path_of(pid)
        if not want or not path:
            return True
        wanted = self._stems(want)
        labels = set()
        for label in path:
            labels |= self._stems(str(label))
        placed = bool(wanted & labels) or any(
            self._stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        if not placed and "smartphone" in wanted:
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                for label in path
            )
        if not placed:
            return False
        qualifier = self.purse.cat_qualifier
        if not qualifier:
            return True
        title = self.title_of(pid)
        words = self._stems(title) | _tokens(title)
        if qualifier == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in _hz_vals(title))
            return "monitor" in words and ("gaming" in words or fast)
        return "over" in words and "ear" in words or "overear" in words or "circumaural" in words

    def maker_match(self, pid: str) -> bool:
        want = self.purse.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self.maker_of(pid)) or wanted <= _tokens(self.title_of(pid)))

    def device_match(self, pid: str, sku: str) -> bool:
        if not _same_word(self.purse.category or "", "smartphone"):
            return True
        verdict = _phone_identity(self.facts_of(pid), self.title_of(pid), self.options_of(pid, sku))
        if self.purse.family in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def resale_title(self, pid: str) -> bool:
        return bool(re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self.title_of(pid), re.I))

    def drift(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        model, color = self.purse.pref_model, self.purse.pref_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self.title_of(pid) + " " + " ".join(str(v) for v in self.facts_of(pid).values()))
        same_model = squash(model) in blob
        shade = (self.options_of(pid, sku) + " " + str(self.facts_of(pid).get("color") or "") + " " + self.title_of(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def reply_text(self) -> str:
        target = self.ranker.pick() or (self.ranker.pool() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target.pid, target.sku
        parts = ["I am going with " + (self.title_of(pid) or pid)[:90]]
        field, want = self.purse.spec_field, self.purse.spec_value
        if field and want and self.sku_value(pid, sku, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if target.price is not None:
            parts.append("it is %.2f %s and in stock" % (target.price, self.purse.currency))
        return ", ".join(parts) + "."

    # ------------------------------------------------------------------ discovery
    def band_signature(self, name: str, cap: float | None, extra: str) -> str:
        return "%s|%.4f|%s" % (name, cap if cap is not None else -1.0, extra)

    def next_band_query(self) -> tuple | None:
        """Issue the next unused survival-band filter or search."""
        caps: list[float | None] = list(_band_caps(self.purse.budget)) or [None]
        extra = "|".join(
            str(x) for x in (self.purse.category or "", self.purse.brand or "", self.purse.spec_value or "")
        )
        if "filter" in self.tool_names and self.purse.category:
            for cap in caps:
                sig = self.band_signature("filter", cap, extra)
                if sig in self.swept_bands:
                    continue
                self.swept_bands.add(sig)
                args: dict[str, Any] = {"category": self.purse.category, "in_stock": True, "k": 10}
                if cap is not None:
                    args["max_price"] = cap
                if self.purse.brand:
                    args["brand"] = self.purse.brand
                self.whisper(
                    "Sweeping survival band max_price=%s (in_stock, k=10). Later bands are 80 percent "
                    "then the full ceiling if this one is thin." % ("full" if cap is None else "%.2f" % cap)
                )
                return ("filter", args)
        if "search" in self.tool_names:
            words = [w for w in (self.purse.brand, self.purse.spec_value, self.purse.category) if w]
            if not words and self.purse.theme:
                words = list(self.purse.theme)
            if words:
                for cap in caps:
                    sig = self.band_signature("search", cap, extra + "|" + " ".join(words))
                    if sig in self.swept_bands:
                        continue
                    self.swept_bands.add(sig)
                    args = {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10}
                    if cap is not None:
                        args["max_price"] = cap
                    self.whisper(
                        "Search sweep at band max_price=%s so the first cart add can absorb a bump."
                        % ("full" if cap is None else "%.2f" % cap)
                    )
                    return ("search", args)
        return None

    def harvest(self) -> tuple | None:
        """Turn listing rows into pages, then widen through unused price bands."""
        fam = self.purse.family
        if fam == "recall" and len(self.shelf.identities) < self.recall_quota():
            return None
        twin = self.twin_hunt()
        if twin is not None:
            return twin
        leads = self.unread_leads()
        if leads and self.forced_compares < 8 and self.turn >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        if self.turn < 2:
            return None
        band = self.next_band_query()
        if band is not None:
            return band
        self.catalog_exhausted = True
        return None

    def twin_hunt(self) -> tuple | None:
        dead = self.shelf.shock_key
        if self.searched_twins or not dead or "search" not in self.tool_names:
            return None
        title = self.title_of(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self.whisper(
            "The item that died is one of several listings sharing a title; searching that title "
            "surfaces its other colors and sizes, which meet the same requirements."
        )
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def nudge_brand(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self.whisper(
            "No %s product that meets every firm requirement and still has leftover budget has been "
            "read yet. Filter by brand=%s through the survival bands and read those pages before any "
            "other brand." % (self.purse.pref_brand, self.purse.pref_brand)
        )

    def sibling_hunt(self) -> tuple | None:
        if self.searched_siblings:
            return None
        rows = [t for t in self.ranker.pool() if t.live and t.viewed and t.relax == 0]
        if rows:
            return None
        near = [t for t in self.ranker.pool() if t.viewed and t.relax == 1]
        if not near:
            return None
        title = self.title_of(near[0].pid)
        if not title:
            return None
        self.searched_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def nudge_exact(self) -> None:
        if self.hinted_exact or not self.purse.pref_model:
            return
        self.hinted_exact = True
        self.whisper(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its "
            "own and view the results so every color variant is on record; only that exact pairing may "
            "be the first cart item."
            % (self.purse.pref_model, ", " + self.purse.pref_color if self.purse.pref_color else "")
        )

    def unread_leads(self) -> list:
        leads = []
        for key, row in self.shelf.listings.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.shelf.pages or str(pid) in self.shelf.dead_pages:
                continue
            st = self.shelf.quotes.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.purse.budget is not None and price > self.purse.budget + _EPS):
                continue
            title = str(row.get("title") or "")
            if self.purse.theme and not all(t in _tokens(title) for t in self.purse.theme):
                continue
            if self.purse.pref_model and self.shelf.add_count == 0 and not self.shelf.shock_key:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.purse.pref_model.lower()) not in squashed:
                    continue
            if self.purse.brand:
                wanted = _tokens(self.purse.brand)
                if not (wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    # ------------------------------------------------------------------ guidance
    def whisper_shortlist(self) -> None:
        tickets = self.ranker.pool()
        if self.purse.family == "preference" and not self.shelf.basket:
            good = [t for t in tickets if t.ok]
            good.sort(key=lambda t: (-self.wish_hits(t.pid, t.sku), t.price if t.price is not None else 1e12))
            if good:
                self.whisper(
                    "Products that meet every firm requirement so far: "
                    + "; ".join("%s::%s at %.2f (slack %.0f%%)" % (t.pid, t.sku, t.price or 0.0, 100 * t.slack) for t in good[:6])
                    + ". Pick the one that best fits the shopper's stated priorities and can absorb a bump, inspect it, then add it."
                )
                return
        best = self.ranker.pick()
        if best:
            self.whisper(
                "Verified cushioned ticket: %s::%s at %.2f (headroom %.0f%%). The controller will inspect, "
                "cart and order it; read other candidates only if one could survive a reprice better."
                % (best.pid, best.sku, best.price or 0.0, 100 * best.slack)
            )
            return
        field, want = self.purse.spec_field, self.purse.spec_value
        blocked = [t for t in tickets if t.viewed and t.live and not t.ok]
        if blocked and field:
            sample = blocked[:3]
            self.whisper(
                "None of the products read so far can be verified. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(t.pid + "::" + t.sku for t in sample))
            )
        elif blocked:
            self.whisper(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search a different survival band."
            )

    def wish_hits(self, pid: str, sku: str) -> int:
        wants = self.wish_vocab()
        if not wants:
            return 0
        have = _tokens(" ".join(str(v) for v in self.facts_of(pid).values()) + " " + self.title_of(pid) + " " + self.options_of(pid, sku))
        return len(wants & have)

    def wish_vocab(self) -> set:
        if self._wishes is None:
            said = self.purse.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _tokens(self.purse.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def hear_budget(self, text: str) -> None:
        new = self.purse.tighten(text)
        if new is None:
            return
        self.shopper_updated = True
        self.whisper("Budget updated by the shopper: at most %.2f %s. Every later choice must fit it." % (new, self.purse.currency))
        for pid, sku in list(self.shelf.basket):
            st = self.shelf.quote(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.shelf.dead_keys.add(Shelf.pair(pid, sku))
                self.whisper(
                    "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                    % (Shelf.pair(pid, sku), price)
                )

    def claim_sheet(self, pid: str, sku: str) -> list:
        """Claims proven by what the agent has already observed about this exact item."""
        out = []
        field, want = self.purse.spec_field, self.purse.spec_value
        if not field and self.purse.family == "justification":
            field, want = "resolution", self.labeled_fact(pid, "resolution")
        if not field or not want:
            return []
        stated = dict(self.purse.specs) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self.labeled_fact(pid, name) == value and self.page_value(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = self.shelf.quote(pid, sku) or {}
        if self.purse.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.purse.budget})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self.maker_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def order_payload(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.purse.family == "justification":
            claims = self.claim_sheet(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def orderable(self, pid: str, sku: str) -> tuple:
        st = self.shelf.quote(pid, sku)
        k = Shelf.pair(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.shelf.basket:
            return False, "add that exact variant to the cart first"
        if self.shelf.last_add_tick is not None and self.shelf.quote_tick.get(k, 0) <= self.shelf.last_add_tick:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.purse.budget is not None and price is not None and price > self.purse.budget + _EPS:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.purse.budget)
        if k in self.shelf.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    # ------------------------------------------------------------------ steering
    def recall_quota(self) -> int:
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return self.purse.required_identities + (2 if early else 0)

    def may_commit(self) -> bool:
        fam = self.purse.family
        if self.shelf.add_count:
            return True
        if self.purse.spec_sealed:
            if self.turn < self.max_steps * 0.6:
                return False
            self.purse.spec_sealed = False
            self.whisper("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if fam in ("intent", "constraint") and self.turn < 4 and not self.shopper_updated:
            return False
        if fam == "preference" and len(self.shelf.listing_ids) < 2:
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.remaining() > 140
        if fam == "recall" and roomy:
            if len(self.shelf.identities) < self.recall_quota() or self.search_calls < 2:
                return False
        if roomy and not self.catalog_exhausted and len({t.pid for t in self.ranker.pool() if t.ok}) < 2:
            return False
        return True

    def freshly_seen(self, pid: str, sku: str) -> bool:
        k = Shelf.pair(pid, sku)
        if k not in self.shelf.verified:
            return False
        return self.shelf.last_add_tick is None or self.shelf.quote_tick.get(k, 0) > self.shelf.last_add_tick

    def steer(self) -> tuple | None:
        """Forced controller move, or nothing so the model can explore."""
        if self.done:
            return None
        if self.shopper_question_pending:
            if self.question_waits < 2 or "message" not in self.tool_names:
                self.question_waits += 1
                return None
            return ("message", {"content": self.reply_text()})
        fam = self.purse.family
        if fam == "ranking" and "compare" in self.tool_names:
            missing = [p for p, _s in self.purse.candidate_keys if p not in self.shelf.pages and p not in self.unreadable]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if fam == "recovery" and not self.shelf.add_count and self.purse.pref_model and "search" in self.tool_names:
            forced = self.sibling_hunt()
            if forced is not None:
                return forced
        target = self.ranker.pick()
        if target is None or not self.may_commit():
            forced = self.harvest()
            if forced is not None:
                return forced
            if target is None:
                return None
        pid, sku = target.pid, target.sku
        if (pid, sku) in self.shelf.basket:
            if not self.freshly_seen(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self.order_payload(pid, sku))
        if not self.may_commit():
            return None
        for cpid, csku in self.shelf.basket:
            if Shelf.pair(cpid, csku) in self.shelf.dead_keys and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if not self.freshly_seen(pid, sku):
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
            resp = _WIRE.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self.whisper("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            if self.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------------ environment
    def post_env(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            result = _WIRE.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.remaining() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def enact(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self.post_env(pend["envelope"])
            if replay is None:
                self.whisper("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self.digest(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
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
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self.digest(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def digest(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.shelf.seen_calls.discard(name + json.dumps(args, sort_keys=True))
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
            self.done = True
        self.shelf.ingest(name, args, obs, self.purse)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _clip(json.dumps(obs), _OBS_CAP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self.hear_budget(text)
            if self.purse.family in ("constraint", "intent") and not self.purse.spec_value:
                stated = self.purse.unseal(text)
                if stated:
                    self.shopper_updated = True
                    self.whisper("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self.whisper("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    # ------------------------------------------------------------------ controller steps
    def after_cart(self) -> None:
        """Recheck the carted variant. Keep it if a bump still leaves it legal."""
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        prior = self.shelf.quote(pid, sku) or {}
        prior_price = prior.get("price")
        obs = self.enact("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _money(obs.get("price"))
        over = self.purse.budget is not None and price is not None and price > self.purse.budget + _EPS
        bumped = prior_price is not None and price is not None and abs(price - prior_price) > _EPS
        if stock and not over:
            if bumped:
                self.whisper(
                    "Reprice after the cart addition: %s moved to %s but is still under the ceiling and "
                    "in stock. Keeping it; a still-legal ticket is not replaced."
                    % (Shelf.pair(pid, sku), price)
                )
            else:
                self.whisper(
                    "Re-check after the cart addition: %s is in stock at %s, still legal. The controller will order it."
                    % (Shelf.pair(pid, sku), price)
                )
            return
        self.shelf.shock_key = Shelf.pair(pid, sku)
        self.event_turn = self.turn
        self.shelf.dead_keys.add(Shelf.pair(pid, sku))
        self.shelf.dead_pages.add(str(pid))
        self.forced_compares = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, now above the budget" % (price or 0.0)
        guidance = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "The remaining named candidates are re-ranked by hz, then inch, then price; headroom is ignored.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
        }.get(self.purse.family, "Choose a different product that meets every firm requirement and can still absorb a bump.")
        self.whisper("Market change: the carted item %s %s. " % (Shelf.pair(pid, sku), reason) + guidance)
        self.whisper_shortlist()

    def hurry(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for pid, sku in reversed(self.shelf.basket):
            ok, _why = self.orderable(pid, sku)
            if ok and not self.done:
                self.enact("place_test_order", self.order_payload(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [t for t in self.ranker.pool() if t.ok]
        if not good and self.purse.family != "justification":
            good = [
                t for t in self.ranker.pool()
                if t.live and t.spec == 2 and self.maker_match(t.pid) and t.cat_ok
                and (not self.purse.candidate_keys or (t.pid, t.sku) in self.purse.candidate_keys)
            ]
        if not good:
            return
        good.sort(key=lambda t: self.ranker.score(t, self.shelf.add_count == 0))
        pid, sku = good[0].pid, good[0].sku
        if (pid, sku) not in self.shelf.basket:
            obs = self.enact("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = self.enact("inspect_stock", {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = _money(obs.get("price"))
        if self.purse.budget is not None and price is not None and price > self.purse.budget + _EPS:
            return
        self.enact("place_test_order", self.order_payload(pid, sku))

    # ------------------------------------------------------------------ main loop
    def shop(self) -> list:
        question = self.first_ask()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.remaining() < 25 or self.turn >= self.max_steps - 3:
                self.hurry()
                break
            if question and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self.enact("message", {"content": question})
                continue
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self.after_cart()
                continue
            move = self.steer()
            if move is not None:
                name, args = move
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.shelf.dead_keys.add(Shelf.pair(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.shopper_question_pending = False
                    last_forced, repeats = None, 0
                    self.whisper("That step changed nothing, so the controller moved on to other candidates.")
                    continue
                obs = self.enact(name, args)
                if name == "message":
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            proposal = self.consult()
            if proposal is None:
                self.hurry()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _call_args(call)
            content = proposal.get("content") or ""
            reprompts = 0
            while True:
                verdict = self.refuse(name, args)
                if verdict is None:
                    break
                if reprompts >= _RETRY_PROMPTS:
                    self.rejected_streak += 1
                    self.whisper("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self.whisper_shortlist()
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
                args = _call_args(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.shelf.basket:
                    pid, sku = self.shelf.basket[-1]
                args = self.order_payload(pid, sku)
            if name == "message":
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
                self.question_waits = 0
            obs = self.enact(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                self.search_calls += 1
                if self.purse.family == "recall" and self.purse.required_identities:
                    have, need = len(self.shelf.identities), self.recall_quota()
                    self.whisper(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name in ("view", "compare"):
                self.whisper_shortlist()
        return self.dialogue

    def refuse(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.shelf.basket:
                pid, sku = self.shelf.basket[-1]
            ok, why = self.orderable(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.purse.family == "recall" and len(self.shelf.identities) < self.recall_quota():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.shelf.identities), self.recall_quota())
                )
            if self.purse.family == "preference" and len(self.shelf.listing_ids) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            t = self.ranker.grade(pid, sku)
            key = Shelf.pair(pid, sku)
            if not t.live:
                return "that variant is out of stock, over budget, or already invalidated"
            if not t.viewed:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.shelf.verified:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if t.spec != 2 and self.purse.spec_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.purse.spec_field, self.purse.spec_value)
                )
            if not t.ok:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.purse.family == "preference" and len(self.shelf.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.purse.family == "recovery" and not self.shelf.add_count and self.purse.pref_model and self.drift(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if not self.may_commit():
                return (
                    "hold: read one more product that also meets every requirement first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return (
                    "the shopper has already answered; decide from the evidence you have rather than asking again"
                )
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.shelf.seen_calls and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.shelf.seen_calls.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return CushionShop(problem_data).shop()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
