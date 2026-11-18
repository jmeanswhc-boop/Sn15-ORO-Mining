"""ORO-Bench shopping agent — Plan 06: pairwise tournament ranker.

A language model scouts the catalog. A deterministic arena owns the endgame.
Eligible SKUs enter a Bracket as Fighters after their pages are read. Control
is match replay, not list.sort: two Fighters meet in a Match scored on the
task's own keys, the winner stays, and the closer carts only that champion.
When the market kills the champion the dead node is cut and the bracket is
replayed from the remaining roster.
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
_CARD_SECONDS = 270.0
_OBS_CAP = 7000
_REPROMPT_CAP = 2
_SLACK = 1e-9

_FIGURE = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_PURSE_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _FIGURE
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_NAMED_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTE_RE = re.compile(r"\"([^\"]{2,80})\"")
_MOTIF_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_HUE_RE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_SOFT_MAKER_RE = re.compile(
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

# ---------------------------------------------------------------- evidence tape
# The environment gates a specification from public evidence only: labelled raw
# facts, the selected variant's option text, then the listing title. A number
# counts only when its unit or concept word sits next to it. Disagreement
# resolves to nothing. These readers mirror that tape so the arena can predict
# the verdict before it carts a champion.
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QUARTER_RE = re.compile(r"\bqHD\b")

_FACT_TAG = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}


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
    return bool(
        _WXH_RE.search(text)
        or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I)
    )


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


# ------------------------------------------------------------ device identity
# Some categories are graded on what the listing IS. A case, a spare panel or a
# keypad handset are not the device a shopper asked for. The runtime decides
# that from the same public evidence the arena sees.
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

_SCORECARDS = {
    "intent": (
        "Intent scorecard: the category, the named hardware specification, the budget and stock "
        "are firm. A named brand is only a preference — seat that maker when a fully compliant "
        "fighter of that brand exists, otherwise any compliant brand. Filter the shelf, then view "
        "or compare so the variant's raw facts are on the tape. Answer the shopper with message; "
        "a later budget cut applies to every later bout."
    ),
    "recall": (
        "Recall scorecard: run several differently worded searches (k=10, in_stock=true) built "
        "from the requested words and close synonyms until enough distinct matching products have "
        "appeared; every requested word must appear as a whole word of the title. Then view the "
        "strongest so the arena can verify one and buy it."
    ),
    "constraint": (
        "Constraint scorecard: the shopper hides one firm hardware requirement and answers it "
        "when asked; obey it exactly for the specific variant, reading the variant's own option "
        "text before any product-wide value. Filter and view. If the carted fighter sells out, "
        "the bracket replays without that node."
    ),
    "preference": (
        "Preference scorecard: you already asked which preference matters most; use the answer. "
        "Search or filter the category within budget, then compare fighters with different "
        "listing_id values so two booths are on the tape before the first add. The first cart "
        "item is graded. If it is repriced above budget, replay without that listing_id."
    ),
    "ranking": (
        "Ranking scorecard: only the named candidate variants count. Compare them in batches of "
        "five so their facts are on the tape. Highest advertised maximum refresh rate wins the "
        "bout, then the largest screen, then the lowest price, reading the selected variant's "
        "own value before a product-wide one. After a market death the remaining named fighters "
        "replay the bracket."
    ),
    "recovery": (
        "Recovery scorecard: find the preferred model and color first. Search the model code, "
        "then view it so every variant and color is on the tape. The first cart item must be "
        "exactly that model and color. If it disappears, the next bout favours the smallest "
        "relax — same model, other color — before any other model. Never relax category, budget "
        "or stock."
    ),
    "justification": (
        "Justification scorecard: find a product whose own raw facts state the requested "
        "specification, in the requested category and brand, in stock and within budget. View "
        "or compare it so its facts are on the tape; the arena then inspects, carts and orders "
        "with claims taken only from that evidence."
    ),
    "generic": (
        "Generic scorecard: understand the firm requirements, filter or search the catalog, "
        "view candidates, and let the arena verify and buy the bout winner."
    ),
}

_RING_RULES = (
    "You are a ringside clerk working a catalog tournament through tools. Exactly one tool call "
    "per turn, at most 30 turns, so keep the card moving. A deterministic arena checks every "
    "proposal and takes over the cart and the order once a fighter is fully verified, so your "
    "job is to surface and read pages: search or filter, then view or compare so raw facts, "
    "variant options and prices are on the scorecard. Never add an item to the cart as a "
    "placeholder — the first cart addition is graded. Prices and stock change while you shop; "
    "the arena re-checks the carted champion and tells you when that node has died and the "
    "bracket will replay. Answer every direct question from the shopper with one short message "
    "before anything else, and never send progress or status updates. Never ask the shopper to "
    "raise the budget. Do not repeat a tool call with identical arguments; one compare of up to "
    "five products beats several view calls."
)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _call_args(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _clip_obs(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


def _purse_from(text: str) -> float | None:
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


def _shelf_from(goal: str) -> str | None:
    m = _SHELF_RE.search(goal or "")
    if m:
        return m.group(1).strip()
    m = re.search(
        r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
        goal or "",
        re.I,
    )
    return m.group(1) if m else None


def _maker_from(goal: str, shelf: str | None) -> str | None:
    if not shelf:
        return None
    m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(shelf), goal or "", re.I)
    if not m:
        return None
    words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_NOISE_RE.sub(" ", m.group(1))) if w]
    keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _res_vals(w)]
    return " ".join(keep) or None


def _soft_maker_from(goal: str) -> str | None:
    m = _SOFT_MAKER_RE.search(goal or "")
    if not m:
        return None
    name = m.group(1).strip(" .,")
    words = name.split()
    if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
        return None
    return name


def _shape_from(goal: str) -> str | None:
    if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
        return "gaming"
    if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
        return "over ear"
    return None


def _gates_from(goal: str) -> dict:
    out = {}
    for field in ("screen_size", "refresh_rate", "resolution", "ram"):
        value = _wanted_value(field, goal)
        if value:
            out[field] = value
    return out


class ShotClock:
    def __init__(self, seconds: float) -> None:
        self.opened = time.monotonic()
        self.limit = seconds

    def left(self) -> float:
        return self.limit - (time.monotonic() - self.opened)


class QueryCard:
    """Public request, parsed into the fields the arena scores bouts against."""

    def __init__(self, query: str) -> None:
        self.source = query or ""
        head, _, footer = self.source.partition("\n\nTask rules:")
        self.ask = head.strip()
        self.rule_blob = footer.lower()
        self.purse = _purse_from(self.ask)
        self.money = "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.ask)
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.money = m.group(1)
        self.named_pairs = [(pid, sku.rstrip(".,;:")) for pid, sku in _NAMED_RE.findall(self.ask)]
        motif = _MOTIF_RE.search(self.ask)
        quoted = _QUOTE_RE.findall(self.ask)
        seed = motif.group(1) if motif else (quoted[0] if quoted else "")
        self.motif = re.findall(r"[a-z0-9]+", seed.lower())
        sm = _SURFACE_RE.search(self.rule_blob)
        self.surface_need = int(sm.group(1)) if sm else 0
        hm = _HUE_RE.search(self.ask)
        self.hue = hm.group(1).lower() if hm else None
        self.code = hm.group(2) if hm else None
        self.shelf = _shelf_from(self.ask)
        self.maker = _maker_from(self.ask, self.shelf)
        self.soft_maker = None if hm else _soft_maker_from(self.ask)
        self.shape = _shape_from(self.ask)
        self.gates = _gates_from(self.ask)
        self.gate_name, self.gate_mark = self._first_gate()
        self.hidden_gate = self.gate_name is None and "firm requirement" in self.ask.lower()
        self.lane = self._lane()

    def _first_gate(self) -> tuple:
        for field, value in self.gates.items():
            return field, value
        return None, None

    def _lane(self) -> str:
        if self.named_pairs:
            return "ranking"
        if self.surface_need:
            return "recall"
        if self.code:
            return "recovery"
        for name, sign in _LANE_MARKS:
            if sign in self.rule_blob:
                return name
        if self.motif:
            return "recall"
        if _WHY_RE.search(self.ask):
            return "justification"
        if self.soft_maker:
            return "intent"
        if self.hidden_gate or self.gate_name or self.maker:
            return "constraint"
        return "preference"

    def take_hidden(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = _wanted_value(field, text)
            if value:
                self.gate_name, self.gate_mark = field, value
                self.gates[field] = value
                self.hidden_gate = False
                return field + " = " + value
        return None


class RingSheet:
    """Observed tape, keyed by product and variant. Not a ranking structure."""

    def __init__(self) -> None:
        self.basket: list = []
        self.quotes: dict = {}
        self.quote_tick: dict = {}
        self.weighed: set = set()
        self.dossiers: dict = {}
        self.slips: dict = {}
        self.option_ink: dict = {}
        self.booths: set = set()
        self.booth_of: dict = {}
        self.faces: set = set()
        self.tick = 0
        self.first_add_tick: int | None = None
        self.last_add_tick: int | None = None
        self.adds = 0
        self.fallen: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.echo: set = set()

    @staticmethod
    def mark(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def ingest(self, name: str, args: dict, obs: dict, card: QueryCard) -> None:
        self.tick += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._tag_slip(row, card)
        elif name == "view":
            self._tag_dossier(obs, card)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._tag_dossier(prod, card)
        elif name == "inspect_stock":
            k = self.mark(obs.get("product_id"), obs.get("sku"))
            self.quotes[k] = {"in_stock": bool(obs.get("in_stock")), "price": _as_float(obs.get("price"))}
            self.quote_tick[k] = self.tick
            self.weighed.add(k)
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
            self._sync_basket(obs)
        elif name == "remove_from_cart":
            self._sync_basket(obs)
            removed = obs.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.basket:
                self.basket.remove(pair)
        elif name == "inspect_cart":
            self._sync_basket(obs)

    def _sync_basket(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.basket = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.mark(r.get("product_id"), r.get("sku"))
            price = _as_float(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.quotes.get(k, {})
            self.quotes[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.quote_tick[k] = self.tick

    def _tag_slip(self, row: dict, card: QueryCard) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if card.motif:
            if all(t in _tokens(title) for t in card.motif):
                self.faces.add(str(pid))
        else:
            self.faces.add(str(pid))
        if not row.get("sku"):
            return
        k = self.mark(pid, row.get("sku"))
        self.slips[k] = row
        if row.get("price") is not None:
            self.quotes.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _as_float(row.get("price"))})

    def _tag_dossier(self, prod: dict, card: QueryCard) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.dossiers[str(pid)] = prod
        if prod.get("listing_id"):
            self.booths.add(str(prod["listing_id"]))
            self.booth_of[str(pid)] = str(prod["listing_id"])
        if card.motif and all(t in _tokens(str(prod.get("title") or "")) for t in card.motif):
            self.faces.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.mark(pid, v.get("sku"))
            self.quotes[k] = {"in_stock": bool(v.get("in_stock")), "price": _as_float(v.get("price"))}
            self.quote_tick[k] = self.tick
            self.option_ink[k] = str(v.get("options") or "")

    def quote_of(self, pid: str, sku: str) -> dict | None:
        return self.quotes.get(self.mark(pid, sku))


class Fighter:
    """One SKU seated in the bracket. Holds the tape the next Match will read."""

    __slots__ = (
        "pid", "sku", "key", "price", "live", "viewed", "spec", "cat_ok", "ok",
        "relax", "hz", "inch", "hits", "proof", "booth",
    )

    def __init__(self) -> None:
        self.pid = ""
        self.sku = ""
        self.key = ""
        self.price: float | None = None
        self.live = False
        self.viewed = False
        self.spec = 2
        self.cat_ok = True
        self.ok = False
        self.relax = 0
        self.hz: float | None = None
        self.inch: float | None = None
        self.hits = 0
        self.proof = 0
        self.booth = ""

    def purse_or_high(self) -> float:
        return self.price if self.price is not None else 1e12


class Match:
    """One pairwise bout. The scorecard is the task lane, never a global sort."""

    def __init__(self, left: Fighter, right: Fighter, arena: "ArenaShop") -> None:
        self.left = left
        self.right = right
        self.arena = arena
        self.reason = ""

    def winner(self) -> Fighter:
        lane = self.arena.card.lane
        if lane == "ranking":
            return self._panel_bout()
        if lane == "recovery":
            return self._relax_bout()
        return self._hit_bout()

    def _panel_bout(self) -> Fighter:
        a, b = self.left, self.right
        if a.hz is None or a.inch is None:
            self.reason = "left missing panel tape"
            return b
        if b.hz is None or b.inch is None:
            self.reason = "right missing panel tape"
            return a
        if a.hz != b.hz:
            self.reason = "higher refresh"
            return a if a.hz > b.hz else b
        if a.inch != b.inch:
            self.reason = "larger screen"
            return a if a.inch > b.inch else b
        if a.purse_or_high() != b.purse_or_high():
            self.reason = "lower price"
            return a if a.purse_or_high() < b.purse_or_high() else b
        self.reason = "key tiebreak"
        return a if a.key < b.key else b

    def _relax_bout(self) -> Fighter:
        a, b = self.left, self.right
        if a.relax != b.relax:
            self.reason = "smaller relax"
            return a if a.relax < b.relax else b
        if a.purse_or_high() != b.purse_or_high():
            self.reason = "lower price after relax"
            return a if a.purse_or_high() < b.purse_or_high() else b
        self.reason = "key tiebreak"
        return a if a.key < b.key else b

    def _hit_bout(self) -> Fighter:
        a, b = self.left, self.right
        if a.hits != b.hits:
            self.reason = "more spec hits"
            return a if a.hits > b.hits else b
        if a.proof != b.proof:
            self.reason = "device-proof"
            return a if a.proof > b.proof else b
        if a.purse_or_high() != b.purse_or_high():
            self.reason = "lower price"
            return a if a.purse_or_high() < b.purse_or_high() else b
        self.reason = "key tiebreak"
        return a if a.key < b.key else b


class Bracket:
    """Seated fighters. Champion is decided by replaying Matches, not sorting."""

    def __init__(self, arena: "ArenaShop") -> None:
        self.arena = arena
        self.nodes: dict[str, Fighter] = {}
        self.banned_booths: set[str] = set()
        self.tape: list[str] = []
        self.holder: Fighter | None = None

    def empty(self) -> bool:
        return not self.nodes

    def clear(self) -> None:
        self.nodes = {}
        self.holder = None

    def admit(self, fighter: Fighter) -> bool:
        if not fighter.ok:
            return False
        if fighter.key in self.arena.sheet.dead_keys:
            return False
        if fighter.booth and fighter.booth in self.banned_booths:
            return False
        self.nodes[fighter.key] = fighter
        return True

    def exile(self, key: str) -> None:
        self.nodes.pop(key, None)
        if self.holder is not None and self.holder.key == key:
            self.holder = None

    def ban_booth(self, listing_id: str | None) -> None:
        if not listing_id:
            return
        self.banned_booths.add(listing_id)
        drop = [k for k, f in self.nodes.items() if f.booth == listing_id]
        for k in drop:
            self.nodes.pop(k, None)
        if self.holder is not None and self.holder.booth == listing_id:
            self.holder = None

    def after_death(self, key: str, listing_id: str | None, preference: bool) -> None:
        """Cut the dead node. Preference also bans that listing_id, then replay."""
        self.exile(key)
        if preference:
            self.ban_booth(listing_id)
        self.replay()

    def restock(self, fighters: list[Fighter]) -> None:
        self.nodes = {}
        for fighter in fighters:
            self.admit(fighter)

    def replay(self) -> Fighter | None:
        """King-of-the-hill: seat the first live node, then each challenger plays a Match."""
        alive: list[Fighter] = []
        for fighter in self.nodes.values():
            if fighter.key in self.arena.sheet.dead_keys:
                continue
            if fighter.booth and fighter.booth in self.banned_booths:
                continue
            if not fighter.ok:
                continue
            alive.append(fighter)
        if not alive:
            self.holder = None
            return None
        champ = alive[0]
        walk = 1
        while walk < len(alive):
            bout = Match(champ, alive[walk], self.arena)
            nxt = bout.winner()
            self.tape.append(
                "bout %s vs %s -> %s (%s)" % (champ.key, alive[walk].key, nxt.key, bout.reason)
            )
            champ = nxt
            walk += 1
        self.holder = champ
        return champ


class ArenaShop:
    """Ringside controller: scouts with the model, closes with the bracket winner."""

    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.kit = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.card = QueryCard(str(self.policy.get("query") or ""))
        self.purse = self.card.purse
        self.sheet = RingSheet()
        self.clock = ShotClock(_CARD_SECONDS)
        self.bracket = Bracket(self)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self._scorecard_prompt()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.owes_reply = False
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
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.fog: set = set()
        self.reply_stalls = 0
        self.closed = False
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _ROSTER.get(provider) or _ROSTER["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _scorecard_prompt(self) -> str:
        play = _SCORECARDS.get(self.card.lane, _SCORECARDS["generic"])
        extra = ""
        if self.card.lane == "recall" and self.card.surface_need:
            extra = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the arena reports the count after each search."
                % self.card.surface_need
            )
        return _RING_RULES + "\n\nFamily scorecard: " + play + extra

    def _first_ask(self) -> str | None:
        if self.card.lane == "constraint" or self.card.hidden_gate:
            return (
                "Before I search: which hardware specification is your firm requirement "
                "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
            )
        if self.card.lane == "preference":
            return (
                "Quick question so I choose well: which of your preferences counts most, and in what "
                "order should I weigh the others? Let me know your priority."
            )
        return None

    def _ring_note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _page(self, pid: str) -> dict:
        return self.sheet.dossiers.get(str(pid)) or {}

    def _facts(self, pid: str) -> dict:
        f = self._page(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def _banner(self, pid: str) -> str:
        title = str(self._page(pid).get("title") or "")
        if title:
            return title
        for key, row in self.sheet.slips.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def _maker_of(self, pid: str) -> str:
        brand = self._page(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.sheet.slips.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def _shelf_walk(self, pid: str) -> list:
        path = self._page(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def _option_ink(self, pid: str, sku: str) -> str:
        return self.sheet.option_ink.get(RingSheet.mark(pid, sku), "")

    def _peel_codes(self, pid: str, title: str) -> str:
        """Drop model codes from a title so a product name is not read as a panel class."""
        for token in re.split(r"[;,]", str(self._facts(pid).get("model") or "")):
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

    def _labelled(self, pid: str, field: str) -> str | None:
        """The value the product's own labelled facts state for a concept."""
        raw = self._facts(pid).get(_FACT_TAG[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _VALUE_SETS[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return _one(found)

    def _catalog_mark(self, pid: str, field: str) -> str | None:
        """Product-wide value: labelled facts plus the title, nothing when they disagree."""
        vals: set = set()
        conflict = False
        raw = self._facts(pid).get(_FACT_TAG[field])
        if raw:
            found = _VALUE_SETS[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and _QUARTER_RE.search(str(raw)):
                conflict = True
            vals |= found
        title = self._banner(pid)
        if field == "resolution":
            title = self._peel_codes(pid, title)
            if _QUARTER_RE.search(title):
                conflict = True
        vals |= _adjacent_vals(field, title)
        return None if conflict else _one(vals)

    def _sku_mark(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        opt = self._option_ink(pid, sku)
        vals = _adjacent_vals(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if _mentions(field, opt):
            return None
        return self._catalog_mark(pid, field)

    def _tape_measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self._sku_mark(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def _shelf_ok(self, pid: str) -> bool:
        want = self.card.shelf
        path = self._shelf_walk(pid)
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
        qualifier = self.card.shape
        if not qualifier:
            return True
        title = self._banner(pid)
        words = self._stems(title) | _tokens(title)
        if qualifier == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in _hz_vals(title))
            return "monitor" in words and ("gaming" in words or fast)
        return "over" in words and "ear" in words or "overear" in words or "circumaural" in words

    def _maker_ok(self, pid: str) -> bool:
        want = self.card.maker
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._maker_of(pid)) or wanted <= _tokens(self._banner(pid)))

    def _device_ok(self, pid: str, sku: str) -> bool:
        if not _same_word(self.card.shelf or "", "smartphone"):
            return True
        verdict = _phone_identity(self._facts(pid), self._banner(pid), self._option_ink(pid, sku))
        if self.card.lane in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def _resale_banner(self, pid: str) -> bool:
        return bool(
            re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self._banner(pid), re.I)
        )

    def _relax_of(self, pid: str, sku: str) -> int:
        model, color = self.card.code, self.card.hue
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self._banner(pid) + " " + " ".join(str(v) for v in self._facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self._option_ink(pid, sku) + " " + str(self._facts(pid).get("color") or "") + " " + self._banner(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _hits_of(self, pid: str, sku: str) -> int:
        n = 0
        for field, want in self.card.gates.items():
            if self._sku_mark(pid, sku, field) == want:
                n += 1
        return n

    def _proof_of(self, pid: str, sku: str) -> int:
        """Phone yes outranks unclear; unclear outranks no. Used only as a bout key."""
        verdict = _phone_identity(self._facts(pid), self._banner(pid), self._option_ink(pid, sku))
        if verdict == "yes":
            return 2
        if verdict == "unclear":
            return 1
        return 0

    def _weigh(self, pid: str, sku: str) -> Fighter:
        """Score one variant against every gate the runtime will apply to it."""
        fam = self.card.lane
        key = RingSheet.mark(pid, sku)
        st = self.sheet.quotes.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.sheet.dead_keys or (str(pid) in self.sheet.dead_products and fam != "recovery")
        live = bool(stock) and price is not None and (self.purse is None or price <= self.purse + _SLACK) and not dead
        viewed = str(pid) in self.sheet.dossiers
        field, want = self.card.gate_name, self.card.gate_mark
        spec = 2
        if field and want:
            got = self._sku_mark(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self._maker_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._shelf_ok(pid)
        ok = ok and cat_ok
        if fam == "justification":
            ok = ok and not self._resale_banner(pid) and bool(self._maker_of(pid))
            if field and want:
                ok = ok and self._labelled(pid, field) == want and self._catalog_mark(pid, field) == want
        if fam == "preference" and self.sheet.fallen:
            dead_listing = self.sheet.booth_of.get(self.sheet.fallen.split("::", 1)[0])
            if dead_listing and self.sheet.booth_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == "recall" and self.card.motif:
            ok = ok and all(t in _tokens(self._banner(pid)) for t in self.card.motif)
        hz = self._tape_measure(pid, sku, "refresh_rate")
        inch = self._tape_measure(pid, sku, "screen_size")
        if fam == "ranking":
            ok = ok and (str(pid), str(sku)) in self.card.named_pairs
            ok = ok and hz is not None and inch is not None
        node = Fighter()
        node.pid = str(pid)
        node.sku = str(sku)
        node.key = key
        node.price = price
        node.live = live
        node.viewed = viewed
        node.spec = spec
        node.cat_ok = cat_ok
        node.ok = ok
        node.relax = self._relax_of(pid, sku) if fam == "recovery" else 0
        node.hz = hz
        node.inch = inch
        node.hits = self._hits_of(pid, sku)
        node.proof = self._proof_of(pid, sku)
        node.booth = self.sheet.booth_of.get(str(pid), "")
        return node

    def _walk_roster(self) -> list[Fighter]:
        seen = set()
        out = []
        for key in list(self.sheet.quotes):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._weigh(pid, sku))
        return out

    def _seat_fighters(self) -> list[Fighter]:
        """Who may enter the bracket this tick. First-add lanes keep a short bench."""
        first = self.sheet.adds == 0
        fam = self.card.lane
        good = [f for f in self._walk_roster() if f.ok]
        if fam == "intent" and first and self.card.soft_maker:
            wanted = _tokens(self.card.soft_maker)
            named = [
                f for f in good
                if wanted <= _tokens(self._maker_of(f.pid)) or wanted <= _tokens(self._banner(f.pid))
            ]
            if named:
                good = named
            elif self.turn < self.max_steps * 0.6:
                self._brand_nudge()
                return []
        if fam == "recovery" and first:
            exact = [f for f in good if f.relax == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                self._exact_nudge()
                return []
        return good

    def _refill_and_replay(self) -> Fighter | None:
        seated = self._seat_fighters()
        self.bracket.restock(seated)
        return self.bracket.replay()

    def _champion(self) -> Fighter | None:
        return self._refill_and_replay()

    def _ringside_answer(self) -> str:
        champ = self._champion()
        if champ is None:
            walk = self._walk_roster()
            champ = walk[0] if walk else None
        if not champ:
            return "I am still checking the options against your requirements; nothing is decided yet."
        parts = ["I am going with " + (self._banner(champ.pid) or champ.pid)[:90]]
        field, want = self.card.gate_name, self.card.gate_mark
        if field and want and self._sku_mark(champ.pid, champ.sku, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if champ.price is not None:
            parts.append("it is %.2f %s and in stock" % (champ.price, self.card.money))
        return ", ".join(parts) + "."

    def _surface_need(self) -> int:
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return self.card.surface_need + (2 if early else 0)

    def _fresh_leads(self) -> list:
        leads = []
        for key, row in self.sheet.slips.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.sheet.dossiers or str(pid) in self.sheet.dead_products:
                continue
            st = self.sheet.quotes.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.purse is not None and price > self.purse + _SLACK):
                continue
            title = str(row.get("title") or "")
            if self.card.motif and not all(t in _tokens(title) for t in self.card.motif):
                continue
            if self.card.code and self.sheet.adds == 0 and not self.sheet.fallen:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.card.code.lower()) not in squashed:
                    continue
            if self.card.maker:
                wanted = _tokens(self.card.maker)
                if not (wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title)):
                    continue
            leads.append((price, str(pid)))
        priced = {}
        for price, pid in leads:
            if pid not in priced or price < priced[pid]:
                priced[pid] = price
        ordered = []
        while priced:
            best_pid = None
            best_price = None
            for pid, price in priced.items():
                if best_pid is None or price < best_price or (price == best_price and pid < best_pid):
                    best_pid = pid
                    best_price = price
            ordered.append(best_pid)
            priced.pop(best_pid)
        return ordered

    def _twin_search(self) -> tuple | None:
        dead = self.sheet.fallen
        if self.searched_twins or not dead or "search" not in self.kit:
            return None
        title = self._banner(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self._ring_note(
            "The fallen champion shares a title with sibling listings; searching that title "
            "surfaces other colors and sizes that still meet the same scorecard."
        )
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _hue_search(self) -> tuple | None:
        if self.searched_siblings:
            return None
        rows = [f for f in self._walk_roster() if f.live and f.viewed and f.relax == 0]
        if rows:
            return None
        near = [f for f in self._walk_roster() if f.viewed and f.relax == 1]
        if not near:
            return None
        title = self._banner(near[0].pid)
        if not title:
            return None
        self.searched_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _brand_nudge(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._ring_note(
            "No %s product that meets every firm requirement has been read yet. The shopper's first "
            "choice is graded on that brand, so filter by brand=%s within the budget and read those "
            "products before any other." % (self.card.soft_maker, self.card.soft_maker)
        )

    def _exact_nudge(self) -> None:
        if self.hinted_exact or not self.card.code:
            return
        self.hinted_exact = True
        self._ring_note(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
            "and view the results so every color variant is on record; only that exact pairing may be the "
            "first cart item." % (self.card.code, ", " + self.card.hue if self.card.hue else "")
        )

    def _scout_more(self) -> tuple | None:
        fam = self.card.lane
        if fam == "recall" and len(self.sheet.faces) < self._surface_need():
            return None
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._fresh_leads()
        if leads and self.forced_compares < 8 and self.turn >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.kit:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        probe = (self.card.maker or "", "%.2f" % (self.purse or 0.0), str(self.card.gate_mark))
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
        if "filter" in self.kit and self.card.shelf:
            args = {"category": self.card.shelf, "in_stock": True, "k": 10}
            if self.purse is not None:
                args["max_price"] = self.purse
            if self.card.maker:
                args["brand"] = self.card.maker
            return ("filter", args)
        if "search" in self.kit:
            words = [w for w in (self.card.maker, self.card.gate_mark, self.card.shelf) if w]
            if words:
                return ("search", {"query": " ".join(words), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def _scheduled_compare(self) -> tuple | None:
        """Compare is the scheduled opener for ranking and for preference booths."""
        if "compare" not in self.kit:
            return None
        if self.card.lane == "ranking":
            missing = [
                p for p, _s in self.card.named_pairs
                if p not in self.sheet.dossiers and p not in self.fog
            ]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if self.card.lane == "preference" and self.sheet.adds == 0 and len(self.sheet.booths) < 2:
            leads = self._fresh_leads()
            unread = [p for p in leads if p not in self.sheet.dossiers][:5]
            if len(unread) >= 2:
                return ("compare", {"product_ids": unread[:5]})
            if unread and self.sheet.dossiers:
                known = [p for p in self.sheet.dossiers if p not in unread]
                pack = (known[:1] + unread)[:5]
                if len(pack) >= 2:
                    return ("compare", {"product_ids": pack})
            if unread:
                return ("view", {"product_id": unread[0]})
        return None

    def _wish_words(self) -> set:
        if self._wishes is None:
            said = self.card.ask + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _tokens(self.card.shelf or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def _priority_hits(self, pid: str, sku: str) -> int:
        wants = self._wish_words()
        if not wants:
            return 0
        have = _tokens(
            " ".join(str(v) for v in self._facts(pid).values())
            + " "
            + self._banner(pid)
            + " "
            + self._option_ink(pid, sku)
        )
        return len(wants & have)

    def _scorecard_note(self) -> None:
        champ = self._champion()
        if champ:
            self._ring_note(
                "Scorecard champion: %s at %.2f. The arena will inspect, cart and order this "
                "winner; read other pages only if a challenger could take the bout."
                % (champ.key, champ.price or 0.0)
            )
            return
        field, want = self.card.gate_name, self.card.gate_mark
        blocked = [f for f in self._walk_roster() if f.viewed and f.live and not f.ok]
        if blocked and field:
            sample = blocked[:3]
            self._ring_note(
                "No fighter on the tape can be seated. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw "
                "facts: %s. Look at different products."
                % (field, want, ", ".join(f.key for f in sample))
            )
        elif blocked:
            self._ring_note(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def _purse_from_message(self, text: str) -> None:
        new = _purse_from(text)
        if new is None or (self.purse is not None and new >= self.purse):
            return
        self.purse = new
        self.shopper_updated = True
        self._ring_note(
            "Budget updated by the shopper: at most %.2f %s. Every later bout must fit it."
            % (new, self.card.money)
        )
        for pid, sku in list(self.sheet.basket):
            st = self.sheet.quote_of(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _SLACK:
                self.sheet.dead_keys.add(RingSheet.mark(pid, sku))
                self._ring_note(
                    "The carted item %s costs %.2f, above the new budget; the arena will replay."
                    % (RingSheet.mark(pid, sku), price)
                )

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims proven by what the arena has already observed about this exact item."""
        out = []
        field, want = self.card.gate_name, self.card.gate_mark
        if not field and self.card.lane == "justification":
            field, want = "resolution", self._labelled(pid, "resolution")
        if not field or not want:
            return []
        stated = dict(self.card.gates) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self._labelled(pid, name) == value and self._catalog_mark(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = self.sheet.quote_of(pid, sku) or {}
        if self.purse is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.purse})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self._maker_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.card.lane == "justification":
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.sheet.quote_of(pid, sku)
        k = RingSheet.mark(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.sheet.basket:
            return False, "add that exact variant to the cart first"
        if self.sheet.last_add_tick is not None and self.sheet.quote_tick.get(k, 0) <= self.sheet.last_add_tick:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.purse is not None and price is not None and price > self.purse + _SLACK:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.purse)
        if k in self.sheet.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    def _gate_open(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        fam = self.card.lane
        if self.sheet.adds:
            return True
        if self.card.hidden_gate:
            if self.turn < self.max_steps * 0.6:
                return False
            self.card.hidden_gate = False
            self._ring_note("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if fam in ("intent", "constraint") and self.turn < 4 and not self.shopper_updated:
            return False
        if fam == "preference" and len(self.sheet.booths) < 2:
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.left() > 140
        if fam == "recall" and roomy:
            if len(self.sheet.faces) < self._surface_need() or self.search_calls < 2:
                return False
        if roomy and not self.reads_done and len({f.pid for f in self._walk_roster() if f.ok}) < 2:
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = RingSheet.mark(pid, sku)
        if k not in self.sheet.weighed:
            return False
        return self.sheet.last_add_tick is None or self.sheet.quote_tick.get(k, 0) > self.sheet.last_add_tick

    def _next_bout(self) -> tuple | None:
        """The action the arena takes itself, or nothing so the model can scout."""
        if self.closed:
            return None
        if self.owes_reply:
            if self.reply_stalls < 2 or "message" not in self.kit:
                self.reply_stalls += 1
                return None
            return ("message", {"content": self._ringside_answer()})
        scheduled = self._scheduled_compare()
        if scheduled is not None:
            return scheduled
        if self.card.lane == "recovery" and not self.sheet.adds and self.card.code and "search" in self.kit:
            forced = self._hue_search()
            if forced is not None:
                return forced
        champ = self._champion()
        if champ is None or not self._gate_open():
            forced = self._scout_more()
            if forced is not None:
                return forced
            if champ is None:
                return None
        pid, sku = champ.pid, champ.sku
        if (pid, sku) in self.sheet.basket:
            if not self._fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self._order_args(pid, sku))
        if not self._gate_open():
            return None
        for cpid, csku in self.sheet.basket:
            if RingSheet.mark(cpid, csku) in self.sheet.dead_keys and "remove_from_cart" in self.kit:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if not self._fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    def _ask_model(self) -> dict | None:
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
                self._ring_note("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            if self.clock.left() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            result = _WIRE.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.clock.left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                self._ring_note("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.closed:
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
        result = self._send(envelope)
        if result is None:
            self.pending_envelope = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.sheet.echo.discard(name + json.dumps(args, sort_keys=True))
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
            self.closed = True
        self.sheet.ingest(name, args, obs, self.card)
        if name in ("view", "compare"):
            self._refill_and_replay()
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _clip_obs(json.dumps(obs), _OBS_CAP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._purse_from_message(text)
            if self.card.lane in ("constraint", "intent") and not self.card.gate_mark:
                stated = self.card.take_hidden(text)
                if stated:
                    self.shopper_updated = True
                    self._ring_note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            if _ASK_RE.search(text):
                self.owes_reply = True
                self._ring_note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.closed = True
        return obs

    def _post_add_recheck(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _as_float(obs.get("price"))
        over = self.purse is not None and price is not None and price > self.purse + _SLACK
        if stock and not over:
            self._ring_note(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The arena will order it."
                % (RingSheet.mark(pid, sku), price)
            )
            return
        key = RingSheet.mark(pid, sku)
        self.sheet.fallen = key
        self.event_turn = self.turn
        self.sheet.dead_keys.add(key)
        self.sheet.dead_products.add(str(pid))
        self.forced_compares = 0
        listing_id = self.sheet.booth_of.get(str(pid))
        self.bracket.after_death(key, listing_id, self.card.lane == "preference")
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        guidance = {
            "recovery": "Replay favours the smallest relax: the same model in another color before any other model.",
            "ranking": "The dead named fighter is cut; remaining named candidates replay the bracket.",
            "preference": "Replay excludes that listing_id; a different booth must win the next bout.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
        }.get(self.card.lane, "Replay the bracket without the dead node.")
        self._ring_note("Market change: the carted item %s %s. " % (key, reason) + guidance)
        self._scorecard_note()

    def _panic_pick(self) -> Fighter | None:
        """Late panic: if the bracket is empty, any still-legal live+viewed item."""
        champ = self._champion()
        if champ is not None:
            return champ
        legal: list[Fighter] = []
        for f in self._walk_roster():
            if not (f.live and f.viewed):
                continue
            if f.key in self.sheet.dead_keys:
                continue
            if self.card.named_pairs and (f.pid, f.sku) not in self.card.named_pairs:
                continue
            if f.booth and f.booth in self.bracket.banned_booths:
                continue
            legal.append(f)
        if not legal:
            return None
        pick = legal[0]
        walk = 1
        while walk < len(legal):
            pick = Match(pick, legal[walk], self).winner()
            walk += 1
        return pick

    def _close_fast(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for pid, sku in reversed(self.sheet.basket):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.closed:
                self._act("place_test_order", self._order_args(pid, sku))
                return
        if self.closed or self.turn > self.max_steps - 3:
            return
        champ = self._panic_pick()
        if champ is None:
            return
        pid, sku = champ.pid, champ.sku
        if (pid, sku) not in self.sheet.basket:
            obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.closed or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if self.closed or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = _as_float(obs.get("price"))
        if self.purse is not None and price is not None and price > self.purse + _SLACK:
            return
        self._act("place_test_order", self._order_args(pid, sku))

    def run_card(self) -> list:
        question = self._first_ask()
        last_forced, repeats = None, 0
        while not self.closed and self.turn < self.max_steps:
            if self.clock.left() < 25 or self.turn >= self.max_steps - 3:
                self._close_fast()
                break
            if question and not self.asked_shopper and "message" in self.kit:
                self.asked_shopper = True
                self._act("message", {"content": question})
                continue
            if self.pending_recheck and "inspect_stock" in self.kit:
                self._post_add_recheck()
                continue
            step = self._next_bout()
            if step is not None:
                name, args = step
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.sheet.dead_keys.add(RingSheet.mark(args.get("product_id"), args.get("sku")))
                        self.bracket.exile(RingSheet.mark(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.fog.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.owes_reply = False
                    last_forced, repeats = None, 0
                    self._ring_note("That step changed nothing, so the arena moved on to other fighters.")
                    continue
                obs = self._act(name, args)
                if name == "message":
                    self.owes_reply = False
                    self.reply_stalls = 0
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            proposal = self._ask_model()
            if proposal is None:
                self._close_fast()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _call_args(call)
            content = proposal.get("content") or ""
            reprompts = 0
            while True:
                verdict = self._veto(name, args)
                if verdict is None:
                    break
                if reprompts >= _REPROMPT_CAP:
                    self.rejected_streak += 1
                    self._ring_note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self._scorecard_note()
                        self.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self._ask_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _call_args(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.rejected_streak = 0
            if name not in self.kit:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.sheet.basket:
                    pid, sku = self.sheet.basket[-1]
                args = self._order_args(pid, sku)
            if name == "message":
                if not self.owes_reply:
                    self.questions_asked += 1
                self.owes_reply = False
                self.reply_stalls = 0
            obs = self._act(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                self.search_calls += 1
                if self.card.lane == "recall" and self.card.surface_need:
                    have, need = len(self.sheet.faces), self._surface_need()
                    self._ring_note(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name in ("view", "compare"):
                self._scorecard_note()
        return self.dialogue

    def _veto(self, name: str, args: dict) -> str | None:
        if name not in self.kit:
            return "unknown tool; choose one of " + str(sorted(n for n in self.kit if n))
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.sheet.basket:
                pid, sku = self.sheet.basket[-1]
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.card.lane == "recall" and len(self.sheet.faces) < self._surface_need():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.sheet.faces), self._surface_need())
                )
            if self.card.lane == "preference" and len(self.sheet.booths) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            f = self._weigh(pid, sku)
            key = RingSheet.mark(pid, sku)
            if not f.live:
                return "that variant is out of stock, over budget, or already invalidated"
            if not f.viewed:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.sheet.weighed:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if f.spec != 2 and self.card.gate_name:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.card.gate_name, self.card.gate_mark)
                )
            if not f.ok:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.card.lane == "preference" and len(self.sheet.booths) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.card.lane == "recovery" and not self.sheet.adds and self.card.code and self._relax_of(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if not self._gate_open():
                return (
                    "hold: read one more product that also meets every requirement first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.owes_reply and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.owes_reply and self.questions_asked >= 2:
                return (
                    "the shopper has already answered; decide from the evidence you have rather than asking again"
                )
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.sheet.echo and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.sheet.echo.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return ArenaShop(problem_data).run_card()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
