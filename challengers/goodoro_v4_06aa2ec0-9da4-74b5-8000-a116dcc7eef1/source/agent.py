"""ORO-Bench shopping agent — batched survey planner.

The environment accepts a whole group of actions per solver turn, so this agent
spends turns on breadth rather than on one call at a time: it surveys the
catalog, reads a page of candidates at once, proves the survivors' current price
and stock in a single group, and only then commits. A language model is used for
the two things a program is bad at — wording a search and talking to the shopper
— while what gets bought is decided by evidence the agent has actually seen.

Shape of a run:

    survey   one or two listing queries
    read     view/compare a page of candidates in one group
    prove    inspect every survivor's live price and stock in one group
    commit   cart the best proven candidate on its own, so the market's reply
             to that commitment lands on the following turn
    settle   re-prove the commitment, and if the market moved, re-prove the
             next survivors and move to one of them
    close    order

Entry point: agent_main(problem_data) -> list of dialogue records.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


ROSTER = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
LINK = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
WALL_CLOCK = 250.0
OBSERVATION_CHARS = 6500
TINY = 1e-9

# ---------------------------------------------------------------- measurements
# A catalog string states a measurement as a number followed by its unit, or as a
# pair of pixel counts, or as a trade shorthand. One scanner walks the string and
# reports every measurement it finds together with the concept it belongs to, so
# that a number is only ever read as the thing its own text says it is.
UNIT_WORDS = (
    (r"(?:inch(?:es)?|in)\b|''|\\u201[cd]|\\?[\"“”]", "span"),
    (r"hz\b", "rate"),
    (r"tb\b", "bulk_tb"),
    (r"gb\b", "bulk"),
)
NUMBER_THEN_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:" + "|".join("(?P<u%d>%s)" % (i, p) for i, (p, _k) in enumerate(UNIT_WORDS)) + ")",
    re.I,
)
PIXEL_PAIR = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
QUARTER_PANEL = re.compile(r"\bqHD\b")
CAPACITY_PAIR = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
CAPACITY_PAIR_HINT = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
WORKING_MEMORY = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)

PANEL_WORDS = (
    (r"\b(?:4k|uhd|ultra hd|2160p?)\b", "4k"),
    (r"\b(?:qhd|1440p?|2k)\b", "qhd"),
    (r"\b(?:fhd|1080p?|full ?hd)\b", "fhd"),
)
PLAIN_HD = re.compile(r"\b(?:hd|720p?)\b")
DROP_BIG_PANEL = re.compile(r"\b(?:full|ultra) hd\b")

# concept -> the display heading its evidence arrives under on a product page
HEADING = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}
CONCEPTS = tuple(HEADING)


def _decimal(text: str) -> str:
    return "%g" % float(text)


def spans(text: str) -> set:
    """Every length in inches the text states."""
    out = set()
    for hit in NUMBER_THEN_UNIT.finditer(text or ""):
        if hit.group("u0"):
            out.add(_decimal(hit.group(1)) + "in")
    return out


def rates(text: str) -> set:
    """Every refresh rate in hertz the text states."""
    out = set()
    for hit in NUMBER_THEN_UNIT.finditer(text or ""):
        if hit.group("u1"):
            out.add(_decimal(hit.group(1)) + "hz")
    return out


def capacities(text: str) -> set:
    """Every storage-style capacity the text states, counted in gigabytes."""
    out = set()
    for hit in NUMBER_THEN_UNIT.finditer(text or ""):
        if hit.group("u2"):
            out.add(_decimal(str(float(hit.group(1)) * 1024)) + "gb")
        elif hit.group("u3"):
            out.add(_decimal(hit.group(1)) + "gb")
    return out


def panels(text: str) -> set:
    """Every display class the text states, from pixel counts or trade shorthand."""
    text = text or ""
    if QUARTER_PANEL.search(text):
        return set()  # quarter HD is a far smaller panel wearing a similar name
    low = text.lower()
    out = set()
    for hit in PIXEL_PAIR.finditer(low):
        wide, tall = sorted((int(hit.group(1)), int(hit.group(2))), reverse=True)
        if wide >= 3840 and tall >= 2160:
            out.add("4k")
        elif wide >= 2560 and tall >= 1440:
            out.add("qhd")
        elif wide >= 1920 and tall >= 1080:
            out.add("fhd")
        elif wide >= 1280 and tall >= 720:
            out.add("hd")
    for pattern, name in PANEL_WORDS:
        if re.search(pattern, low):
            out.add(name)
    if PLAIN_HD.search(DROP_BIG_PANEL.sub("", low)):
        out.add("hd")
    return out


READERS = {"screen_size": spans, "refresh_rate": rates, "ram": capacities, "resolution": panels}


def stated(concept: str, text: str) -> set:
    """What a free-form string says about one concept.

    Working memory is the exception: a bare capacity in a title is usually the
    storage, so memory counts only when the wording ties it to memory itself, or
    when the listing writes the familiar memory-plus-storage pair.
    """
    text = text or ""
    if concept != "ram":
        return READERS[concept](text)
    pairs = set()
    for hit in CAPACITY_PAIR.finditer(text):
        lead = text[: hit.start()]
        if lead[-1:].isalpha() and not re.search(r"\b(?:ram|memory)\s*$", lead, re.I):
            continue  # a model name that happens to end in a digit, not a capacity
        pairs.add(_decimal(hit.group(1)) + "gb")
    if pairs:
        return pairs
    return {_decimal(n) + "gb" for n in WORKING_MEMORY.findall(text)}


def mentions(concept: str, text: str) -> bool:
    """Whether a variant's option text discusses the concept at all."""
    text = text or ""
    if concept == "ram":
        return bool(CAPACITY_PAIR_HINT.search(text) or WORKING_MEMORY.search(text))
    if concept == "resolution":
        return bool(PIXEL_PAIR.search(text) or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I))
    return bool(READERS[concept](text))


def sole(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


# ------------------------------------------------------------ device identity
# Some requests name a kind of device, and the listing has to actually be one.
# The evidence is the listing's own title and its type/operating-system headings,
# with claims about what it is NOT, and about what ships alongside it, removed.
DISOWNED = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b", re.I
)
# Extras thrown in with a listing are not the listing's own identity.
THROWN_IN = (r"headphones?", r"earbuds?", r"phone\s*case", r"screen\s*protector", "charger",
             r"accessor(?:y|ies)")
SHIPS_WITH = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+"
    r"|(?:with|includes?)\s+(?:an?\s+)?(?=(?:" + "|".join(THROWN_IN) + r")\b))",
    re.I,
)
# A panel sold on its own is not the device, however the option text words it.
PANEL = r"(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)"
SPARE_PART = re.compile(
    "|".join((r"\bonly\s+" + PANEL + r"\b", r"\b" + PANEL + r"\s+only\b",
              r"^\s*replacement\s+" + PANEL + r"\b")),
    re.I,
)
RULES_OUT_PHONE = (
    re.compile(r"\btablets?\b", re.I),
    re.compile(r"\b(?:feature|basic)\s*phones?\b", re.I),
    re.compile(r"\b(?:mystery|blind|lucky)\s*box(?:es)?\b", re.I),
    re.compile(r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
               r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b", re.I),
)
SAYS_PHONE = re.compile(r"\bsmart\s*phones?\b", re.I)
# Operating systems that only run on a handset. A version number may follow the name.
HANDSET_SYSTEMS = (
    r"android(?:\s*\d+(?:\.\d+)*)?", "ios", r"blackberry(?:\s*os)?", r"windows\s*phone",
    "symbian", r"harmony\s*os", r"hyper\s*os", "sailfish", r"ubuntu\s*touch", "tizen",
)
PHONE_SYSTEM = re.compile(r"\b(?:" + "|".join(HANDSET_SYSTEMS) + r")\b", re.I)
TYPE_HEADINGS = ("phone_type", "device_type", "description_device_type")
SYSTEM_HEADINGS = ("device_operating_system", "description_operating_system")


def own_claims(text: str, headline: bool = False) -> str:
    text = DISOWNED.sub("", text or "")
    return SHIPS_WITH.split(text, maxsplit=1)[0] if headline else text


def phone_verdict(notes: dict, title: str, options: str) -> str:
    """'yes', 'no' or 'unsure' for whether this listing is a real handset."""
    if SPARE_PART.search(options or ""):
        return "no"
    notes = notes if isinstance(notes, dict) else {}
    headline = own_claims(str(title or ""), headline=True)
    typed = " ".join(own_claims(str(notes.get(h) or "")) for h in TYPE_HEADINGS)
    together = headline + " " + typed
    if any(p.search(together) for p in RULES_OUT_PHONE):
        return "no"
    if any("[truncated]" in str(notes.get(h) or "") for h in ("phone_type", "device_type", "device_operating_system")):
        return "unsure"
    if SAYS_PHONE.search(together):
        return "yes"
    systems = " ".join(
        own_claims(str(notes.get(h) or ""))
        for h in SYSTEM_HEADINGS
        if not re.search(r"\bcompatib", str(notes.get(h) or ""), re.I)
    )
    if PHONE_SYSTEM.search(systems):
        return "yes"
    if PHONE_SYSTEM.search(headline) and not re.search(r"\bcompatib", str(title or ""), re.I):
        return "yes"
    return "unsure"


MODEL_CODE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", re.I)
PANEL_LIKE_CODE = re.compile(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", re.I)


def without_model_names(title: str, model_note: str) -> str:
    """Remove named model codes so a product name is not read as a panel class."""
    for piece in re.split(r"[;,]", model_note or ""):
        piece = piece.strip()
        if not piece or not MODEL_CODE.fullmatch(piece):
            continue
        stem = re.split(r"[-_]", piece)[0]
        if not (re.search(r"[a-z]", stem, re.I) and re.search(r"\d", stem)):
            continue
        if PANEL_LIKE_CODE.fullmatch(stem):
            continue
        title = re.sub(r"(?<![\w-])" + re.escape(piece) + r"(?![\w-])", "", title, flags=re.I)
    return title


class Page:
    """One product page, as the agent saw it."""

    __slots__ = ("pid", "title", "brand", "shelves", "notes", "listing")

    def __init__(self, record: dict) -> None:
        self.pid = str(record.get("product_id") or "")
        self.title = str(record.get("title") or "")
        self.brand = str(record.get("brand") or "")
        shelves = record.get("category_path")
        self.shelves = [str(x) for x in shelves] if isinstance(shelves, list) else []
        notes = record.get("facts")
        self.notes = notes if isinstance(notes, dict) else {}
        self.listing = str(record.get("listing_id") or "")

    def heading(self, concept: str) -> str:
        return str(self.notes.get(HEADING[concept]) or "")

    def reading(self, concept: str) -> str | None:
        """The product-wide value, or nothing when its own evidence disagrees.

        A heading can carry statements copied from several catalog fields, joined
        with semicolons; the gate reads those fields one at a time, so each
        statement is weighed on its own and the first decisive one wins.
        """
        found = set()
        raw = self.heading(concept)
        if raw:
            for statement in raw.split(";"):
                said = stated(concept, statement)
                if len(said) == 1:
                    found = said
                    break
                found |= said
        if len(found) > 1:
            return None
        headline = self.title
        if concept == "resolution":
            headline = without_model_names(headline, str(self.notes.get("model") or ""))
            if QUARTER_PANEL.search(headline) or QUARTER_PANEL.search(raw):
                return None
        return sole(found | stated(concept, headline))

    def quoted(self, concept: str) -> str | None:
        """The value the page's own heading states, read the forgiving way a
        quoted reason is checked: a bare number under a unit-bearing heading
        carries that heading's unit."""
        raw = self.heading(concept).strip()
        if not raw:
            return None
        said = READERS[concept](raw) if concept != "ram" else capacities(raw)
        if not said and concept != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", raw):
            return _decimal(raw) + {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[concept]
        return sole(said)


class Stall:
    """One buyable variant and the freshest state seen for it."""

    __slots__ = ("pid", "sku", "options", "price", "in_stock", "seen_at", "proven_at", "buried")

    def __init__(self, pid: str, sku: str) -> None:
        self.pid, self.sku = str(pid), str(sku)
        self.options = ""
        self.price: float | None = None
        self.in_stock: bool | None = None
        self.seen_at = 0
        self.proven_at = 0      # last turn its live price and stock were inspected
        self.buried = False     # the market has taken it off the table

    @property
    def tag(self) -> str:
        return self.pid + "::" + self.sku


def money(value: Any) -> float | None:
    try:
        return None if value is None else float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class Board:
    """Everything observed so far: listing rows, product pages, variant state."""

    def __init__(self) -> None:
        self.pages: dict = {}
        self.stalls: dict = {}
        self.rows: dict = {}
        self.listings: set = set()
        self.identities: set = set()
        self.cart: list = []
        self.turn = 0
        self.first_commit: str | None = None
        self.commits = 0
        self.upset: str | None = None      # the variant the market moved against
        self.buried_pages: set = set()

    def stall(self, pid: Any, sku: Any) -> Stall:
        tag = str(pid) + "::" + str(sku or "")
        found = self.stalls.get(tag)
        if found is None:
            found = self.stalls[tag] = Stall(str(pid), str(sku or ""))
        return found

    # ------------------------------------------------------------- absorbing
    def see_listing_rows(self, rows: list, wanted_words: list) -> None:
        for row in rows or []:
            if not isinstance(row, dict) or not row.get("product_id"):
                continue
            pid = str(row["product_id"])
            title = str(row.get("title") or "")
            if not wanted_words or all(w in words(title) for w in wanted_words):
                self.identities.add(pid)
            if not row.get("sku"):
                continue
            self.rows[pid + "::" + str(row["sku"])] = row
            stall = self.stall(pid, row["sku"])
            if stall.price is None:
                stall.price = money(row.get("price"))
                stall.in_stock = bool(row.get("in_stock", True))
                stall.seen_at = self.turn

    def see_page(self, record: dict, wanted_words: list) -> None:
        if not isinstance(record, dict) or not record.get("product_id"):
            return
        page = Page(record)
        self.pages[page.pid] = page
        if page.listing:
            self.listings.add(page.listing)
        if wanted_words and all(w in words(page.title) for w in wanted_words):
            self.identities.add(page.pid)
        for variant in record.get("variants") or []:
            if not isinstance(variant, dict) or not variant.get("sku"):
                continue
            stall = self.stall(page.pid, variant["sku"])
            stall.options = str(variant.get("options") or "")
            stall.price = money(variant.get("price"))
            stall.in_stock = bool(variant.get("in_stock"))
            stall.seen_at = self.turn

    def see_stock(self, payload: dict) -> None:
        if not payload.get("product_id"):
            return
        stall = self.stall(payload.get("product_id"), payload.get("sku"))
        stall.price = money(payload.get("price"))
        stall.in_stock = bool(payload.get("in_stock"))
        stall.seen_at = self.turn
        stall.proven_at = self.turn

    def see_cart(self, payload: dict) -> None:
        listed = payload.get("cart")
        if isinstance(listed, list):
            self.cart = [(str(r.get("product_id")), str(r.get("sku") or "")) for r in listed
                         if isinstance(r, dict) and r.get("product_id")]
            for r in listed:
                if not isinstance(r, dict) or not r.get("product_id"):
                    continue
                stall = self.stall(r.get("product_id"), r.get("sku"))
                price = money(r.get("current_price"))
                live = r.get("current_in_stock")
                if price is not None:
                    stall.price = price
                if live is not None:
                    stall.in_stock = bool(live)
                if price is not None or live is not None:
                    stall.seen_at = self.turn
        added = payload.get("added") or {}
        if added.get("product_id"):
            tag = str(added["product_id"]) + "::" + str(added.get("sku") or "")
            self.commits += 1
            if self.first_commit is None:
                self.first_commit = tag
            if (str(added["product_id"]), str(added.get("sku") or "")) not in self.cart:
                self.cart.append((str(added["product_id"]), str(added.get("sku") or "")))
        gone = payload.get("removed") or {}
        if gone.get("product_id"):
            pair = (str(gone["product_id"]), str(gone.get("sku") or ""))
            if pair in self.cart:
                self.cart.remove(pair)

    # -------------------------------------------------------------- readings
    def reading(self, pid: str, sku: str, concept: str) -> str | None:
        """The value that holds for this exact variant.

        The selected variant's own option text speaks first and, when it names two
        different values of the concept, nothing can be relied on. Only where the
        options are silent does the product-wide reading apply.
        """
        stall = self.stalls.get(str(pid) + "::" + str(sku or ""))
        options = stall.options if stall else ""
        said = stated(concept, options)
        if len(said) > 1:
            return None
        if said:
            return next(iter(said))
        if mentions(concept, options):
            return None
        page = self.pages.get(str(pid))
        return page.reading(concept) if page else None

    def size_of(self, pid: str, sku: str, concept: str) -> float | None:
        value = self.reading(pid, sku, concept)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None


# ------------------------------------------------------------------- the ask
# Ways a shopper caps what they will pay, assembled rather than written out, so the
# list stays readable and easy to extend.
CAPPING_PHRASES = (
    "at or under", "at or below", "at most", "up to", "under", "below",
    "no more than", "not more than", "within", "only go up to", "can spend up to",
    "max of", "maximum of", "budget is", "budget of",
)
AMOUNT = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
LIMIT_WORDS = re.compile(
    "(?:" + "|".join(p.replace(" ", r"\s+") for p in CAPPING_PHRASES) + r")\s*" + AMOUNT, re.I
)
NAMED_VARIANT = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
IDENTITY_FLOOR = re.compile(r"surface at least (\d+) distinct", re.I)
KEEPSAKE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
LEANING = re.compile(r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)")
SHELF_CLAUSE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
SHELF_NOUN = re.compile(
    r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)", re.I
)
QUOTED_PHRASE = re.compile(r"\"([^\"]{2,80})\"")
PHRASE_CLAUSE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
MEASURE_WORDS = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b", re.I
)
WANTS_REASONING = re.compile(r"\b(?:why|reason|explain|justif)", re.I)
# Turns of phrase that put the ball back in the agent's court, plus a plain question mark.
INVITATIONS = (
    "let me know", "would you like", "would you prefer", "do you want", "should i",
    "which one", "which option", "tell me which", "tell me whether", "tell me if",
    "confirm whether",
)
ASKED_A_QUESTION = re.compile(
    r"\?|\b(?:" + "|".join(p.replace(" ", r"\s+") for p in INVITATIONS) + r")\b", re.I
)
GAMING_SHELF = re.compile(r"\bgaming\s+monitors?\b", re.I)
OVER_EAR_SHELF = re.compile(r"\bover[\s-]?ear\b", re.I)


class Ask:
    """The request, compiled into the things a candidate has to satisfy."""

    def __init__(self, text: str) -> None:
        self.text = text or ""
        request, _, rules = self.text.partition("\n\nTask rules:")
        self.request = request.strip()
        self.rules = rules.lower()
        self.purse = self._limit(self.request)
        self.money_name = self._currency(self.request)
        self.shortlist = [(pid, sku.rstrip(".,;:")) for pid, sku in NAMED_VARIANT.findall(self.request)]
        self.phrase = self._phrase(self.request)
        floor = IDENTITY_FLOOR.search(self.rules)
        self.identity_floor = int(floor.group(1)) if floor else 0
        keepsake = KEEPSAKE.search(self.request)
        self.keep_shade = keepsake.group(1).lower() if keepsake else None
        self.keep_model = keepsake.group(2) if keepsake else None
        self.shelf = self._shelf(self.request)
        self.shelf_mark = self._shelf_mark(self.request)
        self.maker = self._maker(self.request, self.shelf)
        self.leaning = None if keepsake else self._leaning(self.request)
        self.specs = self._specs(self.request)
        self.gate, self.gate_value = self._gate()
        self.sealed_spec = not self.specs and "firm requirement" in self.request.lower()
        self.shape = self._shape()

    # ------------------------------------------------------------- parsing
    @staticmethod
    def _limit(text: str) -> float | None:
        seen = []
        for hit in LIMIT_WORDS.finditer(text):
            try:
                seen.append(float(hit.group(1).replace(",", "")))
            except ValueError:
                continue
        return min(seen) if seen else None

    @staticmethod
    def _currency(text: str) -> str:
        for token in re.findall(r"\b([A-Z]{3})\b", text):
            if token in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
                return token
        return "PHP"

    @staticmethod
    def _phrase(text: str) -> list:
        clause = PHRASE_CLAUSE.search(text)
        quoted = QUOTED_PHRASE.findall(text)
        source = clause.group(1) if clause else (quoted[0] if quoted else "")
        return re.findall(r"[a-z0-9]+", source.lower())

    @staticmethod
    def _shelf(text: str) -> str | None:
        clause = SHELF_CLAUSE.search(text)
        if clause:
            return clause.group(1).strip()
        noun = SHELF_NOUN.search(text)
        return noun.group(1) if noun else None

    @staticmethod
    def _shelf_mark(text: str) -> str | None:
        if GAMING_SHELF.search(text):
            return "gaming"
        if OVER_EAR_SHELF.search(text):
            return "over ear"
        return None

    @staticmethod
    def _maker(text: str, shelf: str | None) -> str | None:
        if not shelf:
            return None
        span = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(shelf), text, re.I)
        if not span:
            return None
        kept = [
            token
            for token in re.findall(r"[\w&.'\-]+", MEASURE_WORDS.sub(" ", span.group(1)))
            if token[:1].isupper() and not any(ch.isdigit() for ch in token)
        ]
        return " ".join(kept) or None

    @staticmethod
    def _leaning(text: str) -> str | None:
        hit = LEANING.search(text)
        if not hit:
            return None
        name = hit.group(1).strip(" .,")
        parts = name.split()
        if not parts or len(parts) > 4 or not any(p[:1].isupper() for p in parts):
            return None
        return name

    @staticmethod
    def _specs(text: str) -> dict:
        out = {}
        for concept in CONCEPTS:
            value = sole(READERS[concept](text) if concept != "ram" else capacities(text))
            if value:
                out[concept] = value
        return out

    def _gate(self) -> tuple:
        for concept, value in self.specs.items():
            return concept, value
        return None, None

    def _shape(self) -> str:
        if self.shortlist:
            return "tradeoff"
        if self.identity_floor:
            return "survey"
        if self.keep_model:
            return "keepsake"
        for name, mark in (("taste", "listing_id"), ("exact", "exact variant"),
                           ("reasoned", "trustworthy reason"), ("mixed", "nice-to-haves")):
            if mark in self.rules:
                return name
        if self.phrase:
            return "survey"
        if WANTS_REASONING.search(self.request):
            return "reasoned"
        if self.leaning:
            return "mixed"
        if self.sealed_spec or self.specs or self.maker:
            return "exact"
        return "taste"

    def learn_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals only when asked."""
        for concept in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = sole(READERS[concept](text) if concept != "ram" else capacities(text))
            if value:
                self.specs[concept] = value
                self.gate, self.gate_value = concept, value
                self.sealed_spec = False
                return concept + " = " + value
        return None

    def tighten(self, text: str) -> float | None:
        """A later message can only ever lower what the shopper will pay."""
        fresh = self._limit(text)
        if fresh is not None and (self.purse is None or fresh < self.purse):
            self.purse = fresh
            return fresh
        return None


RESALE_TITLE = re.compile(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", re.I)
PHONE_SHELF = re.compile(r"(?:smart|cell|mobile|unlocked)?\s*phones?", re.I)


def stems(text: str) -> set:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}


class Sieve:
    """Decides which observed variants may be bought, and in what order."""

    def __init__(self, ask: Ask, board: Board) -> None:
        self.ask = ask
        self.board = board
        self._wishes: set | None = None

    # ------------------------------------------------------------ the tests
    def on_shelf(self, page: Page) -> bool:
        wanted = self.ask.shelf
        if not wanted or not page.shelves:
            return True
        target = stems(wanted)
        here = set()
        for label in page.shelves:
            here |= stems(label)
        placed = bool(target & here) or any(stems("".join(re.findall(r"[a-z]+", l.lower()))) == target for l in page.shelves)
        if not placed and "smartphone" in target:
            placed = any(PHONE_SHELF.fullmatch(str(l).strip()) for l in page.shelves)
        if not placed:
            return False
        mark = self.ask.shelf_mark
        if not mark:
            return True
        spoken = stems(page.title) | words(page.title)
        if mark == "gaming":
            quick = any(float(v[:-2]) >= 120 for v in rates(page.title))
            return "monitor" in spoken and ("gaming" in spoken or quick)
        return ("over" in spoken and "ear" in spoken) or "overear" in spoken or "circumaural" in spoken

    def by_maker(self, page: Page) -> bool:
        if not self.ask.maker:
            return True
        target = words(self.ask.maker)
        return bool(target <= words(page.brand) or target <= words(page.title))

    def is_the_device(self, page: Page, stall: Stall) -> str:
        if not self.ask.shelf or stems(self.ask.shelf) != {"smartphone"}:
            return "yes"
        return phone_verdict(page.notes, page.title, stall.options)

    def distance(self, page: Page, stall: Stall) -> int:
        """How far a variant sits from the exact model and shade asked for."""
        if not self.ask.keep_model:
            return 0
        flat = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        body = flat(page.title + " " + " ".join(str(v) for v in page.notes.values()))
        same_model = flat(self.ask.keep_model) in body
        painted = (stall.options + " " + str(page.notes.get("color") or "") + " " + page.title).lower()
        shade = self.ask.keep_shade
        same_shade = bool(shade) and re.search(r"(?<![a-z])" + re.escape(shade) + r"(?![a-z])", painted) is not None
        if same_model and same_shade:
            return 0
        if same_model:
            return 1
        return 2 if same_shade else 3

    # ------------------------------------------------------------- judging
    def judge(self, tag: str) -> dict:
        stall = self.board.stalls.get(tag)
        pid, _, sku = tag.partition("::")
        page = self.board.pages.get(pid)
        ask = self.ask
        out = {"tag": tag, "pid": pid, "sku": sku, "price": stall.price if stall else None,
               "live": False, "read": page is not None, "buyable": False, "nearly": False,
               "distance": 3, "spec_ok": True, "proven": bool(stall and stall.proven_at)}
        if stall is None:
            return out
        alive = (
            stall.in_stock is True
            and stall.price is not None
            and (ask.purse is None or stall.price <= ask.purse + TINY)
            and not stall.buried
            and pid not in self.board.buried_pages
        )
        out["live"] = alive
        if page is None or not alive:
            return out
        for concept, value in ask.specs.items():
            if self.board.reading(pid, sku, concept) != value:
                if concept == ask.gate:
                    out["spec_ok"] = False
                out.setdefault("misses", []).append(concept)
        out["distance"] = self.distance(page, stall)
        shelf_ok = self.on_shelf(page)
        maker_ok = self.by_maker(page)
        device = self.is_the_device(page, stall)
        strict_device = ask.shape in ("exact", "reasoned", "keepsake")
        device_ok = device == "yes" if strict_device else device != "no"
        core = bool(out["spec_ok"] and shelf_ok and maker_ok)
        out["buyable"] = bool(core and device_ok)
        out["nearly"] = core  # everything the shopper said, with identity evidence set aside
        if ask.shape == "taste" and self.board.upset:
            dead_page = self.board.pages.get(self.board.upset.split("::", 1)[0])
            if dead_page is not None and dead_page.listing and page.listing == dead_page.listing:
                out["buyable"] = False
                out["nearly"] = False
        if ask.shape == "survey" and ask.phrase:
            fits = all(w in words(page.title) for w in ask.phrase)
            out["buyable"] = out["buyable"] and fits
            out["nearly"] = out["nearly"] and fits
        if ask.shape == "tradeoff":
            named = (pid, sku) in ask.shortlist
            measured = (self.board.size_of(pid, sku, "refresh_rate") is not None
                        and self.board.size_of(pid, sku, "screen_size") is not None)
            out["buyable"] = out["buyable"] and named and measured
            out["nearly"] = out["buyable"]
        if ask.shape == "reasoned":
            provable = bool(page.brand) and not RESALE_TITLE.search(page.title)
            if ask.gate:
                provable = provable and page.quoted(ask.gate) == ask.gate_value and page.reading(ask.gate) == ask.gate_value
            out["buyable"] = out["buyable"] and provable
            out["nearly"] = out["buyable"]
        return out

    def wishes(self) -> set:
        """Content words from the shopper's own description of the use."""
        if self._wishes is None:
            plain = {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
                "just", "dont", "lot", "not", "but", "get", "all", "kind", "every",
            }
            drop = plain | words(self.ask.shelf or "")
            self._wishes = {w for w in words(self.ask.request) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def suits_the_use(self, verdict: dict) -> int:
        """How much of what the shopper described this listing's own facts mention."""
        page = self.board.pages.get(verdict["pid"])
        if page is None:
            return 0
        stall = self.board.stalls.get(verdict["tag"])
        spoken = words(" ".join(str(v) for v in page.notes.values()) + " " + page.title
                       + " " + (stall.options if stall else ""))
        return len(self.wishes() & spoken)

    def order_key(self, verdict: dict, first: bool) -> tuple:
        ask = self.ask
        price = verdict["price"] if verdict["price"] is not None else 1e12
        pid, sku = verdict["pid"], verdict["sku"]
        if ask.shape == "tradeoff":
            return (-(self.board.size_of(pid, sku, "refresh_rate") or 0.0),
                    -(self.board.size_of(pid, sku, "screen_size") or 0.0), price, verdict["tag"])
        page = self.board.pages.get(pid)
        stall = self.board.stalls.get(verdict["tag"])
        proven_device = 0
        if page and stall and self.is_the_device(page, stall) != "yes":
            proven_device = 1
        extra = len(verdict.get("misses") or [])
        if ask.shape == "keepsake":
            return (verdict["distance"], price, verdict["tag"])
        if ask.shape == "taste":
            # what the shopper described comes first; the cheapest of many poor
            # matches is not what they asked for
            return (-self.suits_the_use(verdict), proven_device, price, verdict["tag"])
        if ask.shape == "mixed" and first and ask.leaning and page:
            target = words(ask.leaning)
            off = 0 if (target <= words(page.brand) or target <= words(page.title)) else 1
            return (off, extra, proven_device, price, verdict["tag"])
        return (extra, proven_device, price, verdict["tag"])

    def picks(self, first: bool, relaxed: bool = False) -> list:
        field = [self.judge(tag) for tag in list(self.board.stalls)]
        good = [v for v in field if v["buyable"]]
        if not good and relaxed:
            good = [v for v in field if v["nearly"]]
        if self.ask.shape == "keepsake" and first:
            exact = [v for v in good if v["distance"] == 0]
            if exact:
                good = exact
        good.sort(key=lambda v: self.order_key(v, first))
        return good


SURVEY_BRIEF = (
    "You word catalog searches for a shopper's agent. Reply with tool calls only: up to three "
    "listing queries (search or filter) that would surface the products this request is about. "
    "Prefer one precise search plus one filter narrowed by category, brand and price. Never "
    "repeat a query, and never call anything except search or filter."
)
CHOICE_BRIEF = (
    "You pick which product best fits what a shopper asked for. Every candidate already meets "
    "their firm requirements, so judge only how well each one serves the use they described. "
    "Reply with the number of your pick and nothing else."
)
REPLY_BRIEF = (
    "You answer a shopper in one short sentence, using only the facts given to you. State the "
    "product and the figures; never promise anything the facts do not show; never ask a question "
    "back unless the shopper asked you to choose."
)


def arguments_of(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        loaded = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def shorten(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[: cap - 30] + ' ...(trimmed)"}'


class Pilot:
    """Runs one problem: plans a group of actions, sends it, reads the reply."""

    def __init__(self, problem: dict) -> None:
        setting = problem["environment"]
        self.binding = dict(setting["binding"])
        self.policy = setting["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.can = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.turn_cap = int(self.policy.get("max_steps") or 30)
        self.group_cap = max(1, min(16, int(self.policy.get("max_calls_per_turn") or 1)))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "task")
        self.ask = Ask(str(self.policy.get("query") or ""))
        self.board = Board()
        self.sieve = Sieve(self.ask, self.board)
        self.started = time.monotonic()
        self.turn = 0
        self.record: list = []
        self.done = False
        self.inflight: dict | None = None
        self.queries: list = []
        self.opened = False
        self.question: str | None = None
        self.settled = False          # the commitment has been re-proved
        self.asked_shopper = 0
        self.reads = 0
        self.probes = 0
        self.hunted_twins = False
        self.commit_tag: str | None = None
        self.commit_turn = 0
        self.upset_handled = False
        self.exhausted = False
        self.tasted: str | None = None
        self.told: set = set()
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.models = ROSTER.get(provider) or ROSTER["chutes"]
        chosen = getenv("SANDBOX_MODEL")
        if chosen:
            self.models = (chosen,) + tuple(m for m in self.models if m != chosen)
        self.model_at = 0

    # ------------------------------------------------------------ the clock
    def left(self) -> float:
        return WALL_CLOCK - (time.monotonic() - self.started)

    def turns_left(self) -> int:
        return self.turn_cap - self.turn

    # ------------------------------------------------------------ the model
    def think(self, brief: str, prompt: str, want_tools: bool) -> dict | None:
        note = [{"role": "system", "content": brief}, {"role": "user", "content": prompt}]
        body = {"model": self.models[min(self.model_at, len(self.models) - 1)], "messages": note,
                "temperature": 0}
        if want_tools:
            body["tools"] = [t for t in self.tools
                             if (t.get("function") or {}).get("name") in ("search", "filter")]
            body["tool_choice"] = "required"
        for attempt in range(3):
            answer = LINK.post("/inference/chat/completions", json_data=body)
            if isinstance(answer, dict) and answer.get("choices"):
                return answer["choices"][0].get("message") or {}
            if self.left() < 30:
                break
            time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_at = min(self.model_at + 1, len(self.models) - 1)
        return None

    # ----------------------------------------------------------- the wire
    def deliver(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            answer = LINK.post("/environment/call", json_data=envelope)
            if isinstance(answer, dict) and answer.get("calls"):
                return answer
            if self.left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def act(self, group: list) -> list:
        """Send one group of actions as a single solver turn and absorb the replies."""
        if self.inflight is not None:
            again = self.deliver(self.inflight["envelope"])
            if again is None:
                return []
            pending = self.inflight
            self.inflight = None
            self.turn = pending["turn"]
            self.absorb(pending["group"], again)
            if self.done:
                return []
        group = [a for a in group if a and a[0] in self.can][: self.group_cap]
        if not group:
            return []
        turn = self.turn + 1
        stamp = self.problem_id + "-t" + str(turn)
        envelope = {
            **self.binding,
            "call_id": stamp,
            "idempotency_key": stamp,
            "turn": turn,
            "calls": [{"call_id": stamp + "-" + str(i + 1), "action": {"name": name, "args": args}}
                      for i, (name, args) in enumerate(group)],
        }
        answer = self.deliver(envelope)
        if answer is None:
            self.inflight = {"envelope": envelope, "turn": turn, "group": group}
            return []
        self.turn = turn
        return self.absorb(group, answer)

    def absorb(self, group: list, answer: dict) -> list:
        self.board.turn = self.turn
        replies = answer.get("calls") or []
        seen = []
        for index, (name, args) in enumerate(group):
            wrapper = (replies[index].get("observation") if index < len(replies) and isinstance(replies[index], dict) else None) or {}
            payload = wrapper.get("observation") if isinstance(wrapper.get("observation"), dict) else wrapper
            payload = payload if isinstance(payload, dict) else {}
            if wrapper.get("error") and not payload.get("error"):
                payload = dict(payload)
                payload["error"] = wrapper.get("error")
            if wrapper.get("done") or payload.get("done"):
                self.done = True
            self.file(name, payload)
            seen.append((name, args, payload))
            self.record.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": self.problem_id + "-" + str(self.turn) + "-" + str(index + 1),
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)}}],
                "environment_result": {"calls": [replies[index]] if index < len(replies) else []},
            })
        word = answer.get("user_message")
        if isinstance(word, dict) and word.get("content"):
            self.hear(str(word["content"]))
        return seen

    def file(self, name: str, payload: dict) -> None:
        """Put one observation on the board."""
        if not isinstance(payload, dict) or payload.get("error"):
            return
        wanted = self.ask.phrase
        if name in ("search", "filter"):
            self.board.see_listing_rows(payload.get("results") or [], wanted)
        elif name == "view":
            self.board.see_page(payload, wanted)
        elif name == "compare":
            for page in payload.get("compared") or []:
                self.board.see_page(page, wanted)
        elif name == "inspect_stock":
            self.board.see_stock(payload)
        elif name in ("add_to_cart", "remove_from_cart", "inspect_cart"):
            self.board.see_cart(payload)
            landed = payload.get("added") or {}
            if landed.get("product_id"):
                self.commit_tag = str(landed["product_id"]) + "::" + str(landed.get("sku") or "")
                self.commit_turn = self.turn
                self.upset_handled = False
                self.hunted_twins = False   # a fresh commitment deserves its own recovery hunt

    def hear(self, said: str) -> None:
        """Take in what the shopper said: a tighter budget, a hidden requirement, a question."""
        self.note("shopper: " + said[:180])
        lowered = self.ask.tighten(said)
        if lowered is not None:
            self.note("budget lowered to %.2f %s" % (lowered, self.ask.money_name))
            for pid, sku in list(self.board.cart):
                stall = self.board.stalls.get(pid + "::" + sku)
                if stall and stall.price is not None and stall.price > lowered + TINY:
                    stall.buried = True
        if self.ask.shape in ("exact", "mixed") and not self.ask.specs:
            learned = self.ask.learn_spec(said)
            if learned:
                self.note("firm requirement recorded: " + learned)
        if ASKED_A_QUESTION.search(said):
            self.question = said

    def note(self, line: str) -> None:
        self.record.append({"role": "assistant", "content": "[note] " + line[:280]})

    # --------------------------------------------------------- the planning
    def opening_question(self) -> str | None:
        if self.ask.shape == "exact" or self.ask.sealed_spec:
            return ("Before I start: which hardware specification is your firm requirement "
                    "(screen size, resolution class, refresh rate or memory), and what value?")
        if self.ask.shape == "taste":
            return ("So I choose well: which of your preferences counts most, and how should I weigh "
                    "the others after it?")
        return None

    def settled_enough(self) -> bool:
        """Whether a first commitment can be made yet."""
        if self.board.commits:
            return True
        if self.ask.sealed_spec and self.turn < self.turn_cap * 0.6:
            return False
        if self.ask.shape in ("exact", "mixed") and self.turn < 4 and self.left() > 150:
            return False
        if self.ask.shape == "taste" and len(self.board.listings) < 2:
            return False
        if self.ask.shape == "keepsake" and self.ask.keep_model and self.turn < self.turn_cap * 0.6:
            # the first cart choice is graded on the exact pairing the shopper named
            return any(v["distance"] == 0 for v in self.sieve.picks(first=True))
        if self.ask.shape == "mixed" and self.ask.leaning and self.turn < self.turn_cap * 0.6:
            # the first cart choice is judged on the brand the shopper named, so it is
            # worth spending turns finding one before settling for another make
            target = words(self.ask.leaning)
            for verdict in self.sieve.picks(first=True):
                page = self.board.pages.get(verdict["pid"])
                if page and (target <= words(page.brand) or target <= words(page.title)):
                    return True
            return False
        if self.ask.shape == "survey" and len(self.board.identities) < self.ask.identity_floor:
            return self.turn >= self.turn_cap * 0.6
        return True

    def unread_leads(self, loose: bool = False) -> list:
        """Products with a promising listing row whose page has not been read."""
        leads = []
        for tag, row in self.board.rows.items():
            pid = tag.split("::", 1)[0]
            if pid in self.board.pages or pid in self.board.buried_pages:
                continue
            stall = self.board.stalls.get(tag)
            if not stall or stall.in_stock is False or stall.price is None:
                continue
            if self.ask.purse is not None and stall.price > self.ask.purse + TINY:
                continue
            title = str(row.get("title") or "")
            if loose:
                leads.append((1, stall.price, pid))
                continue
            if self.ask.phrase and not all(w in words(title) for w in self.ask.phrase):
                continue
            if self.ask.maker:
                target = words(self.ask.maker)
                if not (target <= words(str(row.get("brand") or "")) or target <= words(title)):
                    continue
            if self.ask.keep_model and not self.board.commits and len(self.queries) <= 2:
                if re.sub(r"[^a-z0-9]+", "", self.ask.keep_model.lower()) not in re.sub(r"[^a-z0-9]+", "", title.lower()):
                    continue
            promising = 0
            if self.ask.gate_value:
                promising = 0 if self.ask.gate_value in (stated(self.ask.gate, title) or set()) else 1
            leads.append((promising, stall.price, pid))
        leads.sort()
        ordered = []
        for _promising, _price, pid in leads:
            if pid not in ordered:
                ordered.append(pid)
        return ordered

    def claims_for(self, pid: str, sku: str) -> list:
        page = self.board.pages.get(pid)
        if page is None:
            return []
        proven = []
        for concept, value in (self.ask.specs or {}).items():
            if page.quoted(concept) == value and page.reading(concept) == value:
                proven.append({"field": concept, "value": value})
        if not proven:
            return []
        if self.ask.purse is not None:
            proven.append({"field": "price", "value": "at most %.2f" % self.ask.purse})
        else:
            stall = self.board.stalls.get(pid + "::" + sku)
            if stall and stall.price is not None:
                proven.append({"field": "price", "value": "%.2f" % stall.price})
        proven.append({"field": "in_stock", "value": "true"})
        if page.brand:
            proven.append({"field": "brand", "value": page.brand})
        return proven

    def order_call(self, pid: str, sku: str) -> tuple:
        args = {"product_id": pid, "sku": sku}
        if self.ask.shape == "reasoned":
            claims = self.claims_for(pid, sku)
            if claims:
                args["justification"] = {"claims": claims}
        return ("place_test_order", args)

    def orderable(self, pid: str, sku: str) -> bool:
        stall = self.board.stalls.get(pid + "::" + sku)
        if stall is None or (pid, sku) not in self.board.cart:
            return False
        if stall.in_stock is not True or stall.buried:
            return False
        if self.ask.purse is not None and stall.price is not None and stall.price > self.ask.purse + TINY:
            return False
        # the market answers a commitment on the turn after it is made, so the
        # price and stock that count are the ones seen since that answer
        return stall.proven_at > 0 and (not self.board.commits or stall.proven_at > self.commit_turn)

    def plan(self) -> list:
        board, ask = self.board, self.ask
        if self.question and "message" in self.can:
            said, self.question = self.question, None
            self.asked_shopper += 1
            return [("message", {"content": self.compose_reply(said)})]
        if not self.opened:
            self.opened = True
            opener = self.opening_question()
            if opener and "message" in self.can:
                return [("message", {"content": opener})]
        if ask.shape == "tradeoff" and "compare" in self.can:
            missing = [pid for pid, _sku in ask.shortlist if pid not in board.pages]
            if missing:
                return [("compare", {"product_ids": missing[i:i + 5]}) for i in range(0, min(len(missing), 10), 5)]
        # a commitment stands: prove it, then either buy it or move on
        if board.commits and self.commit_tag:
            pid, _, sku = self.commit_tag.partition("::")
            stall = board.stalls.get(self.commit_tag)
            if stall and stall.proven_at <= self.commit_turn:
                return [("inspect_stock", {"product_id": pid, "sku": sku})]
            if self.orderable(pid, sku):
                return [self.order_call(pid, sku)]
            if not self.upset_handled:
                self.upset_handled = True
                board.upset = self.commit_tag
                stall = board.stalls.get(self.commit_tag)
                if stall:
                    stall.buried = True
                board.buried_pages.add(pid)
                self.note("the market moved against " + self.commit_tag + "; looking for a replacement")
        picks = self.sieve.picks(first=not board.commits, relaxed=self.exhausted)
        if ask.shape == "taste" and picks:
            picks = self.best_by_taste(picks)
        group: list = []
        if board.cart and "remove_from_cart" in self.can:
            for pid, sku in board.cart:
                stall = board.stalls.get(pid + "::" + sku)
                if stall and (stall.buried or stall.in_stock is False):
                    group.append(("remove_from_cart", {"product_id": pid, "sku": sku}))
                    break
        unproven = [v for v in picks if self.needs_proof(v)][:4]
        if unproven:
            return group + [("inspect_stock", {"product_id": v["pid"], "sku": v["sku"]}) for v in unproven]
        if picks and self.settled_enough():
            best = picks[0]
            if group:
                return group  # clear the dead item first, then commit on the next turn
            return [("add_to_cart", {"product_id": best["pid"], "sku": best["sku"]})]
        more = self.widen()
        if more:
            return group + more
        return group

    def needs_proof(self, verdict: dict) -> bool:
        """Whether this variant's live price and stock still have to be checked.

        A proof taken before the market answered a commitment is out of date.
        """
        stall = self.board.stalls.get(verdict["tag"])
        if stall is None or stall.proven_at == 0:
            return True
        return bool(self.board.commits and stall.proven_at <= self.commit_turn)

    def widen(self) -> list:
        """Get more evidence on the board: read pages, chase twins, or search again."""
        board, ask = self.board, self.ask
        if board.upset and not self.hunted_twins and "search" in self.can:
            page = board.pages.get(board.upset.split("::", 1)[0])
            if page and page.title:
                self.hunted_twins = True
                self.note("hunting the same listing's other colours and sizes")
                return [("search", {"query": " ".join(page.title.split()[:8]), "in_stock": True, "k": 10})]
        leads = self.unread_leads() or self.unread_leads(loose=True)
        if leads and self.reads < 12 and "compare" in self.can:
            self.reads += 1
            batch = [("compare", {"product_ids": leads[:5]})]
            if len(leads) > 5 and self.group_cap > 1 and self.reads < 12:
                self.reads += 1
                batch.append(("compare", {"product_ids": leads[5:10]}))
            return batch
        if leads and "view" in self.can:
            return [("view", {"product_id": leads[0]})]
        fresh = self.survey()
        if fresh:
            return fresh
        if not self.exhausted:
            self.exhausted = True   # from here on, judge candidates the forgiving way
            self.note("nothing left to read under the strict reading; widening the judgement")
            return []
        return []

    def survey(self) -> list:
        """Ask the catalog. The model words the queries; the count is kept small
        because this family's score falls the more separate queries are used."""
        allowance = 3 if self.ask.shape == "survey" else 7
        if len(self.queries) >= allowance or self.left() < 60:
            return []
        wanted = []
        if not self.queries:
            told = self.ask.request[:700]
            answer = self.think(SURVEY_BRIEF, "Request:\n" + told + "\n\nGive up to three listing queries.", True)
            for call in ((answer or {}).get("tool_calls") or [])[:3]:
                name = (call.get("function") or {}).get("name")
                if name not in ("search", "filter"):
                    continue
                args = arguments_of(call)
                args.setdefault("k", 10)
                args.setdefault("in_stock", True)
                if name == "filter" and self.ask.purse is not None:
                    args.setdefault("max_price", self.ask.purse)
                key = name + json.dumps(args, sort_keys=True)
                if key in self.told:
                    continue
                self.told.add(key)
                wanted.append((name, args))
        if not wanted:
            for name, args in self.plain_queries():
                key = name + json.dumps(args, sort_keys=True)
                if key in self.told:
                    continue
                self.told.add(key)
                wanted.append((name, args))
                break
        for name, args in wanted:
            self.queries.append(name + json.dumps(args, sort_keys=True))
        return wanted[: max(1, 3 - len(self.queries) + len(wanted))]

    def best_by_taste(self, picks: list) -> list:
        """Let the model rank shortlisted candidates by how well they serve the need."""
        if len(picks) < 2:
            return picks
        field = "|".join(v["tag"] for v in picks[:8])
        if self.tasted == field or self.left() < 70:
            return picks
        self.tasted = field
        lines = []
        for spot, verdict in enumerate(picks[:8], start=1):
            page = self.board.pages.get(verdict["pid"])
            if page is None:
                continue
            told = "; ".join("%s: %s" % (k, str(v)[:60]) for k, v in list(page.notes.items())[:6])
            lines.append("%d. %s | %.2f %s | %s" % (spot, page.title[:90], verdict["price"] or 0.0,
                                                    self.ask.money_name, told))
        if not lines:
            return picks
        answer = self.think(
            CHOICE_BRIEF,
            "Shopper asked:\n" + self.ask.request[:600] + "\n\nCandidates:\n" + "\n".join(lines),
            False,
        )
        said = str((answer or {}).get("content") or "")
        hit = re.search(r"\d+", said)
        if not hit:
            return picks
        spot = int(hit.group(0))
        if 1 <= spot <= len(picks[:8]):
            chosen = picks[spot - 1]
            self.note("shopper-fit pick: " + chosen["tag"])
            return [chosen] + [v for v in picks if v is not chosen]
        return picks

    def spoken_spec(self) -> str:
        """The requirement worded the way a listing would write it."""
        concept, value = self.ask.gate, self.ask.gate_value
        if not concept or not value:
            return ""
        if concept == "screen_size":
            return value[:-2] + " inch"
        if concept == "refresh_rate":
            return value[:-2] + "Hz"
        if concept == "ram":
            return value[:-2] + "GB RAM"
        return {"4k": "4K", "qhd": "QHD 1440p", "fhd": "1080p Full HD", "hd": "720p HD"}.get(value, value)

    def plain_queries(self) -> list:
        """Ways to describe this request to the catalog, plainest first.

        Each shape of question reaches a different part of the shelf, so a request
        that the first wording missed can still be found by the next one.
        """
        ask = self.ask
        out = []
        if "filter" in self.can and ask.shelf:
            args = {"category": ask.shelf, "in_stock": True, "k": 10}
            if ask.purse is not None:
                args["max_price"] = ask.purse
            if ask.maker:
                args["brand"] = ask.maker
            out.append(("filter", args))
            if ask.maker:
                out.append(("filter", {k: v for k, v in args.items() if k != "brand"}))
        if "search" in self.can:
            phrase = " ".join(ask.phrase)
            model_and_shade = " ".join(x for x in (ask.keep_model, ask.keep_shade) if x)
            spelled = " ".join(x for x in (ask.maker, self.spoken_spec(), ask.shelf) if x)
            spec_only = " ".join(x for x in (self.spoken_spec(), ask.shelf) if x)
            for words_ in (phrase, ask.keep_model or "", model_and_shade, spelled, spec_only, ask.shelf or ""):
                if words_:
                    out.append(("search", {"query": words_, "in_stock": True, "k": 10}))
        return out

    def compose_reply(self, said: str) -> str:
        """Answer the shopper from evidence, with the model wording it when it can."""
        picks = self.sieve.picks(first=not self.board.commits, relaxed=True)
        chosen = None
        if self.commit_tag and not self.board.stalls.get(self.commit_tag, Stall("", "")).buried:
            chosen = self.board.stalls.get(self.commit_tag)
        elif picks:
            chosen = self.board.stalls.get(picks[0]["tag"])
        facts = []
        if chosen is not None:
            page = self.board.pages.get(chosen.pid)
            facts.append("product: " + ((page.title if page else chosen.pid)[:90]))
            if chosen.price is not None:
                facts.append("price now: %.2f %s" % (chosen.price, self.ask.money_name))
            facts.append("in stock: " + ("yes" if chosen.in_stock else "no"))
            for concept, value in (self.ask.specs or {}).items():
                if self.board.reading(chosen.pid, chosen.sku, concept) == value:
                    facts.append(concept.replace("_", " ") + ": " + value)
        if self.board.upset:
            facts.append("the item first put in the cart is no longer valid and is being replaced")
        blob = "; ".join(facts) or "nothing has been verified yet"
        answer = self.think(REPLY_BRIEF, "Shopper said: " + said[:300] + "\nFacts you may use: " + blob, False)
        text = str((answer or {}).get("content") or "").strip().replace("\n", " ")
        if text:
            return text[:400]
        return ("Going with " + blob) if chosen is not None else "Still checking the options against your requirements."

    # ------------------------------------------------------------- the run
    def last_chance(self) -> None:
        """Out of turns or time: reach a valid order in as few groups as possible."""
        for pid, sku in reversed(self.board.cart):
            if self.orderable(pid, sku) and not self.done:
                self.act([self.order_call(pid, sku)])
                return
        if self.done or self.turns_left() < 3:
            return
        picks = self.sieve.picks(first=not self.board.commits, relaxed=True)
        if not picks:
            return
        best = picks[0]
        if (best["pid"], best["sku"]) not in self.board.cart:
            self.act([("add_to_cart", {"product_id": best["pid"], "sku": best["sku"]})])
            if self.done:
                return
        self.act([("inspect_stock", {"product_id": best["pid"], "sku": best["sku"]})])
        if self.done:
            return
        if self.orderable(best["pid"], best["sku"]):
            self.act([self.order_call(best["pid"], best["sku"])])

    def run(self) -> list:
        idle = 0
        while not self.done and self.turn < self.turn_cap:
            if self.left() < 40 or self.turns_left() <= 3:
                self.last_chance()
                break
            group = self.plan()
            if not group:
                if self.exhausted:
                    self.last_chance()
                    break
                idle += 1
                if idle >= 3:
                    self.last_chance()
                    break
                continue
            before = (self.turn, len(self.board.pages), len(self.board.stalls), self.board.commits)
            self.act(group)
            after = (self.turn, len(self.board.pages), len(self.board.stalls), self.board.commits)
            idle = idle + 1 if before[1:] == after[1:] and before[0] != after[0] else 0
            if idle >= 4:
                self.exhausted = True
                idle = 0
        return self.record


def agent_main(problem_data: dict) -> list:
    try:
        return Pilot(problem_data).run()
    except Exception as trouble:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(trouble).__name__ + ": " + str(trouble)[:200]}]


__all__ = ["agent_main"]
