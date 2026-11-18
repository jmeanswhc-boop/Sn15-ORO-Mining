"""ORO-Bench shopping agent — champion and challenger.

This agent keeps one product as its standing choice and spends every turn trying
to unseat it. A challenger only earns attention if, on the evidence already in
hand, it could beat the champion; the agent then spends one grouped turn proving
the pair's live price and stock. When no challenger can beat the champion, the
champion is bought. The same duel runs again the moment the market moves against
whatever is in the cart.

Deciding what counts as "better" is the task's own business: the cheapest item
that meets every stated requirement, the top of a stated ranking rule, the exact
model and colour a shopper named, or the listing that best serves a described
use. A language model is asked to shortlist listings worth opening and to answer
the shopper; everything that decides the purchase is evidence the agent has seen.

Entry point: agent_main(problem_data) -> list of dialogue records.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


ENGINES = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
GATEWAY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
SECONDS_ALLOWED = 250.0
SLACK = 1e-9

# --------------------------------------------------------------- reading spec
# Every gated concept is described once: where its evidence appears on a product
# page, how a number of that concept is written, and how the value is spelled
# when it is quoted back to the shopper. One reader serves all of them.
INCHES = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|\\u201[cd]|\\?[\"“”])", re.I)
HERTZ = re.compile(r"(\d+(?:\.\d+)?)\s*hz\b", re.I)
GIGA = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
PIXELS = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
QUARTER = re.compile(r"\bqHD\b")
PAIRED_GIGA = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
PAIRED_HINT = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
NAMED_GIGA = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
BIG_PANELS = ((r"\b(?:4k|uhd|ultra hd|2160p?)\b", "4k"),
              (r"\b(?:qhd|1440p?|2k)\b", "qhd"),
              (r"\b(?:fhd|1080p?|full ?hd)\b", "fhd"))
SMALL_PANEL = re.compile(r"\b(?:hd|720p?)\b")
BIG_HD = re.compile(r"\b(?:full|ultra) hd\b")


def tidy(number: str) -> str:
    return "%g" % float(number)


def read_inches(text: str) -> set:
    return {tidy(m.group(1)) + "in" for m in INCHES.finditer(text or "")}


def read_hertz(text: str) -> set:
    return {tidy(m.group(1)) + "hz" for m in HERTZ.finditer(text or "")}


def read_giga(text: str) -> set:
    out = set()
    for m in GIGA.finditer(text or ""):
        scale = 1024 if m.group(2).lower() == "tb" else 1
        out.add(tidy(str(float(m.group(1)) * scale)) + "gb")
    return out


def read_panel(text: str) -> set:
    text = text or ""
    if QUARTER.search(text):
        return set()
    low = text.lower()
    out = set()
    for m in PIXELS.finditer(low):
        long_side, short_side = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        for floor_long, floor_short, name in ((3840, 2160, "4k"), (2560, 1440, "qhd"),
                                              (1920, 1080, "fhd"), (1280, 720, "hd")):
            if long_side >= floor_long and short_side >= floor_short:
                out.add(name)
                break
    for pattern, name in BIG_PANELS:
        if re.search(pattern, low):
            out.add(name)
    if SMALL_PANEL.search(BIG_HD.sub("", low)):
        out.add("hd")
    return out


def read_memory(text: str) -> set:
    """A capacity is memory only when the wording ties it to memory."""
    text = text or ""
    pairs = set()
    for m in PAIRED_GIGA.finditer(text):
        before = text[: m.start()]
        if before[-1:].isalpha() and not re.search(r"\b(?:ram|memory)\s*$", before, re.I):
            continue
        pairs.add(tidy(m.group(1)) + "gb")
    if pairs:
        return pairs
    return {tidy(n) + "gb" for n in NAMED_GIGA.findall(text)}


CONCEPT_SPEC = {
    "screen_size": {"heading": "size", "scan": read_inches, "unit": "in", "spoken": "%s inch"},
    "resolution": {"heading": "display", "scan": read_panel, "unit": "", "spoken": "%s"},
    "refresh_rate": {"heading": "refresh_rate", "scan": read_hertz, "unit": "hz", "spoken": "%sHz"},
    "ram": {"heading": "storage", "scan": read_memory, "unit": "gb", "spoken": "%sGB RAM"},
}
CONCEPTS = tuple(CONCEPT_SPEC)


def scan(concept: str, text: str) -> set:
    if concept == "ram":
        return read_memory(text)
    return CONCEPT_SPEC[concept]["scan"](text)


def raises_concept(concept: str, text: str) -> bool:
    """Whether an option string discusses the concept, even unclearly."""
    text = text or ""
    if concept == "ram":
        return bool(PAIRED_HINT.search(text) or NAMED_GIGA.search(text))
    if concept == "resolution":
        return bool(PIXELS.search(text)
                    or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I))
    return bool(CONCEPT_SPEC[concept]["scan"](text))


def only_one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


# ------------------------------------------------------------ what a listing is
NEGATED = re.compile(r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b", re.I)
BUNDLED = re.compile(r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+"
                     r"|(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))", re.I)
BARE_PANEL = re.compile(r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b"
                        r"|\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b"
                        r"|^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b", re.I)
NOT_A_HANDSET = re.compile(r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b"
                           r"|\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for"
                           r"|replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b", re.I)
CALLS_ITSELF_PHONE = re.compile(r"\bsmart\s*phones?\b", re.I)
MOBILE_OS = re.compile(r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian"
                       r"|harmony\s*os|hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b", re.I)
MODEL_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", re.I)
PANEL_TOKEN = re.compile(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", re.I)


def plain_claim(text: str, is_title: bool = False) -> str:
    text = NEGATED.sub("", text or "")
    return BUNDLED.split(text, maxsplit=1)[0] if is_title else text


def handset_check(facts: dict, title: str, options: str) -> str:
    """'yes', 'no' or 'unsure' — is this listing an actual handset?"""
    if BARE_PANEL.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    head = plain_claim(str(title or ""), is_title=True)
    kinds = " ".join(plain_claim(str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
    both = head + " " + kinds
    if NOT_A_HANDSET.search(both):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unsure"
    if CALLS_ITSELF_PHONE.search(both):
        return "yes"
    running = " ".join(plain_claim(str(facts.get(k) or ""))
                       for k in ("device_operating_system", "description_operating_system")
                       if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I))
    if MOBILE_OS.search(running):
        return "yes"
    if MOBILE_OS.search(head) and not re.search(r"\bcompatib", str(title or ""), re.I):
        return "yes"
    return "unsure"


def drop_model_codes(title: str, model_fact: str) -> str:
    for token in re.split(r"[;,]", model_fact or ""):
        token = token.strip()
        if not token or not MODEL_TOKEN.fullmatch(token):
            continue
        head = re.split(r"[-_]", token)[0]
        if not (re.search(r"[a-z]", head, re.I) and re.search(r"\d", head)) or PANEL_TOKEN.fullmatch(head):
            continue
        title = re.sub(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", "", title, flags=re.I)
    return title


def bag(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def roots(text: str) -> set:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}


def as_money(value: Any) -> float | None:
    try:
        return None if value is None else float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


class Card:
    """One variant and everything known about it."""

    __slots__ = ("pid", "sku", "title", "brand", "shelves", "facts", "listing",
                 "options", "price", "stocked", "checked_on", "vetoed", "opened")

    def __init__(self, pid: str, sku: str) -> None:
        self.pid, self.sku = str(pid), str(sku or "")
        self.title = self.brand = self.listing = self.options = ""
        self.shelves: list = []
        self.facts: dict = {}
        self.price: float | None = None
        self.stocked: bool | None = None
        self.checked_on = 0     # turn its live price and stock were last inspected
        self.vetoed = False     # the market took it away
        self.opened = False     # its product page has been read

    @property
    def ref(self) -> str:
        return self.pid + "::" + self.sku

    def heading(self, concept: str) -> str:
        return str(self.facts.get(CONCEPT_SPEC[concept]["heading"]) or "")

    def page_value(self, concept: str) -> str | None:
        """The product-wide value, or nothing when the page contradicts itself."""
        gathered = set()
        head = self.heading(concept)
        if head:
            for clause in head.split(";"):
                found = scan(concept, clause)
                if len(found) == 1:
                    gathered = found
                    break
                gathered |= found
        if len(gathered) > 1:
            return None
        title = self.title
        if concept == "resolution":
            title = drop_model_codes(title, str(self.facts.get("model") or ""))
            if QUARTER.search(title) or QUARTER.search(head):
                return None
        return only_one(gathered | scan(concept, title))

    def value(self, concept: str) -> str | None:
        """What holds for this exact variant: its own options speak first."""
        said = scan(concept, self.options)
        if len(said) > 1:
            return None
        if said:
            return next(iter(said))
        if raises_concept(concept, self.options):
            return None
        return self.page_value(concept)

    def measure(self, concept: str) -> float | None:
        value = self.value(concept)
        unit = CONCEPT_SPEC[concept]["unit"]
        try:
            return float(value[: -len(unit)]) if value and unit else None
        except ValueError:
            return None

    def quotable(self, concept: str) -> str | None:
        """The value the page's own heading states, read the way a quoted reason is
        checked: a bare number under a unit-bearing heading carries that unit."""
        head = self.heading(concept).strip()
        if not head:
            return None
        found = scan(concept, head)
        if not found and concept != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", head):
            return tidy(head) + CONCEPT_SPEC[concept]["unit"]
        return only_one(found)


# ------------------------------------------------------------ what was asked
CEILING = re.compile(r"(?:at or under|at or below|at most|up to|under|below|no more than|not more than"
                     r"|within|only go up to|can spend up to|max(?:imum)? of|budget (?:is|of))\s*"
                     r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", re.I)
LISTED_VARIANT = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
MIN_IDENTITIES = re.compile(r"surface at least (\d+) distinct", re.I)
EXACT_WISH = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
SOFT_WISH = re.compile(r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)")
SHELF_NAMED = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
SHELF_GUESSED = re.compile(r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)", re.I)
IN_QUOTES = re.compile(r"\"([^\"]{2,80})\"")
PHRASE_NAMED = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
SPEC_WORDING = re.compile(r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b"
                          r"|\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b", re.I)
REASON_WANTED = re.compile(r"\b(?:why|reason|explain|justif)", re.I)
TURNED_TO_US = re.compile(r"\?|\b(?:let me know|would you (?:like|prefer)|do you want|should i"
                          r"|which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\b", re.I)
GAMING_MONITOR = re.compile(r"\bgaming\s+monitors?\b", re.I)
OVER_EAR = re.compile(r"\bover[\s-]?ear\b", re.I)
HANDSET_SHELF = re.compile(r"(?:smart|cell|mobile|unlocked)?\s*phones?", re.I)
SECOND_HAND = re.compile(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", re.I)


class Wants:
    """The request, read once into the things that decide a purchase."""

    def __init__(self, whole: str) -> None:
        self.whole = whole or ""
        spoken, _, printed = self.whole.partition("\n\nTask rules:")
        self.spoken = spoken.strip()
        self.printed = printed.lower()
        self.ceiling = self._ceiling(self.spoken)
        self.currency = self._currency(self.spoken)
        self.listed = [(pid, sku.rstrip(".,;:")) for pid, sku in LISTED_VARIANT.findall(self.spoken)]
        self.phrase = self._phrase(self.spoken)
        floor = MIN_IDENTITIES.search(self.printed)
        self.identity_floor = int(floor.group(1)) if floor else 0
        wish = EXACT_WISH.search(self.spoken)
        self.wish_colour = wish.group(1).lower() if wish else None
        self.wish_model = wish.group(2) if wish else None
        self.shelf = self._shelf(self.spoken)
        self.shelf_extra = ("gaming" if GAMING_MONITOR.search(self.spoken)
                            else "over ear" if OVER_EAR.search(self.spoken) else None)
        self.brand = self._brand(self.spoken, self.shelf)
        self.soft_brand = None if wish else self._soft_brand(self.spoken)
        self.needs = self._needs(self.spoken)
        self.gate = next(iter(self.needs), None)
        self.gate_value = self.needs.get(self.gate) if self.gate else None
        self.hidden_need = not self.needs and "firm requirement" in self.spoken.lower()
        self.errand = self._errand()

    @staticmethod
    def _ceiling(text: str) -> float | None:
        amounts = []
        for hit in CEILING.finditer(text):
            try:
                amounts.append(float(hit.group(1).replace(",", "")))
            except ValueError:
                pass
        return min(amounts) if amounts else None

    @staticmethod
    def _currency(text: str) -> str:
        for code in re.findall(r"\b([A-Z]{3})\b", text):
            if code in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
                return code
        return "PHP"

    @staticmethod
    def _phrase(text: str) -> list:
        named = PHRASE_NAMED.search(text)
        quoted = IN_QUOTES.findall(text)
        source = named.group(1) if named else (quoted[0] if quoted else "")
        return re.findall(r"[a-z0-9]+", source.lower())

    @staticmethod
    def _shelf(text: str) -> str | None:
        named = SHELF_NAMED.search(text)
        if named:
            return named.group(1).strip()
        guessed = SHELF_GUESSED.search(text)
        return guessed.group(1) if guessed else None

    @staticmethod
    def _brand(text: str, shelf: str | None) -> str | None:
        if not shelf:
            return None
        run = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(shelf), text, re.I)
        if not run:
            return None
        keep = [w for w in re.findall(r"[\w&.'\-]+", SPEC_WORDING.sub(" ", run.group(1)))
                if w[:1].isupper() and not any(c.isdigit() for c in w)]
        return " ".join(keep) or None

    @staticmethod
    def _soft_brand(text: str) -> str | None:
        hit = SOFT_WISH.search(text)
        if not hit:
            return None
        name = hit.group(1).strip(" .,")
        parts = name.split()
        if not parts or len(parts) > 4 or not any(p[:1].isupper() for p in parts):
            return None
        return name

    @staticmethod
    def _needs(text: str) -> dict:
        found = {}
        for concept in CONCEPTS:
            value = only_one(scan(concept, text))
            if value:
                found[concept] = value
        return found

    def _errand(self) -> str:
        if self.listed:
            return "rank"
        if self.identity_floor:
            return "sweep"
        if self.wish_model:
            return "exactly"
        for name, tell in (("fit", "listing_id"), ("spec", "exact variant"),
                           ("reason", "trustworthy reason"), ("blend", "nice-to-haves")):
            if tell in self.printed:
                return name
        if self.phrase:
            return "sweep"
        if REASON_WANTED.search(self.spoken):
            return "reason"
        if self.soft_brand:
            return "blend"
        if self.hidden_need or self.needs or self.brand:
            return "spec"
        return "fit"

    def spoken_need(self) -> str:
        if not self.gate or not self.gate_value:
            return ""
        spec = CONCEPT_SPEC[self.gate]
        stem = self.gate_value[: -len(spec["unit"])] if spec["unit"] else self.gate_value
        if self.gate == "resolution":
            return {"4k": "4K", "qhd": "QHD 1440p", "fhd": "1080p Full HD", "hd": "720p HD"}.get(self.gate_value, self.gate_value)
        return spec["spoken"] % stem

    def take_need(self, said: str) -> str | None:
        for concept in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = only_one(scan(concept, said))
            if value:
                self.needs[concept] = value
                self.gate, self.gate_value = concept, value
                self.hidden_need = False
                return concept + " = " + value
        return None

    def lower_ceiling(self, said: str) -> float | None:
        fresh = self._ceiling(said)
        if fresh is not None and (self.ceiling is None or fresh < self.ceiling):
            self.ceiling = fresh
            return fresh
        return None


class Deck:
    """Every card seen, plus what the cart and the market have done."""

    def __init__(self) -> None:
        self.cards: dict = {}
        self.rows: dict = {}
        self.listings: set = set()
        self.identities: set = set()
        self.cart: list = []
        self.turn = 0
        self.adds = 0
        self.staked: str | None = None      # the card most recently put in the cart
        self.staked_on = 0
        self.struck: str | None = None      # the card the market struck down
        self.spent_pages: set = set()

    def card(self, pid: Any, sku: Any) -> Card:
        ref = str(pid) + "::" + str(sku or "")
        held = self.cards.get(ref)
        if held is None:
            held = self.cards[ref] = Card(str(pid), str(sku or ""))
        return held

    def take_rows(self, rows: list, phrase: list) -> None:
        for row in rows or []:
            if not isinstance(row, dict) or not row.get("product_id"):
                continue
            pid = str(row["product_id"])
            title = str(row.get("title") or "")
            if not phrase or all(w in bag(title) for w in phrase):
                self.identities.add(pid)
            if not row.get("sku"):
                continue
            self.rows[pid + "::" + str(row["sku"])] = row
            card = self.card(pid, row["sku"])
            if not card.opened:
                card.title = card.title or title
                card.brand = card.brand or str(row.get("brand") or "")
                if card.price is None:
                    card.price = as_money(row.get("price"))
                    card.stocked = bool(row.get("in_stock", True))

    def take_page(self, page: dict, phrase: list) -> None:
        if not isinstance(page, dict) or not page.get("product_id"):
            return
        pid = str(page["product_id"])
        title = str(page.get("title") or "")
        shelves = page.get("category_path")
        shelves = [str(x) for x in shelves] if isinstance(shelves, list) else []
        facts = page.get("facts") if isinstance(page.get("facts"), dict) else {}
        listing = str(page.get("listing_id") or "")
        if listing:
            self.listings.add(listing)
        if phrase and all(w in bag(title) for w in phrase):
            self.identities.add(pid)
        for variant in page.get("variants") or []:
            if not isinstance(variant, dict) or not variant.get("sku"):
                continue
            card = self.card(pid, variant["sku"])
            card.title, card.brand, card.shelves, card.facts, card.listing = title, str(page.get("brand") or ""), shelves, facts, listing
            card.options = str(variant.get("options") or "")
            card.price = as_money(variant.get("price"))
            card.stocked = bool(variant.get("in_stock"))
            card.opened = True

    def take_stock(self, seen: dict) -> None:
        if not seen.get("product_id"):
            return
        card = self.card(seen.get("product_id"), seen.get("sku"))
        card.price = as_money(seen.get("price"))
        card.stocked = bool(seen.get("in_stock"))
        card.checked_on = self.turn

    def take_cart(self, seen: dict) -> None:
        listed = seen.get("cart")
        if isinstance(listed, list):
            self.cart = [(str(r.get("product_id")), str(r.get("sku") or "")) for r in listed
                         if isinstance(r, dict) and r.get("product_id")]
            for r in listed:
                if isinstance(r, dict) and r.get("product_id"):
                    card = self.card(r.get("product_id"), r.get("sku"))
                    price, live = as_money(r.get("current_price")), r.get("current_in_stock")
                    if price is not None:
                        card.price = price
                    if live is not None:
                        card.stocked = bool(live)
        put = seen.get("added") or {}
        if put.get("product_id"):
            ref = str(put["product_id"]) + "::" + str(put.get("sku") or "")
            self.adds += 1
            self.staked, self.staked_on = ref, self.turn
            pair = (str(put["product_id"]), str(put.get("sku") or ""))
            if pair not in self.cart:
                self.cart.append(pair)
        pulled = seen.get("removed") or {}
        if pulled.get("product_id"):
            pair = (str(pulled["product_id"]), str(pulled.get("sku") or ""))
            if pair in self.cart:
                self.cart.remove(pair)


class Duel:
    """Ranks cards against each other by what this particular errand values."""

    def __init__(self, wants: Wants, deck: Deck) -> None:
        self.wants = wants
        self.deck = deck
        self._spoken_words: set | None = None

    def on_the_right_shelf(self, card: Card) -> bool:
        asked = self.wants.shelf
        if not asked or not card.shelves:
            return True
        target = roots(asked)
        shelf_words = set()
        for label in card.shelves:
            shelf_words |= roots(label)
        placed = bool(target & shelf_words) or any(roots("".join(re.findall(r"[a-z]+", l.lower()))) == target for l in card.shelves)
        if not placed and "smartphone" in target:
            placed = any(HANDSET_SHELF.fullmatch(str(l).strip()) for l in card.shelves)
        if not placed:
            return False
        extra = self.wants.shelf_extra
        if not extra:
            return True
        spoken = roots(card.title) | bag(card.title)
        if extra == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in read_hertz(card.title))
            return "monitor" in spoken and ("gaming" in spoken or fast)
        return ("over" in spoken and "ear" in spoken) or "overear" in spoken or "circumaural" in spoken

    def right_maker(self, card: Card) -> bool:
        if not self.wants.brand:
            return True
        target = bag(self.wants.brand)
        return bool(target <= bag(card.brand) or target <= bag(card.title))

    def right_thing(self, card: Card) -> str:
        if not self.wants.shelf or roots(self.wants.shelf) != {"smartphone"}:
            return "yes"
        return handset_check(card.facts, card.title, card.options)

    def gap_from_wish(self, card: Card) -> int:
        if not self.wants.wish_model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        body = squash(card.title + " " + " ".join(str(v) for v in card.facts.values()))
        same_model = squash(self.wants.wish_model) in body
        painted = (card.options + " " + str(card.facts.get("color") or "") + " " + card.title).lower()
        colour = self.wants.wish_colour
        same_colour = bool(colour) and re.search(r"(?<![a-z])" + re.escape(colour) + r"(?![a-z])", painted) is not None
        if same_model and same_colour:
            return 0
        return 1 if same_model else (2 if same_colour else 3)

    def described_words(self) -> set:
        if self._spoken_words is None:
            common = {"the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                      "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                      "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
                      "just", "dont", "lot", "not", "but", "get", "all", "kind", "every"}
            drop = common | bag(self.wants.shelf or "")
            self._spoken_words = {w for w in bag(self.wants.spoken) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._spoken_words

    def serves_the_use(self, card: Card) -> int:
        told = bag(" ".join(str(v) for v in card.facts.values()) + " " + card.title + " " + card.options)
        return len(self.described_words() & told)

    def standing(self, card: Card) -> tuple:
        """(may be bought, meets everything the shopper said) for one card."""
        wants, deck = self.wants, self.deck
        if card.stocked is not True or card.price is None or card.vetoed or card.pid in deck.spent_pages:
            return False, False
        if wants.ceiling is not None and card.price > wants.ceiling + SLACK:
            return False, False
        if not card.opened:
            return False, False
        for concept, value in wants.needs.items():
            if concept == wants.gate and card.value(concept) != value:
                return False, False
        said_so = self.on_the_right_shelf(card) and self.right_maker(card)
        thing = self.right_thing(card)
        strict = wants.errand in ("spec", "reason", "exactly")
        fine = said_so and (thing == "yes" if strict else thing != "no")
        if wants.errand == "sweep" and wants.phrase:
            matches = all(w in bag(card.title) for w in wants.phrase)
            fine, said_so = fine and matches, said_so and matches
        if wants.errand == "rank":
            named = (card.pid, card.sku) in wants.listed
            gauged = card.measure("refresh_rate") is not None and card.measure("screen_size") is not None
            fine = fine and named and gauged
            said_so = fine
        if wants.errand == "reason":
            citable = bool(card.brand) and not SECOND_HAND.search(card.title)
            if wants.gate:
                citable = citable and card.quotable(wants.gate) == wants.gate_value and card.page_value(wants.gate) == wants.gate_value
            fine = fine and citable
            said_so = fine
        if wants.errand == "fit" and deck.struck:
            fallen = deck.cards.get(deck.struck)
            if fallen is not None and fallen.listing and card.listing == fallen.listing:
                return False, False
        return fine, said_so

    def worth(self, card: Card, first: bool) -> tuple:
        wants = self.wants
        price = card.price if card.price is not None else 1e12
        if wants.errand == "rank":
            return (-(card.measure("refresh_rate") or 0.0), -(card.measure("screen_size") or 0.0), price, card.ref)
        unproven = 0 if self.right_thing(card) == "yes" else 1
        if wants.errand == "exactly":
            return (self.gap_from_wish(card), price, card.ref)
        if wants.errand == "fit":
            return (-self.serves_the_use(card), unproven, price, card.ref)
        missed = sum(1 for c, v in wants.needs.items() if c != wants.gate and card.value(c) != v)
        if wants.errand == "blend" and first and wants.soft_brand:
            liked = bag(wants.soft_brand)
            away = 0 if (liked <= bag(card.brand) or liked <= bag(card.title)) else 1
            return (away, missed, unproven, price, card.ref)
        return (missed, unproven, price, card.ref)

    def field(self, first: bool, forgiving: bool = False) -> list:
        runners = []
        for card in self.deck.cards.values():
            fine, said_so = self.standing(card)
            if fine or (forgiving and said_so):
                runners.append(card)
        if self.wants.errand == "exactly":
            if first:
                exact = [c for c in runners if self.gap_from_wish(c) == 0]
                if exact:
                    runners = exact
            else:
                # relax by the smallest step: the same model in another colour long
                # before any other model
                near = [c for c in runners if self.gap_from_wish(c) <= 1]
                if near:
                    runners = near
        runners.sort(key=lambda c: self.worth(c, first))
        return runners


SHORTLIST_BRIEF = (
    "You pick which catalog listings deserve a closer look for a shopper. Reply with the numbers "
    "of the most promising rows, best first, separated by spaces, and nothing else."
)
WORDING_BRIEF = (
    "You word catalog searches. Reply with tool calls only: one search or filter that would surface "
    "the products this request is about. Never call anything but search or filter."
)
SPEAKING_BRIEF = (
    "You answer a shopper in one short sentence using only the facts you are given. Name the product "
    "and the figures. Never promise anything the facts do not show."
)


def call_args(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class Runner:
    """Plays one problem: run the duel, prove the finalists, buy the winner."""

    def __init__(self, problem: dict) -> None:
        world = problem["environment"]
        self.binding = dict(world["binding"])
        self.view = world["policy_view"]
        self.tools = list(self.view.get("tools") or [])
        self.able = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.turns_allowed = int(self.view.get("max_steps") or 30)
        self.per_turn = max(1, min(16, int(self.view.get("max_calls_per_turn") or 1)))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "task")
        self.wants = Wants(str(self.view.get("query") or ""))
        self.deck = Deck()
        self.duel = Duel(self.wants, self.deck)
        self.began = time.monotonic()
        self.turn = 0
        self.script: list = []
        self.finished = False
        self.held: dict | None = None
        self.asked_opening = False
        self.pending_question: str | None = None
        self.queries: list = []
        self.sent: set = set()
        self.opens = 0
        self.struck_noted = False
        self.chased_kin = False
        self.forgiving = False
        engine = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.engines = ENGINES.get(engine) or ENGINES["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.engines = (pinned,) + tuple(m for m in self.engines if m != pinned)
        self.engine_at = 0

    def spare(self) -> float:
        return SECONDS_ALLOWED - (time.monotonic() - self.began)

    def consult(self, brief: str, prompt: str, tools: bool) -> dict | None:
        body = {"model": self.engines[min(self.engine_at, len(self.engines) - 1)],
                "messages": [{"role": "system", "content": brief}, {"role": "user", "content": prompt}],
                "temperature": 0}
        if tools:
            body["tools"] = [t for t in self.tools if (t.get("function") or {}).get("name") in ("search", "filter")]
            body["tool_choice"] = "required"
        for round_ in range(3):
            reply = GATEWAY.post("/inference/chat/completions", json_data=body)
            if isinstance(reply, dict) and reply.get("choices"):
                return reply["choices"][0].get("message") or {}
            if self.spare() < 30:
                break
            time.sleep(min(2.0 * (round_ + 1), 5.0))
            self.engine_at = min(self.engine_at + 1, len(self.engines) - 1)
        return None

    # -------------------------------------------------------------- the wire
    def post(self, envelope: dict) -> dict | None:
        for round_ in range(4):
            reply = GATEWAY.post("/environment/call", json_data=envelope)
            if isinstance(reply, dict) and reply.get("calls"):
                return reply
            if self.spare() < 20:
                break
            time.sleep(min(2.0 * (round_ + 1), 6.0))
        return None

    def play(self, moves: list) -> None:
        if self.held is not None:
            repeat = self.post(self.held["envelope"])
            if repeat is None:
                return
            kept = self.held
            self.held = None
            self.turn = kept["turn"]
            self.unpack(kept["moves"], repeat)
            if self.finished:
                return
        moves = [m for m in moves if m and m[0] in self.able][: self.per_turn]
        if not moves:
            return
        turn = self.turn + 1
        label = self.problem_id + "-turn" + str(turn)
        envelope = {**self.binding, "call_id": label, "idempotency_key": label, "turn": turn,
                    "calls": [{"call_id": label + "." + str(i + 1), "action": {"name": n, "args": a}}
                              for i, (n, a) in enumerate(moves)]}
        reply = self.post(envelope)
        if reply is None:
            self.held = {"envelope": envelope, "turn": turn, "moves": moves}
            return
        self.turn = turn
        self.unpack(moves, reply)

    def unpack(self, moves: list, reply: dict) -> None:
        self.deck.turn = self.turn
        answers = reply.get("calls") or []
        for spot, (name, args) in enumerate(moves):
            shell = (answers[spot].get("observation") if spot < len(answers) and isinstance(answers[spot], dict) else None) or {}
            seen = shell.get("observation") if isinstance(shell.get("observation"), dict) else shell
            seen = seen if isinstance(seen, dict) else {}
            if shell.get("error") and not seen.get("error"):
                seen = dict(seen)
                seen["error"] = shell.get("error")
            if shell.get("done") or seen.get("done"):
                self.finished = True
            self.shelve(name, seen)
            self.script.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": self.problem_id + "." + str(self.turn) + "." + str(spot + 1),
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)}}],
                "environment_result": {"calls": [answers[spot]] if spot < len(answers) else []},
            })
        spoke = reply.get("user_message")
        if isinstance(spoke, dict) and spoke.get("content"):
            self.listen(str(spoke["content"]))

    def shelve(self, name: str, seen: dict) -> None:
        if not isinstance(seen, dict) or seen.get("error"):
            return
        phrase = self.wants.phrase
        if name in ("search", "filter"):
            self.deck.take_rows(seen.get("results") or [], phrase)
        elif name == "view":
            self.deck.take_page(seen, phrase)
        elif name == "compare":
            for page in seen.get("compared") or []:
                self.deck.take_page(page, phrase)
        elif name == "inspect_stock":
            self.deck.take_stock(seen)
        elif name in ("add_to_cart", "remove_from_cart", "inspect_cart"):
            self.deck.take_cart(seen)
            if (seen.get("added") or {}).get("product_id"):
                self.struck_noted = False
                self.chased_kin = False

    def listen(self, said: str) -> None:
        self.remark("shopper: " + said[:170])
        cut = self.wants.lower_ceiling(said)
        if cut is not None:
            self.remark("ceiling lowered to %.2f %s" % (cut, self.wants.currency))
            for pid, sku in list(self.deck.cart):
                card = self.deck.cards.get(pid + "::" + sku)
                if card and card.price is not None and card.price > cut + SLACK:
                    card.vetoed = True
        if self.wants.errand in ("spec", "blend") and not self.wants.needs:
            taken = self.wants.take_need(said)
            if taken:
                self.remark("firm requirement: " + taken)
        if TURNED_TO_US.search(said):
            self.pending_question = said

    def remark(self, line: str) -> None:
        self.script.append({"role": "assistant", "content": "[note] " + line[:260]})

    # ------------------------------------------------------------- the moves
    def opening(self) -> str | None:
        if self.wants.errand == "spec" or self.wants.hidden_need:
            return ("Before I look: which hardware specification is your firm requirement "
                    "(screen size, resolution class, refresh rate or memory), and what value?")
        if self.wants.errand == "fit":
            return ("So I choose well: which of your preferences counts most, and how should I weigh "
                    "the rest after it?")
        return None

    def may_commit(self) -> bool:
        wants, deck = self.wants, self.deck
        if deck.adds:
            return True
        if wants.hidden_need and self.turn < self.turns_allowed * 0.6:
            return False
        if wants.errand in ("spec", "blend") and self.turn < 4 and self.spare() > 150:
            return False
        if wants.errand == "fit" and len(deck.listings) < 2:
            return False
        if wants.errand == "sweep":
            if len(deck.identities) < wants.identity_floor or len(self.queries) < 2:
                return self.turn >= self.turns_allowed * 0.6
        if wants.errand == "exactly" and wants.wish_model and self.turn < self.turns_allowed * 0.45:
            return any(self.duel.gap_from_wish(c) == 0 for c in self.duel.field(first=True))
        if wants.errand == "blend" and wants.soft_brand and self.turn < self.turns_allowed * 0.45:
            liked = bag(wants.soft_brand)
            return any(liked <= bag(c.brand) or liked <= bag(c.title) for c in self.duel.field(first=True))
        return True

    def fresh_enough(self, card: Card) -> bool:
        """A price checked before the market answered our stake is out of date."""
        if card.checked_on == 0:
            return False
        return not self.deck.adds or card.checked_on > self.deck.staked_on

    def unopened(self) -> list:
        """Listing rows worth opening, best guess first."""
        wants = self.wants
        waiting = []
        for ref, row in self.deck.rows.items():
            pid = ref.split("::", 1)[0]
            card = self.deck.cards.get(ref)
            if card is None or card.opened or pid in self.deck.spent_pages:
                continue
            if card.stocked is False or card.price is None:
                continue
            if wants.ceiling is not None and card.price > wants.ceiling + SLACK:
                continue
            title = str(row.get("title") or "")
            if not self.forgiving:
                if wants.phrase and not all(w in bag(title) for w in wants.phrase):
                    continue
                if wants.brand:
                    liked = bag(wants.brand)
                    if not (liked <= bag(str(row.get("brand") or "")) or liked <= bag(title)):
                        continue
                if wants.wish_model and not self.deck.adds:
                    squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
                    if squash(wants.wish_model) not in squash(title):
                        continue
            near = 0 if (wants.gate_value and wants.gate_value in scan(wants.gate, title)) else 1
            waiting.append((near, card.price, pid))
        waiting.sort()
        order = []
        for _near, _price, pid in waiting:
            if pid not in order:
                order.append(pid)
        return order

    def claims(self, card: Card) -> list:
        proved = [{"field": c, "value": v} for c, v in self.wants.needs.items()
                  if card.quotable(c) == v and card.page_value(c) == v]
        if not proved:
            return []
        if self.wants.ceiling is not None:
            proved.append({"field": "price", "value": "at most %.2f" % self.wants.ceiling})
        elif card.price is not None:
            proved.append({"field": "price", "value": "%.2f" % card.price})
        proved.append({"field": "in_stock", "value": "true"})
        if card.brand:
            proved.append({"field": "brand", "value": card.brand})
        return proved

    def buy(self, card: Card) -> tuple:
        args = {"product_id": card.pid, "sku": card.sku}
        if self.wants.errand == "reason":
            said = self.claims(card)
            if said:
                args["justification"] = {"claims": said}
        return ("place_test_order", args)

    def speak(self, said: str) -> str:
        card = None
        if self.deck.staked:
            held = self.deck.cards.get(self.deck.staked)
            card = held if held and not held.vetoed else None
        if card is None:
            field = self.duel.field(first=not self.deck.adds, forgiving=True)
            card = field[0] if field else None
        facts = []
        if card is not None:
            facts.append("product: " + card.title[:90])
            if card.price is not None:
                facts.append("price now: %.2f %s" % (card.price, self.wants.currency))
            facts.append("in stock: " + ("yes" if card.stocked else "no"))
            for concept, value in self.wants.needs.items():
                if card.value(concept) == value:
                    facts.append(concept.replace("_", " ") + ": " + value)
        if self.deck.struck:
            facts.append("the first item chosen is no longer valid and is being replaced")
        told = "; ".join(facts) or "nothing verified yet"
        reply = self.consult(SPEAKING_BRIEF, "Shopper said: " + said[:280] + "\nFacts: " + told, False)
        wording = str((reply or {}).get("content") or "").strip().replace("\n", " ")
        return wording[:400] if wording else ("Going with " + told)

    def look_wider(self) -> list:
        deck, wants = self.deck, self.wants
        if deck.struck and not self.chased_kin and "search" in self.able:
            fallen = deck.cards.get(deck.struck)
            if fallen is not None and fallen.title:
                self.chased_kin = True
                self.remark("looking for the same listing in another colour or size")
                return [("search", {"query": " ".join(fallen.title.split()[:8]), "in_stock": True, "k": 10})]
        waiting = self.unopened()
        if not waiting and not self.forgiving:
            self.forgiving = True
            waiting = self.unopened()
        if waiting and self.opens < 12 and "compare" in self.able:
            self.opens += 1
            batch = [("compare", {"product_ids": waiting[:5]})]
            if len(waiting) > 5 and self.per_turn > 1 and self.opens < 12:
                self.opens += 1
                batch.append(("compare", {"product_ids": waiting[5:10]}))
            return batch
        if waiting and "view" in self.able:
            return [("view", {"product_id": waiting[0]})]
        return self.ask_catalog()

    def ask_catalog(self) -> list:
        wants = self.wants
        allowed = 3 if wants.errand == "sweep" else 7
        if len(self.queries) >= allowed or self.spare() < 60:
            return []
        if not self.queries:
            reply = self.consult(WORDING_BRIEF, "Request:\n" + wants.spoken[:700], True)
            for call in ((reply or {}).get("tool_calls") or [])[:2]:
                name = (call.get("function") or {}).get("name")
                if name not in ("search", "filter"):
                    continue
                args = call_args(call)
                args.setdefault("k", 10)
                args.setdefault("in_stock", True)
                if name == "filter" and wants.ceiling is not None:
                    args.setdefault("max_price", wants.ceiling)
                mark = name + json.dumps(args, sort_keys=True)
                if mark not in self.sent:
                    self.sent.add(mark)
                    self.queries.append(mark)
                    return [(name, args)]
        for name, args in self.spelled_out():
            mark = name + json.dumps(args, sort_keys=True)
            if mark in self.sent:
                continue
            self.sent.add(mark)
            self.queries.append(mark)
            return [(name, args)]
        return []

    def spelled_out(self) -> list:
        wants = self.wants
        out = []
        if "filter" in self.able and wants.shelf:
            base = {"category": wants.shelf, "in_stock": True, "k": 10}
            if wants.ceiling is not None:
                base["max_price"] = wants.ceiling
            if wants.brand:
                out.append(("filter", dict(base, brand=wants.brand)))
            out.append(("filter", base))
        if "search" in self.able:
            said = wants.spoken_need()
            for phrase in (" ".join(wants.phrase),
                           wants.wish_model or "",
                           " ".join(x for x in (wants.wish_model, wants.wish_colour) if x),
                           " ".join(x for x in (wants.brand, said, wants.shelf) if x),
                           " ".join(x for x in (said, wants.shelf) if x),
                           wants.shelf or ""):
                if phrase:
                    out.append(("search", {"query": phrase, "in_stock": True, "k": 10}))
        return out

    def next_moves(self) -> list:
        deck, wants, duel = self.deck, self.wants, self.duel
        if self.pending_question and "message" in self.able:
            said, self.pending_question = self.pending_question, None
            return [("message", {"content": self.speak(said)})]
        if not self.asked_opening:
            self.asked_opening = True
            question = self.opening()
            if question and "message" in self.able:
                return [("message", {"content": question})]
        if wants.errand == "rank" and "compare" in self.able:
            shut = [pid for pid, _sku in wants.listed if not deck.cards.get(pid + "::" + _sku, Card(pid, _sku)).opened]
            if shut:
                return [("compare", {"product_ids": shut[i:i + 5]}) for i in range(0, min(len(shut), 10), 5)]
        # a stake is on the table: prove it, then buy it or let the duel resume
        if deck.staked:
            staked = deck.cards.get(deck.staked)
            if staked is not None:
                if not self.fresh_enough(staked):
                    return [("inspect_stock", {"product_id": staked.pid, "sku": staked.sku})]
                standing, _ = duel.standing(staked)
                if standing and (staked.pid, staked.sku) in deck.cart:
                    challengers = [c for c in duel.field(first=False) if c.ref != staked.ref]
                    if not challengers or duel.worth(staked, False) <= duel.worth(challengers[0], False):
                        return [self.buy(staked)]
                if not standing and not self.struck_noted:
                    self.struck_noted = True
                    deck.struck = staked.ref
                    staked.vetoed = True
                    deck.spent_pages.add(staked.pid)
                    self.remark("the market went against " + staked.ref)
        clearing = []
        for pid, sku in deck.cart:
            held = deck.cards.get(pid + "::" + sku)
            if held is not None and (held.vetoed or held.stocked is False) and "remove_from_cart" in self.able:
                clearing.append(("remove_from_cart", {"product_id": pid, "sku": sku}))
                break
        field = duel.field(first=not deck.adds, forgiving=self.forgiving)
        if field:
            stale = [c for c in field[:4] if not self.fresh_enough(c)]
            if stale:
                return clearing + [("inspect_stock", {"product_id": c.pid, "sku": c.sku}) for c in stale]
            if clearing:
                return clearing
            if self.may_commit():
                best = field[0]
                if (best.pid, best.sku) not in deck.cart:
                    return [("add_to_cart", {"product_id": best.pid, "sku": best.sku})]
        if (deck.struck and wants.errand == "exactly" and not self.chased_kin
                and not any(duel.gap_from_wish(c) <= 1 for c in field)):
            kin = self.look_wider()
            if kin:
                return clearing + kin
        wider = self.look_wider()
        if wider:
            return clearing + wider
        return clearing

    def salvage(self) -> None:
        """Out of room: reach a valid order in as few groups as possible."""
        for pid, sku in reversed(self.deck.cart):
            card = self.deck.cards.get(pid + "::" + sku)
            if card is None or self.finished:
                continue
            standing, _ = self.duel.standing(card)
            if standing and self.fresh_enough(card):
                self.play([self.buy(card)])
                return
        if self.finished or self.turns_allowed - self.turn < 3:
            return
        field = self.duel.field(first=not self.deck.adds, forgiving=True)
        if not field:
            return
        pick = field[0]
        if (pick.pid, pick.sku) not in self.deck.cart:
            self.play([("add_to_cart", {"product_id": pick.pid, "sku": pick.sku})])
            if self.finished:
                return
        self.play([("inspect_stock", {"product_id": pick.pid, "sku": pick.sku})])
        if self.finished:
            return
        standing, _ = self.duel.standing(pick)
        if standing:
            self.play([self.buy(pick)])

    def go(self) -> list:
        quiet = 0
        while not self.finished and self.turn < self.turns_allowed:
            if self.spare() < 40 or self.turns_allowed - self.turn <= 3:
                self.salvage()
                break
            moves = self.next_moves()
            if not moves:
                if self.forgiving:
                    self.salvage()
                    break
                self.forgiving = True
                self.remark("nothing left under the strict reading; judging more forgivingly")
                continue
            before = (len(self.deck.cards), self.deck.adds, len(self.deck.rows))
            self.play(moves)
            if (len(self.deck.cards), self.deck.adds, len(self.deck.rows)) == before:
                quiet += 1
                if quiet >= 3:
                    self.forgiving = True
                    quiet = 0
            else:
                quiet = 0
        return self.script


def agent_main(problem_data: dict) -> list:
    try:
        return Runner(problem_data).go()
    except Exception as bother:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(bother).__name__ + ": " + str(bother)[:200]}]


__all__ = ["agent_main"]
