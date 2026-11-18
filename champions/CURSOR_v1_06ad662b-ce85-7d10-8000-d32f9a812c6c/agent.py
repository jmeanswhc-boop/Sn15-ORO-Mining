"""Shopping agent that asks about held-back requirements, then searches, carts and orders."""

from __future__ import annotations

import json
import math
import re
from collections import Counter as Tally
from difflib import SequenceMatcher
from itertools import zip_longest
from os import getenv
from time import monotonic
from typing import Any

from src.agent.proxy_client import ProxyClient

MODELS = {
    "chutes": ["zai-org/GLM-5.2-TEE"],
    "openrouter": ["z-ai/glm-5.2"],
}
# per-seat models: RECOMMENDED by default, ALTERNATIVE for comparison; MODEL_SET selects one
SONNET, SOL, HAIKU, GEMINI_FLASH, GLM_NEXT, QWEN_FLASH, GLM_BASE = ("anthropic/claude-sonnet-5.5", "openai/gpt-6-sol", "anthropic/claude-haiku-5.5",
                                                                    "google/gemini-3.8-flash", "z-ai/glm-5.3", "qwen/qwen3.8-flash", "z-ai/glm-5.2")
RECOMMENDED = {"plan": SOL, "judge": SOL, "hear": SONNET, "rank_needs": SOL, "priority_unit": HAIKU, "colour_check": HAIKU, "vocabulary": HAIKU,
               "search_better": GEMINI_FLASH, "rank": HAIKU, "names": HAIKU, "same": HAIKU, "words": HAIKU, "search_plan": GLM_BASE}
ALTERNATIVE = {"plan": SOL, "judge": GLM_NEXT, "hear": SOL, "rank_needs": GLM_NEXT, "priority_unit": QWEN_FLASH, "colour_check": QWEN_FLASH, "vocabulary": QWEN_FLASH,
               "search_better": GLM_NEXT, "rank": QWEN_FLASH, "names": QWEN_FLASH, "same": QWEN_FLASH, "words": QWEN_FLASH, "search_plan": SOL}
MODEL_SET = RECOMMENDED
SEAT_MODELS: dict[str, dict[str, list[str]]] = {"openrouter": {seat: [model, "z-ai/glm-5.2"] for seat, model in MODEL_SET.items()}}
CALLS_PER_TURN = 16
OPEN_BATCH = 5
OPEN_LIMIT = 400
SEARCH_LIMIT = 120            # counts shelf filters as well as searches
JUDGE_CARDS = 18
JUDGE_LIMIT = 120
EPISODE_SECONDS = 200.0
TOOL_REMINDER = "\n\nReply by calling the tool."
INCOMPLETE_REMINDER = "\n\nThe previous reply was incomplete ({}): write every field in full."
MODEL_CUTOFF = 210.0          # no model call starts past this many seconds
QUICK_TIMEOUT = 25
TURN_RESERVE = 6
CHANGE_RESERVE = 4            # turns held back while a shopper change is followed
READ_AHEAD = 5                # extra listings whose stock is rechecked after a cart add
FIRST_CART_ROOM = 9           # turns kept after the first cart add so changes can arrive
READ_FLOOR = 5                # the rows of the latest searches are opened only while more turns than this are left (the cart, the order and a spare)
EXPLORE_ROUNDS = 3
COLOUR_CHECKS = 3
EXPLORE_PER_ROUND = 12
SWAP_MIN = 0.2                # least share of read listings that must state the figure to outrank another amount
SWAP_RATIO = 2.0              # and how many times it must outnumber the first amount's listings
SHOWN_FIRST = 2               # picks that listings show which go ahead of picks only the request names
OVER_REACH = 1.35             # price factor up to which an over-limit listing is still read
NEAREST_REACH = 1.25          # price factor up to which an over-limit listing may be named nearest
OPEN_REACH = 1.6              # price factor for opening a row, since another variant may be cheaper
QUERY_LENGTH = 200
SIBLING_LIMIT = 40            # seen titles searched together with the values
PROBE_LIMIT = 12              # titles whose variants are searched by wanted colour or amount
FAMILY_LIMIT = 10             # seen titles searched on their own
MAKER_LIMIT = 4               # shelves filtered by the make of a matching listing
THINK_TOKENS = 3000
MIN_TOKENS = 900              # floor for every call, since a provider may think before the tool call
SPEND_CAP = 0.08              # an episode past this spend asks only the cheap fallback model; past 1.5x it asks nothing
SAID_NEEDED = r"(?:(?:a|an|is|be|it|the)\s+)*(?:required|requirement|needed|necessary|essential|must|need|included|present|chosen|selected|provided)"
# quantities whose units carry a base-unit size, which allows converting between units
SIZED_QUANTITIES = ("length", "weight", "volume", "time")
WORD_ROUNDS = 4              # vocabulary rounds over the words of newly read listings
WORDS_SENT = 160
LATE_ROUNDS = 2              # extra rounds for words of listings that could still be ordered
FLUSH_ROUNDS = 4             # extra rounds, before a choice, for the unsent words of listings that could still be ordered
UNITS_SENT = 40
# fact labels about terms of sale, page markup or naming rather than the item itself
SALE_LABELS = re.compile(r"warrant|condition|deliver|shipping|service|polic|dangerous|location|postal|origin|certif|description|html|span|"
                         r"div|margin|text_|style_line|li_|video|img|^model$|^brand$|^name$|^series$|^year$|variation", re.I)
CHANGE_WORDS = r"\b(?:instead|chang\w+|switch\w*|swap\w*|forget|no longer|rather than|replac\w+)\b"
BOUNDED = re.compile(r"\b(?:up ?to|below|under|less than|at most|max(?:imum)?|over|above|more than|at least|min(?:imum)?|within)\s*(?:of\s+)?\d|\d\s*[a-z]{0,10}\s+(?:and|or|&)\s+(?:below|under|less|above|over|more|up)\b")     # a figure used as a bound
NOT_A_PROPERTY = re.compile(r"(?<![\w-])(?:categor(?:y|ies)|budget|prices?|availab(?:le|ility)|(?:in[- ]|out of )stock|stock(?: status| level| availability))(?![\w-])|^stock$")
# a question names one property, alone and singular, so the shopper's reader can tell what is meant
ASK_ALONE = "What {} do you need?"
# opening-turn question about the use, sent only when the request invites consultation
PLAIN_QUESTION = "What will you use it for?"
# closing question that hands the reason back to the shopper
REASON_QUESTION = "You want the {} for {}. For that use, what does it need to have?"
CLOSE_AFTER = 3               # property questions per reason before the use is asked about
REASON_AFTER = 6              # property questions per reason before the reason is handed back
CLOSING_ROOM = 3              # extra questions the closing questions may use
PLAIN_DRY_ASKS = 0            # property questions for a reason implying none, after the use question drew nothing (live: 1 of 94 revealed)
FLOOR_QUESTIONS = 5           # costed questions past the free ones; beyond them asking costs nothing
USE_FIRST = True              # ask about the use before handing a reason back
CLOSING_QUESTIONS = True
ASK_WIDE = 4
ASK_AFTER_PAIR = 6            # property questions after the shopper picked between two wishes
WAIT_TURNS = 2                # turns a question waits while only listings over the limit meet the wish the shopper picked
ASK_UNHEARD = 3               # property questions while no held-back requirement has surfaced
ASK_PER_REASON = 6
ASK_NARROW = 3                # questions of three or four properties before dropping to pairs
ASK_LIMIT = 10
ASK_AFTER_ONE = 2             # property questions once one requirement is out and another is expected
ASK_TURNS = 10                # turns kept free of questions for the cart, changes and order
ASK_SHELF = 14
MIXED_COLOURS = 2             # a word filed under more colours than this names none of them
BEST_ROUNDS = 3
BEST_EXTRA = 2                # extra rounds, each only if the previous one found a better figure
FORM_LIMIT = 6
APART_LIMIT = 6               # searches with a decimal figure set apart from its unit that follow the value searches
HUNT_RULES = 4
READ_FIGURES = 3              # figures read before a lack of better ones is believed
FIGURE_LIMIT = 150
# fit confidence, from the firmest evidence (store-stated values) to the faintest (other wording only)
FIRM_FIT, SURE_FIT, LIKELY_FIT, UNSURE_FIT, FAINT_FIT = 1.0, 0.85, 0.5, 0.3, 0.1
AGAINST_FIT = 0.05            # the page states a different value: scores under a merely unclear listing
LIST_ENTRIES = 4              # most entries of a listed fact for a member value to count as the fact's own

_proxy = ProxyClient(timeout=90, max_retries=2)
_proxy_quick = ProxyClient(timeout=QUICK_TIMEOUT, max_retries=1)


def lowered(text: Any) -> str:
    # a double prime after a figure stands for inches
    plain = re.sub(r"(?<=\d)\s?(?:''|\u2033|\u201d|\")(?=[\s,;:/)\]x\u00d7*-]|$)", " inch", str(text or "").lower())
    return " ".join(plain.split())


def word_list(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", lowered(text))


def singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 4 and word.endswith("ie"):
        return word[:-2] + "y"
    if len(word) > 4 and word.endswith(("ve", "fe")):
        return word[:-2] + "f"      # folds -ve and -fe endings into -f
    return word


def closed_up(text: Any) -> str:
    """Joins a figure to the unit that follows it (3 cm -> 3cm)."""
    return re.sub(r"(\d)\s+(?=[a-z]{1,4}\b)", r"\1", lowered(text))


def same_word(one: str, other: str) -> bool:
    if one == other:
        return True
    short, long = sorted((one, other), key=len)
    return len(short) >= 4 and not short[0].isdigit() and long.startswith(short) and len(long) - len(short) <= 2


def stems(text: Any) -> set[str]:
    return {singular(w) for w in word_list(closed_up(text))}


def label_key(text: Any) -> str:
    """Category label as singular words, so two spellings compare equal."""
    return " ".join(singular(w) for w in word_list(str(text).replace("&", " and ")) if w != "and")


def shelf_key(text: Any) -> str:
    """Category label as singular words with 'and' (or '&') kept, so a parent shelf joining two names stays apart."""
    return " ".join(singular(w) for w in word_list(str(text).replace("&", " and ")))


NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
                "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def figures_in(text: Any) -> list[float]:
    """Numbers in a text, as digits or spelled out, with comma separators handled."""
    low = re.sub(r"(?<=\d),(?=\d{1,2}(?!\d))", ".", re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", lowered(text)))
    found = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", low)]
    total, current, inside, scale = 0, 0, False, False
    for word in re.findall(r"[a-z]+", low) + [""]:
        if word in NUMBER_WORDS:
            current, inside, scale = current + NUMBER_WORDS[word], True, False
        elif word in ("hundred", "thousand") and inside:
            total, current, scale = (total, (current or 1) * 100, True) if word == "hundred" else (total + (current or 1) * 1000, 0, True)
        elif word == "and" and inside and scale:
            continue
        else:
            if inside:
                found.append(float(total + current))
            total, current, inside, scale = 0, 0, False, False
    return found


def money(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def one_word_terms(terms: Any) -> set[str]:
    out: set[str] = set()
    for term in terms if isinstance(terms, list) else []:
        word = lowered(term).strip()
        if re.fullmatch(r"[a-z]+", word):
            out.update((word, singular(word)))
    return out


class Glossary:
    """Colour, material and unit vocabulary the reader supplies during an episode."""

    def __init__(self) -> None:
        self.colour_words: dict[str, tuple[str, ...]] = {}   # word -> colour families it is filed under
        self.material_spellings: dict[str, set[str]] = {}
        self.material_answers: dict[str, list[set[str]]] = {}
        self.unit_by_spelling: dict[str, str] = {}
        self.unit_size: dict[str, tuple[str, float]] = {}    # unit -> (what it measures, size in the base unit)
        self.unit_quantity: dict[str, str] = {}
        self.sent: set[str] = set()
        self.omitted: set[str] = set()                       # words an answer left out, sent again once
        self.answered_values: set[str] = set()
        self.pages: set[str] = set()
        self.pages_seen = 0
        self.rounds = 0
        self.word_rounds = 0                                 # counts only rounds that carried listing words
        self.late_rounds = 0
        self.flushes = 0
        self.item_words: set[str] = set()
        self.checked: set[str] = set()
        self.refused: set[str] = set()
        self.checks = 0

    def colours_of_word(self, word: str) -> tuple[str, ...]:
        """Colours a word stands for; item-name words count only as a colour they name."""
        if word in self.colour_words:
            families = self.colour_words[word]
            if len(families) > MIXED_COLOURS:
                return ("multi",)
            return () if word in self.item_words and word not in families else families
        return (word,) if any(word in families for families in self.colour_words.values()) else ()

    def names_a_colour(self, text: Any) -> bool:
        return any(self.colours_of_word(word) for word in word_list(text))

    def may_be_colour(self, value: str) -> bool:
        return self.names_a_colour(value) or lowered(value).strip() not in self.answered_values

    def colour_of(self, value: str) -> str:
        found = {family for word in word_list(value) for family in self.colours_of_word(word)}
        return next(iter(found)) if len(found) == 1 else ""

    def colours_named_in(self, text: Any) -> list[tuple[str, ...]]:
        found: list[tuple[str, ...]] = []
        for word in re.findall(r"[\w-]+", str(text or "").lower()):
            if self.colours_of_word(word):
                found.append(self.colours_of_word(word))
            elif "-" in word:
                parts = [self.colours_of_word(part) for part in word.split("-") if self.colours_of_word(part)]
                found += parts if len(parts) == 1 else []
        return found

    def words_for_colour(self, family: str) -> list[str]:
        return [word for word, families in self.colour_words.items() if family in families and len(families) <= MIXED_COLOURS]

    def material_words(self, value: str) -> set[str]:
        return self.material_spellings.get(" ".join(word_list(value)), set())

    def other_material_words(self, family: set[str]) -> list[set[str]]:
        return [words for words in self.material_spellings.values() if words != family]

    def unit_named(self, word: str) -> str:
        plain = lowered(word).strip()
        return self.unit_by_spelling.get(plain) or self.unit_by_spelling.get(Errand.unit_word(plain), plain)

    def listing_units(self, unit: str) -> dict[str, float]:
        """Other units a listing may use for this quantity, with their size ratio to this unit."""
        known = self.unit_size.get(unit)
        if known is None:
            return {}
        out: dict[str, float] = {}
        for name, (kind, size) in self.unit_size.items():
            if kind != known[0]:
                continue
            ratio = size / known[1]
            step = max(ratio, 1 / ratio)
            tens = 10.0 ** round(math.log10(step))
            if abs(step - tens) <= 0.005 * step:
                step = tens
            elif kind == "time":
                if step > 12.5 or abs(step - round(step)) > 0.02 * step:
                    continue
                step = float(round(step))     # near-whole time ratios (12 months a year, about 30 days a month)
            elif step >= 12:
                continue
            out[name] = step if ratio >= 1 else 1 / step
        return out

    def weighs(self, unit: str) -> bool:
        return self.unit_quantity.get(unit) in ("weight", "power")

    @staticmethod
    def dumped(words: set[str], sent: set[str]) -> bool:
        """True for a long list that mostly echoes the words the reader was sent."""
        return len(words) >= 9 and len(words & sent) * 2 > len(words)

    @staticmethod
    def material_entry(entry: Any, listing_words: set[str]) -> set[str]:
        given = {lowered(w).strip() for w in entry.get("words") or []} if isinstance(entry, dict) and isinstance(entry.get("words"), list) else set()
        given = {w for w in given if re.fullmatch(r"[a-z]+", w)}
        return set() if Glossary.dumped(given, set(listing_words)) else one_word_terms(sorted(given))

    def another_unit(self, name: str, kind: str, size: float | None) -> bool:
        """True when the spelling is already a known unit of another quantity or size."""
        known = self.unit_by_spelling.get(name)
        if known is None:
            return False
        if self.unit_quantity.get(known) and kind and self.unit_quantity[known] != kind:
            return True
        had = self.unit_size.get(known)
        return bool(had and size and abs(had[1] - size) > 0.005 * max(had[1], size))

    def take_in(self, answer: dict, values: list[str], listing_words: set[str] = frozenset(), unit_words: set[str] = frozenset()) -> None:
        """Merges one vocabulary answer into the glossary and marks its values as answered."""
        for entry in answer.get("colours") if isinstance(answer.get("colours"), list) else []:
            if not isinstance(entry, dict):
                continue
            word = lowered(entry.get("word")).strip()
            colours = entry.get("colours") if isinstance(entry.get("colours"), list) else []
            families = tuple(dict.fromkeys(lowered(c).strip() for c in colours if lowered(c).strip()))
            if families and re.fullmatch(r"[a-z]{3,}", word) and word not in self.colour_words and word not in self.refused:      # too short to name a colour
                self.colour_words[word] = families
        for entry in answer.get("materials") if isinstance(answer.get("materials"), list) else []:
            if not isinstance(entry, dict):
                continue
            name = " ".join(word_list(entry.get("material")))
            words = self.material_entry(entry, listing_words)
            if len(name.split()) >= 2:
                # a lone word equal to the last word of a longer material name is only a wider kind
                words = {w for w in words if singular(w) != singular(name.split()[-1])}
            if name and words:
                # vote across rounds: a word stays when enough of the answers include it
                given = self.material_answers.setdefault(name, [])
                given.append(words)
                votes = Tally(w for one in given for w in one)
                need = 1 if len(given) <= 2 else max(2, math.ceil(0.4 * len(given)))
                self.material_spellings[name] = {w for w, n in votes.items() if n >= need} or given[0]
        parsed = []
        for entry in answer.get("units") if isinstance(answer.get("units"), list) else []:
            if not isinstance(entry, dict):
                continue
            spellings = entry.get("spellings") if isinstance(entry.get("spellings"), list) else []
            names = list(dict.fromkeys(n for n in (lowered(x).strip() for x in [entry.get("unit"), *spellings]) if re.fullmatch(r"[a-z]+", n)))
            if names and not self.dumped(set(names), set(unit_words)):
                parsed.append((lowered(entry.get("unit")).strip(), names, lowered(entry.get("quantity")).strip(), money(entry.get("size"))))
        def apart(one: tuple, other: tuple) -> bool:
            """True when two entries are different units, not one unit given in two forms."""
            if one[2] and other[2] and one[2] != other[2]:
                return True
            if one[3] and other[3]:
                return abs(one[3] - other[3]) > 0.005 * max(one[3], other[3])
            return not Errand.same_unit(one[0], other[0])
        for entry in parsed:
            own, names, kind, size = entry
            rivals = [other for other in parsed if other is not entry and apart(entry, other)]

            def ours(name: str) -> bool:
                """True when the spelling is this unit's, not a rival's or one already known."""
                if name == own:
                    return True
                givers = [other for other in rivals if name in other[1]]
                if any(name == other[0] for other in rivals) or givers and (not Errand.same_unit(name, own) or any(Errand.same_unit(name, other[0]) for other in givers)):
                    return False
                return not self.another_unit(name, kind, size)
            names = [name for name in names if ours(name)]
            if not names:
                continue
            # canonical unit: a known one (also via singular/plural), otherwise the shortest spelling
            def same_size(known: str) -> bool:
                have = self.unit_size.get(known)
                return not (size and have) or abs(size - have[1]) <= 0.005 * max(size, have[1])
            canon = next((self.unit_by_spelling[n] for n in names if n in self.unit_by_spelling),
                         next((known for spelling, known in self.unit_by_spelling.items()
                               if (not kind or not self.unit_quantity.get(known) or self.unit_quantity[known] == kind)
                               and any(Errand.unit_word(spelling) == Errand.unit_word(n) for n in names)),
                              # one unit given in two rounds under two spellings ('watt' then 'w') is still one unit
                              next((known for spelling, known in self.unit_by_spelling.items()
                                    if kind and kind not in SIZED_QUANTITIES + ("count", "other", "none") and self.unit_quantity.get(known) == kind and same_size(known)
                                    and any(Errand.same_unit(spelling, n) for n in names)), min(names, key=lambda n: (len(n), n)))))
            for name in names:
                self.unit_by_spelling.setdefault(name, canon)
                self.unit_by_spelling.setdefault(Errand.unit_word(name), canon)
            self.unit_quantity.setdefault(canon, kind)
            if kind in SIZED_QUANTITIES and size:
                self.unit_size.setdefault(canon, (kind, size))
        self.answered_values |= {lowered(v).strip() for v in values}


GLOSSARY = Glossary()     # replaced by a fresh instance every episode


def tool_spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


def strings(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def act(name: str, **args: Any) -> dict:
    return {"name": name, "args": {k: v for k, v in args.items() if v is not None}}


PLAN_TOOL = tool_spec("plan", "The shopper's request, written down for shopping.", {
    "item": {"type": "string", "description": "The shopper's own words for the kind of thing, copied from the request exactly as written, without the article and without the requirements."},
    "category": {"type": "string", "description": "A store category, only when the shopper names one as a category; otherwise empty."},
    "requirements": {"type": "array", "description": "Each property the item itself must have, stated in the request.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string", "description": "The name of the property in the shopper's words (colour, material, pattern, style, capacity, ...)."},
            "value": {"type": "string", "description": "The required value, as the shopper wrote it, with its unit."},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The value in another form: another spelling of the word, or the amount given in a different unit. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value (a synonym; not a part or a wider kind of it). Empty when there is none."}},
        "required": ["property", "value"]}},
    "budget": {"type": "number", "description": "The most the shopper will pay; 0 when none is named."},
    "brand": {"type": "string", "description": "A brand the shopper requires; empty when none, and empty when a brand is only preferred or a first choice."},
    "priority": {"type": "string", "description": "What the shopper hopes to get the most or the least of among the items that fit, in the shopper's words (for example the lowest price). Most requests name one: read the request again before leaving this empty."},
    "queries": strings("Exactly six short catalogue searches (two to five words) that could return the item: the item words with the required values, with a synonym or another spelling of a value where one exists."),
    "priority_queries": strings("When a priority is given: up to four more searches that reach the listings standing best by it; otherwise empty."),
    "doubted_pair": {"type": "array", "items": {"type": "string"}, "maxItems": 2,
                     "description": "When the shopper wishes for two things and doubts one item can have both: the two wishes, each as its value with what it describes (for example 'a leather strap', 'the Acme brand'). Empty otherwise."},
    "purposes": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                 "description": "Each use or recipient the request gives for the item, one entry each, in the shopper's words. Empty when it gives none."},
    "invitation": {"type": "string", "description": "The request's exact words that tell the assistant to raise a question with the shopper before deciding (to ask, check or confirm with the shopper whether the purpose alters what the item calls for); empty when the request holds no such instruction."},
    "look_at": {"type": "integer", "description": "When the request tells the assistant to open, compare or check at least some number of matching listings before choosing: that number; otherwise 0."},
    "open_properties": {"type": "array", "items": {"type": "string"}, "maxItems": 6,
                        "description": "Properties of this kind of item that the request does not settle and a shopper could still have a requirement about, the likeliest first, each in one to three plain words."},
}, ["item", "requirements", "budget", "priority", "queries", "doubted_pair", "purposes", "invitation", "look_at", "open_properties"])

HEAR_TOOL = tool_spec("hear", "What the shopper's new line changes.", {
    "adds": {"type": "array", "description": "Each requirement of the item the line states that was not known before.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string", "description": "The property's name as the shopper said it, with every word of that name (the material of one part is named with the part)."},
            "value": {"type": "string", "description": "The value alone, as the shopper said it: an amount with its unit; never the property's own name, and never a word that only says it is needed."},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The value in another form: another spelling of the word, or the amount given in a different unit. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value (a synonym; not a part or a wider kind of it). Empty when there is none."},
            "replaces": {"type": "string", "description": "When this value takes the place of a value already known, that earlier value as the known requirements list it; empty when the value is added beside what is known."}},
        "required": ["property", "value"]}},
    "drops": strings("Properties whose earlier requirement the line withdraws or replaces."),
    "budget": {"type": "number", "description": "A new spending limit the line states; 0 when it states none."},
    "priority": {"type": "string", "description": "What the line says matters most between fitting items; empty when nothing."},
    "asks": {"type": "boolean", "description": "True when the line asks the assistant something or expects an answer."},
    "about": {"type": "string", "description": "Only when the assistant had just asked about several properties and the line gives a value: the one of them that value belongs to, copied exactly as the assistant named it. Empty when it belongs to none of them or nothing was asked."},
    "about_change": {"type": "boolean", "description": "True when the line only reacts to the stock or the price of the chosen item having changed."},
    "queries": strings("Up to six short catalogue searches that use what the line added (try a synonym or another spelling of the value too); empty when it added nothing."),
}, ["adds", "drops", "budget", "asks", "queries"])

ASK_TOOL = tool_spec("ask_about", "The properties of the item worth asking the shopper about.", {
    "properties": {"type": "array", "items": {"type": "string"}, "maxItems": 8,
                   "description": "The properties to ask about next, the likeliest to carry a requirement first, each named in one to three plain words as a shopper would say it."},
    "measures": strings("One entry for every unit shown under 'Amounts': the unit as shown, a colon, and the name of what it measures - a noun of one or two "
                        "words, as a shopper would name that property of the item (for a unit of mass: weight) - or 'none' when it is no unit of measure of "
                        "the item itself (part of a code, a word that only follows a number, a span of time). Put the entries for how many and how much "
                        "first, and those for how long or how big last.")},
    ["properties"])

JUDGE_TOOL = tool_spec("judge", "Which listings state every requirement.", {
    "listings": {"type": "array", "description": "One entry for every listing shown, in the order shown.", "items": {
        "type": "object", "properties": {
            "n": {"type": "integer", "description": "The listing's number."},
            "kind": {"type": "string", "enum": ["K", "X"], "description": "K when the listing is the kind of thing asked for, X when it is not."},
            "rank_value": {"type": "number", "description": "The listing's number for the shopper's priority when one is given, else 0."},
            "checks": {"type": "array", "description": "One entry per requirement, in order.", "items": {
                "type": "object", "properties": {
                    "words": {"type": "string", "description": "The listing's exact words that state the required value, copied as written; empty when none do."},
                    "met": {"type": "boolean", "description": "True only when those words state the required value."}},
                "required": ["words", "met"]}}},
        "required": ["n", "kind", "checks"]}},
    "queries": strings("When no listing meets every requirement: up to five other short searches worth trying; otherwise empty."),
}, ["listings"])

NAMES_TOOL = tool_spec("names", "What each text names for one property.", {
    "marks": {"type": "array", "items": {"type": "string"}, "description": "One entry per text, in order: '<number> <only|among|other|none>'."}},
    ["marks"])

SAME_TOOL = tool_spec("same", "How each entry relates to the wanted value.", {
    "marks": {"type": "array", "items": {"type": "string"}, "description": "One entry per text, in order: '<number> <same|kind|other|none>'."}},
    ["marks"])

WORDS_TOOL = tool_spec("words", "Other words sellers use for a value.", {
    "words": {"type": "array", "items": {"type": "string"}, "description": "Up to six, the most usual first; empty when there is none."}},
    ["words"])

COLOUR_CHECK_GUIDE = """You check a list of words for a shopping assistant. The words come from the variant names of store listings, and each was filed under the colour that was asked for. Say which of the words, as a shopper would write it in a shop's colour filter, is by itself the name of that colour or of a shade of it. A word is not one when it is a scent, flavour, product line, collection, character or place name, a brand, an abbreviation or cut-off word, a code, or only describes a colour (light, dark, bright, assorted, matte). Fill the colour_check tool with the numbers of the words that are."""

COLOUR_CHECK_TOOL = tool_spec("colour_check", "The words that name the colour.", {
    "names": strings("The numbers of the words that name the colour or one of its shades; empty when none does.")}, ["names"])

RANK_TOOL = tool_spec("rank", "Each listing's figure for the shopper's priority.", {
    "values": {"type": "array", "items": {"type": "string"}, "description": "One entry per listing, in order: '<number> <figure>' with the figure as a plain number, written in a single unit for all listings, or '<number> none' when the listing's words state none."},
    "better": {"type": "string", "enum": ["higher", "lower"], "description": "Whether a higher or a lower figure is better for this priority."},
    "unit": {"type": "string", "description": "The one unit every figure is written in (empty for a plain count)."}},
    ["values", "better", "unit"])

MEASURE_TOOL = tool_spec("priority_unit", "The quantity the shopper's priority is about and its unit.", {
    "quantity": {"type": "string", "description": "The quantity of which the shopper hopes to get the most or the least, in a few words; empty when there is none."},
    "unit": {"type": "string", "description": "The unit listings write that quantity in, lower case; empty for a plain count or the price."}},
    ["quantity", "unit"])

BEST_TOOL = tool_spec("search_better", "Catalogue searches for listings that stand better by the shopper's priority.", {
    "queries": strings("Up to eight new searches of two to six words each, different from those tried."),
}, ["queries"])

SEARCH_TOOL = tool_spec("search_plan", "New catalogue searches.", {
    "queries": strings("Six new searches of two to five words each, different from those tried."),
}, ["queries"])

RECHECK_NOTE = ("A first reading of this request found no requirement of the item itself. Read it again before leaving the list empty: a requirement may "
                "stand inside the words that name the item (a word before the kind of thing that says what it is made of, whose make it is or what kind it is), "
                "or in a sentence of its own about the item. List every one the shopper states.")
PRIORITY_NOTE = ("A first reading of this request found no priority. Read it again before leaving it empty: a request often says, in a sentence of its own or next to the budget, what the shopper hopes to get the most or the least of among the items that fit. Give it in the shopper's words, or leave it empty when the request has none.")
PLAN_GUIDE = """You prepare a purchase for a shopping assistant. Read the shopper's request, then fill the plan tool using its words only.
- item: the words the shopper uses for the kind of thing, copied completely and exactly as written, with a leading word such as 'other' and any '&' or '/' kept (the store files its listings under such names); do not shorten them or replace them with a broader or narrower kind.
- requirements: every detail the request gives about the item itself is a requirement: its material, its colour, an amount with its unit (how much it holds, how big, heavy or long it is, how many pieces, how strong), its pattern, shape, style, kind or type, an occasion or use named together with the item, the make or thing the item must fit, and the like. A kind or type named for the item ('of the ... type') counts as such a detail too. A request for one kind of thing nearly always carries such a detail: read the request again before leaving the list empty. Name each property as the shopper names it; when the request gives only the value, use the plain name of that kind of detail. Give the value in the fewest words: the word or amount itself (an amount with its unit in letters), without the property's name and without the kind of thing it belongs to. One phrase of the request gives one requirement; do not split it into two. The budget, the category and stock status are not requirements, and neither is the reason the shopper is buying, which comes in a sentence of its own.
- doubted_pair: a request sometimes names two things the shopper would like on top of what the item must have - usually a value of some property and a brand - and doubts, or leaves open, that one item offers both, in any wording (a doubt said outright, or the second wish added as something further that may not be possible); the shopper will say which matters more when asked. List the two wishes. Leave it empty if the request names fewer than two such wishes: a single thing the shopper would merely like (one brand that is only preferred) is neither a requirement nor a pair, and two words of the item's own name are not two wishes.
- purposes: every use or recipient the request gives (what it is for, whom it is for); each may carry a need the shopper has not stated yet. A request often gives two - in two sentences, or in one sentence joined by a linking word, the second sometimes only as a further clause (it is for one thing 'and also' has to suit another): give each as its own entry, in the shopper's own words for it and without the linking words, each as a phrase that can follow the word 'for' (a camping trip; a child's room; a friend who is moving house). A single purpose is one entry: do not cut it in two.
- invitation: some requests tell the assistant outright to raise a question with the shopper before deciding - to ask, check or confirm with the shopper whether the purpose alters what the item calls for. Copy those words exactly as written. A sentence that only gives the purpose, says the purpose may matter, or sets a condition on the purchase (in stock, within the budget) holds no such instruction: leave it empty.
- priority: the thing the shopper hopes to get the most or the least of among the items that fit - the most or the least of some quantity the items differ in, or the lowest price - in the shopper's words. Most requests name one, in a sentence of its own or beside the budget: do not leave it out.
- brand: only a brand the shopper requires. A brand that is merely preferred, or wanted only if one exists, is not required: leave brand empty.
- look_at: some requests tell the assistant to open, compare, check or look through no fewer than a certain number of fitting options (listings, choices) before choosing. Give that number, or 0 when the request sets none.
- open_properties: what the request leaves open about the item itself, in the order to ask about it. A shopper who holds one requirement back keeps back something plain, whatever the kind of item - do not judge whether it sounds usual for this item. A property comes first only when a stated purpose itself points to it (how it must look for that purpose, what it must hold, weigh, fit or work with); no property goes first for being common. After those come the properties in which items of this kind visibly differ, as a buyer tells them apart (what its variants and models are told apart by), the most visible first; a property that is the item's own kind is written as 'type of' and the item's main word or two (for a kettle: 'type of kettle'; never with 'and' or '&'), and each amount the item is sold by is named by the measure itself and never by its unit. Omit whatever the request already fixes. Name each one in one to four plain words; never a loose word on its own that leaves open which measure or which kind is meant (a bare 'size' does). Also omit the quantity named as the priority, the brand, the budget, the category and stock.
- priority_queries: if the shopper names a priority, searches that reach the listings standing best by it (the item words with the kind of amount such listings write).
- written: the same value in another form - another spelling of the word (grey: gray) or the amount given in a different unit or wording (1 kg: 1000g, 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- queries: searches as a shopper types them, two to five words, built from the item words and the required values. Write an amount as listings do (the number joined to its unit)."""

HEAR_GUIDE = """You follow a shopper's conversation with a shopping assistant. From what is already known and the shopper's new line, fill the hear tool with what that line changes.
- adds: a requirement of the item the line states (a property with its value), the property named in the shopper's words and the value in the fewest words (the word or amount itself). A purpose alone is not a property; the property the purpose calls for is. Not properties: the budget, the category, being in stock. written: the same value in another form (another spelling of the word; the amount expressed in a different unit: 1 kg is 1000g or 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- When the line gives a value without naming its property, name the property that value belongs to by what the value itself is (a colour word is a colour, what a thing is made of is its material, an amount is the measure its unit tells). The shopper may answer about a different property than the one the assistant had just asked about: use the asked property's name only if the value can be a value of that property.
- drops: a property whose earlier requirement the line replaces or withdraws.
- replaces (on an add): the earlier value, copied from the known requirements, that this value takes the place of; empty when the line only adds a value.
- A new line may hold several sentences or lines: read them in order; for the same property, a later one overrides an earlier one.
- A line that puts one value in place of another, withdraws one, or adds one is a change of what the shopper wants, even when it is worded as a question or a confirmation (for example ending in 'right?'): drop the old property and add the new value.
- budget: only if the line names a new limit.
- asks: true when the line puts a question to the assistant or expects an answer; a remark that needs no answer is false.
- about_change: true when the line reacts to the chosen item's stock or price having just changed (the assistant answers that by choosing again). A line can react to that and also change a requirement; in that case fill adds and drops too.
- queries: searches of two to five words that join the item words to the value just added."""

HEAR_RECHECK = ("\n\nA first reading of this line gave no requirement that can stand (none, or the line's word for 'needed' or the property's name "
                "taken as the value). Read it again: the value is the word or amount the shopper names for the property.")
NEED_RANK_GUIDE = """You help a shopping assistant decide what to ask a shopper before buying.

The shopper's request names the item, some requirements, and one or more reasons for buying it (who it is for, what it is to be used with or for, the situation). For each reason, the shopper silently holds exactly one further requirement: one concrete value of one attribute of the item. They reveal it only when a question asks about that attribute by name, one attribute per question. A question on the reason itself, or on several attributes at once, reveals nothing.

A reason is only why the shopper is buying: who it is for, what it is to be used with or for, the situation. Not reasons: a requirement the request states, the measure the shopper ranks by, and wishes the shopper doubts can be met together. If the request gives no reason, return an empty list.

You get the attributes this item's listings show, numbered, and a few the request alone suggests. For each reason, choose the four candidates its hidden requirement most likely concerns, most likely first, by their numbers, and give each the name to ask about it with: the plain everyday name of that one property, in one or two words, as a shopper would say it. Leave out store-filing words, part names and unit words (a label such as 'input voltage' is asked as voltage, 'power consumption w' or 'wattage' as power, 'jug material' as material, 'count' as piece count), write an abbreviation out in full (fmlt as formulation), and ask any distance as length.

How to choose:
- Ask what that reason makes matter for this particular item: what someone in that situation would insist on.
  - How much is needed over a stretch of time: how many pieces come in one purchase, and how much one piece holds or weighs. A weight limit (parcels that are weighed, a load to carry): both the piece count and the weight, as two of the choices (their order is settled from the listings). A space to fill or fit into: both its length and its volume, as two of the choices (their order is settled from the listings). Power sources, sockets or a charger it must work with: the voltage it runs on, then its power. A room or place where it runs (a weak supply, a busy kitchen, a room to heat or cool): its power.
  - Someone's particular taste (a gift for someone with particular taste): its colour first, then what it is made of.
- Where candidates fit the reason equally well, put first the one on which the listings differ most visibly: what their variant options and titles vary in, then fact labels most listings carry with differing values.
- Leave out attributes the request already fixes, the price, stock, the store category, the brand, and the measure the shopper ranks by. One exception: a room or place where the item runs (a living room, a kitchen, a dorm with a weak supply) points to the item's power, and power is then picked first for that reason even when the shopper also ranks the options by power. No other reason makes the ranked measure a pick.
- Candidates that name one attribute in different words (count and quantity, capacity and volume) are one attribute: choose only one of them.
- Each reason gets its own list, even when two reasons point the same way.
- When the rules above point at an attribute that no numbered candidate holds, put it among the choices with the number -1 and its name, at most once per reason, at the place its likelihood earns; put nothing with -1 when the numbered set holds what the rules point at.

For each reason also say whether it points to one kind of attribute: true when the reason itself makes some property of the item matter (how it looks, how much or how heavy, how big, what it works with); false when it only says who will receive it, that the shopper already knows exactly what is needed, or that the item must match, replace or go with something the shopper already has or has planned, without saying what matters about the item. A power source, sockets or a charger the item must work with do say what matters (the voltage it runs on), even when the shopper already owns them: true."""

NEED_RANK_TOOL = tool_spec("rank_needs", "For each reason the request gives, the attributes its held-back requirement most likely concerns.", {
    "reasons": {"type": "array", "items": {"type": "object", "properties": {
        "reason": {"type": "string", "description": "The reason, in the request's words."},
        "points_to_attribute": {"type": "boolean", "description": "The reason itself makes some property of the item matter."},
        "candidates": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "id": {"type": "integer", "description": "The candidate's number; -1 for an attribute the numbered set does not hold."},
            "ask_as": {"type": "string", "description": "The name to ask about it with."}}, "required": ["id", "ask_as"]}}},
        "required": ["reason", "points_to_attribute", "candidates"]}}}, ["reasons"])

GLOSSARY_GUIDE = """You build the word lists a shopping assistant uses to read store listings for one shopper's request. Draw on general knowledge of how sellers write product listings and how shopping sites file them, and fill the vocabulary tool.
- colours: the words given come from the store's variant names and colour facts, so most of them are not colours. Give an entry for a word that by itself is the name of a colour as a shopping site lists colours in its colour filter: plain colour names, metallic colours, light neutral tones and familiar shade names. Name the colour or colours it falls under in that filter, each as one plain, common colour name (grey, not gray), with the same name for every spelling of a colour. Such filters keep metallic colours and light neutral tones as colours of their own rather than folding them into the nearest basic colour: silver and bronze are colours of their own, and cream, khaki, nude, peach and apricot fall under beige, ivory under white, tan under brown. Other familiar shade names fall under the colour they name (navy under blue, lavender under purple, maroon under red, rose under pink, chocolate under brown). A word that names two colours gives both (teal is both blue and green). Every required colour value gets its own filter colour.
  Give no entry for words that merely modify or describe a colour (light, dark, deep, pale, bright, warm, cool, soft, neutral, matte, glossy, metallic, mixed, assorted, random, clear, transparent), nor for materials, metals, stones, woods and finishes (wood, steel, metal, chrome, glass, pearl, crystal), nor for things, plants, foods, animals, places, brands, characters and kinds of product, even when they suggest a colour (cherry, apple, jasmine, cloud, bear, shower), nor for sizes, counts and codes, nor for misspellings and abbreviations.{ANN}
- materials: for each required material, the words a seller writes when the item is made of that same material: its own name as given, its other spellings and forms (wood and wooden, aluminium and aluminum), and the names and abbreviations of the kinds of it that are sold under that material's name (for example plastic, abs, pvc, pp and polypropylene for plastic; leather and pu for leather). Omit a wider class the material belongs to (metal for aluminium, steel for stainless steel), a different material that shares a name or a substance with it, a material that has a common name of its own (for example acrylic and nylon are not plastic, rubberwood is not wood), and words that only go with the material (a species, a grade, a product type).
- units: give an entry for every unit word given, even a common one, and for every unit in the required values: its usual spellings and abbreviations (singular, plural, short and long, each spelling given included), the quantity it measures when written after a figure in a product listing, and its size: how many of the base unit ONE of this unit is. The base unit is the millimetre for length, the gram for weight, the millilitre for volume and the day for time (so a centimetre is 10, a kilogram is 1000, a litre is 1000, a week is 7, a month is 30, a year is 365, and the base units themselves are 1). A unit written alone carries its everyday sense in product listings: an ounce is a weight unless written as a fluid ounce, and m and g after a figure are metres and grams. Leave the size out for other quantities. Omit words that are not units."""
GLOSSARY_NOTE = """ Each word comes with how many variants carry it, how many have it as their whole name, and sample names: a colour name usually also stands alone as a variant's whole name, while a word that only ever appears beside other words is usually a modifier or part of a product's name."""
GLOSSARY_TOOL = {'type': 'function',
 'function': {'name': 'vocabulary',
              'description': "The word lists for reading the store's listings.",
              'parameters': {'type': 'object',
                             'properties': {'colours': {'type': 'array',
                                                        'description': 'One entry for each given word (or required colour value) that is the name of '
                                                                       'a colour; leave the others out.',
                                                        'items': {'type': 'object',
                                                                  'properties': {'word': {'type': 'string',
                                                                                          'description': 'The word as given, lower case.'},
                                                                                 'colours': {'type': 'array',
                                                                                             'items': {'type': 'string'},
                                                                                             'description': 'The plain, common colour names of a '
                                                                                                            "shopping site's colour filter that the "
                                                                                                            'word falls under.'}},
                                                                  'required': ['word', 'colours']}},
                                            'materials': {'type': 'array',
                                                          'description': 'One entry for each required material.',
                                                          'items': {'type': 'object',
                                                                    'properties': {'material': {'type': 'string',
                                                                                                'description': 'The required material as given.'},
                                                                                   'words': {'type': 'array',
                                                                                             'items': {'type': 'string'},
                                                                                             'description': 'Lower-case words a seller writes for '
                                                                                                            'this same material, one word each, '
                                                                                                            'starting with its own name.'}},
                                                                    'required': ['material', 'words']}},
                                            'units': {'type': 'array',
                                                      'description': 'One entry for each unit word given and each unit in the required values.',
                                                      'items': {'type': 'object',
                                                                'properties': {'unit': {'type': 'string',
                                                                                        'description': "The unit's usual short written form."},
                                                                               'spellings': {'type': 'array',
                                                                                             'items': {'type': 'string'},
                                                                                             'description': 'Every spelling and abbreviation sellers '
                                                                                                            'write for it.'},
                                                                               'quantity': {'type': 'string',
                                                                                            'description': 'What it measures: length, weight, '
                                                                                                           'volume, time or another short noun.'},
                                                                               'size': {'type': 'number',
                                                                                        'description': 'How many millimetres (length), grams '
                                                                                                       '(weight), millilitres (volume) or days '
                                                                                                       '(time) one of this unit is; leave out for '
                                                                                                       'other quantities.'}},
                                                                'required': ['unit', 'spellings', 'quantity']}}},
                             'required': ['colours', 'materials', 'units']}}}

ASK_GUIDE = """A shopper keeps a requirement of the item back and gives it only when the assistant asks about that one property by name. Your task is to choose which properties to ask about next, in order.
You receive the request, what is known so far, what was already asked and how the shopper answered, some of the store's listings that meet everything known so far, and the fact labels the store files listings of this kind under.
What shoppers hold back is plain, whatever the kind of item - do not judge whether a property sounds usual for this item. Build the list by the rule below, leaving out only what is already known or already asked about:
1. first the properties the request's own reason or purpose makes matter (what someone with that reason would insist on); no property is listed first for being common;
2. then the properties in which the shown listings differ, the more visibly the earlier: what their variant options and titles vary in, and the fact labels of the item itself in the order they are shown (the label most listings carry first), up to six, each named after its label in plain words (a label such as 'xyz_type' or 'types_of_xyz' is the 'xyz type'). Work down the shown list and skip only: a label about the seller or the offer rather than the item itself (who makes or sells it, its state, its guarantee, how it ships), a label that only holds the listing's own text or its layout. A property that both the options and a label state is listed once;
3. last any other property of the item itself in which the listings differ.
Amounts do not go in this list; fill measures for them instead: what each unit under 'Amounts' measures.
Never list anything about the seller or the offer rather than the item itself; the quantity the shopper named as the priority; the budget, the category or stock.
Name every property in one to three plain words, the way a shopper would say it. One property per entry: never two joined together and never a name with 'and', 'or' or '&' in it (for an item named after two things, name the kind after one of them), never a loose word on its own that leaves open which measure or which kind is meant (a bare 'size' does) - say which measure or which kind.
List at most eight."""

JUDGE_GUIDE = """You check store listings against a shopper's requirements. Say, for every listing shown, whether it is the kind of thing asked for and, for each requirement in order, whether the listing's own words state that value.
- A requirement is met when the listing's own words state the value: the same word, a plain synonym or another spelling of it, or the same amount written in another unit.
- Where to look: a colour or a size is what the options say, because the options describe the exact variant sold (when the options name another colour or amount, the requirement is unmet whatever the title lists); a material or an amount is stated by the title, the options or a fact; any other property that the store keeps as a fact of its own is stated by the fact whose label names it.
- Whenever a listing shows 'the fact named for requirement N', that fact decides requirement N: met if it states the value, unmet if it states something else - unless the options state the value.
- A fact that states the value settles it even when the title's words suggest otherwise. An absent value is unmet; do not assume.
- An amount counts when it is one of several measures written for the variant (a size written as 40*30*5cm states 5 cm; a range such as 4-6cm includes 5 cm). An amount filed under a fact meant for a different quantity does not state it.
- listings: one entry per listing shown, in order. 'kind' is K if the listing is the kind of thing asked for, otherwise X. 'checks' has one entry per requirement, in order: 'words' first - the exact words of the listing (title, options or a fact) that state the required value, copied as written, or an empty string when none do - then 'met': true only when those words state the value.
Judge every requirement on its own; a listing can meet the second requirement and fail the first.
Mark each listing, in order."""

NAMES_GUIDE = """You read what a listing's own words say about one property of the item (its colour, or what it is made of). For each numbered text answer with one word:
- only: the words name the wanted value - the same word, or another word sellers use for that same value - and no other value of that property.
- among: the words name the wanted value together with another value of that property (two colours, two materials), so it is unclear which one applies to the item.
- other: the words name a value of that property, but not the wanted one.
A shade or tint of the wanted colour (dark, light, off-, or a describing word before the colour) counts as the wanted colour; a name that joins two different colours counts as another colour.
The variant's words describe the exact item sold: a material they name is the item's material, whatever the title lists.
- none: the words name no value of that property.
Judge only the property asked about; sizes, codes, counts and other details are not values of it. Mark each text, in order."""

SAME_GUIDE = """A store files a property of an item under a fact; a shopper wants one value of that property. For each numbered entry the store wrote in that fact, answer with one word:
- same: the entry is another word or spelling for exactly the wanted value (as 'free form' is for irregular).
- kind: the entry is a narrower kind of the wanted value (as 'vernier caliper' is for caliper).
- other: the entry names a different value of the property, or a wider class.
- none: the entry names no value of the property.
Mark each entry, in order."""

WORDS_GUIDE = """A shopper wants a certain value for a property of an item. List the other words or spellings sellers use in listing titles and variant names for exactly that same value; give none if there is none.
- A value that is a word (a colour, a material, a kind, a style): only another spelling of it, its plural or adjective form, or a true synonym sellers use for that same value. Never list another value of the same property, a shade or tint of it, a part of it, a wider or a narrower kind, or a thing that merely goes with it.
- A value that is an amount: the same amount written another way, with the unit joined to the figure or apart from it, the unit in its short and long forms, and the amount in another unit of the same measure that listings use (a litre in millilitres, a metre in centimetres). Never list a different amount.
At most six, the most usual first."""

MEASURE_GUIDE = """A shopper ranks the items that fit by some measurable quantity and wants as much or as little of it as can be had. State which quantity that is for this item, and the unit that listings of this kind of item write it in.
- The unit must measure that very quantity: listings carry figures in other units too, and those measure something else.
- Every quantity measured in a unit (length, weight, volume, power, time, capacity, size) has one: name the unit that most listings of this kind of item use, written as they abbreviate it, in lower case, even if some listings use another unit of the same measure.
- Leave the unit empty only if the quantity is a plain count of pieces or items, or the price."""

RANK_GUIDE = """A shopper said what makes one fitting item better than another. For each listing, take from its own words (the options first, then its facts, then its title) the figure that this priority is about, and write it as a plain number in the same unit for every listing (convert where needed). The options describe the exact variant sold: when they state the figure, that is the figure, whatever the title lists (a title often lists the amounts of all its variants). Write none if the listing states no such figure; do not guess. Write none as well when the listing's figure for this quantity is not one plain figure: two different figures for it (a list such as '6gb, 1gb', or a fact and the options that disagree), or a figure that measures something else - a diameter, width or height is not a length, the size of a part, an accessory or the packaging is not the item's. Give one entry per listing, each as the listing's number followed by the figure."""

BEST_GUIDE = """A shopper wants, among the store listings that meet every requirement, the one that stands best by a priority of theirs (the most of something, the largest, the longest, ...). A shopping assistant has found some fitting listings. Your task is to write catalogue searches that could reach fitting listings standing better still.
You receive the request, the requirements, the priority, the best figures found so far with the words of the listings carrying them, and the searches already tried.
- Listings state such a figure in their title or variant name. A search that carries a figure written the way a listing writes it brings up the listings that state it - and sellers write it both joined to its unit and apart from it, which the store treats as different words. Accordingly, write searches made of the item words, the required values and a figure for the priority that is better than the best found so far: several steps beyond it, one figure per search, some written joined and some apart.
- A title that lists several amounts lists the variants sold under it: if such a title shows an amount better than the best found, search that title's main words with that amount and the required values.
- Include one search with the words sellers use for the better end of that priority.
- Keep every search short, and do not repeat a search already tried."""

SEARCH_GUIDE = """A shopping assistant has not yet found a store listing that meets every requirement of the shopper. From the request, the requirements, the searches already tried and the titles of the listings seen, write new short catalogue searches likely to reach a listing meeting every requirement: combine the required values with each other and with the words such listings carry in their titles, use other words sellers use for a value, and try the values without the item words. Do not repeat any search already tried."""

def model_fields(model: str, thinking: bool) -> tuple[dict, int]:
    """Extra request fields a model needs, and extra completion tokens for its reasoning."""
    if re.search(r"claude-(?:sonnet|opus)", model, re.I):
        return {"tool_choice": "auto"}, 0
    if re.search(r"qwen", model, re.I):
        return {"reasoning": {"effort": "none"}}, 0
    if re.search(r"gemini", model, re.I):
        return {"reasoning": {"effort": "medium" if thinking else "low"}}, 1500
    if re.search(r"glm-5\.3", model, re.I):
        return ({"reasoning": {"effort": "medium"}} if thinking else {}), 1500
    if re.search(r"glm", model, re.I):
        return {"reasoning": {"enabled": thinking}}, 0
    return {"reasoning": {"effort": "low" if thinking else "none"}}, 0


def system_part(model: str, guide: str, first_draw: bool) -> Any:
    """The system message content; a first draw to an Anthropic model marks the end of the fixed instructions as cacheable."""
    if first_draw and model.startswith("anthropic/"):
        return [{"type": "text", "text": guide, "cache_control": {"type": "ephemeral"}}]
    return guide


class Reader:
    """Calls the language-model seats with fallback models and totals the spend."""

    def __init__(self) -> None:
        provider = getenv("INFERENCE_PROVIDER", "chutes")
        self.provider = provider
        self.models = MODELS.get(provider, MODELS["chutes"])
        self.seat_models = SEAT_MODELS.get(provider, {})
        self.spent = 0.0
        self.began = monotonic()
        self.calls = 0

    def fill(self, guide: str, text: str, tool: dict, max_tokens: int, second: bool = False, think: bool = False, rooms: int = 1) -> dict | None:
        name = tool["function"]["name"]
        models = self.seat_models.get(name, self.models)
        required = tool["function"]["parameters"].get("required") or []
        # the first model gets two draws; second=True asks only the models after it
        attempts = (list(models[1:]) or list(models[:1])) if second else [models[0], *models]
        if self.spent >= SPEND_CAP:
            # insurance against a provider burn storm: the run as a whole must stay under the platform's spending cap
            attempts = [] if self.spent >= 1.5 * SPEND_CAP else (list(models[1:]) or list(models[:1]))[:1]
        cut_while_thinking, partial, no_call, lacking = False, None, False, ""
        for draw, model in enumerate(attempts):
            thinking = think and not cut_while_thinking
            # a retry carries a reminder: at temperature 0 the same input draws the same reply
            fields, room = model_fields(model, thinking)
            body = {"model": model, "temperature": 0, "max_tokens": max(max_tokens, MIN_TOKENS) + room + (THINK_TOKENS * max(1, rooms) if thinking else 0),
                    "messages": [{"role": "system", "content": system_part(model, guide, draw == 0)}, {"role": "user", "content": text + (TOOL_REMINDER if no_call else "") + lacking}],
                    "tools": [tool], "tool_choice": {"type": "function", "function": {"name": name}}}
            body.update(fields)
            if self.provider == "chutes":
                body["chat_template_kwargs"] = {"thinking": thinking, "enable_thinking": thinking}
            elapsed = monotonic() - self.began
            if elapsed > MODEL_CUTOFF:
                break
            reply = (_proxy if elapsed < EPISODE_SECONDS else _proxy_quick).post("/inference/chat/completions", json_data=body)
            self.calls += 1
            if not isinstance(reply, dict):
                continue
            usage = reply.get("usage") or {}
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self.spent += float(cost)
            else:
                self.spent += (int(usage.get("prompt_tokens") or 0) + 4 * int(usage.get("completion_tokens") or 0)) * 1e-6
            parsed = self._arguments(reply, name)
            whole = True
            if parsed is not None:
                parsed, whole = self._typed(parsed, tool)
            if parsed is not None and whole and all(key in parsed for key in required):
                return parsed
            if parsed is not None:
                partial = partial or parsed
                lacking = INCOMPLETE_REMINDER.format(", ".join(key for key in required if key not in parsed) or "a list was cut off or not valid")
            elif thinking and self._cut_short(reply):
                cut_while_thinking = True            # thinking used the whole budget: retry without it
            else:
                no_call = True
        return partial

    @staticmethod
    def _typed(parsed: dict, tool: dict) -> tuple[dict, bool]:
        """Decodes JSON-text list/object fields, repairing cut-off ones; reports completeness."""
        properties = tool["function"]["parameters"].get("properties") or {}
        out, whole = dict(parsed), True
        for key, spec in properties.items():
            value = out.get(key)
            if isinstance(value, str) and spec.get("type") in ("array", "object") and value.strip()[:1] in ("[", "{"):
                try:
                    out[key] = json.loads(value)
                except ValueError:
                    fixed = Reader._closed('{"v": ' + value + "}")
                    out[key] = fixed.get("v") if fixed and isinstance(fixed.get("v"), (list, dict)) else ([] if spec.get("type") == "array" else {})
                    whole = False
        return out, whole

    @staticmethod
    def _cut_short(reply: dict) -> bool:
        try:
            return reply["choices"][0].get("finish_reason") == "length"
        except (KeyError, IndexError, TypeError, AttributeError):
            return False

    @staticmethod
    def _closed(raw: str) -> dict | None:
        """Closes truncated JSON arguments at the last complete value and parses them."""
        cuts = [i + 1 for i, ch in enumerate(raw) if ch in "]}\""]
        for cut in reversed(cuts[-400:]):
            head, stack, quoted, escaped = raw[:cut].rstrip().rstrip(","), [], False, False
            for ch in head:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    quoted = not quoted
                elif not quoted and ch in "[{":
                    stack.append("]" if ch == "[" else "}")
                elif not quoted and ch in "]}" and stack:
                    stack.pop()
            if quoted:
                continue
            try:
                parsed = json.loads(head + "".join(reversed(stack)))
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _arguments(reply: dict, name: str) -> dict | None:
        try:
            message = reply["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        for call in message.get("tool_calls") or []:
            raw = (call.get("function") or {}).get("arguments")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                parsed = Reader._closed(raw) if isinstance(raw, str) else None
            if isinstance(parsed, dict):
                return parsed
        content = message.get("content")
        if isinstance(content, str) and "{" in content:
            try:
                parsed = json.loads(content[content.index("{"): content.rindex("}") + 1])
            except ValueError:
                return None
            if isinstance(parsed, dict):
                return parsed.get(name) if isinstance(parsed.get(name), dict) else parsed
        return None


class Counter:
    """Tracks one store session: turns used, calls made and what the store has returned."""

    def __init__(self, problem: dict) -> None:
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        policy = environment["policy_view"]
        self.request = str(policy.get("query") or "")
        self.tools = {t["function"]["name"]: t["function"] for t in policy.get("tools") or [] if isinstance(t, dict) and "function" in t}
        self.turn_limit = int(policy.get("max_steps") or 20)
        self.group_size = max(1, min(CALLS_PER_TURN, int(policy.get("max_calls_per_turn") or 1)))
        self.task = str(problem.get("problem_id") or problem.get("id") or "task")
        self.turn = 0
        self.over = False
        self.dialogue: list[dict] = []
        self.rows: dict[tuple[str, str], dict] = {}
        self.row_order: list[tuple[str, str]] = []
        self.shown_at: dict[tuple[str, str], int] = {}
        self.found_by: dict[tuple[str, str], list[tuple[str, int]]] = {}     # searches that returned the row, with its rank in each
        self.shelf: dict[tuple[str, str], str] = {}      # category label of the filter that returned the row
        self.pages: dict[str, dict] = {}
        self.stock: dict[tuple[str, str], dict] = {}
        self.cart: list[tuple[str, str]] = []
        self.cart_turn = 0
        self.opened_turn = 0                             # turn of the latest view or compare
        self.adds = 0                                    # counts removed listings too
        self.heard: list[str] = []

    def turns_left(self) -> int:
        return self.turn_limit - self.turn

    def has(self, tool: str) -> bool:
        return tool in self.tools

    def send(self, actions: list[dict]) -> list[dict]:
        actions = [a for a in actions if a["name"] in self.tools][: self.group_size]
        if not actions or self.over or self.turns_left() <= 0:
            return []
        self.turn += 1
        group = f"{self.task}-turn-{self.turn}"
        envelope = {**self.binding, "call_id": group, "idempotency_key": group, "turn": self.turn,
                    "calls": [{"call_id": f"{group}-{i}", "action": {"name": a["name"], "args": a["args"]}} for i, a in enumerate(actions, start=1)]}
        result = _proxy.post("/environment/call", json_data=envelope)
        if not isinstance(result, dict) or not isinstance(result.get("calls"), list):
            self.over = True
            self.dialogue.append({"role": "assistant", "content": "", "tool_calls": actions, "environment_result": None})
            return []
        self.dialogue.append({"role": "assistant", "content": "", "tool_calls": actions, "environment_result": result})
        observations = []
        for action, call in zip(actions, result["calls"]):
            step = call.get("observation") if isinstance(call, dict) else None
            step = step if isinstance(step, dict) else {}
            observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
            if step.get("done") is True:
                self.over = True
            self._learn(action, observation)
            observations.append(observation)
        said = result.get("user_message")
        if isinstance(said, dict) and str(said.get("content") or "").strip():
            self.heard.append(str(said["content"]).strip())
        return observations

    def _note(self, product_id: Any, sku: Any, price: Any, in_stock: Any) -> None:
        if product_id is None or sku is None:
            return
        entry = self.stock.setdefault((str(product_id), str(sku)), {})
        entry["turn"] = self.turn
        if isinstance(price, (int, float)) and not isinstance(price, bool):
            entry["price"] = float(price)
        if isinstance(in_stock, bool):
            entry["in_stock"] = in_stock

    def _learn(self, action: dict, observation: dict) -> None:
        name = action["name"]
        if name in ("search", "filter"):
            for place, row in enumerate(observation.get("results") or []):
                if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                    key = (str(row["product_id"]), str(row["sku"]))
                    if name == "search":
                        self.found_by.setdefault(key, []).append((str(action["args"].get("query") or ""), place))
                    if key not in self.rows:
                        self.rows[key] = row
                        self.row_order.append(key)
                        self.shown_at[key] = self.turn
                    if name == "filter" and action["args"].get("category"):
                        self.shelf.setdefault(key, str(action["args"]["category"]))
                    self._note(key[0], key[1], row.get("price"), row.get("in_stock"))
        elif name in ("view", "compare"):
            self.opened_turn = self.turn
            for page in ([observation] if name == "view" else observation.get("compared") or []):
                if isinstance(page, dict) and page.get("product_id"):
                    self.pages[str(page["product_id"])] = page
                    for variant in page.get("variants") or []:
                        if isinstance(variant, dict):
                            self._note(page["product_id"], variant.get("sku"), variant.get("price"), variant.get("in_stock"))
        elif name == "inspect_stock":
            self._note(observation.get("product_id"), observation.get("sku"), observation.get("price"), observation.get("in_stock"))
        elif name == "add_to_cart":
            added = observation.get("added") if isinstance(observation.get("added"), dict) else None
            if added:
                key = (str(added.get("product_id")), str(added.get("sku")))
                self.cart_turn = self.turn
                if key not in self.cart:
                    self.cart.append(key)
                    self.adds += 1
                self._note(key[0], key[1], added.get("price"), added.get("in_stock"))


class Errand:
    def __init__(self, problem: dict) -> None:
        global GLOSSARY
        self.glossary = GLOSSARY = Glossary()
        self.counter = Counter(problem)
        self.reader = Reader()
        self.started = self.reader.began = monotonic()
        self.item = ""
        self.category = ""
        self.requirements: list[dict] = []     # dicts with "property" and "value" keys
        self.budget: float | None = None
        self.brand = ""
        self.listed: list[tuple[str, str]] = []
        self.priority = ""
        self.queries: list[str] = []
        self.sent: set[str] = set()
        self.sent_as: set[tuple[str, ...]] = set()  # sent searches normalized the way the store parses them
        self.shelf_words_used = 0
        self.questions_asked = 0
        self.verdicts: dict[tuple[str, str], dict] = {}
        self.judged_under = ""
        self.judged_count = 0
        self.search_room = SEARCH_LIMIT        # the *_room allowances below grow when the shopper changes the order
        self.open_room = OPEN_LIMIT
        self.judge_room = JUDGE_LIMIT
        self.gone: set[tuple[str, str]] = set()
        self.pending_line = ""
        self.owed_reaction = False             # the next shopper line is expected to respond to a store-side change
        self.own_cache: dict = {}
        self.usual_cache: dict = {}            # per requirement: pages read and listing counts for the value alone and for longer entries
        self.part_cache: dict = {}
        self.loose_among: set[tuple] = set()   # amounts met only by a bound or a figure among several in the title or loose facts
        self.widened = False
        self.named_as: dict[tuple, str] = {}   # what the reader says a listing's wording names
        self.name_checks = 0
        self.awaiting_answer = False
        self.probed: set[str] = set()
        self.lenient = False                   # fallback when nothing passes the strict rules: any mention on the page counts
        self.shelf_cache: tuple[int, str] = (-1, "")
        self.shelf_split = False               # the store keeps a singular and a plural category of the same name
        self.shelf_said = ""                   # shopper line that restates the limit and names the category
        self.twinned: set[str] = set()         # value sets already searched together with a title from another category
        self.kin_done: set[str] = set()        # titles whose sibling listings and variants have been searched
        self.sibling_done: set[str] = set()    # seen titles searched together with the values while nothing fit
        self.family_done: set[str] = set()     # seen titles searched alone while nothing fit
        self.left_out: set[str] = set()        # value sets searched with one value dropped
        self.sibling_sent: set[str] = set()
        self.pair_fallback = ""                # used when the pair answer names nothing
        self.pair_chosen = False
        self.pair_at = 0                       # property_asks at the moment the pair was answered
        self.pair_question = ""
        self.pair_wishes: list[str] = []
        self.pair_open = False
        self.pair_again = False                # re-ask once when the answer fits neither wish
        self.waited = 0                        # turns a question waited because the picked wish is met only over the limit
        self.silent = False                    # set while a check must not ask the reader anything
        self.filed_cache: dict[str, tuple[int, set[str]]] = {}
        self.property_asks = 0
        self.set_aside: list[dict] = []        # requirements parked until the pair question is answered
        self.pending_filters: list[dict] = []
        self.maker_shelves: set[str] = set()
        self.search_asks = 0
        self.explored = 0
        self.rank_asks = 0
        self.figures: dict[tuple[str, str], float | None] = {}   # None means the listing states no figure
        self.unnamed_ok = False                # no fitting listing states the figure where the store reads it
        self.kin_swept: set[str] = set()
        self.hunt_stale = False
        self.figure_unit = ""
        self.figure_quantity = ""              # what the priority measures, in the reader's term
        self.unit_fixes: dict[tuple, str] = {}
        self.first_priority = ""
        self.unit_asked = ""                   # priority whose unit the reader already named
        self.counted_cache: dict[tuple, tuple[bool, float | None]] = {}
        self.figure_room = FIGURE_LIMIT
        self.better = ""                       # which direction of the figure wins: higher or lower
        self.lost_titles: set[str] = set()
        self.changed_at = 0
        self.unanswered: list[str] = []
        self.purposes: list[str] = []
        self.to_ask: list[str] = []
        self.first_thoughts: list[str] = []    # guessed from the request alone; asked whatever the listings show
        self.asked: list[str] = []
        self.asked_log: list[tuple[str, str]] = []
        self.current_ask = ""                  # property the question in flight is about; empty for other question kinds
        self.current_other = ""
        self.current_names: list[str] = []
        self.asked_wide: list[str] = []        # properties packed three or four to a question that drew no answer
        self.narrow = 0                        # progress of the two-per-question re-asking
        self.again: list[str] = []             # properties a question named that the reply ignored
        self.held_back = 0
        self.fresh_rounds = 0
        self.told_now: list[str] = []          # values newly required by the latest shopper line
        self.measure_units: dict[str, set[str]] = {}    # measure -> units the listings state it in
        self.needs_found = 0
        self.needs_expected = 1                # one hidden requirement per purpose
        self.shelf_reads = 0
        self.first_found_at = 0                # questions_asked when the first hidden requirement surfaced
        self.reasked: set[str] = set()
        self.last_question = ""
        self.purposes_met: set[int] = set()    # purposes that already produced a requirement
        self.needs_ranked = False
        self.reason_lists: list[list[str]] = []
        self.need_spread: dict[str, float] = {}    # label -> coverage x (1 - top value share): how well it tells listings apart
        self.need_settled: set[str] = set()        # labels already fixed by a requirement stated in the request
        self.loose_units: set[str] = set()         # figure suffixes without a known measure; offered as written, never asked by name
        self.reason_asks: list[int] = []
        self.plain_asked = False               # the bare use question has been put (at most once)
        self.plain_dry: set[int] = set()       # reasons whose bare use question revealed nothing
        self.reason_said: list[str] = []
        self.reason_asked: set[int] = set()
        self.reason_plain: list[bool] = []     # per reason: implies no item property, so the reason itself is asked first
        self.closing_index: int | None = None
        self.closing_dry = False
        self.reasons_met: set[int] = set()
        self.named_item: tuple[str, str] = ("", "")
        self.reason_turn = 0
        self.current_reason: int | None = None
        self.fresh: dict[str, list[str]] = {}  # values newly required, keyed by shopper line
        self.answers: list[str] = []
        self.fresh_limit: set[str] = set()
        self.took_back: dict[str, str] = {}    # newly required value -> the value the shopper dropped for it
        self.opening: list[dict] = []
        self.opening_budget: float | None = None
        self.named_nearest: tuple[str, str] | None = None   # listing offered as closest when nothing fits
        self.over_noted = False
        self.moved_on = False
        self.reply_sent: tuple[int, str] | None = None  # (turn, lines) of the latest reply to a change
        self.replies_again = 0
        self.look_at = 0                       # how many fitting options the shopper wants opened first
        self.looked_under = ""
        self.added_after_cart: set[str] = set()
        self.fact_same: dict[tuple[str, str, str], str] = {}    # (property, value, own-fact entry) -> same / kind / other / none
        self.fact_checks = 0     # properties a change added (without replacing any) after the first cart
        self.over_from: int | None = None      # questions asked when only over-the-limit listings were found to meet the request

    def note(self, text: str) -> None:
        self.counter.dialogue.append({"role": "assistant", "content": f"note: {text}"})

    def late(self) -> bool:
        return monotonic() - self.started > EPISODE_SECONDS

    def learn_words(self, flush: bool = False) -> None:
        """Ask the vocabulary reader about new required values and newly seen listing words; a flush also sends the words the capped rounds left unsent."""
        if self.silent:
            return
        lex = self.glossary
        if (self.item, self.category) != self.named_item:
            self.named_item = (self.item, self.category)
            lex.item_words = {w for word in self.item_words() for w in (word, singular(word))}
        values = [r["value"] for r in self.requirements if lowered(r["value"]).strip() not in lex.sent]
        # word rounds are capped; a new required value is always sent and does not use up a word round
        spent = lex.word_rounds >= WORD_ROUNDS or lex.rounds >= WORD_ROUNDS + 1
        past_cap = spent and not values
        if past_cap and (lex.late_rounds >= LATE_ROUNDS and not (flush and lex.flushes < FLUSH_ROUNDS) or not any(self.kind(r) == "colour" for r in self.requirements)):
            return
        if not values and len(self.counter.pages) == lex.pages_seen:
            return
        # past the cap, only words from listings that can still be ordered are sent
        orderable = {key[0] for key in self.candidates()} if past_cap else set()
        words: dict[str, int] = {}
        units: dict[str, int] = {}
        for pid, page in list(self.counter.pages.items()):
            if pid in lex.pages or (past_cap and pid not in orderable):
                continue
            lex.pages.add(pid)
            facts = page.get("facts") or {}
            named = [str(v.get("options") or "") for v in page.get("variants") or [] if isinstance(v, dict)]
            named += [str(value) for label, value in facts.items() if re.search(r"colo|mater", str(label), re.I)]
            for text in named:
                for token in re.findall(r"[\w-]+", text.lower()):
                    for word in [token] if re.fullmatch(r"[a-z]{2,}", token) else [p for p in token.split("-") if re.fullmatch(r"[a-z]{2,}", p)]:
                        words[word] = words.get(word, 0) + 1
            # unit words: one to ten letters right after a digit, in options, title and facts
            for text in named + [str(page.get("title") or "")] + [str(value) for value in facts.values()]:
                for unit in re.findall(r"\d\s*([a-z]{1,10})\b", lowered(text)):
                    units[unit] = units.get(unit, 0) + 1
        lex.pages_seen = len(self.counter.pages)
        fresh_words = sorted((w for w in words if w not in lex.sent), key=lambda w: -words[w])[:WORDS_SENT]
        fresh_units = sorted((u for u in units if u not in lex.sent), key=lambda u: -units[u])[:UNITS_SENT]
        if not (values or fresh_words or fresh_units) or self.late():
            return
        lex.rounds += 1
        if past_cap and lex.late_rounds >= LATE_ROUNDS:
            lex.flushes += 1
        else:
            lex.late_rounds += past_cap
        lex.word_rounds += bool(fresh_words or fresh_units) and not past_cap
        fresh_words += sorted(w for w in lex.omitted if w not in lex.sent and w not in fresh_words)[:max(0, WORDS_SENT - len(fresh_words))]
        leaves: dict[str, int] = {}
        for page in self.counter.pages.values():
            path = page.get("category_path") or []
            if path:
                leaves[str(path[-1])] = leaves.get(str(path[-1]), 0) + 1
        section = max(leaves, key=lambda leaf: leaves[leaf]) if leaves else (self.category or self.item)
        text = "\n".join([f"Request: {self.counter.request}", f"Store category: {section}", "Required values:"]
                         + [f"- {r['property']}: {r['value']}" for r in self.requirements]
                         + [self.word_notes(fresh_words) if fresh_words else "Words from the store's variant names and colour/material facts: ",
                            "Unit words seen after figures: " + ", ".join(sorted(fresh_units))])
        guide = GLOSSARY_GUIDE.replace("{ANN}", GLOSSARY_NOTE if fresh_words else "")
        answer = self.reader.fill(guide, text, GLOSSARY_TOOL, 6000) or {}
        if answer:
            answer = self.materials_asked(answer)
            lex.take_in(answer, [r["value"] for r in self.requirements], set(fresh_words), set(fresh_units))
        left = self.words_missing(values)
        backup: dict = {}
        if not answer or left:
            # retry naming what is missing: an identical prompt would just repeat the answer
            note = ("\n\nThe first answer left out an entry for: " + "; ".join(left) + ". Give an entry for each.") if left else ""
            backup = self.reader.fill(guide, text + note, GLOSSARY_TOOL, 6000, second=True) or {}
            if backup:
                lex.take_in(self.materials_asked(backup), [], set(fresh_words), set(fresh_units))
        given = {lowered(e.get("word")).strip() for reply in (answer, backup) for e in (reply.get("colours") if isinstance(reply.get("colours"), list) else []) if isinstance(e, dict)}
        left_out = {w for w in fresh_words if w not in given and w not in lex.omitted}
        lex.omitted |= left_out
        self.check_colour_words()
        if past_cap and flush:
            self.judged_under = ""          # words that came late change how the listings read
        if answer or lex.colour_words or lex.material_spellings or lex.unit_by_spelling or spent:
            lex.sent |= {lowered(v).strip() for v in values} | (set(fresh_words) - left_out) | set(fresh_units)
        self.note(f"vocabulary round {lex.rounds}: {len(lex.colour_words)} colour words, {len(lex.material_spellings)} materials, {len(lex.unit_by_spelling)} unit spellings"
                  + ("" if answer else " (no answer)"))

    def materials_asked(self, answer: dict) -> dict:
        """Drop material entries that do not match a required material value."""
        wanted = {" ".join(word_list(r["value"])) for r in self.requirements if re.search(r"mater|fabric|ingredient", lowered(r["property"]))}
        entries = answer.get("materials") if isinstance(answer.get("materials"), list) else []
        return dict(answer, materials=[e for e in entries if isinstance(e, dict) and " ".join(word_list(e.get("material"))) in wanted])

    def word_notes(self, words: list[str]) -> str:
        """Build the reader prompt block: each word with its counts and sample variant names."""
        seen: dict[str, list] = {w: [0, 0, set()] for w in words}
        for page in self.counter.pages.values():
            for variant in page.get("variants") or []:
                options = lowered(variant.get("options"))
                parts = [p.strip() for p in re.split(r"[/,;|]+", options) if p.strip()]
                for word in set(re.findall(r"[a-z]{2,}", options)) & seen.keys():
                    entry = seen[word]
                    entry[0] += 1
                    entry[1] += word in parts
                    if len(options) <= 40:
                        entry[2].add(options)
        lines = ["Words from the store's variant names and colour/material facts, each with the number of variants that carry it, "
                 "the number whose name has that word standing alone, and sample names:"]
        for word in sorted(words):
            count, alone, names = seen[word]
            examples = sorted(names, key=lambda n: (len(n), n))[:2]
            lines.append(f"- {word} ({count}; alone {alone}" + ("; e.g. " + ", ".join(f'"{n}"' for n in examples) if examples else "") + ")")
        return "\n".join(lines)

    def words_missing(self, asked: list[str] = ()) -> list[str]:
        """List vocabulary gaps for the required values: materials, plain colours and units."""
        lex = self.glossary
        left: list[str] = []
        for r in self.requirements:
            value = lowered(r["value"]).strip()
            if (value in {lowered(a).strip() for a in asked} and re.search(r"colou?r", lowered(r["property"])) and re.fullmatch(r"[a-z]+", value)
                    and not lex.colours_of_word(value)):
                left.append(f"the colour {value}")
            if "material" in lowered(r["property"]) and not re.search(r"\d", r["value"]) and " ".join(word_list(r["value"])) not in lex.material_spellings:
                left.append(f"the material {value}")
            left += [f"the unit {unit}" for unit in re.findall(r"\d\s*([a-z]+)", value) if unit not in lex.unit_by_spelling and Errand.unit_word(unit) not in lex.unit_by_spelling]
        return list(dict.fromkeys(left))

    def check_colour_words(self) -> None:
        """Re-check, with the reader, extra words filed under the required colour, then drop false ones."""
        lex = self.glossary
        for requirement in self.requirements:
            family = self.colour_family(requirement) if self.kind(requirement) == "colour" else ""
            if not family or lex.checks >= COLOUR_CHECKS or self.late():
                continue
            own = self.forms(requirement) | self.forms(requirement, loose=True)
            words = sorted(w for w in lex.words_for_colour(family) if w != family and w not in lex.checked and w not in own and singular(w) not in own)
            if not words:
                continue
            lines = []
            for number, word in enumerate(words, start=1):
                names, titles = [], []
                for page in self.counter.pages.values():
                    for variant in page.get("variants") or []:
                        options = lowered(variant.get("options"))
                        if word in re.findall(r"[a-z]{2,}", options):
                            names += [options] if len(options) <= 40 and options not in names else []
                            titles += [str(page.get("title") or "")[:60]] if str(page.get("title") or "")[:60] not in titles else []
                lines.append(f"[{number}] {word}" + (f" (variant names: {', '.join(repr(n) for n in names[:2])}; listing: {titles[0]!r})" if titles else ""))
            lex.checks += 1
            lex.checked |= set(words)
            answer = self.reader.fill(COLOUR_CHECK_GUIDE, f"Colour asked for: {family}\nWords:\n" + "\n".join(lines), COLOUR_CHECK_TOOL, 200) or {}
            numbers = answer.get("names")
            if not isinstance(numbers, list):
                continue
            kept = {int(m.group(0)) for n in numbers for m in [re.search(r"\d+", str(n))] if m}
            out = [w for number, w in enumerate(words, start=1) if number not in kept]
            for word in out:
                del lex.colour_words[word]
            lex.refused |= set(out)
            if out:
                self.judged_under = ""          # forces listings to be judged again with the new word list
                self.note(f"colour words taken out of {family}: {out}")

    def read_request(self) -> str:
        request = self.counter.request
        plan = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 900) or {}
        if not [r for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]:
            again = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}\n\n{RECHECK_NOTE}", PLAN_TOOL, 700, second=True) or {}
            if again.get("requirements"):
                plan = {**plan, **{k: v for k, v in again.items() if v}} if plan else again
        if not self.grounded_priority(plan.get("priority"), request):
            # the first reading left the priority empty: one more reading by the fallback model, whose priority is kept only when the request backs it
            again = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}\n\n{PRIORITY_NOTE}", PLAN_TOOL, 700, second=True) or {}
            if self.grounded_priority(again.get("priority"), request):
                plan = {**plan, "priority": again["priority"], "priority_queries": again.get("priority_queries") or plan.get("priority_queries") or []}
        self.item = str(plan.get("item") or "").strip()
        self.category = str(plan.get("category") or "").strip()
        self.requirements = [self.requirement(r) for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        self.requirements = [dict(r, stated=True) for r in self.requirements if not NOT_A_PROPERTY.search(lowered(f"{r['property']} {r['value']}"))]
        # a value found only inside a stated purpose is the use itself, not an item property
        uses = [lowered(" ".join(str(w).split())) for w in plan.get("purposes") or [] if str(w).strip()]
        whole = lowered(" ".join(request.split()))
        rest = whole
        for use in uses:
            rest = rest.replace(use, " ")

        def same_root(a: str, b: str) -> bool:
            shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return shared >= 5 and shared * 10 >= 7 * min(len(a), len(b))

        def only_the_use(value: str) -> bool:
            words = [w for w in word_list(lowered(value)) if len(w) >= 4]
            if not words or re.search(r"\d", value) or rest == whole:
                return False
            in_use = all(any(w == u or same_root(w, u) for use in uses for u in word_list(use)) for w in words)
            return in_use and not any(w == u or same_root(w, u) for w in words for u in word_list(rest))

        self.requirements = [r for r in self.requirements if not only_the_use(r["value"])]
        self.beside: list[str] = []
        for r in list(self.requirements):
            use = self.beside_a_purpose(request, r, uses)
            if use:
                self.requirements.remove(r)
                self.beside.append(use)
        self.budget = money(plan.get("budget"))
        if self.budget is None:
            amounts = [float(m.replace(",", "")) for m in re.findall(r"(\d[\d,]*\.\d{2})", request)]
            self.budget = max(amounts) if amounts else None
        self.brand = str(plan.get("brand") or "").strip()
        # a make or model the request states as a requirement is judged by that property's own evidence, not by the page's brand field
        if self.brand and any(set(word_list(self.brand)) <= set(word_list(r["value"])) and "brand" not in lowered(r["property"]) for r in self.requirements):
            self.brand = ""
        self.listed = [(m.group(1), m.group(2)) for m in re.finditer(r"(\d{6,})::([\w-]+)", request)]
        self.priority = self.first_priority = self.grounded_priority(plan.get("priority"), request)
        if self.priority:
            # a requirement that merely restates the priority could never be met, so it is dropped
            most = self.name_words(self.priority)
            self.requirements = [r for r in self.requirements if re.search(r"\d", r["value"])
                                 or not (self.name_words(r["value"]) - {"the", "a", "an"} <= most and self.name_words(r["property"]) & most)]
        self.add_queries(self.value_queries() + list(plan.get("queries") or []))
        if self.priority:
            self.add_queries([q for q in plan.get("priority_queries") or [] if isinstance(q, str)][:4])
        pair = self.real_pair([" ".join(str(w).split()) for w in plan.get("doubted_pair") or [] if str(w).strip()][:2])
        purposes = ([" ".join(str(w).split()) for w in plan.get("purposes") or [] if str(w).strip()] + self.beside)[:3]
        self.note(f"plan item={self.item!r} requirements={self.requirements} budget={self.budget} pair={pair} purposes={purposes} "
                  f"priority={self.priority!r} invitation={plan.get('invitation')!r} open={plan.get('open_properties')}")
        if not self.queries:
            self.add_queries([" ".join(word_list(request)[:5])])
        purposes = [p for i, p in enumerate(purposes) if not any(i != j and self.within(p, q) and (not self.within(q, p) or j < i)
                                                                for j, q in enumerate(purposes))]
        self.purposes = purposes
        try:
            self.look_at = max(0, min(8, int(plan.get("look_at") or 0)))
        except (TypeError, ValueError):
            self.look_at = 0
        quoted = " ".join(str(plan.get("invitation") or "").split()).strip(" .\"'")
        asking = [w for w in word_list(lowered(quoted)) if len(w) >= 4]
        of_a_purpose = sum(any(same_root(w, u) or w == u for p in purposes for u in word_list(lowered(p))) for w in asking)
        apart = (len(quoted.split()) >= 2 and self.plain_quotes(quoted) in self.plain_quotes(request)
                 and lowered(quoted.split()[0]) not in ("who", "whom", "whose", "which", "that")
                 and not any(self.within(quoted, p) or self.within(p, quoted) for p in purposes)
                 and of_a_purpose * 2 < max(1, len(asking)))
        # phrases by which the request tells the assistant to ask or check with the shopper
        asks_me = r"\b(?:ask|check|confirm|consult)\b[^.?!]{0,24}\bme\b|\btell me (?:if|whether|when)\b|\blet me know\b|\bif that\b|\bjust say\b"
        of_the_pair = any(self.content_words(w) and self.content_words(w) <= self.content_words(quoted) for w in pair)
        # an explicit invitation to consult makes the bare use question go first
        invited = len(quoted.split()) >= 2 and self.plain_quotes(quoted) in self.plain_quotes(request) and bool(re.search(asks_me, self.plain_quotes(quoted)))
        if apart and len(purposes) < 2 and not re.search(asks_me, self.plain_quotes(quoted)) and not of_the_pair:
            # an instruction-like clause that is really a purpose becomes one, cut from its first article or -ing word
            words = quoted.split()
            start = next((i for i, w in enumerate(words) if lowered(w) in ("a", "an", "the", "my", "our") or (lowered(w).endswith("ing") and len(w) > 5)), None)
            if start is not None and len(words) - start >= 3:
                purposes = purposes + [" ".join(words[start:])]
                self.purposes = purposes
                apart = invited = False
        if len(purposes) < 2:
            purposes = purposes + self.unread_purposes(request, purposes, pair)[: 2 - len(purposes)]
            self.purposes = purposes
        self.needs_expected = max(1, min(2, len(purposes)))
        self.queue_properties([w for w in plan.get("open_properties") or [] if isinstance(w, str)])
        self.first_thoughts = self.to_ask[:4]
        # each wish needs at least half its words in the request, one of them distinctive
        said = set(word_list(request))
        shouted = {x.lower() for x in re.findall(r"\b[A-Z][A-Z0-9]+\b", request)}
        wishes = [[x for x in word_list(w) if x not in ("the", "a", "an", "of", "as", "its", "it", "to", "be", "with")] for w in pair]
        spoken = all(words and sum(x in said for x in words) * 2 >= len(words)
                     and any(x in said and (len(x) >= 3 or x[0].isdigit() or x in shouted) for x in words) for words in wishes)
        if len(pair) == 2 and spoken and len(" ".join(pair).split()) <= 16 and not self.mistaken_pair(pair, purposes):
            # a doubted pair is asked as its own question; neither wish is required until answered
            wished = lowered(" ".join(pair))
            self.set_aside = [r for r in self.requirements if lowered(r["value"]) in wished]
            self.requirements = [r for r in self.requirements if lowered(r["value"]) not in wished]
            self.pair_fallback = "pair"
            if self.brand and set(word_list(self.brand)) <= set(word_list(wished)):
                self.brand = ""
            self.pair_wishes = list(pair)
            self.pair_question = f"Which one matters more to you, {pair[0]} or {pair[1]}?"
            return self.pair_question
        if invited and purposes:
            # must go out in the opening turn: once an item is carted, a question goes unheard
            self.plain_asked = True
            return PLAIN_QUESTION
        return ""

    def beside_a_purpose(self, request: str, requirement: dict, uses: list[str]) -> str:
        """Return the use phrase when a value appears only in a purpose clause, else an empty string."""
        words = [w for w in word_list(lowered(requirement["value"])) if len(w) >= 4]
        if not words or re.search(r"\d", requirement["value"]) or not uses:
            return ""
        named = {singular(w) for w in word_list(self.item) if len(w) >= 3}
        found = ""
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(request).replace("\u2019", "'").split())):
            said = word_list(sentence)
            if not all(w in said for w in words):
                continue
            if (re.search(r"\d", sentence) or {singular(w) for w in said} & named
                    or not any(len(self.content_words(use) & self.content_words(sentence)) >= 2 for use in uses)):
                return ""
            for clause in re.split(r",\s*(?:and|but|while)\s+|;\s*", sentence):
                if not all(w in word_list(clause) for w in words):
                    continue
                tokens = clause.strip(" .!").split()
                start = next((i for i, w in enumerate(tokens) if lowered(w).endswith("ing") and len(w) > 5), None)
                position = next((i for i, w in enumerate(tokens) if words[0] in lowered(w)), None)
                if start is None or position is None or start >= position:
                    return ""
                found = " ".join(tokens[start:])
        return found

    def unread_purposes(self, request: str, purposes: list[str], pair: list[str]) -> list[str]:
        """Find purposes the plan reader missed: linked clauses that state no requirement or limit."""
        known = [self.content_words(p) for p in purposes]
        taken = [self.content_words(r["value"]) for r in self.requirements] + [self.content_words(w) for w in pair]
        priority = self.content_words(self.priority)
        named = self.content_words(self.item) | self.content_words(self.category) | {singular(w) for w in word_list(self.item) if len(w) >= 3}
        out: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(request).replace("\u2019", "'").split())):
            lead = re.match(r"(?:since|because|as)\s+(?:i'm|i am|we're|we are)\s+([^,]{8,80}),", lowered(sentence))
            if lead and lowered(lead.group(1)).split()[0].endswith("ing"):
                # a fronted reason clause that opens with an -ing verb is itself a purpose
                found = lead.group(1).strip()
                words = self.content_words(found)
                if words and not words & named and not any(len(k & words) >= 2 for k in known) and not any(t and t <= words for t in taken):
                    out.append(found)
                    known.append(words)
            if re.search(r"\d", sentence) or sentence.rstrip().endswith("?"):
                continue
            for clause in re.split(r",\s*(?:and|but|while)\s+|;\s*", sentence):
                said = self.content_words(clause)
                if (NOT_A_PROPERTY.search(lowered(clause)) or any(len(k & said) >= 2 or (k and k <= said) for k in known)
                        or any(t and t <= said for t in taken) or len(priority & said) >= 2
                        or re.search(r"\b(?:ask|check|confirm|justify|order|choose|pick|select|find)\b", lowered(clause))):
                    continue
                for link in re.finditer(r"\b(?:for|match(?:es|ing)?|follow(?:s|ing)?|correspond(?:s|ing)?|based on|goal is|purpose is|use is|meant|intended)\b", lowered(clause)):
                    words = clause[link.end():].strip(" .!").split()
                    start = next((i for i, w in enumerate(words) if lowered(w) in ("a", "an", "the", "my", "our")
                                  or (lowered(w).endswith("ing") and len(w) > 5)), None)
                    if start is None or len(words) - start < 3:
                        continue
                    found = " ".join(words[start:])
                    if self.content_words(found) & named:
                        continue
                    if not any(self.within(found, o) for o in out):
                        out.append(found)
                        known.append(self.content_words(found))
                    break
        if out:
            self.note(f"purposes the request gives besides: {out}")
        return out

    def queue_properties(self, names: list[str], first: bool = False, compared: bool = False) -> None:
        """Queue plain property names to ask about, skipping duplicates and settled or asked ones."""
        fresh: list[str] = []
        settled = [self.name_words(r["property"]) for r in self.requirements + self.set_aside]
        taken = [self.name_words(p) for p in self.asked + ([] if first else self.to_ask)]
        for name in names:
            raw = lowered(str(name).replace("_", " "))
            if re.search(r"[&/,+]|\b(?:and|or)\b", raw):
                continue            # a joined name would be heard as two questions
            text = " ".join(word_list(raw))
            words = self.name_words(text)
            if not words or len(text.split()) > 4 or NOT_A_PROPERTY.search(text) or "brand" in words:
                continue
            if any(len(self.content_words(text) & self.content_words(purpose)) >= 2 for purpose in self.purposes):
                continue
            if not compared and self.content_words(text) & self.content_words(self.priority):
                continue            # the priority's own quantity is rarely a hidden requirement
            if any(words == other for other in taken + [self.name_words(f) for f in fresh]):
                continue
            if any(words <= other or other <= words for other in settled if other):
                continue
            # names nested in one another are queued side by side; the shopper may know either
            beside = next((i for i, f in enumerate(fresh) if len(words) >= 2 and len(self.name_words(f)) >= 2
                           and (words < self.name_words(f) or self.name_words(f) < words)), None)
            if beside is None:
                fresh.append(text)
            else:
                fresh.insert(beside + 1, text)
        if first:
            kept = [p for p in self.to_ask if not any(self.name_words(p) == self.name_words(f) for f in fresh)]
            self.to_ask = fresh + kept
        else:
            self.to_ask += fresh

    def open_names(self) -> list[str]:
        known = self.asked + [r["property"] for r in self.requirements + self.set_aside]
        return [n for n in self.to_ask if not any(self.overlaps(n, other) or self.same_property(n, other) for other in known)]

    @staticmethod
    def property_core(name: str) -> list[str]:
        """Normalize a property name to singular words, minus trailing type/kind/family and tiny words."""
        words = [singular(w) for w in word_list(str(name).replace("_", " ").lower().replace("color", "colour")) if w not in ("the", "of")]
        while len(words) > 1 and (words[-1] in ("type", "kind", "family") or len(words[-1]) <= 2):
            words = words[:-1]
        return words

    @staticmethod
    def same_property(one: str, other: str) -> bool:
        """True when the shorter name is the tail of the longer, so both name one property."""
        short, longer = sorted((Errand.property_core(one), Errand.property_core(other)), key=len)
        return bool(short) and longer[len(longer) - len(short):] == short

    def next_property(self, plain: bool = False) -> str:
        """Take the next unasked property; with plain, prefer short names or a leading type."""
        open_names = self.open_names()
        short = [n for n in open_names if len(n.split()) <= 2 or n.split()[0] == "type"] if plain else []
        name = (short or open_names or [""])[0]
        self.to_ask = [n for n in self.to_ask if n in open_names and n != name]
        if name:
            self.asked.append(name)
        return name

    def next_question(self) -> str:
        """Return the next question text, naming exactly one property in the singular."""
        if self.reason_lists:
            first = self.plain_first()
            if first:
                self.current_names, self.current_ask, self.current_other = [], "", ""
                return first
            name = "" if self.pair_spent() else self.next_for_reason()
            if not name:
                closing = self.closing_question()
                if closing:
                    self.current_names, self.current_ask, self.current_other = [], "", ""
                    return closing
        else:
            name, self.current_reason = self.next_property(), None
        self.current_names = [name] if name else []
        self.property_asks += bool(name)
        self.current_ask, self.current_other = name, ""
        return ASK_ALONE.format(self.spoken(name)) if name else ""

    @staticmethod
    def spoken(name: str) -> str:
        """Return the property name with its last word singularized, for use in a question."""
        words = str(name).split()
        last = words[-1] if words else ""
        if last.endswith(("series", "species")):
            pass                    # same form for singular and plural
        elif len(last) > 4 and last.endswith("ies"):
            last = last[:-3] + "y"
        elif len(last) > 4 and last.endswith(("ches", "shes", "sses", "xes")):
            last = last[:-2]
        elif len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is")):
            last = last[:-1]
        return " ".join(words[:-1] + [last])

    def go_over(self) -> bool:
        """Re-queue wide-asked properties for a two-at-a-time retry; True when any were queued."""
        again = [n for n in dict.fromkeys(self.asked_wide) if n in self.asked][: 2 * ASK_NARROW]
        if self.narrow or not again or self.needs_found:
            return False
        self.narrow = 1
        self.asked = [p for p in self.asked if p not in again]
        self.to_ask = again + [n for n in self.to_ask if n not in again]
        return True

    def purpose_of(self, line: str) -> int | None:
        """Pick the open purpose sharing the most words with the line, else the first open one."""
        open_ones = [i for i in range(min(2, len(self.purposes))) if i not in self.purposes_met]
        said = self.content_words(line)
        shared = {i: len(self.content_words(self.purposes[i]) & said) for i in open_ones}
        ranked = sorted(open_ones, key=lambda i: -shared[i])
        return ranked[0] if ranked else None

    def put(self, question: str) -> dict:
        """Build the message action that asks the shopper one question."""
        self.last_question = question
        self.pair_open = bool(self.pair_question) and question == self.pair_question
        self.questions_asked += 1
        self.awaiting_answer = True
        return act("message", content=question[:400])

    def listen(self) -> tuple[bool, bool]:
        """Read the shopper's reply; no reply counts as an empty answer to a pending question."""
        if self.awaiting_answer and not self.counter.heard:
            self.awaiting_answer = False
            self.answered("", 0, spoke_of_else=False)
        return self.hear()

    def wants_to_ask(self) -> bool:
        counter = self.counter
        if self.reason_lists:
            if not self.reasons_open() and not self.closing_due():
                return False
        elif self.needs_found and self.questions_asked >= self.first_found_at + ASK_AFTER_ONE:
            return False
        elif not self.needs_found and self.property_asks >= ASK_UNHEARD:
            return False            # many property questions with no find suggests nothing is hidden
        if self.narrow > ASK_NARROW:
            return False
        if self.pair_spent() and not (self.reason_lists and self.closing_due()):
            return False            # after a pair question little is usually hidden, so only the likeliest are asked
        if self.waited >= WAIT_TURNS and self.unbuyable():
            # a request met only over the limit still hides a need that decides the closest listing: one more question
            if self.over_from is None:
                self.over_from = self.questions_asked
            if self.questions_asked > self.over_from:
                return False
        return (self.needs_found < self.needs_expected and self.questions_asked < ASK_LIMIT + (CLOSING_ROOM if self.closing_due() else 0) and counter.has("message")
                and not counter.over and counter.turns_left() > ASK_TURNS and not self.late()
                and not counter.cart)

    def unbuyable(self) -> bool:
        """True when one of two wishes was picked, nothing held back has come out, and of the listings read none within the limit meets the request while one just over it does."""
        picked = [r for r in self.requirements if r.get("chosen")]
        if not self.pair_chosen or len(picked) != 1 or self.needs_found or self.moved_on or self.counter.cart or not self.can_abstain():
            return False
        kept = (dict(self.verdicts), self.judged_under)
        self.silent = True          # the check reads the pages as they stand and sends the reader nothing
        try:
            self.judge()
            return not any(self.fits(k) for k in self.verdicts) and bool(self.only_over())
        finally:
            self.silent = False
            self.verdicts, self.judged_under = kept     # the verdicts of the real judging are made later, with the reader's words

    def need_candidates(self) -> list[tuple[str, str]]:
        """List (property, evidence) candidates for a hidden requirement, from colour, measures and labels."""
        pages = list(self.counter.pages.values())[:60]
        out: list[tuple[str, str]] = []
        coloured = sum(1 for p in pages if any(GLOSSARY.colours_of_word(w) for v in p.get("variants") or [] for w in word_list(lowered(v.get("options")))))
        spreads: dict[str, float] = {}
        if coloured:
            out.append(("colour", f"colour words in the variant names of {coloured} of {len(pages)} listings"))
            hues: dict[str, int] = {}
            for page in pages:
                for hue in {h for v in page.get("variants") or [] for w in word_list(lowered(v.get("options"))) for h in GLOSSARY.colours_of_word(w)}:
                    hues[hue] = hues.get(hue, 0) + 1
            if hues:
                spreads["colour"] = coloured / len(pages) * (1 - max(hues.values()) / sum(hues.values()))
        measured: dict[str, int] = {}
        amounts: dict[str, dict[str, int]] = {}
        loose: dict[str, dict[str, int]] = {}
        for page in pages:
            text = " / ".join([lowered(page.get("title"))] + [lowered(v.get("options")) for v in page.get("variants") or []])
            written: dict[str, set[str]] = {}
            unread: dict[str, str] = {}
            for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?\s?([a-z]{1,10})\b", text):
                quantity = GLOSSARY.unit_quantity.get(self.unit_spelled(m.group(1)))
                if quantity not in {None, "", "time", "other", "none"}:
                    written.setdefault(quantity, set()).add("".join(m.group(0).split()))
                elif quantity is None and len(m.group(1)) >= 2:
                    unread.setdefault(self.unit_word(m.group(1)), "".join(m.group(0).split()))
            for word, figure in unread.items():
                loose.setdefault(word, {})[figure] = loose.setdefault(word, {}).get(figure, 0) + 1
            for name, figures in written.items():
                measured[name] = measured.get(name, 0) + 1
                for figure in figures:
                    amounts.setdefault(name, {})[figure] = amounts.setdefault(name, {}).get(figure, 0) + 1
        for name, count in sorted(measured.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append((name, f"amounts in its units in the titles or variant names of {count} of {len(pages)} listings"))
            spreads[name] = count / len(pages) * (1 - max(amounts[name].values()) / sum(amounts[name].values()))
        # unit-like words with no known measure are offered as written for the priority reader
        for word, figures in sorted(loose.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
            count = sum(figures.values())
            if count >= 3 and 4 * count >= len(pages):
                shown = ", ".join(f for f, _ in sorted(figures.items(), key=lambda kv: (-kv[1], kv[0]))[:3])
                out.append((word, f"figures straight before the word '{word}' in the titles or variant names of {count} of {len(pages)} listings, as {shown}"))
                self.loose_units.add(word)
        # main category: the requested one if listings sit in it, else the most common leaf
        leaves: dict[str, int] = {}
        for page in pages:
            leaf = lowered((page.get("category_path") or [""])[-1])
            leaves[leaf] = leaves.get(leaf, 0) + 1
        main = lowered(self.category) if lowered(self.category) in leaves else (max(leaves, key=lambda leaf: leaves[leaf]) if leaves else "")
        own = [p for p in pages if lowered((p.get("category_path") or [""])[-1]) == main]
        own_labels = {str(label) for p in own for label in (p.get("facts") or {})}
        labels: dict[str, dict[str, int]] = {}
        for page in pages:
            for label, value in (page.get("facts") or {}).items():
                # labels found only on other kinds of item are skipped once five own-kind listings are read; materials stay
                elsewhere = len(own) >= 5 and str(label) not in own_labels and not re.search(r"mater|fabric", str(label), re.I)
                if not SALE_LABELS.search(str(label)) and not elsewhere:
                    seen = labels.setdefault(str(label), {})
                    for entry in self.entries(value)[:4]:
                        seen[entry] = seen.get(entry, 0) + 1
        carried = {label: sum(1 for p in pages if label in (p.get("facts") or {})) for label in labels}
        stated = [r for r in self.requirements + self.set_aside if r.get("stated")]
        self.need_settled = {" ".join(word_list(label.replace("_", " "))) for label, seen in labels.items()
                             if any(self.entry_is(entry, r, label) for entry in seen for r in stated)}
        for r in stated:
            for unit in re.findall(r"\d\s*([a-z]+)", lowered(r["value"])):
                quantity = GLOSSARY.unit_quantity.get(self.unit_spelled(unit))
                if quantity:
                    self.need_settled.add(quantity)
        self.need_spread = {" ".join(word_list(label.replace("_", " "))): carried[label] / len(pages) * (1 - max(seen.values()) / sum(seen.values()))
                            for label, seen in labels.items() if seen}
        # colour and measures use the same spread score so everything orders together
        for name, value in spreads.items():
            self.need_spread[name] = max(value, self.need_spread.get(name, -1.0))
        for label in sorted(labels, key=lambda name: (-carried[name], name))[:40]:
            name = " ".join(word_list(label.replace("_", " ")))
            top = ", ".join(v[:24] for v, _ in sorted(labels[label].items(), key=lambda kv: (-kv[1], kv[0]))[:5])
            evidence = f"store fact '{name}' on {carried[label]} of {len(pages)} listings, {len(labels[label])} different values: {top}"
            # a label for an already listed property merges its values into that entry
            plain = self.name_words(name) - {"family", "the"}
            twin = next((i for i, (other, _) in enumerate(out) if plain and self.name_words(other) - {"family", "the"} == plain), None)
            if twin is None:
                out.append((name, evidence))
            else:
                out[twin] = (out[twin][0], out[twin][1] + "; also " + evidence)
                self.need_spread[out[twin][0]] = max(self.need_spread.get(out[twin][0], -1.0), self.need_spread.get(name, -1.0))
        # properties guessed from the request are candidates even if no listing shows them
        for name in self.first_thoughts:
            # a stray label (shown on few listings) that only adds words to the guess is not the guessed property itself
            if not any(self.overlaps(name, other) and not (self.name_words(name) < self.name_words(other) and self.shown_share(evidence) < 0.25)
                       for other, evidence in out):
                out.append((name, "named from the request alone; no listing shows it"))
        return out

    @staticmethod
    def shown_share(evidence: str) -> float:
        """Fraction of read listings an evidence line says show the attribute; 1.0 when it names no count."""
        found = re.search(r"(\d+) of (\d+) listings", evidence)
        return int(found.group(1)) / max(1, int(found.group(2))) if found else 1.0

    def rank_needs(self) -> None:
        """Ask the reader once which shown properties each stated reason's hidden need concerns."""
        if self.needs_ranked or not self.counter.pages:
            return
        self.needs_ranked = True
        self.learn_words()
        candidates = self.need_candidates()
        if not candidates:
            return
        pages = [p for p in self.counter.pages.values() if self.in_category(p)] or list(self.counter.pages.values())
        leaves: dict[str, int] = {}
        for page in pages:
            leaf = str((page.get("category_path") or [""])[-1])
            leaves[leaf] = leaves.get(leaf, 0) + 1
        shelf = max(leaves, key=lambda leaf: leaves[leaf]) if leaves else (self.category or self.item)
        listing = "\n".join(f"{i}: {name} - {evidence}" for i, (name, evidence) in enumerate(candidates))
        if self.priority and not self.priced_priority():
            self.priority_unit()
        order_note = f"THE QUANTITY THE OPTIONS ARE RANKED BY (never a pick, except power for a room or place where the item runs): {self.priority}\n\n" if self.priority and not self.priced_priority() else ""
        text = (f"REQUEST:\n{self.counter.request}\n\nSTORE CATEGORY: {shelf}\n\n{order_note}"
                f"CANDIDATE ATTRIBUTES (number: name - what the listings show):\n{listing}")
        answer = self.reader.fill(NEED_RANK_GUIDE, text, NEED_RANK_TOOL, 900, think=True, rooms=max(1, min(2, len(self.purposes)))) or {}
        lists: list[list[str]] = []
        said: list[str] = []
        plains: list[bool] = []
        picked: set[int] = set()
        for reason in answer.get("reasons") or [] if isinstance(answer.get("reasons"), list) else []:
            names: list[str] = []
            shares: list[float | None] = []
            guesses: list[bool] = []
            unlisted = False
            unlisted_names: set[str] = set()
            for pick in (reason.get("candidates") or []) if isinstance(reason, dict) else []:
                if isinstance(pick, dict) and isinstance(pick.get("id"), int) and 0 <= pick["id"] < len(candidates):
                    picked.add(pick["id"])
                    name = self.askable(str(pick.get("ask_as") or "").strip() or candidates[pick["id"]][0])
                    # a property the request already fixes (a count in the stated amount) is no hidden need
                    if name and candidates[pick["id"]][0] not in self.need_settled and not any(self.overlaps(name, n) for n in names):
                        names.append(name)
                        shares.append(self.stated_share(candidates[pick["id"]][1]))
                        guesses.append(candidates[pick["id"]][1].startswith("named from the request alone"))
                elif isinstance(pick, dict) and pick.get("id") == -1 and not unlisted:
                    # an attribute no listing shows, which the rules point at for this reason
                    name = self.askable(str(pick.get("ask_as") or "").strip())
                    if name and not any(self.overlaps(name, n) for n in names):
                        names.append(name)
                        shares.append(None)
                        guesses.append(False)
                        unlisted = True
                        unlisted_names.add(name)
            # among two measurable picks, the one more listings quote a figure for goes first
            spots = [i for i, share in enumerate(shares) if share is not None]
            if len(spots) >= 2 and shares[spots[1]] >= SWAP_MIN and shares[spots[1]] >= SWAP_RATIO * shares[spots[0]]:
                names[spots[0]], names[spots[1]] = names[spots[1]], names[spots[0]]
            names = self.parcel_order(self.shown_first(names, guesses))
            # an attribute no listing shows (picked with -1) is rarely the held-back one: the first shown pick goes ahead of it
            blind = [name for name in names if name in unlisted_names]
            if names and names[0] in blind:
                ahead = next((n for n in names if n not in blind), None)
                if ahead:
                    names = [ahead] + [n for n in names if n != ahead]
            if names:
                lists.append(names[:4])
                said.append(str(reason.get("reason") or ""))
                plains.append(isinstance(reason, dict) and reason.get("points_to_attribute") is False)
        # leftover candidates after the picks: unspent request-named ones first, then widest spread
        spent = [candidates[i][0] in self.need_settled or self.within(candidates[i][0], self.priority) for i in range(len(candidates))]
        ahead = [i for i in range(len(candidates)) if i not in picked and not spent[i] and candidates[i][1].startswith("named from the request alone")][:2 if picked else 0]
        rest = sorted((i for i in range(len(candidates)) if i not in picked),
                      key=lambda i: (spent[i], i not in ahead, -self.need_spread.get(candidates[i][0], -1.0), i))
        tail = [name for name in (self.askable(candidates[i][0]) for i in rest if candidates[i][0] not in self.loose_units) if name and not self.ranks_options(name)]
        if not lists and tail and self.purposes:
            # no usable pick: each purpose still gets a list, filled from the leftovers
            lists = [[] for _ in self.purposes[:2]]
            said = [str(p) for p in self.purposes[:2]]
            plains = [False] * len(lists)
        if self.purposes and len(lists) > len(self.purposes):
            # reasons sharing no word with any purpose are not hidden needs, so drop them
            kept = [i for i, text in enumerate(said) if any(self.content_words(text) & self.content_words(p) for p in self.purposes)]
            if kept and len(kept) < len(lists):
                lists, said, plains = [lists[i] for i in kept], [said[i] for i in kept], [plains[i] for i in kept]
        if self.purposes and len(lists) > len(self.purposes):
            # one purpose split into several reasons is merged back into one list
            owners = [max(range(len(self.purposes)), key=lambda j: len(self.content_words(text) & self.content_words(self.purposes[j]))) for text in said]
            merged_lists, merged_said, merged_plain = [], [], []
            for j in sorted(set(owners)):
                members = [i for i, o in enumerate(owners) if o == j]
                names: list[str] = []
                for i in members:
                    names += [n for n in lists[i] if not any(self.overlaps(n, k) for k in names)]
                merged_lists.append(names)
                merged_said.append(self.purposes[j] if len(members) > 1 else said[members[0]])
                merged_plain.append(any(plains[i] for i in members))
            lists, said, plains = merged_lists, merged_said, merged_plain
        own = self.priority_pick(candidates)
        for names in lists:
            if own and not any(self.overlaps(own, k) for k in names):
                names.append(own)           # asked after the picks: the held-back requirement may concern the very quantity the options are ranked by
            for n in tail:
                if not any(self.overlaps(n, k) for k in names):
                    names.append(n)
        self.note(f"rank: {[names[:4] for names in lists]}")
        if isinstance(answer.get("reasons"), list) and not answer["reasons"] and not self.purposes:
            self.needs_expected = 0
        if lists:
            self.reason_lists = lists[:2]
            self.reason_said = said[:2]
            self.reason_plain = plains[:2]
            self.reason_asks = [0] * len(self.reason_lists)
            self.needs_expected = len(self.reason_lists)
            for met in sorted(self.purposes_met):
                # a purpose the use question already answered marks its closest reason as met
                if met < len(self.purposes):
                    shared = [len(self.content_words(self.purposes[met]) & self.content_words(text)) for text in said[:2]]
                    self.reasons_met.add(max(range(len(shared)), key=lambda i: shared[i]))

    def parcel_order(self, names: list[str]) -> list[str]:
        """When a reason's picks hold both the weight and the piece count, the shelf decides which goes first: the weight
        only when about a third or more of the opened listings write a weight, else the count."""
        weight, count = GLOSSARY.unit_quantity.get(GLOSSARY.unit_named("g")), GLOSSARY.unit_quantity.get(GLOSSARY.unit_named("piece"))
        if not weight or not count or weight == count:
            return names
        heavy = [i for i, name in enumerate(names) if weight in self.name_words(name)]
        counted = [i for i, name in enumerate(names) if count in self.name_words(name)]
        if not heavy or not counted or heavy[0] == counted[0]:
            return names
        pages = [p for p in self.counter.pages.values() if self.in_category(p)] or list(self.counter.pages.values())
        if not pages:
            return names
        written = 0
        for page in pages:
            texts = [page.get("title")] + [v.get("options") for v in page.get("variants") or [] if isinstance(v, dict)] + list((page.get("facts") or {}).values())
            if any(GLOSSARY.unit_quantity.get(self.unit_spelled(m.group(1))) == weight
                   for text in texts for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?\s?([a-z]+)", lowered(text))):
                written += 1
        first, second = (heavy[0], counted[0]) if 3 * written >= len(pages) else (counted[0], heavy[0])
        rest = [name for i, name in enumerate(names) if i not in (first, second)]
        at = min(first, second)
        self.note(f"parcel order: weight written on {written} of {len(pages)} listings -> {names[first]!r} first")
        return rest[:at] + [names[first], names[second]] + rest[at:]

    @staticmethod
    def shown_first(names: list[str], guesses: list[bool]) -> list[str]:
        """Put picks that only the request names behind the first picks that the listings show."""
        out: list[str] = []
        held: list[str] = []
        shown = 0
        for name, guess in zip(names, guesses):
            if guess and shown < SHOWN_FIRST:
                held.append(name)
                continue
            out.append(name)
            shown += not guess
            if shown >= SHOWN_FIRST:
                out, held = out + held, []
        return out + held

    def ranks_options(self, name: str) -> bool:
        """True when the name is the quantity the shopper ranks the options by, which cannot also be a held-back requirement."""
        if not self.priority or self.priced_priority():
            return False
        measure = GLOSSARY.unit_quantity.get(self.unit_spelled(self.counted_unit()), "")
        said = self.content_words(self.priority)
        if self.sized_priority():
            # what the quantity is taken of ('length of the piece') is the thing measured, not the quantity, so a piece count stays a candidate
            measured = re.split(r"\s(?:of|in|per|for)\s", lowered(self.figure_quantity), maxsplit=1)[1:]
            said -= self.content_words(measured[0]) if measured else set()
        return bool(self.content_words(name) & (said | {measure} - {""}))

    def priority_pick(self, candidates: list[tuple[str, str]]) -> str:
        """The first candidate that names the quantity the options are ranked by, worded for a question; empty when none can be asked."""
        return next((self.askable(name) for name, _ in candidates if self.ranks_options(name) and self.askable(name)), "")

    @staticmethod
    def stated_share(evidence: str) -> float | None:
        """Fraction of read listings stating a figure, for amount candidates; None for the rest."""
        found = re.match(r"amounts in its units in the titles or variant names of (\d+) of (\d+)", evidence)
        return int(found.group(1)) / max(1, int(found.group(2))) if found else None

    def askable(self, name: str) -> str:
        """The property worded for a question, or an empty string when it cannot be asked alone."""
        raw = lowered(name.replace("_", " "))
        if re.search(r"[&/,+]|\b(?:and|or)\b", raw):
            return ""
        parts = word_list(raw)
        while len(parts) > 1 and GLOSSARY.unit_quantity.get(self.unit_spelled(parts[-1])) not in (None, "", "count", "none", "other"):
            parts = parts[:-1]      # a store label ending in its unit ('power consumption w') is asked without it
        text = " ".join(parts)
        words = self.name_words(text)
        if not words or len(text.split()) > 4 or NOT_A_PROPERTY.search(text) or "brand" in words:
            return ""
        settled = [self.name_words(r["property"]) for r in self.requirements + self.set_aside]
        if any(words <= self.name_words(wish) for wish in self.pair_wishes):
            return ""  # a pick between the two wishes already settled this
        return "" if any(words <= other or other <= words for other in settled if other) else text

    def next_for_reason(self) -> str:
        """Next unasked property, with open reasons taking turns; empty when none is left."""
        count = len(self.reason_lists)
        for _ in range(count):
            index = self.reason_turn % count
            self.reason_turn += 1
            if index in self.reasons_met or not self.reason_room(index):
                continue
            for name in self.reason_lists[index]:
                if self.askable(name) and not self.asked_already(name):
                    self.reason_asks[index] += 1
                    self.current_reason = index
                    self.asked.append(name)
                    return name
        return ""

    def closing_words(self, index: int) -> str:
        text = self.reason_said[index].strip() if index < len(self.reason_said) else ""
        return text or (self.purposes[index] if index < len(self.purposes) else "")

    def closing_reasons(self) -> list[int]:
        """Reasons the closing question may target, widened when the need count still falls short."""
        out = [i for i in range(len(self.reason_lists)) if i not in self.reasons_met and i not in self.plain_dry]
        if self.closing_dry and self.needs_found < self.needs_expected:
            out += [i for i in range(len(self.reason_lists)) if i in self.reasons_met and i not in self.plain_dry]
        return out

    def closing_due(self) -> bool:
        out = self.closing_reasons()
        return CLOSING_QUESTIONS and bool(out) and (not self.plain_asked or any(i not in self.reason_asked and self.closing_words(i) for i in out))

    def owner_of(self, line: str, asked: int) -> int:
        """Reason that a revealed requirement belongs to, judged by word overlap with the shopper's line."""
        said = self.content_words(line)
        shared = {i: len(self.content_words(self.closing_words(i)) & said) for i in range(len(self.reason_lists)) if i == asked or i not in self.reasons_met}
        best = max(shared, key=lambda i: (shared[i], i == asked), default=asked)
        return best if shared.get(best, 0) > shared.get(asked, 0) else asked

    def pair_spent(self) -> bool:
        """True once a chosen pair has drawn nothing and its allowance of property questions is used."""
        return self.pair_chosen and not self.needs_found and self.property_asks - self.pair_at >= ASK_AFTER_PAIR

    def put_back(self, index: int) -> str:
        """Question that hands one reason back to the shopper."""
        self.reason_asked.add(index)
        self.current_reason = self.closing_index = index
        return REASON_QUESTION.format(self.item, self.closing_words(index))

    def plain_first(self) -> str:
        """Question for reasons that name no property or got stuck: the reason itself, then the use."""
        for index in range(len(self.reason_lists)):
            plain = index < len(self.reason_plain) and self.reason_plain[index]
            stuck = CLOSING_QUESTIONS and index < len(self.reason_asks) and self.reason_asks[index] >= CLOSE_AFTER
            if index in self.reasons_met or index in self.plain_dry or not (plain or stuck):
                continue
            if USE_FIRST and not self.plain_asked:
                self.plain_asked = True
                self.current_reason = index
                return PLAIN_QUESTION
            if index not in self.reason_asked:
                if self.closing_words(index) and (plain or self.reason_asks[index] >= REASON_AFTER):
                    return self.put_back(index)
            elif self.closing_dry and not self.plain_asked:
                self.plain_asked = True
                self.current_reason = index
                return PLAIN_QUESTION
        return ""

    def closing_question(self) -> str:
        if not self.closing_due():
            return ""
        out = self.closing_reasons()
        if USE_FIRST and not self.plain_asked:
            self.plain_asked = True
            self.current_reason = out[0]
            return PLAIN_QUESTION
        for index in out:
            if index not in self.reason_asked and self.closing_words(index):
                return self.put_back(index)
        if self.plain_asked:
            return ""
        self.plain_asked = True
        self.current_reason = out[0]
        return PLAIN_QUESTION

    def same_purpose(self, index: int) -> set[int]:
        def owner(i: int) -> int | None:
            words = self.content_words(self.reason_said[i]) if i < len(self.reason_said) else set()
            shared = [len(words & self.content_words(p)) for p in self.purposes]
            return max(range(len(shared)), key=lambda j: shared[j]) if shared and max(shared) else None
        mine = owner(index)
        return {i for i in range(len(self.reason_lists)) if i == index or (mine is not None and owner(i) == mine)}

    def reason_room(self, index: int) -> bool:
        """Whether the reason may still be asked about, under its cap or the free-question floor."""
        if index in self.plain_dry and self.reason_asks[index] >= PLAIN_DRY_ASKS:
            self.reasons_met.add(index)  # a silent use question means the reason hides nothing
            return False
        return self.reason_asks[index] < ASK_PER_REASON or self.questions_asked >= max(1, self.needs_expected, len(self.purposes)) + FLOOR_QUESTIONS

    def reasons_open(self) -> bool:
        return any(i not in self.reasons_met and self.reason_room(i)
                   and any(self.askable(n) and not self.asked_already(n) for n in names)
                   for i, names in enumerate(self.reason_lists))

    def asked_already(self, name: str) -> bool:
        return any(self.overlaps(name, p) or self.same_property(name, p) for p in self.asked)

    @staticmethod
    def overlaps(one: str, other: str) -> bool:
        """Two names for one property: every word of one appears in the other."""
        first, second = Errand.name_words(one), Errand.name_words(other)
        return bool(first and second) and (first <= second or second <= first)

    def shelf_properties(self) -> str:
        """Prompt text on the shelf: sample fitting listings, fact labels with values, amount units."""
        counter = self.counter
        pages = [p for p in counter.pages.values() if self.in_category(p)] or list(counter.pages.values())
        fitting = [k for k in self.verdicts if self.fits_words(k) and self.affordable(k)]
        shown, titles = [], set()
        for key in fitting + [k for k in self.verdicts if k not in fitting and (self.verdicts[k] or {}).get("kind_ok")]:
            title = lowered((counter.pages.get(key[0]) or {}).get("title"))
            if (title, self.variant_of(key).get("options")) in titles or len(shown) >= 12:
                continue
            if len(shown) >= 4 and key not in fitting:
                break
            titles.add((title, self.variant_of(key).get("options")))
            shown.append(key)
        labels: dict[str, dict[str, int]] = {}
        for page in pages:
            for label, value in (page.get("facts") or {}).items():
                seen = labels.setdefault(str(label), {})
                for entry in self.entries(value)[:4]:
                    seen[entry] = seen.get(entry, 0) + 1
        lines = []
        ranked = sorted(labels.items(), key=lambda kv: (-(len(kv[1]) > 1), -sum(kv[1].values()), kv[0]))
        for label, values in ranked[:ASK_SHELF]:
            top = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            lines.append(f"- {label} ({sum(values.values())} listings): " + ", ".join(v[:24] for v, _ in top))
        cards = "\n".join(self.card(i, key) for i, key in enumerate(shown, start=1)) or "(none read yet)"
        units = "\n".join(f"- {unit} ({count} listings, {figures} different figures)" for unit, count, figures in self.shelf_units()) or "(none)"
        return (f"Listings that meet what is known so far:\n{cards}\n\n"
                f"Fact labels on the {len(pages)} listings of this kind read:\n" + ("\n".join(lines) or "(none)")
                + f"\n\nAmounts the names and options of these listings state, by unit:\n{units}")

    def shelf_units(self) -> list[tuple[str, int, int]]:
        """Units in which listing names state amounts, with listing and figure counts, minus known ones."""
        counter = self.counter
        pages = [p for p in counter.pages.values() if self.in_category(p)] or list(counter.pages.values())
        known = {self.unit_word(m.group(2)) for r in self.requirements + self.set_aside for form in self.forms(r)
                 for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        seen: dict[str, tuple[set[int], set[float]]] = {}
        for index, page in enumerate(pages):
            text = " / ".join([lowered(page.get("title"))] + [lowered(v.get("options")) for v in page.get("variants") or []])
            for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s?([a-z]{1,10})\b", text):
                unit = self.unit_word(m.group(2))
                listings, figures = seen.setdefault(unit, (set(), set()))
                listings.add(index)
                figures.add(float(m.group(1)))
        rows = [(unit, len(listings), len(figures)) for unit, (listings, figures) in seen.items()
                if unit not in known and len(listings) >= 3 and len(figures) >= 2]
        return sorted(rows, key=lambda row: (-row[1], row[0]))[:8]

    @staticmethod
    def unit_word(word: str) -> str:
        """A unit as one singular word; three-letter plurals like pcs lose their s."""
        return word[:-1] if len(word) == 3 and word.endswith("s") else singular(word)

    @staticmethod
    def unit_spelled(word: str) -> str:
        word = lowered(word).strip()
        return GLOSSARY.unit_named(word)

    def read_shelf(self) -> None:
        """Have the reader pick what to ask about, given the request and how the listings differ."""
        if self.shelf_reads >= 4 or not self.counter.pages or (self.shelf_reads and len(self.open_names()) >= ASK_WIDE):
            return
        self.shelf_reads += 1
        self.judge()
        asked = "\n".join(f"- {question} -> {result}" for question, result in self.asked_log) or "(nothing yet)"
        still = "\n".join(f"- {name}" for name in self.to_ask) or "(none)"
        text = (f"{self.state_text()}\nPurposes named in the request: {'; '.join(self.purposes) or '(none)'}\n\n"
                f"Already asked:\n{asked}\n\nThought of before the listings were read, not asked yet:\n{still}\n\n{self.shelf_properties()}")
        answer = self.reader.fill(ASK_GUIDE, text, ASK_TOOL, 300) or {}
        names = answer.get("properties")
        names = re.split(r"[\n;,]+", names) if isinstance(names, str) else (names or [])
        shown = {unit for unit, _, _ in self.shelf_units()}
        measures: list[str] = []
        for entry in answer.get("measures") or [] if isinstance(answer.get("measures"), list) else []:
            unit, _, measure = str(entry).partition(":")
            measure = " ".join(word_list(measure))
            unit = self.unit_word((re.findall(r"[a-z]+", lowered(unit)) or [""])[0])
            if unit in shown and measure and measure != "none" and measure.split()[0] != "how" and len(measure.split()) <= 3:
                self.measure_units.setdefault(measure, set()).add(unit)
                if measure not in measures:
                    measures.append(measure)
        self.note(f"ask about: {names} measures: {measures}")
        if measures or [n for n in names if isinstance(n, str) and n.strip()]:
            # the reader's order replaces the earlier guesses
            first = [n for n in self.to_ask if n in self.first_thoughts]
            listed = [n for n in names if isinstance(n, str)]
            mixed: list[str] = []
            for index in range(max(len(listed), 3)):
                mixed += measures[index: index + 1] if index < 3 else []
                mixed += listed[index: index + 1]
            self.to_ask = []
            self.queue_properties(mixed[:4])
            self.queue_properties(first)
            self.queue_properties(mixed[4:] + measures[3:])
            # the quantity the priority compares goes last, flagged as compared
            self.queue_properties(measures, compared=True)

    def find_out(self) -> None:
        """Ask one property per turn, each question sharing its turn with reading or searching."""
        counter = self.counter
        opening = True
        while self.wants_to_ask():
            if counter.pages:
                self.rank_needs()
                if not self.reason_lists:
                    self.read_shelf()
            holding = False
            if not counter.pages and counter.rows and self.held_back < 2:
                # hold the question until some listing is read
                self.held_back += 1
                question = ""
            elif self.waited < WAIT_TURNS and self.unbuyable():
                # the picked wish is met only over the limit: more listings are read first, a fit within it brings the questions back
                self.waited += 1
                question, holding = "", True
            else:
                question = self.next_question()
                if not question and counter.pages and not self.reason_lists and self.shelf_reads < 4:
                    self.read_shelf()
                    question = self.next_question()
                if not question and counter.pages and not self.reason_lists and self.go_over():
                    question = self.next_question()
            if not question and not holding:
                if counter.pages or not counter.rows:
                    break
                question = ""
            work: list[dict] = []
            room = counter.group_size - (1 if question else 0)
            if opening:
                work = self.to_open(room * OPEN_BATCH)[:room]
            if not work:
                work = self.shelf_filters(room // 2)
                work += self.next_searches(room - len(work))
                if not work:
                    work = self.to_open(room * OPEN_BATCH)[:room]
            opening = not opening
            if not question and not work:
                break
            calls = ([self.put(question)] if question else []) + work
            counter.send(calls)
            self.listen()

    @staticmethod
    def content_words(text: str) -> set[str]:
        return {singular(w) for w in word_list(text) if len(w) >= 4} - {"that", "what", "with", "this", "from", "have", "them", "they", "their"}

    def grounded_priority(self, priority: Any, source: str) -> str:
        """Keep a priority only when the shopper's own text backs it; price priorities need every word."""
        text = " ".join(str(priority or "").split())
        words, have = self.content_words(text), self.content_words(source)
        if not words:
            return ""
        if re.search(r"\b(cheap|price|cost|expensive|afford)", lowered(text)):
            return text if words <= have else ""
        return text if words & have else ""

    def mentions(self, line: str, requirement: dict) -> bool:
        said = stems(line) | self.name_words(line)
        named = self.name_words(requirement["property"])
        if named and named <= said:
            return True
        if self.kind(requirement) == "amount":
            # a line quoting this figure beside another amount of the measure counts as speaking of it
            same, others = self.amounts(line, requirement)
            if (same or others) and self.says_value(line, requirement["value"]):
                return True
        return any(set(form.split()) <= said for form in self.forms(requirement) if form)

    @staticmethod
    def plain_quotes(text: Any) -> str:
        return lowered(text).replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')

    def within(self, inner: str, outer: str) -> bool:
        words = self.content_words(inner)
        return bool(words) and words <= self.content_words(outer)

    def is_wish(self, value: str) -> bool:
        """True when the value's words fit inside one of the two doubted wishes."""
        words = {singular(w) for w in word_list(value)}
        return bool(words) and any(words <= {singular(w) for w in word_list(wish)} for wish in self.pair_wishes)

    def mistaken_pair(self, pair: list[str], purposes: list[str]) -> bool:
        """True when a pair entry is really a purpose, the priority or the item rather than a wish."""
        said = [self.content_words(p) for p in purposes]
        priority = self.content_words(self.priority)
        named = self.content_words(self.item) | self.content_words(self.category)
        if all(self.content_words(w) and self.content_words(w) <= named for w in pair):
            return True  # two names of the one item are not a real pair
        return any(any(self.content_words(w) & s for s in said) or len(self.content_words(w) & priority) >= 2 for w in pair)

    def real_pair(self, pair: list[str]) -> list[str]:
        """The two wishes of a doubted pair, dropping purchase terms and splitting one joined member."""
        named = [self.content_words(self.item), self.content_words(self.category)]
        kept = [w for w in pair if not NOT_A_PROPERTY.search(lowered(w)) and not (self.content_words(w) and self.content_words(w) in named)]
        if len(kept) == len(pair):
            return pair
        if len(kept) != 1:
            return []
        parts = [part.strip() for part in re.split(r"\s+(?:and|&)\s+|\s*,\s*", kept[0]) if part.strip()]
        values = [self.content_words(r["value"]) for r in self.requirements]
        holders = [next((i for i, v in enumerate(values) if v and v <= self.content_words(part)), None) for part in parts]
        return parts if len(parts) == 2 and None not in holders and holders[0] != holders[1] else []

    @staticmethod
    def requirement(raw: dict) -> dict:
        def texts(name: str) -> list[str]:
            return [str(w).strip() for w in raw.get(name) or [] if str(w).strip()][:6] if isinstance(raw.get(name), list) else []
        value = re.sub(r"(\d)\s*(?:\"|\u201d|\u2033|'')", r"\1 inch", str(raw.get("value") or "").strip())  # inch marks become the word inch
        name = lowered(raw.get("property"))
        if re.fullmatch(r"\d+(?:\.\d+)?", value) and re.fullmatch(r"[a-z]{1,3}", name.strip()):
            value = f"{value} {name.strip()}"  # a bare number under a short unit-like name takes that name as its unit
        filed = not re.search(r"\d", value) and not re.search(r"colou?r|material", name)
        written = texts("written")
        written += [form for form in Errand.turned(value) if closed_up(form) not in {closed_up(w) for w in written}]
        return {"filed": filed, "property": str(raw.get("property") or "").strip(), "value": value,
                "written": written, "alike": texts("alike")}

    @staticmethod
    def turned(value: str) -> list[str]:
        """A metric amount restated in the unit a thousand times larger or smaller, if that is tidy."""
        found = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([cmk]?[lgm])", lowered(value).strip())  # litre, gram or metre with an optional c, m or k prefix
        if not found:
            return []
        number, unit = float(found.group(1)), found.group(2)
        if len(unit) == 2 and unit[0] == "m" and number >= 1000:
            number, unit = number / 1000, unit[1]
        elif len(unit) == 2 and unit[0] == "k":
            number, unit = number * 1000, unit[1]
        elif unit in ("l", "g") and number != int(number):
            number, unit = number * 1000, "m" + unit  # a fractional litre or gram is restated in milli units
        else:
            return []
        if abs(number * 100 - round(number * 100)) > 1e-6:
            return []
        return [f"{number:g}{unit}"]

    def state_text(self) -> str:
        lines = [f"Request: {self.counter.request}", f"Item: {self.item}",
                 "Requirements: " + ("; ".join(f"{r['property']} = {r['value']}" for r in self.requirements) or "(none)"),
                 f"Budget: {self.budget if self.budget is not None else 'none'}"]
        if self.changed_at:
            lines.insert(1, "The shopper has changed some of this since: where the request and the lines below differ, only the "
                            "requirements and the budget below hold now.")
        if self.priority:
            lines.append(f"Priority between fitting items: {self.priority}")
        return "\n".join(lines)

    @staticmethod
    def readable(text: str) -> bool:
        """False for the fragment of a malformed list ('["type"', a lone quote) that a truncated reader answer leaves behind."""
        return bool(re.search(r"[a-z0-9]", text)) and text[0] not in '[{"'

    def add_queries(self, queries: list, first: bool = False) -> None:
        fresh = []
        for query in queries:
            text = lowered(" ".join(str(query or "").split())[:QUERY_LENGTH])
            if text and self.readable(text) and text not in self.sent and self.read_as(text) not in self.sent_as and text not in fresh and (first or text not in self.queries):
                fresh.append(text)
        # when going first, entries already queued move up along with the fresh ones
        self.queries = fresh + [q for q in self.queries if q not in fresh] if first else self.queries + fresh

    @staticmethod
    def read_as(query: str) -> tuple[str, ...]:
        """Key under which the store treats a search alike: singular words, sorted, split on operators."""
        return tuple(sorted(singular(w) for w in re.split(r"[\s+\-&|!(){}\[\]^\"~*?:\\/,;]+", lowered(query)) if w))

    def values(self) -> list[str]:
        """Search words for each required value, closed up and without repeating the property name."""
        out = []
        for requirement in self.requirements:
            named = self.name_words(requirement["property"])
            value = requirement["value"]
            if re.search(r"\d", value) and len(closed_up(value).split()) > 1:
                # prefer a one-word written form with the same figures, the way listings spell it
                figures = re.findall(r"\d+(?:\.\d+)?", value)
                value = next((w for w in requirement.get("written") or [] if len(closed_up(w).split()) == 1
                              and re.findall(r"\d+(?:\.\d+)?", w) == figures), value)
            words = word_list(closed_up(value))
            if re.search(r"\d[.,]\d", value):
                # a decimal figure stays glued to its unit as a single word
                words = [w.strip(".,;:()") for w in closed_up(value).split() if w.strip(".,;:()")]
            kept = [w for w in words if singular(w) not in named]
            if kept and not re.search(r"\d", " ".join(kept)) and all(len(w) <= 3 for w in kept):
                kept = []  # short leftovers alone carry no meaning
            out.append(" ".join(kept or words))
        return out

    def store_spellings(self, requirement: dict) -> list[str]:
        """Other ways listings may write the value: converted units or other colour words."""
        self.learn_words()
        found = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)", lowered(requirement["value"]).strip())
        if found:
            number, unit = float(found.group(1)), self.unit_spelled(found.group(2))
            scale = GLOSSARY.listing_units(unit)
            apart = {GLOSSARY.unit_named("inch"): " inch", GLOSSARY.unit_named("year"): " year", GLOSSARY.unit_named("month"): " months"}
            out = []
            for other, size in scale.items():
                figure = number / size
                if other != unit and 0.1 <= figure < 1000 and abs(figure * 10 - round(figure * 10)) < 1e-6:
                    out.append(f"{figure:g}" + apart.get(other, other))
            return out
        family = self.colour_family(requirement) if self.kind(requirement) == "colour" else ""
        own = set(word_list(lowered(requirement["value"])))
        return [word for word in GLOSSARY.words_for_colour(family) if word not in own] if family and family != "multicolor" else []

    @staticmethod
    def spelled_amount(requirement: dict) -> list[str]:
        found = re.fullmatch(r"(\d+(?:\.\d+)?)[\s-]*([a-z]+)", lowered(requirement["value"]).strip())
        if not found:
            return []
        unit = GLOSSARY.unit_named(found.group(2))
        names = sorted({singular(name): name for name, known in GLOSSARY.unit_by_spelling.items()
                        if known == unit and len(name) >= 3 and re.fullmatch(r"[a-z]+", name) and singular(name) != singular(found.group(2))}.values(), key=len)
        return [f"{found.group(1)} {name}" for name in names[:2]]

    def item_words(self) -> list[str]:
        return [w for w in word_list(self.category or self.item) if len(w) >= 3 and w not in ("and", "the", "for", "with", "other")]

    def value_queries(self) -> list[str]:
        values, item = self.values(), self.category or self.item
        label = self.store_label()
        if label and values and lowered(label) != lowered(item):
            self.add_queries([f"{' '.join(values)} {label}", f"{values[-1]} {label}"])
        if not values or not item:
            return [item] if item else []
        every, newest = " ".join(values), values[-1]
        head = (self.item_words() or [""])[-1]
        out = [f"{every} {item}", f"{newest} {item}", f"{values[0]} {item}"]
        # a category joining several kinds is also searched under each kind alone
        parts = [part for part in (" ".join(word_list(piece)) for piece in re.split(r"[&,]|\band\b", item)) if part]
        if len(parts) > 1:
            out[1:1] = [f"{every} {part}" for part in parts[:3]]
        # short forms: values with the item's last word, values alone, and respelled values
        short = [f"{every} {head}".strip()] + ([every] if len(every.split()) >= 2 else [])
        for requirement in self.requirements:
            for other in self.store_spellings(requirement)[:3]:
                turned = " ".join(other if r is requirement else v for r, v in zip(self.requirements, values))
                short += [f"{turned} {head}".strip()] + ([turned] if len(turned.split()) >= 2 else [])
        out[1:1] = short
        kinds = [" ".join(word_list(part)) for part in re.split(r"&|/|,|\band\b", item)]
        if len([k for k in kinds if k]) > 1:
            out[1:1] = [f"{every} {kind}" for kind in kinds if kind]
        for requirement in self.requirements:
            for other in self.amount_spellings(requirement)[:3]:
                if closed_up(other) != closed_up(requirement["value"]):
                    turned = " ".join(closed_up(other if r is requirement else r["value"]) for r in self.requirements)
                    out.append(f"{turned} {item}")
                    if head and len(item.split()) > 1 and re.search(r"\d", other) and other in self.turned(requirement["value"]):
                        out.append(f"{turned} {head}")
        for requirement in self.requirements:
            for other in self.spelled_amount(requirement):
                turned = " ".join(other if r is requirement else v for r, v in zip(self.requirements, values))
                out += [f"{turned} {head}".strip(), turned] if len(turned.split()) > 2 else [f"{turned} {head}".strip()]
        out += [f"{every} {w}" for w in self.item_words()[:3]]
        if len(values) >= 2:
            out.append(every)
        out += [f"{newest} {w}" for w in self.item_words()[:2]]
        earlier = " ".join(values[:-1])
        whole = " ".join(word_list(closed_up(self.requirements[-1]["value"])))
        if whole and not re.search(r"\d", whole):
            # newest value whole and with its last word toggled between singular and plural
            head, _, tail = whole.rpartition(" ")
            turned = tail[:-1] if tail.endswith("s") and len(tail) > 3 else tail + "s"
            forms = ([whole] if whole != newest else []) + [f"{head} {turned}".strip()]
            out[3:3] = [" ".join(f"{earlier} {form} {item}".split()) for form in forms]
        # synonyms for the newest value, for listings that use neither the shopper's word
        for other in (self.requirements[-1].get("alike") or [])[:2]:
            if closed_up(other) != closed_up(self.requirements[-1]["value"]):
                out.append(" ".join(f"{earlier} {closed_up(other)} {item}".split()))
                plain_word = (self.item_words() or [""])[-1]
                if plain_word and len(item.split()) > 1:
                    out.insert(3, " ".join(f"{earlier} {closed_up(other)} {plain_word}".split()))
        # the store reads 2.19inch and 2.19 inch as different words and titles write a decimal figure either way: the apart forms follow the rest
        apart = [self.figures_apart(text) for text in out]
        out += list(dict.fromkeys(form for form, text in zip(apart, list(out)) if form != text and form not in out))[:APART_LIMIT]
        return out

    @staticmethod
    def figures_apart(text: str) -> str:
        """The search with each decimal figure set apart from its unit (2.19inch -> 2.19 inch)."""
        return re.sub(r"(?<![\w.])(\d+\.\d+)([a-z]{1,4})(?![\w])", r"\1 \2", text)

    def shelf_queries(self, count: int) -> list[str]:
        """Searches pairing frequent title words of this category with the required values."""
        pages = [p for p in self.counter.pages.values() if self.in_category(p)]
        titles = [self.counter.rows[k].get("title") for k in self.counter.shelf if k in self.counter.rows] + [p.get("title") for p in pages]
        if len(titles) < 3 or not self.requirements:
            return []
        title_words = [set(word_list(title)) for title in titles]
        values = self.shelf_values(title_words, [set(word_list(v.get("options"))) for p in pages for v in p.get("variants") or []])
        known = set(self.item_words()) | {w for v in self.values() + values for w in word_list(closed_up(v))}
        # titles already stating a just-told value are left out of the word counts, unless most titles state it
        told = [r for r in self.requirements if lowered(r["value"]) in {lowered(v) for v in self.told_now}]
        others = [words for title, words in zip(titles, title_words) if not any(self.says(title, r) for r in told)]
        counted = others if told and len(others) >= 3 and 2 * len(others) > len(titles) else title_words
        counts: dict[str, int] = {}
        for words in counted:
            for word in words:
                if len(word) >= 4 and not word.isdigit() and word not in known:
                    counts[word] = counts.get(word, 0) + 1
        common = [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2]
        chosen = common[self.shelf_words_used: self.shelf_words_used + count]
        self.shelf_words_used += len(chosen)
        every = " ".join(values)
        # head word: the item word the most titles on this shelf carry
        kinds = [{singular(w) for w in words} for words in title_words]
        head = max(reversed(self.item_words()), key=lambda w: sum(singular(w) in k for k in kinds), default="")
        out = [f"{w} {every}" for w in chosen] + [f"{w} {every} {head}".strip() for w in chosen]
        apart = re.sub(r"(\d)([a-z])", r"\1 \2", every)
        if apart != every:
            # also try the amount with a space between figure and unit
            out[1:1] = [f"{w} {apart}" for w in chosen[:3]]
        return out

    def shelf_values(self, titles: list[set[str]], options: list[set[str]]) -> list[str]:
        """Required values worded as this shelf's variants most often name them."""
        def used(phrase: str, texts: list[set[str]]) -> int:
            return sum(set(word_list(phrase)) <= words for words in texts)
        out = []
        for requirement, value in zip(self.requirements, self.values()):
            spellings = self.store_spellings(requirement) if self.kind(requirement) == "colour" else []
            best = max(spellings, key=lambda word: used(word, options), default="")
            swap = best and used(best, options) > max(used(value, options), used(best, titles))
            out.append(best if swap else value)
        return out

    def twin_queries(self) -> list[str]:
        """Search built from an off-shelf listing that states every value and may be the same article."""
        every = " ".join(self.values())
        if every in self.twinned or not any(self.in_category(p) for p in self.counter.pages.values()):
            return []
        self.twinned.add(every)
        given = set(word_list(closed_up(every)))
        kind = {singular(w) for w in self.item_words()}
        for product_id, page in self.counter.pages.items():
            title = page.get("title")
            if self.in_category(page) or not kind <= {singular(w) for w in word_list(title)}:
                continue
            keys = [(product_id, str(v.get("sku"))) for v in page.get("variants") or []]
            if any(self.affordable(key) and all(self.plain(key, r) >= 2 for r in self.requirements) for key in keys):
                return [lowered(" ".join([every] + [w for w in word_list(title) if w not in given][:10]))[:QUERY_LENGTH]]
        return []

    def kin_titles(self) -> list[str]:
        return list(dict.fromkeys(lowered((self.counter.pages.get(k[0]) or {}).get("title")) for k in self.verdicts if self.fits_words(k)))

    def kin_pending(self) -> bool:
        return len(self.kin_done) < 4 and any(title not in self.kin_done for title in self.kin_titles())

    def kin_queries(self) -> list[str]:
        """Searches pairing the values with titles of fitting listings, to reach their other variants."""
        every = " ".join(self.values())
        out = []
        for key in [k for k in self.verdicts if self.fits_words(k)]:
            page = self.counter.pages.get(key[0]) or {}
            title = lowered(page.get("title"))
            if title in self.kin_done or len(self.kin_done) >= 4:
                continue
            self.kin_done.add(title)
            words = word_list(title)
            if len(words) < 2:
                continue
            out += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.item} {' '.join(words[:3])}"]
            if len(words) > 10:
                out.append(f"{every} {' '.join(words)}")
            options = " ".join(word_list(self.variant_of(key).get("options")))
            if options:
                out.append(f"{options} {' '.join(words[:10])}")
            brand = lowered(page.get("brand"))
            if brand and brand not in ("none", "no brand", "oem"):
                out.append(f"{every} {brand} {self.item}")
            out.append(f"{every} {' '.join(words[:2])}")
            elsewhere = {w for row in self.counter.rows.values() if lowered(row.get("title")) != title for w in word_list(row.get("title"))}
            own = sorted((w for w in dict.fromkeys(words) if len(w) >= 5 and not w[0].isdigit() and w not in elsewhere), key=len, reverse=True)
            if own:
                out.append(own[0])  # a title word no other listing carries points straight at this article
        return out

    def maker_filters(self) -> None:
        """Queue a filter on the shelf and brand of a near-fitting listing to reach its sibling listings."""
        if not self.counter.has("filter") or self.budget is None:
            return
        price = lambda k: (self.counter.stock.get(k) or {}).get("price") or 0.0
        seeds = self.ranked() + sorted((k for k in self.verdicts if self.fits_words(k)), key=price)
        shelved = any(self.in_category(p) for p in self.counter.pages.values())
        for key in dict.fromkeys(seeds):
            if len(self.maker_shelves) >= MAKER_LIMIT:
                return
            page = self.counter.pages.get(key[0]) or {}
            brand = str(page.get("brand") or "").strip()
            label = str((page.get("category_path") or [""])[-1] or "").strip()
            tag = f"{lowered(label)}|{lowered(brand)}|{self.budget}"
            if not brand or lowered(brand) in ("none", "no brand", "oem") or not label or tag in self.maker_shelves:
                continue
            if shelved and not self.in_category(page) or self.brand and not set(word_list(self.brand)) <= set(word_list(brand)):
                continue
            self.maker_shelves.add(tag)
            self.pending_filters.append(act("filter", category=label, brand=brand, max_price=self.budget, in_stock=True, k=10))

    def variant_queries(self, wide: bool = False) -> list[str]:
        """Searches pairing a listing's title with the wanted value, to reach a variant not shown."""
        out = []
        stated = lambda item: -sum(max(self.plain(item[0], r), 0) for r in self.requirements) if self.requirements else 0
        for key, verdict in sorted(self.verdicts.items(), key=stated):
            if len(self.probed) >= (PROBE_LIMIT if wide else 4) or len(out) >= 8:
                break
            if not verdict.get("kind_ok") or verdict.get("by") != "words" or not self.requirements:
                continue
            levels = [self.plain(key, r) for r in self.requirements]
            # an unmet colour or amount may sit on a sibling variant; other unmet values rule it out unless wide
            off = [r for r, level in zip(self.requirements, levels) if level < 1 and self.kind(r) in ("colour", "amount")]
            unmet_other = any(level < 1 and self.kind(r) not in ("colour", "amount") for r, level in zip(self.requirements, levels))
            if not off or (unmet_other and not wide):
                continue
            title = lowered((self.counter.pages.get(key[0]) or {}).get("title"))
            if not title or title in self.probed:
                continue
            self.probed.add(title)
            words = " ".join(word_list(title)[:10])
            for requirement in off:
                spelled = self.store_spellings(requirement)[:1] if self.kind(requirement) == "colour" else []
                for word in dict.fromkeys([closed_up(requirement["value"])] + spelled + list(requirement.get("alike") or [])[: 2 - len(spelled)]):
                    out.append(f"{word} {words}")
        return out

    def sibling_queries(self, count: int) -> list[str]:
        """While nothing fits, pair the values with titles of seen listings to reach other variants."""
        every = " ".join(self.values())
        if not every.strip():
            return []
        wanted = set(word_list(closed_up(every)))

        # values the shopper withdrew still count toward a title's score
        before = {w for old in self.took_back.values() for w in word_list(closed_up(old))}

        def holds(text: str) -> int:
            return len(wanted & set(word_list(text))) + len(before & set(word_list(text)))
        read = [p for p in self.counter.pages.values() if self.in_category(p)]
        read.sort(key=lambda p: -holds(" ".join([str(p.get("title") or "")] + [str(v.get("options") or "") for v in p.get("variants") or []])))
        shown = [row for key, row in self.counter.rows.items() if key[0] not in self.counter.pages]
        shown.sort(key=lambda row: -holds(str(row.get("title") or "")))
        titles = [p.get("title") for p in read] + [row.get("title") for row in shown]
        if before:
            scored = [(holds(" ".join([str(p.get("title") or "")] + [str(v.get("options") or "") for v in p.get("variants") or []])), p.get("title")) for p in read]
            scored += [(holds(str(row.get("title") or "")), row.get("title")) for row in shown]
            titles = [title for _, title in sorted(scored, key=lambda item: -item[0])]
        out = []
        for title in titles:
            if len(out) >= count or len(self.sibling_done) >= SIBLING_LIMIT:
                break
            head = " ".join(word_list(lowered(title))[:10])
            # titles already used by the kin and variant searches are skipped
            if not head or head in self.sibling_done or lowered(title) in self.kin_done or lowered(title) in self.probed:
                continue
            self.sibling_done.add(head)
            query = lowered(" ".join(f"{every} {head}".split())[:QUERY_LENGTH])
            self.sibling_sent.add(query)
            out.append(query)
        return out

    def leave_one_queries(self) -> list[str]:
        """Queries made of the required values with each one omitted in turn."""
        values = [v for v in self.values() if v]
        key = " ".join(values)
        if len(values) < 2 or key in self.left_out:
            return []
        self.left_out.add(key)
        head = (self.item_words() or [""])[-1]
        out = [f"{' '.join(v for j, v in enumerate(values) if j != i)} {head}".strip() for i in range(len(values))]
        if len(values) >= 3:
            out += [" ".join(v for j, v in enumerate(values) if j != i) for i in range(len(values))]
        return out

    def family_queries(self, count: int) -> list[str]:
        """Searches by the bare title of listings already read, to surface their sibling variants."""
        read = [p for p in self.counter.pages.values() if self.in_category(p)]
        out = []
        for page in read:
            if len(out) >= count or len(self.family_done) >= FAMILY_LIMIT:
                break
            head = " ".join(word_list(lowered(page.get("title")))[:8])
            if len(head.split()) < 3 or head in self.family_done:
                continue
            self.family_done.add(head)
            out.append(head)
        return out

    def label_guesses(self) -> list[str]:
        """Likely store category names built from the item words, in singular and plural forms."""
        base = self.category or self.item
        if not base:
            return []

        def plural(word: str) -> str:
            if word.endswith("s"):
                return word
            if re.search(r"(x|ch|sh)$", word):
                return word + "es"
            if re.search(r"ife$", word):
                return word[:-2] + "ves"
            if re.search(r"(?:[lr]f|[oe]af)$", word):
                return word[:-1] + "ves"
            if re.search(r"[^aeiou]y$", word):
                return word[:-1] + "ies"
            return word + "s"

        out = []
        parts = base.split()
        for skip in range(min(3, len(parts))):
            phrase = " ".join(parts[skip:])
            if skip and not re.match(r"[A-Za-z]{3}", phrase):
                continue
            last = re.sub(r"([A-Za-z]+)(\W*)$", lambda m: plural(m.group(1)) + m.group(2), phrase)
            every = re.sub(r"([A-Za-z]+)(?=\s*(?:&|/|,|\band\b)|\W*$)", lambda m: plural(m.group(1)), phrase)
            out += [phrase, last, every] if not skip else [last]
        return list(dict.fromkeys(out))

    def store_label(self) -> str:
        named = self.named_shelf()
        if named:
            return named
        if self.counter.shelf:
            return next(iter(self.counter.shelf.values()))
        wanted = {label_key(self.category), label_key(self.item)} - {""}
        for page in self.counter.pages.values():
            for label in page.get("category_path") or []:
                if label_key(label) in wanted:
                    return str(label)
        return ""

    def shelf_filters(self, count: int = 4) -> list[dict]:
        """Category filters at several fractions of the budget, each exposing other listings."""
        label = self.store_label()
        if not label or not self.counter.has("filter") or self.budget is None:
            return []
        out = []
        for share in (0.7, 0.45, 0.25, 0.12, 0.85, 0.55, 0.35, 0.18, 0.08, 0.04):
            tag = f"filter|{lowered(label)}|{share}"
            if tag not in self.sent and len(out) < count and len(self.sent) < self.search_room:
                self.sent.add(tag)
                out.append(act("filter", category=label, max_price=round(self.budget * share, 2), in_stock=True, k=10))
        return out

    def queue_filters(self) -> None:
        """Queue category and brand filters under the current budget for the next searches."""
        brand = next((r["value"] for r in self.requirements if "brand" in lowered(r["property"])), None)
        label = self.store_label()
        for category in ([label] if label else self.label_guesses()[:2]):
            self.pending_filters.append(act("filter", category=category, brand=brand, max_price=self.budget, in_stock=True, k=10))
        if brand:
            self.pending_filters.append(act("filter", brand=brand, max_price=self.budget, in_stock=True, k=10))

    def next_searches(self, room: int) -> list[dict]:
        out = self.pending_filters[:room] if self.counter.has("filter") else []
        self.pending_filters = self.pending_filters[len(out):] if out else []
        while self.queries and len(out) < room and len(self.sent) < self.search_room:
            query = self.queries.pop(0)
            if query in self.sent or self.read_as(query) in self.sent_as:
                continue
            self.sent.add(query)
            self.sent_as.add(self.read_as(query))
            out.append(act("search", query=query, k=10, in_stock=True))
        return out

    @staticmethod
    def listed_fields(answer: dict | None) -> dict:
        """Normalize a reader answer so that adds, drops and queries are always lists."""
        answer = dict(answer or {})
        for name in ("adds", "drops", "queries"):
            if isinstance(answer.get(name), str):
                try:
                    answer[name] = json.loads(answer[name])
                except ValueError:
                    # truncated or unclosed list: keep the entries that are complete
                    fixed = Reader._closed('{"v": ' + answer[name] + "}")
                    answer[name] = fixed.get("v") if fixed and isinstance(fixed.get("v"), list) else []
            if not isinstance(answer.get(name), list):
                answer[name] = []
        return answer

    def hear(self) -> tuple[bool, bool]:
        """Process queued shopper lines; returns (something changed, a reply is expected)."""
        changed = asks = False
        while self.counter.heard:
            line = self.counter.heard.pop(0)
            parts = [part.strip() for part in line.split("\n") if part.strip()]
            if len(parts) > 1:
                # a multi-line message is handled line by line, in order
                self.counter.heard[:0] = parts
                continue
            before = [(r["property"], r["value"]) for r in self.requirements]
            held, changed_so_far = list(self.requirements), changed
            limit_before = self.budget
            said = f"{self.state_text()}\n\n"
            if self.awaiting_answer and self.last_question:
                said += f"The assistant had just asked: {self.last_question}\n"
            said += f"The shopper's new line:\n{line}"
            if self.awaiting_answer and self.restates_budget(line):
                # a bare repeat of the known limit carries nothing to read
                heard = self.listed_fields(None)
                if line != self.shelf_said:
                    self.shelf_said, self.shelf_cache = line, (-1, "")
            else:
                heard = self.own_value(self.listed_fields(self.reader.fill(HEAR_GUIDE, said, HEAR_TOOL, 1500)), line)

                swap = re.search(r"\binstead of\b([^,.;!?]*)", lowered(line))

                def real_adds(reading: dict) -> list[dict]:
                    """Keep only added requirements that are real values, not labels or values being given up."""
                    kept = []
                    for a in reading.get("adds") or []:
                        value = lowered(a.get("value")) if isinstance(a, dict) else ""
                        if not isinstance(a, dict):
                            continue
                        if re.fullmatch(SAID_NEEDED, value):
                            continue
                        named = {w for n in self.asked[-4:] for w in self.name_words(n)}
                        # a property label or the whole item name read as the value (one word of a longer item name is fine)
                        if self.awaiting_answer and value and (self.name_words(value) <= named or self.name_words(value) == self.name_words(self.item)):
                            continue
                        if value and self.name_words(value) == self.name_words(str(a.get("property") or "")) and not NOT_A_PROPERTY.search(value):
                            continue
                        # the value after "instead of" is the one given up
                        if swap and value and value in swap.group(1) and value not in lowered(line)[:swap.start(1)] + " " + lowered(line)[swap.end(1):]:
                            continue
                        if self.names_itself(a):
                            continue
                        kept.append(a)
                    return kept
                found = [a for a in heard.get("adds") or [] if isinstance(a, dict)]
                if len(real_adds(heard)) < len(found) or (self.awaiting_answer and not found):
                    # nothing usable came out of the first reading: read the line once more
                    again = self.own_value(self.listed_fields(self.reader.fill(HEAR_GUIDE, said + HEAR_RECHECK, HEAR_TOOL, 1500, second=True)), line)
                    if real_adds(again):
                        heard = again
                if any(isinstance(a, dict) and re.fullmatch(SAID_NEEDED, lowered(a.get("value"))) for a in heard.get("adds") or []):
                    # the value came out as the "needed" word: the last word of the property label is the real value
                    mended = []
                    for a in heard["adds"]:
                        words = str(a.get("property") or "").split() if isinstance(a, dict) else []
                        if words and re.fullmatch(SAID_NEEDED, lowered(a.get("value"))) and len(words) >= 2 and self.says_value(line, words[-1]):
                            a = dict(a, property=" ".join(words[:-1]), value=words[-1])
                        mended.append(a)
                    heard["adds"] = mended
                if heard.get("adds") and not real_adds(heard) and self.awaiting_answer:
                    # still only labels: the regex finds a "the X is required" phrase, and X minus the asked names is the value
                    said_so = re.search(r"\bthe ([a-z0-9 \-]+?) (?:is|are) (?:required|needed|necessary|essential|a must)\b", lowered(line))
                    named = self.name_words(self.item) | {w for n in self.asked[-4:] for w in self.name_words(n)}
                    left = [w for w in word_list(said_so.group(1)) if singular(w) not in named and w not in ("of", "a", "an")] if said_so else []
                    asked_for = next((n for n in reversed(self.asked[-4:]) if self.name_words(n) & self.name_words(said_so.group(1))), "") if said_so else ""
                    if left and len(left) <= 3 and asked_for:
                        heard["adds"] = [{"property": asked_for, "value": " ".join(left)}]
                if heard.get("adds"):
                    heard["adds"] = real_adds(heard)
            answering, self.awaiting_answer = self.awaiting_answer, False
            nothing = answering and not any(heard.get(k) for k in ("adds", "drops", "budget", "priority", "about_change"))
            if nothing and len(self.current_names) == 2:
                heard["asks"] = False       # two properties were asked together; the follow-up questions answer this
            given = replaced = 0
            withdrawn = []
            if heard.get("about_change") is True:
                # a figure in a reaction to the market change is a price, not a budget
                heard["budget"] = 0
            changes = not answering and bool(re.search(CHANGE_WORDS, lowered(line)))

            def gives_up(r: dict) -> bool:
                return self.says_value(line, r["value"]) or changes

            def supported(a: dict) -> bool:
                """True when the line itself backs the add (the add loop below skips the others), so the value it replaces really goes."""
                back = next((now for now, before in self.took_back.items() if lowered(before) == lowered(a.get("value"))), None)
                return self.says_value(line, str(a.get("value") or "")) and (back is None or self.says_value(line, back))
            explicit = {lowered(a.get("replaces")).strip() for a in heard.get("adds") or [] if isinstance(a, dict) and str(a.get("replaces") or "").strip() and supported(a)}

            def unspoken(r: dict) -> bool:
                """True when the line only answers our question and speaks neither of the requirement's property nor of its value."""
                return answering and not self.pair_open and not re.search(CHANGE_WORDS, lowered(line)) and not (self.mentions(line, r) or self.says_value(line, r["value"]))

            def replaced_by_reader(r: dict) -> bool:
                value = lowered(r["value"]).strip()
                return not unspoken(r) and any(value == said or (len(said) >= 3 and (said in value or value in said)) for said in explicit)
            drops = list(heard.get("drops") or [])
            drops += [r["property"] for r in self.requirements if replaced_by_reader(r) and lowered(r["property"]) not in {lowered(d) for d in drops}]
            for name in drops:
                # a requirement survives a drop when the line never speaks of it, or when an answer gives its value under another property
                again = {lowered(a.get("value")) for a in heard.get("adds") or [] if isinstance(a, dict) and lowered(a.get("property")) != lowered(name)}
                kept = [r for r in self.requirements if lowered(r["property"]) != lowered(name) or not (replaced_by_reader(r) or (self.mentions(line, r) and gives_up(r)))
                        or (answering and lowered(r["value"]) in again)]
                changed = changed or len(kept) != len(self.requirements)
                replaced += len(self.requirements) - len(kept)
                withdrawn += [r for r in self.requirements if r not in kept]
                self.requirements = kept
            added = [lowered(a.get("value")) for a in heard.get("adds") or [] if isinstance(a, dict) and str(a.get("value") or "").strip()
                     and not NOT_A_PROPERTY.search(lowered(f"{a.get('property')} {a.get('value')}"))]
            if heard.get("drops") and added and not withdrawn:
                # the drop used a name no requirement has: remove the one whose value the line states and does not keep
                said = stems(line)
                gone = [r for r in self.requirements if lowered(r["value"]) not in added and stems(r["value"]) - {"a", "an", "the"}
                        and stems(r["value"]) - {"a", "an", "the"} <= said and self.says_value(line, r["value"])
                        and not re.search(r"\bstill\b[^.!?]*" + re.escape(lowered(r["value"])), lowered(line))]
                if len(gone) == 1:
                    self.requirements = [r for r in self.requirements if r is not gone[0]]
                    withdrawn += gone
                    replaced += 1
                    changed = True
            if withdrawn and not added:
                # an amount was withdrawn and nothing replaced it: a different amount in the same unit is the new one
                for old in withdrawn:
                    was = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)", lowered(old["value"]).strip())
                    if not was:
                        continue
                    others = [m for m in re.finditer(r"(?<![a-z0-9.,])(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", lowered(line))
                              if self.same_unit(self.unit_word(m.group(2)), self.unit_word(was.group(2))) and float(m.group(1)) != float(was.group(1))]
                    if len({float(m.group(1)) for m in others}) == 1:
                        heard["adds"] = list(heard.get("adds") or []) + [{"property": old["property"], "value": f"{others[0].group(1)} {was.group(2)}"}]
                        self.note(f"the line's own new amount for {old['property']!r}: {others[0].group(1)} {was.group(2)}")
            for add in heard.get("adds") or []:
                if not isinstance(add, dict) or not str(add.get("value") or "").strip():
                    continue
                new = self.requirement(add)
                if NOT_A_PROPERTY.search(lowered(f"{new['property']} {new['value']}")):
                    continue
                if not self.says_value(line, new["value"]):
                    continue
                new["property"] = self.shopper_property(new, line)
                if not re.search(r"\d", new["value"]) and any(self.within(new["value"], purpose) for purpose in self.purposes):
                    continue            # words of a purpose describe a use, not a requirement
                stands_now = next((now for now, before in self.took_back.items() if lowered(before) == lowered(new["value"])), None)
                if stands_now is not None and not self.says_value(line, stands_now):
                    continue            # a line repeating a value the shopper already swapped out
                same = next((r for r in self.requirements if lowered(r["property"]) == lowered(new["property"])), None)
                if same is not None and lowered(same["value"]) == lowered(new["value"]):
                    continue
                twin = next((r for r in self.requirements if lowered(r["value"]) == lowered(new["value"])), None)
                said = stems(line) | self.name_words(line)
                if same is None and twin is not None and not (self.name_words(new["property"]) <= said and not self.named_in(self.name_words(twin["property"]), said)):
                    continue            # same value under another property name is not new, unless the line names only this property
                if same is not None and answering and not self.mentions(line, same):
                    # an answer mentioning neither the property nor its value adds a separate requirement
                    new["property"], same = "requirement", None
                if (same is not None and answering and same.get("stated") and not re.search(CHANGE_WORDS, lowered(line))
                        and (self.within(same["value"], self.item) or self.within(same["value"], self.category or ""))):
                    # the old reading was the reader's (usually the item name taken for a detail): the answer replaces it
                    self.requirements.remove(same)
                    self.opening = [r for r in self.opening if lowered(r["value"]) != lowered(same["value"])]
                    same = None
                if same is not None and not (gives_up(same) or replaced_by_reader(same)):
                    same = None
                if same is not None:
                    self.requirements.remove(same)
                    replaced += 1           # a new value for a known property is a change, not an answer
                    withdrawn.append(same)
                else:
                    given += 1
                # spellings of a withdrawn value are not alternates of its replacement
                old = {closed_up(w) for r in withdrawn for w in [r["value"]] + list(r.get("written") or []) + list(r.get("alike") or [])}
                for kind in ("written", "alike"):
                    if isinstance(new.get(kind), list):
                        new[kind] = [w for w in new[kind] if closed_up(str(w)) not in old]
                if answering and self.pair_open and self.is_wish(new["value"]):
                    new["chosen"] = True
                instead_of = [r for r in withdrawn if lowered(r["property"]) == lowered(new["property"])] or (withdrawn if len(withdrawn) == 1 else [])
                unit_before = re.fullmatch(r"\d+(?:\.\d+)?\s*([a-z][a-z ]*)", lowered(instead_of[0]["value"]).strip()) if instead_of else None
                if unit_before and re.fullmatch(r"\d+(?:\.\d+)?", new["value"].strip()):
                    # a bare number replacing a figure with a unit counts the same thing
                    new["value"] = f"{new['value'].strip()} {unit_before.group(1).strip()}"
                    new["filed"] = False
                if instead_of and lowered(instead_of[0]["value"]) != lowered(new["value"]):
                    self.took_back[lowered(new["value"])] = instead_of[0]["value"]
                self.requirements.append(new)
                changed = True
                if "brand" in lowered(new["property"]):
                    self.queue_filters()
                    self.add_queries([f"{new['value']} {self.item}", new["value"]], first=True)
            if sorted((lowered(p), lowered(v)) for p, v in before) == sorted((lowered(r["property"]), lowered(r["value"])) for r in self.requirements):
                # the line only repeated known requirements: undo the bookkeeping
                self.requirements, changed, given, replaced = held, changed_so_far, 0, 0
            else:
                self.fresh[line] = [r["value"] for r in self.requirements if r not in held]
            budget = money(heard.get("budget"))
            if budget is not None and not (self.says_amount(line, budget) or self.percent_of_limit(line, budget)):
                budget = None
            if budget is not None and any(abs(budget - float((self.counter.stock.get(k) or {}).get("price") or -1.0)) < 0.005 for k in self.counter.cart):
                budget = None               # that is a carted item's price, not a new limit
            if budget is not None and budget != self.budget:
                lowered_limit = self.budget is None or budget < self.budget
                self.budget, changed = budget, True
                self.fresh_limit.add(line)
                if lowered_limit:
                    self.queue_filters()
            if self.grounded_priority(heard.get("priority"), line):
                self.priority, changed = self.grounded_priority(heard.get("priority"), line), True
                self.figures, self.figure_unit, self.figure_quantity = {}, "", ""
            moved = [(r["property"], r["value"]) for r in self.requirements] != before or self.budget != limit_before
            wants_reply = heard.get("asks") is True and heard.get("about_change") is not True and not self.owed_reaction
            # while a question is open, a line that alters known values is not its answer, unless it only supplies the held-back requirement
            told_back = answering and given > 0 and not re.search(CHANGE_WORDS, lowered(line))
            spoke_of_else = self.budget != limit_before or (not told_back and (replaced > 0 or heard.get("about_change") is True))
            if replaced > 0 or self.budget != limit_before or (moved and not answering):
                self.moved_on = True
            was_answer = answering and not spoke_of_else
            if (wants_reply or (moved and not was_answer)) and line not in self.unanswered:
                # a changed want or a question gets a plain reply before the order
                self.unanswered.append(line)
            if answering:
                self.answered(line, given, spoke_of_else=spoke_of_else, about=" ".join(word_list(lowered(heard.get("about")))), nothing=nothing)
            if changed:
                self.changed_at = self.counter.turn
                # (a figure and its unit are one word closed up and two words apart: queued searches carry either)
                gone_words = [set(word_list(spelling)) for r in withdrawn if r not in self.requirements for spelling in (closed_up(r["value"]), r["value"])]
                kept_words = {w for v in self.values() for w in word_list(v)} | {w for r in self.requirements for w in word_list(r["value"])}
                gone_words = [words - kept_words for words in gone_words if words - kept_words]
                if gone_words:
                    # queued searches built on withdrawn values are dropped
                    self.queries = [q for q in self.queries if not any(words <= set(q.split()) for words in gone_words)]
                if given or replaced:
                    # new values: redo the looks and hunts that used the old ones
                    self.fresh_rounds = 2
                    self.hunt_stale = True
                    self.glossary.late_rounds = 0
                    self.glossary.flushes = 0
                    self.kin_swept.clear()
                    self.shelf_words_used = 0
                    self.sibling_done, self.kin_done, self.probed = set(), set(), set()
                    self.told_now = list(self.fresh.get(line) or [])
                    if replaced or not answering:
                        self.widened = False    # a changed value's other spellings (another unit, a word form) get searched too
                self.add_queries(self.value_queries() + list(heard.get("queries") or []), first=True)
                self.search_room = max(self.search_room, len(self.sent) + 16)
                self.search_asks = 0
                self.open_room = max(self.open_room, len(self.counter.pages) + 2 * self.counter.group_size * OPEN_BATCH)
                self.figure_room = max(self.figure_room, len(self.figures) + 3 * JUDGE_CARDS)
                self.judge_room = max(self.judge_room, self.judged_count + 36)
            if changed and self.counter.cart and given and not replaced and not withdrawn:
                # a change that only adds a requirement leaves the carted line's other evidence standing
                self.added_after_cart |= {lowered(str(a.get("property") or "")) for a in heard.get("adds") or [] if isinstance(a, dict)}
            self.note(f"heard {line[:80]!r} -> adds={heard.get('adds')} drops={heard.get('drops')} budget={heard.get('budget')} asks={heard.get('asks')}"
                      + (f" about={heard.get('about')!r}" if heard.get("about") else ""))
            if heard.get("asks") and not heard.get("about_change"):
                if self.owed_reaction:
                    self.owed_reaction = False
                else:
                    asks = True
                    self.pending_line = line
            elif heard.get("about_change"):
                self.owed_reaction = False
            if heard.get("about_change") is True and self.reply_sent and self.reply_sent[0] == self.counter.turn and self.replies_again < 2:
                # the market change was raised in the same turn as our reply, so the reply is queued again
                self.replies_again += 1
                for said_line in self.reply_sent[1].split("\n"):
                    text = said_line[2:] if said_line.startswith("- ") else said_line
                    if text.strip() and text not in self.unanswered:
                        self.unanswered.append(text)
                self.reply_sent = None
        return changed, asks

    @staticmethod
    def named_in(names: set[str], said: set[str]) -> bool:
        """True when each content word of the name occurs, in some form, among the words said."""
        names = {name for name in names if name not in ("as", "at", "by", "for", "in", "on", "to", "with", "the", "a", "an", "its")}
        return bool(names) and all(any(same_word(name, word) for word in said) for name in names)

    @staticmethod
    def names_itself(add: dict) -> bool:
        """True when the reading used the property's name, or its leading words, as the value."""
        own, said = [singular(w) for w in word_list(add.get("property"))], [singular(w) for w in word_list(add.get("value"))]
        return (bool(said) and not NOT_A_PROPERTY.search(lowered(f"{add.get('property')} {add.get('value')}")) and set(said) <= set(own)
                and (said == own or own[-len(said):] != said))

    def own_value(self, heard: dict, line: str) -> dict:
        """Swap a value that only repeats the property name for the phrase the reading's searches add."""
        adds = [a for a in heard.get("adds") or [] if isinstance(a, dict)]
        mended = []
        for a in adds:
            # also when the line puts the value before the property name and the reading reused the name's words for both
            if self.names_itself(a) or f" of the {' '.join(word_list(a.get('value')))} {' '.join(word_list(a.get('property')))} " in f" {' '.join(word_list(line))} ":
                known = (self.name_words(self.item) | self.name_words(a.get("property")) | {w for v in self.values() for w in self.name_words(v)}
                         | {w for b in adds if b is not a for w in self.name_words(b.get("value"))})
                found = {" ".join(w for w in word_list(q) if singular(w) not in known) for q in heard.get("queries") or []} - {""}
                found = {f for f in found if self.says_value(line, f)}
                if len(found) == 1:
                    a = dict(a, value=found.pop())
            mended.append(a)
        return dict(heard, adds=mended) if adds else heard

    def restates_budget(self, line: str) -> bool:
        """True when the line only repeats the known limit to the cent and mentions no requirement."""
        if self.budget is None or f"{self.budget:.2f}" not in line.replace(",", ""):
            return False
        figures = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", line.replace(",", ""))]
        if any(abs(figure - self.budget) > 0.005 for figure in figures):
            return False
        return not any(self.mentions(line, r) for r in self.requirements + self.set_aside)

    def answered(self, line: str, given: int, spoke_of_else: bool, about: str = "", nothing: bool = False) -> None:
        """Record what the last question produced: a requirement, nothing, or an unrelated line."""
        question, name, other = self.last_question, self.current_ask, self.current_other
        reason, self.current_reason = self.current_reason, None
        closing, self.closing_index = self.closing_index, None
        named = list(self.current_names) or [n for n in (name, other) if n]
        self.current_ask = self.current_other = ""
        self.current_names = []
        if line and line in self.answers:
            # a repeated earlier answer tells nothing new, so the remaining properties are still asked
            self.asked_log.append((question, "the same answer as before"))
            self.closing_dry = self.closing_dry or closing is not None
            if closing is not None and closing not in self.reasons_met and self.needs_found:
                self.reasons_met.add(closing)
                self.needs_expected = max(self.needs_found, self.needs_expected - 1)
            return
        unheard = spoke_of_else and bool(question) and (name or question) not in self.reasked
        if closing is not None and not unheard and not (given and not spoke_of_else):
            self.closing_dry = True
        if given and not spoke_of_else:
            self.answers.append(line)
            said = self.name_words(line)
            told = self.requirements[-given:]
            kinds = {self.kind(r) for r in told}
            units = {self.unit_word(m.group(2)) for r in told for form in self.forms(r) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}

            def about_this(n: str) -> bool:
                words = self.name_words(n)
                if words <= said or any(words <= self.name_words(r["property"]) or self.name_words(r["property"]) <= words for r in self.requirements):
                    return True
                if ("colour" in kinds and re.search(r"colou?r", n)) or ("material" in kinds and "material" in n):
                    return True
                return bool(self.measure_units.get(n, set()) & units) or " ".join(word_list(n)) == about

            group = named if len(named) >= 2 else []
            unsaid = [n for n in group if not about_this(n)]
            apart = bool(group) and len(unsaid) == len(group)
            for n in unsaid:
                # the answer covered none of the named properties: they are asked again separately
                self.asked = [p for p in self.asked if p != n]
                self.again.append(n)
            met = self.purpose_of(line)
            choice = self.pair_open and any(r.get("chosen") for r in told)
            if self.pair_open and not choice and not self.pair_again and not self.pair_chosen:
                self.pair_again = True
            if met is not None and not choice:
                self.purposes_met.add(met)
            self.asked_log.append((question, f"the shopper said: {line[:160]}"))
            if choice:
                # the pair is settled, but more properties may still be held back
                self.pair_chosen, self.pair_at = True, self.property_asks
            else:
                self.needs_found += 1
                self.first_found_at = self.first_found_at or self.questions_asked
                if reason is not None:
                    reason = self.owner_of(line, reason)
                    self.reasons_met.add(reason)
                    # a reason sharing at least half its words with the line is the same purpose split in two
                    said_here = self.content_words(line)
                    for other in range(len(self.reason_lists)):
                        words = self.content_words(self.closing_words(other))
                        if other not in self.reasons_met and words and 2 * len(words & said_here) >= len(words):
                            self.reasons_met.add(other)
                            self.needs_expected = max(self.needs_found, self.needs_expected - 1)
                    # the reader's property-to-reason assignment is a guess: interleave the lists of the reasons still open
                    for other in range(len(self.reason_lists)):
                        if other not in self.reasons_met:
                            mine = [n for n in self.reason_lists[other] if not self.asked_already(n)]
                            theirs = [n for n in self.reason_lists[reason] if not self.asked_already(n)]
                            merged = mine[:1] + [n for pair in zip(mine[1:], theirs) for n in pair] + mine[1 + len(theirs):] + theirs[len(mine) - 1:]
                            self.reason_lists[other] = list(dict.fromkeys(merged))
            # properties named beside the answered one go back into the queue behind the next two
            rest = [n for n in self.to_ask if n not in self.again]
            if apart:
                half = (len(self.again) + 1) // 2
                self.to_ask = rest[:1] + self.again[:half] + rest[1:2] + self.again[half:] + rest[2:]
            else:
                self.to_ask = rest[:2] + self.again + rest[2:]
            self.again = []
            return
        if spoke_of_else and question and (name or question) not in self.reasked:
            # the shopper moved to another subject, so the question is asked once more
            self.reasked.add(name or question)
            if name:
                for again in reversed(named):
                    self.asked = [p for p in self.asked if p != again]
                    self.to_ask.insert(0, again)
            if reason is not None:
                if name:
                    self.reason_asks[reason] -= 1
                else:
                    if closing is not None:
                        self.reason_asked.discard(closing)
                    if question == PLAIN_QUESTION:
                        self.plain_asked = False
                self.reason_turn = reason
            return
        if line and nothing and len(named) == 2 and question not in self.reasked and not self.restates_budget(line):
            # an empty answer to a two-property question: each property is asked again, alone
            self.reasked.add(question)
            self.asked_log.append((question, "not answered"))
            self.asked = [p for p in self.asked if p not in named]
            self.to_ask = named + [n for n in self.to_ask if n not in named]
            return
        self.asked_log.append((question, "no requirement"))
        if (question == PLAIN_QUESTION and reason is not None and reason < len(self.reason_plain) and self.reason_plain[reason] and not spoke_of_else
                and (not line.strip() or self.restates_budget(line))):
            self.plain_dry |= self.same_purpose(reason)
        if len(named) >= 3:
            self.asked_wide += named
        elif self.narrow and named:
            self.narrow += 1

    def affordable(self, key: tuple[str, str]) -> bool:
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        if seen.get("in_stock") is False or key in self.gone:
            return False
        return self.budget is None or not isinstance(price, float) or price <= self.budget + 1e-9

    def just_over(self, key: tuple[str, str], reach: float = OVER_REACH) -> bool:
        """In stock and slightly above the budget: the closest buy when nothing within budget fits."""
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        if self.budget is None or not isinstance(price, float) or seen.get("in_stock") is False or key in self.gone:
            return False
        return self.budget + 1e-9 < price <= self.budget * reach

    def only_over(self) -> list[tuple[str, str]]:
        """Listings that meet everything but the budget, by a small margin: most trusted, then cheapest."""
        if self.moved_on:
            # once the shopper moved the limit or a value, a price near-miss is no fallback
            return []
        any_shelved = any(self.in_category(p) for p in self.counter.pages.values())
        out = []
        for key in self.verdicts:
            page = self.counter.pages.get(key[0]) or {}
            if not self.just_over(key, NEAREST_REACH) or not self.fits_words(key) or (any_shelved and not self.in_category(page)):
                continue
            if self.brand and not set(word_list(self.brand)) <= set(word_list(page.get("brand"))):
                continue
            out.append(key)
        return sorted(out, key=lambda k: (-self.trust(k), (self.counter.stock.get(k) or {}).get("price") or 0.0))

    def named_shelf(self) -> str:
        """The store's own shelf label: the longest one read on listings whose words sit in the request."""
        pages = self.counter.pages
        if self.shelf_cache[0] == len(pages):
            return self.shelf_cache[1]
        request = f" {label_key(self.counter.request)} "
        said = f" {label_key(self.shelf_said)} " if self.shelf_said else ""
        seen: dict[str, int] = {}
        told: dict[str, int] = {}
        for page in pages.values():
            path = [str(label) for label in page.get("category_path") or []]
            if path and label_key(path[-1]) and f" {label_key(path[-1])} " in request:
                seen[path[-1]] = seen.get(path[-1], 0) + 1
            if path and said and label_key(path[-1]) and f" {label_key(path[-1])} " in said:
                told[path[-1]] = told.get(path[-1], 0) + 1
        own = {label_key(self.category), label_key(self.item)} - {""}
        best = max(seen, key=lambda label: (len(label_key(label).split()), label_key(label) in own, seen[label]), default="")
        if told:
            # a shelf the shopper named in the store's wording wins, preferring the exact spelling over its singular
            spelled = [label for label in told if f" {' '.join(word_list(label))} " in f" {' '.join(word_list(self.shelf_said))} "]
            best = max(spelled or told, key=lambda label: (len(label_key(label).split()), told[label]))
        self.shelf_split = len({tuple(word_list(str(page["category_path"][-1]))) for page in pages.values() if page.get("category_path")
                                and label_key(page["category_path"][-1]) == label_key(best)}) > 1 if best else False
        self.shelf_cache = (len(pages), best)
        return best

    def in_category(self, page: dict) -> bool:
        named = self.named_shelf()
        if named and self.shelf_split:
            # a singular twin shelf exists, so only the exactly named shelf counts
            return any(word_list(label) == word_list(named) for label in page.get("category_path") or [])
        # ('&' is kept as a word: a parent shelf 'Dolls & Accessories' is not the shelf 'Doll Accessories')
        wanted = {shelf_key(named)} if named else {shelf_key(self.category), shelf_key(self.item)} - {""}
        labels = {shelf_key(label) for label in page.get("category_path") or []}
        return bool(wanted & labels)

    def variant_of(self, key: tuple[str, str]) -> dict:
        page = self.counter.pages.get(key[0]) or {}
        return next((v for v in page.get("variants") or [] if str(v.get("sku")) == key[1]), {})

    @staticmethod
    def name_words(text: Any) -> set[str]:
        words = {singular(w) for w in word_list(str(text).replace("_", " "))} - {"of"}
        return {"color" if w == "colour" else w for w in words}

    @staticmethod
    def amount_spellings(requirement: dict) -> list[str]:
        """Alternate spellings of the value; an amount with a unit keeps only those of the same amount."""
        spellings = list(requirement.get("written") or [])
        if re.search(r"\d\s*[a-z]", lowered(requirement["value"])):      # a bare figure cannot stand for an amount with a unit
            spellings = [form for form in spellings if not re.fullmatch(r"[\d.,\s]+", str(form).strip())
                         and Errand.same_amount(requirement["value"], form)]
        return spellings

    @staticmethod
    def forms(requirement: dict, loose: bool = False) -> set[str]:
        """Ways the required value may be written; loose adds the words sellers use as synonyms."""
        spellings = Errand.amount_spellings(requirement)
        if requirement.get("filed"):
            written = spellings + list(requirement.get("alike") or []) if loose else [requirement["value"]]
        elif re.search(r"\d", requirement["value"]):
            written = list(requirement.get("alike") or []) if loose else [requirement["value"]] + spellings
        else:                               # close spellings count as the word itself; the rest are seller synonyms
            close = [form for form in spellings if SequenceMatcher(None, lowered(form), lowered(requirement["value"])).ratio() >= 0.75]
            written = [form for form in spellings if form not in close] + list(requirement.get("alike") or []) if loose else [requirement["value"]] + close
        # a hyphenated word-plus-suffix value is also matched by its first part
        head = re.fullmatch(r"([a-z]{3,})-[a-z]{3,}", lowered(requirement["value"]).strip())
        return {" ".join(singular(w) for w in word_list(closed_up(form))) for form in written + ([head.group(1)] if head else [])} - {""}

    @staticmethod
    def same_amount(value: str, form: Any) -> bool:
        """True when every figure with a unit in the spelling matches the value's amount, after conversion."""
        def figures(text: Any) -> list[tuple[float, str]]:
            return [(float(m.group(1)), Errand.unit_word(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", lowered(text))]
        wanted, found = figures(value), figures(form)
        if not wanted:
            return True
        for number, name in found:
            same = any(number == n and Errand.same_unit(name, u) for n, u in wanted)
            turned = any(abs(n / GLOSSARY.listing_units(GLOSSARY.unit_named(u)).get(GLOSSARY.unit_named(name), -1.0) - number) < 1e-6
                         for n, u in wanted)
            if not same and not turned:
                return False
        return True

    def forms_under(self, requirement: dict, label: str) -> set[str]:
        """Forms of the value for one fact label, also without the words the label itself says."""
        named = self.name_words(label) | {"at", "the", "of", "a", "an", "in", "on", "for", "with", "its"}
        out = set(self.forms(requirement))
        if out & self.filed_under(label):
            return out          # the shelf files the whole value here, so a shorter entry would be a different value
        for form in list(out):
            words = form.split()
            kept = [w for w in words if not any(same_word(w, n) for n in named)]
            if kept and len(kept) < len(words):
                out.add(" ".join(kept))
        return out

    def filed_under(self, label: str) -> set[str]:
        pages = self.counter.pages
        cached = self.filed_cache.get(label)
        if cached and cached[0] == len(pages):
            return cached[1]
        found = {entry for page in pages.values() for entry in self.entries((page.get("facts") or {}).get(label))}
        self.filed_cache[label] = (len(pages), found)
        return found

    def states(self, text: Any, requirement: dict, label: str = "") -> bool:
        """True when one entry of a fact, which may list several, is exactly the required value."""
        entries = {" ".join(singular(w) for w in word_list(reading(part))) for part in re.split(r"[,;/|]+", lowered(text)) for reading in (lowered, closed_up)}
        if not label:
            return bool(entries & self.forms(requirement))
        named = self.name_words(label)
        for entry in list(entries):
            kept = [w for w in entry.split() if not any(same_word(w, n) for n in named)]
            if kept:
                entries.add(" ".join(kept))
        return bool(entries & self.forms_under(requirement, label))

    def kind(self, requirement: dict) -> str:
        """Where a store states a property: amount, colour (variant options), material (name), else fact."""
        if re.search(r"\d", requirement["value"]):
            return "amount"
        name = lowered(requirement["property"])
        if re.search(r"colou?r", name) and GLOSSARY.may_be_colour(requirement["value"]):
            return "colour"     # only listed colours count as colour; other values are facts
        if "material" in name:
            # a part's material is a separate fact when the shelf keeps labels for it
            return "fact" if self.part_labels(requirement) else "material"
        return "fact"

    def shopper_property(self, requirement: dict, line: str) -> str:
        """The property name as the shopper wrote it, else a shelf fact label that holds the value."""
        named = word_list(requirement["property"])
        if re.search(r"\d", requirement["value"]):
            return requirement["property"]
        wrote = stems(line) | self.name_words(line)
        if len(named) >= 2:
            kind, rest = singular(named[-1]), {singular(w) for w in named[:-1]}
            if kind in wrote and not rest & (wrote | stems(self.counter.request)) and not any(lowered(r["property"]) == kind for r in self.requirements):
                return kind
        if self.name_words(requirement["property"]) & wrote:
            return requirement["property"]
        # no overlap with the line: use the shelf fact label, made of words the line wrote, that holds the value
        filed: dict[str, int] = {}
        for page in self.counter.pages.values():
            for label, value in (page.get("facts") or {}).items():
                words = self.name_words(label)
                if words and words <= wrote and not SALE_LABELS.search(str(label)) and any(self.entry_says(e, requirement, str(label)) for e in self.entries(value)):
                    filed[str(label)] = filed.get(str(label), 0) + 1
        if not filed:
            return requirement["property"]
        return " ".join(word_list(max(filed, key=lambda label: (len(self.name_words(label)), filed[label])).replace("_", " ")))

    def part_labels(self, requirement: dict) -> set[str]:
        """Fact labels under which the shelf files the material of one named part, beyond plain material."""
        stamp = (requirement["property"], len(self.counter.pages))
        cached = self.part_cache.get(requirement["property"])
        if cached and cached[0] == stamp:
            return cached[1]
        wanted = self.name_words(requirement["property"])
        part = wanted - self.name_words("material type")
        found: set[str] = set()
        if part:
            for page in self.counter.pages.values():
                for label in page.get("facts") or {}:
                    have = self.name_words(label)
                    if have and have <= wanted and have & part:
                        found.add(str(label))
        self.part_cache[requirement["property"]] = (stamp, found)
        return found

    @staticmethod
    def entries(text: Any) -> list[str]:
        """Split a fact into its separate entries, as plain words."""
        parts = (" ".join(singular(w) for w in word_list(closed_up(part))) for part in re.split(r"[,;/|]+", lowered(text)))
        return [part for part in parts if part]

    def entry_says(self, entry: str, requirement: dict, label: str, loose: bool = False) -> bool:
        """True when an entry under the label holds every word of one form of the value."""
        forms = self.forms(requirement, loose=True) if loose else self.forms_under(requirement, label)
        have = entry.split()
        return any(all(any(same_word(n, h) for h in have) for n in form.split()) for form in forms)

    def entry_is(self, entry: str, requirement: dict, label: str) -> bool:
        """True when the entry equals a form of the value once the label's own words are dropped."""
        named = self.name_words(label)
        kept = " ".join(w for w in entry.split() if not any(same_word(w, n) for n in named)) or entry
        forms = self.forms_under(requirement, label)
        return entry in forms or kept in forms

    def own_labels(self, requirement: dict) -> tuple[set[str], set[str], set[str]]:
        """Returns (own labels, inferred labels, single-word entries under them) for the property."""
        pages = self.counter.pages
        stamp = (requirement["property"], requirement["value"], len(requirement.get("written") or []), len(pages), self.named_shelf())
        cached = self.own_cache.get(stamp)
        if cached is not None:
            return cached
        wanted = self.name_words(requirement["property"])
        kind = self.kind(requirement)
        shelf = [page for page in pages.values() if self.in_category(page)] or list(pages.values())
        written = " ".join(word_list(requirement["property"]))
        if kind == "amount" and wanted and not any(self.name_words(label) & wanted for page in shelf for label in page.get("facts") or {}):
            # the shopper tied the amount to a part ('a 40-watt motor'): the store files it under the measure its unit measures
            measure = next((GLOSSARY.unit_quantity.get(self.unit_spelled(m.group(2))) for form in self.forms(requirement)
                            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)
                            if GLOSSARY.unit_quantity.get(self.unit_spelled(m.group(2))) not in (None, "", "count", "none", "other")), "")
            if measure:
                wanted, written = self.name_words(measure), " ".join(word_list(measure))
        tiers: list[set[str]] = [set(), set(), set(), set(), set()]     # tiers, closest first: identical, same words, nested, one shared word
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        unit_names = units | {self.unit_spelled(unit) for unit in units}
        stated: dict[str, int] = {}
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                have = self.name_words(label)
                if wanted and have:
                    if " ".join(word_list(str(label).replace("_", " "))) == written:
                        tiers[0].add(str(label))
                    elif have == wanted:
                        tiers[1].add(str(label))
                    elif wanted <= have:
                        # extra word = the unit: still the property's own label; any other extra word may be another measure
                        tiers[2 if (have ^ wanted) <= units else 3].add(str(label))
                    elif have & unit_names and have - unit_names and have - unit_names <= wanted:
                        tiers[2].add(str(label))
                    elif kind == "amount" and have & wanted and (units & have or any(self.amounts(value, requirement))):
                        tiers[4].add(str(label))
                    elif kind == "fact" and have < wanted and singular(word_list(written)[0]) in have:
                        # the property named without its last word ('pets' for pet type): still the property's own label
                        tiers[3].add(str(label))
                entries = self.entries(value)
                if len(entries) == 1 and self.entry_is(entries[0], requirement, str(label)):
                    # for a material the entry itself must equal the value, not merely contain its word
                    if kind != "material" or entries[0] in self.forms_under(requirement, str(label)):
                        stated[str(label)] = stated.get(str(label), 0) + 1
        related = next((tier for tier in tiers if tier), set())
        if kind != "amount":
            # a property may sit under a general label and a department one, so every like-named label counts
            related = set().union(*tiers)
            if "material" in lowered(requirement["property"]) and self.part_labels(requirement):
                related = self.part_labels(requirement)
        most = max(stated.values(), default=0)
        inferred: set[str] = set()
        if kind != "colour" and not any(stated.get(label) for label in related):
            inferred = {label for label, count in stated.items() if count >= 2 and 3 * count >= most}
        if kind == "fact" and not related and not inferred:
            # last resort: the label whose entries hold the value on the most listings, skipping sale-term labels
            held: dict[str, int] = {}
            for page in shelf:
                for label, value in (page.get("facts") or {}).items():
                    if not SALE_LABELS.search(str(label)) and any(self.entry_says(entry, requirement, str(label)) for entry in self.entries(value)):
                        held[str(label)] = held.get(str(label), 0) + 1
            inferred = {label for label, count in held.items() if count >= 3 and 3 * count >= max(held.values())}
        words: set[str] = set()
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    words |= {entry for entry in self.entries(value) if " " not in entry}
            if kind == "colour":
                for variant in page.get("variants") or []:
                    words |= {entry for entry in self.entries(variant.get("options")) if re.fullmatch(r"[a-z]{3,}", entry)}
        answer = (related, inferred, words)
        # keyed per requirement so a changed property keeps both values cached; stale page counts are dropped
        self.own_cache = {key: value for key, value in self.own_cache.items() if key[3:] == stamp[3:]}
        self.own_cache[stamp] = answer
        return answer

    @staticmethod
    def colour_family(requirement: dict) -> str:
        return GLOSSARY.colour_of(requirement["value"])

    @staticmethod
    def colours_in(text: Any) -> list[tuple[str, ...]]:
        return GLOSSARY.colours_named_in(text)

    def store_colours(self, key: tuple[str, str]) -> tuple[list[tuple[str, ...]], int]:
        """Colours of the variant with a level: 3 when read from its options, 2 from colour facts."""
        found = self.colours_in(self.variant_of(key).get("options"))
        if found:
            return found, 3
        facts = ((self.counter.pages.get(key[0]) or {}).get("facts") or {}).items()
        return [c for label, value in facts if "colo" in str(label).lower() for c in self.colours_in(value)], 2

    def named_facts(self, page: dict, requirement: dict) -> list[tuple[str, str]]:
        related, inferred, _ = self.own_labels(requirement)
        return [(str(label), str(value)) for label, value in (page.get("facts") or {}).items() if str(label) in related or str(label) in inferred]

    def says(self, text: Any, requirement: dict, whole: bool = False, loose: bool = False) -> bool:
        """True when the text holds every word of some form of the value; whole bars hyphen-joined words."""
        if whole:
            have = {singular(w) for reading in (lowered(text), closed_up(text)) for w in re.split(r"[\s/,;()\[\]|+]+", reading) if w}
            return any(set(form.split()) <= have for form in self.forms(requirement, loose))
        have = {singular(w) for reading in (lowered(text), closed_up(text)) for w in word_list(reading)}
        return any(all(any(same_word(n, h) for h in have) for n in form.split()) for form in self.forms(requirement, loose))

    def amounts(self, text: Any, requirement: dict) -> tuple[set[tuple[float, str]], set[tuple[float, str]]]:
        """Returns (amounts equal to the value, other amounts) the text gives in the value's measure."""
        unit = self.unit_spelled
        wanted: set[tuple[float, str]] = set()
        for form in self.forms(requirement):
            wanted |= {(float(m.group(1)), unit(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        said = lowered(text)
        # standalone figure and unit, not fused to neighbours; a range gives its unit-bearing end; a quote mark is inches
        found = {(float(m.group(1)), unit(m.group(2))) for m in re.finditer(r"(?<![a-z0-9])(?<![0-9]\.)(\d+(?:\.\d+)?)\s*([a-z]+)(?![a-z0-9])", said)}
        found |= {(float(m.group(1)), GLOSSARY.unit_named("inch")) for m in re.finditer(r"(?<![a-z0-9])(?<![0-9]\.)(\d+(?:\.\d+)?)\s*(?:\"|''|\u201d|\u2033)", said)}
        # the same measure in another unit is the same amount after conversion; screen sizes read in inches only
        screen = bool(self.name_words(requirement["property"]) & {"display", "screen"})
        for number, name in list(wanted):
            scale = GLOSSARY.listing_units(name)
            if scale and not (screen and name == GLOSSARY.unit_named("inch")):
                wanted |= {(round(number / scale[seen], 6), seen) for seen in {u for _, u in found} if seen in scale}
        piece = GLOSSARY.unit_named("piece")
        if piece in {name for _, name in wanted} and not (requirement.get("stated") or lowered(requirement["value"]) in self.took_back):
            found = set(found)      # (a pack or a set of N is not N pieces)
        units = {name for _, name in wanted}
        found = {pair for pair in found if pair[1] in units}
        same = {pair for pair in found if any(abs(pair[0] - number) < 1e-6 and pair[1] == name for number, name in wanted)}
        return same, found - same

    @staticmethod
    def other_words(tokens: set[str]) -> set[str]:
        """Plain alphabetic words among option tokens, leaving out codes and size letters."""
        return {w for w in tokens if re.fullmatch(r"[a-z]{3,}", w) and not re.fullmatch(r"x*[sml]|x+l", w)}

    def unit_in_label(self, label: str, requirement: dict) -> bool:
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        return bool(units & self.name_words(label))

    def named_otherwise(self, options: Any, requirement: dict) -> bool:
        """True when the options give the property by name with a figure other than the required one."""
        named = [w for w in self.name_words(requirement["property"]) if len(w) >= 5]
        for part in re.split(r"[/,;|]+", lowered(options)):
            if not re.search(r"\d", part) or self.has_amount(part, requirement):
                continue
            if any(SequenceMatcher(None, word, name).ratio() >= 0.8 for word in re.findall(r"[a-z]{4,}", part) for name in named):
                return True
        return False

    def has_amount(self, text: Any, requirement: dict) -> bool:
        """True when the text states the required amount; a bare number borrows the property's unit."""
        same, others = self.amounts(text, requirement)
        if same or (not others and self.says(text, requirement)):
            return True
        bare = re.fullmatch(r"\d+(?:\.\d+)?", requirement["value"].strip())
        if not bare:
            return False
        named = [w for w in word_list(requirement["property"]) if len(w) >= 3]
        return any(float(m.group(1)) == float(bare.group(0)) and any(self.same_unit(m.group(2), w) for w in named)
                   for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]+)", lowered(text)))

    @staticmethod
    def same_unit(one: str, other: str) -> bool:
        """True when one unit spelling abbreviates the other, or both are the same."""
        unit = Errand.unit_word
        a, b = sorted((unit(lowered(one)), unit(lowered(other))), key=len)
        if not a or a[0] != b[0] or len(b) <= 3 or (len(a) == 1 and len(b) > 6):
            return bool(a) and a == b
        rest = iter(b)
        return all(letter in rest for letter in a)

    @staticmethod
    def scaled(unit: str, other: str, any_step: bool = False) -> float | None:
        """Factor between two units of one measure (1000x apart, or any metric step), else None."""
        prefixes = {"milli": 1e-3, "centi": 1e-2, "deci": 1e-1, "kilo": 1e3, "m": 1e-3, "c": 1e-2, "d": 1e-1, "k": 1e3}

        def readings(name: str) -> list[tuple[float, str]]:
            return [(1.0, name)] + [(size, name[len(p):]) for p, size in prefixes.items() if name.startswith(p) and len(name) > len(p)]
        for size, base in readings(unit):
            for other_size, other_base in readings(other):
                if size != other_size and (base == other_base or (len(base) > 1 and len(other_base) > 1 and Errand.same_unit(base, other_base))):
                    ratio = size / other_size
                    if any_step or abs(ratio - 1000.0) < 1e-6 or abs(ratio - 0.001) < 1e-9:
                        return ratio
        return None

    def plain(self, key: tuple[str, str], requirement: dict) -> int:
        """Level 3..0 of how plainly a listing states the value; -1 when it states something else."""
        self.learn_words()
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        title = page.get("title")
        kind = self.kind(requirement)
        related, inferred, words = self.own_labels(requirement)
        own = [(label, self.entries(value)) for label, value in self.named_facts(page, requirement)]
        singles = [(label, entries[0]) for label, entries in own if len(entries) == 1]
        lists = [(label, entry) for label, entries in own if len(entries) > 1 for entry in entries]
        stated = any(self.entry_says(entry, requirement, label) for label, entry in singles)
        listed = any(self.entry_says(entry, requirement, label) for label, entry in lists)
        value_words = {w for form in self.forms(requirement) for w in form.split()}
        elsewhere = any(self.says(value, requirement) for value in (page.get("facts") or {}).values())

        if kind == "amount":
            loose_tag = (key, requirement["property"], requirement["value"])
            self.loose_among.discard(loose_tag)
            if self.named_otherwise(options, requirement):
                return -1
            number = re.search(r"\d+(?:\.\d+)?", requirement["value"])
            asked = bool(requirement.get("stated")) or lowered(requirement["value"]) in self.took_back
            unit = self.unit_spelled(re.sub(r"[\d.\s-]+", "", lowered(requirement["value"])))
            # a fact whose label names the amount's unit also counts as own and may give a bare figure in that unit
            own = [(label, value) for label, value in self.named_facts(page, requirement)
                   if label in related or self.unit_in_label(label, requirement)]
            # the own label's twin that adds a unit of the measure ('capacity_l' beside 'capacity') is the property's own as well
            own += [(str(label), str(value)) for label, value in (page.get("facts") or {}).items()
                    if str(label) not in {x for x, _ in own} and any(self.name_words(label) > self.name_words(r) for r in related)
                    and all(self.unit_spelled(w) in GLOSSARY.listing_units(self.unit_spelled(re.sub(r"[\d.\s-]+", "", lowered(requirement["value"]))))
                            for r in related for w in self.name_words(label) - self.name_words(r))]
            bounded = False
            kept_parts = []
            for label, value in (page.get("facts") or {}).items():
                if re.search(r"brand|descri", str(label)):
                    continue
                if str(label) not in related | inferred:
                    # parts that only bound the amount (an upper or lower limit) do not state it and are dropped
                    parts = [part for part in re.split(r"[;|]+|,\s", str(value)) if part.strip()]
                    kept = [part for part in parts if not BOUNDED.search(lowered(part))]
                    if len(kept) < len(parts):
                        if self.amounts("; ".join(part for part in parts if part not in kept), requirement)[0]:
                            bounded = True
                        value = "; ".join(kept)
                kept_parts.append(str(value))
            rest = "; ".join(kept_parts)

            deferred: list[int] = []

            def read(text: str, level: int) -> int | None:
                same, others = self.amounts(text, requirement)
                if same and not others:
                    if re.search(r"(?<![a-z0-9.])\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*[a-z]", lowered(text)):
                        deferred.append(level)      # a range holds only if no later place contradicts it
                        return 0
                    return level
                if others and not same:
                    return -1
                return 0 if same else None      # 0: value among several figures, next place decides; None: no figure of this measure

            def read_own(items: list | None = None) -> int | None:
                items = own if items is None else items
                found = read("; ".join(value for _, value in items), 3)
                scales = GLOSSARY.listing_units(unit)

                facts_here = {str(k): str(v).strip() for k, v in (page.get("facts") or {}).items()}

                def scale_of(label: str) -> float:
                    # a bare figure under a label naming another unit of the measure ('capacity_l: 0.5') is read in that unit,
                    # and so is the same figure under the label's unit-less twin ('capacity: 0.5' beside 'capacity_l: 0.5')
                    named = [scales[u] for u in (self.unit_spelled(w) for w in self.name_words(label)) if u in scales]
                    if not named:
                        twins = [scales[u] for other, value in facts_here.items() if other != label and value == facts_here.get(label)
                                 and self.name_words(other) > self.name_words(label)
                                 for u in (self.unit_spelled(w) for w in self.name_words(other) - self.name_words(label)) if u in scales]
                        named = twins
                    return named[0] if len(set(named)) == 1 else 1.0
                figures = {round(float(m.group(1)) * scale_of(label), 6) for label, value in items for m in [re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", value)] if m}
                if found is None and figures and number:
                    want = float(number.group(0))
                    found = 3 if all(abs(f - want) <= 1e-6 * max(1.0, want) for f in figures) else -1 if not any(abs(f - want) <= 1e-6 * max(1.0, want) for f in figures) else 0
                return found

            # places are tried in an order that depends on the measure; the first definite answer decides
            if GLOSSARY.weighs(unit):
                own = [(label, value) for label, value in own if unit in {self.unit_spelled(w) for w in self.name_words(label)}]
            places = {"options": lambda: read(options, 3), "title": lambda: read(title, 2), "own": read_own, "facts": lambda: read(rest, 2),
                      "named": lambda: read_own([(label, value) for label, value in own if unit in {self.unit_spelled(w) for w in self.name_words(label)}])}
            if unit == GLOSSARY.unit_named("piece"):
                # a count is the figure written with its piece unit in the variant or the title; a bare figure under a set or
                # pieces label is not one
                order = ("options", "title")
            elif unit == GLOSSARY.unit_named("volt"):
                order = ("own", "options", "title", "facts") if asked else ("options", "own", "title", "facts")
            elif GLOSSARY.weighs(unit) and asked or unit == GLOSSARY.unit_named("piece") and not asked:
                order = ("options", "title", "own", "facts")
            elif unit == GLOSSARY.unit_named("inch") and self.name_words(requirement["property"]) & {"display", "screen"}:
                order = ("options", "own", "title", "facts")
            else:
                # a bare figure under a label that names the unit (power_w = 1400) is read last
                order = ("options", "title", "facts", "named")
            several, loose = False, False
            for place in order:
                if place == "facts" and "own" not in order and own:
                    # the property's own labelled fact stating exactly this amount outweighs other facts' figures
                    mine = read_own()
                    if mine and mine > 0:
                        return mine
                ranges = len(deferred)
                found = places[place]()
                if found:
                    return found
                if found == 0:
                    # a figure among several in the title or the loose facts states nothing; in the options or a range it still does
                    if place in ("options", "own") or len(deferred) > ranges:
                        several = True
                    else:
                        loose = True
            if bounded and not (several or listed):
                self.loose_among.add(loose_tag)
                return 1
            if loose and not (several or listed):
                self.loose_among.add(loose_tag)
                return 1
            return max(deferred, default=1) if several or listed or loose else 0

        named_as = self.named_as.get((key, requirement["property"], requirement["value"]))
        if kind == "colour" and self.colour_family(requirement):
            # every colour read for the variant (options, else colour facts) must be the asked colour
            found, level = self.store_colours(key)
            if found:
                return level if all(self.colour_family(requirement) in colours for colours in found) else -1
            if GLOSSARY.names_a_colour(options):
                return 0
            if not options.strip() and not any("colo" in str(label).lower() for label in page.get("facts") or {}):
                return 0
            # no colour word of the known list: judge by words like any other value
        if kind == "colour":
            wanted = self.forms(requirement)
            parts = [{singular(w) for reading in (lowered(part), closed_up(part)) for w in re.split(r"[\s()\[\]+]+", reading) if w}
                     for part in re.split(r"[/,;|]+", options)]
            # a hyphen between a colour and a code with a digit separates the colour from the seller's code
            parts = [part | {singular(piece) for w in part if "-" in w and re.search(r"\d", w) for piece in w.split("-") if piece} for part in parts]
            named = [part for part in parts if any(set(form.split()) <= part for form in wanted)]
            if named:
                # a part that also carries another known colour word names a different colour
                alone = any(not ((part & words) - value_words) and not self.other_words(part - value_words) for part in named)
                if all((part & words) - value_words for part in named):
                    return 0
                return 3 if alone and named_as not in ("among", "other") else 1
            # splitting on any other sign finds colours fused to text, which reads less plainly
            joined = [{singular(w) for reading in (lowered(part), closed_up(part)) for w in re.split(r"[^a-z0-9.\-]+", reading) if w}
                      for part in re.split(r"[/,;|]+", options)]
            if any(set(form.split()) <= part for form in wanted for part in joined):
                return 1
            if any(form.replace(" ", "") in lowered(options).replace(" ", "") for form in wanted):
                return -1           # fused inside a longer token: a code or another name, not this colour
            if stated:
                return 1 if options.strip() else 3
            before = self.took_back.get(lowered(requirement["value"]))
            if before and any(set(form.split()) <= part for form in self.forms(dict(requirement, value=before, written=[], alike=[])) for part in parts):
                return -1           # the variant shows the colour this requirement replaced
            alike = any(set(form.split()) <= part for form in self.forms(requirement, loose=True) for part in parts)
            if alike or named_as in ("only", "among"):
                return 1
            if options.strip():
                return -1
            return 1 if listed else 0

        if kind == "material":
            before = None if requirement.get("withdrawn") else self.took_back.get(lowered(requirement["value"]))
            if before and lowered(before) != lowered(requirement["value"]) and self.plain(key, dict(requirement, value=before, written=[], alike=[], withdrawn=True)) >= 2:
                return -1           # the listing still reads as the material this requirement replaced
            # materials match by family across material facts, title and options
            kin = GLOSSARY.material_words(requirement["value"])
            # only a fact named for the material states it; one filed under another name (a fabric, a type) does not
            own = [(str(label), str(value)) for label, value in (page.get("facts") or {}).items() if "mater" in lowered(label)]
            singles = [(label, self.entries(value)[0]) for label, value in own if len(self.entries(value)) == 1]
            lists = [(label, entry) for label, value in own if len(self.entries(value)) > 1 for entry in self.entries(value)]
            elsewhere = False
            stated = any(self.entry_says(entry, requirement, label) for label, entry in singles) or any(kin & set(entry.split()) for label, entry in singles)
            listed = any(self.entry_says(entry, requirement, label) for label, entry in lists) or any(kin & set(entry.split()) for label, entry in lists)
            # the own fact must be exactly the material; near spellings and blends are other entries
            exactly = any((self.entry_is(entry, requirement, label) or set(entry.split()) <= kin) and not re.search(r"[+&]", str(raw))
                          for (label, entry), raw in zip(singles, [value for label, value in own if len(self.entries(value)) == 1]))
            named = {singular(w) for w in word_list(f"{title} {options}")}
            # only a material fact settles which of two materials a title names
            filed = any(("mater" in label or "fabric" in label) and (self.entry_says(entry, requirement, label) or kin & set(entry.split()))
                        for label, entry in singles + lists)
            two = bool(kin & named) and any(other & named for other in GLOSSARY.other_material_words(kin))
            # the reader saw the value beside another material, with no material fact or options to decide
            two = two or (named_as == "among" and not own and not (kin & {singular(w) for w in word_list(options)}) and not self.says(options, requirement))
            if stated and exactly and (filed or not two):
                return 3
            if two and not filed and own:
                return 1                # (with no material fact at all, a title naming the material beside another one still states it)
            if stated or listed or self.says(title, requirement) or self.says(options, requirement) or kin & named:
                return 2
            if named_as in ("among", "other"):
                # the reader sees the title and variant only: an 'among' needs a word of the value in the wording or the facts
                nameable = any(self.says(text, requirement) or self.says(text, requirement, loose=True) for text in (title, options, *(page.get("facts") or {}).values()))
                return (1 if nameable else 0) if named_as == "among" else -1
            contrary = bool(singles)
            # the reader's 'only' counts only when a seller's word for the material stands in the variant or the facts (it saw the title and variant, and marks a text with no such word 'only' too)
            if named_as == "only" and any(self.says(text, requirement, loose=True) for text in (options, *(page.get("facts") or {}).values())):
                return 1
            if self.says(title, requirement, loose=True):
                return 1
            if contrary:
                return -1
            return 1 if elsewhere else 0

        # a generic property: the fact must be the value itself, not merely contain its words
        if stated:
            return 3 if any(self.entry_is(entry, requirement, label) for label, entry in singles) else 1
        # a seller's synonym never states a listing's own fact value; spellings and shortened forms made of the value's own words still do
        spelled = dict(requirement, alike=[a for a in requirement.get("alike") or [] if {singular(w) for w in word_list(closed_up(a))} <= value_words])
        if listed or any(self.entry_says(entry, spelled, label, loose=True) for label, entry in singles):
            return 1
        if own:
            if self.filed_alike(page, requirement):
                return 1
            # an own fact that only names a synonym or a broader class counts when another fact is exactly the value
            if self.filed_elsewhere(page, requirement) and any(self.entry_says(entry, requirement, label, loose=True) for label, entry in singles):
                return 1
            if any(self.entry_says(entry, requirement, label, loose=True) for label, entry in singles):
                return 0            # the own fact holds a seller's synonym for the value ('type c' for usb c): neither met nor contrary
            if any(self.fact_same.get((requirement["property"], requirement["value"], lowered(entry))) == "same" for label, entry in singles):
                return 0            # the reader says the own fact's entry is the value in other words ('free form' for irregular)
            return -1
        if self.filed_elsewhere(page, requirement):
            return 1
        if related:
            return -1               # the shelf files this property as a fact but this listing lacks it
        if inferred:
            return 0
        return 1 if elsewhere or self.says(title, requirement) else 0

    def exact(self, key: tuple[str, str], requirement: dict) -> int:
        """Net score: own facts and options that are exactly the value add, ones stating another subtract."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        score = 0
        for label, value in self.named_facts(page, requirement):
            entries = self.entries(value)
            if len(entries) == 1:
                bare = self.kind(requirement) == "amount" and self.unit_in_label(label, requirement) and self.bare_figure(entries[0], requirement)
                score += 1 if bare or self.entry_is(entries[0], requirement, label) else (0 if self.entry_says(entries[0], requirement, label) else -1)
        return score + int(self.states(options, requirement))

    @staticmethod
    def bare_figure(entry: str, requirement: dict) -> bool:
        number = re.search(r"\d+(?:\.\d+)?", requirement["value"])
        return bool(number) and bool(re.fullmatch(r"\d+(?:\.\d+)?", entry.strip())) and float(entry) == float(number.group(0))

    def filed_elsewhere(self, page: dict, requirement: dict) -> bool:
        """True when a fact outside the property's own labels has the value as its only entry."""
        mine = {label for label, _ in self.named_facts(page, requirement)}
        for label, value in (page.get("facts") or {}).items():
            entries = self.entries(value)
            if str(label) not in mine and len(entries) == 1 and self.entry_is(entries[0], requirement, str(label)):
                return True
        return False

    def filed_alike(self, page: dict, requirement: dict) -> bool:
        """True when a fact sharing a name word (other than "type") has the value as its only entry."""
        mine = {label for label, _ in self.named_facts(page, requirement)}
        wanted = self.name_words(requirement["property"]) - {"type"}
        for label, value in (page.get("facts") or {}).items():
            entries = self.entries(value)
            if (str(label) not in mine and wanted & self.name_words(label) and len(entries) == 1
                    and self.entry_is(entries[0], requirement, str(label))):
                return True
        return False

    def label_nearness(self, label: str, requirement: dict) -> tuple[int, int, int]:
        """Closeness of a label to the property name, as a tuple that compares higher when nearer."""
        wanted, have = self.name_words(requirement["property"]), self.name_words(label)
        shelf = self.name_words(self.named_shelf() or self.category or self.item)
        other = have - wanted
        departments = bool(other) and other <= shelf
        return (len(wanted & have) - (0 if departments else len(other)), int(departments), len(wanted & have))

    def nearest_label(self, key: tuple[str, str], requirement: dict) -> int:
        """1 when the nearest-named fact is the value, -1 when it differs, 0 on a tie or no evidence."""
        page = self.counter.pages.get(key[0]) or {}
        mine = {label for label, _ in self.named_facts(page, requirement)}
        wanted = self.name_words(requirement["property"])
        agree: tuple[int, int, int] | None = None
        differ: tuple[int, int, int] | None = None
        for label, value in (page.get("facts") or {}).items():
            label, entries = str(label), self.entries(value)
            if not entries:
                continue
            is_value = len(entries) == 1 and self.entry_is(entries[0], requirement, label)
            if not is_value and label not in mine and len(wanted & self.name_words(label)) < 2:
                continue
            near = self.label_nearness(label, requirement)
            if not is_value and len(entries) == 1 and self.entry_says(entries[0], requirement, label):
                is_value = True
            if is_value:
                agree = near if agree is None else max(agree, near)
            else:
                differ = near if differ is None else max(differ, near)
        if agree is None:
            return -1 if differ is not None else 0
        if differ is None:
            return 1
        return (agree > differ) - (differ > agree)

    def gainsaid(self, key: tuple[str, str], requirement: dict) -> bool:
        """True when an own fact has a single entry that is not the value."""
        page = self.counter.pages.get(key[0]) or {}
        for label, value in self.named_facts(page, requirement):
            entries = self.entries(value)
            if len(entries) == 1 and not self.entry_says(entries[0], requirement, label):
                return True
        return False

    def level_needed(self, requirement: dict) -> int:
        """Minimum plain() level that counts as stated for this kind of property."""
        return 2 if self.kind(requirement) in ("amount", "material") else 3

    def settled(self, key: tuple[str, str], requirement: dict) -> bool:
        """True when the page states the value at any level."""
        return self.plain(key, requirement) >= 1

    def row_score(self, key: tuple[str, str]) -> float:
        row = self.counter.rows.get(key) or {}
        title = stems(row.get("title"))
        score = 4.0 if key in self.counter.shelf else 0.0
        for requirement in self.requirements:
            need = stems(requirement["value"])
            others = [stems(form) for form in list(requirement.get("written") or []) + list(requirement.get("alike") or [])]
            if need and need <= title or any(other and other <= title for other in others):
                score += 3
            elif need & title:
                score += 1
        score += sum(1 for w in {singular(w) for w in self.item_words()} if w in title)
        if lowered(row.get("title")) in self.kin_titles() or lowered(row.get("title")) in self.probed:
            score += 20
        elif any(query in self.sibling_sent and place < 5 for query, place in self.counter.found_by.get(key) or []):
            score += 20             # top hit of a search built from a seen listing's title
        elif "-" in key[1] and key[1].rsplit("-", 1)[0] in {k[1].rsplit("-", 1)[0] for k in self.verdicts if "-" in k[1] and self.fits_words(k)}:
            score += 20
        return score + (50 if self.listed and key in self.listed else 0)

    def search_place(self, key: tuple[str, str], spelled: list[set[str]]) -> int:
        """Best rank of the row in searches carrying every required value in some spelling; 10 if none."""
        best = 10
        for query, place in self.counter.found_by.get(key) or []:
            said = {singular(w) for w in word_list(closed_up(query))}
            if place < best and all(any(set(form.split()) <= said for form in forms) for forms in spelled):
                best = place
        return best

    def spellings(self, requirement: dict) -> set[str]:
        found = self.forms(requirement) | self.forms(requirement, loose=True)
        return (found | {" ".join(singular(w) for w in word_list(closed_up(w))) for w in self.store_spellings(requirement)}) - {""}

    def to_open(self, limit: int, allowance: int | None = None, since: int = 0, worth: float = 0.0) -> list[dict]:
        allowance = self.open_room if allowance is None else allowance
        unopened = [k for k in self.counter.row_order if k[0] not in self.counter.pages and self.counter.shown_at.get(k, 0) >= since
                    and (self.affordable(k) or self.just_over(k, OPEN_REACH)) and (worth <= 0 or self.row_score(k) >= worth)]
        if self.listed:
            unopened = [k for k in self.listed if k[0] not in self.counter.pages] + [k for k in unopened if k in self.listed]
        # rows first shown after the latest change go first: later searches carry the current values
        fresh = (lambda k: self.counter.shown_at.get(k, 0) >= self.changed_at) if self.changed_at else (lambda k: False)
        # a row just over the limit loses 2 points: it can be named but not bought
        spelled = [self.spellings(r) for r in self.requirements]
        # a top hit of a search carrying every required value is opened early even when its title shows none of them: the
        # store matched the values in its facts
        top = (lambda k: self.search_place(k, spelled) <= 2) if spelled else (lambda k: False)
        ranked = sorted(dict.fromkeys(unopened), key=lambda k: (not fresh(k), not top(k), -(self.row_score(k) - (0 if self.affordable(k) else 2)), self.search_place(k, spelled)))
        ids = list(dict.fromkeys(k[0] for k in ranked))[: max(0, min(limit, allowance - len(self.counter.pages)))]
        if self.counter.has("compare"):
            return [act("compare", product_ids=ids[i:i + OPEN_BATCH]) for i in range(0, len(ids), OPEN_BATCH)]
        return [act("view", product_id=i) for i in ids]

    def candidates(self) -> list[tuple[str, str]]:
        out = []
        any_shelved = any(self.in_category(p) for p in self.counter.pages.values())
        for product_id, page in self.counter.pages.items():
            if any_shelved and not self.in_category(page):
                continue
            for variant in page.get("variants") or []:
                key = (product_id, str(variant.get("sku")))
                if (self.affordable(key) or self.just_over(key)) and (not self.listed or key in self.listed):
                    out.append(key)
        return out

    def card(self, number: int, key: tuple[str, str]) -> str:
        page = self.counter.pages[key[0]]
        named_words = {w for r in self.requirements for w in self.name_words(r["property"])}
        value_words = {w for r in self.requirements for w in stems(r["value"])}
        close, other = [], []
        for label, value in (page.get("facts") or {}).items():
            if self.name_words(label) & named_words or stems(value) & value_words:
                close.append(f"{label} = {str(value)[:110]}")
            else:
                other.append(f"{label} = {str(value)[:36]}")
        facts = "; ".join(close + other[:5])
        for position, requirement in enumerate(self.requirements, start=1):
            named = self.named_facts(page, requirement)
            if named:
                facts += f" || the fact named for requirement {position}: " + "; ".join(f"{label} = {value[:110]}" for label, value in named)
        price = (self.counter.stock.get(key) or {}).get("price")
        return (f"[{number}] title: {str(page.get('title') or '')[:150]} | options: {self.variant_of(key).get('options') or '-'} | price: {price} | "
                f"category: {' > '.join(str(x) for x in (page.get('category_path') or [])[-2:])} | brand: {page.get('brand') or '-'} | facts: {facts or '-'}")

    def judge(self) -> None:
        """Gives each unjudged candidate a verdict from the page's own words."""
        shelf_words = set(label_key(self.named_shelf()).split())

        def the_shelf_itself(r: dict) -> bool:
            return bool(r.get("stated") and shelf_words and set(label_key(r["value"]).split()) <= shelf_words)
        self.requirements = [r for r in self.requirements if not the_shelf_itself(r)]
        self.opening = [r for r in self.opening if not the_shelf_itself(r)]
        self.name_check()
        self.fact_check()
        stamp = json.dumps([self.requirements, self.priority, self.lenient, len(self.named_as), len(self.fact_same)], sort_keys=True)
        if stamp != self.judged_under:
            self.verdicts, self.judged_under = {}, stamp
        for key in self.candidates():
            if key not in self.verdicts:
                in_shelf = self.in_category(self.counter.pages[key[0]])
                met = [self.settled(key, r) or (self.lenient and self.plain(key, r) >= 1) for r in self.requirements]
                self.verdicts[key] = {"kind_ok": in_shelf, "met": met, "by": "words"}

    def judged_marks(self, answer: dict, text: str) -> tuple[list[str], list[dict]]:
        """Turns the reader's answer into mark strings and proofs; a quote must occur on the card."""
        cards = {int(m.group(1)): m.group(0) for m in re.finditer(r"^\[(\d+)\] .*$", text, re.M)}
        if not isinstance(answer.get("listings"), list):
            old = answer.get("marks")
            return (re.split(r"[\n;,]+", old) if isinstance(old, str) else (old or [])), [p for p in answer.get("proof") or [] if isinstance(p, dict)]
        marks, proofs = [], []
        for item in answer["listings"]:
            if not isinstance(item, dict) or not isinstance(item.get("n"), int) or item["n"] not in cards:
                continue
            card = cards[item["n"]]
            seen = " ".join(word_list(card))
            letters, where, said = "", [], []
            for check in item.get("checks") or []:
                quoted = str(check.get("words") or "") if isinstance(check, dict) else ""
                if " = " in quoted:
                    quoted = quoted.split(" = ", 1)[1].split(";")[0]
                words = " ".join(word_list(quoted))
                met = bool(words) and isinstance(check, dict) and check.get("met") is True and f" {words} " in f" {seen} "
                letters += "Y" if met else "N"
                said.append(quoted.strip())
                where.append(self.place_of(card, words) if met else "")
            marks.append(f"{item['n']} {str(item.get('kind') or 'K').strip().upper()[:1] or 'K'} {letters}")
            if letters and set(letters) == {"Y"}:
                proofs.append({"n": item["n"], "where": where, "words": said, "rank_value": item.get("rank_value")})
        return marks, proofs

    @staticmethod
    def place_of(card: str, words: str) -> str:
        """Name of the card part (title, options or fact label) that contains the words."""
        parts = {}
        for part in card.split(" | "):
            label, _, value = part.partition(": ")
            parts[label.split("] ")[-1]] = value
        for label in ("title", "options"):
            if f" {words} " in f" {' '.join(word_list(parts.get(label, '')))} ":
                return label
        for fact in parts.get("facts", "").split(" || ")[0].split("; "):
            label, _, value = fact.partition(" = ")
            if value and f" {words} " in f" {' '.join(word_list(value))} ":
                return label.strip()
        return ""

    def ask_reader(self) -> None:
        """Sends the nearest misses to the reader, who may spot a synonym or another unit."""
        open_keys = [k for k in self.verdicts if self.affordable(k) and self.verdicts[k].get("by") == "words" and self.verdicts[k].get("kind_ok")
                     and not self.fits_words(k) and not any(self.plain(k, r) == -1 for r in self.requirements)]
        open_keys.sort(key=lambda k: -sum(self.plain(k, r) for r in self.requirements))
        batch = open_keys[:JUDGE_CARDS]
        if not batch or self.judged_count >= self.judge_room:
            return
        self.judged_count += len(batch)
        wanted = "\n".join(f"{i}. {r['property']}: {r['value']}" for i, r in enumerate(self.requirements, start=1)) or "(none beyond the kind of thing)"
        text = (f"The shopper wants: {self.item}\nRequirements, in order:\n{wanted}\n"
                + (f"Priority between fitting items: {self.priority}\n" if self.priority else "")
                + "\nListings:\n" + "\n".join(self.card(i, key) for i, key in enumerate(batch, start=1)))
        answer = self.reader.fill(JUDGE_GUIDE, text, JUDGE_TOOL, 2600) or {}
        marks, proofs = self.judged_marks(answer, text)
        for mark in marks:
            found = re.match(r"\s*\[?(\d+)\]?\W+([KkXx])\W*([YyNn\s,]*)", str(mark))
            if not found or not 1 <= int(found.group(1)) <= len(batch):
                continue
            key = batch[int(found.group(1)) - 1]
            said = [x in "Yy" for x in re.findall(r"[YyNn]", found.group(3))]
            said += [False] * (len(self.requirements) - len(said))
            # the reader's yes covers wording differences only, never an amount or a colour
            met = [self.settled(key, r) or (said[i] and self.plain(key, r) != -1 and not re.search(r"\d", r["value"]) and self.kind(r) != "colour")
                   for i, r in enumerate(self.requirements)]
            self.verdicts[key] = {"kind_ok": found.group(2) in "Kk", "met": met, "by": "reader"}
        for key in batch:
            if self.verdicts[key].get("by") == "words":
                self.verdicts[key]["by"] = "reader"      # each listing goes to the reader once
        for proof in proofs:
            if isinstance(proof, dict) and isinstance(proof.get("n"), int) and 1 <= proof["n"] <= len(batch):
                self.verdicts[batch[proof["n"] - 1]].update({k: proof.get(k) for k in ("where", "words", "rank_value")})
        self.note(f"reader judged {len(batch)}: " + " | ".join(str(m)[:12] for m in marks)[:260])
        self.add_queries(answer.get("queries") or [])

    def fits_words(self, key: tuple[str, str]) -> bool:
        verdict = self.verdicts.get(key) or {}
        met = verdict.get("met") or []
        return bool(verdict.get("kind_ok")) and len(met) >= len(self.requirements) and all(met[: len(self.requirements)])

    def fits(self, key: tuple[str, str]) -> bool:
        if not self.fits_words(key) or not self.affordable(key):
            return False
        page = self.counter.pages.get(key[0]) or {}
        shelved = self.in_category(page) or not any(self.in_category(p) for p in self.counter.pages.values())
        brand = not self.brand or set(word_list(self.brand)) <= set(word_list(page.get("brand")))
        return shelved and brand

    def ranked(self) -> list[tuple[str, str]]:
        """Fitting listings, best first: by plainness and price, or by expected value of the priority."""
        if self.better in ("higher", "lower"):
            higher = self.better == "higher"
        else:
            # default direction: more is better unless the priority's words say less
            higher = not re.search(r"\b(low|least|small|short|few|min|light)", lowered(self.priority))
        by_price = self.priced_priority()
        by_figure = bool(self.priority) and not by_price
        fitting = [k for k in self.verdicts if self.fits(k)]

        def number(key: tuple[str, str]) -> float | None:
            value = self.figure(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        self.unnamed_ok = False
        figures = {k: number(k) for k in fitting} if by_figure else {}
        if figures and all(v is None for v in figures.values()):
            # no listing has a figure where a store counts one, so figures from any fact are allowed
            self.unnamed_ok = True
            figures = {k: number(k) for k in fitting}
        trust = {k: self.trust(k) for k in fitting}
        fixed = self.fixed_figure() if by_figure else None
        if fixed is not None and any(v is not None and abs(v - fixed) < 1e-9 for v in figures.values()):
            # a fixed amount that some listing meets demotes listings with a different amount
            for key in fitting:
                if figures[key] is not None and abs(figures[key] - fixed) >= 1e-9:
                    trust[key] = round(trust[key] * FAINT_FIT, 4)

        # a figure standing alone is trusted fully; one inside a longer measurement or among several is not
        weight = {k: (1.0 if self.figure_stated(k, figures[k]) else UNSURE_FIT) for k in fitting if figures.get(k) is not None}
        named: dict[str, tuple[float, float]] = {}
        shared: dict[str, tuple] = {}
        for k in weight:
            title = lowered((self.counter.pages.get(k[0]) or {}).get("title")) or k[0]
            value = figures[k] if higher else -figures[k]
            if title not in named or (value, trust[k] * weight[k]) > named[title]:
                named[title] = (value, trust[k] * weight[k])
                shared[title] = self.doubt(k, trust[k])
        rivals = sorted(named.items(), key=lambda item: -item[1][0])
        # plainly stated listings sort ahead of weakly stated ones with bigger figures
        sure = {k for k in fitting if trust[k] >= SURE_FIT - 1e-9}

        def standing(key: tuple[str, str]) -> float:
            """Expected share of the best figure this listing brings by the priority, if it fits."""
            own = lowered((self.counter.pages.get(key[0]) or {}).get("title")) or key[0]
            mine = None if figures[key] is None else (figures[key] if higher else -figures[key])
            expected, left = 0.0, 1.0
            mine_doubt = self.doubt(key, trust[key])
            counted: set = set()
            for title, (value, chance) in rivals:
                if title == own or (mine is not None and value <= mine):
                    continue
                if shared.get(title) is not None:
                    # listings unsure for the same reason stand or fall together, so only the best counts as a rival
                    if shared[title] in counted:
                        continue
                    counted.add(shared[title])
                    if shared[title] == mine_doubt:
                        chance = 1.0
                if mine is not None and chance > 0:
                    share = (mine / value) if higher else (value / mine)
                    expected += left * chance * max(0.0, min(1.0, share))
                left *= 1.0 - chance
            if mine is None:
                return left
            # if its figure is not its own after all, it ranks like a listing with no figure
            none = 1.0
            counted = set()
            for title, (_, chance) in rivals:
                if title != own:
                    if shared.get(title) is not None:
                        if shared[title] in counted:
                            continue
                        counted.add(shared[title])
                    none *= 1.0 - chance
            return (expected + left) * weight[key] + (1.0 - weight[key]) * none

        def order(key: tuple[str, str]) -> tuple:
            price = (self.counter.stock.get(key) or {}).get("price") or 0.0
            plainly = sum(self.plain(key, r) for r in self.requirements) + sum(self.exact(key, r) for r in self.requirements)
            twin = lowered((self.counter.pages.get(key[0]) or {}).get("title")) in self.lost_titles
            rest = (0 if self.firm(key) else 1, 0 if self.solid(key) else 1, 0 if self.verdicts[key].get("by") == "words" else 1, -plainly,
                    0 if twin else 1, price)
            wishes = -self.pair_met(key)
            if by_price:
                return (wishes, -trust[key], price) + rest
            if not by_figure:
                return (wishes, -trust[key]) + rest
            value = figures[key]
            worth = trust[key] * standing(key)
            # equal worth: the surer fit first, then the listing whose options state the figure
            # (a figure glued to a quote mark or an escape, 11.6\u201chp, is not stated in the options the way a figure with its unit is)
            in_options = value is not None and any(abs(float(n) - value) < 1e-9 for n in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)(?!\d|\.\d|\s?[\"\u201d\u201c\u2033'\u2019\\])", str(self.variant_of(key).get("options") or "")))
            # a sure fit with no figure leads only when no rival states one
            ahead = key in sure and (value is not None or not rivals)
            # (then a figure the title writes with its unit: the store reads a figure written out more surely than a fact's)
            titled = value is not None and any(abs(float(n) - value) < 1e-9 for n in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s?[a-z]", lowered((self.counter.pages.get(key[0]) or {}).get("title"))))
            return (wishes, 0 if ahead else 1, -round(worth, 6), 0 if self.worded(key) else 1, 1 if value is None else 0, -trust[key],
                    -((value if higher else -value) if value is not None else 0.0), 0 if in_options else 1, 0 if in_options or titled else 1, -self.spelled_as_asked(key)) + rest

        return sorted(fitting, key=order)

    def pair_met(self, key: tuple[str, str]) -> int:
        """While the doubted pair is unanswered, how many of its two wishes the listing writes: the shopper wants both if one
        listing can have them, so a listing holding both goes first."""
        if not self.pair_wishes or self.pair_chosen:
            return 0
        page = self.counter.pages.get(key[0]) or {}
        text = set(word_list(" ".join([str(page.get("title") or ""), str(self.variant_of(key).get("options") or "")]
                                      + [str(v) for v in (page.get("facts") or {}).values()])))
        have = {singular(w) for w in text}
        plain = self.content_words(self.item) | self.content_words(self.category) | {"type", "brand", "kind", "material", "colour", "color", "style"}
        met = 0
        for wish in self.pair_wishes:
            words = {singular(w) for w in word_list(wish) if len(w) >= 3 and w not in ("the", "its", "and", "for", "with", "one")} - {singular(w) for w in plain}
            if words and words <= have:
                met += 1
        return met

    def spelled_as_asked(self, key: tuple[str, str]) -> int:
        """How many required amounts the listing's title or options write in the shopper's own spelling ('1kg' or '1 kg', not
        '1kilo'): the store reads an amount in its usual unit spelling more surely than in another word for the unit."""
        page = self.counter.pages.get(key[0]) or {}
        text = " ".join(word_list(f"{page.get('title') or ''} {self.variant_of(key).get('options') or ''}"))
        glued = f" {closed_up(text)} "
        count = 0
        for requirement in self.requirements:
            value = lowered(requirement["value"]).strip()
            if self.kind(requirement) == "amount" and value and (f" {' '.join(word_list(value))} " in f" {text} " or f" {closed_up(value).replace(' ', '')} " in glued):
                count += 1
        return count

    def worded(self, key: tuple[str, str]) -> bool:
        """True when every required material is written out as that very word in the listing's title, options or facts, or is named by its own material fact."""
        page = self.counter.pages.get(key[0]) or {}
        text = " ".join([str(page.get("title") or ""), str(self.variant_of(key).get("options") or "")] + [str(v) for v in (page.get("facts") or {}).values()])
        have = {singular(w) for w in word_list(lowered(text))}
        # (a lone material fact naming a kind of the material, abs for plastic, states it as plainly as the word itself; a list of materials or a title beside a contrary fact does not)
        own = {singular(w) for label, value in (page.get("facts") or {}).items() if "mater" in lowered(label) and len(self.entries(value)) == 1 for w in word_list(lowered(str(value)))}
        return all({singular(w) for w in word_list(lowered(r["value"]))} <= have or bool(own & GLOSSARY.material_words(r["value"]))
                   for r in self.requirements if self.kind(r) == "material")

    def doubt(self, key: tuple[str, str], trust: float) -> tuple | None:
        """Plainness per requirement plus rounded trust; None when the listing surely fits."""
        if trust >= SURE_FIT - 1e-9:
            return None
        return (tuple(self.plain(key, r) for r in self.requirements), round(trust, 3))

    def counted_unit(self) -> str:
        if re.search(r"\bpieces?\b", lowered(self.priority)) and not self.sized_priority():
            return GLOSSARY.unit_named("piece")
        return self.own_unit(self.unit_word((re.findall(r"[a-z]+", lowered(self.figure_unit)) or [""])[0]))

    def sized_priority(self) -> bool:
        """True when the unit reader's own term for the priority opens with a measured quantity (length, weight, volume, time), whatever nouns the priority holds."""
        return any(w in SIZED_QUANTITIES for w in word_list(self.figure_quantity)[:2])

    def own_unit(self, unit: str) -> str:
        """Swap a unit of the wrong measure for the one the shelf's fact labels end in, else the one its listings write most."""
        said = word_list(self.figure_quantity)
        quantity = said[0] if said and (len(said) == 1 or said[0] in {singular(w) for w in word_list(self.priority)}) else ""
        if not unit or quantity not in SIZED_QUANTITIES or GLOSSARY.unit_quantity.get(self.unit_spelled(unit)) == quantity:
            return unit
        stamp = (unit, quantity, len(self.counter.pages) // 25)
        if stamp not in self.unit_fixes:
            ends = Tally(self.unit_spelled(words[-1]) for page in self.counter.pages.values() for label in page.get("facts") or {}
                         for words in [word_list(str(label).replace("_", " "))] if words)
            own = [name for name, _ in ends.most_common() if GLOSSARY.unit_quantity.get(name) == quantity]
            if not own:
                # no fact label ends in one: the unit of the quantity that the options and titles write after a number most often
                written = Tally(self.unit_spelled(m.group(1)) for page in self.counter.pages.values()
                                for text in [page.get("title")] + [v.get("options") for v in page.get("variants") or []]
                                for m in re.finditer(r"(?<![a-z0-9.])\d+(?:\.\d+)?\s?([a-z]+)(?![a-z0-9])", lowered(text)))
                own = [name for name, _ in written.most_common() if GLOSSARY.unit_quantity.get(name) == quantity]
            self.unit_fixes[stamp] = own[0] if own else unit
        return self.unit_fixes[stamp]

    def bare_unread(self) -> bool:
        """True when the priority counts pieces or a length: the store reads a bare number under a fact label as no figure there."""
        unit = self.unit_spelled(self.counted_unit())
        return bool(unit) and (unit == GLOSSARY.unit_named("piece") or GLOSSARY.unit_quantity.get(unit) == "length")

    def counted_figure(self, key: tuple[str, str], unnamed: bool = False) -> tuple[bool, float | None]:
        """Return (settled, figure) as the store counts it: facts, then options, then title."""
        unit = self.unit_spelled(self.counted_unit())
        if not unit:
            return (False, None)
        stamp = (unit, key, self.priority, unnamed)
        if stamp in self.counted_cache:
            return self.counted_cache[stamp]
        page = self.counter.pages.get(key[0]) or {}
        measure = GLOSSARY.listing_units(unit)
        named = {self.unit_spelled(singular(w)) for w in word_list(self.priority) if len(w) >= 4}

        def written(text: Any) -> list[float]:
            # number after a space or one of - ( / , : * ; [ followed by a unit; a pack is one piece
            out: list[float] = []
            for m in re.finditer(r"(?<![^\s\-(/,:*;\[])(\d+(?:\.\d+)?)\)?\s*([a-z]+|\")(?![^\s)\-/.,(:*\[|;\uff08\uff09])", str(text or "").lower()):
                raw = m.group(2)
                word = GLOSSARY.unit_named("inch") if raw == '"' else unit if unit == GLOSSARY.unit_named("piece") and raw in ("pack", "packs") else self.unit_spelled(raw)
                if word == unit or (unit == GLOSSARY.unit_named("piece") and self.same_unit(word, unit)):      # (pcs and pieces are one unit, though a glossary may file them apart)
                    out.append(float(m.group(1)))
                elif word in measure:
                    out.append(round(float(m.group(1)) * measure[word], 6))
                elif (word.endswith(unit) or unit.endswith(word)) and self.scaled(word, unit, any_step=True):
                    out.append(round(float(m.group(1)) * (self.scaled(word, unit, any_step=True) or 0.0), 6))
            return out

        def own(label: str) -> bool:
            # label names the priority or ends in its unit; length units are excluded as widths use them too
            words = word_list(label.replace("_", " "))
            ends_in_unit = bool(words) and self.unit_spelled(words[-1]) in (measure or {unit: 1.0}) and GLOSSARY.unit_quantity.get(unit) != "length"
            return "descri" not in label and (bool(named & {self.unit_spelled(singular(w)) for w in words}) or ends_in_unit)

        places = []
        for label, fact in (page.get("facts") or {}).items():
            if own(str(label)):
                found = written(fact)
                words = word_list(str(label).replace("_", " "))
                bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(fact))
                if not found and bare and len(words[-1]) <= 4 and self.unit_spelled(words[-1]) == unit and not self.bare_unread():
                    found = [float(bare.group(1))]
                if found:
                    places.append(found)
                    break
        if unnamed and not places:
            for label, fact in (page.get("facts") or {}).items():
                found = written(fact) if "descri" not in str(label) else []
                if found and max(found) <= 1.05 * min(found):
                    places.append(found)
                    break
        places += [written(self.variant_of(key).get("options")), written(page.get("title"))]
        answer: tuple[bool, float | None] = (False, None)
        listed: list[float] = []
        for found in places:
            if not found:
                continue
            low, high = min(found), max(found)
            single = high <= 1.05 * low
            if not single and not listed:
                # a place listing several sizes cannot tell which one this variant is; the options may
                listed, answer = found, (True, None)
                continue
            if listed and single and any(abs(low - size) <= 0.05 * size for size in listed):
                answer = (True, low)
            elif not listed:
                answer = (True, low if single else None)
            break
        self.counted_cache[stamp] = answer
        return answer

    def figure_stated(self, key: tuple[str, str], value: float) -> bool:
        """True when the figure stands alone on the page with its unit; an unknown unit passes."""
        unit = self.counted_unit()
        if not unit:
            return True
        settled, counted = self.counted_figure(key, unnamed=self.unnamed_ok)
        if settled:
            return counted is not None and abs(counted - float(value)) < 1e-9

        def alike(found: str) -> bool:
            return self.same_unit(found, unit)

        def stands_in(text: Any) -> bool | None:
            """True if the value is the only figure in the unit in the text; None when there is no figure."""
            found = [(float(m.group(1)), m.group(2)) for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", lowered(text))]
            same = {number for number, word in found if alike(word)}
            if same:
                return same == {float(value)}
            other = [word for number, word in found if number == float(value) and self.scaled(self.unit_word(word), unit, any_step=True)]
            if other:
                return len({number for number, word in found if word == other[0]}) == 1
            return None
        options, title = self.variant_of(key).get("options"), (self.counter.pages.get(key[0]) or {}).get("title")
        for text in (options, title):
            stated = stands_in(text)
            if stated is not None:
                return stated
        # no standalone figure: it came from a fact, accepted only with its unit or under a label naming it
        for label, fact in ((self.counter.pages.get(key[0]) or {}).get("facts") or {}).items():
            if stands_in(fact) is True:
                return True
            bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(fact))
            if bare and float(bare.group(1)) == float(value) and any(self.same_unit(word, unit) for word in word_list(str(label).replace("_", " "))) and not self.bare_unread():
                return True
        return False

    def fixed_figure(self) -> float | None:
        """The amount a requirement fixes in the unit of the priority's figures, if any."""
        unit = self.unit_word((re.findall(r"[a-z]+", lowered(self.figure_unit)) or [""])[0])
        if not unit:
            return None
        for requirement in self.requirements:
            for form in self.forms(requirement):
                for found in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form):
                    if self.unit_word(found.group(2)) == unit:
                        return float(found.group(1))
        return None

    def trust(self, key: tuple[str, str]) -> float:
        worth = 1.0
        for requirement in self.requirements:
            worth *= self.reliance(key, requirement)
        return round(worth, 4)

    def reliance(self, key: tuple[str, str], requirement: dict) -> float:
        """How far one stated value can be relied on, given where the page states it."""
        level, kind = self.plain(key, requirement), self.kind(requirement)
        if level < 0:
            return AGAINST_FIT
        if kind == "fact":
            nearest = self.nearest_label(key, requirement)
            if level >= 3:
                # a long list of entries weakens any single value in it
                crowd = max((len(self.entries(value)) for _, value in self.named_facts(self.counter.pages.get(key[0]) or {}, requirement)), default=1)
                share = min(1.0, LIST_ENTRIES / max(1, crowd))
                # the property's own label has the last word on whether the listing is filed under the value
                if nearest < 0:
                    return FAINT_FIT
                if nearest > 0 and self.exact(key, requirement) > 0 and not self.gainsaid(key, requirement):
                    return round(FIRM_FIT * share, 4)
                return round((SURE_FIT if nearest > 0 else LIKELY_FIT) * share, 4)
            if level >= 1 and nearest < 0:
                # (a whole entry of a short list of the property's own fact: the seller filed the listing under the value among others)
                listed = [len(self.entries(value)) for label, value in self.named_facts(self.counter.pages.get(key[0]) or {}, requirement)
                          if 1 < len(self.entries(value)) <= LIST_ENTRIES and any(self.entry_is(entry, requirement, label) for entry in self.entries(value))]
                return round(UNSURE_FIT * min(1.0, LIST_ENTRIES / max(listed)), 4) if listed else FAINT_FIT
            if level >= 1:
                # the value is on file for the listing (long entry, or exact in another fact); a name-only value is not
                page = self.counter.pages.get(key[0]) or {}
                own = self.own_labels(requirement)[0]
                if own and not any(str(label) in own for label in page.get("facts") or {}):
                    return UNSURE_FIT
                within = any(len(self.entries(value)) == 1 and self.entry_says(self.entries(value)[0], requirement, label) and not self.copies_title(page, value)
                             for label, value in self.named_facts(page, requirement))
                if within and not self.filed_elsewhere(page, requirement) and not self.usual_entry(key, requirement):
                    return UNSURE_FIT
                if within or self.filed_elsewhere(page, requirement):
                    return SURE_FIT
            return FAINT_FIT
        if kind == "amount":
            if level == 1 and (key, requirement["property"], requirement["value"]) in self.loose_among:
                return FAINT_FIT
            return FIRM_FIT if level >= 2 else LIKELY_FIT if level == 1 else FAINT_FIT
        if kind == "colour" and self.colour_family(requirement) and self.store_colours(key)[0]:
            if level >= 3 and not self.named_as_asked(key, requirement):
                return SURE_FIT
            return FIRM_FIT if level >= 3 else SURE_FIT if level >= 1 else FAINT_FIT
        if kind == "colour":
            if level >= 3:
                own_word = dict(requirement, written=[], alike=[])
                if self.second_colour(key, requirement) or (self.fused_colour(key, requirement) and self.exact(key, requirement) <= 0):
                    return LIKELY_FIT
                return FIRM_FIT if self.states(self.variant_of(key).get("options"), own_word) or self.exact(key, requirement) > 0 else SURE_FIT
            own = self.says(self.variant_of(key).get("options"), requirement) or self.exact(key, requirement) > 0
            if level >= 1 and own and (self.second_colour(key, requirement) or self.fused_colour(key, requirement)) and self.exact(key, requirement) <= 0:
                return LIKELY_FIT       # two-colour variants are often filed under the other colour
            return SURE_FIT if level >= 1 and own else UNSURE_FIT if level >= 1 else FAINT_FIT
        if kind == "material":
            if level == 2 and lowered(requirement["value"]) in self.took_back and self.named_facts(self.counter.pages.get(key[0]) or {}, requirement) and not self.fact_holds(key, requirement):
                return FAINT_FIT
            if level == 2 and self.only_listed(key, requirement):
                return LIKELY_FIT
            page = self.counter.pages.get(key[0]) or {}
            if level == 2 and self.named_as.get((key, requirement["property"], requirement["value"])) == "other" and not (
                    self.fact_holds(key, requirement) or self.says(page.get("title"), requirement) or self.says(self.variant_of(key).get("options"), requirement)):
                return UNSURE_FIT       # a family word alone (an alloy, an abbreviation) against the reader's mark that the wording names another material
            if level >= 3 and not self.named_as_asked(key, requirement):
                return UNSURE_FIT       # a kind-of word without the material itself is weak
            return FIRM_FIT if level >= 3 else LIKELY_FIT if level >= 2 else UNSURE_FIT
        return FIRM_FIT if level >= 3 else SURE_FIT if level >= 1 else UNSURE_FIT

    @staticmethod
    def copies_title(page: dict, text: Any) -> bool:
        """True when a fact only repeats the listing's title (six or more of its words in a row): the seller stated nothing of its own."""
        fact, title = " ".join(word_list(text)), " ".join(word_list(page.get("title")))
        return len(fact.split()) >= 6 and (fact in title or title in fact)

    def fact_holds(self, key: tuple[str, str], requirement: dict) -> bool:
        page = self.counter.pages.get(key[0]) or {}
        kin = GLOSSARY.material_words(requirement["value"]) if self.kind(requirement) == "material" else set()
        return any(self.entry_says(entry, requirement, label) or ("mater" in label and bool(kin & set(entry.split())))
                   for label, value in self.named_facts(page, requirement) for entry in self.entries(value))

    def usual_entry(self, key: tuple[str, str], requirement: dict) -> bool:
        """True when the longer entry is the wording most listings on the shelf use for the value."""
        page = self.counter.pages.get(key[0]) or {}
        longer = [entries[0] for label, value in self.named_facts(page, requirement) for entries in [self.entries(value)]
                  if len(entries) == 1 and self.entry_says(entries[0], requirement, label) and not self.entry_is(entries[0], requirement, label)]
        if not longer:
            return True
        stamp = (requirement["property"], requirement["value"])
        cached = self.usual_cache.get(stamp)
        if not cached or cached[0] != len(self.counter.pages):
            related, inferred, _ = self.own_labels(requirement)
            carried: dict[str, int] = {}
            alone = 0
            shelf = [p for p in self.counter.pages.values() if self.in_category(p)] or list(self.counter.pages.values())
            for other in shelf:
                kept = {(str(label), entry) for label, value in (other.get("facts") or {}).items()
                        if str(label) in related or str(label) in inferred for entry in self.entries(value)}
                alone += any(self.entry_is(entry, requirement, label) for label, entry in kept)
                for entry in {entry for label, entry in kept if not self.entry_is(entry, requirement, label) and self.entry_says(entry, requirement, label)}:
                    carried[entry] = carried.get(entry, 0) + 1
            cached = self.usual_cache[stamp] = (len(self.counter.pages), alone, carried)
        _, alone, carried = cached
        # usual means no other wording of the value is carried by more listings than this one
        return any(carried.get(entry, 0) >= max([alone, 1] + list(carried.values())) for entry in longer)

    def named_as_asked(self, key: tuple[str, str], requirement: dict) -> bool:
        """True when the page names the value itself, not only a related word for it."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        wanted = set(word_list(lowered(requirement["value"])))
        if self.kind(requirement) == "colour":
            own = {w for w in wanted if GLOSSARY.colours_of_word(w)} or wanted
            return bool(set(word_list(lowered(options))) & (own | {self.colour_family(requirement)}))
        facts = [str(value) for label, value in (page.get("facts") or {}).items() if re.search(r"mater|fabric", str(label), re.I)]
        said = set(word_list(lowered(" ".join([str(page.get("title") or ""), options] + facts))))
        return bool(wanted) and wanted <= said | {singular(w) for w in said}

    def only_listed(self, key: tuple[str, str], requirement: dict) -> bool:
        """True when the material appears only as one entry of a fact that lists several."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        title = page.get("title")
        kin = GLOSSARY.material_words(requirement["value"])
        if self.says(title, requirement) or self.says(options, requirement) or kin & {singular(w) for w in word_list(f"{title} {options}")}:
            return False
        for label, value in self.named_facts(page, requirement):
            entries = self.entries(value)
            if len(entries) == 1 and (self.entry_says(entries[0], requirement, label) or kin & set(entries[0].split())):
                return False
        return True

    def fused_colour(self, key: tuple[str, str], requirement: dict) -> bool:
        """True when the colour appears in the options only as a piece of a hyphen-joined word."""
        options = lowered(self.variant_of(key).get("options"))
        wanted = self.forms(requirement)
        alone = [{singular(w) for w in re.split(r"[\s()\[\]+]+", part) if w} for part in re.split(r"[/,;|]+", options)]
        if any(set(form.split()) <= part for form in wanted for part in alone):
            return False
        pieces = {singular(piece) for word in re.split(r"[\s/,;|()\[\]+]+", options) if "-" in word for piece in word.split("-") if piece}
        return any(set(form.split()) <= pieces for form in wanted)

    def second_colour(self, key: tuple[str, str], requirement: dict) -> bool:
        """True when the options also name a colour the shelf files as a colour of its own."""
        wanted = {w for form in self.forms(requirement) for w in form.split()}
        # single-word entries under the shelf's colour facts
        related, inferred, _ = self.own_labels(requirement)
        known: set[str] = set()
        for page in self.counter.pages.values():
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    known |= {entry for entry in self.entries(value) if re.fullmatch(r"[a-z]{3,}", entry)}
        known -= wanted
        options = str(self.variant_of(key).get("options") or "")
        parts = [{singular(w) for w in re.split(r"[^a-z0-9]+", lowered(part)) if w} for part in re.split(r"[/,;|]+", options)]
        named = any(set(form.split()) <= words for form in self.forms(requirement) for words in parts)
        return named and any(words & known for words in parts)

    def keeps_request(self, key: tuple[str, str]) -> bool:
        """The listing is the right kind and states every requirement the request itself wrote."""
        stated = [r for r in self.requirements if r.get("stated")]
        return bool((self.verdicts.get(key) or {}).get("kind_ok")) and all(self.plain(key, r) >= self.level_needed(r) for r in stated)

    def meets_chosen(self, key: tuple[str, str]) -> bool:
        """True when the listing meets every requirement the shopper marked as mattering more."""
        met = ((self.verdicts.get(key) or {}).get("met") or [])[: len(self.requirements)]
        marks = [x for x, r in zip(met, self.requirements) if r.get("chosen")]
        return bool(marks) and all(marks)

    def priced_priority(self) -> bool:
        """True when the price itself is what the shopper wants least of, not another quantity."""
        text = lowered(self.priority)
        paid = r"(?:price|cost|expensive|afford\w*|pay\w*|spend\w*)"
        return bool(re.search(r"\bcheap|\bbest (?:price|value|deal)\b|\bmost afford", text)
                    or re.search(rf"\b(?:low\w*|least|less|small\w*|minim\w*)\b(?:\s+\w+){{0,2}}?\s+{paid}", text)
                    or re.search(rf"\b{paid}\s+(?:\w+\s+)?(?:as\s+)?(?:low|little|less|least)\b", text))

    def figure(self, key: tuple[str, str]) -> Any:
        """The listing's figure for the priority: as its page states it, else as the reader read it."""
        settled, counted = self.counted_figure(key) if key[0] in self.counter.pages else (False, None)
        if settled:
            return counted
        read = self.figures[key] if key in self.figures else (self.verdicts.get(key) or {}).get("rank_value")
        if read is None and self.unnamed_ok and key[0] in self.counter.pages:
            settled, counted = self.counted_figure(key, unnamed=True)
            return counted if settled else None
        return read

    def rank_by_priority(self, every: bool = False) -> None:
        """Read a figure for each fitting listing; every=True batches until all are read."""
        if not self.priority or self.priced_priority():
            return
        for _ in range(4 if every else 1):
            fitting = self.ranked()
            keys = [k for k in fitting if k not in self.figures][:JUDGE_CARDS]
            if not keys or len(self.figures) >= self.figure_room or (len(fitting) < 2 and not every):
                return
            self.priority_unit()
            self.rank_asks += 1
            text = (f"Item: {self.item}\nThe shopper's priority: {self.priority}\n"
                    + (f"Write every figure in this unit, as for the listings read before: {self.figure_unit}\n" if self.figure_unit else "")
                    + "\nListings:\n" + "\n".join(self.card(i, key) for i, key in enumerate(keys, start=1)))
            answer = self.reader.fill(RANK_GUIDE, text, RANK_TOOL, 400) or {}
            if str(answer.get("better") or "").lower() in ("higher", "lower"):
                self.better = str(answer["better"]).lower()
            if not self.figure_unit and isinstance(answer.get("unit"), str):
                self.figure_unit = " ".join(answer["unit"].split())[:20]    # later batches must use this unit
            values = answer.get("values")
            values = re.split(r"[\n;]+", values) if isinstance(values, str) else (values or [])
            entries = [str(entry) for entry in values]
            for position, figure in self.figure_entries(entries, len(keys)).items():
                figure = self.range_end(keys[position - 1], figure)
                if self.figure_stated(keys[position - 1], figure):
                    self.figures[keys[position - 1]] = figure       # a figure the reader inferred, which the page never states, is not kept
            for key in keys:
                self.figures.setdefault(key, None)
            self.note(f"ranked by {self.priority!r}: " + " ".join(entry[:14] for entry in entries)[:200])

    def range_end(self, key: tuple[str, str], figure: float) -> float:
        """The end of a range that carries the unit, when the reader gave the other end (a range states the end that carries the unit)."""
        unit = self.unit_spelled(self.counted_unit())
        if not unit:
            return figure
        measure = GLOSSARY.listing_units(unit)
        page = self.counter.pages.get(key[0]) or {}
        for text in [*(page.get("facts") or {}).values(), self.variant_of(key).get("options"), page.get("title")]:
            for found in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:to|-|\u2013|~)\s*(\d+(?:\.\d+)?)\s*([a-z]+)(?![a-z0-9])", lowered(text)):
                word = self.unit_spelled(found.group(3))
                size = 1.0 if word == unit else measure.get(word)
                if size and found.group(1) != found.group(2) and abs(float(found.group(1)) * size - figure) <= 1e-6 * max(1.0, abs(figure)):
                    return round(float(found.group(2)) * size, 6)
        return figure

    def figure_entries(self, entries: list[str], count: int) -> dict[int, float]:
        """Map the reader's entries to listing positions (from 1) and figures: an entry may lead with the listing number or follow the shown order, and may carry a unit word; an entry with no figure or with several (dimensions) gives none."""
        led = [re.match(r"\s*\[?(\d+)\]?(?!\.\d)[\s:.)-]+(.*)$", entry) for entry in entries]
        led = [m if m and (re.search(r"\d", m.group(2)) or re.fullmatch(r"\W*(?:none|null|n/a)\W*", lowered(m.group(2)))) else None for m in led]
        numbers = [int(m.group(1)) for m in led if m]
        indexed = len(numbers) == len(entries) and len(set(numbers)) == len(numbers) and all(1 <= n <= count for n in numbers)
        if not indexed and len(entries) != count:
            return {}
        unit = self.unit_spelled((re.findall(r"[a-z]+", lowered(self.figure_unit)) or [""])[0])
        out: dict[int, float] = {}
        for position, entry in enumerate(entries, start=1):
            text = led[position - 1].group(2) if indexed else entry
            found = re.findall(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*([a-z]*)", lowered(text).replace(",", ""))
            if len(found) != 1 or re.search(r"\d\s*(?:x|\*|by)\s*\d", lowered(text)):
                continue
            figure, word = float(found[0][0]), self.unit_spelled(found[0][1]) if found[0][1] else ""
            if unit and word and word != unit and GLOSSARY.unit_quantity.get(word):
                # a known unit of this measure is converted by size; a unit of another measure gives no figure
                sizes = (GLOSSARY.unit_size.get(word), GLOSSARY.unit_size.get(unit))
                if not all(sizes) or sizes[0][0] != sizes[1][0]:
                    continue
                figure *= sizes[0][1] / sizes[1][1]
            out[numbers[position - 1] if indexed else position] = figure
        return out

    def priority_unit(self) -> None:
        """Name the priority's unit from the shopper's words before any listing's figures are read."""
        if self.figure_unit or self.unit_asked == self.priority:
            return
        self.unit_asked = self.priority
        text = f"Request: {self.counter.request}" + (f"\nThe shopper's priority now: {self.priority}" if self.priority != self.first_priority else f"\nThe shopper's priority: {self.priority}")
        answer = self.reader.fill(MEASURE_GUIDE, text, MEASURE_TOOL, 300) or {}
        unit = " ".join(str(answer.get("unit") or "").split())[:20]
        if unit and not re.search(r"\d", unit):
            self.figure_unit = unit
            self.figure_quantity = lowered(str(answer.get("quantity") or "")).strip()
        self.note(f"unit of {self.priority!r}: {answer.get('quantity')!r} in {unit!r}")

    def best_figures(self) -> list[tuple[float, tuple[str, str]]]:
        lower = self.better == "lower"
        known = [(float(self.figure(k)), k) for k in self.ranked() if k not in self.gone
                 and isinstance(self.figure(k), (int, float)) and not isinstance(self.figure(k), bool)
                 and self.figure_stated(k, float(self.figure(k)))]      # skip figures the page never states
        return sorted(known, key=lambda pair: pair[0] if lower else -pair[0])

    def figure_forms(self, queries: list[str]) -> list[str]:
        """Searches carrying a priority-unit figure, redone with the unit spelled out and with none."""
        unit = self.unit_spelled(self.counted_unit())
        if not unit:
            return []
        spelled = {}
        for spelling, canon in GLOSSARY.unit_by_spelling.items():
            if canon == unit and len(spelling) >= 4 and re.fullmatch(r"[a-z]+", spelling):
                spelled.setdefault(singular(spelling), spelling)
        have = {lowered(q) for q in queries}
        out: list[str] = []
        for query in queries:
            text = lowered(query)
            found = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]{1,10})(?![a-z0-9])", text)
            if not found or self.unit_spelled(found.group(2)) != unit:
                continue
            rest = " ".join((text[:found.start()] + " " + text[found.end():]).split())
            for form in [f"{rest} {found.group(1)}"] + [f"{rest} {found.group(1)} {name}" for name in list(spelled.values())[:2]]:
                if form not in out and form not in have:
                    out.append(form)
        return out[:FORM_LIMIT]

    def hunt_rules(self, taken: list[str]) -> list[str]:
        """Rule-made searches (see explore_queries) not yet sent and not already among those taken."""
        have = {lowered(q) for q in taken}
        fresh = [q for q in self.explore_queries() if lowered(q) not in self.sent and self.read_as(lowered(q)) not in self.sent_as and lowered(q) not in have]
        return list(dict.fromkeys(fresh))[:HUNT_RULES]

    def hunt_best(self) -> None:
        """Read every fitting listing's figure and keep searching while a round still finds a better one."""
        counter = self.counter
        again, self.hunt_stale = self.hunt_stale and bool(counter.cart), False     # a hunt before any cart keeps the usual reserve
        if not self.priority or self.priced_priority() or not counter.has("search"):
            return
        best_so_far: float | None = None
        for round_number in range(BEST_ROUNDS + BEST_EXTRA):
            # with a listing carted, the hunt resumes only after the shopper changes a requirement
            if counter.over or (counter.cart and not again) or self.late() or counter.turns_left() <= (TURN_RESERVE + 1 if again else max(TURN_RESERVE + 4, self.reserve())):
                break
            self.rank_by_priority(every=True)
            found = self.best_figures()
            best = found[0][0] if found else None
            better_now = best is not None and (best_so_far is None or (best < best_so_far if self.better == "lower" else best > best_so_far))
            if round_number >= BEST_ROUNDS and not better_now:
                break               # extra rounds continue only while the last one improved the best
            if round_number and not better_now and len(found) >= READ_FIGURES:
                break
            best_so_far = best
            shown = "\n".join(f"- {value:g}: {str((counter.pages.get(key[0]) or {}).get('title') or '')[:90]} | {self.variant_of(key).get('options') or '-'}"
                              for value, key in found[:6]) or "(no fitting listing states a figure yet)"
            text = (f"{self.state_text()}\n\nBest figures found so far for the priority, with the listings that carry them:\n{shown}"
                    f"\n\nSearches tried:\n" + "\n".join(sorted(self.sent)[-30:]))
            answer = self.listed_fields(self.reader.fill(BEST_GUIDE, text, BEST_TOOL, 300))
            queries = [q for q in answer.get("queries") or [] if isinstance(q, str) and q.strip()][:8]
            # the fitting listings' titles plus the required values bring up their other variants
            every = " ".join(self.values())
            forms = self.figure_forms(queries)
            rules = self.hunt_rules(queries + forms)
            siblings = []
            for key in [k for _, k in found] + [k for k in self.ranked() if k not in self.gone]:
                title = lowered((counter.pages.get(key[0]) or {}).get("title"))
                if title and title not in self.kin_swept and len(siblings) < counter.group_size - len(queries) - len(forms) - len(rules):
                    self.kin_swept.add(title)
                    siblings.append(f"{every} {' '.join(word_list(title)[:12])}")
            queries = siblings + queries + forms + rules
            self.note(f"better by {self.priority!r} than {best}: {queries}")
            if not queries:
                break
            self.search_room = max(self.search_room, len(self.sent) + len(queries))
            self.add_queries(queries, first=True)
            self.maker_filters()
            searches = self.next_searches(min(counter.group_size, len(queries) + len(self.pending_filters)))
            if not searches:
                break
            counter.send(searches)
            self.hear_quietly()
            opens = self.to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + 60)
            if opens and not counter.over and counter.turns_left() > TURN_RESERVE + 2:
                counter.send(opens[: counter.group_size])
                self.hear_quietly()
            self.judge()
        self.rank_by_priority(every=True)

    def rethink_searches(self) -> bool:
        """Nothing meets every requirement: the reader writes new searches; True when it searched."""
        if self.search_asks >= 2 or self.late() or self.counter.turns_left() <= self.reserve() + 2:
            return False
        self.search_asks += 1
        seen = list(dict.fromkeys(str(p.get("title") or "")[:90] for p in self.counter.pages.values()))[-25:]
        text = (f"{self.state_text()}\n\nSearches tried:\n" + "\n".join(sorted(self.sent)[-25:])
                + "\n\nTitles seen (none meets every requirement):\n" + "\n".join(seen))
        answer = self.listed_fields(self.reader.fill(SEARCH_GUIDE, text, SEARCH_TOOL, 400))
        queries = [q for q in answer.get("queries") or [] if isinstance(q, str) and q.strip()][:6]
        self.note(f"new searches: {queries}")
        if not queries:
            return False
        self.search_room = max(self.search_room, len(self.sent) + len(queries))
        self.add_queries(queries, first=True)
        self.look(rounds=3, enough=1)
        return True

    def explore_queries(self) -> list[str]:
        """Rule-made searches from the values, the item word, shared title words and makers seen."""
        values = [v for v in self.values() if v]
        if not values:
            return []
        every = " ".join(values)
        head = (self.item_words() or [""])[-1]
        rows = list(self.counter.rows.values())
        titles = [str(row.get("title") or "") for row in rows] + [str(page.get("title") or "") for page in self.counter.pages.values()]
        known = set(word_list(every)) | set(self.item_words())
        shared = Tally(w for title in titles for w in sorted(self.content_words(title)) if len(w) >= 4 and w not in known)
        words = [q for w, count in shared.most_common(30) if count >= 2 for q in (f"{w} {every}", f"{w} {every} {head}".strip())]
        makers = Tally(lowered(row.get("brand")) for row in rows if row.get("brand"))
        shelves = [q for maker, _ in makers.most_common(10) for q in (f"{maker} {every}", f"{maker} {every} {head}".strip())]
        less = [f"{head} " + " ".join(v for j, v in enumerate(values) if j != i) for i in range(len(values)) if len(values) > 1]
        turns = [q for group in zip_longest(words, shelves, less) for q in group if q]
        return [f"{every} {head}".strip()] + turns

    def explore_more(self) -> bool:
        """Send rule-made searches in small rounds once the reader's own are used up; True if sent."""
        if self.explored >= EXPLORE_ROUNDS or self.late() or self.counter.turns_left() <= TURN_RESERVE + 2:
            return False
        try:
            fresh = [q for q in self.explore_queries() if lowered(q) not in self.sent and self.read_as(lowered(q)) not in self.sent_as]
            batch = list(dict.fromkeys(fresh))[:EXPLORE_PER_ROUND]
            if not batch:
                return False
            self.explored += 1
            self.note(f"rule searches {self.explored}: {batch}")
            self.search_room = max(self.search_room, len(self.sent) + len(batch))
            self.add_queries(batch, first=True)
            # sent now: look() would otherwise put its own searches ahead of these
            searches = self.next_searches(self.counter.group_size)
            if searches:
                self.counter.send(searches)
                self.hear_quietly()
            self.look(rounds=2, enough=1)
            return True
        except Exception as exc:    # an extra search failing must not block the order
            self.explored = EXPLORE_ROUNDS
            self.note(f"rule searches stopped: {type(exc).__name__}: {exc}")
            return False

    def read_pending(self) -> None:
        """Open the rows searches showed after the last opening: a search is worth its turn only once its rows are read, and the
        last searches before a choice run to the edge of the reserve."""
        counter = self.counter
        if counter.over or self.late() or counter.turns_left() <= READ_FLOOR:
            return
        # (only rows whose title names the item or a required value: a row of something else is not worth the turn)
        opens = self.to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + counter.group_size * OPEN_BATCH, since=counter.opened_turn, worth=1.0)
        if opens:
            counter.send(opens[: counter.group_size])
            self.hear_quietly()
            self.judge()

    def read_rows(self, choice: tuple[str, str] | None = None) -> None:
        counter = self.counter
        for number in range(2):
            opens = self.to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + counter.group_size * OPEN_BATCH)
            if not opens or counter.over or (number and counter.turns_left() <= TURN_RESERVE + 1):
                break
            opens = opens[: counter.group_size]
            counter.send(opens + (self.stock_reads(counter.group_size - len(opens), choice) if choice else []))
            self.hear_quietly()

    def stock_reads(self, room: int, skip: tuple[str, str] | None = None, adding: bool = False) -> list[dict]:
        """Stock checks for listings last read before the first cart (adding: before the one being carted)."""
        counter = self.counter
        if room <= 0 or not counter.has("inspect_stock") or not (adding or counter.cart):
            return []
        since = counter.turn + 1 if adding else counter.cart_turn
        stale = [k for k in self.ranked() if k not in self.gone and k != skip and k not in counter.cart
                 and int((counter.stock.get(k) or {}).get("turn") or 0) < since]
        return [act("inspect_stock", product_id=k[0], sku=k[1]) for k in stale[: min(room, READ_AHEAD)]]

    def dig_after_change(self) -> None:
        counter = self.counter
        left = [k for k in self.ranked() if k not in self.gone]
        if len([k for k in left if self.firm(k)]) >= 2 or counter.turns_left() <= TURN_RESERVE or self.late():
            return
        self.kin_done.clear()
        self.add_queries(self.kin_queries() + self.shelf_queries(6), first=True)
        self.look(rounds=3 if left else 5, enough=2 if left else 1)

    def surer(self, key: tuple[str, str], other: tuple[str, str]) -> bool:
        """True when the listing states the requirements more plainly than the other."""
        mine, theirs = self.trust(key), self.trust(other)

        def plainly(one: tuple[str, str]) -> int:
            return sum(self.plain(one, r) + self.exact(one, r) for r in self.requirements)
        return mine > theirs * 1.34 or (mine >= theirs and plainly(key) > plainly(other))

    def cart_keeper(self) -> tuple[str, str] | None:
        """The latest carted variant that is still on sale, still fits and states every value firmly."""
        for key in reversed(self.counter.cart):
            if key not in self.gone and self.on_sale(key) and key in self.verdicts and self.fits(key) and self.firm(key):
                return key
        return None

    def worth_swap(self, new: tuple[str, str], old: tuple[str, str]) -> bool:
        """A carted firm fit gives way only to a firm fit that is clearly better by the shopper's priority (a third or more)."""
        if not self.firm(new) or not self.fits(new):
            return False
        if self.priced_priority():
            mine, theirs = (self.counter.stock.get(new) or {}).get("price"), (self.counter.stock.get(old) or {}).get("price")
            return isinstance(mine, float) and isinstance(theirs, float) and mine <= 0.75 * theirs
        if not self.priority:
            return False

        def number(key: tuple[str, str]) -> float | None:
            value = self.figure(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None
        mine, theirs = number(new), number(old)
        if mine is None:
            return False
        if theirs is None:
            return True             # the carted fit states no figure for the priority, so it cannot be the best option
        higher = self.better == "higher" if self.better in ("higher", "lower") else not re.search(r"\b(low|least|small|short|few|min|light)", lowered(self.priority))
        return mine >= theirs * 4 / 3 if higher else mine <= 0.75 * theirs

    def recart(self) -> tuple[str, str] | None:
        """Cart the next fitting listing, rereading its stock and price; returns its key."""
        counter = self.counter
        for key in [k for k in self.ranked() if k not in self.gone][:2]:
            if counter.over or counter.turns_left() < 3:
                break
            group = [act("add_to_cart", product_id=key[0], sku=key[1])]
            if counter.has("inspect_stock"):
                group.append(act("inspect_stock", product_id=key[0], sku=key[1]))
            counter.send(group)
            self.hear()
            seen = counter.stock.get(key) or {}
            if seen.get("in_stock") is not False and (self.budget is None or not isinstance(seen.get("price"), float) or seen["price"] <= self.budget + 1e-9):
                return key
            self.gone.add(key)
        return None

    def reply_before_paying(self, choice: tuple[str, str]) -> None:
        lines, self.unanswered = self.unanswered[-4:], []
        self.answer("\n".join(f"- {line}" for line in lines))

    def last_resort(self) -> None:
        """After a failure, still order the cart's listing or the best one left."""
        counter = self.counter
        if counter.over or counter.turns_left() < 1 or not counter.has("place_test_order"):
            return
        alive = [k for k in counter.cart if k not in self.gone]
        key = alive[-1] if alive else next((k for k in self.ranked() if k not in self.gone), None)
        if key is None:
            return
        if key not in counter.cart:
            if counter.turns_left() < 2:
                return
            counter.send([act("add_to_cart", product_id=key[0], sku=key[1])])
        counter.send([self.final_action(key)])

    def near_misses(self) -> list[tuple[str, str]]:
        def score(key: tuple[str, str]) -> tuple:
            verdict = self.verdicts[key]
            met = (verdict.get("met") or [])[: len(self.requirements)]
            chosen = sum(1 for x, r in zip(met, self.requirements) if x and r.get("chosen"))
            # requirements the shopper marked as mattering more outrank the raw count of those met
            # a near miss whose page states another value (a contradiction) is farther than one silent on it
            return (-int(bool(verdict.get("kind_ok"))), -chosen, -sum(1 for x in met if x), self.contradicted(key), -self.said_in_part(key, met),
                    (self.counter.stock.get(key) or {}).get("price") or 0.0)
        return sorted((k for k in self.verdicts if self.affordable(k)), key=score)

    def hear_quietly(self) -> None:
        if self.counter.heard:
            self.hear()

    def fact_check(self) -> None:
        """For a word value filed under the property's own fact, have the reader say whether an entry the words model did not
        list is the same value in other words; such an entry is then unknown rather than contrary."""
        if self.silent or self.fact_checks >= 4:
            return
        for requirement in self.requirements:
            if self.kind(requirement) != "fact" or re.search(r"\d", requirement["value"]):
                continue
            entries = []
            for key in self.candidates():
                page = self.counter.pages.get(key[0]) or {}
                for label, value in self.named_facts(page, requirement):
                    found = self.entries(value)
                    if len(found) != 1:
                        continue
                    entry = found[0]
                    tag = (requirement["property"], requirement["value"], lowered(entry))
                    if tag in self.fact_same or tag in [t for t, _ in entries] or self.entry_is(entry, requirement, label) \
                            or self.entry_says(entry, requirement, label, loose=True):
                        continue
                    entries.append((tag, entry))
            entries = entries[:12]
            if not entries or self.fact_checks >= 4:
                continue
            self.fact_checks += 1
            asked = (f"Item: {self.item}\nProperty: {requirement['property']}\nWanted value: {requirement['value']}\n\nEntries:\n"
                     + "\n".join(f"[{i}] {entry}" for i, (_, entry) in enumerate(entries, start=1)))
            answer = self.reader.fill(SAME_GUIDE, asked, SAME_TOOL, 300) or {}
            marks = answer.get("marks")
            marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
            for position, mark in enumerate(marks, start=1):
                word = re.search(r"\b(same|kind|other|none)\b", lowered(mark))
                number = re.match(r"\s*\[?(\d+)", str(mark))
                at = int(number.group(1)) if number and 1 <= int(number.group(1)) <= len(entries) else position
                if word and 1 <= at <= len(entries):
                    self.fact_same[entries[at - 1][0]] = word.group(1)
            for tag, _ in entries:
                self.fact_same.setdefault(tag, "unread")
            self.note(f"same {requirement['value']}: " + " ".join(f"{e}={self.fact_same[t]}" for t, e in entries)[:240])

    def name_check(self) -> None:
        """For materials, have the reader classify how each title names the wanted value."""
        if self.silent:
            return
        for requirement in self.requirements:
            kind = self.kind(requirement)
            if kind != "material" or self.name_checks >= 10:
                continue
            others = [r for r in self.requirements if r is not requirement]
            waiting = []
            for key in self.candidates():
                tag = (key, requirement["property"], requirement["value"])
                if tag in self.named_as or any(self.plain(key, r) < 1 for r in others):
                    continue
                page = self.counter.pages.get(key[0]) or {}
                options = str(self.variant_of(key).get("options") or "").strip()
                if kind == "colour":
                    text = f"variant: {options}" if options else f"title: {str(page.get('title') or '')[:150]}"
                else:
                    text = f"title: {str(page.get('title') or '')[:150]}" + (f" | variant: {options}" if options else "")
                waiting.append((tag, text))
            waiting = waiting[:12]
            if not waiting:
                continue
            self.name_checks += 1
            asked = (f"Item: {self.item}\nProperty: {requirement['property']}\nWanted value: {requirement['value']}\n\nTexts:\n"
                     + "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(waiting, start=1)))
            answer = self.reader.fill(NAMES_GUIDE, asked, NAMES_TOOL, 500) or {}
            marks = answer.get("marks")
            marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
            spoken = [re.match(r"\s*\[?(\d+)", str(mark)) for mark in marks]
            # the numbers index the texts only when distinct and inside the list ('0 none' / '1 only' is a flag, not an index)
            numbered = bool(marks) and all(spoken) and len({int(m.group(1)) for m in spoken}) == len(spoken) and all(1 <= int(m.group(1)) <= len(waiting) for m in spoken)
            if numbered or len(marks) == len(waiting):      # unnumbered marks count only with one per text
                for position, mark in enumerate(marks, start=1):
                    word = re.search(r"\b(only|among|other|none)\b", lowered(mark))
                    number = re.match(r"\s*\[?(\d+)", str(mark))
                    at = int(number.group(1)) if numbered else position
                    if word and 1 <= at <= len(waiting):
                        self.named_as[waiting[at - 1][0]] = word.group(1)
            for tag, _ in waiting:
                self.named_as.setdefault(tag, "unread")
            self.note(f"names {requirement['value']}: " + " ".join(f"{i}:{self.named_as[tag]}" for i, (tag, _) in enumerate(waiting, start=1))[:240])

    def widen(self) -> None:
        """Ask the reader for other seller wordings of each value and search for them."""
        self.widened = True
        for requirement in self.requirements:
            answer = self.reader.fill(WORDS_GUIDE, f"Item: {self.item}\nProperty: {requirement['property']}\nValue: {requirement['value']}", WORDS_TOOL, 200) or {}
            words = [str(w).strip() for w in answer.get("words") or [] if str(w).strip()][:6]
            name = "written" if re.search(r"\d", requirement["value"]) else "alike"
            requirement[name] = list(dict.fromkeys(list(requirement.get(name) or []) + words))
            others = " ".join(r["value"] for r in self.requirements if r is not requirement)
            searched = self.amount_spellings(dict(requirement, written=words)) if name == "written" else words
            self.search_room = max(self.search_room, len(self.sent) + len(self.queries) + 2 * len(searched[:3]))
            plain_word = (self.item_words() or [""])[-1]
            self.add_queries([f"{word} {others} {plain_word}" for word in searched[:3]] + [f"{word} {others} {self.item}" for word in searched[:3]], first=True)
        self.note(f"widened: {[(r['value'], r.get('written'), r.get('alike')) for r in self.requirements]}")

    def refill_queries(self, lost: tuple[str, str], wide: bool = True) -> list[str]:
        """Searches for a lost variant's siblings and relistings; wide adds full-title and shelf searches."""
        words = word_list((self.counter.pages.get(lost[0]) or {}).get("title"))
        if len(words) < 2:
            return []
        every = " ".join(self.values())
        own = " ".join(w for w in word_list(self.variant_of(lost).get("options")) if not w.replace(".", "").isdigit())
        queries = [" ".join(words), " ".join(words[:10])]
        if own:
            queries.append(f"{own} {' '.join(words[:10])}")
        queries += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.item} {' '.join(words[:3])}"]
        if wide:
            queries += [f"{every} {' '.join(words)}"] + self.shelf_queries(6)
        return list(dict.fromkeys(lowered(" ".join(q.split())[:QUERY_LENGTH]) for q in queries))

    def refill(self, lost: tuple[str, str], wide: bool = True) -> None:
        """The chosen variant went out of stock: search for the listing's next variant or a relisting."""
        counter = self.counter
        texts = self.refill_queries(lost, wide)
        if not texts or not counter.has("search"):
            return
        calls = [act("search", query=text, k=10, in_stock=True) for text in texts]
        label, price = self.store_label(), (counter.stock.get(lost) or {}).get("price")
        if wide and label and isinstance(price, float) and counter.has("filter"):
            # same shelf at that price: a listing's variants share one price
            ceiling = round(min(self.budget or price * 1.1, price * 1.1), 2)
            brand = str((counter.pages.get(lost[0]) or {}).get("brand") or "").strip()
            calls.append(act("filter", category=label, max_price=ceiling, in_stock=True, k=10))
            if brand and lowered(brand) not in ("none", "no brand", "oem"):
                calls.append(act("filter", category=label, brand=brand, max_price=ceiling, in_stock=True, k=10))
        counter.send(calls[: counter.group_size])
        self.hear_quietly()
        opens = self.to_open(4 * OPEN_BATCH, allowance=len(counter.pages) + 4 * OPEN_BATCH)
        if opens and not counter.over:
            counter.send(opens)
            self.hear_quietly()
        self.judge()

    def solid(self, key: tuple[str, str]) -> bool:
        """Every value is stated as required in the options, the property's own fact or the title."""
        return (self.verdicts.get(key) or {}).get("by") == "words" and all(self.plain(key, r) >= 2 for r in self.requirements)

    def firm(self, key: tuple[str, str]) -> bool:
        """Every value stands in the options or the property's own fact (the title too for amounts)."""
        if (self.verdicts.get(key) or {}).get("by") != "words":
            return False
        return all(self.plain(key, r) >= self.level_needed(r) for r in self.requirements)

    def reserve(self) -> int:
        """Turns held back from looking; before the first cart, extra room for what the cart sets off."""
        if self.counter.cart or not [k for k in self.ranked() if k not in self.gone]:
            return TURN_RESERVE
        return min(TURN_RESERVE + FIRST_CART_ROOM, self.counter.turn_limit // 2)

    def look(self, rounds: int, enough: int = 2) -> None:
        """Search and read listings until enough fit or the rounds run out."""
        counter = self.counter
        for round_number in range(rounds):
            if counter.over or counter.turns_left() <= self.reserve() or self.late():
                break
            opens = self.to_open(counter.group_size * OPEN_BATCH)
            if opens:
                counter.send(opens[: counter.group_size])
                self.hear_quietly()
            self.judge()
            fitting = [k for k in self.ranked() if self.solid(k)]
            if len([k for k in fitting if self.firm(k)]) >= enough:
                return
            if len(fitting) >= enough and not self.kin_pending() and round_number >= 2:
                return
            self.maker_filters()
            told_new = self.shelf_queries(8) if self.fresh_rounds > 0 and len(fitting) < enough else []
            if told_new:
                # too little fits right after a requirement change: search the new values in the shelf's own words
                self.fresh_rounds -= 1
                named = [q for q in (lowered(" ".join(str(v).split())[:QUERY_LENGTH]) for v in self.value_queries()) if q and q not in self.sent]
                named = list(dict.fromkeys(named))[: counter.group_size // 2]
                self.add_queries(told_new, first=True)
                twins = [q for q in self.twin_queries() if q not in self.sent and q not in named]
                self.queries = twins + named + [q for q in self.queries if q not in named and q not in twins]
                searches = self.next_searches(counter.group_size)
            else:
                self.add_queries(self.kin_queries(), first=True)
                if len(fitting) < enough:
                    self.add_queries(self.variant_queries(wide=not fitting), first=True)
                if not fitting and round_number >= 1:
                    self.add_queries(self.leave_one_queries(), first=True)
                # nothing fits yet: title-based sibling searches go first and displace the price-share filters
                siblings = self.sibling_queries(counter.group_size // 2) if not fitting else []
                if siblings:
                    self.add_queries(siblings, first=True)
                    self.add_queries(self.family_queries(2), first=True)
                if len(self.queries) < counter.group_size // 2:
                    self.add_queries(self.shelf_queries(6))
                searches = [] if siblings else self.shelf_filters(counter.group_size // 2)
                searches += self.next_searches(counter.group_size - len(searches))
            if not searches and not opens:
                break
            if searches:
                counter.send(searches)
                self.hear_quietly()
        opens = self.to_open(counter.group_size * OPEN_BATCH)
        # the rows of the last search wave are read even when the wave left exactly the reserve of turns
        if opens and not counter.over and counter.turns_left() >= TURN_RESERVE:
            counter.send(opens[: counter.group_size])
            self.hear_quietly()
        self.judge()
        if len(self.ranked()) < enough and not self.widened and not counter.over:
            self.widen()
            return self.look(rounds=3, enough=enough)
        for _ in range(2):                  # too few by literal words: let the reader judge other wordings
            if len(self.ranked()) >= enough:
                return
            self.ask_reader()
        if len(self.ranked()) < enough and not self.lenient:
            self.lenient = True
            self.judge()

    def places(self, key: tuple[str, str]) -> dict[str, str]:
        """Named text fields of a listing: title, chosen options and each labelled fact."""
        page = self.counter.pages.get(key[0]) or {}
        places = {"title": str(page.get("title") or ""), "options": str(self.variant_of(key).get("options") or "")}
        places.update({str(label): str(value) for label, value in (page.get("facts") or {}).items()})
        return places

    def stated_at(self, key: tuple[str, str], requirement: dict) -> dict | None:
        """The field and wording on the page that states the required value, or None."""
        places = self.places(key)
        page = self.counter.pages.get(key[0]) or {}
        own = sorted((label for label, _ in self.named_facts(page, requirement)), key=lambda label: tuple(-n for n in self.label_fit(label, requirement)))
        order = own + ["options", "title"] + [label for label in places if label not in own and label not in ("options", "title")]
        forms = sorted(self.forms(requirement) | {" ".join(word_list(closed_up(requirement["value"])))}, key=len, reverse=True)
        for field in order:
            text = places.get(field) or ""
            for reading in (lowered(text), closed_up(text)):
                for form in forms:
                    if not form:
                        continue
                    # digits may be split from a trailing unit on the page
                    found = re.search(r"(?<![a-z0-9])" + re.sub(r"(\d)(?=[a-z])", r"\1\\s*", re.escape(form).replace(r"\ ", r"\W+")) + r"(?![a-z0-9])", reading)
                    if found and f" {' '.join(word_list(found.group(0)))} " in f" {' '.join(word_list(text))} ":
                        return {"field": field, "value": found.group(0)}
        if self.kind(requirement) == "amount":
            # the page may state the same quantity in another unit of the measure
            for field in order:
                text = places.get(field) or ""
                same, others = self.amounts(text, requirement)
                for found in re.finditer(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([a-z]+)(?![a-z0-9])", lowered(text)):
                    if same and not others and (float(found.group(1)), self.unit_spelled(found.group(2))) in same:
                        return {"field": field, "value": found.group(0)}
        return None

    def label_fit(self, label: Any, requirement: dict) -> tuple[int, int, int]:
        """Closeness of a fact label to the property name and to the evidence sentence, as a sort key."""
        named = self.name_words(label)
        return (len(named & self.name_words(requirement["property"])), len(named & self.cited_words()), len(named))

    def holding_fact(self, key: tuple[str, str], requirement: dict) -> dict | None:
        """The property-labelled fact whose value satisfies the requirement, best label first."""
        page = self.counter.pages.get(key[0]) or {}
        held = [(str(label), str(value).strip()) for label, value in self.named_facts(page, requirement)
                if any(self.entry_says(entry, requirement, str(label)) for entry in self.entries(value))]
        if not held:
            return None
        label, value = max(held, key=lambda entry: self.label_fit(entry[0], requirement))
        return {"field": label, "value": value}

    def labelled_claim(self, key: tuple[str, str], requirement: dict) -> dict | None:
        """The fact under the best-matching label, whatever value it carries."""
        if self.kind(requirement) == "colour":
            return None
        page = self.counter.pages.get(key[0]) or {}
        named = [(self.label_fit(label, requirement), str(label), str(value).strip())
                 for label, value in (page.get("facts") or {}).items() if word_list(str(value)) and lowered(label) != "brand"]
        named = [entry for entry in named if entry[0][0] > 0]
        if not named:
            return None
        _, label, value = max(named)
        return {"field": label, "value": value}

    def cited_words(self) -> set[str]:
        """Words of the request sentence that says what the order must be justified with."""
        words: set[str] = set()
        for sentence in re.split(r"(?<=[.!?])\s+", getattr(self, "request", "")):
            low = lowered(sentence)
            if re.search(r"\bprice", low) and re.search(r"\bstock", low) and re.search(r"justif|\bback|support|explain|\bbase|\bcit(?:e|ing)\b|referenc|point|evidence|\bfacts?\b|checked|\bsaw\b|\bseen\b|observ|verified|confirm", low):
                words |= self.name_words(sentence) - {"price", "stock", "brand", "status", "listing", "detail", "order", "current", "mention", "include", "add", "fine", "want", "like"}
        return words

    def sibling_label(self, key: tuple[str, str], requirement: dict, fact: dict) -> dict:
        """Swap in a better-matching label when it carries the same wording as the chosen fact."""
        page = self.counter.pages.get(key[0]) or {}
        facts = page.get("facts") or {}
        said = f" {' '.join(word_list(fact['value']))} "
        if fact["field"] not in facts or said == "  ":
            return fact
        best = max(((self.label_fit(label, requirement), str(label)) for label, value in facts.items()
                    if lowered(label) != "brand" and said in f" {' '.join(word_list(value))} "), default=((0, 0, 0), fact["field"]))
        if best[0] > self.label_fit(fact["field"], requirement):
            return {"field": best[1], "value": fact["value"]}
        return fact

    def claims(self, key: tuple[str, str], about: dict | None = None) -> dict | None:
        """The facts shown with an order: price, stock flag and one page fact for a requirement."""
        seen = self.counter.stock.get(key) or {}
        places = self.places(key)
        claims: list[dict] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        claims.append({"field": "in_stock", "value": seen.get("in_stock") is not False})
        fact = None
        verdict = self.verdicts.get(key) or {}
        # chosen or stated requirements lead; wishes offered as a choice do not
        wish = [r for r in self.requirements if r.get("chosen") or self.is_wish(r["value"])]
        firm = [r for r in self.requirements if r not in wish and (r.get("stated") or lowered(r["value"]) in self.took_back)]
        asked = self.cited_words()
        named = [r for r in self.requirements if r not in wish and self.name_words(r["property"]) & asked]
        ordered = ([about] if about else []) + named + [r for r in firm if r not in named] + [r for r in self.requirements if r not in firm and r not in named]
        for requirement in ordered:
            fact = self.stated_at(key, requirement)
            if requirement in firm or requirement in named:
                held = None if self.kind(requirement) in ("amount", "colour") else self.holding_fact(key, requirement)
                if held and (fact is None or fact["field"] not in {label for label, _ in self.named_facts(self.counter.pages.get(key[0]) or {}, requirement)}):
                    fact = held                 # cite the property's own fact, not options or title
                elif fact is None:
                    fact = self.labelled_claim(key, requirement)
                else:
                    fact = self.sibling_label(key, requirement, fact)
            if fact:
                break
        if fact is None:
            for where, words in zip(verdict.get("where") or [], verdict.get("words") or []):
                where = re.sub(r"^facts?\s*[:.]\s*", "", str(where or ""))
                field = next((label for label in places if lowered(label) == lowered(where)), None)
                said = " ".join(word_list(words))
                if field and said and f" {said} " in f" {' '.join(word_list(places[field]))} ":
                    fact = {"field": field, "value": str(words).strip()}
                    break
        if fact:
            claims.append(fact)
        return {"claims": claims}

    def shortfall(self, key: tuple[str, str]) -> dict | None:
        """The facts shown with an abstaining order: price, stock flag and what the page offers instead."""
        seen = self.counter.stock.get(key) or {}
        page = self.counter.pages.get(key[0]) or {}
        places = self.places(key)
        claims: list[dict] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        claims.append({"field": "in_stock", "value": seen.get("in_stock") is not False})
        missed = sorted((r for r in self.requirements if self.plain(key, r) < 1), key=lambda r: self.plain(key, r)) or list(self.requirements)
        for requirement in missed:
            for label, value in sorted(self.named_facts(page, requirement), key=lambda item: self.label_fit(item[0], requirement), reverse=True):
                words = " ".join(word_list(value))
                if words:
                    claims.append({"field": label, "value": str(value).strip()})
                    return {"claims": claims}
        for field in ("options", "title"):
            if " ".join(word_list(places.get(field))):
                claims.append({"field": field, "value": places[field].strip()})
                break
        return {"claims": claims}

    def look_again(self, choice: tuple[str, str]) -> None:
        """Open the requested number of fitting listings before the final choice, once things are settled."""
        counter = self.counter
        stamp = json.dumps([[(r["property"], r["value"]) for r in self.requirements], self.budget], sort_keys=True)
        if not self.look_at or self.looked_under == stamp or counter.over or counter.turns_left() < 2 or not (counter.has("compare") or counter.has("view")):
            return
        self.looked_under = stamp
        wanted = self.look_at + 2
        picked: list[str] = []
        titles: set[str] = set()
        for key in [choice] + [k for k in self.ranked() if k not in self.gone] + self.near_misses():
            title = lowered((counter.pages.get(key[0]) or {}).get("title"))
            if key[0] in picked or (title in titles and key != choice) or not self.on_sale(key):
                continue
            picked.append(key[0])
            titles.add(title)
            if len(picked) >= wanted:
                break
        if counter.has("compare"):
            calls = [act("compare", product_ids=picked[i:i + OPEN_BATCH]) for i in range(0, len(picked), OPEN_BATCH)]
        else:
            calls = [act("view", product_id=i) for i in picked]
        self.note(f"opened {len(picked)} fitting options before choosing")
        counter.send(calls)
        self.hear_quietly()

    def final_action(self, choice: tuple[str, str]) -> dict:
        if choice == self.named_nearest and self.can_abstain():
            # the nearest-listing order carries the same justification as a normal one
            return act("place_test_order", product_id=choice[0], sku=choice[1], abstain=True, justification=self.claims(choice))
        if choice in self.counter.cart or not self.can_abstain():
            return act("place_test_order", product_id=choice[0], sku=choice[1], justification=self.claims(choice))
        self.note(f"nothing could be put in the cart: abstain, closest {choice}")
        return act("place_test_order", product_id=choice[0], sku=choice[1], abstain=True, justification=self.shortfall(choice))

    def nearest_now(self, key: tuple[str, str], read: bool = True) -> tuple[str, str] | None:
        """Re-read the named nearest listing and return the one to order or to name as nearest."""
        counter = self.counter
        if read and counter.has("inspect_stock") and counter.turns_left() >= 2:
            counter.send([act("inspect_stock", product_id=key[0], sku=key[1])])
            if counter.heard and self.hear()[0]:
                self.judge()
        left = [k for k in self.ranked() if k not in self.gone]
        if left and max(self.trust(k) for k in left[:12]) < LIKELY_FIT and any(self.trust(k) >= SURE_FIT for k in self.only_over()):
            left = []       # only loose matches within budget: a clearly matching over-budget listing wins
        if left:
            self.named_nearest = None
            if left[0] not in counter.cart and counter.turns_left() >= 2:
                counter.send([act("add_to_cart", product_id=left[0][0], sku=left[0][1])])
                self.hear_quietly()
            if left[0] in counter.cart:
                return left[0]
            alive = [k for k in counter.cart if k not in self.gone]
            return alive[-1] if alive else self.closest()
        over = self.only_over()
        if over:
            self.named_nearest = key if key in over else over[0]
            return self.named_nearest
        self.named_nearest = None
        alive = [k for k in counter.cart if k not in self.gone]
        return alive[-1] if alive else self.closest()

    def can_abstain(self) -> bool:
        spec = (self.counter.tools.get("place_test_order") or {}).get("parameters") or {}
        return "abstain" in (spec.get("properties") or {})

    def closest(self) -> tuple[str, str] | None:
        def nearness(key: tuple[str, str]) -> tuple:
            verdict = self.verdicts.get(key) or {}
            met = (verdict.get("met") or [])[: len(self.requirements)]
            stated = sum(1 for x, r in zip(met, self.requirements) if x and r.get("stated"))
            return (-int(bool(verdict.get("kind_ok"))), -stated, -sum(1 for x in met if x), self.contradicted(key), -self.said_in_part(key, met),
                    -int(key in self.counter.cart), (self.counter.stock.get(key) or {}).get("price") or 0.0)
        open_keys = [k for k in self.verdicts if self.on_sale(k)]
        return min(open_keys, key=nearness) if open_keys else None

    def contradicted(self, key: tuple[str, str]) -> int:
        """How many requirements the listing's own words state another value for; a variant that names no colour at all and
        carries no colour fact is silent on a colour, not against it."""
        page = self.counter.pages.get(key[0])
        if not page:
            return 0
        options = str(self.variant_of(key).get("options") or "")
        silent = not GLOSSARY.names_a_colour(options) and not any("colo" in str(label).lower() for label in page.get("facts") or {})
        return sum(1 for r in self.requirements if self.plain(key, r) < 0 and not (silent and self.kind(r) == "colour"))

    def said_in_part(self, key: tuple[str, str], met: list) -> int:
        """Count unmet requirements for which the property's fact holds part of the value's words."""
        page = self.counter.pages.get(key[0]) or {}
        count = 0
        for position, requirement in enumerate(self.requirements):
            if position < len(met) and met[position] or re.search(r"\d", requirement["value"]):
                continue
            wanted = {singular(w) for form in self.forms(requirement) for w in form.split()}
            for label, value in self.named_facts(page, requirement):
                named = self.name_words(label)
                words = [w for entry in self.entries(value) for w in entry.split() if not any(same_word(w, n) for n in named)]
                if words and all(any(same_word(w, v) for v in wanted) for w in words):
                    count += 1
                    break
        return count

    def on_sale(self, key: tuple[str, str]) -> bool:
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        return seen.get("in_stock") is not False and (self.budget is None or not isinstance(price, float) or price <= self.budget + 1e-9)

    @staticmethod
    def says_value(line: str, value: str) -> bool:
        """True when the line mentions the value: all its figures, or at least one telling word."""
        figures = figures_in(value)
        if figures:
            said = set(figures_in(line))
            return all(figure in said for figure in figures)
        said_words = stems(line) | set(word_list(lowered(line)))
        words = [w for w in word_list(lowered(value)) if len(w) >= 3]
        # for multi-word values the longer words are the distinguishing ones
        words = [w for w in words if len(w) >= 4] or words
        return not words or any(w in said_words or stems(w) & said_words for w in words)

    @staticmethod
    def says_amount(text: str, amount: float) -> bool:
        return any(abs(figure - amount) < 0.005 for figure in figures_in(text))

    def percent_of_limit(self, line: str, amount: float) -> bool:
        """True when the amount is the budget raised or lowered by a percentage stated in the line."""
        if self.budget is None:
            return False
        for share in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|per ?cent)", lowered(line)):
            for sign in (-1, 1):
                if abs(self.budget * (1 + sign * float(share) / 100) - amount) <= max(0.5, 0.002 * amount):
                    return True
        return False

    def holds_now(self, said: str) -> tuple[list[dict], bool]:
        """Requirements newly set by the lines being answered, and whether one set the budget."""
        lines = [line for line in list(self.fresh) + list(self.fresh_limit) if line and line in said]
        values = {lowered(v) for line in lines for v in self.fresh.get(line, [])}
        return [r for r in self.requirements if lowered(r["value"]) in values], any(line in self.fresh_limit for line in lines)

    def owed_news(self, said: str) -> list[dict]:
        """Requirements a reply must state; for a confirm request with no new ones, the values named."""
        news, _ = self.holds_now(said)
        if not news and re.search(r"\bconfirm|\bright\?|\bok(?:ay)?\?", lowered(said)):
            news = [r for r in self.requirements if self.says_value(said, r["value"])]
        return news

    def answer(self, line: str, beside: list[dict] | None = None) -> bool:
        """Reply to lines that changed a requirement; True when a message went out."""
        counter = self.counter
        if counter.over or not counter.has("message") or counter.turns_left() < 2:
            return False
        if line in self.unanswered:
            self.unanswered.remove(line)
        news = self.owed_news(line)
        if not news:
            if re.search(r"\bconfirm|\bright\?|\bok(?:ay)?\?", lowered(line)) and not counter.over:
                # a confirm request with no detected change still gets a yes
                counter.send([act("message", content="Yes, that is right, confirmed.")] + list(beside or []))
                self.reply_sent = (counter.turn, line)
                return True
            return False
        parts = []
        for requirement in news:
            before = self.took_back.get(lowered(requirement["value"]))
            parts.append(f"the {requirement['property']} is now {requirement['value']}" + (f" instead of {before}" if before else ""))
        counter.send([act("message", content=("Yes, confirmed: " + ", and ".join(parts) + ". Nothing else changes, your budget included.")[:400])]
                     + list(beside or []))
        self.reply_sent = (counter.turn, line)
        return True

    def run(self) -> list[dict]:
        counter = self.counter
        question = self.read_request()
        self.opening, self.opening_budget = [dict(r) for r in self.requirements if r.get("stated")], self.budget
        first: list[dict] = []
        if question and counter.has("message"):
            first.append(self.put(question))
        if self.listed and counter.has("compare"):
            first.append(act("compare", product_ids=list(dict.fromkeys(k[0] for k in self.listed))[:OPEN_BATCH]))
        filters = [act("filter", category=guess, max_price=self.budget, in_stock=True, k=10) for guess in self.label_guesses()[:5]] if counter.has("filter") else []
        first.extend(self.next_searches(counter.group_size - len(first) - len(filters)))
        counter.send(first + filters)
        self.listen()
        if self.pair_again and not self.pair_chosen and counter.has("message") and not counter.over:
            again = [self.put(self.pair_question)] + self.next_searches(counter.group_size - 1)
            counter.send(again)
            self.listen()
        if self.pair_fallback and not self.pair_chosen:
            # the pair question led nowhere: set-aside wishes count as requirements again
            self.requirements += self.set_aside
            self.opening += [dict(r) for r in self.set_aside if r.get("stated")]
            self.set_aside, self.pair_fallback = [], ""
            self.add_queries(self.value_queries(), first=True)
        try:
            self.find_out()
        except Exception as exc:    # a failed question step must not block the order
            self.note(f"asking stopped: {type(exc).__name__}: {exc}")
            self.awaiting_answer = False
        self.look(rounds=9)
        try:
            self.hunt_best()
        except Exception as exc:    # a failed search for better must not block the order
            self.note(f"search for the best stopped: {type(exc).__name__}: {exc}")
        commits = 0
        choice: tuple[str, str] | None = None
        while not counter.over and counter.turns_left() >= 2 and commits < 5:
            if self.hunt_stale and counter.cart and not self.late():
                try:
                    self.hunt_best()
                except Exception as exc:    # a failed search for better must not block the order
                    self.note(f"search for the best stopped: {type(exc).__name__}: {exc}")
            self.read_pending()
            flushed = self.glossary.flushes
            self.learn_words(flush=True)
            if self.glossary.flushes != flushed:
                self.judge()
            self.rank_by_priority(every=True)
            open_keys = [k for k in self.ranked() if k not in self.gone]
            weak = (bool(open_keys) and not self.search_asks and counter.turns_left() >= 14
                    and max(self.trust(k) for k in open_keys[:12]) < LIKELY_FIT)
            wanting = not open_keys or weak or not counter.cart and not any(self.keeps_request(k) for k in open_keys)
            if wanting and (self.rethink_searches() or (not open_keys and self.explore_more())):
                # nothing fits, or only loosely: search again first (by rule once the reader's own queries run out)
                continue
            if not open_keys and self.can_abstain():
                over = self.only_over()
                if not over and not self.over_noted:
                    self.over_noted = True
                    near = [k for k in self.verdicts if self.just_over(k)]
                    self.note(f"nothing fits within {self.budget} and no listing just over it does: moved_on={self.moved_on} brand={self.brand!r} over the limit: {len(near)} "
                              + "; ".join(f"{str((counter.pages.get(k[0]) or {}).get('title'))[:28]}|{self.variant_of(k).get('options')}|{(counter.stock.get(k) or {}).get('price')}|"
                                          f"{(self.verdicts[k].get('met') or [])}|{self.verdicts[k].get('kind_ok')}" for k in near[:8]))
                if over:
                    # nothing in budget meets everything but a pricier listing does: name it as nearest
                    choice = self.named_nearest = over[0]
                    title = str((counter.pages.get(choice[0]) or {}).get("title") or "")[:50]
                    self.note(f"nothing fits within {self.budget}: nearest {choice} {title!r} at {(counter.stock.get(choice) or {}).get('price')} "
                              f"turn={counter.turn} pages={len(counter.pages)}")
                    break
            order = open_keys or [k for k in self.near_misses() if k not in self.gone]
            if not order:
                break
            if not open_keys and self.added_after_cart:
                # nothing fits after a change that only added a requirement: the carted line, which still meets every earlier
                # requirement, stays ahead of a near miss that breaks one of them
                def keeps_rest(k: tuple[str, str]) -> bool:
                    return all(self.plain(k, r) >= 1 for r in self.requirements if lowered(r["property"]) not in self.added_after_cart)
                def buyable(k: tuple[str, str]) -> bool:
                    seen = self.counter.stock.get(k) or {}
                    price = seen.get("price")
                    return seen.get("in_stock") is not False and (self.budget is None or not isinstance(price, float) or price <= self.budget + 1e-9)
                carted = [k for k in self.counter.cart if buyable(k) and keeps_rest(k)]
                if carted and order[0] not in carted and not keeps_rest(order[0]):
                    self.note(f"added-only change: carted {carted[0]} kept over near miss {order[0]}")
                    order = [carted[0]] + [k for k in order if k != carted[0]]
            if not counter.cart:
                # the first cart favours the shopper's chosen values, then the request as written
                order.sort(key=lambda k: (not self.meets_chosen(k), not self.keeps_request(k)))
            choice = order[0]
            kept = self.cart_keeper()
            if kept is not None and kept != choice and not self.worth_swap(choice, kept):
                # every carted variant left unordered is charged unless a change broke it: a still-firm cart stays
                self.note(f"cart kept: {kept} over {choice}")
                choice = kept
            commits += 1
            title = str((counter.pages.get(choice[0]) or {}).get("title") or "")[:50]
            fitting = f"{len(self.ranked())}: " + "; ".join(f"{str((counter.pages.get(k[0]) or {}).get('title') or '')[:28]}|{self.variant_of(k).get('options')}|{[self.plain(k, r) for r in self.requirements]}|{(self.verdicts.get(k) or {}).get('by')}" for k in self.ranked()[:4])
            self.note(f"commit {commits}: {choice} {title!r} fits={self.fits(choice)} turn={counter.turn} pages={len(counter.pages)} spent={self.reader.spent:.4f} calls={self.reader.calls} fitting: {fitting}")
            group = []
            if choice not in counter.cart:
                group.append(act("add_to_cart", product_id=choice[0], sku=choice[1]))
            if counter.has("inspect_stock"):
                group.append(act("inspect_stock", product_id=choice[0], sku=choice[1]))
            if group and group[0]["name"] == "add_to_cart":
                group += self.stock_reads(counter.group_size - len(group), choice, adding=True)
            counter.send(group)
            seen = counter.stock.get(choice) or {}
            price = seen.get("price")
            over_budget = self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9
            held = [(r["property"], r["value"]) for r in self.requirements]
            changed, asks = self.hear()
            rewritten = [(r["property"], r["value"]) for r in self.requirements] != held
            if seen.get("in_stock") is False or over_budget:
                self.gone.add(choice)
                self.lost_titles.add(lowered((counter.pages.get(choice[0]) or {}).get("title")))
                self.owed_reaction = True
                self.pending_line = ""
                if counter.turns_left() > TURN_RESERVE and not self.late():
                    # a lost listing frequently has a twin elsewhere on the shelf
                    self.refill(choice, wide=not any(self.firm(k) for k in self.ranked() if k not in self.gone))
                left = [k for k in self.ranked() if k not in self.gone]
                if len([k for k in left if self.firm(k)]) < 2 and counter.turns_left() > TURN_RESERVE and not self.late():
                    # too few firm listings left: widen the search before another sell-out strands the order
                    self.kin_done.clear()
                    self.add_queries(self.kin_queries() + self.shelf_queries(6), first=True)
                    self.look(rounds=3 if left else 5, enough=2 if left else 1)
                choice = None
                continue
            if changed:
                self.judge()
                if not self.fits(choice):
                    self.gone.add(choice)
                    if counter.turns_left() > CHANGE_RESERVE and not self.late():
                        replying = bool(asks and self.pending_line and counter.has("message") and counter.turns_left() > CHANGE_RESERVE + 1)
                        fresh = self.next_searches(counter.group_size - 1 if replying else counter.group_size)
                        sent = False
                        if replying:
                            # the reply rides along with these searches to spare a turn
                            line, self.pending_line = self.pending_line, ""
                            sent = self.answer(line, beside=fresh)
                        if fresh and not sent:
                            counter.send(fresh)
                        if fresh or sent:
                            self.hear_quietly()
                        self.read_rows(choice)
                        self.judge()
                    if counter.turns_left() > TURN_RESERVE + 2 and not self.late():
                        # also search by rule: the best fit for the new values may not surface from their own queries
                        self.explored = 0
                        self.explore_more()
                    if not [k for k in self.ranked() if k not in self.gone] and counter.turns_left() > CHANGE_RESERVE and not self.late():
                        # a changed want is usually met by another variant or a relisting of the same article
                        self.refill(choice, wide=False)
                    if not [k for k in self.ranked() if k not in self.gone] and counter.turns_left() > CHANGE_RESERVE and not self.late():
                        self.look(rounds=4, enough=1)
                    if [k for k in self.ranked() if k not in self.gone]:
                        self.dig_after_change()
                    choice = None
                    continue
                if rewritten and not self.firm(choice) and not self.late():
                    self.dig_after_change()
                    top = next((k for k in self.ranked() if k not in self.gone), None)
                    if top is not None and top != choice and self.surer(top, choice):
                        choice = None
                        continue
            if asks and self.pending_line:
                self.answer(self.pending_line)
                self.pending_line = ""
                more_changed, _ = self.hear()
                if more_changed:
                    continue
            if counter.turn > int((counter.stock.get(choice) or {}).get("turn") or 0) and counter.has("inspect_stock") and counter.turns_left() >= 3:
                # price and stock may have moved since the last read
                counter.send([act("inspect_stock", product_id=choice[0], sku=choice[1])])
                late_change, _ = self.hear()
                seen = counter.stock.get(choice) or {}
                if seen.get("in_stock") is False or (self.budget is not None and isinstance(seen.get("price"), float) and seen["price"] > self.budget + 1e-9):
                    self.gone.add(choice)
                    self.lost_titles.add(lowered((counter.pages.get(choice[0]) or {}).get("title")))
                    self.owed_reaction = True
                    choice = None
                    continue
                if late_change:
                    self.judge()
                    if not self.fits(choice):
                        self.gone.add(choice)
                        choice = None
                        continue
            if self.unanswered and counter.has("message") and counter.turns_left() >= 3:
                self.reply_before_paying(choice)
                if self.hear()[0]:
                    self.judge()
                    if not self.fits(choice):
                        self.gone.add(choice)
                        choice = None
                        continue
            break
        if choice is None and not counter.over and self.can_abstain() and not [k for k in self.ranked() if k not in self.gone]:
            over = self.only_over()
            if over:
                choice = self.named_nearest = over[0]
        if choice is None and counter.cart and not counter.over:
            alive = [k for k in counter.cart if k not in self.gone]
            choice = alive[-1] if alive else None
        if choice is None and not counter.over:
            # no choice left: order the closest listing on sale, or name it as closest where abstaining is allowed
            self.judge()
            nearest = self.closest()
            if nearest is not None and nearest not in counter.cart and counter.turns_left() >= 2:
                counter.send([act("add_to_cart", product_id=nearest[0], sku=nearest[1])])
                self.hear_quietly()
            if nearest is not None and (nearest in counter.cart or self.can_abstain()):
                choice = nearest
        for _ in range(3):
            if choice is None or counter.over or counter.turns_left() < 1:
                break
            if self.unanswered and counter.has("message") and counter.turns_left() >= 2:
                self.reply_before_paying(choice)
            if choice == self.named_nearest:
                choice = self.nearest_now(choice)
                if choice is None:
                    break
            else:
                self.look_again(choice)
            if counter.heard:
                if self.hear()[0]:
                    self.judge()
                    if choice == self.named_nearest:
                        choice = self.nearest_now(choice, read=False)
                        if choice is None:
                            break
                    elif not self.fits(choice):
                        self.gone.add(choice)
                        choice = self.recart() or choice
            observations = counter.send([self.final_action(choice)])
            if counter.over:
                break
            # a refused order (a pending shopper line, say) means: read the line in, then retry
            changed, asks = self.hear()
            if asks and self.pending_line and counter.turns_left() >= 2:
                self.answer(self.pending_line)
                self.pending_line = ""
                changed = self.hear()[0] or changed
            if changed and counter.turns_left() >= 3:
                self.judge()
                if choice == self.named_nearest:
                    choice = self.nearest_now(choice, read=False)
                elif not self.fits(choice):
                    self.gone.add(choice)
                    choice = self.recart()
            if not observations:
                break
        return counter.dialogue


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    errand = Errand(problem_data)
    try:
        return errand.run()
    except Exception as exc:  # still return the dialogue, after one last order attempt
        errand.counter.dialogue.append({"role": "assistant", "content": f"agent error: {type(exc).__name__}: {exc}"})
        try:
            errand.last_resort()
        except Exception as late:
            errand.counter.dialogue.append({"role": "assistant", "content": f"agent error at the last order: {type(late).__name__}: {late}"})
        return errand.counter.dialogue


__all__ = ["agent_main"]
