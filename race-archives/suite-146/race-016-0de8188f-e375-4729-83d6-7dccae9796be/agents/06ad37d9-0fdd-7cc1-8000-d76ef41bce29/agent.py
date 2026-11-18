"""Shopping agent for shopper situations: one shopper, one order.

The shopper states some requirements, keeps others until asked, and the market or the shopper may
change while the agent shops. The agent reads the request once, finds out what the request leaves
open by asking about one property of the item at a time, searches the catalogue, reads the listings
it might buy, puts its best match in the cart, checks what the cart did to price and stock, follows
any change, and orders - or, when no listing meets everything, says so and names the closest one.

How the open point is found: a shopper who keeps a requirement back gives it only to a question
about that property. So the agent asks property by property - first the ones the request points at,
then the ones the store's own listings of that kind differ in - until the shopper names a
requirement for each purpose the request gives, and it stops asking as soon as that is done.

A language model reads the request, the shopper's lines and the listing pages; the code keeps the
episode in order (turns, questions, searches, the cart, stock checks, the final order).
"""

from __future__ import annotations

import html
import json
import re
import threading
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from time import monotonic
from typing import Any

from src.agent.proxy_client import ProxyClient

# ----------------------------------------------------------------------------- settings
MODELS = ["deepseek/deepseek-v4.1-flash", "z-ai/glm-5.2"]     # the first reads; the second takes over a call the first failed
SPEND_CAP_USD = 0.05          # per episode; the order is still placed when the cap is reached
CALLS_PER_TURN = 16           # tool calls sent in one turn (at most what the task allows)
OPEN_BATCH = 5                # listings per compare call
OPEN_LIMIT = 250              # listings read in full over the episode
SEARCH_LIMIT = 140            # searches and shelf filters sent while looking (the turns, not the searches, run out last)
JUDGE_CARDS = 18              # listings shown to the reader in one call
JUDGE_LIMIT = 120             # listings the reader is asked about over the episode
EPISODE_SECONDS = 150.0       # after this the agent stops looking and orders what it has
EPISODE_BUDGET_S = 240        # hard budget for the whole episode: a model call that would run past it is given up
CALL_LIMIT_S = 95.0           # the longest one model call is waited for
TURN_RESERVE = 6              # turns kept for the cart, the stock checks and the order
OVER_REACH = 1.35             # a listing this far over the spending limit is still read: it may be the nearest when nothing fits
OPEN_REACH = 1.6              # a search row this far over the limit is still opened: the row shows one variant, another may cost less
QUERY_LENGTH = 200            # a search is cut to this many characters
THINK_TOKENS = 3000           # room for a thinking pass before the tool call is written
MIN_TOKENS = 900              # every call: a provider may think before it writes the tool call, even when asked not to
SAID_NEEDED = r"(?:(?:a|an|is|be|it|the|must|should|has|have|to|needs?)\s+)*(?:required|requirement|needed|necessary|essential|must|need|included|present|chosen|selected|provided)"
NOT_A_PROPERTY = re.compile(r"categor|budget|price|\bstock|availab")     # terms of the purchase, not properties of the item
PURCHASE_TERMS = re.compile(r"categor|budget|price|\bstock\b|availab")     # the same, as whole words: a longer word that only contains one of them is no purchase term
CODE_TAIL = r"(?![a-z])(?!\d(?!\d*(?:\.\d+)?\s?[a-z]+(?![a-z0-9-])))"      # after a figure's unit: not a code running on
DOUBT_WORDS = r"\b(?:sure|doubt\w*|certain\w*|maybe|perhaps|possibl\w*|both|either|whether|might)\b"     # a clause that doubts the request can be met
PLAIN_QUESTION = "What will you use it for?"
# A question is about ONE property of the item, named alone and in the singular. The shopper's reader tells a
# requirement only when it is plain that one thing is asked about: two properties in a question, the use tied to
# the property, or a name in the plural leave that in doubt.
USE_WORDS = "what you will use it for"
ASK_ALONE = "What {} do you need?"
# Before any property, once for each purpose the request gives: what the item must have for that use. A shopper
# holds a requirement back for a purpose, and gives it when asked what that purpose needs.
ASK_BY_USE = "You want {it} for {purpose}. Used that way, what do you need it to have?"
ASK_BY_USE_AGAIN = "Thinking of {purpose}: what should {it} be like?"
ASK_WIDE = 4                  # properties named in one question
ASK_AFTER_PAIR = 5            # questions about properties after the shopper has chosen between two wishes
NO_BUY_HOLD = 2               # turns the next question waits, once one of two doubted wishes is chosen, while only listings over the limit meet the request
ASK_AFTER_USE = 4             # questions about properties in all once a question about a purpose was answered with no requirement
ASK_QUIET = 5                 # questions in all while the shopper has told no held-back requirement
ASK_AFTER_BOTH = 4            # questions about properties once one requirement is out and every purpose was asked about
ASK_NARROW = 3                # questions that go over properties asked about three or four at a time, two at a time
# (a purpose in the shopper's words, said back to the shopper)
TO_SHOPPER = {"i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself", "we": "you", "us": "you", "our": "your",
              "i'm": "you're", "i've": "you've", "i'll": "you'll", "i'd": "you'd", "am": "are"}
ASK_LIMIT = 10                # questions put to the shopper over an episode
ASK_MORE = 6                  # further questions once one held-back requirement is out and another purpose remains
ASK_TURNS = 10                # turns kept free of questions at the end, for the cart, the changes and the order
ASK_SHELF = 14                # properties of the shelf shown to the reader when it chooses what to ask about
BEST_ROUNDS = 3               # rounds of searching for a listing that stands better by the shopper's priority
FIGURE_LIMIT = 150            # listings whose figure for the priority is read over the episode
# how far a listing is trusted to fit: every value stated where the store states that kind of property (the variant's
# options, the property's own fact) / every value stated plainly somewhere / in the listing's own words / only in other words
FIRM_FIT, SURE_FIT, LIKELY_FIT, UNSURE_FIT, FAINT_FIT = 1.0, 0.85, 0.5, 0.3, 0.1
LONG_LIST = 6                 # values a page fact lists beyond which it states none of them in particular

_proxy = ProxyClient(timeout=90, max_retries=2)


# ----------------------------------------------------------------------------- small helpers
_NORMALIZED: dict[str, str] = {}      # (each text is read many times over: it is put in plain form once)


def normalized_lower(text: Any) -> str:
    text = str(text or "")
    known = _NORMALIZED.get(text)
    if known is not None:
        return known
    raw = text
    if not text.isascii():
        text = unicodedata.normalize("NFKC", text)      # (full-width letters, digits and brackets written as plain ones)
    # (the double-prime sign after a figure is a way of writing the inch: 22'' is 22 inch)
    plain = " ".join(re.sub(r"(?<=\d)\s?(?:''|\u2033|\u201d|\")(?=[\s,;:/)\]x\u00d7*-]|$)", " inch", text.lower()).split())
    if len(_NORMALIZED) < 250000:
        _NORMALIZED[raw] = plain
    return plain


_FORMS: dict[tuple, frozenset] = {}     # (a requirement's spellings, worked out once for each way it is written)


def letter_digit_tokens(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalized_lower(text))


def singular_form(word: str) -> str:
    """One form for a word and its plural (also for words written with -ie or -y)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 4 and word.endswith("ie"):
        return word[:-2] + "y"
    if len(word) > 4 and word.endswith(("ve", "fe")):
        return word[:-2] + "f"      # (a plural in -ves and its singular in -f or -fe are one word; so is any word in -ve)
    return word


def join_number_unit(text: Any) -> str:
    """Amounts written as listings write them: the number and its unit as one word (3 cm -> 3cm)."""
    return re.sub(r"(\d)\s+(?=[a-z]{1,4}\b)", r"\1", normalized_lower(text))


def same_word_stem(one: str, other: str) -> bool:
    """Two words are the same word when they differ only by an ending (dot / dotted)."""
    if one == other:
        return True
    short, long = sorted((one, other), key=len)
    return len(short) >= 4 and not short[0].isdigit() and long.startswith(short) and len(long) - len(short) <= 2


def stem_set(text: Any) -> set[str]:
    text = str(text or "")
    known = _STEM_CACHE.get(text)
    if known is None:
        known = frozenset(singular_form(w) for w in letter_digit_tokens(join_number_unit(text)))
        if len(_STEM_CACHE) < 250000:
            _STEM_CACHE[text] = known
    return set(known)


_STEM_CACHE: dict[str, frozenset] = {}


def singular_phrase_key(text: Any) -> str:
    """A category label reduced to its words in singular form, for comparing two spellings of it."""
    return _singular_phrase_key(str(text))


@lru_cache(maxsize=65536)
def _singular_phrase_key(text: str) -> str:
    return " ".join(singular_form(w) for w in letter_digit_tokens(text.replace("&", " and ")) if w != "and")


def positive_amount(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def make_tool_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


def string_array_schema(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def make_tool_call(name: str, **args: Any) -> dict:
    return {"name": name, "args": {k: v for k, v in args.items() if v is not None}}


# ----------------------------------------------------------------------------- what the reader fills in
PLAN_TOOL = make_tool_schema("plan", "The shopper's request, written down for shopping.", {
    "item": {"type": "string", "description": "The shopper's own words for the kind of thing, copied from the request exactly as written, without the article and without the requirements."},
    "category": {"type": "string", "description": "A store category, only when the shopper names one as a category; otherwise empty."},
    "requirements": {"type": "array", "description": "Each property the item itself must have, stated in the request.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string", "description": "The name of the property in the shopper's words (colour, material, pattern, style, capacity, ...)."},
            "value": {"type": "string", "description": "The required value, as the shopper wrote it, with its unit."},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The same value written another way: another spelling of the word, or the same amount in another unit. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value (a synonym; not a part or a wider kind of it). Empty when there is none."}},
        "required": ["property", "value"]}},
    "budget": {"type": "number", "description": "The most the shopper will pay; 0 when none is named."},
    "brand": {"type": "string", "description": "A brand the shopper requires; empty when none, and empty when a brand is only preferred or a first choice."},
    "brand_wish": {"type": "string", "description": "A brand the shopper only prefers or names as a first choice (not required); empty otherwise."},
    "priority": {"type": "string", "description": "What the shopper wants the most or the least of among the items that fit, in the shopper's words (for example the lowest price). Most requests name one: read the request again before leaving this empty."},
    "queries": string_array_schema("Exactly six short catalogue searches (two to five words) that could return the item: the item words with the required values, with a synonym or another spelling of a value where one exists."),
    "priority_queries": string_array_schema("When a priority is given: up to four more searches that reach the listings standing best by it; otherwise empty."),
    "doubted_pair": {"type": "array", "items": {"type": "string"}, "maxItems": 2,
                     "description": "When the shopper wishes for two things and doubts one item can have both: the two wishes, each as its value with what it describes (for example 'a leather strap', 'the Acme brand'). Empty otherwise."},
    "purposes": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                 "description": "Each use or recipient the request gives for the item, one entry each, in the shopper's words. Empty when it gives none."},
    "invitation": {"type": "string", "description": "The request's exact words that tell the assistant to put a question to the shopper before deciding (to ask, check or confirm with the shopper whether the purpose changes what the item needs); empty when the request holds no such instruction."},
    "look_at": {"type": "integer", "description": "When the request tells the assistant to open, compare or check at least some number of fitting options before choosing: that number; otherwise 0."},
    "open_properties": {"type": "array", "items": {"type": "string"}, "maxItems": 6,
                        "description": "Properties of this kind of item that the request does not settle and a shopper could still have a requirement about, the likeliest first, each in one to three plain words."},
    "use_properties": {"type": "array", "items": {"type": "string"}, "maxItems": 6,
                       "description": "The measures a stated purpose puts a figure on, every measure of its kind in the order the guide gives (a requirement the request states left out), each one measure in one to three plain words; empty when no purpose does."},
    "cite": {"type": "string", "description": "The property of the item (besides its price and stock) that the request asks the order's reasons to point to on the listing, in the request's words (the colour, the voltage); empty when it asks for none."},
    "settled": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                "description": "The purposes, copied from purposes, whose item was already decided outside this request (by a list or plan someone made, by a person's own precise wishes, by an item it must match or stand in for, by the shopper's usual purchase); empty when none is."},
    "priority_measure": {"type": "string", "description": "The measure the priority compares, named as a property of the item in one to three plain words (the most pieces: piece count; the longest one: length; the largest volume: volume); empty when price decides, or nothing does."},
    "priority_units": string_array_schema("The units listings write the priority's measure in, shortest forms first (for a length: mm, cm, m); empty when price decides, the measure is a count of pieces, or nothing does."),
}, ["item", "requirements", "budget", "priority", "priority_measure", "queries", "doubted_pair", "purposes", "invitation", "look_at", "open_properties", "use_properties"])

HEAR_TOOL = make_tool_schema("hear", "What the shopper's new line changes.", {
    "adds": {"type": "array", "description": "Each requirement of the item the line states that was not known before.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string", "description": "The property's name as the shopper said it, with every word of that name (the material of one part is named with the part)."},
            "value": {"type": "string", "description": "The value alone, as the shopper said it: an amount with its unit; never the property's own name, and never a word that only says it is needed."},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The same value written another way: another spelling of the word, or the same amount in another unit. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value (a synonym; not a part or a wider kind of it). Empty when there is none."}},
        "required": ["property", "value"]}},
    "drops": string_array_schema("Properties whose earlier requirement the line withdraws or replaces."),
    "budget": {"type": "number", "description": "A new spending limit the line states; 0 when it states none."},
    "priority": {"type": "string", "description": "What the line says matters most between fitting items; empty when nothing."},
    "asks": {"type": "boolean", "description": "True when the line asks the assistant something or expects an answer."},
    "about": {"type": "string", "description": "Only when the assistant had just asked about several properties and the line gives a value: the one of them that value belongs to, copied exactly as the assistant named it. Empty when it belongs to none of them or nothing was asked."},
    "about_change": {"type": "boolean", "description": "True when the line only reacts to the stock or the price of the chosen item having changed."},
    "queries": string_array_schema("Up to six short catalogue searches that use what the line added (try a synonym or another spelling of the value too); empty when it added nothing."),
}, ["adds", "drops", "budget", "asks", "queries"])

ASK_TOOL = make_tool_schema("ask_about", "The properties of the item worth asking the shopper about.", {
    "properties": {"type": "array", "items": {"type": "string"}, "maxItems": 8,
                   "description": "The properties to ask about next, the likeliest to carry a requirement first, each named in one to three plain words as a shopper would say it."},
    "measures": string_array_schema("One entry for every unit shown under 'Amounts': the unit as shown, a colon, and the name of what it measures - a noun of one or two "
                        "words, as a shopper would name that property of the item (for a unit of mass: weight) - or 'none' when it is no unit of measure of "
                        "the item itself (part of a code, a word that only follows a number, a span of time). Put the entries for how many and how much "
                        "first, and those for how long or how big last.")},
    ["properties", "measures"])

JUDGE_TOOL = make_tool_schema("judge", "Which listings state every requirement.", {
    "marks": string_array_schema("One short entry for every listing shown, in the order shown: the listing's number, then K when it is the kind of thing asked for or X when it is not, "
                     "then one letter per requirement in order - Y when the listing's own words state that value, N when they do not. Example for two requirements: '3 K YN'."),
    "proof": {"type": "array", "description": "Only for listings marked K with every requirement Y: where each requirement is stated.", "items": {
        "type": "object", "properties": {
            "n": {"type": "integer", "description": "The listing's number."},
            "where": string_array_schema("For each requirement in order: title, options, or the fact's label exactly as written."),
            "words": string_array_schema("For each requirement in order: the listing's exact words that state it."),
            "rank_value": {"type": "number", "description": "The listing's number for the shopper's priority when one is given, else 0."}},
        "required": ["n", "where", "words"]}},
    "queries": string_array_schema("When no listing meets every requirement: up to five other short searches worth trying; otherwise empty."),
}, ["marks"])

NAMES_TOOL = make_tool_schema("names", "What each text names for one property.", {
    "marks": {"type": "array", "items": {"type": "string"}, "description": "One entry per text, in order: '<number> <only|among|other|none>'."}},
    ["marks"])

WORDS_TOOL = make_tool_schema("words", "Other words sellers use for a value.", {
    "words": {"type": "array", "items": {"type": "string"}, "description": "Up to six, the most usual first; empty when there is none."}},
    ["words"])

CHECK_GUIDE = """A shopping assistant is about to order an item for a shopper. You check the candidate listings against everything the shopper wants now. You are given the record of the shopper's requirements (numbered), the spending limit, what the shopper said makes one item better, the whole conversation between the shopper and the assistant, and the candidate listings with the title, the options of the exact variant sold, the price and the facts.
- Read the conversation yourself: a need the shopper told in an answer, a requirement they changed and a limit they changed count as they stand at the end of the conversation, even where the record misses them or words them otherwise. A later line replaces an earlier one on the same point.
- For each listing say Y when its own words (the options of the variant first, then its facts, then its title) show that it meets every requirement now in force; N when its words show that it misses one, naming that requirement (its number in the record, or 0 for a need told in the conversation that the record lacks); ? when its words leave a requirement unsettled.
- Say N only on a clear miss: a value that differs, a property the item plainly lacks, another kind of item than the one wanted. A value written in other words, units or spelling that means the same is a match; a priority is not a requirement."""

CHECK_TOOL = make_tool_schema("check", "Whether each listing meets every requirement of the shopper now in force.", {
    "verdicts": {"type": "array", "description": "One entry per listing, in order.", "items": {"type": "object", "properties": {
        "number": {"type": "integer", "description": "The listing's number."},
        "meets": {"type": "string", "enum": ["Y", "N", "?"], "description": "Y meets every requirement, N misses one, ? unsettled."},
        "violated": {"type": "integer", "description": "For N: the number of the requirement missed in the record, or 0 for a need told in the conversation that the record lacks."},
        "why": {"type": "string", "description": "A few words: what the listing states against what is wanted."}},
        "required": ["number", "meets"]}}},
    ["verdicts"])

RANK_TOOL = make_tool_schema("rank", "Each listing's figure for the shopper's priority.", {
    "values": {"type": "array", "items": {"type": "string"}, "description": "One entry per listing, in order: '<number> <figure>' with the figure as a plain number in one unit for every listing, or '<number> none' when the listing's words state none."},
    "better": {"type": "string", "enum": ["higher", "lower"], "description": "Whether a higher or a lower figure is better for this priority."},
    "unit": {"type": "string", "description": "The one unit every figure is written in (empty for a plain count)."}},
    ["values", "better", "unit"])

BEST_TOOL = make_tool_schema("search_better", "Catalogue searches for listings that stand better by the shopper's priority.", {
    "queries": string_array_schema("Up to eight new searches of two to six words each, different from those tried."),
}, ["queries"])

SEARCH_TOOL = make_tool_schema("search_plan", "New catalogue searches.", {
    "queries": string_array_schema("Six new searches of two to five words each, different from those tried."),
}, ["queries"])

PLAN_GUIDE = """You prepare a purchase for a shopping assistant. Read the shopper's request and fill the plan tool from its words only.
- item: the words the shopper uses for the kind of thing, copied completely and exactly as written, with a leading word such as 'other' and any '&' or '/' kept (the store files its listings under such names); do not shorten them or replace them with a broader or narrower kind.
- requirements: every detail the request gives about the item itself is a requirement: its material, its colour, an amount with its unit (how much it holds, how big, heavy or long it is, how many pieces, how strong), its pattern, shape, style, kind or type, an occasion or use named together with the item, the make or thing the item must fit, and the like. A kind or type named for the item ('of the ... type') is such a detail too. A request for one kind of thing nearly always carries such a detail: read the request again before leaving the list empty. Name each property the way the shopper names it; when the request gives only the value, use the plain name of that kind of detail. Give the value in the fewest words: the word or amount itself (an amount with its unit in letters), without the property's name and without the kind of thing it belongs to. One phrase of the request gives one requirement; do not split it into two. The budget, being in stock and the category are not requirements, and neither is why the shopper is buying, which comes in a sentence of its own.
- doubted_pair: a request sometimes names two things the shopper would like on top of what the item must have - usually a value of some property and a brand - and doubts, or leaves open, that one item offers both, in any wording (a doubt said outright, or the second wish added as something further that may not be possible); the shopper will say which matters more when asked. Give the two wishes. Leave it empty when the request names fewer than two such wishes: a single thing the shopper would merely like (one brand that is only preferred) is neither a requirement nor a pair, and two words of the item's own name are not two wishes.
- purposes: every use or recipient the request gives (what the item is for, whom it is for); each can carry a need the shopper has not stated yet. A request often gives two - in two sentences, or in one sentence joined by a linking word, the second sometimes only as a further clause (it is for one thing 'and also' has to suit another): give each as its own entry, in the shopper's own words for it and without the linking words, each as a phrase that can follow the word 'for' (a camping trip; a child's room; a friend who is moving house). One purpose is one entry: do not cut a single purpose in two.
- invitation: some requests tell the assistant outright to put a question to the shopper before deciding - to ask, check or confirm with the shopper whether the purpose changes what the item needs. Copy those words exactly as written. A sentence that only gives the purpose, says the purpose may matter, or tells the assistant what the item must be (in stock, within the budget) holds no such instruction: leave it empty.
- priority: what the shopper wants the most or the least of among the items that fit - the most or the least of some quantity the items differ in, or the lowest price - in the shopper's words. Most requests name one, in a sentence of its own or beside the budget: do not leave it out.
- brand: only a brand the shopper requires. A brand that is only preferred, or wanted only when there is one, is not required: leave brand empty.
- brand_wish: a brand the shopper only prefers or names as a first choice; it goes here and never in brand.
- look_at: some requests tell the assistant to open, compare, check or go through at least a certain number of fitting options (listings, choices) before choosing. Give that number; 0 when the request sets none.
- cite: when the request asks the order's justification or reasons to point to what the listing shows, the property of the item it names there besides the price and the stock (the colour, the voltage, the pet type), in its words; empty when it asks for none.
- open_properties: what the request leaves open about the item itself, in the order to ask about it. A shopper who holds one requirement back holds back something plain, whatever the kind of item - do not judge whether it sounds usual for this item. When a stated purpose itself points to a property of the item (how it has to look, what it has to hold, weigh or work with for that purpose), that property comes first. After it the list always runs in this order, leaving out only what the request already fixes: the colour; then the material; then which type of the item it is, written as 'type of' and the item's main word or two (for a kettle: 'type of kettle'; never with 'and' or '&'); then the piece count (how many pieces come together); then the amounts this kind of item is sold by, each named by the measure itself and never by its unit. Nothing else goes ahead of these. Name each one in one to four plain words; never a loose word on its own that leaves open which measure or which kind is meant (a bare 'size' does). Leave out the quantity named as the priority, the brand, the budget, the category and stock.
- priority_queries: when the shopper names a priority, searches that reach the listings standing best by it (the item words with the kind of amount such listings write).
- use_properties: some purposes put a figure on the item - how many pieces come together, how much it weighs or holds, how big it may be, what it draws or runs on. For each such purpose give every measure of its kind below, in that order, each by the measure's own plain name in one to three words, never a description of it - leave out only a measure the request already states as a requirement:
  a stock that gets used up or has to last a while: piece count, weight, volume;
  something carried, posted or shipped: piece count, weight;
  something that has to fit a space: length, volume, width;
  something run hard, all day, or on a supply that can only give so much: power, capacity, voltage;
  something that has to work with chargers, sockets or devices already owned: voltage, plug type;
  something that makes or serves a lot at once: capacity, power.
  Give the whole list of the purpose's kind, also when the priority compares one of its measures (the most volume for a stock still gives piece count, weight, volume); the priority itself is never a requirement. A purpose about looks, taste, a gift, matching or replacing something, or following someone's list puts no figure on the item: give nothing for it (empty when every purpose is such).
- settled: of the purposes, each one whose item was decided before this request by someone or something else - a list or a plan someone made, a person's own precise wishes, an item the new one must match or stand in for, the shopper's usual purchase. The shopper knows what such a purpose fixes and says it when asked what the item is for. A purpose about looks or taste (a gift for someone's taste), or one that only puts a figure on the item (a stock that has to last, something shipped, a space, power, chargers and sockets), is not settled. Copy each as you wrote it in purposes; empty when none is.
- written: the same value written another way - another spelling of the word (grey: gray) or the same amount in another unit or wording (1 kg: 1000g, 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- queries: searches as a shopper types them, two to five words, built from the item words and the required values. Write an amount the way listings do (the number joined to its unit)."""

HEAR_GUIDE = """You follow a shopper's conversation with a shopping assistant. Given what is already known and the shopper's new line, fill the hear tool with what the line changes.
- adds: a requirement of the item the line states (a property with its value), the property named in the shopper's words and the value in the fewest words (the word or amount itself). A purpose alone is not a property; the property the purpose calls for is. The budget, the category and being in stock are not properties. written: the same value written another way (another spelling of the word; the same amount in another unit: 1 kg is 1000g or 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- When the line gives a value without naming its property, name the property that value belongs to by what the value itself is (a colour word is a colour, what a thing is made of is its material, an amount is the measure its unit tells).
- When the line says the item must have or be some thing (a part, a feature, a way it is powered or fitted, a form), the property is what kind of thing that is (power source, fitting, form, feature) and the value is the thing itself in the fewest words; a value never just repeats the property's name.
- A line that only adds a value - one more thing it must have or include, with the rest staying as it is - withdraws nothing: leave drops empty unless the line says an earlier value no longer holds. The shopper may answer about another property than the one the assistant had just asked about: use the asked property's name only when the value can be a value of that property.
- drops: a property whose earlier requirement the line replaces or withdraws.
- A new line can hold several sentences or lines: read them in order; about the same property, a later one overrides an earlier one.
- A line that puts one value in place of another, withdraws one, or adds one is a change of what the shopper wants, even when it is worded as a question or a confirmation (for example ending in 'right?'): drop the old property and add the new value.
- budget: only when the line names a new limit.
- asks: true when the line puts a question to the assistant or expects an answer; a remark that needs no answer is false.
- about_change: true when the line reacts to the chosen item's stock or price having just changed (the assistant answers that by choosing again). A line can do that and also change a requirement: then fill adds and drops too.
- queries: searches of two to five words that combine the item words with the value just added."""

ASK_GUIDE = """A shopper keeps a requirement of the item back and gives it only when the assistant asks about that one property by name. You choose which properties to ask about next, in order.
You are given the request, what is known so far, what was already asked and how the shopper answered, some of the store's listings that meet everything known so far, and the fact labels the store files listings of this kind under.
What shoppers hold back is plain, whatever the kind of item - do not judge whether a property sounds usual for this item. Build the list in exactly this order, leaving out only what is already known or already asked about:
1. the colour;
2. the material;
3. the fact labels of the item itself, in the order they are shown (the label most listings carry comes first), up to six, each named after its label in plain words (a label such as 'xyz_type' or 'types_of_xyz' is the 'xyz type'). Go down the shown list and skip only: a label about the seller or the offer rather than the item itself (who makes or sells it, its state, its guarantee, how it ships), a label that only holds the listing's own text or its layout, and a label that says the colour;
4. which type of the item it is, written as 'type of' and the item's main word or two (for a kettle: 'type of kettle'; never with 'and' or '&');
5. any other property of the item itself in which the listings differ.
Amounts do not go in this list. Fill measures instead: what each unit under 'Amounts' measures.
Never list anything about the seller or the offer rather than the item itself; the quantity the shopper named as the priority; the budget, the category or stock.
Name each property in one to three plain words, as a shopper would say it. One property per entry: never two joined together and never a name with 'and', 'or' or '&' in it (for an item named after two things, name the kind after one of them), never a loose word on its own that leaves open which measure or which kind is meant (a bare 'size' does) - say which measure or which kind.
List up to eight."""

JUDGE_GUIDE = """You check store listings against a shopper's requirements. For every listing shown, say whether it is the kind of thing asked for and, for each requirement in order, whether the listing's own words state that value.
- A requirement is met when the listing's own words state the value: the same word, a plain synonym or another spelling of it, or the same amount written in another unit.
- Where to look: a colour or a size is what the options say, because the options describe the exact variant sold (when the options name another colour or amount, the requirement is unmet whatever the title lists); a material or an amount is stated by the title, the options or a fact; any other property the store keeps as a fact of its own is stated by the fact whose label names that property.
- When a listing shows 'the fact named for requirement N', that fact decides requirement N: met when it states the value, unmet when it states something else - unless the options state the value.
- A fact that states the value settles it even when the title's words suggest otherwise. A value that is absent is unmet; do not assume.
- An amount counts when it is one of several measures written for the variant (a size written as 40*30*5cm states 5 cm; a range such as 4-6cm includes 5 cm). An amount filed under a fact for a different quantity does not state it.
- marks: one entry per listing, in order, as '<number> <K or X> <Y or N for each requirement>'.
- proof: only for the listings that are K with every requirement Y - the place (title, options, or the fact's label exactly as written) and the listing's exact words that state each value; rank_value is the listing's number for the shopper's priority when one is given.
Mark every listing, in order."""

NAMES_GUIDE = """You read what a listing's own words say about one property of the item (its colour, or the material it is made of). For each numbered text answer with one word:
- only: the words name the wanted value - the same word, or another word sellers use for that same value - and no other value of that property. A narrower kind of the wanted value (one particular kind of it) is the wanted value; a wider kind is not.
- among: the words name the wanted value together with another value of that property (two colours, two materials), so it is unclear which one the item is.
- other: the words name a value of that property, but not the wanted one.
A shade or tint of the wanted colour (dark, light, off-, or a describing word before the colour) is the wanted colour; a name that joins two different colours is another colour.
A word before the colour that is itself a colour's name counts as a shade only when, on its own, it names the same colour family as the wanted colour; when it names a different colour family, answer other.
A colour name of its own that sellers use for a shade within the wanted colour's family (a lighter, darker or particular kind of that colour, though it does not contain the colour's own word) names the wanted colour: answer only.
A colour word that describes another thing named beside it (an add-on, a part, or another thing sold together with the item) does not give the item's colour: answer none for that text.
The variant's words describe the exact item sold: a material they name is the item's material, whatever the title lists.
- none: the words name no value of that property.
Judge only the property asked about; sizes, codes, counts and other details are not values of it. Mark every text, in order."""

SHADES_GUIDE = """A shopper wants an item of a certain colour. List the names sellers give to shades of that colour in listing titles and variant names: other colour words for a shade of it, and the names of things of that colour (a fruit, a food, a stone, a material) used as colour names. Only names a shopper would take for that colour; not another colour and not a mix of two colours."""

WORDS_GUIDE = """A shopper wants a certain value for a property of an item. List the other words or spellings that sellers use in listing titles and variant names for that same value: synonyms and other spellings of a word, or the same amount in other units and notations. Only words that mean the same value; not a part, a shade or a wider kind of it."""

RANK_GUIDE = """A shopper said what makes one fitting item better than another. For each listing, read from its own words (the options first, then its facts, then its title) the figure that this priority is about, and write it as a plain number in the same unit for every listing (convert where needed). The figure must measure what the priority names: how much an item holds is not what it weighs, and its length is not its width - a listing that states only another measure states none. The options describe the exact variant sold: when they state the figure, that is the figure, whatever the title lists (a title often lists the amounts of all its variants). Write none when the listing states no such figure; do not guess. One entry per listing, each as the listing's number and then the figure."""

BEST_GUIDE = """A shopper wants, among the store listings that meet every requirement, the one that stands best by a priority of theirs (the most of something, the largest, the longest, ...). A shopping assistant has found some fitting listings. You write catalogue searches that could reach fitting listings that stand better still.
You are given the request, the requirements, the priority, the best figures found so far with the words of the listings that carry them, and the searches already tried.
- Listings state such a figure in their title or variant name. A search that carries a figure written the way a listing writes it brings up the listings that state it - and sellers write it both joined to its unit and apart from it, which the store treats as different words. So write searches made of the item words, the required values and a figure for the priority that is better than the best found so far: several steps beyond it, one figure per search, some written joined and some apart.
- A title that lists several amounts lists the variants sold under it: when such a title shows an amount better than the best found, search that title's main words with that amount and the required values.
- Add one search with the words sellers use for the better end of that priority.
- Keep every search short, and do not repeat a search already tried."""

SEARCH_GUIDE = """A shopping assistant has not yet found a store listing that meets every requirement of the shopper. Given the request, the requirements, the searches already tried and the titles of the listings seen, write new short catalogue searches that are likely to reach a listing meeting every requirement: combine the required values with each other and with the words such listings carry in their titles, use other words sellers use for a value, and try the values without the item words. Do not repeat a search already tried."""

# ----------------------------------------------------------------------------- inference
class ModelCaller:
    """Language-model calls with a fallback model and a spending guard."""

    def __init__(self) -> None:
        self.models = list(MODELS)
        self.spent = 0.0
        self.calls = 0
        self.started = monotonic()
        self.abandoned = 0

    def seconds_remaining(self) -> float:
        return EPISODE_BUDGET_S - (monotonic() - self.started)

    def post_with_deadline(self, body: dict) -> Any:
        """One model call, waited for at most what the episode can spare: a call that does not come back in time
        is left to finish on its own, and the episode goes on without it (the order must still be placed)."""
        bound = min(CALL_LIMIT_S, self.seconds_remaining() - 8.0)
        if bound < 4.0:
            return None
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                box["reply"] = _proxy.post("/inference/chat/completions", json_data=body)
            except Exception as exc:     # (a failed call is a missing answer, like a timeout)
                box["error"] = type(exc).__name__
        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(bound)
        if thread.is_alive():
            self.abandoned += 1
            return None
        return box.get("reply")

    def spend_limit_reached(self) -> bool:
        return self.spent >= SPEND_CAP_USD

    def request_tool_fill(self, guide: str, text: str, tool: dict, max_tokens: int, second: bool = False, think: bool = False) -> dict | None:
        name = tool["function"]["name"]
        for model in (self.models[1:] if second else self.models):
            body = {"model": model, "temperature": 0, "max_tokens": max(max_tokens, MIN_TOKENS) + (THINK_TOKENS if think else 0),
                    "messages": [{"role": "system", "content": guide}, {"role": "user", "content": text}],
                    "tools": [tool], "tool_choice": {"type": "function", "function": {"name": name}}}
            # a thinking pass spends the completion budget before the tool call is written
            if re.search(r"glm-5\.3-flash", model, re.I):
                body["reasoning"] = {"effort": "low"}       # (this endpoint answers only with some reasoning)
            else:
                body["reasoning"] = {"enabled": think} if re.search(r"glm|kimi|deepseek", model, re.I) else {"effort": "low"}
            reply = self.post_with_deadline(body)
            self.calls += 1
            if not isinstance(reply, dict):
                continue
            usage = reply.get("usage") or {}
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self.spent += float(cost)
            else:
                self.spent += (int(usage.get("prompt_tokens") or 0) + 4 * int(usage.get("completion_tokens") or 0)) * 1e-6
            parsed = self._parsed_tool_arguments(reply, name)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _closed_json_object(raw: str) -> dict | None:
        """Arguments cut off at the length limit: the longest prefix that ends with a complete value, closed."""
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
    def _unescape_markup(value: Any) -> Any:
        """The reader's strings with any web entity written as its character (a reader may write & as &amp;, and a
        search or a shelf name with the entity in it finds nothing)."""
        if isinstance(value, str):
            return html.unescape(value) if "&" in value else value
        if isinstance(value, list):
            return [ModelCaller._unescape_markup(v) for v in value]
        if isinstance(value, dict):
            return {k: ModelCaller._unescape_markup(v) for k, v in value.items()}
        return value

    @staticmethod
    def _parsed_tool_arguments(reply: dict, name: str) -> dict | None:
        try:
            message = reply["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        for call in message.get("tool_calls") or []:
            raw = (call.get("function") or {}).get("arguments")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                parsed = ModelCaller._closed_json_object(raw) if isinstance(raw, str) else None
            if isinstance(parsed, dict):
                return ModelCaller._unescape_markup(parsed)
        content = message.get("content")
        if isinstance(content, str) and "{" in content:
            try:
                parsed = json.loads(content[content.index("{"): content.rindex("}") + 1])
            except ValueError:
                return None
            if isinstance(parsed, dict):
                return ModelCaller._unescape_markup(parsed.get(name) if isinstance(parsed.get(name), dict) else parsed)
        return None


# ----------------------------------------------------------------------------- the store session
class StoreSession:
    """One environment session: turns, tool calls and what the store has shown so far."""

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
        self.rows: dict[tuple[str, str], dict] = {}      # (product_id, sku) -> search / filter row
        self.row_order: list[tuple[str, str]] = []
        self.shown_at: dict[tuple[str, str], int] = {}     # the turn each row was first shown at
        self.hits: dict[tuple[str, str], list] = {}        # row -> (search text, place in its results) for each search showing it
        self.shelf: dict[tuple[str, str], str] = {}      # rows a category filter returned -> the label that returned them
        self.pages: dict[str, dict] = {}                 # product_id -> page from view / compare
        self.stock: dict[tuple[str, str], dict] = {}     # (product_id, sku) -> latest price / stock seen
        self.cart: list[tuple[str, str]] = []
        self.adds = 0                                    # listings put in the cart so far (those taken out again too)
        self.heard: list[str] = []                       # shopper lines not yet handled

    def turns_remaining(self) -> int:
        return self.turn_limit - self.turn

    def tool_available(self, tool: str) -> bool:
        return tool in self.tools

    def send_environment_turn(self, actions: list[dict]) -> list[dict]:
        """One turn: the actions in order. Returns one observation per action sent."""
        actions = [a for a in actions if a["name"] in self.tools][: self.group_size]
        if not actions or self.over or self.turns_remaining() <= 0:
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
            self._record_observation(action, observation)
            observations.append(observation)
        said = result.get("user_message")
        if isinstance(said, dict) and str(said.get("content") or "").strip():
            # (an amount written with a thousands separator - 12,000 - is read as the one figure it is: a line that
            # restates a held amount so is no new value, and one that changes it gives the figure it means)
            self.heard.append(re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(said["content"]).strip()))
        return observations

    def _remember_price_stock(self, product_id: Any, sku: Any, price: Any, in_stock: Any) -> None:
        if product_id is None or sku is None:
            return
        entry = self.stock.setdefault((str(product_id), str(sku)), {})
        entry["turn"] = self.turn
        if isinstance(price, (int, float)) and not isinstance(price, bool):
            entry["price"] = float(price)
        if isinstance(in_stock, bool):
            entry["in_stock"] = in_stock

    def _record_observation(self, action: dict, observation: dict) -> None:
        name = action["name"]
        if name in ("search", "filter"):
            for place, row in enumerate(observation.get("results") or []):
                if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                    key = (str(row["product_id"]), str(row["sku"]))
                    if name == "search" and action["args"].get("query"):
                        self.hits.setdefault(key, []).append((str(action["args"]["query"]).lower(), place))
                    if key not in self.rows:
                        self.rows[key] = row
                        self.row_order.append(key)
                        self.shown_at[key] = self.turn
                    if name == "filter" and action["args"].get("category"):
                        self.shelf.setdefault(key, str(action["args"]["category"]))
                    self._remember_price_stock(key[0], key[1], row.get("price"), row.get("in_stock"))
        elif name in ("view", "compare"):
            for page in ([observation] if name == "view" else observation.get("compared") or []):
                if isinstance(page, dict) and page.get("product_id"):
                    self.pages[str(page["product_id"])] = page
                    for variant in page.get("variants") or []:
                        if isinstance(variant, dict):
                            self._remember_price_stock(page["product_id"], variant.get("sku"), variant.get("price"), variant.get("in_stock"))
        elif name == "inspect_stock":
            self._remember_price_stock(observation.get("product_id"), observation.get("sku"), observation.get("price"), observation.get("in_stock"))
        elif name == "add_to_cart":
            added = observation.get("added") if isinstance(observation.get("added"), dict) else None
            if added:
                key = (str(added.get("product_id")), str(added.get("sku")))
                if key not in self.cart:
                    self.cart.append(key)
                    self.adds += 1
                self._remember_price_stock(key[0], key[1], added.get("price"), added.get("in_stock"))


# ----------------------------------------------------------------------------- the shopper's order
class ShopOrder:
    def __init__(self, problem: dict) -> None:
        self.counter = StoreSession(problem)
        self.reader = ModelCaller()
        self.started = self.reader.started
        self.item = ""
        self.category = ""
        self.requirements: list[dict] = []     # {"property", "value"}
        self.budget: float | None = None
        self.brand = ""
        self.brand_wish = ""                   # a brand only preferred: its listings are filtered for once, never required
        self.wish_filtered = False
        self.listed: list[tuple[str, str]] = []
        self.priority = ""
        self.queries: list[str] = []
        self.sent: set[str] = set()            # searches and filters already sent
        self.shelf_words_used = 0
        self.questions_asked = 0
        self.verdicts: dict[tuple[str, str], dict] = {}
        self.judged_under = ""
        self.judged_count = 0
        self.search_room = SEARCH_LIMIT        # searches allowed so far; a change by the shopper opens more
        self.open_room = OPEN_LIMIT            # listings that may be read so far
        self.judge_room = JUDGE_LIMIT          # listings the reader may be asked about so far
        self.gone: set[tuple[str, str]] = set()
        self.pending_line = ""
        self.owed_reaction = False             # a market change was just seen: the shopper's next question is about it
        self.reader_asked = False
        self.own_cache: dict = {}
        self.part_cache: dict = {}
        self.widened = False
        self.widened_at = 0                    # the turn the searches for the values' other words went out
        self.read_before_miss = False          # the rows shown were read once more before a near miss went in first
        self.named_as: dict[tuple, str] = {}   # (listing, property, value) -> what the reader found its words to name
        self.shelf_colours: dict[tuple, set] = {}   # (value, pages read) -> words the shelf names variants by on their own
        self.kinds_tried: set[str] = set()      # longer entries of a value's own fact already searched as its kinds
        self.amounts_cache: dict[tuple, tuple] = {}    # (text, requirement, requirements in force) -> amounts read in it
        self.name_checks = 0
        self.awaiting_answer = False           # a question was just put to the shopper
        self.probed: set[str] = set()          # titles whose other variants were already searched for
        self.lenient = False                   # nothing fits by the firm rules: a value stated anywhere on the page counts
        self.shelf_cache: tuple[int, str] = (-1, "")
        self.form_cache: dict[tuple, str] = {}       # (pages and rows read, requirement) -> the value as this shelf writes it
        self.texts_cache: tuple = (None, ([], [], []), frozenset())   # (pages and rows read, shelf) -> what this shelf's listings write
        self.page_reads: dict[str, tuple] = {}       # product_id -> its category path, what its page writes, its colour words (all; its facts')
        self.wanted_cache: tuple = (None, [])        # (pages and rows read, values) -> each value's words, as a search carries them
        self.wording_cache: tuple = (None, [])       # (pages read, priority, values) -> the searches for the priority in the shelf's words
        self.shelf_said = ""                   # the shopper's line that repeats the limit and names the shelf
        self.kin_done: set[str] = set()        # titles whose other listings and variants were already searched for
        self.dug = False                       # the last look before a near miss is ordered was taken
        self.lone_cache: tuple[int, set[str]] = (-1, set())   # the words the shelf's variants are named by on their own
        self.pair_fallback = ""                # asked when the answer to the pair question names nothing
        self.pair_chosen = False               # the shopper said which of the two wishes matters more
        self.told_to_ask = False               # the request tells the assistant to ask the shopper before deciding
        self.pair_question = ""                # the question that asks which of the two wishes matters more
        self.pair_wishes: list[str] = []       # the two wishes, in the request's words
        self.pair_open = False                 # the question just put is that one
        self.pair_again = False                # it drew a requirement that is neither wish: it is put once more
        self.no_buy_holds = 0                  # turns a question waited because only listings over the limit met the chosen wish
        self.filed_cache: dict[str, tuple[int, set[str]]] = {}     # per fact label: the entries read under it
        self.use_asked: set[int] = set()       # purposes whose requirement was asked for by the use itself
        self.use_again: set[int] = set()       # purposes to ask about once more, in other words (the answer was another purpose's)
        self.use_twice: set[int] = set()       # purposes asked about twice
        self.use_same: set[int] = set()        # of those to ask once more: purposes whose question drew another purpose's
                                               # requirement (asked once more in the same words)
        self.by_use: set[str] = set()          # those questions, as put
        self.use_drew_nothing = False          # such a question was answered with no requirement: properties from then on
        self.property_asks = 0                 # questions put about a property by its name
        self.set_aside: list[dict] = []        # requirements read as the wishes of a doubted pair, until the answer
        self.pending_filters: list[dict] = []  # store filters to send with the next searches
        self.search_asks = 0                   # times the reader wrote new searches because nothing fitted
        self.rank_asks = 0
        self.figures: dict[tuple[str, str], float | None] = {}   # listing -> its figure for the shopper's priority (None: states none)
        self.kin_swept: set[str] = set()       # titles whose other variants were searched for when looking for the best
        self.figure_unit = ""                  # the unit the figures for the priority are read in
        self.priority_measure = ""             # the measure the priority compares, as the plan reader names it
        self.plan_units: list[str] = []        # the units listings write that measure in, as the plan reader names them
        self.unit_settled = False              # the figures' unit was checked against them
        self.counted_cache: dict[tuple, tuple[bool, float | None]] = {}    # (unit, listing) -> the figure its page writes
        self.figure_room = FIGURE_LIMIT        # figures read so far at most; a change by the shopper opens more
        self.better = ""                       # the reader's word on the priority: a higher or a lower figure is better
        self.lost_titles: set[str] = set()     # titles of chosen listings that went out of stock or over the budget
        self.changed_at = 0                    # the turn at which the shopper last changed what is wanted
        self.unanswered: list[str] = []        # lines that changed what the shopper wants, not yet replied to
        # ---- finding out what the request leaves open
        self.purposes: list[str] = []          # each use or recipient the request gives
        self.invited = False                   # the request itself tells the assistant to ask about the purpose
        self.to_ask: list[str] = []            # properties of the item still worth asking about, likeliest first
        self.lead: list[str] = []              # properties a purpose bears on directly: asked about before the colour
        self.vacated = False                   # a purpose measure the plan named was the priority's own quantity
        self.filled_lead = False               # the shelf's measures filled that slot
        self.lead_mark = 0                     # properties asked about when the last requirement came out
        self.first_thoughts: list[str] = []    # the first properties thought of from the request alone: asked whatever the shelf shows
        self.asked: list[str] = []             # properties already asked about (as plain words)
        self.asked_log: list[tuple[str, str]] = []   # (question, what came of it) for the reader
        self.current_ask = ""                  # the property the question just sent asks about ('' for another kind of question)
        self.current_other = ""                # the second property that question names, when it names two
        self.current_names: list[str] = []     # every property that question names
        self.asked_wide: list[str] = []        # properties named three or four to a question that drew nothing
        self.narrow = 0                        # questions put two at a time over those properties
        self.alone = 0                         # questions still to be put about one property each, after a question about two went unanswered
        self.early_tries = 0                   # times a listing was looked for to put in the cart early
        self.again: list[str] = []             # properties a question named and the answer did not speak of
        self.held_back = 0                     # turns a question waited for the first listings to be read
        self.fresh_rounds = 0                  # rounds of looking still owed to what the shopper has just said they want
        self.measure_units: dict[str, set[str]] = {}    # a measure asked about -> the units the shelf states it in
        self.needs_found = 0                   # requirements the shopper gave in answer to a question
        self.needs_expected = 1                # how many the request can hold back: one per purpose it gives
        self.shelf_reads = 0                   # times the reader has chosen what to ask from the store's own listings
        self.first_found_at = 0                # the number of questions asked when the first held-back requirement came
        self.reasked: set[str] = set()         # questions put a second time because the shopper spoke of something else
        self.last_question = ""                # the question just sent, for reading its answer
        self.plain_pending = False             # the purpose itself is still to be asked about
        self.plain_asked = False               # the purpose itself was asked about
        self.fixed_uses: set[int] = set()      # purposes whose item someone or something else decided (a list, a plan, an item to match)
        self.fixed_quiet = False               # asked what the item is for, the shopper named nothing such a purpose fixes
        self.quiet_after = 0                   # questions after which such a purpose is asked about no further
        self.quiet_closed = False              # (and it no longer counts as holding something back)
        self.use_unheard = False               # a question that spoke of the use drew nothing: ask about properties alone
        self.led: list[int] = []               # purposes (by position) already said back in a question
        self.leading: int | None = None        # the purpose the open question was put for
        self.purposes_met: set[int] = set()    # purposes whose requirement the shopper has named
        self.fresh: dict[str, list[str]] = {}  # shopper line -> the values it newly required
        self.answers: list[str] = []           # the shopper's answers that named a requirement
        self.fresh_limit: set[str] = set()     # shopper lines that set a new spending limit
        self.took_back: dict[str, str] = {}    # a newly required value -> the value the shopper withdrew for it
        self.opening: list[dict] = []          # the requirements as the request first wrote them
        self.opening_budget: float | None = None
        self.stand_in: tuple[str, str] | None = None    # the listing put in the cart first, in place of the choice
        self.went_ahead = False                # a second listing went into the cart ahead of the choice
        self.named_nearest: tuple[str, str] | None = None   # the listing named as the nearest when the order says nothing fits
        self.over_noted = False                # (the note on why no listing over the limit was named, written once)
        self.changed_early = False             # the shopper changed a requirement or the limit before anything was in the cart
        self.moved_on = False                  # the shopper changed a requirement or the limit at any point
        self.reply_sent: tuple[int, str] | None = None  # the turn of the last reply to a change, and the lines it answered
        self.replies_again = 0
        self.spoke_of_use = False              # the question now open spoke of the use
        self.look_at = 0                       # fitting options the shopper wants opened before the choice
        self.cite = ""                         # the property the shopper wants the order's reasons to point to
        self.deferred = False                  # the first question waits for the shelf: several measures could be it
        self.plan_lead: list[str] = []         # the measures the request's purposes point at, as the plan read them
        self.looked_under = ""                 # the requirements under which those options were last opened

    def record_note(self, text: str) -> None:
        self.counter.dialogue.append({"role": "assistant", "content": f"note: {text}"})

    def time_is_short(self) -> bool:
        return monotonic() - self.started > EPISODE_SECONDS

    def seconds_remaining(self) -> float:
        return self.reader.seconds_remaining()

    def can_think(self) -> bool:
        return not self.reader.spend_limit_reached()

    # ---- the request
    @staticmethod
    def restates_item(plan: dict) -> bool:
        """A stated value of the plan is wholly words of the plan's own item name."""
        named = set(letter_digit_tokens(normalized_lower(str(plan.get("item") or ""))))
        values = [str(r.get("value") or "") for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        return bool(named) and any(set(letter_digit_tokens(normalized_lower(v))) <= named for v in values if letter_digit_tokens(normalized_lower(v)))

    @staticmethod
    def kept_values(plan: dict) -> list[dict]:
        """The plan's stated requirements without those whose value is wholly words of its own item name or is its own
        property's name."""
        named = set(letter_digit_tokens(normalized_lower(str(plan.get("item") or ""))))
        kept = []
        for r in plan.get("requirements") or []:
            if not isinstance(r, dict) or not str(r.get("value") or "").strip():
                continue
            words = set(letter_digit_tokens(normalized_lower(str(r.get("value")))))
            if words and ((named and words <= named) or words == set(letter_digit_tokens(normalized_lower(str(r.get("property") or ""))))):
                continue
            kept.append(r)
        return kept

    def read_opening_request(self) -> str:
        request = self.counter.request
        # (no thinking pass: it reads the request no better, and one in ten ran to the token limit, which sent the
        # request to the slower model and used up the episode's time before anything was asked or searched)
        plan = self.reader.request_tool_fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 900) or {}
        if not [r for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]:
            # a request without any detail of the item is rare: have the other model read it too
            again = self.reader.request_tool_fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 700, second=True) or {}
            if again.get("requirements"):
                plan = {**plan, **{k: v for k, v in again.items() if v}} if plan else again
        elif self.restates_item(plan):
            # a value that is only words of the plan's own item name: the kind was read as a value ("ladder & workbench:
            # step stool" -> type = ladder). The other model reads the request too; its reading stands when it does not
            again = self.reader.request_tool_fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 900, second=True) or {}
            kept_again, kept_first = self.kept_values(again), self.kept_values(plan)
            if kept_again:
                self.record_note(f"plan read again: {plan.get('item')!r} {plan.get('requirements')} -> {again.get('item')!r} {kept_again}")
                plan = {**plan, "item": again.get("item") or plan.get("item"), "requirements": kept_again}
            elif kept_first:
                self.record_note(f"plan kept without the values that restate the item: {kept_first}")
                plan = {**plan, "requirements": kept_first}
        self.item = str(plan.get("item") or "").strip()
        self.category = str(plan.get("category") or "").strip()
        self.requirements = [self.requirement(r, request) for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        self.requirements = [dict(r, stated=True) for r in self.requirements if not NOT_A_PROPERTY.search(normalized_lower(f"{r['property']} {r['value']}"))]
        # what the item is for is no property of it: a value the request says once, inside a purpose it gives, is the
        # use itself (it can hold a requirement back, and that one is asked for)
        uses = [normalized_lower(" ".join(str(w).split())) for w in plan.get("purposes") or [] if str(w).strip()]
        whole = normalized_lower(" ".join(request.split()))
        rest = whole
        for use in uses:
            rest = rest.replace(use, " ")

        def same_root(a: str, b: str) -> bool:
            """Two forms of one word (paint, painting): they share most of their letters from the start."""
            shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return shared >= 5 and shared * 10 >= 7 * min(len(a), len(b))

        def only_the_use(value: str) -> bool:
            words = [w for w in letter_digit_tokens(normalized_lower(value)) if len(w) >= 4]
            if not words or re.search(r"\d", value) or rest == whole:
                return False
            in_use = all(any(w == u or same_root(w, u) for use in uses for u in letter_digit_tokens(use)) for w in words)
            return in_use and not any(w == u or same_root(w, u) for w in words for u in letter_digit_tokens(rest))

        self.requirements = [r for r in self.requirements if not only_the_use(r["value"])]
        self.beside: list[str] = []
        for r in list(self.requirements):
            use = self.beside_a_purpose(request, r, uses)
            if use:
                self.requirements.remove(r)
                self.beside.append(use)
        self.budget = positive_amount(plan.get("budget"))
        if self.budget is None:
            amounts = [float(m.replace(",", "")) for m in re.findall(r"(\d[\d,]*\.\d{2})", request)]
            self.budget = max(amounts) if amounts else None
        self.brand = str(plan.get("brand") or "").strip()
        wish = str(plan.get("brand_wish") or "").strip()
        self.brand_wish = wish if wish and normalized_lower(wish) != normalized_lower(self.brand) and self.said_in_request(wish) else ""
        self.listed = [(m.group(1), m.group(2)) for m in re.finditer(r"(\d{6,})::([\w-]+)", request)]
        self.priority = self.grounded_priority(plan.get("priority"), request)
        self.priority_measure = " ".join(letter_digit_tokens(str(plan.get("priority_measure") or ""))) if self.priority else ""
        units = plan.get("priority_units") or []
        self.plan_units = [normalized_lower(u).strip() for u in (units if isinstance(units, list) else [units])
                           if isinstance(u, str) and re.fullmatch(r"[a-z]{1,8}", normalized_lower(u).strip())] if self.priority else []
        if self.priority:
            # what the shopper wants the most or least of is the priority between fitting items, not a value an item
            # must state: a "requirement" that only says the priority again would be met by nothing
            most = self.word_stem_set(self.priority)
            # (the value alone may say it again under another name for the property - power: highest wattage)
            self.requirements = [r for r in self.requirements if re.search(r"\d", r["value"])
                                 or not (self.word_stem_set(r["value"]) - {"the", "a", "an"} <= most
                                         and (self.word_stem_set(r["property"]) & most or len(self.word_stem_set(r["value"]) & most) >= 2))]
        self.add_queries(self.value_queries() + list(plan.get("queries") or [])
                         + ([f"{self.brand_wish} {self.item}"] if self.brand_wish and self.item else []))
        if self.priority:
            # the listings that stand best by the shopper's priority are searched for from the start
            self.add_queries([q for q in plan.get("priority_queries") or [] if isinstance(q, str)][:4])
        pair = [" ".join(str(w).split()) for w in plan.get("doubted_pair") or [] if str(w).strip()][:2]
        purposes = ([" ".join(str(w).split()) for w in plan.get("purposes") or [] if str(w).strip()] + self.beside)[:3]
        # (words of the item's own name read as what it is for are no purpose: the shopper keeps nothing back for them)
        purposes = [p for p in purposes if not (self.content_words(p) and self.content_words(p) <= self.content_words(self.item))]
        # (a clause about the terms of the purchase - the price, the stock, the facts the order is to be backed with -
        # is an instruction, not what the item is for: it holds no requirement back)
        purposes = [p for p in purposes if not PURCHASE_TERMS.search(normalized_lower(p))]
        self.record_note(f"plan item={self.item!r} requirements={self.requirements} budget={self.budget} pair={pair} purposes={purposes} "
                  f"priority={self.priority!r} invitation={plan.get('invitation')!r} open={plan.get('open_properties')}")
        if not self.queries:
            self.add_queries([" ".join(letter_digit_tokens(request)[:5])])
        purposes = [p for i, p in enumerate(purposes) if not any(i != j and self.within(p, q) and (not self.within(q, p) or j < i)
                                                                for j, q in enumerate(purposes))]
        self.purposes = purposes
        try:
            self.look_at = max(0, min(8, int(plan.get("look_at") or 0)))
            self.cite = " ".join(str(plan.get("cite") or "").split())[:60]
        except (TypeError, ValueError):
            self.look_at = 0
        # (an instruction to ask counts only in the request's own words)
        quoted = " ".join(str(plan.get("invitation") or "").split()).strip(" .\"'")
        asking = [w for w in letter_digit_tokens(normalized_lower(quoted)) if len(w) >= 4]
        of_a_purpose = sum(any(same_root(w, u) or w == u for p in purposes for u in letter_digit_tokens(normalized_lower(p))) for w in asking)
        apart = (len(quoted.split()) >= 2 and self.plain_quotes(quoted) in self.plain_quotes(request)
                 and normalized_lower(quoted.split()[0]) not in ("who", "whom", "whose", "which", "that")
                 and not any(self.within(quoted, p) or self.within(p, quoted) for p in purposes)
                 and of_a_purpose * 2 < max(1, len(asking)))
        # (the request itself tells the assistant to ask, check or confirm with the shopper)
        asks_me = r"\b(?:ask|check|confirm|consult|tell)\b[^.?!]{0,24}\bme\b|\blet me know\b|\bif that\b|\bjust say\b"
        self.told_to_ask = bool(re.search(asks_me, self.plain_quotes(request)))
        # (a clause that asks to be asked is an instruction, never a purpose: "if that means the specs change, tell me")
        # (nor is the sentence that names the two wishes and doubts both can be had: "I'd like X and Y, though ...")
        of_the_pair = any(self.content_words(w) and self.content_words(w) <= self.content_words(quoted) for w in pair)
        # (nor is a clause that doubts the request can be met)
        if (apart and len(purposes) < 2 and not re.search(asks_me, self.plain_quotes(quoted)) and not of_the_pair
                and not re.search(DOUBT_WORDS, normalized_lower(quoted)) and not PURCHASE_TERMS.search(normalized_lower(quoted))):
            # a clause the reader took for an instruction ("it needs to follow a plan someone made")
            # is one more purpose of the request: what it is for, from its first article or -ing word on
            words = quoted.split()
            start = next((i for i, w in enumerate(words) if normalized_lower(w) in ("a", "an", "the", "my", "our") or (normalized_lower(w).endswith("ing") and len(w) > 5)), None)
            if start is not None and len(words) - start >= 3:
                purposes = purposes + [" ".join(words[start:])]
                self.purposes = purposes
                apart = False
        if len(purposes) < 2:
            purposes = purposes + self.unread_purposes(request, purposes, pair)[: 2 - len(purposes)]
            self.purposes = purposes
        self.invited = apart and not purposes     # (a request that gives its purpose is asked what that purpose needs)
        joined = self.joined_purposes(request, purposes)
        if joined != purposes:
            self.record_note(f"purposes said as one phrase: {joined}")
            purposes = self.purposes = joined
        # each purpose the request gives can hold one requirement back
        self.needs_expected = max(1, min(2, len(purposes)))
        # (a purpose whose item was decided elsewhere - a list someone wrote, a plan, a person's exact wish, an item to
        # match or replace - is told by the shopper when asked what the item is for, whatever property it fixes)
        decided = [self.content_words(str(w)) for w in plan.get("settled") or [] if str(w).strip()]
        self.fixed_uses = {i for i, p in enumerate(purposes[:2])
                           if any(words and len(words & self.content_words(p)) * 2 >= min(len(words), len(self.content_words(p)) or 1)
                                  for words in decided)}
        if self.fixed_uses:
            self.record_note(f"purposes decided elsewhere: {[purposes[i] for i in sorted(self.fixed_uses)]}")
        # (one property to a question. The colour goes first, whether the reader thought of it or not: it is what a
        # shopper most often holds back, and a request that fixes it takes it out of the line)
        self.queue_properties(["colour", "material"] + [w for w in plan.get("open_properties") or [] if isinstance(w, str)])
        # (a property a purpose bears on directly - what it has to weigh, measure, hold or work with - goes before it)
        uses = [w for w in plan.get("use_properties") or [] if isinstance(w, str)][:6]
        planned = list(uses)
        # (the quantity the priority compares is held back only when a purpose caps it - a weak supply caps the power -
        # so it is asked about, but after every other measure the purposes point at)
        others = [u for u in uses if not self.priority_quantity(" ".join(letter_digit_tokens(normalized_lower(u))))]
        # (and only beside another measure the purposes point at: alone, it is the priority read again)
        uses = others + ([u for u in uses if self.priority_quantity(" ".join(letter_digit_tokens(normalized_lower(u))))] if others else [])
        self.queue_properties(uses, first=True, compared=True)
        self.lead = [n for n in self.to_ask if any(self.word_stem_set(n) == self.word_stem_set(u) for u in uses)]
        # (the plan named the priority's own quantity for a purpose: the shelf's other measures take its place)
        self.vacated = any(self.priority_quantity(" ".join(letter_digit_tokens(normalized_lower(u)))) for u in planned)
        self.plan_lead = list(self.lead)
        self.to_ask.sort(key=lambda name: 0 if name in self.lead else 1 if re.search(r"colou?r", name) else 2 if name == "material" else 3)
        self.first_thoughts = self.to_ask[:4]
        # (both wishes must stand in the request's own words - at least half of each, a telling word among them: a
        # reader's guess is no wish, a reader's word for the property beside the shopper's value is)
        said = set(letter_digit_tokens(request))
        wishes = [[x for x in letter_digit_tokens(w) if x not in ("the", "a", "an", "of", "as", "its", "it", "to", "be", "with")] for w in pair]
        spoken = all(words and sum(x in said for x in words) * 2 >= len(words)
                     and any(x in said and (len(x) >= 3 or x[0].isdigit()) for x in words) for words in wishes)
        if len(pair) == 2 and spoken and len(" ".join(pair).split()) <= 16 and not self.mistaken_pair(pair, purposes) \
                and not self.pair_far_apart(request, wishes):
            # the shopper wants one of two things and says which when asked: that is an open point of its own;
            # until then neither wish is a firm requirement
            wished = normalized_lower(" ".join(pair))
            self.set_aside = [r for r in self.requirements if normalized_lower(r["value"]) in wished]
            self.requirements = [r for r in self.requirements if normalized_lower(r["value"]) not in wished]
            self.pair_fallback = PLAIN_QUESTION
            if self.brand and set(letter_digit_tokens(self.brand)) <= set(letter_digit_tokens(wished)):
                self.brand = ""         # (one of the two wishes: required only once the shopper says it matters more)
            self.pair_wishes = list(pair)
            self.pair_question = f"Which matters more to you: {pair[0]} or {pair[1]}?"
            return self.pair_question
        if self.invited or (not self.to_ask and not self.use_question_left()) or self.fixed_uses:
            # the request asks to be asked about the purpose (or leaves nothing else to ask about: a purpose it gives
            # is asked about in its own words first; and a purpose decided elsewhere is told when the use is asked)
            return PLAIN_QUESTION
        if (len(self.lead) >= 2 and self.to_ask and self.to_ask[0] in self.lead) or (self.vacated and not self.lead):
            # (a purpose points at several measures - how many, how heavy, how much it holds: which one the shopper
            # holds back is as likely the one this kind of item is sold by, which the first listings read show. The
            # first question waits for them; the searches go out at once)
            self.deferred = True
            return ""
        return self.next_question()

    # ---- finding out what the request leaves open
    @staticmethod
    def joined_purposes(request: str, purposes: list[str]) -> list[str]:
        """Two purposes the request writes as one phrase - one word between them that starts no clause of its own - are
        one purpose: the shopper gives one reason, and one requirement at most is held back for it."""
        flat = " ".join(str(request).replace("\u2019", "'").split())
        low = flat.lower()
        out = list(purposes)
        while len(out) >= 2:
            pair = None
            for i, a in enumerate(out):
                for j, b in enumerate(out):
                    at, bt = low.find(" ".join(a.split()).lower()), low.find(" ".join(b.split()).lower())
                    if i == j or at < 0 or bt < 0 or bt <= at:
                        continue
                    gap = low[at + len(" ".join(a.split())): bt]
                    if (re.fullmatch(r"\s+(?!and\b|or\b|but\b|nor\b|while\b|also\b|then\b)[a-z]+\s+", gap)
                            or re.fullmatch(r"\s*[:–—-]\s*", gap)
                            # (a clause that only says more about the first - "a trip abroad, where bags are
                            # weighed" - is that same purpose)
                            or re.fullmatch(r"\s*,?\s*(?:where|which|whose|who|when)\s+", gap)
                            # (or the reader kept the clause's own first word with it: "where bags are weighed")
                            or (re.fullmatch(r"\s*,?\s*", gap) and re.match(r"(?:where|which|whose|who|when)\s", " ".join(b.split()).lower()))):
                        pair = (i, j, flat[at: bt + len(" ".join(b.split()))])
                        break
                if pair:
                    break
            if not pair:
                break
            i, j, whole = pair
            out = [p for k, p in enumerate(out) if k not in (i, j)]
            out.insert(min(i, j), whole)
        return out

    def beside_a_purpose(self, request: str, requirement: dict, uses: list[str]) -> str:
        """A value the request says only in the sentence that gives a purpose, tied to a use by an -ing word before it
        ("... and it has to go with brightening a small hallway"): the use itself, in its own words. "" when the
        value is said where the item or an amount is, or with no purpose beside it: then it is a requirement."""
        words = [w for w in letter_digit_tokens(normalized_lower(requirement["value"])) if len(w) >= 4]
        if not words or re.search(r"\d", requirement["value"]) or not uses:
            return ""
        named = {singular_form(w) for w in letter_digit_tokens(self.item) if len(w) >= 3}
        found = ""
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(request).replace("\u2019", "'").split())):
            said = letter_digit_tokens(sentence)
            if not all(w in said for w in words):
                continue
            if (re.search(r"\d", sentence) or {singular_form(w) for w in said} & named
                    or not any(len(self.content_words(use) & self.content_words(sentence)) >= 2 for use in uses)):
                return ""
            for clause in re.split(r",\s*(?:and|but|while)\s+|;\s*", sentence):
                if not all(w in letter_digit_tokens(clause) for w in words):
                    continue
                tokens = clause.strip(" .!").split()
                start = next((i for i, w in enumerate(tokens) if normalized_lower(w).endswith("ing") and len(w) > 5), None)
                position = next((i for i, w in enumerate(tokens) if words[0] in normalized_lower(w)), None)
                if start is None or position is None or start >= position:
                    return ""
                found = " ".join(tokens[start:])
        return found

    def unread_purposes(self, request: str, purposes: list[str], pair: list[str]) -> list[str]:
        """What the purchase is for, has to match or follow, where the reader of the request left it out: a clause that
        states no requirement, wish, limit or priority and ties the purchase to something by a linking word, from the
        first article or -ing word after that word on ("it has to follow a plan someone made" -> "a plan someone made")."""
        known = [self.content_words(p) for p in purposes]
        taken = [self.content_words(r["value"]) for r in self.requirements] + [self.content_words(w) for w in pair]
        priority = self.content_words(self.priority)
        named = self.content_words(self.item) | self.content_words(self.category) | {singular_form(w) for w in letter_digit_tokens(self.item) if len(w) >= 3}
        out: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(request).replace("\u2019", "'").split())):
            lead = re.match(r"(?:since|because|as)\s+(?:i'm|i am|we're|we are)\s+([^,]{8,80}),", normalized_lower(sentence))
            if lead and normalized_lower(lead.group(1)).split()[0].endswith("ing"):
                # ("Since I'm refilling what we keep at the office, I need ...": the reason given first is the purpose)
                found = lead.group(1).strip()
                words = self.content_words(found)
                if words and not words & named and not any(len(k & words) >= 2 for k in known) and not any(t and t <= words for t in taken):
                    out.append(found)
                    known.append(words)
            if re.search(r"\d", sentence) or sentence.rstrip().endswith("?"):
                continue
            for clause in re.split(r",\s*(?:and|but|while)\s+|;\s*", sentence):
                said = self.content_words(clause)
                if (NOT_A_PROPERTY.search(normalized_lower(clause)) or re.search(DOUBT_WORDS, normalized_lower(clause))
                        or any(len(k & said) >= 2 or (k and k <= said) for k in known)
                        or any(t and t <= said for t in taken) or len(priority & said) >= 2
                        or re.search(r"\b(?:ask|check|confirm|justify|order|choose|pick|select|find)\b", normalized_lower(clause))
                        or re.search(r"\bprefer\w*", normalized_lower(clause))):
                    # (nor is a clause that voices a preference - a maker the shopper would merely like to have is a wish,
                    # not a purpose: asking what it needs only costs a question)
                    continue
                for link in re.finditer(r"\b(?:for|match(?:es|ing)?|follow(?:s|ing)?|correspond(?:s|ing)?|based on|according to|from(?!\s+(?:the|those|these|all|any)\s+(?:\w+\s+)?(?:options?|choices?|ones?|listings?|items?|matching))|(?:goal|purpose|use)(?:\s+(?!is\b)[a-z']+){0,2}\s+is|meant|intended|(?:this|it) is|it's)\b", normalized_lower(clause)):
                    words = clause[link.end():].strip(" .!").split()
                    start = next((i for i, w in enumerate(words) if normalized_lower(w) in ("a", "an", "the", "my", "our")
                                  or (normalized_lower(w).endswith("ing") and len(w) > 5)), None)
                    if start is None or len(words) - start < 3:
                        continue
                    found = " ".join(words[start:])
                    if self.content_words(found) & named:
                        continue        # (the thing wanted itself: "looking for a kettle")
                    if not any(self.within(found, o) for o in out):
                        out.append(found)
                        known.append(self.content_words(found))
                    break
        if out:
            self.record_note(f"purposes the request gives besides: {out}")
        return out

    def queue_properties(self, names: list[str], first: bool = False, compared: bool = False) -> None:
        """Add properties to ask about: plain names of one to four words, none twice, none already settled or asked
        (compared: the quantity the shopper's priority compares may be among them)."""
        fresh: list[str] = []
        settled = [self.word_stem_set(r["property"]) for r in self.requirements + self.set_aside]
        taken = [self.word_stem_set(p) for p in self.asked + ([] if first else self.to_ask)]
        for name in names:
            raw = normalized_lower(str(name).replace("_", " "))
            if re.search(r"[&/,+]|\b(?:and|or)\b", raw):
                continue            # two things joined: the shopper would hear two questions
            text = " ".join(letter_digit_tokens(raw))
            words = self.word_stem_set(text)
            if not words or len(text.split()) > 4 or NOT_A_PROPERTY.search(text) or "brand" in words:
                continue
            if any(len(self.content_words(text) & self.content_words(purpose)) >= 2 for purpose in self.purposes):
                continue            # words of a purpose the request gives: a clause of the request, not a property
            if not compared and self.priority_quantity(text):
                continue            # the quantity the shopper wants most or least of is seldom a requirement held back
            if any(words == other for other in taken + [self.word_stem_set(f) for f in fresh]):
                continue
            if any(words <= other or other <= words for other in settled if other):
                continue            # the shopper already fixed this property
            # one name inside another (lid type, pot lid type): the shopper may know the property by either, so
            # both are asked about, one beside the other where the first is still in line
            beside = next((i for i, f in enumerate(fresh) if len(words) >= 2 and len(self.word_stem_set(f)) >= 2
                           and (words < self.word_stem_set(f) or self.word_stem_set(f) < words)), None)
            if beside is None:
                fresh.append(text)
            else:
                fresh.insert(beside + 1, text)
        if first:
            kept = [p for p in self.to_ask if not any(self.word_stem_set(p) == self.word_stem_set(f) for f in fresh)]
            self.to_ask = fresh + kept
        else:
            self.to_ask += fresh

    def open_names(self) -> list[str]:
        """The properties in line that were not asked about and are not known."""
        known = self.asked + [r["property"] for r in self.requirements + self.set_aside]
        return [n for n in self.to_ask if not any(self.word_stem_set(n) == self.word_stem_set(other) for other in known)]

    def next_property(self, plain: bool = False) -> str:
        """The next property to ask about, one not asked about yet (plain: one named in a plain word or two, or the
        type of the item, when there is one)."""
        open_names = self.open_names()
        short = [n for n in open_names if len(n.split()) <= 2 or n.split()[0] == "type"] if plain else []
        name = (short or open_names or [""])[0]
        self.to_ask = [n for n in self.to_ask if n in open_names and n != name]
        if name:
            self.asked.append(name)
        return name

    def next_question(self) -> str:
        """The next question: about ONE property, by its name alone. The shopper tells a requirement only when it is
        plain that the question asks about one thing. A question that names two properties, or ties the property to
        the use, leaves that in doubt, and so does a name in the plural: each is asked about on its own, in the singular."""
        # (the first question names a property: the shopper of one race tells what a purpose holds back when asked what
        # that purpose needs, the shopper of another only when its property is named - a named property is heard by
        # both. The use is asked about next, then the properties one by one)
        if (self.needs_found and self.property_asks == self.lead_mark and len(self.purposes) >= 2
                and 0 not in self.purposes_met and 1 in self.purposes_met and not self.plain_asked
                and self.needs_found < self.needs_expected):
            # (the second purpose's detail is out, the first's is not: the plain use question tells it; if it draws
            # nothing, the first purpose's own question follows)
            self.lead_mark = -1
            self.current_names, self.leading = [], 0
            self.current_ask, self.current_other = "", ""
            return PLAIN_QUESTION
        run = self.property_asks - self.lead_mark
        open_now = self.open_names()
        second = (run == 1 and bool(self.asked) and (self.asked[-1] in self.lead or self.filled_lead)
                  and bool(open_now) and open_now[0] in self.lead)
        lead = ((run == 0 or second) and self.needs_found < self.needs_expected
                and not self.use_question_again() and bool(open_now))
        by_use = "" if lead else self.use_question()
        if by_use:
            return by_use
        name = self.next_property()
        self.current_names, self.leading = ([name] if name else []), None
        self.property_asks += bool(name)
        self.current_ask, self.current_other = name, ""
        return ASK_ALONE.format(self.spoken(name)) if name else ""

    @staticmethod
    def pair_far_apart(request: str, wishes: list) -> bool:
        """Two wishes the request writes two or more sentences apart: no doubted pair (a doubt is written in one breath)."""
        sentences = [set(letter_digit_tokens(s)) for s in re.split(r"(?<=[.!?])\s+", request) if s.strip()]
        spots = []
        for words in wishes:
            spot = [i for i, said in enumerate(sentences) if words and sum(x in said for x in words) * 2 >= len(words)]
            if not spot:
                return False                  # (a wish not located keeps the pair)
            spots.append(spot)
        return len(spots) == 2 and all(abs(a - b) >= 2 for a in spots[0] for b in spots[1])

    def use_question_again(self) -> bool:
        """A purpose whose needs question drew the other purpose's requirement is due to be asked again in the same words
        (the shopper answers a single point per reply): that goes before any property."""
        return any(i in self.use_again and i in self.use_same and i not in self.purposes_met for i in range(min(2, len(self.purposes))))

    def use_question_left(self) -> bool:
        """A purpose of the request has not been asked about yet (only once a needs question was put: no first one is)."""
        return bool(self.use_asked) and any(i not in self.purposes_met and (i not in self.use_asked or i in self.use_again) and self.said_back(i)
                   for i in range(min(2, len(self.purposes))))

    def use_question(self) -> str:
        """What the item must have for a purpose the request gives: asked once for each purpose (two at most) that
        has no requirement yet, before any property - until such a question is answered with nothing."""
        indices = range(min(2, len(self.purposes)))
        # (each purpose once; then once more one whose question drew another purpose's answer, or nothing. After another
        # purpose's requirement in the same words: the shopper tells one thing at a time, the question was heard for
        # both and the other's came first - with it told, the same words reach this one. Other words are for a
        # question that drew nothing, or an answer given before)
        for again in (False, True):
            for index in indices:
                said = self.said_back(index)
                if index in self.purposes_met or not said or (index not in self.use_again if again else index in self.use_asked):
                    continue
                if not again:
                    continue        # (a question naming no property is seldom answered with one: the properties are asked by name)
                self.use_asked.add(index)
                if again:
                    self.use_again.discard(index)
                    self.use_twice.add(index)
                self.current_names, self.leading = [], index
                self.current_ask = self.current_other = ""
                # (an item named by two things joined would be heard as two things asked about)
                plain = bool(self.item) and len(self.item.split()) <= 3 and not re.search(r"[&/,+]|\b(?:and|or)\b", self.item)
                question = (ASK_BY_USE_AGAIN if again and index not in self.use_same else ASK_BY_USE).format(
                    it=f"the {self.item}" if plain else "it", purpose=said)
                self.by_use.add(question)
                return question
        return ""

    @staticmethod
    def spoken(name: str) -> str:
        """A property's name as a question says it: its last word in the singular ("special features" -> "special feature")."""
        words = str(name).split()
        last = words[-1] if words else ""
        if last.endswith(("series", "species")):
            pass                    # (one form for one and for several)
        elif len(last) > 4 and last.endswith("ies"):
            last = last[:-3] + "y"
        elif len(last) > 4 and last.endswith(("ches", "shes", "sses", "xes")):
            last = last[:-2]
        elif len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is")):
            last = last[:-1]
        return " ".join(words[:-1] + [last])

    def go_over(self) -> bool:
        """Properties named three or four to a question may not all have been heard: when nothing more is in line and
        no requirement has come of them, the first of them are asked about again, two at a time. True when some are."""
        again = [n for n in dict.fromkeys(self.asked_wide) if n in self.asked][: 2 * ASK_NARROW]
        if self.narrow or not again or self.needs_found:
            return False
        self.narrow = 1
        self.asked = [p for p in self.asked if p not in again]
        self.to_ask = again + [n for n in self.to_ask if n not in again]
        return True

    def said_back(self, index: int) -> str:
        """The purpose at that position as it is said back to the shopper: its main clause, in the second person, with
        the clause the request writes right after it (a comma, a relative word, the rest of that clause). That clause
        tells which use is meant: without it the question reads as one about another use, and what the shopper holds
        back for this one may stay unsaid."""
        text = " ".join(str(self.purposes[index]).replace("\u2019", "'").split()).strip(" .\"")
        head = text.split(",")[0]
        text = head if len(head.split()) >= 3 else text
        words = text.split()
        while words and normalized_lower(words[0]) in ("for", "to", "as", "it's", "its"):
            words = words[1:]
        if len(words) < 2:
            return ""
        whole = (" ".join(words) + self.clause_after(text)).split()
        words = whole if len(whole) <= 16 else words        # (a clause the question could not say whole is left out)
        return " ".join(TO_SHOPPER.get(normalized_lower(w), w) for w in words[:16])

    def clause_after(self, phrase: str) -> str:
        """The clause the request writes right after a phrase of its own: a comma, a relative word and the rest of
        that clause, up to the next stop, comma or 'and' - a figure's point or comma stops nothing ("" when the
        request writes none there)."""
        flat = " ".join(self.counter.request.replace("\u2019", "'").split())
        words = phrase.split()
        found = re.search(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", flat, re.I) if words else None
        clause = re.match(r"\s*,\s*(?:who|whom|whose|which|where)\b(?:[^.;:!?,]|(?<=\d)[.,](?=\d))*?(?=\s+and\b|[.;:!?,](?!\d)|$)",
                          flat[found.end():], re.I) if found else None
        return clause.group(0).rstrip() if clause else ""

    def reason_beyond(self, line: str) -> str:
        """The reason a shopper's answer gives for what it requires ("because it's for X, ...", "for X, ..."), when no
        purpose the request was read for carries most of its words; "" otherwise."""
        low = normalized_lower(line)
        found = (re.search(r"\b(?:because|since|given that|as)\s+(?:it'?s|it is|this is|that's|that is)?\s*(?:for|meant for|meant as)?\s*([^,.;]{8,90})", low)
                 or re.match(r"\s*for\s+([^,.;]{8,90}),", low))
        if not found:
            return ""
        reason = " ".join(found.group(1).split())
        words = self.content_words(reason)
        if len(words) < 2:
            return ""
        for purpose in self.purposes:
            mine = self.content_words(purpose)
            if mine and len(words & mine) * 2 >= min(len(words), len(mine)):
                return ""
        return reason

    def purpose_of(self, line: str, asked_for: int | None = None) -> int | None:
        """The purpose, of those still without a requirement, that the shopper's answer speaks of: the one whose
        words it carries, else the one the question was put for."""
        open_ones = [i for i in range(min(2, len(self.purposes))) if i not in self.purposes_met]
        said = self.content_words(line)
        shared = {i: len(self.content_words(self.purposes[i]) & said) for i in open_ones}
        ranked = sorted(open_ones, key=lambda i: (-shared[i], i != asked_for))
        if ranked and not shared[ranked[0]] and asked_for in open_ones:
            return asked_for
        return ranked[0] if ranked else None

    def shopper_message(self, question: str) -> dict:
        """The message that puts one question to the shopper; its answer is read by the next hear()."""
        self.last_question = question
        self.pair_open = bool(self.pair_question) and question == self.pair_question
        self.plain_asked = self.plain_asked or question == PLAIN_QUESTION
        self.spoke_of_use = question == PLAIN_QUESTION or USE_WORDS in question
        self.questions_asked += 1
        self.awaiting_answer = True
        return make_tool_call("message", content=question[:400])

    def listen(self) -> tuple[bool, bool]:
        """Take in what the shopper said after a turn that may have put a question; silence answers it with nothing."""
        if self.awaiting_answer and not self.counter.heard:
            self.awaiting_answer = False
            self.answered("", 0, spoke_of_else=False)
        return self.absorb_shopper_lines()

    def wants_to_ask(self) -> bool:
        """Something may still be held back, and there is room to ask."""
        counter = self.counter
        if self.fixed_quiet and not self.quiet_closed and self.questions_asked >= self.quiet_after:
            # a purpose decided elsewhere that the use question and one property question drew nothing for holds
            # nothing back: it no longer counts among those that may
            self.quiet_closed = True
            quiet = [i for i in self.fixed_uses if i not in self.purposes_met]
            self.needs_expected = max(0, self.needs_expected - len(quiet))
            self.record_note(f"purposes decided elsewhere hold nothing back: {len(quiet)} fewer to find")
        if self.needs_found and self.questions_asked >= self.first_found_at + ASK_MORE:
            return False            # one requirement came out; a further purpose earns a few more questions, no more
        if self.narrow > ASK_NARROW:
            return False            # the properties asked about widely were gone over, two at a time
        if self.pair_chosen and not self.needs_found and self.questions_asked > ASK_AFTER_PAIR:
            return False            # after a doubted pair often nothing is held back: only the likeliest are asked about
        if self.no_buy_holds >= NO_BUY_HOLD and self.no_buy_in_view():
            return False            # the chosen wish is met only over the limit, after a further look as well: a request that
                                    # cannot be met holds nothing back - the nearest listing is named with what is known
        if not self.needs_found and self.questions_asked >= ASK_QUIET and not self.plain_pending:
            return False            # many properties asked about and nothing told: the shopper mostly holds nothing back
        if (self.use_drew_nothing and self.questions_asked >= ASK_QUIET and not self.use_question_left()
                and not self.plain_pending):
            return False            # asked what its purpose needs, a shopper who names nothing mostly holds nothing back
                                    # (and one who has just chosen between two wishes, nothing at all)
        if (self.needs_found >= self.needs_expected and self.use_asked and self.property_asks >= ASK_AFTER_BOTH
                and not self.use_question_left()):
            return False            # one requirement is out and each purpose was asked what it needs: a couple of properties, no more
        return (self.needs_found < self.needs_expected and self.questions_asked < ASK_LIMIT and counter.tool_available("message")
                and not counter.over and counter.turns_remaining() > ASK_TURNS and not self.time_is_short()
                and not [k for k in counter.cart if k != self.stand_in])

    def no_buy_in_view(self) -> bool:
        """The shopper chose ONE of two doubted wishes and nothing held back has come out yet, and of the listings
        read none within the limit meets the request as it now stands while one just over the limit does: what is
        asked for is not to be had at this price. Such a request keeps nothing back behind its purpose. (A shopper who
        wants both wishes, or a listing within the limit that fits, leaves the asking as it was.)"""
        chosen = [r for r in self.requirements if r.get("chosen")]
        if (not self.pair_chosen or len(chosen) != 1 or self.needs_found or self.moved_on or self.counter.cart
                or not self.can_abstain()):
            return False
        self.mark_listings()
        return not any(self.listing_fits(k) for k in self.verdicts) and bool(self.only_over())

    def shelf_properties(self) -> str:
        """What the store shows about listings of this kind: some of the listings that meet everything known so
        far, and each fact label with the number of listings that carry it and its most frequent values."""
        counter = self.counter
        pages = [p for p in counter.pages.values() if self.in_category(p)] or list(counter.pages.values())
        fitting = [k for k in self.verdicts if self.listing_clears(k, "words") and self.affordable(k)]
        shown, titles = [], set()
        for key in fitting + [k for k in self.verdicts if k not in fitting and (self.verdicts[k] or {}).get("kind_ok")]:
            title = normalized_lower((counter.pages.get(key[0]) or {}).get("title"))
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
        """The units in which the names and options of this shelf's listings state amounts, for those in which the
        listings differ: (unit, listings that state one, different figures), the unit most listings state first.
        A unit the shopper's own amounts are in is left out: that amount is known."""
        counter = self.counter
        pages = [p for p in counter.pages.values() if self.in_category(p)] or list(counter.pages.values())
        known = {self.unit_word(m.group(2)) for r in self.requirements + self.set_aside for form in self.forms(r)
                 for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        seen: dict[str, tuple[set[int], set[float]]] = {}
        for index, page in enumerate(pages):
            text = " / ".join([normalized_lower(page.get("title"))] + [normalized_lower(v.get("options")) for v in page.get("variants") or []])
            for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s?([a-z]{1,10})\b", text):
                unit = self.unit_word(m.group(2))
                listings, figures = seen.setdefault(unit, (set(), set()))
                listings.add(index)
                figures.add(float(m.group(1)))
        rows = [(unit, len(listings), len(figures)) for unit, (listings, figures) in seen.items()
                if unit not in known and len(listings) >= 3 and len(figures) >= 2]
        return sorted(rows, key=lambda row: (-row[1], row[0]))[:8]

    @staticmethod
    @lru_cache(maxsize=65536)
    def unit_word(word: str) -> str:
        """A unit as one word, whatever its number: a short unit takes its plural like any word (pcs is pc, kgs is kg)."""
        word = word[:-1] if len(word) == 3 and word.endswith("s") else singular_form(word)
        # (one unit spelled two ways: a word ending in a consonant and -re is the same word ending in -er)
        return word[:-2] + "er" if len(word) >= 5 and re.fullmatch(r"[a-z]+[^aeiou\W\d]re", word) else word

    def order_lead_by_shelf(self) -> None:
        """The measures the purpose points at, in the order of how many listings read state an amount in a unit of
        each (as the reader of the shelf named what each unit measures): what this kind of item is sold by comes
        first. The quantity the priority compares stays last."""
        listings = {unit: count for unit, count, _ in self.shelf_units()}

        def support(name: str) -> int:
            units = set()
            for measure, found in self.measure_units.items():
                if self.word_stem_set(measure) == self.word_stem_set(name):
                    units |= found
            return sum(listings.get(unit, 0) for unit in units)
        lead = [n for n in self.lead if n in self.to_ask]
        if len(lead) < 2:
            return
        # (the measures the purpose itself names go before those only the shelf adds: the shelf decides the order
        # within each)
        ordered = sorted(lead, key=lambda n: (self.priority_quantity(n), n not in self.plan_lead, -support(n), lead.index(n)))
        if ordered != lead:
            self.record_note(f"purpose measures by what the shelf states: {[(n, support(n)) for n in ordered]}")
        self.to_ask = ordered + [n for n in self.to_ask if n not in ordered]
        self.lead = ordered + [n for n in self.lead if n not in ordered]

    def read_shelf(self) -> None:
        """Once listings of the wanted kind are read: have the reader choose what to ask about from the request and
        from what those listings differ in."""
        if self.shelf_reads >= 4 or not self.counter.pages or not self.can_think() or (self.shelf_reads and len(self.open_names()) >= ASK_WIDE):
            return
        self.shelf_reads += 1
        self.mark_listings()
        asked = "\n".join(f"- {question} -> {result}" for question, result in self.asked_log) or "(nothing yet)"
        still = "\n".join(f"- {name}" for name in self.to_ask) or "(none)"
        text = (f"{self.state_text()}\nPurposes named in the request: {'; '.join(self.purposes) or '(none)'}\n\n"
                f"Already asked:\n{asked}\n\nThought of before the listings were read, not asked yet:\n{still}\n\n{self.shelf_properties()}")
        answer = self.reader.request_tool_fill(ASK_GUIDE, text, ASK_TOOL, 300) or {}
        names = answer.get("properties")
        names = re.split(r"[\n;,]+", names) if isinstance(names, str) else (names or [])
        shown = {unit for unit, _, _ in self.shelf_units()}
        measures: list[str] = []
        for entry in answer.get("measures") or [] if isinstance(answer.get("measures"), list) else []:
            unit, _, measure = str(entry).partition(":")
            measure = " ".join(letter_digit_tokens(measure))
            unit = self.unit_word((re.findall(r"[a-z]+", normalized_lower(unit)) or [""])[0])
            if unit in shown and measure and measure != "none" and measure.split()[0] != "how" and len(measure.split()) <= 3:
                self.measure_units.setdefault(measure, set()).add(unit)
                if measure not in measures:
                    measures.append(measure)
        if not measures and shown:
            # (the reader named no measure though the listings show units: those read before for these units stand)
            measures = [m for m, units in self.measure_units.items() if units & shown]
        self.record_note(f"ask about: {names} measures: {measures}")
        if self.deferred and not self.needs_found and len(self.lead) >= 2:
            self.order_lead_by_shelf()
        if measures or [n for n in names if isinstance(n, str) and n.strip()]:
            # the reader has seen the listings: its order replaces what was thought of before. The colour stays
            # first (what a shopper most often holds back shows on no shelf); then the amounts in which this
            # shelf's listings differ and the facts most of its listings carry; then the other first thoughts
            if self.vacated and self.needs_found < self.needs_expected and measures:
                # (the purpose's planned measure was the priority's own quantity: the shelf's next measures lead in
                # its place, ahead of the colour and the material)
                self.vacated = False
                lead_now = [n for n in self.lead if n in self.to_ask]
                taken = [self.word_stem_set(n) for n in self.asked + self.lead] + [
                    self.word_stem_set(r["property"]) for r in self.requirements + self.set_aside]
                fill = [m for m in measures if not self.priority_quantity(m)
                        and not any(self.word_stem_set(m) <= t or t <= self.word_stem_set(m) for t in taken if t)][:max(0, 2 - len(lead_now))]
                if fill:
                    self.queue_properties(fill, first=True)
                    self.lead += [n for n in self.to_ask if n not in self.lead and any(self.word_stem_set(n) == self.word_stem_set(f) for f in fill)]
                    self.filled_lead = True
                    self.record_note(f"lead filled from the shelf's measures: {fill}")
            if self.lead and self.needs_found < self.needs_expected and measures:
                # (a purpose that puts a figure on the item: the amounts this shelf's listings are told apart by are
                # the likeliest figures it holds back, after the ones the purpose names and before colour or material)
                taken = [self.word_stem_set(n) for n in self.asked + self.lead] + [
                    self.word_stem_set(r["property"]) for r in self.requirements + self.set_aside]
                more = [m for m in measures if not self.priority_quantity(m)
                        and not any(self.word_stem_set(m) <= t or t <= self.word_stem_set(m) for t in taken if t)]
                if more:
                    known = list(self.lead)
                    self.queue_properties(more)
                    self.lead = known + [n for n in self.to_ask if n not in known and any(self.word_stem_set(n) == self.word_stem_set(m) for m in more)]
                    lead_now = [n for n in self.lead if n in self.to_ask]
                    self.to_ask = lead_now + [n for n in self.to_ask if n not in lead_now]
                    self.record_note(f"shelf measures join the purpose's: {more}")
                    self.order_lead_by_shelf()
            first = [n for n in self.to_ask if n in self.first_thoughts]
            colour = ([n for n in self.to_ask if n in self.lead] + [n for n in first if re.search(r"colou?r", n) and n not in self.lead]
                      + [n for n in first if n == "material" and n not in self.lead])
            listed = [n for n in names if isinstance(n, str)]
            # (the guide's own order: the shelf's fact labels before the type of the item - a 'type of' name that is no
            # label shown goes right after the last listed name that is one)
            labelled = {frozenset(self.word_stem_set(label)) for page in self.counter.pages.values() for label in (page.get("facts") or {})}
            is_label = [frozenset(self.word_stem_set(n)) in labelled for n in listed]
            loose = [n for n, hit in zip(listed, is_label) if not hit and re.match(r"\s*type of\b", normalized_lower(n))]
            if loose and any(is_label):
                last = max(i for i, hit in enumerate(is_label) if hit)
                listed = ([n for n in listed[: last + 1] if n not in loose] + loose
                          + [n for n in listed[last + 1:] if n not in loose])
            mixed: list[str] = []
            for index in range(max(len(listed), 3)):
                # an amount and a fact label in turn: each is about as likely to be what the shopper holds back
                mixed += measures[index: index + 1] if index < 3 else []
                mixed += listed[index: index + 1]
            self.to_ask = colour
            self.queue_properties(mixed[:4])
            self.queue_properties([n for n in first if n not in colour])
            self.queue_properties(mixed[4:] + measures[3:])
            # (last: the quantity the priority compares - now and then the shopper needs one exact amount of it)
            self.queue_properties(measures, compared=True)

    def find_out(self) -> None:
        """Ask about one property a turn until the shopper has named what the request held back. Every question
        shares its turn with reading or searching, so asking costs no turn of its own."""
        counter = self.counter
        opening = True
        while self.wants_to_ask():
            if (self.first_cart_due() and self.cart_early()) or self.reply_early():
                opening = True
                continue
            if counter.pages:
                self.read_shelf()
            holding = False
            if (self.questions_asked or self.deferred) and not counter.pages and counter.rows and self.held_back < 2:
                # one question is out and no listing is read yet: the next one is chosen with the shelf in view
                self.held_back += 1
                question = ""
            elif self.no_buy_holds < NO_BUY_HOLD and self.no_buy_in_view():
                # (only listings over the limit meet the wish the shopper chose: the next question waits while more
                # listings are read - one within the limit that fits makes this a purchase, and the asking goes on)
                self.no_buy_holds += 1
                question, holding = "", True
            elif self.plain_pending or (not self.needs_found and not self.plain_asked and not self.use_unheard
                                        and not self.purposes and self.questions_asked >= min(ASK_LIMIT, ASK_QUIET) - 1):
                # (the last question, when no property drew a requirement out, is about the purpose itself)
                question, self.plain_pending = PLAIN_QUESTION, False
            else:
                question = self.next_question()
                if not question and counter.pages and self.shelf_reads < 4:
                    self.read_shelf()   # the list ran out: the reader looks again, knowing what was asked
                    question = self.next_question()
                if not question and counter.pages and self.go_over():
                    question = self.next_question()
            if not question and not holding and counter.pages and not self.needs_found and not self.plain_asked and not self.use_unheard and not self.purposes:
                question = PLAIN_QUESTION   # no property drew a requirement out: the purpose itself is asked about once
            if not question and not holding:
                if counter.pages or not counter.rows:
                    break
                question = ""           # nothing to ask yet: read listings first, then choose
            work: list[dict] = []
            room = counter.group_size - (1 if question else 0)
            if opening:
                work = self.listings_to_open(room * OPEN_BATCH)[:room]
            if not work:
                work = self.shelf_filters(room // 2)
                work += self.next_searches(room - len(work))
                if not work:
                    work = self.listings_to_open(room * OPEN_BATCH)[:room]
            opening = not opening
            if not question and not work:
                break
            calls = ([self.shopper_message(question)] if question else []) + work
            counter.send_environment_turn(calls)
            self.listen()

    def cart_early(self, asking: bool = True) -> bool:
        """Once the questions are done (or the first cart is due), the choice as it stands goes into the cart: the best
        listing that firmly fits everything known and meets the request as first written. What the shopper changes and
        what the store does at the first cart come out then, with the episode still ahead, and the listing is kept at
        the order while it still fits. True when the turn was spent on it."""
        counter = self.counter
        # (the asking over, a few more tries while looking: each costs a judging pass, not a turn, until one goes in)
        if (counter.cart or counter.adds or self.early_tries >= (3 if asking else 8) or not counter.pages or counter.over
                or self.time_is_short() or not counter.tool_available("add_to_cart") or counter.turns_remaining() < ASK_TURNS + 4):
            return False
        if self.pair_fallback or self.pair_chosen or self.set_aside:
            return False            # (two wishes the shopper doubts between: what the request first asked for is not settled)
        self.early_tries += 1
        self.mark_listings()
        # (the first cart is held to the request as first written, value by value as the store files them: only a
        # listing that states each of those where it counts, with nothing on its page against it, goes in this early)
        meeting = [k for k in self.verdicts if self.affordable(k) and self.meets_opening(k)]
        if self.name_stand_ins(meeting):
            # (the reader has said what the words of the listings in reach name for a colour or material the request
            # wrote: one it finds naming another beside it, or instead, no longer meets the request as written)
            self.mark_listings()
            meeting = [k for k in self.verdicts if self.affordable(k) and self.meets_opening(k)]
        # (the first cart is the choice as it stands now: of the listings that meet the request as first written, only
        # one that states every requirement known now firmly, with nothing on its page leaving a value in doubt, goes
        # in - the one the final ranking puts first - so that what goes in is what would be ordered)
        keys = [k for k in meeting if self.stands_firm(k)]
        if not keys and not self.first_cart_due(asking):
            return False
        if self.priority:
            self.rank_by_priority(every=True)
        # (once the asking is over, a listing that misses something the shopper has told goes in only at the
        # stand-in's own turn: put in earlier, it is swapped out at the order for nothing)
        # (and once the shopper has told a held-back value, none that misses it goes in at all: it is swapped out for nothing)
        key = self.best_of(keys) if keys else self.as_asked(meeting if asking and not self.needs_found else [k for k in meeting if self.listing_fits(k)])
        if key is None:
            return False
        title = str((counter.pages.get(key[0]) or {}).get("title") or "")[:50]
        self.record_note(f"early cart: {key} {title!r} {self.variant_of(key).get('options')!r} turn={counter.turn} of {len(keys)} firmly fitting")
        group = [make_tool_call("add_to_cart", product_id=key[0], sku=key[1])]
        if counter.tool_available("inspect_stock"):
            group.append(make_tool_call("inspect_stock", product_id=key[0], sku=key[1]))
        group += self.next_searches(counter.group_size - len(group))
        counter.send_environment_turn(group)
        if key not in counter.cart:
            return True
        self.stand_in = key
        self.stand_in_basis = self.held_basis()
        seen = counter.stock.get(key) or {}
        price = seen.get("price")
        self.absorb_shopper_lines()
        if seen.get("in_stock") is False or (self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9):
            self.gone.add(key)
            self.owed_reaction = True
            self.pending_line = ""          # the shopper's reaction to the change is answered by the choice
        return True

    def first_cart_due(self, asking: bool = True) -> bool:
        """While questions are still out, the first cart waits for them - unless the episode is far enough on that
        what the store and the shopper do at the first cart would otherwise come too late to be seen. Once the asking
        is over, it waits less: a change the first cart draws needs turns to be followed, and every turn spent
        looking for a firmer fit is taken from them."""
        counter = self.counter
        limit = counter.turn + counter.turns_remaining()
        share = 0.4 if asking else 0.3
        return counter.turn >= max(6, int(share * limit)) or counter.turns_remaining() <= ASK_TURNS + 6

    def stands_firm(self, key: tuple[str, str]) -> bool:
        """The listing meets the request as first written and every requirement held now, each stated firmly, and its
        page leaves no value in doubt: no other amount of the kind beside the required one (in its options, in the
        property's own fact as a range or a list, or in a title its options do not settle), no second colour."""
        if key in self.gone or not self.listing_fits(key) or not self.meets_opening(key):
            return False
        if any(self.reliance(key, r) < FIRM_FIT or self.plainness_level(key, r) < self.level_needed(r) for r in self.requirements):
            return False
        # (a value the store may read otherwise - a limit, a near spelling, a longer label - is no firm statement)
        return not self.doubtful(key)

    def page_doubt(self, key: tuple[str, str]) -> bool:
        """The page leaves a required value in doubt: another amount of the kind beside the required one, or a second colour."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        for requirement in self.requirements:
            kind = self.property_place(requirement)
            if kind == "amount":
                if self.amounts(options, requirement)[1]:
                    return True
                if not self.has_amount(options, requirement) and self.amounts(page.get("title"), requirement)[1]:
                    return True
                for _, value in self.named_facts(page, requirement):
                    entries = self.entries(value)
                    if len(entries) > 1 or any(self.amounts(entry, requirement)[1] for entry in entries):
                        return True
            elif kind == "colour" and self.second_colour(key, requirement):
                return True
        return False

    def exact_words(self, text: Any, requirement: dict) -> bool:
        """The text carries every word of one of the value's spellings as a word of its own, letter for letter (a
        hyphen parts words; a longer or shorter word does not count)."""
        have = {singular_form(w) for reading in (normalized_lower(text), join_number_unit(text)) for w in letter_digit_tokens(reading)}
        return any(set(singular_form(w) for w in form.split()) <= have for form in self.forms(requirement))

    @staticmethod
    def without_joined_amounts(text: Any) -> str:
        """The text without the amounts it joins to their unit by a hyphen, as a listing's name may do."""
        return re.sub(r"(?<![\d.])\d+(?:\.\d+)?-[a-z]+\b", " ", normalized_lower(text))

    @staticmethod
    def without_amounts(text: Any) -> str:
        """The text without any figure written with a unit."""
        return re.sub(r"\d+(?:\.\d+)?\s*[a-z]+", " ", normalized_lower(text))

    def masked_plain(self, key: tuple[str, str], requirement: dict) -> int:
        """How plainly the page states the required amount where the store reads such an amount: the variant's options,
        the listing's name (an amount joined to its unit by a hyphen is part of a name, not a measurement), and the
        property's own facts when this shelf files the property under facts of its own (another fact's figure - a
        wattage band beside the power - is not that property)."""
        page = self.counter.pages.get(key[0]) or {}
        self.shelf_texts()      # (the shelf's own reading is taken from the real pages first, never from this masked one)
        related, inferred, _ = self.own_labels(requirement)
        own = related | inferred
        # (a fact that repeats the listing's name - a model line - carries its joined amounts too: none is a measurement)
        facts = {label: self.without_joined_amounts(value if not own or str(label) in own else self.without_amounts(value))
                 for label, value in (page.get("facts") or {}).items()}
        masked = dict(page, title=self.without_joined_amounts(page.get("title")), facts=facts)
        self.counter.pages[key[0]] = masked
        try:
            return self.plainness_level(key, requirement)
        finally:
            self.counter.pages[key[0]] = page

    def doubtful(self, key: tuple[str, str]) -> bool:
        """A store may well read the listing as missing a required value that other listings state plainly: the page
        leaves an amount or a colour in doubt, or a property kept as a fact of its own is stated only as one entry of
        a list of several, or only under a longer label while this shelf files that property under its own name."""
        if self.page_doubt(key):
            return True
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        labels_here = [self.word_stem_set(label) for label in (page.get("facts") or {})]
        texts = [str(page.get("title") or ""), options] + [str(v) for v in (page.get("facts") or {}).values()]
        for requirement in self.requirements:
            kind = self.property_place(requirement)
            if kind == "amount":
                # (an amount the page gives only in a fact that is not the property's own - a wattage band beside the
                # power - or joined to its unit by a hyphen in its name - is not the amount the store reads)
                if self.plainness_level(key, requirement) >= 1 and self.masked_plain(key, requirement) < 1:
                    return True
                continue
            if kind == "material" or kind == "fact":
                # (a value matched only in a near spelling - silicon for silicone - and never as its own word)
                if self.plainness_level(key, requirement) >= 1 and not any(self.exact_words(text, requirement) for text in texts):
                    return True
            if kind != "fact" or self.states(options, requirement):
                continue
            own = [(label, self.entries(value)) for label, value in self.named_facts(page, requirement)]
            alone = any(len(entries) == 1 and self.fact_entry_matches(entries[0], requirement, label) for label, entries in own)
            listed = any(len(entries) > 1 and any(self.fact_entry_matches(e, requirement, label) for e in entries) for label, entries in own)
            if listed and not alone:
                return True
            wanted = self.word_stem_set(requirement["property"])
            if (alone and len(wanted) >= 2 and wanted not in labels_here
                    and any(self.word_stem_set(label) == wanted for label in self.shelf_labels())):
                return True
        return False

    def list_span(self, key: tuple[str, str]) -> int:
        """How many values the page lists where it states a required value of a property kept as a fact of its own:
        1 when the fact holds that value alone, the size of the shortest such list otherwise (0: no such statement). A
        fact that lists a great many values (every style, every diet) is one a store seldom reads as stating any one."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        span = 0
        for requirement in self.requirements:
            if self.property_place(requirement) != "fact" or self.states(options, requirement):
                continue
            sizes = [len(entries) for label, entries in ((label, self.entries(value)) for label, value in self.named_facts(page, requirement))
                     if any(self.fact_entry_matches(e, requirement, label) for e in entries)]
            if sizes:
                span = max(span, min(sizes))
        return span

    def shelf_labels(self) -> set[str]:
        """The fact labels the listings of the wanted kind read so far carry."""
        pages = self.counter.pages
        stamp = ("shelf labels", len(pages))
        cached = self.own_cache.get("shelf labels")
        if cached and cached[0] == stamp:
            return cached[1]
        found = {str(label) for page in ([p for p in pages.values() if self.in_category(p)] or list(pages.values()))
                 for label in (page.get("facts") or {})}
        self.own_cache["shelf labels"] = (stamp, found)
        return found

    def as_asked(self, keys: list[tuple[str, str]]) -> tuple[str, str] | None:
        """When the first cart is due and nothing firmly fits everything known yet: the listing that meets the request
        as first written with nothing on its page in doubt - one that fits what was heard since first, then the one
        stating the most plainly."""
        usable = [k for k in keys if k not in self.gone and self.meets_opening(k) and self.affordable(k) and not self.page_doubt(k)]
        usable = [k for k in usable if not self.doubtful(k)] or usable
        if not usable:
            return None
        return self.best_of(usable) or min(usable, key=lambda k: (not self.listing_fits(k), -sum(self.plainness_level(k, r) for r in self.requirements)))

    def best_of(self, keys: list[tuple[str, str]]) -> tuple[str, str] | None:
        """Of the given listings, the one the final ranking puts first (for a first cart: what the shopper said matters
        more, then the request as written, as the commit has it)."""
        allowed = set(keys)
        order = [k for k in self.ranked() if k in allowed and k not in self.gone]
        order.sort(key=lambda k: (not self.meets_chosen(k), not self.keeps_request(k)))
        return order[0] if order else None

    def held_basis(self) -> tuple:
        """What the shopper holds the item to now: every requirement and the limit."""
        return (tuple((str(r.get("property")), str(r.get("value"))) for r in self.requirements), self.budget)

    def stand_in_holds(self, order: list[tuple[str, str]]) -> tuple[str, str] | None:
        """The listing already in the cart is ordered rather than another, when it still fits everything held now, the
        store has not sold it out or raised it over the limit, and the top of the ranking is not clearly better by
        the priority (taking another in its place would put one more listing through the cart for nothing)."""
        counter = self.counter
        key = self.stand_in
        if not order or key is None or key not in counter.cart or key in self.gone or key not in order:
            return None
        if counter.heard:
            return None  # a line not yet taken in: the commit re-checks everything instead
        # (a change since the add - a lower limit, a new value - is checked below like everything else: the listing
        # stays when it still fits all of it, since putting another in its place costs the order a share of its worth)
        if key == order[0]:
            return key
        seen = counter.stock.get(key) or {}
        price = seen.get("price")
        if seen.get("in_stock") is False or (self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9):
            return None
        if not self.listing_fits(key):
            return None
        top = order[0]
        if self.trust(key) < min(self.trust(top), SURE_FIT) - 1e-9:
            return None
        if self.doubtful(key) and not self.doubtful(top):
            return None     # (the listing in the cart may be read as missing a value; the top states every one plainly)

        def number(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if self.priority:
            if self.priced_priority():
                best = (counter.stock.get(top) or {}).get("price")
                if number(price) and number(best) and float(price) > float(best) * 1.02 + 0.01:
                    return None
            else:
                if self.better in ("higher", "lower"):
                    higher = self.better == "higher"
                else:
                    higher = not re.search(r"\b(low|least|small|short|few|min|light)", normalized_lower(self.priority))
                mine, best = self.figure(key), self.figure(top)
                if number(best):
                    if not number(mine):
                        return None
                    mine, best = float(mine), float(best)
                    margin = abs(best) * 0.02 + 1e-9
                    if (higher and mine < best - margin) or (not higher and mine > best + margin):
                        return None
        return key

    def talk_lines(self) -> list[str]:
        """The conversation so far: the request, each message put to the shopper and each line the shopper said."""
        counter = self.counter
        lines = [f"Shopper: {counter.request[:600]}"]
        for entry in counter.dialogue:
            for action in entry.get("tool_calls") or []:
                if isinstance(action, dict) and action.get("name") == "message":
                    lines.append(f"Assistant: {str((action.get('args') or {}).get('content') or '')[:400]}")
            result = entry.get("environment_result")
            said = result.get("user_message") if isinstance(result, dict) else None
            if isinstance(said, dict) and str(said.get("content") or "").strip():
                lines.append(f"Shopper: {str(said['content']).strip()[:600]}")
        return lines[:1] + lines[1:][-40:]

    def whole_check(self, order: list[tuple[str, str]], choice: tuple[str, str]) -> tuple[str, str]:
        """Before the choice goes into the cart: one reading of the whole conversation and the requirement record
        against the first listings of the ranking. The reading only vetoes: when it finds the choice missing a
        requirement now in force and a ranked listing meeting every one, that listing is chosen instead. A miss it
        names on an amount or a colour the code reads as stated firmly is not taken (the words settle it). A failed or
        empty reading, or no time or budget left for it, leaves the choice as it is."""
        counter = self.counter
        done = getattr(self, "whole_checks", None)
        if done is None:
            done = self.whole_checks = {}
        if (not self.can_think() or counter.over or counter.turns_remaining() <= 3 or len(done) >= 3
                or monotonic() - self.started > EPISODE_SECONDS + 30 or not self.requirements):
            return choice
        pool = [choice] + [k for k in order[:8] if k != choice]
        pool = [k for k in dict.fromkeys(pool) if k[0] in counter.pages and k not in self.gone][:5]
        if len(pool) < 2 or pool[0] != choice:
            return choice
        stamp = json.dumps([self.held_basis(), self.priority, pool, len(counter.dialogue) if counter.heard else 0], default=str)
        if stamp in done:
            picked = done[stamp]
            return picked if picked in pool else choice
        done[stamp] = choice
        record = "\n".join(f"{i}. {r.get('property')}: {r.get('value')}" + ("" if r.get("stated") else " (heard in the conversation)")
                            for i, r in enumerate(self.requirements, start=1))
        text = (f"Item wanted: {self.item}\nRequirement record:\n{record}\n"
                f"Spending limit: {self.budget if self.budget is not None else 'none'}\n"
                f"Priority (not a requirement): {self.priority or 'none'}\n\nConversation:\n" + "\n".join(self.talk_lines())
                + "\n\nCandidate listings:\n" + "\n".join(self.card(n, k) for n, k in enumerate(pool, start=1)))
        try:
            answer = self.reader.request_tool_fill(CHECK_GUIDE, text, CHECK_TOOL, 700, think=True) or {}
        except Exception:
            answer = {}
        verdicts: dict[int, tuple[str, int | None]] = {}
        for item in answer.get("verdicts") or []:
            if not isinstance(item, dict):
                continue
            try:
                number = int(item.get("number"))
            except (TypeError, ValueError):
                continue
            meets = str(item.get("meets") or "").strip().upper()[:1]
            violated = item.get("violated")
            verdicts[number] = (meets, violated if isinstance(violated, int) and not isinstance(violated, bool) else None)
        if not verdicts:
            return choice

        def firmly_met(key: tuple[str, str], violated: int | None) -> bool:
            if violated is None or not 1 <= violated <= len(self.requirements):
                return False
            requirement = self.requirements[violated - 1]
            return (self.property_place(requirement) in ("amount", "colour") and self.reliance(key, requirement) >= FIRM_FIT
                    and self.plainness_level(key, requirement) >= self.level_needed(requirement))
        meets, violated = verdicts.get(1, ("", None))
        if meets != "N" or firmly_met(choice, violated):
            if meets == "N":
                self.record_note(f"whole check: {choice} read as missing requirement {violated}, which its words state firmly: kept")
            return choice
        choice_fits = self.listing_fits(choice)
        for number, key in enumerate(pool[1:], start=2):
            if verdicts.get(number, ("", None))[0] != "Y":
                continue
            seen = counter.stock.get(key) or {}
            price = seen.get("price")
            if seen.get("in_stock") is False or (self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9):
                continue
            if choice_fits and not self.listing_fits(key):
                continue
            if not counter.cart and self.meets_opening(choice) and not self.meets_opening(key):
                continue
            done[stamp] = key
            self.record_note(f"whole check: {choice} misses requirement {violated}; {key} "
                      f"{str((counter.pages.get(key[0]) or {}).get('title') or '')[:40]!r} meets every one: chosen instead")
            return key
        self.record_note(f"whole check: {choice} misses requirement {violated}, and no ranked listing meets every one: kept")
        return choice

    def name_stand_ins(self, meeting: list[tuple[str, str]]) -> bool:
        """Before a listing goes into the cart in place of the choice: the reader says what the words of the listings in
        reach of the request as first written name for a colour or a material it wrote - those that meet it by their
        words, then those that state the value less plainly, which the reader's word may settle - where it has not read
        them. The general reading shows it only listings that meet every requirement held now, and a need told since
        does not bind the first cart. One reading a try, for one value. True when the reader was asked."""
        if not self.can_think():
            return False
        for requirement in self.opening:
            kind = self.property_place(requirement)
            if kind not in ("colour", "material"):
                continue
            rest = [r for r in self.opening if r is not requirement]
            need = self.level_needed(requirement)
            near = [k for k in self.verdicts if k not in meeting and (self.verdicts.get(k) or {}).get("kind_ok") and self.affordable(k)
                    and k not in self.gone and (self.counter.stock.get(k) or {}).get("in_stock") is not False
                    and 1 <= self.plainness_level(k, requirement) < need and all(self.plainness_level(k, r) >= self.level_needed(r) for r in rest)]

            def plainest(key: tuple[str, str]) -> int:
                # (the plainest first: of more than the reader is shown at once, those likeliest to go in)
                return -sum(self.plainness_level(key, r) + self.exact_fact_score(key, r) for r in self.opening)
            waiting = []
            for key in sorted(meeting, key=plainest) + sorted(near, key=plainest):
                tag = (key, requirement["property"], requirement["value"])
                if tag in self.named_as or len(waiting) >= 12:
                    continue
                page = self.counter.pages.get(key[0]) or {}
                options = str(self.variant_of(key).get("options") or "").strip()
                if kind == "colour":
                    text = f"variant: {options}" if options else f"title: {str(page.get('title') or '')[:150]}"
                else:
                    text = f"title: {str(page.get('title') or '')[:150]}" + (f" | variant: {options}" if options else "")
                    text += "".join(f" | {label}: {str(value)[:60]}" for label, value in self.named_facts(page, requirement)[:2])
                waiting.append((tag, text))
            if not waiting:
                continue
            asked = (f"Item: {self.item}\nProperty: {requirement['property']}\nWanted value: {requirement['value']}\n\nTexts:\n"
                     + "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(waiting, start=1)))
            answer = self.reader.request_tool_fill(NAMES_GUIDE, asked, NAMES_TOOL, 500) or {}
            marks = answer.get("marks")
            marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
            numbered = all(re.match(r"\s*\[?\d+", str(mark)) for mark in marks)
            if numbered or len(marks) == len(waiting):      # (marks without numbers count only when there is one per text)
                for position, mark in enumerate(marks, start=1):
                    word = re.search(r"\b(only|among|other|none)\b", normalized_lower(mark))
                    number = re.match(r"\s*\[?(\d+)", str(mark))
                    at = int(number.group(1)) if numbered else position
                    if word and 1 <= at <= len(waiting):
                        self.named_as[waiting[at - 1][0]] = word.group(1)
            for tag, _ in waiting:
                self.named_as.setdefault(tag, "unread")      # asked once, not asked again
            self.record_note(f"names {requirement['value']} (first cart): " + " ".join(f"{i}:{self.named_as[tag]}" for i, (tag, _) in enumerate(waiting, start=1))[:240])
            return True
        return False

    def settle_early(self) -> None:
        """When the asking ended before anything went into the cart (the shopper answered at once, or nothing was
        left to ask): read the first listings, put the stand-in in the cart and reply to what that draws."""
        counter = self.counter
        for _ in range(3):
            if counter.cart or counter.over or self.early_tries >= 3 or counter.turns_remaining() < ASK_TURNS + 4 or self.time_is_short():
                break
            if not counter.pages or self.early_tries:
                # (no listing read yet, or none of those read meets the request as written: read further first)
                more = self.listings_to_open(counter.group_size * OPEN_BATCH)[: counter.group_size] or self.next_searches(counter.group_size)
                if not more:
                    break
                counter.send_environment_turn(more)
                self.absorb_quietly()
            self.cart_early(asking=False)
        self.reply_early()

    def reply_early(self) -> bool:
        """The lines that changed a requirement while the stand-in went into the cart get their reply at once, in a
        turn that also reads or searches. True when the turn was spent on it."""
        counter = self.counter
        if not self.unanswered or not counter.cart or counter.over or not counter.tool_available("message") or counter.turns_remaining() <= ASK_TURNS:
            return False
        lines = self.unanswered[-4:]
        said = "\n".join(f"- {line}" for line in lines)
        if not self.holds_now(said)[0]:
            # (nothing newly holds - the value a line asks for was held already: the lines stay owed, and the reply
            # before the order says what they ask to have confirmed)
            return False
        self.unanswered = []
        room = counter.group_size - 1
        work = self.listings_to_open(room * OPEN_BATCH)[:room] or self.next_searches(room)
        turn = counter.turn
        self.answer(said, beside=work)
        if counter.turn == turn:
            return False
        self.pending_line = ""
        self.absorb_shopper_lines()
        return True

    @staticmethod
    def content_words(text: str) -> set[str]:
        return {singular_form(w) for w in letter_digit_tokens(text) if len(w) >= 4} - {"that", "what", "with", "this", "from", "have", "them", "they", "their"}

    def grounded_priority(self, priority: Any, source: str) -> str:
        """What the shopper wants most or least of, kept only when the shopper's own words carry it: a telling
        word of it stands in the text it was read from (every word of it, when it is about the price - the
        cheapest is wanted only when the shopper says so). A reader's guess is no priority."""
        text = " ".join(str(priority or "").split())
        words, have = self.content_words(text), self.content_words(source)
        if not words:
            return ""
        if re.search(r"\b(cheap|price|cost|expensive|afford)", normalized_lower(text)):
            return text if words <= have else ""
        return text if words & have else ""

    def mentions(self, line: str, requirement: dict) -> bool:
        """The line speaks of this requirement: it names the property or carries the value's own words."""
        said = stem_set(line) | self.word_stem_set(line)
        # (the property's name without its linking words - compatible with: compatible - and a value of several words
        # said nearly whole - a make and model without the kind of vehicle it names - speak of it too)
        named = self.word_stem_set(requirement["property"]) - {"with", "of", "for", "to", "the", "a", "an", "and", "in", "on"}
        if named and named <= said:
            return True
        for form in self.forms(requirement):
            words = set(form.split())
            if words and (words <= said or (len(words) >= 3 and len(words & said) * 3 >= 2 * len(words))):
                return True
        return False

    @staticmethod
    def plain_quotes(text: Any) -> str:
        """Text with its quotation marks written one way, for finding words copied from the request in the request."""
        return normalized_lower(text).replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')

    def within(self, inner: str, outer: str) -> bool:
        """Every content word of one phrase stands in the other."""
        words = self.content_words(inner)
        return bool(words) and words <= self.content_words(outer)

    def is_wish(self, value: str) -> bool:
        """The value is one of the two wishes the request doubts it can have both of (every word of it stands in one)."""
        words = {singular_form(w) for w in letter_digit_tokens(value)}
        return bool(words) and any(words <= {singular_form(w) for w in letter_digit_tokens(wish)} for wish in self.pair_wishes)

    def wish_chosen(self, line: str, most: Any) -> dict | None:
        """Of the two doubted wishes, the one an answer chose that was read as what matters most between fitting items:
        the wish whose words (or its property's name) that reading carries, and the other's not, when the line says
        its value. The requirement it is: the one set aside for that wish, else the wish's words the request itself
        says, as the value, under the reader's word for the property beside them (a value the request gives alone, its
        property named by the reader). None when either is not plain."""
        words = self.content_words(str(most or ""))
        held = {w: next((r for r in self.set_aside if letter_digit_tokens(r["value"])
                         and {singular_form(x) for x in letter_digit_tokens(r["value"])} <= {singular_form(x) for x in letter_digit_tokens(w)}), None)
                for w in self.pair_wishes}
        named = [w for w in self.pair_wishes
                 if words & (self.content_words(w) | (self.content_words(held[w]["property"]) if held[w] else set()))]
        if len(named) != 1:
            return None
        kept = held[named[0]]
        if kept is not None:
            chosen = {k: kept[k] for k in ("property", "value", "written", "alike") if k in kept}
        else:
            said = set(letter_digit_tokens(self.counter.request))
            wished = [w for w in letter_digit_tokens(named[0]) if w not in ("the", "a", "an", "of", "as", "its", "it", "to", "be", "with")]
            value, name = [w for w in wished if w in said], [w for w in wished if w not in said]
            if not value or not name:
                return None
            chosen = {"property": " ".join(name), "value": " ".join(value)}
        return chosen if self.says_value(line, chosen["value"]) else None

    def mistaken_pair(self, pair: list[str], purposes: list[str]) -> bool:
        """A pair entry that shares a word with a purpose, or two with the priority, is not a wish for the item."""
        said = [self.content_words(p) for p in purposes]
        priority = self.content_words(self.priority)
        named = self.content_words(self.item) | self.content_words(self.category)
        if all(self.content_words(w) and self.content_words(w) <= named for w in pair):
            return True             # both stand in the item's own name: one kind of thing, not two wishes
        return any(any(self.content_words(w) & s for s in said) or len(self.content_words(w) & priority) >= 2 for w in pair)

    @staticmethod
    def requirement(raw: dict, said: str = "") -> dict:
        """A requirement as read; said: the shopper's words it was read from."""
        def texts(name: str) -> list[str]:
            return [str(w).strip() for w in raw.get(name) or [] if str(w).strip()][:6] if isinstance(raw.get(name), list) else []
        value = re.sub(r"(\d)\s*(?:\"|\u201d|\u2033|'')", r"\1 inch", str(raw.get("value") or "").strip())      # 5" is 5 inch
        name = normalized_lower(raw.get("property"))
        if re.fullmatch(r"\d+(?:\.\d+)?", value) and re.fullmatch(r"[a-z]{1,3}", name.strip()):
            value = f"{value} {name.strip()}"       # (a short unit read as the property's name - 15 v - is the value's unit)
        elif re.fullmatch(r"\d+(?:\.\d+)?", value) and said:
            # (a figure read bare takes the unit the shopper's own words write right after it: 3 read from "... 3 pcs
            # ..." is 3 pcs)
            unit = ShopOrder.unit_said(value, said, name, texts("written"))
            value = f"{value} {unit}" if unit else value
        filed = not re.search(r"\d", value) and not re.search(r"colou?r|material", name)
        written = texts("written")
        written += [form for form in ShopOrder.turned(value) if join_number_unit(form) not in {join_number_unit(w) for w in written}]
        return {"filed": filed, "property": str(raw.get("property") or "").strip(), "value": value,
                "written": written, "alike": texts("alike")}

    @staticmethod
    def unit_said(figure: str, said: str, name: str, written: list[str]) -> str:
        """The unit the text writes right after the figure, one wherever it writes the figure with a word after it: a
        short unit that is no word of the grammar ("4 in 1", "2 of them"), or a longer word that the property's name or
        the reader's own spellings of the value give as its unit ("4 pieces" for a count property). '' when the
        text writes none, or two."""
        grammar = r"a|an|the|and|or|nor|but|to|of|in|on|at|by|for|as|is|are|was|be|it|its|so|if|per|x|off|up|than"
        agreed = ([w for w in letter_digit_tokens(name) if len(w) >= 3 and not re.fullmatch(grammar, w)]
                  + [u.group(1) for form in written for u in re.finditer(r"\d\s*-?\s*([a-z]+)", normalized_lower(form))])
        found: set[str] = set()
        for m in re.finditer(r"(?<![a-z0-9.,])(\d+(?:\.\d+)?)\s?-?\s?([a-z]+)(?![a-z0-9])", normalized_lower(said)):
            word = m.group(2)
            if float(m.group(1)) != float(figure) or re.fullmatch(grammar, word):
                continue
            if len(word) <= 3 or any(ShopOrder.same_unit(word, w) for w in agreed):
                found.add(word)
        return next(iter(found)) if len({ShopOrder.unit_word(w) for w in found}) == 1 else ""

    @staticmethod
    def turned(value: str) -> list[str]:
        """An amount in a metric unit, written in the unit a thousand times larger or smaller when that gives a plain
        figure (1500 ml is 1.5l, 4 kg is 4000g): a listing states the amount in either."""
        found = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([cmk]?[lgm])", normalized_lower(value).strip())      # (litres, grams and metres only)
        if not found:
            return []
        number, unit = float(found.group(1)), found.group(2)
        if len(unit) == 2 and unit[0] == "m" and number >= 1000:
            number, unit = number / 1000, unit[1]                   # milli -> the unit itself
        elif len(unit) == 2 and unit[0] == "k":
            number, unit = number * 1000, unit[1]                   # kilo -> the unit itself
        elif unit in ("l", "g") and number != int(number):
            number, unit = number * 1000, "m" + unit                # a part of a litre or gram -> the unit a thousand times smaller
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

    # ---- searches
    def add_queries(self, queries: list, first: bool = False) -> None:
        fresh = []
        for query in queries:
            text = normalized_lower(" ".join(str(query or "").split())[:QUERY_LENGTH])
            if text and text not in self.sent and text not in fresh and (first or text not in self.queries):
                fresh.append(text)
        # (a search planned again ahead of the rest is wanted now: one already waiting moves up with the new ones)
        self.queries = fresh + [q for q in self.queries if q not in fresh] if first else self.queries + fresh

    def values(self, store: bool = True) -> list[str]:
        """Each required value as a search word, in the shopper's own spelling (the request and the shopper's lines
        give the store's own values): an amount closed up, without a word that only repeats the property's name.
        store: a value this shelf's listings write otherwise, and seldom as the shopper does, in the shelf's spelling."""
        out = []
        for requirement in self.requirements:
            named = self.word_stem_set(requirement["property"])
            shelf = self.store_form(requirement) if store else ""
            words = shelf.split() if shelf else letter_digit_tokens(self.search_word(requirement))
            if re.search(r"\d[.,]\d", requirement["value"]) and not shelf:
                # (a figure with a decimal point stays one word with its unit, as a listing writes it: 1.5 l is 1.5l)
                words = [w.strip(".,;:()") for w in join_number_unit(requirement["value"]).split() if w.strip(".,;:()")]
            # (the shelf's spelling is kept whole: its unit is no word that repeats the property's name)
            kept = words if shelf else [w for w in words if singular_form(w) not in named]
            if kept and not re.search(r"\d", " ".join(kept)) and all(len(w) <= 3 for w in kept):
                kept = []       # ("no lid" under "lid type": the word left over says nothing alone)
            out.append(" ".join(kept or words))
        return out

    def search_word(self, requirement: dict) -> str:
        """The value as a search writes it: closed up, and an amount whose unit is written out in full (220 volts) in
        the short form the reader gave for the same figure (220v) - listings write it so, and a search finds only the
        spelling it is given."""
        value = requirement["value"]
        whole = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-z]{5,})\s*", normalized_lower(value))
        if whole:
            for form in requirement.get("written") or []:
                short = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-z]{1,4})\s*", normalized_lower(form))
                if short and float(short.group(1)) == float(whole.group(1)) and self.same_unit(short.group(2), whole.group(2)):
                    return join_number_unit(form)
        return join_number_unit(value)

    def store_spellings(self, requirement: dict) -> list[str]:
        """The value as the store's own titles write it when they part a closed word in two (a word value whose
        letters two neighbouring title words make up) or close two words up in one: a search finds only the
        spelling it is given."""
        value = normalized_lower(requirement["value"])
        if re.search(r"\d", value) or not value.strip():
            return []
        words = value.split()
        titles = [row.get("title") for row in self.counter.rows.values()] + [page.get("title") for page in self.counter.pages.values()]
        found: dict[str, int] = {}
        for title in titles:
            tokens = letter_digit_tokens(title)
            if len(words) == 1 and len(value) >= 6:
                for one, two in zip(tokens, tokens[1:]):
                    if len(one) >= 3 and len(two) >= 3 and singular_form(one + two) == singular_form(value):
                        found[f"{one} {singular_form(two)}"] = found.get(f"{one} {singular_form(two)}", 0) + 1
            elif len(words) == 2 and any(singular_form(token) == singular_form(words[0] + words[1]) for token in tokens):
                found[words[0] + words[1]] = found.get(words[0] + words[1], 0) + 1
        return [form for form, _ in sorted(found.items(), key=lambda kv: -kv[1])][:2]

    @staticmethod
    def amount_of(value: Any) -> tuple[float, str] | None:
        """A value that is one figure with its unit: the figure, and the unit as one word."""
        found = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-?\s*([a-z]+)\s*", normalized_lower(value))
        return (float(found.group(1)), ShopOrder.unit_word(found.group(2))) if found else None

    @staticmethod
    @lru_cache(maxsize=4096)
    def written_in(text: str) -> tuple[frozenset, tuple]:
        """A text's words, and each figure it writes with a unit after it: (figure, the space between, the unit as written).
        (A small cache: the rows' names come again at every look, and each page is read once anyway.)"""
        return (frozenset(letter_digit_tokens(text)),
                tuple((float(m.group(1)), m.group(2), m.group(3)) for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(\s?)([a-z]{1,10})\b", text)))

    def shelf_texts(self) -> tuple[list[tuple], list[frozenset], list[frozenset]]:
        """What the listings of the wanted kind read so far write: the words and figures of each in-category page (its
        name, options and facts) and of the name of each row a filter on the shelf returned and not read yet (a search's
        rows carry the words it was given); the colour words (the options, and the facts named for a colour) of each
        in-category page, and of each page of the shelf's department - the store's category path above the shelf's
        own label. (Kept with them: the words the department's facts named for a colour carry.)"""
        counter = self.counter
        stamp = (len(counter.pages), len(counter.rows), len(counter.shelf), self.named_shelf())
        if self.texts_cache[0] == stamp:
            return self.texts_cache[1]
        for product_id, page in counter.pages.items():
            if product_id not in self.page_reads:
                # (each page is read once: its category path, the words and figures of its name, options and facts, and
                # its colour words - the options, and the facts named for a colour - and those facts' words apart)
                facts = [str(value) for label, value in (page.get("facts") or {}).items() if re.search(r"colou?r", normalized_lower(label))]
                said = facts + [str(variant.get("options") or "") for variant in page.get("variants") or []]
                text = " / ".join([str(page.get("title") or "")] + [str(v.get("options") or "") for v in page.get("variants") or []]
                                  + [str(value) for value in (page.get("facts") or {}).values()])
                self.page_reads[product_id] = (tuple(str(label) for label in page.get("category_path") or []), self.written_in(normalized_lower(text)),
                                               frozenset(singular_form(w) for part in said for w in letter_digit_tokens(part)),
                                               frozenset(singular_form(w) for part in facts for w in letter_digit_tokens(part)))
        named = self.named_shelf()
        wanted = {singular_phrase_key(named)} if named else {singular_phrase_key(self.category), singular_phrase_key(self.item)} - {""}
        shelf, above = [], {}
        for product_id in counter.pages:
            path = self.page_reads[product_id][0]
            at = next((i for i, label in enumerate(path) if singular_phrase_key(label) in wanted), None)
            if at is not None:
                shelf.append(product_id)
                above[path[:at]] = above.get(path[:at], 0) + 1
        parent = max(above, key=lambda path: above[path]) if above else ()
        department = [product_id for product_id in counter.pages if parent and self.page_reads[product_id][0][: len(parent)] == parent] or shelf
        rows = dict.fromkeys(normalized_lower(counter.rows[key].get("title")) for key in counter.shelf if key in counter.rows and key[0] not in counter.pages)
        answer = ([self.page_reads[product_id][1] for product_id in shelf] + [self.written_in(title) for title in rows],
                  [self.page_reads[product_id][2] for product_id in shelf], [self.page_reads[product_id][2] for product_id in department])
        self.texts_cache = (stamp, answer, frozenset().union(*(self.page_reads[product_id][3] for product_id in department)))
        return answer

    def store_form(self, requirement: dict) -> str:
        """The value as this shelf's own listings write it, where they write it otherwise than the shopper and seldom
        as the shopper does - else empty. A search finds only the spelling it is given. A colour: the shade (the
        reader's, kept with the requirement) that the shelf and its department file as a colour, where the shelf
        files the shopper's word at most once. An amount: in another unit it converts to that the shelf's listings
        write it in, or write most of their figures in, spelled as they spell that unit. A code: the reader's other
        notation of it that shares its figures, where the shelf writes that one more often. The spelling found is
        kept among the value's other words, so a listing that writes it reads as the value (a colour loosely)."""
        counter = self.counter
        stamp = (len(counter.pages), len(counter.rows) + len(counter.shelf), str(requirement["property"]), str(requirement["value"]),
                 tuple(str(w) for w in requirement.get("written") or []), tuple(requirement.get("shades") or []))
        if stamp in self.form_cache:
            return self.form_cache[stamp]
        texts, shelf, department = self.shelf_texts()
        value, amount, form = normalized_lower(requirement["value"]), self.amount_of(requirement["value"]), ""
        colour = self.property_place(requirement) == "colour"
        if colour:
            own = [set(f.split()) for f in self.forms(requirement)]

            def filed(pages: list, forms: list) -> int:
                return sum(1 for page in pages if any(words <= page for words in forms))
            if filed(shelf, own) <= 1:
                best, mine = (0, ""), filed(shelf, own) + filed(department, own)
                for shade in requirement.get("shades") or []:
                    words = {singular_form(w) for w in letter_digit_tokens(shade)}
                    # (a narrower name of the colour itself - its own word with another - is no other word for it; and
                    # a shade the department's colour facts never carry may name a scent, a flavour or a form where an
                    # option carries it)
                    if words and not any(words >= w for w in own) and words <= self.texts_cache[2]:
                        score = filed(shelf, [words]) + filed(department, [words])
                        if score > best[0]:
                            best = (score, " ".join(letter_digit_tokens(shade)))
                if best[0] >= 3 and best[0] > mine:
                    form = best[1]
        elif amount:
            figure, unit = amount
            stated: dict[str, list[tuple[int, float, str]]] = {}
            for index, (_, figures) in enumerate(texts):
                for number, space, raw in figures:
                    stated.setdefault(self.unit_word(raw), []).append((index, number, space + raw))

            def stating(name: str, number: float | None, loose: bool = False) -> tuple[set[int], dict[str, int]]:
                # (the texts that write a figure in that very unit; loose: in any spelling of it - for the shopper's
                # own unit, so that a shelf writing it short or long counts as writing it)
                at, spelled = set(), {}
                for seen, found in stated.items():
                    if seen == name or (loose and self.same_unit(seen, name)):
                        for index, n, spelling in found:
                            if number is None or abs(n - number) < 1e-6:
                                at.add(index)
                                spelled[spelling] = spelled.get(spelling, 0) + 1
                return at, spelled

            def places(number: float) -> int:
                return len(f"{number:g}".partition(".")[2])
            mine, mine_anywhere = len(stating(unit, figure, True)[0]), len(stating(unit, None, True)[0])
            best: tuple = ((0, 0), "")
            # (the amount in another unit: another metric step of it, or another unit the reader converted it to - not
            # the same figure under another unit's name, which is a spelling of the unit, or another measure)
            for other in sorted(self.converted(requirement)):
                parsed = self.amount_of(other)
                if not parsed or parsed[1] == unit or self.same_unit(parsed[1], unit):
                    continue
                number, name = parsed
                # the amount itself written in the other unit on more of the shelf's texts than in the shopper's; else,
                # where the shelf writes the amount as the shopper does once at most, most of its figures written in
                # that unit, the amount within the sizes they give in it (a plain figure in it, not a finer one than
                # asked; a shelf that writes only some other, smaller measure in that unit - a part's - says nothing of
                # how it writes this one) - spelled as the shelf spells that figure in it, else as it spells that unit
                at, spelled = stating(name, number)
                rank = (2, len(at)) if len(at) >= 2 and len(at) > mine else (0, 0)
                if not rank[0] and mine <= 1 and number >= 1 and places(number) <= places(figure):
                    anywhere, usual = stating(name, None)
                    sizes = [n for _, n, _ in stated.get(name, [])]
                    rank = ((1, len(anywhere)) if len(anywhere) >= 3 and len(anywhere) >= 2 * mine_anywhere
                            and min(sizes) <= number <= max(sizes) else (0, 0))
                    spelled = spelled or usual
                if rank > best[0]:
                    best = (rank, f"{number:g}" + max(spelled, key=lambda s: spelled[s]))
            form = best[1]
        elif re.search(r"\d", value):
            runs = set(re.findall(r"\d{3,}", value))

            def holding(spelled: str) -> int:
                words = set(letter_digit_tokens(spelled))
                return sum(1 for tokens, _ in texts if words and words <= tokens)
            mine, best = holding(value), (0, "")
            for other in requirement.get("written") or []:
                spelled = join_number_unit(other)
                # (a notation that shares none of its longer figures names another measure of it, not a spelling)
                if runs & set(re.findall(r"\d{3,}", spelled)) and letter_digit_tokens(spelled) != letter_digit_tokens(value):
                    count = holding(spelled)
                    if count > best[0]:
                        best = (count, spelled)
            if best[0] >= 2 and best[0] > mine:
                form = best[1]
        if form:
            name = "alike" if colour else "written"
            if form not in {normalized_lower(w) for w in requirement.get(name) or []}:
                requirement[name] = list(requirement.get(name) or []) + [form]
        self.form_cache[stamp] = form
        self.form_cache[stamp[:4] + (tuple(str(w) for w in requirement.get("written") or []),) + stamp[5:]] = form
        return form

    def colour_shades(self, requirement: dict) -> list[str]:
        """The names sellers give to shades of a required colour (the reader's), asked for once and kept with the
        requirement: the shelf may file the colour under one of them."""
        if "shades" not in requirement and self.can_think():
            answer = self.reader.request_tool_fill(SHADES_GUIDE, f"Item: {self.item}\nColour: {requirement['value']}", WORDS_TOOL, 200)
            if isinstance(answer, dict):
                requirement["shades"] = [str(w).strip() for w in answer.get("words") or [] if str(w).strip() and not re.search(r"\d", str(w))][:6]
        return list(requirement.get("shades") or [])

    def close_spelling(self, requirement: dict) -> str:
        """Another spelling of a word value close to it (grey, gray) that a search tells apart, if the reader gave one."""
        return next((join_number_unit(form) for form in requirement.get("written") or [] if not re.search(r"\d", str(form))
                     and letter_digit_tokens(join_number_unit(form)) != letter_digit_tokens(join_number_unit(requirement["value"]))
                     and SequenceMatcher(None, normalized_lower(form), normalized_lower(requirement["value"])).ratio() >= 0.75), "")

    def close_values(self) -> list[str]:
        """The values as search words, each word value in its close other spelling where it has one."""
        return [self.close_spelling(r) or value for r, value in zip(self.requirements, self.values())]

    def item_words(self) -> list[str]:
        return [w for w in letter_digit_tokens(self.category or self.item) if len(w) >= 3 and w not in ("and", "the", "for", "with", "other")]

    def kind_words(self, count: int) -> list[str]:
        """The item words a search fans the values out over: a name that lists several kinds (A, B & C) names each of
        them, and a listing of the last kind is found by its own word - so each is searched, not only the first ones.
        A kind named by one word before the last kind's words (A & B C: an A C and a B C) goes out with the word the
        last kind ends in, as the name writes it: the lone word names no kind of thing, and a search matches whole words.
        (Not a word the last kind repeats - A & A C: that one names a kind of its own.)"""
        name = str(self.category or self.item or "")
        words = list(dict.fromkeys(self.item_words()))
        if not re.search(r",|&|/|\band\b", name):
            return words[:count]
        parts = [part for part in (letter_digit_tokens(text) for text in re.split(r",|&|/|\band\b", normalized_lower(name))) if part]
        if len(parts) >= 2 and len(parts[-1]) >= 2:
            head, last = parts[-1][-1], {singular_form(w) for w in parts[-1]}
            lone = {part[0] for part in parts[:-1] if len(part) == 1 and singular_form(part[0]) not in last}
            words = [f"{w} {head}" if w in lone else w for w in words]
        return words[: max(count, 5)]

    def value_queries(self) -> list[str]:
        """The required values with the item words: every value, the newest alone, and each item word in turn."""
        values, item = self.values(), self.category or self.item
        label = self.store_label()
        if label and values and normalized_lower(label) != normalized_lower(item):
            self.add_queries([f"{' '.join(values)} {label}", f"{values[-1]} {label}"])
        if not values or not item:
            return [item] if item else []
        every, newest = " ".join(values), values[-1]
        out = [f"{every} {item}", f"{newest} {item}", f"{values[0]} {item}"]
        head = (self.item_words() or [""])[-1]
        for requirement in self.requirements:       # the same search in the value's other spelling
            for other in (requirement.get("written") or [])[:3] + self.store_spellings(requirement):
                if join_number_unit(other) != join_number_unit(requirement["value"]):
                    turned = " ".join(join_number_unit(other) if r is requirement else self.search_word(r) for r in self.requirements)
                    out.append(f"{turned} {item}")
                    # (an amount in another unit written apart too: the store finds only the spelling it is given)
                    apart = " ".join(normalized_lower(other) if r is requirement else self.search_word(r) for r in self.requirements)
                    if re.search(r"\d", other) and apart != turned:
                        out.append(f"{apart} {item}")
                    if head and len(item.split()) > 1 and re.search(r"\d", other) and other in self.turned(requirement["value"]):
                        out.append(f"{turned} {head}")      # (an amount in its other unit, with the item's plain word alone)
        out += [f"{every} {w}" for w in self.kind_words(3)]
        # (and the values in their close other spellings, fanned out the same way: the store finds a variant written so
        # only in that spelling, and with the item's plainer words the item phrase's own matches do not crowd it out)
        turned = " ".join(self.close_values())
        if turned != every:
            out += [f"{turned} {w}" for w in self.kind_words(3)] + [f"{self.close_values()[-1]} {w}" for w in self.kind_words(2)]
        # (a whole amount written with a decimal point, as some sellers write it: a search finds only the spelling it
        # is given, and the variant written so may show for no other search)
        plain = self.values(store=False)
        decimal = re.sub(r"(?<![\d.])(\d+)(?!pc\b)([a-z]{1,2})\b", r"\1.0\2", " ".join(plain))
        if decimal != " ".join(plain):
            out.append(f"{decimal} {item}")
        if len(values) >= 2:
            out.append(every)               # the values alone: listings that state them all, whatever the title calls the item
        out += [f"{newest} {w}" for w in self.kind_words(2)]
        earlier = " ".join(values[:-1])
        whole = " ".join(letter_digit_tokens(join_number_unit(self.requirements[-1]["value"])))
        if whole and not re.search(r"\d", whole):
            # the newest value with every word of it, and with its last word in the other number (glove, gloves): the
            # store files a value in its own spelling, and a search finds the spelling it is given
            head, _, tail = whole.rpartition(" ")
            turned = tail[:-1] if tail.endswith("s") and len(tail) > 3 else tail + "s"
            forms = ([whole] if whole != newest else []) + [f"{head} {turned}".strip()]
            out[3:3] = [" ".join(f"{earlier} {form} {item}".split()) for form in forms]
        # the newest value in the other words sellers use for it: a listing filed under one of those says neither
        # the shopper's word in its name nor in its options
        for other in (self.requirements[-1].get("alike") or [])[:2]:
            if join_number_unit(other) != join_number_unit(self.requirements[-1]["value"]):
                out.append(" ".join(f"{earlier} {join_number_unit(other)} {item}".split()))
        if plain != values:
            # (a value this shelf writes otherwise than the shopper: the item phrase goes out in the shopper's spelling,
            # the values in the shelf's go out alone and with the item's plainer words - beside the whole item phrase,
            # the listings that carry its words crowd out the one that carries the values)
            out[0] = f"{' '.join(plain)} {item}"
            out[1:1] = [every] + [f"{every} {w}" for w in self.kind_words(2)]
        out[3:3] = self.priority_wording()
        if len(values) == 1 and len(every.split()) >= 2 and self.property_place(self.requirements[0]) != "colour":
            # (so does a value of several words that is the only one - a make, a model, a kind - among the first
            # searches: the listing that states it may call the item by other words, and the item's words crowd it
            # out; not a colour, which alone names no kind of thing)
            out.insert(3, every)
        return out

    def priority_wording(self, best: float | None = None) -> list[str]:
        """Searches for the shopper's priority in the words this shelf's listings write it in: a count with its unit
        spelled as most of them spell it after a figure (the very spelling - a store may count two spellings alike, a
        search does not), or a measure with the unit they write right before its name. A search finds only the words
        it is given, and the words the priority is said in stand in no listing. best: also the best figure found so
        far and the figure a better one would be near (twice it, or half it where a lower figure is better), written
        apart from the unit as listings write a count."""
        if not self.priority or self.priced_priority() or not self.requirements:
            return []
        every, item, head = " ".join(self.values()), self.category or self.item, (self.item_words() or [""])[-1]
        unit = self.counted_unit()
        stamp = (len(self.counter.pages), self.named_shelf(), self.priority, unit, every, item, best, self.better)
        if self.wording_cache[0] != stamp:
            self.wording_cache = (stamp, self._priority_wording(every, item, head, unit, best))
        return list(self.wording_cache[1])

    def _priority_wording(self, every: str, item: str, head: str, unit: str, best: float | None) -> list[str]:
        texts = [normalized_lower(" / ".join([str(page.get("title") or "")] + [str(v.get("options") or "") for v in page.get("variants") or []]))
                 for page in self.counter.pages.values() if self.in_category(page)]
        if unit:
            spelled: dict[str, set[int]] = {}
            for index, text in enumerate(texts):
                for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?\s?([a-z]{1,10})\b", text):
                    if self.unit_word(m.group(1)) == unit:
                        spelled.setdefault(m.group(1), set()).add(index)
            usual = sorted((s for s in spelled if len(spelled[s]) >= 2), key=lambda s: (-len(spelled[s]), s))
            if usual:
                out = [f"{every} {item} {usual[0]}", f"{every} {head} {usual[0]}"]
                if best:
                    near = best / 2 if self.better == "lower" else best * 2
                    out += [f"{every} {item} {figure:g} {usual[0]}" for figure in (best, round(near) if near >= 2 else near)]
                return out
        # (a measure: the unit the shelf writes right before the measure's own word - the priority's last word that
        # says something)
        for word in [w for w in letter_digit_tokens(self.priority) if singular_form(w) in self.content_words(self.priority)][-1:]:
            units: dict[str, int] = {}
            for text in texts:
                for m in re.finditer(rf"\d[\d./]*\s*([a-z]{{2,10}})\s+{re.escape(word)}\b", text):
                    units[m.group(1)] = units.get(m.group(1), 0) + 1
            if units:
                found = max(sorted(units), key=lambda u: units[u])
                return [f"{every} {found} {word}", f"{every} {(self.kind_words(1) or [''])[0]} {found} {word}"]
        return []

    def shelf_queries(self, count: int) -> list[str]:
        """Words the store's own listings of this category carry most often, each with the required values:
        a search shows ten rows, so other words reach other listings of the same shelf."""
        titles = [self.counter.rows[k].get("title") for k in self.counter.shelf if k in self.counter.rows]
        titles += [p.get("title") for p in self.counter.pages.values() if self.in_category(p)]
        if len(titles) < 3 or not self.requirements:
            return []
        values, plain = self.values(), self.values(store=False)
        # (the values are no words to search them with, in the shelf's spelling or the shopper's, and neither is a
        # value the shopper has taken back: a search with it finds what they no longer want)
        known = set(self.item_words()) | {w for v in values + plain + list(self.took_back.values()) for w in letter_digit_tokens(join_number_unit(v))}
        known |= {form for v in values if v not in plain for w in letter_digit_tokens(v) for form in (w, singular_form(w))}
        # (a seller's abbreviation - LED, COB, E27 - is a word of three characters the shelf writes in capitals; one
        # joined to a figure - 30PCS - is a count)
        caps = {w.lower() for title in titles if title and not str(title).isupper()
                for w in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{2}(?![A-Za-z0-9])", str(title))}
        caps -= {w.lower() for title in titles if title for w in re.findall(r"(?<=\d)([A-Za-z]{3})(?![A-Za-z])", str(title))}
        counts: dict[str, int] = {}
        for title in titles:
            for word in set(letter_digit_tokens(title)):
                if (len(word) >= 4 or word in caps) and not word.isdigit() and word not in known:
                    counts[word] = counts.get(word, 0) + 1
        common = [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2]
        # (the words of facts that list several entries - what a listing suits, holds or is for - are words its title
        # may not carry, and a search finds them all the same: every other word is taken from those)
        listed: dict[str, int] = {}
        for page in self.counter.pages.values():
            if not self.in_category(page):
                continue
            for value in (page.get("facts") or {}).values():
                entries = self.entries(value)
                if len(entries) >= 2:
                    for word in {w for entry in entries for w in entry.split()}:
                        if len(word) >= 4 and not word.isdigit() and word not in known and word not in counts:
                            listed[word] = listed.get(word, 0) + 1
        extra = [w for w, n in sorted(listed.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2]
        merged: list[str] = []
        for index in range(max(len(common), len(extra))):
            merged += common[index: index + 1] + extra[index: index + 1]
        common = merged
        # (where the shelf writes a value otherwise than the shopper, each word goes out with the values in the shelf's
        # spelling - alone, and with the item's own word for a listing that the bare values crowd out - and then in the
        # shopper's: half as many words a round)
        store = values != plain
        # (the most frequent words not searched yet: the list is sorted again as pages come in, so a position in it is
        # no record of what was searched)
        done = self.__dict__.setdefault("shelf_words_done", set())
        chosen = [w for w in common if w not in done][: (max(2, count // 2) if store else count)]
        done.update(chosen)
        self.shelf_words_used += len(chosen)
        every, newest = " ".join(values), values[-1]
        head = (self.item_words() or [""])[-1]
        if store:
            out = [query for w in chosen for query in (f"{w} {every}", f"{w} {every} {head}")] + [f"{w} {' '.join(plain)}" for w in chosen]
            # (an amount the shopper wrote closed up goes out apart too - not the shelf's own spelling, nor a code: those
            # are written one way)
            apart = " ".join(re.sub(r"(\d)([a-z])", r"\1 \2", v) if v == p and self.amount_of(r["value"]) else v
                             for r, v, p in zip(self.requirements, values, plain))
            return out + ([f"{w} {apart}" for w in chosen] if apart != every else [])
        # (a value with another spelling close to it - grey, gray - is written so in every other search: a variant's
        # options are found only in the spelling a search gives, and sellers write both)
        turned = " ".join(self.close_values())
        # (a name that lists several kinds may end in a broad word few of the shelf's titles carry: the item word most
        # of them do carry goes out with the other half of the words, which the last word does not reach; every search
        # with the last word is kept)
        usual = ""
        if re.search(r",|&|/|\band\b", str(self.category or self.item or "")):
            shelf = [{singular_form(w) for w in letter_digit_tokens(p.get("title"))} for p in self.counter.pages.values() if self.in_category(p)]
            carried = {w: sum(1 for words in shelf if singular_form(w) in words) for w in self.item_words()}
            if len(shelf) >= 3 and carried and carried.get(head, 0) * 2 < len(shelf):
                usual = max(carried, key=lambda w: carried[w])
                usual = usual if usual != head and carried[usual] > carried.get(head, 0) else ""
        # (with the item's own word too for half of them: a listing may name the kind of thing in other words - tab, pad)
        out = ([f"{w} {turned if index % 2 and turned != every else every}" for index, w in enumerate(chosen)]
               + [f"{w} {every} {head}" for w in chosen[: count // 2]] + [f"{w} {every} {usual}" for w in chosen[count // 2:] if usual]
               + [f"{w} {newest}" for w in chosen[: count // 2]])
        apart = re.sub(r"(\d)([a-z])", r"\1 \2", every)
        if apart != every:
            # a listing writes an amount closed up or with a space (10pcs, 10 pcs), and a search finds the spelling it is given
            out[1:1] = [f"{w} {apart}" for w in chosen[: max(3, count // 2)]]
        return out

    def kin_titles(self) -> list[str]:
        return list(dict.fromkeys(normalized_lower((self.counter.pages.get(k[0]) or {}).get("title")) for k in self.verdicts if self.listing_clears(k, "words")))

    def kin_pending(self) -> bool:
        return len(self.kin_done) < 4 and any(title not in self.kin_done for title in self.kin_titles())

    def kin_queries(self) -> list[str]:
        """The required values with the title of a fitting listing: so specific a search returns that article's
        other variants and its other listings."""
        every = " ".join(self.values())
        out = []
        for key in [k for k in self.verdicts if self.listing_clears(k, "words")]:
            page = self.counter.pages.get(key[0]) or {}
            title = normalized_lower(page.get("title"))
            if title in self.kin_done or len(self.kin_done) >= 4:
                continue
            self.kin_done.add(title)
            words = letter_digit_tokens(title)
            if len(words) < 2:
                continue
            out += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.item} {' '.join(words[:3])}"]
            if len(words) > 10:
                out.append(f"{every} {' '.join(words)}")
            options = " ".join(letter_digit_tokens(self.variant_of(key).get("options")))
            if options:
                out.append(f"{options} {' '.join(words[:10])}")     # the listing's own words for this variant
            brand = normalized_lower(page.get("brand"))
            if brand and brand not in ("none", "no brand", "oem"):
                out.append(f"{every} {brand} {self.item}")
            out.append(f"{every} {' '.join(words[:2])}")
            elsewhere = {w for row in self.counter.rows.values() if normalized_lower(row.get("title")) != title for w in letter_digit_tokens(row.get("title"))}
            own = sorted((w for w in dict.fromkeys(words) if len(w) >= 5 and not w[0].isdigit() and w not in elsewhere), key=len, reverse=True)
            if own:
                out.append(own[0])          # a word no other listing carries leads straight to this article's variants
        return out

    def weighted_queries(self) -> list[str]:
        """A search weighs a listing by how many of its words the listing carries: a value of two words
        outweighs a value of one. When most rows the searches with every value showed name a longer value in their
        titles and hardly any names a shorter one, the listings that name the shorter one in theirs (and state the
        longer one only in their facts) are ranked below the ten shown: that value is written twice in one more
        search, to weigh as much."""
        values = self.values()
        if len(values) < 2:
            return []
        wanted = [(set(letter_digit_tokens(join_number_unit(v))), set(letter_digit_tokens(v))) for v in values]
        rows = [k for k, hits in self.counter.hits.items()
                if any(all(closed <= set(letter_digit_tokens(query)) or apart <= set(letter_digit_tokens(query)) for closed, apart in wanted) for query, _ in hits)]
        if len(rows) < 8:
            return []           # (too few rows to tell what the store puts first)
        titles = [stem_set((self.counter.rows.get(k) or {}).get("title")) for k in rows]
        named = []
        for requirement in self.requirements:
            need = stem_set(requirement["value"])
            others = [stem_set(form) for form in list(requirement.get("written") or []) + list(requirement.get("alike") or [])]
            named.append(sum(1 for title in titles if need and need <= title or any(other and other <= title for other in others)))
        sizes = [len(value.split()) for value in values]
        every, item = " ".join(values), self.category or self.item
        # (a value at most one row in ten names, beside a value of more words that half of them name)
        return [f"{value} {every} {item}" for value, count, size in zip(values, named, sizes)
                if count * 10 <= len(rows) and any(other * 2 >= len(rows) and longer > size for other, longer in zip(named, sizes))]

    def filed_as_list(self, prop: str) -> bool:
        """Some listing of the shelf read so far files several entries under a fact named like the property: the store
        keeps it as a list, and a listing may have more than one value of it."""
        wanted = self.word_stem_set(prop)
        if not wanted:
            return False
        shelf = [page for page in self.counter.pages.values() if self.in_category(page)] or list(self.counter.pages.values())
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                have = self.word_stem_set(label)
                if have and (have <= wanted or wanted <= have) and len(self.entries(value)) > 1:
                    return True
        return False

    def kind_queries(self) -> list[str]:
        """A value the store never files alone under the property's own fact, but only inside longer entries there -
        the kinds of it this shelf sells - searched in those entries' words with the other values: a listing of such
        a kind may carry the value nowhere on its own, and a search finds only the words it is given."""
        out = []
        values = self.values()
        for index, requirement in enumerate(self.requirements):
            if self.property_place(requirement) != "fact":
                continue
            related, inferred, _ = self.own_labels(requirement)
            labels = related | inferred
            words = set(normalized_lower(requirement["value"]).split())
            exact, longer = 0, {}
            for page in self.counter.pages.values():
                if not self.in_category(page):
                    continue
                for label, value in (page.get("facts") or {}).items():
                    if str(label) not in labels:
                        continue
                    for entry in self.entries(value):
                        tokens = entry.split()
                        if self.fact_entry_matches(entry, requirement, str(label), exact=True):
                            exact += 1
                        elif words and words <= set(tokens) and len(tokens) <= len(words) + 2:
                            longer[entry] = longer.get(entry, 0) + 1
            if exact or not longer:
                continue
            others = " ".join(v for i, v in enumerate(values) if i != index)
            for entry, count in sorted(longer.items(), key=lambda kv: (-kv[1], kv[0]))[:2]:
                if count >= 2 and entry not in self.kinds_tried:
                    self.kinds_tried.add(entry)
                    out.append(f"{others} {entry}".strip())
        return out

    def scarce_queries(self) -> list[str]:
        """While nothing fits: a value that far fewer of the shelf's listings state than another is the one hard to come
        by, and the searches with the values keep bringing up the many listings that state the commoner one. A listing
        that states the scarce value and misses one other is searched by its own title with that other value: its other
        listings and variants may state both."""
        if len(self.requirements) < 2 or any(self.listing_fits(k) for k in self.verdicts):
            return []
        # (the value searched with and the title, kept from the first call on and apart from the probed titles: a listing
        # that lacks a value in its facts lacks it in every listing of the same name, so those rows earn no place ahead
        # of others; a value the shopper names later is searched with the same titles again)
        tried = self.__dict__.setdefault("scarce_tried", set())
        shelf = []
        for key, verdict in self.verdicts.items():
            if verdict.get("kind_ok"):
                if len(verdict.get("levels") or []) != len(self.requirements):
                    verdict["levels"] = [self.plainness_level(key, r) for r in self.requirements]     # (read once for each verdict)
                shelf.append((key, verdict["levels"]))
        stating = [len({key[0] for key, levels in shelf if levels[i] >= 1}) for i in range(len(self.requirements))]
        scarce, after = sorted(range(len(stating)), key=lambda i: stating[i])[:2]
        if 3 * stating[scarce] > stating[after]:
            return []           # (the scarcest has to be clearly scarcer: a third as many listings state it, or fewer)
        out = self.scarce_mirror(shelf, scarce, tried)
        for key, levels in sorted(shelf, key=lambda pair: -sum(pair[1])):
            off = [i for i, level in enumerate(levels) if level < 1]
            if levels[scarce] < 1 or len(off) != 1:
                continue
            value = join_number_unit(self.requirements[off[0]]["value"])
            title = normalized_lower((self.counter.pages.get(key[0]) or {}).get("title"))
            words = letter_digit_tokens(title)
            if len(words) < 2 or (value, title) in tried or sum(1 for v, _ in tried if v == value) >= 8:
                continue
            tried.add((value, title))
            out.append(f"{value} {' '.join(words[:10])}")
        return out

    def scarce_mirror(self, shelf: list, scarce: int, tried: set) -> list[str]:
        """The scarce value searched with the title head of each affordable listing that meets every other value and
        misses only that one: another rating or size of the same line is usually a sibling listing. Those whose name or
        options state an amount in the scarce value's unit first; six titles at most."""
        value_text = self.requirements[scarce]["value"]
        value = join_number_unit(value_text)
        unit = (re.findall(r"\d\s*([a-z]+)", normalized_lower(value_text)) or [""])[0]
        heads = []
        for key, levels in shelf:
            off = [i for i, level in enumerate(levels) if level < 1]
            if off != [scarce] or not self.affordable(key):
                continue
            title = normalized_lower((self.counter.pages.get(key[0]) or {}).get("title"))
            words = letter_digit_tokens(title)
            if len(words) < 2 or (value, title) in tried:
                continue
            said = f"{title} {normalized_lower(self.variant_of(key).get('options'))}"
            same_unit = bool(unit) and bool(re.search(r"\d\s*" + re.escape(unit) + r"\b", said))
            heads.append((not same_unit, -sum(levels), title, words))
        out = []
        for _, _, title, words in sorted(heads)[:6]:
            if sum(1 for v, _ in tried if v == value) >= 8:
                break
            tried.add((value, title))
            query = f"{value} {' '.join(words[:6])}"
            if query not in out:
                out.append(query)
        return out

    def variant_queries(self) -> list[str]:
        """A search shows one variant of a listing. When a listing meets everything except what differs between
        its variants (the options shown name another colour or amount), the wanted variant may stand behind the
        one shown: the listing's title with the wanted value brings it up. A listing that misses one value only,
        and that one because the options shown name another, goes first, and even once the others have had their
        turn: its wanted variant is the likeliest to stand behind the one shown."""
        out = []

        def probe(key: tuple[str, str], off: list[dict]) -> None:
            title = normalized_lower((self.counter.pages.get(key[0]) or {}).get("title"))
            if not title or title in self.probed:
                return
            self.probed.add(title)
            words = " ".join(letter_digit_tokens(title)[:10])
            for requirement in off:
                spelled = [self.close_spelling(requirement)] if self.close_spelling(requirement) else []
                for word in [join_number_unit(requirement["value"])] + spelled + list(requirement.get("alike") or [])[:2]:
                    out.append(f"{word} {words}")
        # (first the listings that miss one requirement only, a colour or an amount the shown variant's options name
        # otherwise: the plainest and cheapest first, three a round - a listing read later gets its turn - and up to
        # twice as many titles as the rest. Each listing's levels are read once for its verdict)
        lone = []
        for key, verdict in self.verdicts.items():
            if not verdict.get("kind_ok") or verdict.get("by") != "words" or not self.requirements:
                continue
            if len(verdict.get("levels") or []) != len(self.requirements):
                verdict["levels"] = [self.plainness_level(key, r) for r in self.requirements]
            off = [(r, level) for r, level in zip(self.requirements, verdict["levels"]) if level < 1]
            if len(off) != 1:
                continue
            options = str(self.variant_of(key).get("options") or "")
            missed, level = off[0]
            if options.strip() and (self.property_place(missed) == "colour" and level == -1 or self.property_place(missed) == "amount"
                                    and (self.other_amount(options, missed) or self.named_otherwise(options, missed))):
                lone.append((-sum(verdict["levels"]), (self.counter.stock.get(key) or {}).get("price") or 0.0, key, missed))
        ceiling = min(8, len(self.probed) + 3)
        for _, _, key, missed in sorted(lone, key=lambda entry: entry[:2]):
            if len(self.probed) >= ceiling:
                break
            probe(key, [missed])
        for key, verdict in list(self.verdicts.items()):
            if len(self.probed) >= 4:
                break
            if not verdict.get("kind_ok") or verdict.get("by") != "words" or not self.requirements:
                continue
            levels = [self.plainness_level(key, r) for r in self.requirements]
            off = [r for r, level in zip(self.requirements, levels) if level < 1]
            rest = [level for r, level in zip(self.requirements, levels) if level >= 1]
            if not off or len(rest) != len(self.requirements) - len(off):
                continue
            if any(self.property_place(r) not in ("colour", "amount") for r, level in zip(self.requirements, levels) if level < 1):
                continue
            probe(key, off)
        return out

    def label_guesses(self) -> list[str]:
        """The item words as a category name, with plural endings as stores write them; a describing word in
        front may not belong to the name, so the same is tried without the first word or two."""
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
        """The category label the store answers to: one a filter returned rows for, or the one read on a page."""
        named = self.named_shelf()
        if named:
            return named
        if self.counter.shelf:
            return next(iter(self.counter.shelf.values()))
        wanted = {singular_phrase_key(self.category), singular_phrase_key(self.item)} - {""}
        for page in self.counter.pages.values():
            for label in page.get("category_path") or []:
                if singular_phrase_key(label) in wanted:
                    return str(label)
        return ""

    def shelf_filters(self, count: int = 4) -> list[dict]:
        """The category at several price ceilings: each ceiling shows other listings of the shelf."""
        label = self.store_label()
        if not label or not self.counter.tool_available("filter") or self.budget is None:
            return []
        out = []
        for share in (0.7, 0.45, 0.25, 0.12, 0.85, 0.55, 0.35, 0.18, 0.08, 0.04):
            tag = f"filter|{normalized_lower(label)}|{share}"
            if tag not in self.sent and len(out) < count and len(self.sent) < self.search_room:
                self.sent.add(tag)
                out.append(make_tool_call("filter", category=label, max_price=round(self.budget * share, 2), in_stock=True, k=10))
        return out

    def said_in_request(self, text: str) -> bool:
        """Every word of the text stands in the shopper's request (a reader's guess at a brand is no wish)."""
        said = set(letter_digit_tokens(normalized_lower(self.counter.request)))
        words = letter_digit_tokens(normalized_lower(text))
        return bool(words) and all(w in said for w in words)

    def queue_filters(self) -> None:
        """The shelf under the present budget, and under the brand the shopper requires, to send with the next searches."""
        brand = next((r["value"] for r in self.requirements if "brand" in normalized_lower(r["property"])), None)
        label = self.store_label()
        for category in ([label] if label else self.label_guesses()[:2]):
            self.pending_filters.append(make_tool_call("filter", category=category, brand=brand, max_price=self.budget, in_stock=True, k=10))
        if brand:
            self.pending_filters.append(make_tool_call("filter", brand=brand, max_price=self.budget, in_stock=True, k=10))
        elif self.brand_wish and not self.wish_filtered:
            # (a brand only preferred: its listings on the shelf come into view once; it stays no requirement)
            self.wish_filtered = True
            for category in ([label] if label else self.label_guesses()[:1]):
                self.pending_filters.append(make_tool_call("filter", category=category, brand=self.brand_wish, max_price=self.budget, in_stock=True, k=10))

    def next_searches(self, room: int) -> list[dict]:
        out = self.pending_filters[:room] if self.counter.tool_available("filter") else []
        self.pending_filters = self.pending_filters[len(out):] if out else []
        while self.queries and len(out) < room and len(self.sent) < self.search_room:
            query = self.queries.pop(0)
            if query in self.sent:
                continue
            self.sent.add(query)
            out.append(make_tool_call("search", query=query, k=10, in_stock=True))
        return out

    # ---- the shopper's lines
    @staticmethod
    def listed_fields(answer: dict | None) -> dict:
        """A reader's answer with its lists as lists (a model sometimes writes a list as text)."""
        answer = dict(answer or {})
        for name in ("adds", "drops", "queries"):
            if isinstance(answer.get(name), str):
                try:
                    answer[name] = json.loads(answer[name])
                except ValueError:
                    answer[name] = []
            if not isinstance(answer.get(name), list):
                answer[name] = []
        return answer

    @staticmethod
    def spilled_answer(reading: Any) -> bool:
        """A reader's answer whose names carry pieces of the answer's own structure: quotes, braces or a field's key."""
        if not isinstance(reading, dict):
            return False
        names = [d for d in reading.get("drops") or [] if isinstance(d, str)] + [
            str(a.get("property") or "") for a in reading.get("adds") or [] if isinstance(a, dict)]
        return any(re.search(r"[\"{}\[\]]|\b(?:adds|drops|budget|asks|queries)\b\s*[\"']?\s*:", name) for name in names)

    def absorb_shopper_lines(self) -> tuple[bool, bool]:
        """Take in the shopper lines received. Returns (something changed, a line expects an answer)."""
        changed = asks = False
        while self.counter.heard:
            line = self.counter.heard.pop(0)
            parts = [part.strip() for part in line.split("\n") if part.strip()]
            if len(parts) > 1:
                # several lines in one turn: each is heard on its own, in order (only the first answers a question)
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
                # asked something it holds nothing back about, the shopper only repeats the limit already known
                # (and the store's shelf the item is on, in the store's own word for it)
                heard = self.listed_fields(None)
                if line != self.shelf_said:
                    self.shelf_said, self.shelf_cache = line, (-1, "")
            else:
                reading = self.reader.request_tool_fill(HEAR_GUIDE, said, HEAR_TOOL, 1500)
                if not isinstance(reading, dict) or not any(k in reading for k in ("adds", "drops", "budget", "asks")):
                    # (no reader answered at all - a failed call, not a line that changes nothing: it is read once more)
                    reading = self.reader.request_tool_fill(HEAR_GUIDE, said, HEAR_TOOL, 1500)
                if self.spilled_answer(reading):
                    # (an answer whose fields spill into one another - a property name holding the rest of the answer -
                    # has lost what it was meant to say, a new limit among it: the other model reads the line)
                    again = self.reader.request_tool_fill(HEAR_GUIDE, said, HEAR_TOOL, 1500, second=True)
                    if isinstance(again, dict) and not self.spilled_answer(again):
                        reading = again
                heard = self.listed_fields(reading)

                swap = re.search(r"\binstead of\b", normalized_lower(line))

                def real_adds(reading: dict) -> list[dict]:
                    """The requirements a reading found, without those that cannot be: the line's own word for
                    "this is needed" taken as the value, or - on a line that puts one value in place of another -
                    the value given up taken as the new one (it stands only after "instead of")."""
                    kept = []
                    for a in reading.get("adds") or []:
                        value = normalized_lower(a.get("value")) if isinstance(a, dict) else ""
                        if not isinstance(a, dict):
                            continue
                        if re.fullmatch(SAID_NEEDED, value):
                            continue
                        named = {w for n in self.asked[-4:] for w in self.word_stem_set(n)}
                        if self.awaiting_answer and value and (self.word_stem_set(value) <= named or self.word_stem_set(value) == self.word_stem_set(self.item)):
                            continue        # (the property's own name, or the item's whole name, read as its value; one
                                            # word of a longer item name - the primer of "paint, primer & solvent" - is a value)
                        if swap and value and value in normalized_lower(line)[swap.end():] and value not in normalized_lower(line)[:swap.start()]:
                            continue
                        if (not self.awaiting_answer and len(letter_digit_tokens(value)) >= 2
                                and self.word_stem_set(value) <= self.word_stem_set(str(a.get("property") or ""))):
                            continue        # (a value that only says its property's name again - "make sure it still has the
                                            # lid" read as lid = lid - names no value: read once more)
                        kept.append(a)
                    return kept
                found = [a for a in heard.get("adds") or [] if isinstance(a, dict)]
                if len(real_adds(heard)) < len(found) or (self.awaiting_answer and not found):
                    # the first reading found no requirement in a line that answers the question just asked, or one
                    # that cannot be: the line is read once more
                    again = self.listed_fields(self.reader.request_tool_fill(HEAR_GUIDE, said, HEAR_TOOL, 1500, second=True))
                    if real_adds(again):
                        heard = again
                if any(isinstance(a, dict) and re.fullmatch(SAID_NEEDED, normalized_lower(a.get("value"))) for a in heard.get("adds") or []):
                    # both readings took the line's word for "this is needed" as the value: the value then stands at the
                    # end of what was read as the property's name ("the handle type folding must be included")
                    mended = []
                    for a in heard["adds"]:
                        words = str(a.get("property") or "").split() if isinstance(a, dict) else []
                        if words and re.fullmatch(SAID_NEEDED, normalized_lower(a.get("value"))) and len(words) >= 2 and self.says_value(line, words[-1]):
                            a = dict(a, property=" ".join(words[:-1]), value=words[-1])
                        mended.append(a)
                    heard["adds"] = mended
                if heard.get("adds") and not real_adds(heard) and self.awaiting_answer:
                    # both readings took the label for the value. A shopper may say the value straight after the
                    # property's name: what is left of that phrase without the name's own words is the value
                    said_so = re.search(r"\bthe ([a-z0-9 \-]+?) (?:is|are) (?:required|needed|necessary|essential|a must)\b", normalized_lower(line))
                    named = self.word_stem_set(self.item) | {w for n in self.asked[-4:] for w in self.word_stem_set(n)}
                    left = [w for w in letter_digit_tokens(said_so.group(1)) if singular_form(w) not in named and w not in ("of", "a", "an")] if said_so else []
                    asked_for = next((n for n in reversed(self.asked[-4:]) if self.word_stem_set(n) & self.word_stem_set(said_so.group(1))), "") if said_so else ""
                    if left and len(left) <= 3 and asked_for:
                        heard["adds"] = [{"property": asked_for, "value": " ".join(left)}]
                if heard.get("adds"):
                    heard["adds"] = real_adds(heard)
                if (self.awaiting_answer and self.pair_open and heard.get("priority")
                        and not any(isinstance(a, dict) and self.is_wish(str(a.get("value") or "")) for a in heard.get("adds") or [])):
                    # the answer to which of two wishes matters more, read as what matters most between fitting items
                    # and with no wish among its requirements: the wish that reading names is the one chosen
                    chosen = self.wish_chosen(line, heard.get("priority"))
                    if chosen and not any(isinstance(a, dict) and normalized_lower(a.get("property")) == normalized_lower(chosen["property"])
                                          for a in heard.get("adds") or []):
                        # (a line that gives that property another value gives the wish up: it is not the one chosen)
                        heard["adds"] = list(heard.get("adds") or []) + [chosen]
                        self.record_note(f"pair answer read as a priority ({heard.get('priority')!r}): the wish chosen is {chosen}")
            answering, self.awaiting_answer = self.awaiting_answer, False
            # an answer that states nothing at all: no requirement, no limit, no change
            nothing = answering and not any(heard.get(k) for k in ("adds", "drops", "budget", "priority", "about_change"))
            if nothing and len(self.current_names) == 2:
                heard["asks"] = False       # (it wants the two put one at a time: the next questions are the reply)
            given = replaced = 0
            withdrawn = []
            if heard.get("about_change") is True:
                # a line reacting to the market change: its figure is a price, not a budget (a requirement it states still counts)
                heard["budget"] = 0
            # (the answer to which of two wishes matters more chooses between the wishes: what a purpose was said to need
            # stays, even where the chosen wish is of the same property)
            choosing = answering and self.pair_open and any(isinstance(a, dict) and self.is_wish(str(a.get("value") or "")) for a in heard.get("adds") or [])
            # (a line that adds a value beside one already wanted - also, as well, too - with no word of changing anything,
            # for a property this shelf files as a list of entries, takes nothing back: both values are wanted)
            # (so does one that says the value is to be included, or that everything else stays as it was)
            besides = bool(re.search(r"\b(?:also|as well|too|in addition|besides|plus|includ\w*|one more thing)\b"
                                     r"|\b(?:everything else|the rest|all else)\b[^.!?]{0,24}\b(?:stays?|same|holds?|unchanged)\b", normalized_lower(line))
                           and not re.search(r"\b(?:instead|chang\w+|switch\w*|swap\w*|forget|no longer|rather than|replac\w+|remov\w+|drop\w*)\b", normalized_lower(line)))
            kept_beside = {normalized_lower(a.get("property")) for a in heard.get("adds") or []
                           if besides and isinstance(a, dict) and self.filed_as_list(str(a.get("property") or ""))}
            # (an answer to the question that gives its reason - because / since / as this is for ... - with no word of
            # changing anything tells what a purpose needs: it takes nothing back)
            reasoned_answer = answering and bool(re.search(r"^\s*as\b|\b(?:because|since|given that)\b", normalized_lower(line))) and not re.search(
                r"\b(?:instead|chang\w+|switch\w*|swap\w*|forget|no longer|rather than|replac\w+|remov\w+|drop\w*)\b", normalized_lower(line))
            for name in heard.get("drops") or []:
                if normalized_lower(name) in kept_beside or reasoned_answer:
                    continue
                # (a property named as the shelf files it - milk type, where the request said fat content for 'low fat' and
                # the shelf's listings file that value under milk_type: the line speaks of that requirement by its label)
                by_label = [] if any(self.same_property(r, name) for r in self.requirements) else [
                    r for r in self.requirements if self.property_place(r) == "fact" and self.word_stem_set(name)
                    and any(self.word_stem_set(label) == self.word_stem_set(name) for label in set().union(*self.own_labels(r)[:2]))]
                if len(by_label) == 1 and any(isinstance(a, dict) for a in heard.get("adds") or []):
                    self.record_note(f"the line drops {name!r}: the shelf files {by_label[0]['value']!r} under that label")
                    withdrawn += by_label
                    replaced += 1
                    changed = True
                    self.requirements = [r for r in self.requirements if r is not by_label[0]]
                    continue
                # (a line withdraws only what it speaks of: a reader's slip on a line about stock drops nothing - but a
                # line that says outright it changes something and gives the new value replaces the property the reader
                # names, even where it no longer repeats the old value: "changing the lamp type to pendant")
                changing = bool(re.search(r"\b(?:chang\w*|switch\w*|instead|replac\w*|swap\w*|forget|no longer|rather than)\b", normalized_lower(line))
                                and any(isinstance(a, dict) and str(a.get("value") or "").strip() for a in heard.get("adds") or []))
                kept = [r for r in self.requirements if not self.same_property(r, name) or (not self.mentions(line, r) and not changing)
                        or (choosing and r.get("told"))
                        or any(self.same_property(r, a.get("property")) and self.said_again(line, r, str(a.get("value") or ""), answering)
                               for a in heard.get("adds") or [] if isinstance(a, dict))]
                changed = changed or len(kept) != len(self.requirements)
                replaced += len(self.requirements) - len(kept)
                withdrawn += [r for r in self.requirements if r not in kept]
                self.requirements = kept
            # (a new spending limit read as an added value is no value put in place of a requirement: a line that names
            # a limit and says what is STILL wanted takes nothing back)
            added = [normalized_lower(a.get("value")) for a in heard.get("adds") or [] if isinstance(a, dict) and str(a.get("value") or "").strip()
                     and not NOT_A_PROPERTY.search(normalized_lower(f"{a.get('property')} {a.get('value')}"))]
            if heard.get("drops") and added and not withdrawn and not reasoned_answer:
                # the line withdraws something under a name no requirement carries (the request was read with another
                # name for the property): what goes is the requirement whose value the line says and does not keep
                said = set(singular_phrase_key(line).split())
                gone = [r for r in self.requirements if normalized_lower(r["value"]) not in added and singular_phrase_key(r["value"])
                        and set(singular_phrase_key(r["value"]).split()) <= said and self.says_value(line, r["value"])
                        and not re.search(r"\bstill\b[^.!?]*" + re.escape(normalized_lower(r["value"])), normalized_lower(line))]
                if len(gone) == 1:
                    self.requirements = [r for r in self.requirements if r is not gone[0]]
                    withdrawn += gone
                    replaced += 1
                    changed = True
            if withdrawn and not added:
                # the line takes an amount back and the reader gave nothing in its place: another amount the line
                # writes in the same unit is the new one ("drop the 2 l size, make it 750ml")
                for old in withdrawn:
                    was = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)", normalized_lower(old["value"]).strip())
                    if not was:
                        continue
                    others = [m for m in re.finditer(r"(?<![a-z0-9.,])(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", normalized_lower(line))
                              if self.same_unit(self.unit_word(m.group(2)), self.unit_word(was.group(2))) and float(m.group(1)) != float(was.group(1))]
                    if len({float(m.group(1)) for m in others}) == 1:
                        heard["adds"] = list(heard.get("adds") or []) + [{"property": old["property"], "value": f"{others[0].group(1)} {was.group(2)}"}]
                        self.record_note(f"the line's own new amount for {old['property']!r}: {others[0].group(1)} {was.group(2)}")
            for add in heard.get("adds") or []:
                if not isinstance(add, dict) or not str(add.get("value") or "").strip():
                    continue
                new = self.requirement(add, line)
                if NOT_A_PROPERTY.search(normalized_lower(f"{new['property']} {new['value']}")):
                    continue
                if not self.says_value(line, new["value"]):
                    continue            # a value the line itself does not say is the reader's, not the shopper's
                if not re.search(r"\d", new["value"]) and any(self.within(new["value"], purpose) for purpose in self.purposes):
                    continue            # the use the line gives its requirement for (words of a purpose), not a requirement
                stands_now = next((now for now, before in self.took_back.items() if normalized_lower(before) == normalized_lower(new["value"])), None)
                if stands_now is not None and not self.says_value(line, stands_now):
                    continue            # a line written before the shopper's change, saying the old value "still" holds
                same = next((r for r in self.requirements if self.same_property(r, new["property"])), None)
                if same is not None and singular_phrase_key(same["value"]) == singular_phrase_key(new["value"]):
                    continue            # (the value said again, in another spelling or not: '&' for 'and', a plural)
                if same is not None and self.said_again(line, same, new["value"], answering):
                    continue            # (said again in other words, with no word of changing it: a line that
                                        # still needs the value, said more briefly, restates what holds)
                if same is not None and normalized_lower(new["property"]) in kept_beside:
                    same = None             # (added beside the earlier value, not in its place)
                if same is not None and answering and self.pair_open and self.is_wish(new["value"]) and same.get("told"):
                    # (the side of a doubted pair the shopper chose, of the same property as what a purpose was said to
                    # need: both hold - the choice adds to that need, it does not replace it)
                    same = None
                if same is not None and besides and not answering and not self.mentions(line, same):
                    # (a line that adds a value beside what is wanted - also, too - with no word of changing anything,
                    # naming neither that property nor its value: a requirement of its own, whatever name the reader
                    # gave it, and nothing is taken back)
                    new["property"], same = "requirement", None
                if same is None and any(singular_phrase_key(r["value"]) == singular_phrase_key(new["value"]) for r in self.requirements):
                    continue            # the same value again under another name for the property: nothing new
                if same is not None and answering and not self.mentions(line, same):
                    # the answer to a question names neither that property nor its value: it adds a requirement of
                    # its own and takes nothing back, whatever name the reader gave it
                    new["property"], same = "requirement", None
                if (same is not None and answering and same.get("stated")
                        and not re.search(r"\b(?:instead|chang\w+|switch\w*|swap\w*|forget|no longer|rather than|replac\w+)\b", normalized_lower(line))):
                    # the answer to the question just put, with no word of changing anything, names a value for a
                    # property the request was read to fix already: that reading was the reader's (most often the
                    # item's own name taken for a detail). What the shopper says is what was held back - it takes the
                    # other's place without a word, and counts as the answer it is
                    self.requirements.remove(same)
                    self.opening = [r for r in self.opening if normalized_lower(r["value"]) != normalized_lower(same["value"])]
                    same = None
                if same is not None:
                    self.requirements.remove(same)
                    replaced += 1           # a new value for a property already known: a change, not an answer
                    withdrawn.append(same)
                else:
                    given += 1
                    new["told"] = answering and not self.is_wish(new["value"])     # (what a purpose was said to need)
                # what the shopper has just withdrawn is no other word for what replaces it
                old = {join_number_unit(w) for r in withdrawn for w in [r["value"]] + list(r.get("written") or []) + list(r.get("alike") or [])}
                for kind in ("written", "alike"):
                    if isinstance(new.get(kind), list):
                        new[kind] = [w for w in new[kind] if join_number_unit(str(w)) not in old]
                if answering and self.pair_open and self.is_wish(new["value"]):
                    new["chosen"] = True        # the side of the doubted pair the shopper said matters more
                    if not any(normalized_lower(o["value"]) == normalized_lower(new["value"]) for o in self.opening):
                        # (a request that doubts a pair asked for one of the two: the first cart states the one chosen)
                        self.opening.append(dict(new))
                instead_of = [r for r in withdrawn if self.same_property(r, new["property"])] or (withdrawn if len(withdrawn) == 1 else [])
                unit_before = re.fullmatch(r"\d+(?:\.\d+)?\s*([a-z][a-z ]*)", normalized_lower(instead_of[0]["value"]).strip()) if instead_of else None
                if unit_before and re.fullmatch(r"\d+(?:\.\d+)?", new["value"].strip()):
                    # ("12 instead of 40": the new figure counts what the old one counted - 12 pcs for 40 pcs)
                    new["value"] = f"{new['value'].strip()} {unit_before.group(1).strip()}"
                    new["filed"] = False
                if instead_of and re.search(r"\d", new["value"]):
                    # (a new amount for one also written in another unit - 10 cm for 25 cm, which was written in inches
                    # too - is written in that unit as well: a shelf that writes the measure only so states it there)
                    new["written"] = list(dict.fromkeys(list(new.get("written") or []) + self.spelled_as_before(new, instead_of[0])))
                if instead_of and normalized_lower(instead_of[0]["value"]) != normalized_lower(new["value"]):
                    self.took_back[normalized_lower(new["value"])] = instead_of[0]["value"]
                self.requirements.append(new)
                changed = True
                if "brand" in normalized_lower(new["property"]):
                    # a brand the shopper now requires: ask the store for it directly
                    self.queue_filters()
                    self.add_queries([f"{new['value']} {self.item}", new["value"]], first=True)
            if sorted((normalized_lower(p), singular_phrase_key(v)) for p, v in before) == sorted((normalized_lower(r["property"]), singular_phrase_key(r["value"])) for r in self.requirements):
                # the line only repeats what already holds: nothing changed, and nothing new was given
                self.requirements, changed, given, replaced = held, changed_so_far, 0, 0
            else:
                self.fresh[line] = [r["value"] for r in self.requirements if r not in held]
            budget = positive_amount(heard.get("budget"))
            if budget is not None and not self.says_amount(line, budget):
                budget = None               # a limit the line itself does not state is the reader's, not the shopper's
            if budget is not None and any(abs(budget - float((self.counter.stock.get(k) or {}).get("price") or -1.0)) < 0.005 for k in self.counter.cart):
                budget = None               # the price of an item in the cart, not a new limit
            if budget is not None and budget != self.budget:
                lowered_limit = self.budget is None or budget < self.budget
                self.budget, changed = budget, True
                self.fresh_limit.add(line)
                if lowered_limit:
                    self.queue_filters()        # the shelf again, under the new limit
            told_priority = self.grounded_priority(heard.get("priority"), line)
            # (the shopper saying which of two wishes matters more - the property that matters most to them - answers the
            # pair question: it names a requirement, not what makes one fitting item better than another, and the
            # priority the request gave stands)
            if told_priority and ((answering and self.pair_open)
                                  or any(self.content_words(told_priority) and self.content_words(told_priority) <= self.word_stem_set(r["property"])
                                         for r in self.requirements)):
                told_priority = ""
            if told_priority:
                self.priority, changed = told_priority, True
                self.priority_measure = ""                  # (a new priority: its measure is not read)
                self.plan_units, self.unit_settled = [], False
                self.figures, self.figure_unit = {}, ""     # figures read for the earlier priority say nothing about this one
            moved = [(r["property"], r["value"]) for r in self.requirements] != before or self.budget != limit_before
            wants_reply = heard.get("asks") is True and heard.get("about_change") is not True and not self.owed_reaction
            # with a question outstanding, a line that changes something already known (another value, a new limit,
            # the market) is the shopper speaking of something else, not the answer
            spoke_of_else = replaced > 0 or self.budget != limit_before or heard.get("about_change") is True
            if (replaced > 0 or self.budget != limit_before) and not self.counter.cart:
                self.changed_early = True
            if replaced > 0 or self.budget != limit_before or (moved and not answering):
                self.moved_on = True        # the shopper changed what they want, or added to it unasked
            was_answer = answering and not spoke_of_else
            if (wants_reply or (moved and not was_answer)) and line not in self.unanswered:
                # the shopper changed what they want (an item's property or the spending limit) or asked for an
                # answer: the line gets a plain reply before the order, wherever in the episode it was said
                self.unanswered.append(line)
            if answering:
                self.answered(line, given, spoke_of_else=spoke_of_else, about=" ".join(letter_digit_tokens(normalized_lower(heard.get("about")))), nothing=nothing)
            elif given and not replaced and re.search(r"^\s*as\b|\b(?:because|since|given that)\b", normalized_lower(line)):
                # (told unasked, with a purpose of the request as its reason: that purpose's requirement is out)
                said = self.content_words(line)
                shared = {i: len(self.content_words(self.purposes[i]) & said) for i in range(min(2, len(self.purposes)))
                          if i not in self.purposes_met}
                met = max(shared, key=lambda i: shared[i], default=None)
                if met is not None and shared[met] * 2 >= max(1, len(self.content_words(self.purposes[met]))):
                    self.purposes_met.add(met)
                    self.needs_found += 1
                    self.lead_mark = self.property_asks
                    self.first_found_at = self.first_found_at or self.questions_asked
                    self.asked_log.append(("(told without a question)", f"the shopper said: {line[:160]}"))
            if changed:
                self.changed_at = self.counter.turn
                gone_words = [set(letter_digit_tokens(join_number_unit(r["value"]))) for r in withdrawn if r not in self.requirements]
                kept_words = {w for v in self.values() for w in letter_digit_tokens(v)}
                gone_words = [words - kept_words for words in gone_words if words - kept_words]
                if gone_words:
                    # a search written for a value the shopper has just taken back finds what they no longer want: one
                    # that carries any telling word of it (not a word of the item's own name) is written for it
                    named = self.content_words(f"{self.item} {self.category}")
                    telling = [{w for w in words if (len(w) >= 4 or re.search(r"\d", w)) and singular_form(w) not in named} for words in gone_words]
                    self.queries = [q for q in self.queries
                                    if not any(words <= set(q.split()) or tell & set(q.split()) for words, tell in zip(gone_words, telling))]
                shade_queries: list[str] = []
                if given or replaced:
                    self.fresh_rounds = 2       # the next looks go for what the shopper wants now, in the shelf's own words
                    self.shelf_words_used = 0   # (from the shelf's most usual words again: they were tried with other values)
                    self.__dict__["shelf_words_done"] = set()
                    for fresh_one in [r for r in self.requirements if r not in held and self.property_place(r) == "colour"]:
                        # (a colour just told: the shelf may file it under a shade, and its searches go out in that word
                        # as soon as the listings read show it - the two likeliest shades at once, since a listing named
                        # only by its shade shows for no search in the colour's own word)
                        shades = [s for s in self.colour_shades(fresh_one) if join_number_unit(s) != join_number_unit(fresh_one["value"])]
                        others = " ".join(v for r, v in zip(self.requirements, self.values()) if r is not fresh_one)
                        head = (self.item_words() or [""])[-1]
                        shade_queries += [" ".join(f"{shade} {others} {self.item}".split()) for shade in shades[:2]]
                        shade_queries += [" ".join(f"{others} {shade} {head}".split()) for shade in shades[:2] if head]
                if replaced:
                    # a word value put in place of another: the words a store files it under are asked for at once, so
                    # the first searches for it carry them, and a fact written in the store's word reads as this value
                    taken_back = {join_number_unit(w) for r in withdrawn for w in [r["value"]] + list(r.get("written") or []) + list(r.get("alike") or [])}
                    for fresh_one in [r for r in self.requirements if r not in held]:
                        if (self.property_place(fresh_one) in ("fact", "material") and "brand" not in normalized_lower(fresh_one["property"])
                                and not fresh_one.get("alike")):
                            self.seller_words(fresh_one)
                            fresh_one["alike"] = [w for w in fresh_one.get("alike") or [] if join_number_unit(str(w)) not in taken_back]
                self.add_queries(self.value_queries() + shade_queries + list(heard.get("queries") or []), first=True)
                # what the shopper wants moved: room to look again, whatever was used before
                self.search_room = max(self.search_room, len(self.sent) + 32)
                self.search_asks = 0
                self.open_room = max(self.open_room, len(self.counter.pages) + 40)
                self.figure_room = max(self.figure_room, len(self.figures) + 3 * JUDGE_CARDS)
                self.judge_room = max(self.judge_room, self.judged_count + 36)
            self.record_note(f"heard {line[:80]!r} -> adds={heard.get('adds')} drops={heard.get('drops')} budget={heard.get('budget')} asks={heard.get('asks')}"
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
                # the shopper spoke of the store's change in the very turn of the reply: a shopper taken up with that
                # does not read a message, so the reply is given once more
                self.replies_again += 1
                for said_line in self.reply_sent[1].split("\n"):
                    text = said_line[2:] if said_line.startswith("- ") else said_line
                    if text.strip() and text not in self.unanswered:
                        self.unanswered.append(text)
                self.reply_sent = None
        return changed, asks

    def same_property(self, requirement: dict, name: Any) -> bool:
        """A property a shopper's line names is this requirement's: the same name, the same words in another order or
        form (lamp type for types of lamp), or the name it carried before it was read under the shelf's label."""
        said = normalized_lower(name)
        if not said:
            return False
        if said == normalized_lower(requirement.get("property")) or (requirement.get("was") and said == normalized_lower(requirement.get("was"))):
            return True
        words, own = self.word_stem_set(said), self.word_stem_set(requirement.get("property"))
        if words and words == own:
            return True
        # (a property named by its kind alone is the item's own kind: "lamp type" on a lamp is that "type")
        item = self.word_stem_set(self.item) | self.word_stem_set(self.named_shelf()) | self.word_stem_set(self.category)
        return bool(own) and own <= {"type", "kind"} and own < words and (words - own) <= item

    def said_again(self, line: str, held: dict, value: str, answering: bool) -> bool:
        """The line, with no word of changing anything, says a held value again: the same amount spelled otherwise (1
        piece for 1 pcs, 2L for 2 l), or a word value in fewer of its words - or, when nothing was asked, in more
        (a line that still needs "the X one" for a held "X Y Z"). An answer that says more than the value held is what
        was asked for, and is read as before."""
        if not value.strip() or re.search(r"\b(?:instead|chang\w+|switch\w*|swap\w*|forget|no longer|rather than|replac\w+)\b", normalized_lower(line)):
            return False
        if re.search(r"\d", held["value"]) or re.search(r"\d", value):
            return (bool(re.search(r"\d", held["value"])) and bool(re.search(r"\d", value))
                    and self.same_amount(held["value"], value) and self.same_amount(value, held["value"]))
        held_words, said_words = set(singular_phrase_key(held["value"]).split()), set(singular_phrase_key(value).split())
        return bool(said_words) and (said_words <= held_words or (held_words <= said_words and not answering))

    def spelled_as_before(self, new: dict, old: dict) -> list[str]:
        """A new amount put in place of one the reader had also written in another unit (inches beside centimetres), heard
        with no spelling in that unit: its spellings there. The reader of other words gives them, told the units; when
        it gives none, the old spellings are scaled as the two amounts are, where that gives plain figures. [] when the
        new amount has such a spelling already, or the old one had none."""
        def amount(text: Any) -> tuple[float, str] | None:
            found = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]+)", normalized_lower(text).strip())
            return (float(found.group(1)), self.unit_word(found.group(2))) if found and float(found.group(1)) > 0 else None

        def foreign(requirement: dict, unit: str) -> list[tuple[str, tuple[float, str]]]:
            # (the reader's spellings in a unit that is not the value's own, nor another metric step of it)
            out = []
            for form in requirement.get("written") or []:
                one = amount(form)
                if one and not self.same_unit(one[1], unit) and not self.scaled(unit, one[1], any_step=True):
                    out.append((str(form), one))
            return out
        now, was = amount(new["value"]), amount(old["value"])
        if not now or not was:
            return []
        step = 1.0 if self.same_unit(now[1], was[1]) else self.scaled(now[1], was[1], any_step=True)
        before = foreign(old, was[1])
        if not step or not before or foreign(new, now[1]):
            return []
        units = list(dict.fromkeys(re.sub(r"[\d.,\s-]+", " ", normalized_lower(form)).strip() for form, _ in before))
        if self.can_think():
            answer = self.reader.request_tool_fill(WORDS_GUIDE, f"Item: {self.item}\nProperty: {new['property']}\nValue: {new['value']}\n"
                                                   f"Write it in these units too: {', '.join(units)}", WORDS_TOOL, 200) or {}
            words = [str(w).strip() for w in answer.get("words") or [] if str(w).strip()][:6]
            given = [w for w in words if any(self.same_unit((amount(w) or (0.0, ""))[1], u) for _, (_, u) in before)]
            if given:
                self.record_note(f"the new amount {new['value']!r} in the units it was written in before: {given}")
                return given
        scaled = []
        for form, (figure, _) in before:
            # (the old spelling scaled as the amounts are: 2 inch for 5.08 cm gives 4 inch for 10.16 cm)
            number = round(figure * now[0] * step / was[0], 6)
            if number > 0 and abs(number * 100 - round(number * 100)) < 1e-6:
                scaled.append(re.sub(r"\d+(?:\.\d+)?", f"{number:g}", normalized_lower(form), count=1))
        if scaled:
            self.record_note(f"the new amount {new['value']!r} scaled into the units it was written in before: {scaled}")
        return list(dict.fromkeys(scaled))

    def restates_budget(self, line: str) -> bool:
        """The line only names the spending limit already known, to the cent: no other figure stands in it and it
        speaks of no requirement. Such a line says nothing new about the item."""
        if self.budget is None or f"{self.budget:.2f}" not in line.replace(",", ""):
            return False
        figures = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", line.replace(",", ""))]
        if any(abs(figure - self.budget) > 0.005 for figure in figures):
            return False
        return not any(self.mentions(line, r) for r in self.requirements + self.set_aside)

    def answered(self, line: str, given: int, spoke_of_else: bool, about: str = "", nothing: bool = False) -> None:
        """Book what came of the question just asked: a requirement, nothing, or a line about something else."""
        question, name, other = self.last_question, self.current_ask, self.current_other
        named = list(self.current_names) or [n for n in (name, other) if n]
        self.current_ask = self.current_other = ""
        self.current_names = []
        if line and line in self.answers:
            # the shopper only said the earlier answer again: the question named, in other words, the property that
            # answer was about (or spoke of the use, which that answer was given for). It tells nothing about what
            # else is held back: the next properties in line are asked about, without the words about the use
            self.asked_log.append((question, "the same answer as before"))
            self.use_unheard = True
            if question in self.by_use and self.leading is not None and self.leading not in self.use_twice and self.leading not in self.purposes_met:
                self.use_again.add(self.leading)    # (the answer of another purpose: this one is asked about once more, in other words)
            return
        if given and not spoke_of_else:
            self.answers.append(line)
            said = self.word_stem_set(line)
            told = self.requirements[-given:]
            kinds = {self.property_place(r) for r in told}
            units = {self.unit_word(m.group(2)) for r in told for form in self.forms(r) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}

            def about_this(n: str) -> bool:
                """The answer is about this property: it names it (or the shopper's requirements do), the value is of
                its kind - a colour, a material, an amount in the unit it is measured in - or the reader says so."""
                words = self.word_stem_set(n)
                if words <= said or any(words <= self.word_stem_set(r["property"]) or self.word_stem_set(r["property"]) <= words for r in self.requirements):
                    return True
                if ("colour" in kinds and re.search(r"colou?r", n)) or ("material" in kinds and "material" in n):
                    return True
                return bool(self.measure_units.get(n, set()) & units) or " ".join(letter_digit_tokens(n)) == about

            group = named if len(named) >= 2 else []
            unsaid = [n for n in group if not about_this(n)]
            by_the_use = bool(group) and len(unsaid) == len(group) and self.spoke_of_use
            apart = bool(group) and len(unsaid) == len(group) and not by_the_use
            for n in unsaid:
                # the shopper answers one thing at a time: a property the question named and the answer was not about
                # may be held back still. When the answer shows neither as its own: the words about the use may have
                # drawn it (then neither was answered: both are asked about once more, first, without those words);
                # in a question without them one of the two was meant under another name, and which one is not known
                self.asked = [p for p in self.asked if p != n]
                self.again.append(n)
            if self.spoke_of_use:
                self.use_unheard = True     # the use has had its answer: saying it again would only draw that answer again
            met = self.purpose_of(line, self.leading)
            unseen = self.reason_beyond(line)
            if unseen and len(self.purposes) < 2:
                # the answer gives its reason in words no purpose the request was read for carries ("because it's for
                # a list someone wrote, ..."): the reader of the request took two purposes for one. That reason is a
                # purpose of its own, it held this detail back, and the purpose read before may still hold its own
                self.purposes = self.purposes + [unseen]
                self.needs_expected = max(self.needs_expected, min(2, len(self.purposes)))
                met = len(self.purposes) - 1
                self.record_note(f"the answer names a purpose of its own: {unseen!r}")
            asked_for = self.leading if question in self.by_use else None
            if asked_for is not None and asked_for < len(self.purposes):
                mine, spoken = self.content_words(self.purposes[asked_for]), self.content_words(line)
                # (the line gives its reason - "because / since / given that this is for ..." - or carries another
                # purpose of the request nearly whole)
                reasoned = bool(re.search(r"^\s*as\b|\b(?:because|since|given that|given)\b", normalized_lower(line)))
                others = [self.content_words(p) for i, p in enumerate(self.purposes[:2]) if i != asked_for]
                another = any(o and len(o & spoken) * 5 >= len(o) * 3 for o in others)
                if mine and (reasoned or another) and len(mine & spoken) * 5 < len(mine) * 3 and asked_for not in self.use_twice:
                    # the shopper gave the requirement of another purpose (the line names that one, not the one asked
                    # about): what this purpose holds back is still out, and it is asked about once more
                    self.use_again.add(asked_for)
                    self.use_same.add(asked_for)        # (heard for both purposes, the other's came first: same words)
                    self.needs_expected = max(self.needs_expected, 2)       # (two purposes, each with something held back)
                    met = None if met == asked_for else met
            choice = self.pair_open and any(r.get("chosen") for r in told)
            if self.pair_open and not choice and not self.pair_again and not self.pair_chosen:
                self.pair_again = True          # (asked which wish matters more, the shopper named a requirement that is
                                                # neither wish - what a purpose holds back: the question is put once more)
            if met is not None and not choice:
                self.purposes_met.add(met)      # (the choice between two wishes is no purpose's requirement)
            self.asked_log.append((question, f"the shopper said: {line[:160]}"))
            if choice:
                # which of two wishes matters more: that settles the pair, not what a purpose holds back.
                # A requirement may be held back still, so a few properties are asked about after it
                self.pair_chosen = True
            else:
                self.needs_found += 1
                self.lead_mark = self.property_asks
                self.first_found_at = self.first_found_at or self.questions_asked
            # (a property the told requirement is about leaves the line: asked again, it would draw the same answer)
            gone = [n for n in self.to_ask if n not in self.again and about_this(n)]
            lead_told = any(n in self.lead for n in gone) or any(n in self.lead for n in named if about_this(n))
            if gone:
                self.to_ask = [n for n in self.to_ask if n not in gone]
                self.lead = [n for n in self.lead if n not in gone]
                self.asked += [n for n in gone if n not in self.asked]
            if lead_told and self.lead:
                # (a purpose holds one detail back: its other planned measure goes to the back of the line)
                self.to_ask = [n for n in self.to_ask if n not in self.lead] + [n for n in self.lead if n in self.to_ask]
                self.lead = []
            # the line stays as it is; a property the question named beside the answered one goes back into it,
            # behind the next two: it is as likely to be held back as any other, no more
            rest = [n for n in self.to_ask if n not in self.again]
            if by_the_use:
                self.to_ask = self.again + rest
            elif apart:
                # (beside other properties: asked together again, they would draw the same answer)
                half = (len(self.again) + 1) // 2
                self.to_ask = rest[:1] + self.again[:half] + rest[1:2] + self.again[half:] + rest[2:]
            else:
                self.to_ask = rest[:2] + self.again + rest[2:]
            self.again = []
            return
        if question in self.by_use and line and not spoke_of_else:
            # asked what the purpose needs, the shopper named nothing: what is held back is asked for by property
            self.use_drew_nothing = True
            # (not put again in other words: what it holds back is asked for by property - a shopper who tells only what
            # a named property asks for answers that question with the limit again)
            if len(self.purposes) >= 2 and 0 not in self.purposes_met and not self.plain_asked:
                self.plain_pending = True       # (two purposes, the first still out: the plain use question, once)
            if self.told_to_ask and not self.plain_asked and not self.needs_found:
                self.plain_pending = True       # (a request that says to ask: the use itself is asked about once, in the plainest words)
            self.asked_log.append((question, "no requirement"))
            return
        if question == PLAIN_QUESTION and self.fixed_uses and not self.fixed_quiet and not spoke_of_else:
            # asked what the item is for, the shopper named nothing that a purpose decided elsewhere fixes: such a
            # purpose holds nothing back, and no property is asked about on its account
            # (one property is still asked about first: a shopper of another race told such a purpose's detail only
            # when its property was named)
            # (across races, the plain use question has told every detail such a purpose held back: when it draws only
            # the limit and the shelf, there is none)
            self.fixed_quiet = True
            self.quiet_after = self.questions_asked
            self.record_note("the use question drew nothing for the purposes decided elsewhere: no more questions for them")
        if spoke_of_else and question and (name or question) not in self.reasked:
            # the shopper spoke of a change instead (a new limit, another value for something already known, the
            # market): the question went unanswered and is put once more
            self.reasked.add(name or question)
            if name:
                for again in reversed(named):
                    self.asked = [p for p in self.asked if p != again]
                    self.to_ask.insert(0, again)
            elif question == PLAIN_QUESTION:
                self.plain_pending, self.plain_asked = True, False
            elif question in self.by_use:
                self.use_asked = {i for i in self.use_asked if self.said_back(i) not in question}
            return
        if line and nothing and len(named) == 2 and question not in self.reasked and not self.restates_budget(line):
            # the shopper answered, and with nothing at all: no requirement, and not the limit they hold to either
            # (which is what a shopper says who holds nothing back about the property asked). A question about two
            # properties was not taken as one question: each is asked about once more, alone
            self.reasked.add(question)
            self.asked_log.append((question, "not answered"))
            self.asked = [p for p in self.asked if p not in named]
            self.to_ask = named + [n for n in self.to_ask if n not in named]
            self.alone, self.use_unheard = len(named), True
            return
        self.asked_log.append((question, "no requirement"))
        if len(named) >= 3:
            self.asked_wide += named        # (named three or four at a time: any of them may have gone unheard)
        elif self.narrow and named:
            self.narrow += 1
        if line and self.spoke_of_use:
            self.use_unheard = True     # the shopper answered, and not about the use: properties are asked about alone

    # ---- reading listings
    def affordable(self, key: tuple[str, str]) -> bool:
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        if seen.get("in_stock") is False or key in self.gone:
            return False
        return self.budget is None or not isinstance(price, float) or price <= self.budget + 1e-9

    def just_over(self, key: tuple[str, str], reach: float = OVER_REACH) -> bool:
        """In stock and a little over the spending limit: not for ordering, but what a store has nearest when
        nothing within the limit meets every requirement."""
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        if self.budget is None or not isinstance(price, float) or seen.get("in_stock") is False or key in self.gone:
            return False
        return self.budget + 1e-9 < price <= self.budget * reach

    def only_over(self) -> list[tuple[str, str]]:
        """The listings that meet everything but the spending limit, and miss that by a little: the surest first,
        then the one that costs least."""
        if self.moved_on:
            # a shopper who has lowered the limit or changed a value was asking for something that could be bought:
            # a listing just over the limit now is one that fitted before, not a sign that nothing ever did
            return []
        any_shelved = any(self.in_category(p) for p in self.counter.pages.values())
        out = []
        for key in self.verdicts:
            page = self.counter.pages.get(key[0]) or {}
            if not self.just_over(key) or not self.listing_clears(key, "words") or (any_shelved and not self.in_category(page)):
                continue
            if self.brand and not set(letter_digit_tokens(self.brand)) <= set(letter_digit_tokens(page.get("brand"))):
                continue
            out.append(key)
        # (of the likely or surely fitting ones, the least over the limit first: that is the one a shopper would take)
        return sorted(out, key=lambda k: (self.trust(k) < LIKELY_FIT - 1e-9, (self.counter.stock.get(k) or {}).get("price") or 0.0, -self.trust(k)))

    def named_shelf(self) -> str:
        """The store's own label for the shelf the request names: of the labels read on listings, the longest
        whose words stand together in the request."""
        pages = self.counter.pages
        if self.shelf_cache[0] == len(pages):
            return self.shelf_cache[1]
        request = f" {singular_phrase_key(self.counter.request)} "
        said = f" {singular_phrase_key(self.shelf_said)} " if self.shelf_said else ""
        seen: dict[str, int] = {}
        told: dict[str, int] = {}
        for page in pages.values():
            path = [str(label) for label in page.get("category_path") or []]
            if path and singular_phrase_key(path[-1]) and f" {singular_phrase_key(path[-1])} " in request:
                seen[path[-1]] = seen.get(path[-1], 0) + 1
            if path and said and singular_phrase_key(path[-1]) and f" {singular_phrase_key(path[-1])} " in said:
                told[path[-1]] = told.get(path[-1], 0) + 1
        own = {singular_phrase_key(self.category), singular_phrase_key(self.item)} - {""}
        best = max(seen, key=lambda label: (len(singular_phrase_key(label).split()), singular_phrase_key(label) in own, seen[label]), default="")
        if told:
            # the shopper named the shelf in the store's own word: that is the shelf, whatever the request called the item
            best = max(told, key=lambda label: (len(singular_phrase_key(label).split()), told[label]))
        self.shelf_cache = (len(pages), best)
        return best

    def in_category(self, page: dict) -> bool:
        named = self.named_shelf()
        if named:
            # (the page's own shelf, or the named shelf's own label on its path: a parent label that only reduces to
            # the same words - 'Shower Curtains & Accessories' for 'Shower Curtain Accessories' - is another shelf's)
            path = [str(label) for label in page.get("category_path") or []]
            return bool(path) and (singular_phrase_key(path[-1]) == singular_phrase_key(named)
                                   or any(normalized_lower(label).strip() == normalized_lower(named).strip() for label in path))
        wanted = {singular_phrase_key(self.category), singular_phrase_key(self.item)} - {""}
        labels = {singular_phrase_key(label) for label in page.get("category_path") or []}
        return bool(wanted & labels)

    def variant_of(self, key: tuple[str, str]) -> dict:
        page = self.counter.pages.get(key[0]) or {}
        return next((v for v in page.get("variants") or [] if str(v.get("sku")) == key[1]), {})

    @staticmethod
    def word_stem_set(text: Any) -> set[str]:
        return set(ShopOrder._cached_word_stems(str(text)))

    @staticmethod
    @lru_cache(maxsize=65536)
    def _cached_word_stems(text: str) -> frozenset:
        words = {singular_form(w) for w in letter_digit_tokens(text.replace("_", " "))} - {"of"}
        return frozenset("color" if w == "colour" else w for w in words)

    @staticmethod
    def forms(requirement: dict, loose: bool = False) -> set[str]:
        stamp = (str(requirement.get("value")), tuple(str(w) for w in requirement.get("written") or []),
                 tuple(str(w) for w in requirement.get("alike") or []), bool(requirement.get("filed")), loose)
        known = _FORMS.get(stamp)
        if known is None:
            known = _FORMS[stamp] = frozenset(ShopOrder._forms(requirement, loose))
        return set(known)

    @staticmethod
    def _forms(requirement: dict, loose: bool = False) -> set[str]:
        """The required value in each way it may be written, as plain words; loose: the words sellers use alike."""
        spellings = list(requirement.get("written") or [])
        if requirement.get("filed"):        # (a value filed as a fact is filed in the store's own word)
            written = spellings + list(requirement.get("alike") or []) if loose else [requirement["value"]]
        elif re.search(r"\d", requirement["value"]):
            # (a figure alone is no measurement of an amount with a unit; two figures joined by x with no unit after
            # them are a pair that has none, and the reader's other spellings of it stand)
            pair = re.fullmatch(r"\s*\d+(?:\.\d+)?(?:\s*[x*\u00d7]\s*\d+(?:\.\d+)?)+\s*", normalized_lower(requirement["value"]))
            if re.search(r"\d\s*[a-z]", normalized_lower(requirement["value"])) and not pair:
                spellings = [form for form in spellings if not re.fullmatch(r"[\d.,\s]+", str(form).strip())
                             and ShopOrder.same_amount(requirement["value"], form)]
            elif pair:
                # (of a pair's other spellings, those that carry one of its own figures or none stand for it; one with
                # another figure measures it another way, and only roughly)
                own = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", requirement["value"])}
                spellings = [form for form in spellings
                             if not re.search(r"\d", str(form)) or {float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(form))} & own]
            written = list(requirement.get("alike") or []) if loose else [requirement["value"]] + spellings
        else:                               # a word: another spelling of it is the word, anything else is a seller's other word
            close = [form for form in spellings if SequenceMatcher(None, normalized_lower(form), normalized_lower(requirement["value"])).ratio() >= 0.75]
            written = [form for form in spellings if form not in close] + list(requirement.get("alike") or []) if loose else [requirement["value"]] + close
        return {" ".join(singular_form(w) for w in letter_digit_tokens(join_number_unit(form))) for form in written} - {""}

    @staticmethod
    def same_amount(value: str, form: Any) -> bool:
        return ShopOrder._same_amount(str(value), str(form))

    @staticmethod
    @lru_cache(maxsize=65536)
    def _same_amount(value: str, form: str) -> bool:
        """Another spelling of an amount is that amount: every figure it gives with a unit is a figure of the value in
        the same unit, or in the same measure with another metric prefix ("4G" is not 4 GB, "14 days" is not 2 weeks)."""
        def figures(text: Any) -> list[tuple[float, str]]:
            return [(float(m.group(1)), ShopOrder.unit_word(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]+)", normalized_lower(text))]
        wanted, found = figures(value), figures(form)
        if not wanted:
            return True
        if not found:
            return False        # (a spelling that gives no figure with a unit - the number in words - is no spelling of the amount)
        for number, name in found:
            same = any(number == n and ShopOrder.same_unit(name, u) for n, u in wanted)
            turned = any(ShopOrder.scaled(u, name) and abs(n * (ShopOrder.scaled(u, name) or 0.0) - number) < 1e-6 for n, u in wanted)
            if not same and not turned:
                return False
        return True

    @staticmethod
    def converted(requirement: dict) -> set[str]:
        stamp = ("converted", str(requirement.get("value")), tuple(str(w) for w in requirement.get("written") or []),
                 tuple(str(w) for w in requirement.get("alike") or []))
        known = _FORMS.get(stamp)
        if known is None:
            known = _FORMS[stamp] = frozenset(ShopOrder._converted(requirement))
        return set(known)

    @staticmethod
    def _converted(requirement: dict) -> set[str]:
        """The required amount written in another unit: in each other metric step of a short unit (centimetres as
        millimetres), and in the other units the reader gave for it (months as years, centimetres as inches) - each such
        spelling that names another unit with another figure. A listing that states the amount so states it less
        plainly than in the shopper's own unit."""
        wanted = [(float(m.group(1)), ShopOrder.unit_word(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", normalized_lower(requirement.get("value")))]
        if len(wanted) != 1:
            return set()
        number, unit = wanted[0]
        out: set[str] = set()
        if len(unit) <= 3:
            base = unit[1:] if len(unit) > 1 and unit[0] in "mcdk" else unit
            for other in {base, "m" + base, "c" + base, "d" + base, "k" + base} - {unit}:
                ratio = ShopOrder.scaled(unit, other, any_step=True)
                figure = round(number * ratio, 6) if ratio else 0.0
                if 0 < figure < 1e5 and abs(figure * 100 - round(figure * 100)) < 1e-6:
                    out.add(f"{figure:g} {other}")
        for form in list(requirement.get("written") or []) + list(requirement.get("alike") or []):
            text = normalized_lower(form)
            found = [(float(m.group(1)), ShopOrder.unit_word(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]+)", text)]
            if len(found) != 1 or re.search(r"\d", re.sub(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]+)", " ", text)):
                continue
            figure, name = found[0]
            ratio = ShopOrder.scaled(unit, name, any_step=True)
            if ratio and abs(number * ratio - figure) > 1e-6 * max(1.0, figure):
                continue        # (the same measure in another step, with a rounded figure: another amount)
            if figure != number and len(name) >= 2 and not ShopOrder.same_unit(name, unit):
                out.add(f"{figure:g} {name}")
        return out

    def forms_under(self, requirement: dict, label: str) -> set[str]:
        """Under a fact's label the value may drop the words the label already says (a make is filed by its
        name alone, not by the name and the kind of thing)."""
        named = self.word_stem_set(label) | {"at", "the", "of", "a", "an", "in", "on", "for", "with", "its"}
        out = set(self.forms(requirement))
        if out & self.filed_under(label):
            return out          # (the shelf files the whole value under this label: a shorter entry - "all" beside
                                # "all season" - is another value of the property, not this one said briefly)
        for form in list(out):
            words = form.split()
            kept = [w for w in words if not any(same_word_stem(w, n) for n in named)]
            if kept and len(kept) < len(words):
                out.add(" ".join(kept))
        return out

    def filed_under(self, label: str) -> set[str]:
        """The entries the listings read so far carry under one fact label."""
        pages = self.counter.pages
        cached = self.filed_cache.get(label)
        if cached and cached[0] == len(pages):
            return cached[1]
        found = {entry for page in pages.values() for entry in self.entries((page.get("facts") or {}).get(label))}
        self.filed_cache[label] = (len(pages), found)
        return found

    def states(self, text: Any, requirement: dict, label: str = "") -> bool:
        """One entry of a fact (facts often list several) is exactly the required value."""
        entries = {" ".join(singular_form(w) for w in letter_digit_tokens(reading(part))) for part in re.split(r"[,;/|]+", normalized_lower(text)) for reading in (normalized_lower, join_number_unit)}
        if not label:
            return bool(entries & self.forms(requirement))
        named = self.word_stem_set(label)
        for entry in list(entries):         # an entry may repeat the label's own words
            kept = [w for w in entry.split() if not any(same_word_stem(w, n) for n in named)]
            if kept:
                entries.add(" ".join(kept))
        return bool(entries & self.forms_under(requirement, label))

    def property_place(self, requirement: dict) -> str:
        """Where a store states a property: an amount; a colour, which the variant's options name; a material,
        which the listing's name carries; anything else, which is kept as a fact of its own."""
        if re.search(r"\d", requirement["value"]):
            return "amount"
        name = normalized_lower(requirement["property"])
        if re.search(r"colou?r", name):
            return "colour"
        if "material" in name:
            # (the material of one part - a seat's covering, a strap - is a fact of its own where the shelf keeps one)
            return "fact" if self.part_labels(requirement) else "material"
        return "fact"

    def part_labels(self, requirement: dict) -> set[str]:
        """The fact labels under which this shelf files the material of one named part: labels made of the
        property's own words that say more than 'material'. A store that keeps such a fact files the value there,
        and a listing's name then says what the item is made of, not what that part is."""
        stamp = (requirement["property"], len(self.counter.pages))
        cached = self.part_cache.get(requirement["property"])
        if cached and cached[0] == stamp:
            return cached[1]
        wanted = self.word_stem_set(requirement["property"])
        part = wanted - self.word_stem_set("material type")
        found: set[str] = set()
        if part:
            for page in self.counter.pages.values():
                for label in page.get("facts") or {}:
                    have = self.word_stem_set(label)
                    if have and have <= wanted and have & part:
                        found.add(str(label))
        self.part_cache[requirement["property"]] = (stamp, found)
        return found

    @staticmethod
    def entries(text: Any) -> list[str]:
        """The separate entries of a fact (a fact may list several), as plain words. An entry that names a value under
        a heading ("group: member") is the member: the heading only says what kind of value follows."""
        out = []
        for part in re.split(r"[,;/|]+", normalized_lower(text)):
            head, colon, tail = part.rpartition(":")
            if colon and re.search(r"[a-z]", head) and re.search(r"[a-z]", tail):
                part = tail
            words = " ".join(singular_form(w) for w in letter_digit_tokens(join_number_unit(part)))
            if words:
                out.append(words)
        return out

    def fact_entry_matches(self, entry: str, requirement: dict, label: str, loose: bool = False, *, exact: bool = False) -> bool:
        """Whether one filed entry states the required value.
        exact: the entry is that value and nothing else. Words the label itself says are set aside first.
        loose: every word of a form sellers use for the same value.
        Otherwise: every word of one form under this label. Words the label itself says need not be repeated, and may be repeated."""
        if exact:
            named = self.word_stem_set(label)
            kept = " ".join(w for w in entry.split() if not any(same_word_stem(w, n) for n in named)) or entry
            forms = self.forms_under(requirement, label)
            return entry in forms or kept in forms
        forms = self.forms(requirement, loose=True) if loose else self.forms_under(requirement, label)
        have = entry.split()
        return any(all(any(same_word_stem(n, h) for h in have) for n in form.split()) for form in forms)



    def own_labels(self, requirement: dict) -> tuple[set[str], set[str], set[str]]:
        """The fact labels that are this property's own, and the values the shelf files under them:
        the labels named like the property (one name holding all words of the other); the labels under which
        the shelf's listings state exactly such a value, as their only entry, more than once; and the single
        words that stand as entries under those labels."""
        pages = self.counter.pages
        stamp = (requirement["property"], requirement["value"], len(requirement.get("written") or []), len(pages), self.named_shelf())
        cached = self.own_cache.get(("own", requirement["property"], requirement["value"]))  # (two values of one property each keep theirs)
        if cached and cached[0] == stamp:
            return cached[1]
        wanted = self.word_stem_set(requirement["property"])
        kind = self.property_place(requirement)
        shelf = [page for page in pages.values() if self.in_category(page)] or list(pages.values())
        written = " ".join(letter_digit_tokens(requirement["property"]))
        tiers: list[set[str]] = [set(), set(), set(), set()]     # named exactly so; the same words; one name within the other
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        # (a label may write the unit by its first letter: power_w for watts)
        unit_marks = units | {singular_form(u)[0] for u in units if 4 <= len(singular_form(u)) <= 5}
        stated: dict[str, int] = {}
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                have = self.word_stem_set(label)
                if wanted and have:
                    if " ".join(letter_digit_tokens(str(label).replace("_", " "))) == written:
                        tiers[0].add(str(label))
                    elif have == wanted:
                        tiers[1].add(str(label))
                    elif have <= wanted or wanted <= have:
                        # a name that adds only the required amount's unit (power_w for 1200 W) is the property's own;
                        # one that adds another word (suction_power) may name another measure
                        tiers[2 if (have ^ wanted) <= unit_marks else 3].add(str(label))
                    elif kind == "amount" and have & unit_marks and (have - unit_marks) and (have - unit_marks) < wanted:
                        # (a shorter name of the measure with its unit - power_w for a power rating in watts - is its own too)
                        tiers[2].add(str(label))
                entries = self.entries(value)
                own_name = any(str(label) in tier for tier in tiers)
                # (under a label not named like the property, a material counts only as the whole entry: 'silicone brush'
                # under a brush-type label names a kind of brush, not the material)
                if len(entries) == 1 and (self.fact_entry_matches(entries[0], requirement, str(label), exact=True) if own_name or kind != "material"
                                          else entries[0] in self.forms_under(requirement, str(label))):
                    stated[str(label)] = stated.get(str(label), 0) + 1
        related = next((tier for tier in tiers if tier), set())
        if kind == "amount" and not related:
            # (no label of the shelf is named like the property - wattage - but labels that write the required amount's
            # unit beside the measure's name - power_w, capacity_l - file this amount: they are its own)
            related = {str(label) for page in shelf for label in (page.get("facts") or {})
                       if self.word_stem_set(label) & unit_marks and len(self.word_stem_set(label) - unit_marks) >= 1}
        if kind != "amount":
            # (a store may keep one property under two labels - a general one and its own department's - and file
            # the value under either: each label named like the property is the property's own)
            related = set().union(*tiers)
            if "material" in normalized_lower(requirement["property"]) and self.part_labels(requirement):
                related = self.part_labels(requirement)     # (the part's own fact, not the item's material)
        most = max(stated.values(), default=0)
        inferred: set[str] = set()
        if kind != "colour" and not any(stated.get(label) for label in related):
            # no fact named like the property carries such a value anywhere on the shelf: the label that does
            inferred = {label for label, count in stated.items() if count >= 2 and 3 * count >= most}
        if kind == "amount" and inferred:
            # (a label under which most listings state an amount in another unit files another measure: a few sellers
            # writing this amount under it does not make it the property's own. Bare figures leave the unit open.)
            def foreign(label: str) -> bool:
                values = [(page.get("facts") or {}).get(label) for page in shelf if label in (page.get("facts") or {})]
                other = sum(1 for value in values if not any(self.amounts(value, requirement)) and re.search(r"\d\s*[a-z]", normalized_lower(value)))
                return other * 2 > len(values)
            inferred = {label for label in inferred if not foreign(label)}
        words: set[str] = set()
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    words |= {entry for entry in self.entries(value) if " " not in entry}
            if kind == "colour":        # a word that stands alone as a variant's name is a value variants are told apart by
                for variant in page.get("variants") or []:
                    words |= {entry for entry in self.entries(variant.get("options")) if re.fullmatch(r"[a-z]{3,}", entry)}
        answer = (related, inferred, words)
        self.own_cache[("own", requirement["property"], requirement["value"])] = (stamp, answer)
        return answer

    def named_facts(self, page: dict, requirement: dict) -> list[tuple[str, str]]:
        related, inferred, _ = self.own_labels(requirement)
        found = [(str(label), str(value)) for label, value in (page.get("facts") or {}).items() if str(label) in related or str(label) in inferred]
        if not found and self.property_place(requirement) == "amount":
            # (a page that carries none of the labels the shelf files this property under may file the amount under a
            # label that writes its unit beside the measure's name - power_w for a wattage: that label is its own here)
            units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
            marks = units | {singular_form(u)[0] for u in units if 4 <= len(singular_form(u)) <= 5}
            found = [(str(label), str(value)) for label, value in (page.get("facts") or {}).items()
                     if self.word_stem_set(label) & marks and self.word_stem_set(label) - marks]
        return found

    def says(self, text: Any, requirement: dict, whole: bool = False, loose: bool = False) -> bool:
        """The text carries every word of the required value, in one of the ways it may be written.
        whole: only words standing on their own count (a word joined to another by a hyphen is a different word)."""
        if whole:
            have = {singular_form(w) for reading in (normalized_lower(text), join_number_unit(text)) for w in re.split(r"[\s/,;()\[\]|+]+", reading) if w}
            return any(set(form.split()) <= have for form in self.forms(requirement, loose))
        have = {singular_form(w) for reading in (normalized_lower(text), join_number_unit(text)) for w in letter_digit_tokens(reading)}
        # (a figure alone counts only where it stands alone: not the tail of a code joined to it by a hyphen, nor one end
        # of a range - a-4, 2-3 years; an option or a fact that is the figure still says it)
        return any(self.figure_alone(str(text), form) if form.replace(".", "", 1).isdigit()
                   else all(any(same_word_stem(n, h) for h in have) for n in form.split()) for form in self.forms(requirement, loose))

    @staticmethod
    @lru_cache(maxsize=131072)
    def figure_alone(text: str, figure: str) -> bool:
        """The figure is one of the text's words, and not one joined by a hyphen to the word or figure before it (the
        tail of a code, the end of a range) nor to a figure after it (the start of a range)."""
        low = normalized_lower(text)
        return any(singular_form(m.group()) == figure and not re.search(r"[a-z0-9][-\u2013]$", low[: m.start()]) and not re.match(r"[-\u2013]\d", low[m.end():])
                   for m in re.finditer(r"[a-z0-9]+(?:\.[0-9]+)?", low))

    def amounts(self, text: Any, requirement: dict, convert: bool = False) -> tuple[set[tuple[float, str]], set[tuple[float, str]]]:
        # (the same text is read for the same amount many times over: each reading is worked out once while the
        # requirements stand as they are)
        stamp = (str(text), str(requirement.get("value")), tuple(str(w) for w in requirement.get("written") or []),
                 tuple(str(w) for w in requirement.get("alike") or []), bool(requirement.get("filed")), convert,
                 tuple((str(r.get("value")), tuple(str(w) for w in r.get("written") or [])) for r in self.requirements))
        known = self.amounts_cache.get(stamp)
        if known is None:
            known = self._amounts(text, requirement, convert)
            if len(self.amounts_cache) < 200000:
                self.amounts_cache[stamp] = known
        return set(known[0]), set(known[1])

    def _amounts(self, text: Any, requirement: dict, convert: bool = False) -> tuple[set[tuple[float, str]], set[tuple[float, str]]]:
        """The amounts a text states in the units of the required value: (those that are the value, the others).
        convert: the value written in any metric step of its unit, or in another unit the reader gave for it, is the value."""
        unit = self.unit_word
        wanted: set[tuple[float, str]] = set()
        for form in self.forms(requirement) | (self.converted(requirement) if convert else set()):
            wanted |= {(float(m.group(1)), unit(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        # (a figure right after a letter of another script or a sign such as # - a diameter mark, a number - is no amount,
        # and neither is one glued to the word before it - heat1500w, a seller's code: a store reads no measurement there)
        # (nor is one whose unit runs straight on into more figures that are no amount of their own - a seller's code
        # such as 108v1-s - while two amounts written together, 12v2a, are both read)
        found = {(float(m.group(1)), unit(m.group(2))) for m in re.finditer(r"(?<![^\x00-\x7f\W])(?<![#\d.:a-z])(\d+(?:\.\d+)?)\s*([a-z]+)" + CODE_TAIL, normalized_lower(text))}
        found |= set(self.listed_amounts(text))
        # one measure written with a larger or smaller metric unit is the same amount (1 l and 1000 ml, 2.5 cm and 25 mm)
        for number, name in list(wanted):
            for seen in {u for _, u in found}:
                scale = self.scaled(name, seen, any_step=convert)
                if scale:
                    wanted.add((round(number * scale, 6), seen))
        units = {name for _, name in wanted}
        # (a longer word after a figure matches a one-letter unit only when the reader wrote that word for the value -
        # 5 litres for 5 l - not any word that starts with the letter: 5 layers are no amount of litres)
        numbers = {number for number, _ in wanted}
        spelled = {self.unit_word(m.group(2)) for r in [requirement] + [r for r in self.requirements if r is not requirement]
                   for form in list(r.get("written") or []) + [r.get("value") or ""]
                   for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]+)", normalized_lower(form)) if r is requirement or float(m.group(1)) in numbers}

        # (a short unit read in a longer word - its letters stand in that word in order - is that unit only if the word
        # also agrees with the unit written out in full, where the value's own spellings write it so: 12 months for
        # 12 mo, not the 1 of a "1 memory")
        full = {name: {u for u in units if len(u) > len(name) + 1 and self.same_unit(u, name)} for name in units if len(name) <= 3}

        def unit_is(word: str, name: str) -> bool:
            if not self.same_unit(word, name) or not (len(name) > 1 or len(word) <= 2 or word in spelled):
                return False
            longer = full.get(name)
            return not longer or len(word) <= len(name) + 1 or word in spelled or any(self.same_unit(word, u) for u in longer)
        found = {pair for pair in found if any(unit_is(pair[1], name) for name in units)}
        same = {pair for pair in found if any(pair[0] == number and unit_is(pair[1], name) for number, name in wanted)}
        return same, found - same

    @staticmethod
    def other_words(tokens: set[str]) -> set[str]:
        """Plain words among option tokens: not a code with digits or signs, and not a size letter code (xl, xxl)."""
        return {w for w in tokens if re.fullmatch(r"[a-z]{3,}", w) and not re.fullmatch(r"x*[sml]|x+l", w)}

    def unit_in_label(self, label: str, requirement: dict) -> bool:
        """The fact's label names the unit of the required amount (capacity_l, display_size_inch)."""
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        words = self.word_stem_set(label)
        # (or names it by its first letter, as labels shorten a unit written out in full - power_w for watts; only a
        # short unit word, which no other unit shares a first letter with in the same label)
        return bool(units & words) or any(len(w) == 1 and 4 <= len(singular_form(u)) <= 5 and singular_form(u)[0] == w
                                          for u in units for w in words)

    def named_otherwise(self, options: Any, requirement: dict) -> bool:
        """A part of the variant's options gives the property by its name with a figure that is not the required
        amount (however the seller spells the name): that figure is the property's, and the required one, where the
        options also carry it, measures something else."""
        named = [w for w in self.word_stem_set(requirement["property"]) if len(w) >= 5]
        for part in re.split(r"[/,;|]+", normalized_lower(options)):
            if not re.search(r"\d", part) or self.has_amount(part, requirement):
                continue
            if any(SequenceMatcher(None, word, name).ratio() >= 0.8 for word in re.findall(r"[a-z]{4,}", part) for name in named):
                return True
        return False

    def listed_amounts(self, text: Any) -> list[tuple[float, str]]:
        """A list of three or more figures sharing one unit (sizes on offer written one after another with slashes):
        each figure with that unit. (Two figures joined so are a fraction, not a list.)"""
        out = []
        for m in re.finditer(r"(?<![\d./])((?:\d+(?:\.\d+)?\s*/\s*){2,}\d+(?:\.\d+)?)\s*([a-z]+)", normalized_lower(text)):
            out += [(float(n), self.unit_word(m.group(2))) for n in re.findall(r"\d+(?:\.\d+)?", m.group(1))]
        return out

    def has_amount(self, text: Any, requirement: dict, convert: bool = False) -> bool:
        """The text states the required amount: the same number with the same unit, however it is spaced. An
        amount required as a bare number takes its unit from the property's name."""
        same, others = self.amounts(text, requirement, convert)
        if same or (not others and self.says(text, requirement)):
            return True         # (where the text gives that unit with another number, the words of the value standing
                                # apart in it are not the amount: "50*23mm (1 piece)" is one piece)
        bare = re.fullmatch(r"\d+(?:\.\d+)?", requirement["value"].strip())
        if not bare:
            return False
        named = [w for w in letter_digit_tokens(requirement["property"]) if len(w) >= 3]
        return any(float(m.group(1)) == float(bare.group(0)) and any(self.same_unit(m.group(2), w) for w in named)
                   for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]+)" + CODE_TAIL, normalized_lower(text)))

    @staticmethod
    @lru_cache(maxsize=131072)
    def same_unit(one: str, other: str) -> bool:
        """One unit written short and long: the short one's letters stand in the long one in order, from its first
        letter. Short units are told apart by every letter."""
        unit = ShopOrder.unit_word
        a, b = sorted((unit(normalized_lower(one)), unit(normalized_lower(other))), key=len)
        if not a or a[0] != b[0] or len(b) <= 3 or (len(a) == 1 and len(b) > 6):
            return bool(a) and a == b       # (a single letter stands for a short word only)
        rest = iter(b)
        return all(letter in rest for letter in a)

    @staticmethod
    @lru_cache(maxsize=131072)
    def scaled(unit: str, other: str, any_step: bool = False) -> float | None:
        """How many of `other` make one `unit`, when the two are one measure a thousand times apart (l and ml, kg and
        g, m and mm): a store that asks for one counts the other. None when they are not: a smaller step (cm and
        mm) is another way of writing the size, and a listing that writes it so is not filed under the amount asked
        for. any_step: every metric step counts (for comparing sizes, not for meeting an amount)."""
        prefixes = {"milli": 1e-3, "centi": 1e-2, "deci": 1e-1, "kilo": 1e3, "m": 1e-3, "c": 1e-2, "d": 1e-1, "k": 1e3}

        def readings(name: str) -> list[tuple[float, str]]:
            return [(1.0, name)] + [(size, name[len(p):]) for p, size in prefixes.items() if name.startswith(p) and len(name) > len(p)]
        for size, base in readings(unit):
            for other_size, other_base in readings(other):
                if size != other_size and (base == other_base or (len(base) > 1 and len(other_base) > 1 and ShopOrder.same_unit(base, other_base))):
                    ratio = size / other_size
                    if any_step or abs(ratio - 1000.0) < 1e-6 or abs(ratio - 0.001) < 1e-9:
                        return ratio
        return None

    def other_amount(self, text: Any, requirement: dict, convert: bool = False) -> bool:
        """The text states an amount in a unit of the required value, but not that amount."""
        same, others = self.amounts(text, requirement, convert)
        return bool(others) and not same

    def amount_level(self, page: dict, options: str, title: Any, requirement: dict, singles: list, listed: bool, convert: bool = False) -> int:
        """How plainly the page states a required amount (see plain); convert: written in another unit counts."""
        if self.other_amount(options, requirement, convert) or self.named_otherwise(options, requirement):
            return -1           # the variant's own options state another amount of that kind
        number = re.search(r"\d+(?:\.\d+)?", requirement["value"])
        if self.has_amount(options, requirement, convert):
            # (options that write several amounts of that kind - a set of two sizes - sell them together: the required
            # one is only a part of what is sold)
            if self.amounts(options, requirement, convert)[1]:
                return 1
            # (the amount only as one side of a size - 1.5 x 2.5mm - is stated, but not on its own: a page that states
            # it alone is plainer)
            sizes = re.sub(r"\d+(?:\.\d+)?\s*[a-z]{0,3}\s*(?:[x*\u00d7]\s*\d+(?:\.\d+)?\s*[a-z]{0,3}\s*)+", " ", normalized_lower(options))
            return 3 if self.has_amount(sizes, requirement, convert) or sizes == normalized_lower(options) else 2
        # (the property's own fact that gives the amount only as one end of a range or one of a list - 10 l - 20 l,
        # 10l/20l - states it weakly: the listing may be sold in the other amount)
        if any(self.has_amount(entry, requirement, convert) and self.amounts(entry, requirement, convert)[1] for _, entry in singles) and not any(
                self.has_amount(entry, requirement, convert) and not self.amounts(entry, requirement, convert)[1] for _, entry in singles):
            return 1
        # (the property's own fact - named like it, or the one this shelf files such amounts under - gives the figure bare)
        bare = [label for label, entry in singles if number and re.fullmatch(r"\d+(?:\.\d+)?", entry) and float(entry) == float(number.group(0))]
        # (a bare count under a label that names pieces - a set's number of pieces - counts what the seller lists in the
        # set, colours and tools and a case, as for the priority's count: it is no plain statement of the pieces sold)
        counted = any(self.same_unit(self.unit_word(m.group(2)), "pc") for form in self.forms(requirement)
                      for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form))
        # (a bare number under a label naming another step of the same unit - capacity_l: 0.6 for 600 ml - is that
        # amount written in the label's unit)
        # (any fact named like the property with one unit word more - capacity_l beside a bare capacity - counts so)
        own_units = [m.group(2) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", normalized_lower(requirement["value"]))]
        wanted = self.word_stem_set(requirement["property"])

        def step(word: str, unit: str) -> float | None:
            if self.same_unit(word, unit) or (len(word) == 1 and 4 <= len(singular_form(unit)) <= 5 and singular_form(unit)[0] == word):
                return 1.0
            return self.scaled(word, unit)
        stepped = []
        for label, fact in (page.get("facts") or {}).items():
            extra = self.word_stem_set(label) - wanted
            bare_fact = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(fact))
            if not (number and bare_fact and wanted and wanted <= self.word_stem_set(label) and len(extra) == 1):
                continue
            word = next(iter(extra))
            if any(abs(float(bare_fact.group(1)) * (step(word, unit) or 0.0) - float(number.group(0))) <= 1e-6 * max(1.0, float(number.group(0)))
                   for unit in own_units):
                stepped.append(str(label))
        if (any(self.unit_in_label(label, requirement) and not counted for label in bare) or (stepped and not counted)
                or any(self.has_amount(entry, requirement, convert) for _, entry in singles)):
            if self.other_amount(title, requirement, convert) and not self.has_amount(title, requirement, convert):
                return 1        # (the name states only another amount of that kind - 14pcs for a set of 3: unclear)
            return 3
        if bare:
            # the property's own fact gives the figure without its unit: the same number, but no measurement on its own
            return 2 if self.has_amount(title, requirement, convert) else 1
        if any(re.search(r"\d", entry) for _, entry in singles):
            return -1           # the property's own fact gives another figure
        if self.has_amount(title, requirement, convert):
            if self.sold_together(title, options, requirement, convert):
                return 2        # (the sizes the name lists are sold together, one piece of each: the required one is among them)
            # (a title with several such amounts lists its variants; so does one that lists several sizes of another
            # measure, one of which the variant's options name - its one figure of this kind may be another variant's)
            return 1 if self.amounts(title, requirement, convert)[1] or self.variant_list(title, options, requirement) else 2
        if self.other_amount(title, requirement, convert) and not self.amounts(options, requirement, convert)[0]:
            return -1           # the listing's name gives another amount of that kind ("trial 2ml"): a fact that
                                # says the required one is the seller's form filled in, not what is sold
        for value in (page.get("facts") or {}).values():
            if self.amounts(value, requirement, convert)[1]:
                continue        # (a fact with several amounts of that kind lists what is on offer, not this variant)
            for entry in self.entries(value):
                # one entry of a fact that is this amount and nothing else (not one side of a 30x20x5cm)
                if self.amounts(entry, requirement, convert)[0] and len(re.findall(r"\d+(?:\.\d+)?", entry)) == 1:
                    return 2
        return 1 if listed or any(self.has_amount(value, requirement, convert) for value in (page.get("facts") or {}).values()) else 0

    def sold_together(self, title: Any, options: Any, requirement: dict, convert: bool = False) -> bool:
        """The name lists three or more sizes of the required kind one after another, the required one among them, and
        the listing counts one piece for each size (in its options, else in its name): one of each is sold together."""
        sizes = self.listed_amounts(title)
        figures = {n for n, _ in sizes}
        if len(figures) < 3 or not any(self.amounts(f"{n:g}{u}", requirement, convert)[0] for n, u in sizes):
            return False

        def pieces(text: Any) -> set[int]:
            return {int(m.group(1)) for m in re.finditer(r"(?<![\w.])(\d+)\s?([a-z]+)(?![a-z0-9])", normalized_lower(text))
                    if self.same_unit(self.unit_word(m.group(2)), "pc")}
        return (pieces(options) or pieces(title)) == {len(figures)}

    def variant_list(self, title: Any, options: Any, requirement: dict) -> bool:
        """The title lists two or more figures of one other unit (the sizes of the variants), and the variant's options
        name one of them: the title describes several variants, not this one alone."""
        own = {self.unit_word(m.group(2)) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        listed: dict[str, set[float]] = {}
        for m in re.finditer(r"(?<![^\x00-\x7f\W])(?<![#\d.:])(\d+(?:\.\d+)?)\s*([a-z]+)" + CODE_TAIL, normalized_lower(title)):
            unit = self.unit_word(m.group(2))
            if len(unit) >= 2 and not any(self.same_unit(unit, u) for u in own):
                listed.setdefault(unit, set()).add(float(m.group(1)))
        for number, unit in self.listed_amounts(title):       # (and sizes written one after another with one unit: 16/18/20 inch)
            if len(unit) >= 2 and not any(self.same_unit(unit, u) for u in own):
                listed.setdefault(unit, set()).add(float(number))
        said = {(float(m.group(1)), self.unit_word(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", normalized_lower(options))}
        return any(len(numbers) >= 2 and any(n in numbers and self.same_unit(u, unit) for n, u in said)
                   for unit, numbers in listed.items())

    def options_outvoted(self, page: dict, options: str, title: Any, requirement: dict) -> bool:
        """The options state the amount only in another unit, while the listing's name states one other amount of
        that unit and a fact of the page the same: the name and the fact say what is sold."""
        units = {unit for _, unit in self.amounts(options, requirement, convert=True)[0]}
        in_title = {pair for pair in self.amounts(title, requirement, convert=True)[1] if pair[1] in units}
        if len(in_title) != 1:
            return False
        pair = next(iter(in_title))
        return any(pair in self.amounts(value, requirement, convert=True)[1] for value in (page.get("facts") or {}).values())

    @staticmethod
    def names_other_thing(text: str, forms: set[str], title: Any) -> bool:
        """The value is followed by a word that the listing's name does not carry: it may describe that other thing
        (an add-on, a part) rather than the item."""
        named = {singular_form(w) for w in letter_digit_tokens(title)}
        for form in forms:
            value = r"\s+".join(re.escape(w) for w in form.split())
            for m in re.finditer(rf"(?<![a-z]){value}s?\s+([a-z]{{3,}})", text):
                if singular_form(m.group(1)) not in named and m.group(1) not in form.split():
                    return True
        return False

    @staticmethod
    def joined_value(text: str, forms: set[str]) -> bool:
        """The value stands joined to another word by a sign or a joining word (one colour joined to another by a plus
        sign, an ampersand or 'and'): the text names two values as one."""
        join = r"\s*(?:\+|&|\band\b|\bwith\b)\s*"
        for form in forms:
            value = r"\s+".join(re.escape(w) for w in form.split())
            if value and re.search(rf"[a-z]{{2,}}{join}{value}(?![a-z])|(?<![a-z]){value}s?{join}[a-z]{{2,}}", text):
                return True
        return False

    def colour_plainness(self, requirement: dict, options: str, title: Any, value_words: set, stated: bool, listed: bool, words: set, named_as: Any) -> int:
        """How plainly the variant options state a required colour: the same scale as plain."""
        wanted = self.forms(requirement)
        parts = [{singular_form(w) for reading in (normalized_lower(part), join_number_unit(part)) for w in re.split(r"[\s()\[\]+]+", reading) if w}
                 for part in re.split(r"[/,;|]+", options)]
        # (a colour joined by a hyphen to a code or another word - green-a09, white-upgrade - is that colour: a store
        # reads a hyphen as the gap between two words)
        parts = [part | {singular_form(piece) for w in part if "-" in w for piece in w.split("-") if piece} for part in parts]
        named = [part for part in parts if any(set(form.split()) <= part for form in wanted)]
        if named:
            # the part of the options that names the colour: another colour word beside it makes it another colour,
            # and so does a word joined to it by a sign or 'and'. A describing word beside it that this shelf does
            # not file as a colour (a finish, a model's word) leaves it that colour; a colour followed by a word the
            # listing's name does not carry may be the colour of that other thing - the reader decides
            texts = [normalized_lower(part) for part in re.split(r"[/,;|]+", options)]
            alone = any(not ((part & words) - value_words) and not self.joined_value(text, wanted)
                        and (named_as == "only" or not self.names_other_thing(text, wanted, title))
                        for part, text in zip(parts, texts) if part in named)
            return 3 if alone and named_as not in ("among", "other") else 1
        # (a sign that is not a letter, a digit, a point or a hyphen parts words too - cap*red: the colour is
        # named, but joined to something else by the seller, so less plainly than a word standing on its own)
        joined = [{singular_form(w) for reading in (normalized_lower(part), join_number_unit(part)) for w in re.split(r"[^a-z0-9.\-]+", reading) if w}
                  for part in re.split(r"[/,;|]+", options)]
        if any(set(form.split()) <= part for form in wanted for part in joined):
            return 1
        if any(form.replace(" ", "") in normalized_lower(options).replace(" ", "") for form in wanted):
            # the word joined to other characters is a code or another option's name, not this colour - unless the
            # reader finds the options name this colour alone ('c-beige（retractable）'): then unclear, not another
            return 1 if named_as == "only" else -1
        if stated:
            return 1 if options.strip() else 3      # (options that name the variant otherwise leave it unclear)
        before = self.took_back.get(normalized_lower(requirement["value"]))
        if before and any(set(form.split()) <= part for form in self.forms(dict(requirement, value=before, written=[], alike=[])) for part in parts):
            return -1           # the variant is named by the colour the shopper took back for this one
        alike = any(set(form.split()) <= part for form in self.forms(requirement, loose=True) for part in parts)
        if alike or named_as in ("only", "among"):
            return 1            # another word for the colour: the reader's or the seller's
        if options.strip():
            return -1           # the options name the variant: when they say something else, the variant is something else
        return 1 if listed else 0

    def material_plainness(self, key: tuple[str, str], page: dict, requirement: dict, options: str, title: Any, value_words: set, stated: bool, listed: bool, words: set, named_as: Any, singles: list, elsewhere: bool) -> int:
        """How plainly the page states a required material: the same scale as plain."""
        # (the property's own fact has to BE the material: a near word - silicon for silicone - or a blend of two
        # - wood+silicone - is another entry of the store's list, and such listings failed where the exact one passed)
        raws = [value for label, value in self.named_facts(page, requirement) if len(self.entries(value)) == 1]
        exactly = any(self.fact_entry_matches(entry, requirement, label, exact=True) and not re.search(r"[+&]", str(raw))
                      for (label, entry), raw in zip(singles, raws))
        if stated and exactly:
            return 3
        if stated:
            # (an entry that names one kind of the material before the material's own word is that material, where
            # the reader finds the words name it alone; a word after the value leaves the entry another thing)
            return 2 if named_as == "only" and any(
                not re.search(r"[+&]", str(raw)) and len(entry.split()) > len(form.split()) and entry.split()[-len(form.split()):] == form.split()
                for (label, entry), raw in zip(singles, raws) for form in self.forms_under(requirement, label)) else 1
        loose_in_options = any(re.search(r"(?<![\w-])" + re.escape(form) + r"(?![\w-])", normalized_lower(options)) for form in self.forms(requirement) if form)
        if loose_in_options and self.says(options, requirement) and not re.search(r"[+&]", options):
            # (the variant's own words naming the material describe the very item sold, whatever its fact files or
            # the reader takes the word to name - a glass cover is glass to a store)
            return 2
        if named_as in ("among", "other"):
            # (a name that carries the material's own word is taken at that word, whatever else it names - what the
            # item is used on or with, a material beside it: only the variant's own words naming another material
            # make the item another)
            # (a word of the options that is another requirement's value - the colour asked for - is that requirement's
            # word, not a second material the variant names)
            others = {w for r in self.requirements if r is not requirement for form in self.forms(r) for w in form.split()}
            if named_as == "other" and self.says(title, requirement) and not (({singular_form(w) for w in letter_digit_tokens(options)} & words) - value_words - others):
                return 1
            return 1 if named_as == "among" else -1
        contrary = bool(singles)        # its own fact names another material
        if named_as == "only" and self.own_words_say(key, requirement):
            return 2            # (the name says what the item is made of; a filed material is often that of a part)
        if named_as == "only":
            return 1            # (the reader takes another word on the page for the material - pvc for plastic: a
                                # store reads the material's own word, and only loosely its kinds)
        if self.says(title, requirement):
            named_too = ({singular_form(w) for w in letter_digit_tokens(title)} & words) - value_words
            # (another material beside it: unclear what the item is made of; so too where the store's own fact nearest
            # the property files another material as its only entry - the reader, who is shown that fact, decides)
            return 1 if named_too or (contrary and self.nearest_label(key, requirement) < 0) else 2
        if listed or self.says(title, requirement, loose=True):
            return 1
        if contrary:
            return -1
        return 1 if elsewhere else 0

    def plainness_level(self, key: tuple[str, str], requirement: dict) -> int:
        """How plainly the page states one required value: 3 where the store states that kind of property (the
        variant's options, the property's own fact), 2 in the listing's name, 1 unclearly (beside other values, in
        another fact, in another word for it), 0 nowhere, -1 when the page states something else for it."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        title = page.get("title")
        kind = self.property_place(requirement)
        if kind == "amount":
            # (a figure that is the maker's name - a brand written as a number and a letter - is no amount of the item)
            for token in {w for w in letter_digit_tokens(page.get("brand")) if re.fullmatch(r"\d+[a-z]{1,2}", w)}:
                options = re.sub(rf"(?i)(?<![a-z0-9.]){token}(?![a-z0-9])", " ", options)
                title = re.sub(rf"(?i)(?<![a-z0-9.]){token}(?![a-z0-9])", " ", str(title or ""))
        related, inferred, words = self.own_labels(requirement)
        own = [(label, self.entries(value)) for label, value in self.named_facts(page, requirement)]
        singles = [(label, entries[0]) for label, entries in own if len(entries) == 1]
        lists = [(label, entry) for label, entries in own if len(entries) > 1 for entry in entries]
        stated = any(self.fact_entry_matches(entry, requirement, label) for label, entry in singles)
        listed = any(self.fact_entry_matches(entry, requirement, label) for label, entry in lists)
        value_words = {w for form in self.forms(requirement) for w in form.split()}
        elsewhere = any(self.says(value, requirement) for value in (page.get("facts") or {}).values())
        before = self.took_back.get(normalized_lower(requirement["value"]))
        if before and kind in ("material", "fact") and not re.search(r"\d", before) and not stem_set(before) & stem_set(requirement["value"]):
            # the shopper took that value back for this one: a page that still states it anywhere is what they no
            # longer want, whatever else it says
            gone = dict(requirement, value=before, written=[], alike=[])
            texts = [title, options, *(page.get("facts") or {}).values()]
            if kind == "fact" and self.says(title, requirement, whole=True) and any(self.fact_entry_matches(entry, requirement, label, exact=True) for label, entry in singles):
                # (a name that states the new value of a property kept as a fact beside the old one leaves the property
                # open: its own fact that is the new value and nothing else settles it. Not so a material: the name says
                # what the item is made of, and a withdrawn material it still names is still in it)
                texts = texts[1:]
            if any(self.says(text, gone, whole=True) for text in texts):
                return -1

        if kind == "amount":
            level = self.amount_level(page, options, title, requirement, singles, listed)
            if level >= 1:
                return level
            # (the amount written in another unit of its measure - in millimetres for centimetres, in years for months -
            # states it less plainly: it counts only where nothing states it as asked, and not where the listing's name and a fact
            # both give one other amount against the options)
            same_in_title, other_in_title = self.amounts(title, requirement, convert=True)
            # (the variant's own options write the amount in another step of its unit - 105*105cm for 1.05 m - and no
            # other amount of that kind: the options describe the very variant sold, and are read before the name)
            in_options, other_in_options = self.amounts(options, requirement, convert=True)
            if in_options and not other_in_options and not self.named_otherwise(options, requirement):
                return 2
            # (a title listing the sizes on offer, the required one among them, names the variants: the options choose)
            if (self.amount_level(page, options, title, requirement, singles, listed, convert=True) >= 2
                    and not self.options_outvoted(page, options, title, requirement) and (not other_in_title or same_in_title)):
                return 1
            return level

        named_as = self.named_as.get((key, requirement["property"], requirement["value"]))
        if kind == "colour":
            return self.colour_plainness(requirement, options, title, value_words, stated, listed, words, named_as)

        if kind == "material":
            return self.material_plainness(key, page, requirement, options, title, value_words, stated, listed, words, named_as, singles, elsewhere)

        # a property the store keeps as a fact of its own: the fact has to be the value, not only hold its words
        # (a "table lamp shade" is filed as another kind of thing than a "table lamp")
        if stated:
            return 3 if any(self.fact_entry_matches(entry, requirement, label, exact=True) for label, entry in singles) else 1
        # (another word sellers use for the value, as the only entry of its own fact or as one entry of its own list:
        # stated unclearly, not something else)
        if listed or any(self.fact_entry_matches(entry, requirement, label, loose=True) for label, entry in singles + lists):
            return 1
        if own:
            if self.filed_alike(page, requirement):
                return 1            # (the store keeps the property under a second label, its department's own, and that one is the value)
            # (an entry whose words begin with every word of the value may be the value in a longer form of its word:
            # the page is not read against it - the reader, who sees it then, decides)
            if value_words and any(all(any(w.startswith(v) and len(v) >= 4 for w in entry.split()) for v in value_words) for _, entry in singles):
                return 0
            return -1               # its own fact states something else
        if self.filed_elsewhere(page, requirement):
            return 1                # (another fact of the listing is the value and nothing else: the store has it on file)
        if related:
            return -1               # the store files this property as a fact, and this listing has none
        if inferred:
            return 0
        return 1 if elsewhere or self.says(title, requirement) else 0

    def exact_fact_score(self, key: tuple[str, str], requirement: dict) -> int:
        """How firmly the page states the value: each of the property's own facts whose only entry is exactly the
        value counts for the listing, each that states something else counts against it; options that are
        exactly the value count for it."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        score = 0
        for label, value in self.named_facts(page, requirement):
            entries = self.entries(value)
            if len(entries) == 1:
                score += 1 if self.fact_entry_matches(entries[0], requirement, label, exact=True) else (0 if self.fact_entry_matches(entries[0], requirement, label) else -1)
        return score + int(self.states(options, requirement))

    def filed_elsewhere(self, page: dict, requirement: dict) -> bool:
        """A fact that is not the property's own states the value as its only entry."""
        mine = {label for label, _ in self.named_facts(page, requirement)}
        for label, value in (page.get("facts") or {}).items():
            entries = self.entries(value)
            if str(label) not in mine and len(entries) == 1 and self.fact_entry_matches(entries[0], requirement, str(label), exact=True):
                return True
        return False

    def filed_alone(self, requirement: dict) -> set[str]:
        """The fact labels under which this shelf's listings file the value as their only entry, the value and nothing
        else, more often than inside a longer entry: where the longer entry is the store's usual way of filing the
        value, it is that value. (Worked out once for the listings read.)"""
        pages = self.counter.pages
        stamp = ("filed alone", requirement["property"], requirement["value"])
        mark = (len(pages), self.named_shelf(), len(requirement.get("written") or []))
        cached = self.own_cache.get(stamp)
        if cached and cached[0] == mark:
            return cached[1]
        alone: dict[str, int] = {}
        for page in [page for page in pages.values() if self.in_category(page)] or list(pages.values()):
            for label, value in (page.get("facts") or {}).items():
                entries = self.entries(value)
                if len(entries) == 1 and self.fact_entry_matches(entries[0], requirement, str(label)):
                    alone[str(label)] = alone.get(str(label), 0) + (1 if self.fact_entry_matches(entries[0], requirement, str(label), exact=True) else -1)
        found = {label for label, count in alone.items() if count > 0}
        self.own_cache[stamp] = (mark, found)
        return found

    def filed_alike(self, page: dict, requirement: dict) -> bool:
        """A fact that shares a word of its name with the property (beyond "type") states the value as its only entry."""
        mine = {label for label, _ in self.named_facts(page, requirement)}
        wanted = self.word_stem_set(requirement["property"]) - {"type"}
        for label, value in (page.get("facts") or {}).items():
            entries = self.entries(value)
            if (str(label) not in mine and wanted & self.word_stem_set(label) and len(entries) == 1
                    and self.fact_entry_matches(entries[0], requirement, str(label), exact=True)):
                return True
        return False

    def label_nearness(self, label: str, requirement: dict) -> tuple[int, int, int]:
        """How near a fact's label is to the property's name: the words they share less the label's other words;
        then whether its other words are the shelf's own name (a store keeps a property under a general label and
        under its department's - "garden_tools_type" on the garden tools shelf - and goes by the
        department's); then the words they share."""
        wanted, have = self.word_stem_set(requirement["property"]), self.word_stem_set(label)
        shelf = self.word_stem_set(self.named_shelf() or self.category or self.item)
        other = have - wanted
        departments = bool(other) and other <= shelf
        return (len(wanted & have) - (0 if departments else len(other)), int(departments), len(wanted & have))

    def nearest_label(self, key: tuple[str, str], requirement: dict) -> int:
        """Of the facts on the page that bear on the property - those named like it, and any whose only entry is the
        value: 1 when the one named nearest to the property is the value (alone, or among others under the property's
        own label), -1 when the nearest says something else, 0 when two as near disagree or no fact bears on it. A
        store keeps one property under several labels and goes by the one that is the property's own name."""
        page = self.counter.pages.get(key[0]) or {}
        mine = {label for label, _ in self.named_facts(page, requirement)}
        wanted = self.word_stem_set(requirement["property"])
        agree: tuple[int, int, int] | None = None
        differ: tuple[int, int, int] | None = None
        for label, value in (page.get("facts") or {}).items():
            label, entries = str(label), self.entries(value)
            if not entries:
                continue
            is_value = len(entries) == 1 and self.fact_entry_matches(entries[0], requirement, label, exact=True)
            # (a label that carries two words of the property's name bears on it too, whatever the shelf files under it)
            if not is_value and label not in mine and len(wanted & self.word_stem_set(label)) < 2:
                continue
            near = self.label_nearness(label, requirement)
            if not is_value and len(entries) == 1 and self.fact_entry_matches(entries[0], requirement, label):
                if label in self.filed_alone(requirement):
                    continue        # (but where the shelf's listings mostly file the value alone under that label, the longer
                                    # entry is a narrower kind the store files apart from it: it says neither way)
                is_value = True     # (one longer entry that holds the value - "indoor ceiling lights" - is still that value)
            if not is_value and len(entries) > 1 and label in mine and any(self.fact_entry_matches(entry, requirement, label, exact=True) for entry in entries):
                is_value = True     # (the property's own fact listing the value among others files the listing under it too:
                                    # such listings were paid as often as those that file the value alone)
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
        """One of the property's own facts on the page states, as its only entry, something else than the value."""
        page = self.counter.pages.get(key[0]) or {}
        for label, value in self.named_facts(page, requirement):
            entries = self.entries(value)
            if len(entries) == 1 and not self.fact_entry_matches(entries[0], requirement, label):
                return True
        return False

    def level_needed(self, requirement: dict) -> int:
        """The level at which a value counts as stated where the store states that kind of property."""
        return 2 if self.property_place(requirement) in ("amount", "material") else 3

    def settled(self, key: tuple[str, str], requirement: dict) -> bool:
        """The page's own words state the value, plainly or not."""
        return self.plainness_level(key, requirement) >= 1

    def row_score(self, key: tuple[str, str]) -> float:
        row = self.counter.rows.get(key) or {}
        title = stem_set(row.get("title"))
        score = 4.0 if key in self.counter.shelf else 0.0
        for requirement in self.requirements:
            need = stem_set(requirement["value"])
            others = [stem_set(form) for form in list(requirement.get("written") or []) + list(requirement.get("alike") or [])]
            if need and need <= title or any(other and other <= title for other in others):
                score += 3
            elif need & title:
                score += 1
        score += sum(1 for w in {singular_form(w) for w in self.item_words()} if w in title)
        # (a row that a search carrying every value now wanted showed near its top likely states them in its facts, which
        # the store searches too, even where its title shows none of them)
        # (in the shelf's spelling or the shopper's, closed up or apart - worked out once while the values
        # and the listings read stand as they are: rows are scored by the hundred)
        stamp = (len(self.counter.pages), len(self.counter.rows) + len(self.counter.shelf),
                 tuple((r["property"], r["value"], len(r.get("written") or []), len(r.get("shades") or [])) for r in self.requirements))
        if self.wanted_cache[0] != stamp:
            self.wanted_cache = (stamp, [{frozenset(letter_digit_tokens(form)) for v in pair for form in (v, join_number_unit(v))}
                                         for pair in zip(self.values(), self.values(store=False))])
        wanted = self.wanted_cache[1]
        if wanted:
            # (each such search counts: a row that several of them show near their top states the values somewhere the
            # store reads, whatever its title says)
            backing = {query for query, place in self.counter.hits.get(key, []) for said in [set(letter_digit_tokens(query))]
                       if place < 5 and all(any(words <= said for words in forms) for forms in wanted)}
            score += 3 * min(len(backing), 3)
        if normalized_lower(row.get("title")) in self.kin_titles() or normalized_lower(row.get("title")) in self.probed:
            score += 20
        elif "-" in key[1] and key[1].rsplit("-", 1)[0] in {k[1].rsplit("-", 1)[0] for k in self.verdicts if "-" in k[1] and self.listing_clears(k, "words")}:
            score += 20             # another variant of a fitting listing
        return score + (50 if self.listed and key in self.listed else 0)

    def listings_to_open(self, limit: int, allowance: int | None = None) -> list[dict]:
        allowance = self.open_room if allowance is None else allowance
        unopened = [k for k in self.counter.row_order if k[0] not in self.counter.pages and (self.affordable(k) or self.just_over(k, OPEN_REACH))]
        if self.listed:
            unopened = [k for k in self.listed if k[0] not in self.counter.pages] + [k for k in unopened if k in self.listed]
        # (rows first shown since the shopper last changed something go first: the searches sent since then carry the
        # values as they now stand, and a search matches a variant's options, which a row does not show)
        since = max(self.changed_at, self.widened_at)     # (and since the other words for the values were searched)
        fresh = (lambda k: self.counter.shown_at.get(k, 0) >= since) if since else (lambda k: False)
        # (a row a little over the limit is read after the rows within it that look as good: it can only be named, not bought)
        ranked = sorted(dict.fromkeys(unopened), key=lambda k: (not fresh(k), -(self.row_score(k) - (0 if self.affordable(k) else 2))))
        ids = list(dict.fromkeys(k[0] for k in ranked))[: max(0, min(limit, allowance - len(self.counter.pages)))]
        if self.counter.tool_available("compare"):
            return [make_tool_call("compare", product_ids=ids[i:i + OPEN_BATCH]) for i in range(0, len(ids), OPEN_BATCH)]
        return [make_tool_call("view", product_id=i) for i in ids]

    def candidates(self) -> list[tuple[str, str]]:
        out = []
        any_shelved = any(self.in_category(p) for p in self.counter.pages.values())
        for product_id, page in self.counter.pages.items():
            if any_shelved and not self.in_category(page):
                continue           # the store files the wanted kind under its own label: only those listings count
            for variant in page.get("variants") or []:
                key = (product_id, str(variant.get("sku")))
                if (self.affordable(key) or self.just_over(key)) and (not self.listed or key in self.listed):
                    out.append(key)
        return out

    def card(self, number: int, key: tuple[str, str]) -> str:
        page = self.counter.pages[key[0]]
        named_words = {w for r in self.requirements for w in self.word_stem_set(r["property"])}
        value_words = {w for r in self.requirements for w in stem_set(r["value"])}
        close, other = [], []
        for label, value in (page.get("facts") or {}).items():
            if self.word_stem_set(label) & named_words or stem_set(value) & value_words:
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

    def mark_listings(self) -> None:
        """Settle every unjudged candidate by the page's own words."""
        shelf_words = set(singular_phrase_key(self.named_shelf()).split())
        # (the request with the shelf's own label taken out where it writes it whole - a label with a note in brackets: the words
        # of that label said only there name the shelf, not a requirement of the item)
        shelf_phrase = singular_phrase_key(self.named_shelf())
        rest = f" {singular_phrase_key(self.counter.request)} "
        if shelf_phrase and f" {shelf_phrase} " in rest:
            rest = rest.replace(f" {shelf_phrase} ", " ", 1)
        said, named = rest.split(), singular_phrase_key(self.item).split()

        def the_shelf_itself(r: dict) -> bool:
            words = set(singular_phrase_key(r["value"]).split())
            # (a value whose words the request says again apart from the item's own name narrows a shelf that files
            # several kinds together to one of them: it is no restatement of the shelf)
            apart = bool(words) and all(said.count(w) > named.count(w) for w in words)
            return bool(r.get("stated") and shelf_words and words <= shelf_words and not apart)
        self.requirements = [r for r in self.requirements if not the_shelf_itself(r)]
        # (nor is it a value the first cart has to state: the request as first written is held to the same reading)
        self.opening = [r for r in self.opening if not the_shelf_itself(r)]
        self.refile()
        self.name_check()
        stamp = json.dumps([self.requirements, self.priority, self.lenient, len(self.named_as)], sort_keys=True)
        if stamp != self.judged_under:
            self.verdicts, self.judged_under = {}, stamp
        for key in self.candidates():
            if key not in self.verdicts:
                in_shelf = self.in_category(self.counter.pages[key[0]])
                met = [self.settled(key, r) or (self.lenient and self.plainness_level(key, r) >= 1) for r in self.requirements]
                self.verdicts[key] = {"kind_ok": in_shelf, "met": met, "by": "words"}

    def refile(self) -> None:
        """A value is read where this shelf files it, when the words it was read with point elsewhere. A value read as a
        colour that the shelf's listings name nowhere as one, and that two of them or more file as the only entry of
        one other fact, is a value of that fact: the requirement takes the label's words for its property (the request
        as first written too), so its listings are read under that fact, and a line about that property later is
        about this requirement. A value put in place of another that no listing files under the property's labels,
        whose first word two listings or more file there alone, is that entry: the rest only says what it is."""
        for requirement in self.requirements:
            kind = self.property_place(requirement)
            if kind == "colour":
                label = self.colour_filed_as(requirement)
                if not label:
                    continue
                old, name = requirement["property"], " ".join(letter_digit_tokens(label.replace("_", " ")))
                for r in [requirement] + [o for o in self.opening if (normalized_lower(o["property"]), normalized_lower(o["value"])) == (normalized_lower(old), normalized_lower(requirement["value"]))]:
                    r["property"], r["filed"] = name, not re.search(r"colou?r|material", name)
                self.record_note(f"{old} {requirement['value']!r} read as {name!r}: this shelf files it under {label!r} and names it nowhere as a colour")
            elif kind == "fact" and normalized_lower(requirement["value"]) in self.took_back:
                entry = self.filed_first_word(requirement)
                if entry:
                    old = requirement["value"]
                    self.took_back[entry] = self.took_back.pop(normalized_lower(old))
                    requirement["value"] = entry
                    self.record_note(f"{requirement['property']} {old!r} read as {entry!r}: this shelf files that word alone, and the value nowhere")
            if kind == "fact":
                label = self.type_filed_as(requirement)
                if label:
                    # a value given only as the item's "type" is read under the shelf's own label for that kind of type
                    # (a garden_tool_type label on a shelf of garden tools): among labels that each carry the word, the one that
                    # names the value alone is the store's, not one that names it within a longer entry
                    old, name = requirement["property"], " ".join(letter_digit_tokens(label.replace("_", " ")))
                    for r in [requirement] + [o for o in self.opening if (normalized_lower(o["property"]), normalized_lower(o["value"])) == (normalized_lower(old), normalized_lower(requirement["value"]))]:
                        r["property"], r["was"] = name, old
                    self.record_note(f"{old} {requirement['value']!r} read as {name!r}: this shelf files the value alone under {label!r}")

    def colour_filed_as(self, requirement: dict) -> str:
        """The label of the one fact, not a colour's, under which two listings or more of this shelf file a value read as
        a colour as their only entry, where the shelf keeps a colour fact on two listings or more and no listing names
        the value as a colour - no variant's options and no colour fact carry its words. A shelf that files no colour
        as a fact says nothing either way: the value is read as a colour. Empty otherwise. (Worked out once for the
        listings read.)"""
        pages = self.counter.pages
        stamp = ("colour filed as", requirement["property"], requirement["value"])
        cached = self.own_cache.get(stamp)
        if cached and cached[0] == (len(pages), self.named_shelf()):
            return cached[1]
        shelf = [page for page in pages.values() if self.in_category(page)] or list(pages.values())
        filed: dict[str, int] = {}
        coloured, found = 0, ""
        for page in shelf:
            facts = page.get("facts") or {}
            colours = [value for label, value in facts.items() if re.search(r"colou?r", normalized_lower(label))]
            if (any(self.says(variant.get("options"), requirement, whole=True) for variant in page.get("variants") or [])
                    or any(self.says(value, requirement, whole=True) for value in colours)):
                break               # (the shelf names it as a colour: it is read as one)
            coloured += bool(colours)
            for label, value in facts.items():
                entries = self.entries(value)
                # (the maker's name is no value of a property, whatever it says)
                if len(entries) == 1 and normalized_lower(value) != normalized_lower(page.get("brand")) and self.fact_entry_matches(entries[0], requirement, str(label), exact=True):
                    filed[str(label)] = filed.get(str(label), 0) + 1
        else:
            label = max(filed, key=lambda name: (filed[name], -len(name)), default="")
            found = label if filed.get(label, 0) >= 2 and coloured >= 2 else ""
        self.own_cache[stamp] = ((len(pages), self.named_shelf()), found)
        return found

    def type_filed_as(self, requirement: dict) -> str:
        """For a property named by its kind alone ("type"): the label carrying that word and more under which two
        listings or more of this shelf file the value as their only entry, the most often. Empty where the shelf keeps
        a label named by the bare word itself, or no such label files the value so. (Worked out once for the listings
        read.)"""
        bare = self.word_stem_set(requirement["property"])
        if not bare or not bare <= {"type", "kind"}:
            return ""
        pages = self.counter.pages
        stamp = ("type filed as", requirement["property"], requirement["value"])
        cached = self.own_cache.get(stamp)
        if cached and cached[0] == (len(pages), self.named_shelf()):
            return cached[1]
        shelf = [page for page in pages.values() if self.in_category(page)] or list(pages.values())
        filed: dict[str, int] = {}
        found = ""
        if not any(self.word_stem_set(label) == bare for page in shelf for label in (page.get("facts") or {})):
            for page in shelf:
                for label, value in (page.get("facts") or {}).items():
                    have, entries = self.word_stem_set(label), self.entries(value)
                    if have & bare and have - bare and len(entries) == 1 and self.fact_entry_matches(entries[0], requirement, str(label), exact=True):
                        filed[str(label)] = filed.get(str(label), 0) + 1
            label = max(filed, key=lambda name: (filed[name], -len(name)), default="")
            found = label if filed.get(label, 0) >= 2 else ""
        self.own_cache[stamp] = ((len(pages), self.named_shelf()), found)
        return found

    def filed_first_word(self, requirement: dict) -> str:
        """The first word of a value of two words or more, where no listing of this shelf files the value under the
        property's labels and two listings or more file that word there as their only entry. Empty otherwise."""
        words = letter_digit_tokens(requirement["value"])
        if len(words) < 2 or re.search(r"\d", requirement["value"]):
            return ""
        pages = self.counter.pages
        stamp = ("first word filed", requirement["property"], requirement["value"])
        cached = self.own_cache.get(stamp)
        if cached and cached[0] == (len(pages), self.named_shelf()):
            return cached[1]
        related, inferred, _ = self.own_labels(requirement)
        first = singular_form(words[0])
        filed = alone = 0
        for page in [page for page in pages.values() if self.in_category(page)] or list(pages.values()):
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    entries = self.entries(value)
                    filed += any(self.fact_entry_matches(entry, requirement, str(label)) for entry in entries)
                    alone += entries == [first]
        found = first if not filed and alone >= 2 else ""
        self.own_cache[stamp] = ((len(pages), self.named_shelf()), found)
        return found

    def ask_reader(self) -> None:
        """Have the reader look at the nearest misses: a synonym, another spelling or another unit may state what plain words did not find."""
        open_keys = [k for k in self.verdicts if self.affordable(k) and self.verdicts[k].get("by") == "words" and self.verdicts[k].get("kind_ok")
                     and not self.listing_clears(k, "words") and not any(self.plainness_level(k, r) == -1 for r in self.requirements)]
        open_keys.sort(key=lambda k: -sum(self.plainness_level(k, r) for r in self.requirements))
        batch = open_keys[:JUDGE_CARDS]
        if not batch or not self.can_think() or self.judged_count >= self.judge_room:
            return
        self.judged_count += len(batch)
        wanted = "\n".join(f"{i}. {r['property']}: {r['value']}" for i, r in enumerate(self.requirements, start=1)) or "(none beyond the kind of thing)"
        text = (f"The shopper wants: {self.item}\nRequirements, in order:\n{wanted}\n"
                + (f"Priority between fitting items: {self.priority}\n" if self.priority else "")
                + "\nListings:\n" + "\n".join(self.card(i, key) for i, key in enumerate(batch, start=1)))
        answer = self.reader.request_tool_fill(JUDGE_GUIDE, text, JUDGE_TOOL, 1400) or {}
        marks = answer.get("marks")
        marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
        for mark in marks:
            found = re.match(r"\s*\[?(\d+)\]?\W+([KkXx])\W*([YyNn\s,]*)", str(mark))
            if not found or not 1 <= int(found.group(1)) <= len(batch):
                continue
            key = batch[int(found.group(1)) - 1]
            said = [x in "Yy" for x in re.findall(r"[YyNn]", found.group(3))]
            said += [False] * (len(self.requirements) - len(said))
            # the reader may see a word stated in other words; an amount has to be found written on the page
            # (nor a colour: where no word of the page names one for the variant, the store files none for it)
            met = [self.settled(key, r) or (said[i] and self.plainness_level(key, r) != -1 and not re.search(r"\d", r["value"]) and self.property_place(r) != "colour")
                   for i, r in enumerate(self.requirements)]
            self.verdicts[key] = {"kind_ok": found.group(2) in "Kk", "met": met, "by": "reader"}
        for key in batch:
            if self.verdicts[key].get("by") == "words":
                self.verdicts[key]["by"] = "reader"      # shown once: not shown again
        for proof in answer.get("proof") or []:
            if isinstance(proof, dict) and isinstance(proof.get("n"), int) and 1 <= proof["n"] <= len(batch):
                self.verdicts[batch[proof["n"] - 1]].update({k: proof.get(k) for k in ("where", "words", "rank_value")})
        self.record_note(f"reader judged {len(batch)}: " + " | ".join(str(m)[:12] for m in marks)[:260])
        self.add_queries(answer.get("queries") or [])

    def listing_clears(self, key: tuple[str, str], bar: str) -> bool:
        """Whether a listing clears one bar.
        words: its words meet every requirement, whatever its stock is now.
        solid: every value is stated as required - in the options, the property's own fact or the title -
        not only in another word for it or in some other fact.
        firm: every value is stated where it counts most: the options or the property's own fact
        (an amount: the title too)."""
        if bar == "words":
            verdict = self.verdicts.get(key) or {}
            met = verdict.get("met") or []
            return bool(verdict.get("kind_ok")) and len(met) >= len(self.requirements) and all(met[: len(self.requirements)])
        if (self.verdicts.get(key) or {}).get("by") != "words":
            return False
        if bar == "solid":
            return all(self.plainness_level(key, r) >= 2 for r in self.requirements)
        if bar == "firm":
            return all(self.plainness_level(key, r) >= self.level_needed(r) for r in self.requirements)
        return False


    def listing_fits(self, key: tuple[str, str]) -> bool:
        if not self.listing_clears(key, "words") or not self.affordable(key):
            return False
        page = self.counter.pages.get(key[0]) or {}
        shelved = self.in_category(page) or not any(self.in_category(p) for p in self.counter.pages.values())
        brand = not self.brand or set(letter_digit_tokens(self.brand)) <= set(letter_digit_tokens(page.get("brand")))
        return shelved and brand

    def ranked(self) -> list[tuple[str, str]]:
        """Fitting listings, best first. Without a priority: the page that states the values most plainly, then
        price. With one: by what ordering the listing is expected to bring - how well it stands by the priority
        among the fitting listings, weighed by how sure it is that it fits at all (a listing that states every
        requirement plainly against one that states it unclearly)."""
        if self.better in ("higher", "lower"):
            higher = self.better == "higher"
        else:
            # (until the reader has said which way the priority runs: more is better, unless its words say less)
            higher = not re.search(r"\b(low|least|small|short|few|min|light)", normalized_lower(self.priority))
        by_price = self.priced_priority()
        by_figure = bool(self.priority) and not by_price
        fitting = [k for k in self.verdicts if self.listing_fits(k)]

        counting = higher and self.same_unit(self.counted_unit(), "pc")

        def number(key: tuple[str, str]) -> float | None:
            value = self.figure(key)
            value = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            # (a count the page writes is a figure, one piece included: a listing that states one piece stands above
            # one that states no count at all, which a store counts as the worst)
            return None if counting and value is not None and value <= 0 else value

        figures = {k: number(k) for k in fitting} if by_figure else {}
        trust = {k: self.trust(k) for k in fitting}
        # (a listing whose every value is at least surely stated fits as well as any other such listing: between those
        # the priority decides, not the small differences in how plainly each is worded)
        for key in fitting:
            if (trust[key] < FIRM_FIT and min((self.reliance(key, r) for r in self.requirements), default=FIRM_FIT) >= SURE_FIT - 1e-9
                    and all(self.plainness_level(key, r) >= 2 for r in self.requirements)):
                trust[key] = FIRM_FIT
        # (a listing a store may read as missing a value - a second amount or colour on its page, a value only listed
        # among others, only under a longer label - stands behind every listing that states each value plainly and is
        # likely to fit: a wrong order is worth nothing, a better figure on a listing that fits is worth only a share)
        doubts = {k: self.doubtful(k) for k in fitting}
        plain_fit = any(not doubts[k] and trust[k] >= LIKELY_FIT - 1e-9 for k in fitting)
        behind = {k: 1 if plain_fit and doubts[k] else 0 for k in fitting}
        # (and among listings that state a value only inside a list, one whose list runs on through a great many values
        # stands behind one with a short list: a store reads the long one as stating none of them in particular)
        spans = {k: self.list_span(k) for k in fitting}
        short_list = any(0 < spans[k] < LONG_LIST for k in fitting)
        spread = {k: 1 if short_list and spans[k] >= LONG_LIST else 0 for k in fitting}
        fixed = self.fixed_figure() if by_figure else None
        if fixed is not None and any(v is not None and abs(v - fixed) < 1e-9 for v in figures.values()):
            # the shopper fixed the very amount the priority compares, and some listing is that amount: one whose
            # figure is another amount (several of the item in one pack, a larger size) is not what was asked for
            for key in fitting:
                if figures[key] is not None and abs(figures[key] - fixed) >= 1e-9:
                    trust[key] = round(trust[key] * FAINT_FIT, 4)

        # how far each read figure can be taken as the listing's own: fully when it stands on the page on its own,
        # seldom when it is one side of a longer measurement or one of several figures of its kind
        weight = {k: (1.0 if self.figure_stated(k, figures[k]) else UNSURE_FIT) for k in fitting if figures.get(k) is not None}
        # the listings that state a figure, one for each name (variants and relistings of one article stand or fall
        # together): the surest of a name, with its figure and how likely it is to fit and to be counted
        named: dict[str, tuple[float, float]] = {}
        for k in weight:
            title = normalized_lower((self.counter.pages.get(k[0]) or {}).get("title")) or k[0]
            value = figures[k] if higher else -figures[k]
            if title not in named or (value, trust[k] * weight[k]) > named[title]:
                named[title] = (value, trust[k] * weight[k])
        rivals = sorted(named.items(), key=lambda item: -item[1][0])
        # the listings that state every requirement plainly come before the others whatever the figures say: many
        # weakly stated listings with a large figure must not outvote the one that surely fits
        sure = {k for k in fitting if trust[k] >= SURE_FIT - 1e-9}

        def standing(key: tuple[str, str]) -> float:
            """What ordering the listing is expected to bring by the priority, were it to fit: its figure as a share
            of the best figure among the listings that fit - each other listing counted as far as it is trusted to
            fit and its figure to be its own. A listing whose page states no figure for the priority is taken by a
            store to have none of it: it is the best only when no listing that states one fits after all."""
            own = normalized_lower((self.counter.pages.get(key[0]) or {}).get("title")) or key[0]
            mine = None if figures[key] is None else (figures[key] if higher else -figures[key])
            expected, left = 0.0, 1.0
            for title, (value, chance) in rivals:
                if title == own or (mine is not None and value <= mine):
                    continue
                if mine is not None and chance > 0:
                    share = (mine / value) if higher else (value / mine)       # (lower is better: both are negated)
                    expected += left * chance * max(0.0, min(1.0, share))
                left *= 1.0 - chance
            if mine is None:
                return left
            # (when its figure is not counted as its own after all, the listing stands like one that states none:
            # the best only if no other listing's figure counts either - never below such a listing)
            none = 1.0
            for title, (_, chance) in rivals:
                if title != own:
                    none *= 1.0 - chance
            return (expected + left) * weight[key] + (1.0 - weight[key]) * none

        def order(key: tuple[str, str]) -> tuple:
            price = (self.counter.stock.get(key) or {}).get("price") or 0.0
            plainly = sum(self.plainness_level(key, r) for r in self.requirements) + sum(self.exact_fact_score(key, r) for r in self.requirements)
            twin = normalized_lower((self.counter.pages.get(key[0]) or {}).get("title")) in self.lost_titles
            options = str(self.variant_of(key).get("options") or "").strip()
            # (options that state none of the wanted values - a tester, an add-on - go after plain or stating ones)
            blank = 1 if options and not any(self.says(options, r) or self.says(options, r, loose=True) for r in self.requirements) else 0
            rest = (0 if self.listing_clears(key, "firm") else 1, 0 if self.listing_clears(key, "solid") else 1, 0 if self.verdicts[key].get("by") == "words" else 1, -plainly,
                    0 if twin else 1, blank, price)
            if by_price:
                return (0, behind[key], spread[key], -trust[key], price) + rest
            if not by_figure:
                return (0, behind[key], spread[key], -trust[key]) + rest
            value = figures[key]
            worth = trust[key] * standing(key)
            # equal worth: the surer fit first (a wrong order is worth nothing, a fitting one mostly its full worth),
            # then the listing whose own options state the figure (they describe the exact variant sold)
            in_options = value is not None and any(abs(float(n) - value) < 1e-9 for n in re.findall(r"\d+(?:\.\d+)?", str(self.variant_of(key).get("options") or "")))
            # (a surely fitting listing that states no figure does not stand before the rest on that account: where a
            # fitting listing states one, a store pays nothing for it - it is weighed by its worth like the others)
            ahead = key in sure and (value is not None or not rivals)
            # (a store counts a listing whose page states no figure for the priority as the worst of the fitting ones:
            # it stands behind every listing that states one and may well fit)
            # (a figure only the reader gave - the page writes none - is no figure a store counts: it puts nothing behind)
            # (nor is a figure the page gives only loosely - without its unit, one of several sizes, one side of a longer
            # measurement - where another listing that may fit states one plainly: a store reads none there)
            unstated = ((value is None or weight.get(key, 0.0) < 1.0)
                        and any(k != key and figures.get(k) is not None and weight.get(k) == 1.0 and trust[k] >= LIKELY_FIT - 1e-9
                                for k in fitting))
            # (a listing the store may read as missing a value stands behind the plainly stated ones even where it alone
            # states a figure: a gate it fails pays nothing, while a page that shows no figure may still hold one)
            return (0, behind[key], spread[key], 1 if unstated else 0, 0 if ahead else 1, -round(worth, 6), -trust[key], 1 if value is None else 0,
                    -((value if higher else -value) if value is not None else 0.0), 0 if in_options else 1) + rest

        return sorted(fitting, key=order)

    def plan_unit_override(self) -> bool:
        """The reader's unit stands when it is (or scales to) a unit the plan says the priority's measure is written in, or
        when no such unit is known. Otherwise the first plan unit a fitting page writes an amount in counts, and the figures
        read so far (another measure, or none) give way to the pages' own amounts: True."""
        if self.unit_settled or self.counted_unit() == "pc":
            return False
        wanted = list(self.plan_units) + [u for name, units in self.measure_units.items()
                                          if self.priority_measure and self.word_stem_set(name) == self.word_stem_set(self.priority_measure)
                                          for u in sorted(units)]
        have = self.counted_unit()
        if not wanted or any(have and (self.same_unit(have, self.unit_word(u)) or self.scaled(have, self.unit_word(u), any_step=True))
                             for u in wanted):
            self.unit_settled = True
            return False
        old = self.figure_unit
        for unit in wanted:
            self.figure_unit = unit
            if any(self.counted_figure(k)[0] for k in self.ranked()):
                self.unit_settled = True
                for k in list(self.figures):
                    self.figures[k] = None
                self.record_note(f"priority unit {old!r} -> {unit!r} (the plan's unit for {self.priority_measure!r})")
                return True
        self.figure_unit = old
        if have and self.plan_units:
            # (no fitting page writes the priority's measure in any of its units, and the reader's figures are in a unit
            # of another measure - ounces for a length: they are no figures for it, and searches for better listings go
            # out in the plan's own unit)
            self.unit_settled = True
            self.figure_unit = self.plan_units[0]
            for k in list(self.figures):
                self.figures[k] = None
            self.record_note(f"priority unit {old!r} measures something else: figures dropped, {self.figure_unit!r} instead")
            return True
        return False

    def counted_unit(self) -> str:
        """The unit the priority is counted in: the one the shopper's own words name (a count of pieces), else the one
        the reader gave the figures in."""
        if re.search(r"\bpieces?\b", normalized_lower(self.priority)):
            return "pc"
        return self.unit_word((re.findall(r"[a-z]+", normalized_lower(self.figure_unit)) or [""])[0])

    def counted_figure(self, key: tuple[str, str]) -> tuple[bool, float | None]:
        """The figure a store counts for the priority, by the page's own writing: the number written directly before
        the unit - in the variant's options first, then in a fact, then in the listing's name. (settled, figure):
        the first of those places that writes such an amount settles it, and where it writes several different
        ones (the sizes on offer: 250ml/400ml) the listing counts as stating none. A number joined to another by a
        letter (160x70cm) is no amount of its own; the last of a measurement joined by signs (90*40*55 cm) is.
        Not settled: no place writes an amount in that unit, or the unit is not known yet."""
        unit = self.counted_unit()
        if not unit:
            return (False, None)
        stamp = (unit, key)
        if stamp in self.counted_cache:
            return self.counted_cache[stamp]
        page = self.counter.pages.get(key[0]) or {}

        def written(text: Any) -> set[float]:
            out: set[float] = set()
            pairs = [(float(m.group(1)), m.group(2)) for m in re.finditer(r"(?<![\w.,#:~\-])(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", normalized_lower(text))]
            for number, word in pairs + self.listed_amounts(text):
                word = self.unit_word(word)
                if self.same_unit(word, unit):
                    out.add(number)
                elif self.scaled(word, unit, any_step=True):
                    out.add(round(number * (self.scaled(word, unit, any_step=True) or 0.0), 6))
            return out
        facts: set[float] = set()
        for label, fact in (page.get("facts") or {}).items():
            facts |= written(fact)
            bare = self.bare_figure(label, fact, unit)
            if bare is not None:
                facts.add(bare)     # (a bare number under a label that names the unit: content_ml)
        answer: tuple[bool, float | None] = (False, None)
        options = str(self.variant_of(key).get("options") or "")
        from_options = written(options)
        if from_options and written(page.get("title")) and self.count_of_other(options, page.get("title"), unit):
            from_options = set()        # (the options count another thing they name, not the item: its name says how many)
        named = written(page.get("title"))
        if not from_options and facts and len(named) > 1:
            # (a name that writes several figures of the unit sells several models or sizes: where the variant's options
            # point at one of them - glued to the unit or to a short form of it - and a fact written once for the whole
            # listing gives another, the fact is another model's, and the options' figure is this one's)
            pointed = {float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)([a-z]+)", normalized_lower(options))
                       if unit.startswith(self.unit_word(m.group(2))) or self.same_unit(self.unit_word(m.group(2)), unit)} & named
            if len(pointed) == 1 and not facts & pointed:
                facts = pointed
        for found in (from_options, facts, named):
            if found:
                answer = (True, next(iter(found)) if len(found) == 1 else None)
                break
        self.counted_cache[stamp] = answer
        return answer

    def count_of_other(self, options: str, title: Any, unit: str) -> bool:
        """The options give the amount right after a word the listing's name does not carry, and that this shelf never
        names a variant by on its own (as it does a colour or a size): the amount of that other thing - an add-on sold
        with the item."""
        named = {singular_form(w) for w in letter_digit_tokens(title)}
        if self.lone_cache[0] != len(self.counter.pages):
            alone = {entry for page in self.counter.pages.values() for variant in page.get("variants") or []
                     for entry in self.entries(variant.get("options")) if re.fullmatch(r"[a-z]{3,}", entry)}
            self.lone_cache = (len(self.counter.pages), alone)
        for m in re.finditer(r"([a-z]{3,})\s+(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", normalized_lower(options)):
            word = singular_form(m.group(1))
            if self.same_unit(self.unit_word(m.group(3)), unit) and word not in named and word not in self.lone_cache[1]:
                return True
        return False

    def bare_figure(self, label: Any, fact: Any, unit: str) -> float | None:
        """A fact that is a bare number under a label naming the unit (content_ml: 250): that figure in the unit. Not
        for a count of pieces: a store takes a bare number under a pieces label for no count of the listing's - a count
        is read where it is written with its word (5 pcs)."""
        if self.same_unit(unit, "piece"):
            return None
        bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(fact))
        if bare and any(self.same_unit(word, unit) for word in letter_digit_tokens(str(label).replace("_", " "))):
            return float(bare.group(1))
        return None

    def figure_stated(self, key: tuple[str, str], value: float) -> bool:
        """The figure stands on the page as a figure of its own with its unit - in the variant's options, or as the
        only such figure in the listing's name - and not as one side of a longer measurement, one of several sizes
        on offer, or a count given loosely. Unknown unit: nothing can be said, so yes."""
        unit = self.counted_unit()
        if not unit:
            return True
        settled, counted = self.counted_figure(key)
        if settled:             # (read from the page as a store counts it)
            return counted is not None and abs(counted - float(value)) < 1e-9

        def alike(found: str) -> bool:
            return self.same_unit(found, unit)

        def stands_in(text: Any) -> bool | None:
            """Figures that stand on their own in the text, each with the word after it (not one glued to the next
            figure of a measurement): the listing's figure is the only one in its unit there. None: no such figure."""
            # (a hyphen before the figure parts it from a word or code - blue-15l, a-10pcs: the figure stands on its own,
            # as a store reads it, when the unit is written as such - "-2 layer" is no litre; one glued to a letter, a
            # digit or a point does not)
            low, found = normalized_lower(text), []
            for m in re.finditer(r"(?<![\w.#:~])(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", low):
                word = m.group(2)
                if (m.start() and low[m.start() - 1] == "-" and self.unit_word(word) != self.unit_word(unit)
                        and not (len(word) <= 3 and self.scaled(self.unit_word(word), self.unit_word(unit), any_step=True))):
                    continue
                found.append((float(m.group(1)), word))
            same = {number for number, word in found if alike(word)}
            if same:
                return same == {float(value)}
            other = [word for number, word in found if number == float(value) and self.scaled(self.unit_word(word), unit, any_step=True)]
            if other:       # (the figure in another unit of the same measure - "3 or more" names no unit)
                return len({number for number, word in found if word == other[0]}) == 1
            return None
        options, title = self.variant_of(key).get("options"), (self.counter.pages.get(key[0]) or {}).get("title")
        for text in (options, title):
            stated = stands_in(text)
            if stated is not None:
                return stated
        # neither gives a figure of its own: it was read from a fact. A store takes such a figure as the listing's
        # when the fact writes it with its unit, or bare under a label that names the unit (not a count of pieces); a
        # figure inferred from another kind of fact (a single item for one piece) is not on the page at all
        for label, fact in ((self.counter.pages.get(key[0]) or {}).get("facts") or {}).items():
            if stands_in(fact) is True:
                return True
            if self.bare_figure(label, fact, unit) == float(value):
                return True
        return False

    def fixed_figure(self) -> float | None:
        """The amount a requirement fixes in the unit the figures for the priority are read in, when there is one."""
        unit = self.unit_word((re.findall(r"[a-z]+", normalized_lower(self.figure_unit)) or [""])[0])
        if not unit:
            return None
        for requirement in self.requirements:
            for form in self.forms(requirement):
                for found in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form):
                    if self.unit_word(found.group(2)) == unit:
                        return float(found.group(1))
        return None

    def trust(self, key: tuple[str, str]) -> float:
        """How far the listing is trusted to fit: for each requirement, how far the way the page states the value
        can be relied on, all of them together."""
        worth = 1.0
        for requirement in self.requirements:
            worth *= self.reliance(key, requirement)
        return round(worth, 4)

    def reliance(self, key: tuple[str, str], requirement: dict) -> float:
        """How far one stated value can be relied on, by where the page states it. A store files a property it
        keeps as a fact under that fact, and a listing whose own fact does not say the value is rarely filed under
        it, whatever its name says; an amount is as good in the name as in the options; a colour named beside
        other words is still mostly that colour; a material is mostly what the name says, surest where it is filed."""
        level, kind = self.plainness_level(key, requirement), self.property_place(requirement)
        if kind == "fact":
            nearest = self.nearest_label(key, requirement)
            if level >= 3:
                # (a store may keep the property under several labels and grades by the one that is the property's
                # own name: where that one says something else, or lists the value among others, the listing is not
                # filed under the value, whatever a label named less like the property says)
                if nearest < 0:
                    return FAINT_FIT
                if nearest > 0 and self.exact_fact_score(key, requirement) > 0 and not self.gainsaid(key, requirement):
                    return FIRM_FIT
                return SURE_FIT if nearest > 0 else LIKELY_FIT
            if level >= 1 and nearest < 0:
                return FAINT_FIT
            if level >= 1:
                # the property's own fact carries the value inside a longer entry, or another fact of the listing is
                # exactly the value: the store has the value on file for this listing, and such listings are nearly
                # always what was asked for; a value only the name carries is not
                page = self.counter.pages.get(key[0]) or {}
                within = [label for label, value in self.named_facts(page, requirement)
                          if len(self.entries(value)) == 1 and self.fact_entry_matches(self.entries(value)[0], requirement, label)]
                elsewhere = self.filed_elsewhere(page, requirement)
                if within and nearest <= 0 and not elsewhere and set(within) <= self.filed_alone(requirement):
                    # (the value only inside a longer entry - a narrower kind named beside it - under labels where the
                    # shelf's listings mostly file the value alone: the store files that kind apart from the value)
                    return LIKELY_FIT
                if within or elsewhere or nearest > 0:
                    return SURE_FIT
            return FAINT_FIT
        if kind == "amount":
            if level == 1 and self.amount_in_text_only(key, requirement):
                return UNSURE_FIT   # (only inside another fact's longer text - a description, a line of sizes - not filed so)
            return FIRM_FIT if level >= 2 else LIKELY_FIT if level == 1 else FAINT_FIT
        if kind == "colour":
            if (level >= 1 and self.named_as.get((key, requirement["property"], requirement["value"])) in ("among", "other")
                    and self.says(self.variant_of(key).get("options"), requirement) and self.exact_fact_score(key, requirement) <= 0
                    and self.beside_colour(key, requirement)):
                # (the variant's name carries the wanted colour's own word, and the reader finds it beside another colour
                # or part of a mix it takes for another colour: two colours in one name are another colour to a store,
                # however plainly the wanted word is written - such a variant is the last resort, never ahead of one
                # that is the colour alone)
                return FAINT_FIT
            if (level >= 1 and self.named_as.get((key, requirement["property"], requirement["value"])) == "other"
                    and self.says(self.variant_of(key).get("options"), requirement) and self.exact_fact_score(key, requirement) <= 0):
                # (the reader finds the variant's name another colour though it carries the wanted word - a word before it
                # that names another colour of its own - and no colour fact of its own files it: likely at most, never as
                # sure as a variant named by the colour alone, whatever this shelf names its variants by)
                return LIKELY_FIT
            if (level >= 1 and self.named_as.get((key, requirement["property"], requirement["value"])) == "other"
                    and self.variant_of(key).get("options") and not self.says(self.variant_of(key).get("options"), requirement)
                    and (self.exact_fact_score(key, requirement) > 0 or level >= 3)):
                # (the variant's own options name another colour and only a fact gives the wanted one: the store reads
                # the options first - likely at most, behind a variant whose options name the colour)
                return LIKELY_FIT
            if level >= 3:
                # the variant named by the colour's own word (or filed under it) is that colour; a shortened or
                # other spelling of the word in its place is nearly as sure, not as sure
                own_word = dict(requirement, written=[], alike=[])
                if self.second_colour(key, requirement) or (self.fused_colour(key, requirement) and self.exact_fact_score(key, requirement) <= 0):
                    return LIKELY_FIT       # (another part of the variant's name carries another colour, or the colour
                                            # is fused into a longer word by a hyphen and no fact of its own files it)
                # (firmly so where a part of the options is the colour and nothing else, or its own fact is: beside a
                # seller's code - kx12-white - the store may file the variant under the code)
                return FIRM_FIT if self.states(self.variant_of(key).get("options"), own_word) or self.exact_fact_score(key, requirement) > 0 else SURE_FIT
            # beside other words the variant's own options still mostly mean that colour, and so does the colour the
            # store files for the listing; a colour only the name carries, or only another word for it, seldom is
            own = self.says(self.variant_of(key).get("options"), requirement) or self.exact_fact_score(key, requirement) > 0
            if level >= 1 and own and (self.second_colour(key, requirement) or self.fused_colour(key, requirement) or self.colour_shared(key, requirement)) and self.exact_fact_score(key, requirement) <= 0:
                return LIKELY_FIT       # (a variant named by two colours is as often filed under the other one)
            return SURE_FIT if level >= 1 and own else UNSURE_FIT if level >= 1 else FAINT_FIT
        if kind == "material" and level >= 1 and self.among_materials(key, requirement):
            return LIKELY_FIT       # (one entry of its own fact beside other materials, and no word of its name or options)
        if kind == "material" and level == 1 and not self.own_words_say(key, requirement):
            return UNSURE_FIT       # (only the reader's other words for the material reach it: no word of the value itself)
        return FIRM_FIT if level >= 3 else SURE_FIT if level >= 1 else UNSURE_FIT

    def own_words_say(self, key: tuple[str, str], requirement: dict) -> bool:
        """The page's title, the variant's options or a fact of the page carries the value in its own words (or a
        spelling of them), not only in another word for it."""
        page = self.counter.pages.get(key[0]) or {}
        texts = [page.get("title"), self.variant_of(key).get("options")] + [value for _, value in self.named_facts(page, requirement)]
        if any(self.says(text, requirement) for text in texts if text):
            return True
        forms = self.forms(requirement)
        # (another fact counts when its one entry is the value itself, not the value beside the label's own word)
        return any(len(self.entries(value)) == 1 and self.entries(value)[0] in forms for value in (page.get("facts") or {}).values())

    def among_materials(self, key: tuple[str, str], requirement: dict) -> bool:
        """The page states the material only as one entry among others of the property's own facts, no fact of the
        page gives it as its only entry, and neither the listing's name nor the variant's options carry its word: what
        the item is made of is left open."""
        page = self.counter.pages.get(key[0]) or {}
        own = [(label, self.entries(value)) for label, value in self.named_facts(page, requirement)]
        if not own or any(len(set(entries)) < 2 for _, entries in own):
            return False

        def says(entry: str, label: str) -> bool:
            return self.fact_entry_matches(entry, requirement, label) or self.fact_entry_matches(entry, requirement, label, loose=True)
        marks = [says(entry, label) for label, entries in own for entry in entries]
        if not any(marks) or all(marks):
            return False
        for label, value in (page.get("facts") or {}).items():
            entries = self.entries(value)
            if len(entries) == 1 and says(entries[0], str(label)):
                return False
        return not any(self.says(text, requirement) or self.says(text, requirement, loose=True)
                       for text in (page.get("title"), self.variant_of(key).get("options")))

    def amount_in_text_only(self, key: tuple[str, str], requirement: dict) -> bool:
        """The page states the amount only inside the longer text of a fact that is not the property's own - every fact
        that has it gives several figures - and nowhere in the options, the title or the property's own facts."""
        page = self.counter.pages.get(key[0]) or {}
        if any(self.has_amount(text, requirement) for text in [self.variant_of(key).get("options"), page.get("title")]):
            return False
        if any(self.has_amount(value, requirement) for _, value in self.named_facts(page, requirement)):
            return False
        holding = [str(value) for value in (page.get("facts") or {}).values() if self.has_amount(value, requirement)]
        return bool(holding) and all(len(re.findall(r"\d+(?:\.\d+)?", value)) > 1 for value in holding)

    def colour_shared(self, key: tuple[str, str], requirement: dict) -> bool:
        """The part of the variant's options that names the colour joins it to another value, or names another thing
        the colour may belong to (and the reader did not say the colour is the item's alone)."""
        if self.named_as.get((key, requirement["property"], requirement["value"])) == "only":
            return False
        wanted = self.forms(requirement)
        title = (self.counter.pages.get(key[0]) or {}).get("title")
        for text in re.split(r"[/,;|]+", normalized_lower(self.variant_of(key).get("options"))):
            if any(set(form.split()) <= {singular_form(w) for w in letter_digit_tokens(text)} for form in wanted):
                if self.joined_value(text, wanted) or self.names_other_thing(text, wanted, title):
                    return True
        return False

    def beside_colour(self, key: tuple[str, str], requirement: dict) -> bool:
        """The part of the variant's options that names the wanted colour also carries a word this shelf names
        variants by on its own (another variant's whole name, or an entry of a colour fact): two colours in one name,
        not the colour with a describing word."""
        stamp = (requirement["value"], len(self.counter.pages))
        known = self.shelf_colours.get(stamp)
        if known is None:
            related, inferred, _ = self.own_labels(requirement)
            known = set()
            for page in self.counter.pages.values():
                for label, value in (page.get("facts") or {}).items():
                    if str(label) in related or str(label) in inferred:
                        known |= {singular_form(e) for e in self.entries(value) if re.fullmatch(r"[a-z]{3,}", e)}
                for variant in page.get("variants") or []:
                    known |= {singular_form(part.strip()) for part in re.split(r"[/,;|]+", normalized_lower(variant.get("options")))
                              if re.fullmatch(r"[a-z]{3,}", part.strip())}
            known -= {w for form in self.forms(requirement) for w in form.split()}
            self.shelf_colours[stamp] = known
        for text in re.split(r"[/,;|]+", normalized_lower(self.variant_of(key).get("options"))):
            words = {singular_form(w) for w in re.split(r"[^a-z]+", text) if w}
            if any(set(form.split()) <= words for form in self.forms(requirement)) and words & known:
                return True
        return False

    def fused_colour(self, key: tuple[str, str], requirement: dict) -> bool:
        """The variant's options carry the colour only as one piece of a word joined by a hyphen (to a seller's code, an
        amount, another word): such a name is that colour in some stores' filing and no colour at all in others."""
        options = normalized_lower(self.variant_of(key).get("options"))
        wanted = self.forms(requirement)
        alone = [{singular_form(w) for w in re.split(r"[\s()\[\]+]+", part) if w} for part in re.split(r"[/,;|]+", options)]
        if any(set(form.split()) <= part for form in wanted for part in alone):
            return False
        pieces: set[str] = set()
        for word in re.split(r"[\s/,;|()\[\]+]+", options):
            if "-" not in word:
                continue
            parts = [singular_form(p) for p in word.split("-") if p]
            others = [p for p in parts if not any(set(form.split()) <= {p} for form in wanted)]
            if others and all(re.search(r"\d", p) for p in others):
                continue        # (the colour beside an amount or a size, not fused into a code or another word)
            pieces |= set(parts)
        return any(set(form.split()) <= pieces for form in wanted)

    def second_colour(self, key: tuple[str, str], requirement: dict) -> bool:
        """The variant's options name another colour too, beside the required one or in another part of the name: a
        word that stands on this shelf as a colour of its own (another listing's whole variant name, or an entry of
        the colour fact)."""
        wanted = {w for form in self.forms(requirement) for w in form.split()}
        # (colours this shelf files as such: the single words under its colour facts - not every word a variant is named by)
        related, inferred, _ = self.own_labels(requirement)
        known: set[str] = set()
        for page in self.counter.pages.values():
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    known |= {entry for entry in self.entries(value) if re.fullmatch(r"[a-z]{3,}", entry)}
        known -= wanted
        options = str(self.variant_of(key).get("options") or "")
        parts = [{singular_form(w) for w in re.split(r"[^a-z0-9]+", normalized_lower(part)) if w} for part in re.split(r"[/,;|]+", options)]
        named = any(set(form.split()) <= words for form in self.forms(requirement) for words in parts)
        return named and any(words & known for words in parts)

    def keeps_request(self, key: tuple[str, str]) -> bool:
        """The listing is the kind asked for and states every requirement the request itself wrote, where the store
        states that kind of property: the first cart is the shopper's first sight of the choice and holds to those."""
        stated = [r for r in self.requirements if r.get("stated")]
        return bool((self.verdicts.get(key) or {}).get("kind_ok")) and all(self.plainness_level(key, r) >= self.level_needed(r) for r in stated)

    def meets_opening(self, key: tuple[str, str]) -> bool:
        """The listing is what the request first asked for: the kind, on its shelf, on sale within the first spending
        limit (and the present one), every value the request wrote stated where the store states that kind of
        property - whatever the shopper has said since."""
        counter = self.counter
        page = counter.pages.get(key[0]) or {}
        seen = counter.stock.get(key) or {}
        price = seen.get("price")
        if key in self.gone or seen.get("in_stock") is False or not isinstance(price, float):
            return False
        if any(limit is not None and price > limit + 1e-9 for limit in (self.opening_budget, self.budget)):
            return False
        if not (self.verdicts.get(key) or {}).get("kind_ok"):
            return False
        if not self.in_category(page) and any(self.in_category(p) for p in counter.pages.values()):
            return False
        if self.brand and not set(letter_digit_tokens(self.brand)) <= set(letter_digit_tokens(page.get("brand"))):
            return False
        return all(self.plainness_level(key, r) >= self.level_needed(r) or self.in_shelf_unit(key, r) for r in self.opening)

    def in_shelf_unit(self, key: tuple[str, str], requirement: dict) -> bool:
        """For the first cart: the listing states a required amount plainly, where the store states it, but only in
        another unit of its measure (inches for centimetres: that reads 1), on a shelf whose listings read so far never write
        that measure in the shopper's unit. There the shelf's own unit is the plain way to state it. (The reading
        itself, and so the choice, stays as it is.)"""
        if self.property_place(requirement) != "amount" or self.plainness_level(key, requirement) != 1:
            return False
        if not hasattr(self, "unit_on_shelf"):
            self.unit_on_shelf: dict[tuple, bool] = {}     # (amount, pages read) -> a listing on the shelf writes it in its unit
        stamp = (requirement["value"], tuple(str(w) for w in requirement.get("written") or []), len(self.counter.pages))
        writes = self.unit_on_shelf.get(stamp)
        if writes is None:
            pages = [p for p in self.counter.pages.values() if self.in_category(p)] or list(self.counter.pages.values())
            writes = self.unit_on_shelf[stamp] = any(
                any(self.amounts(text, requirement)) for p in pages
                for text in [p.get("title"), *((v or {}).get("options") for v in p.get("variants") or []), *(p.get("facts") or {}).values()] if text)
        if writes:
            return False
        page = self.counter.pages.get(key[0]) or {}
        options, title = str(self.variant_of(key).get("options") or ""), page.get("title")
        for token in {w for w in letter_digit_tokens(page.get("brand")) if re.fullmatch(r"\d+[a-z]{1,2}", w)}:
            options = re.sub(rf"(?i)(?<![a-z0-9.]){token}(?![a-z0-9])", " ", options)
            title = re.sub(rf"(?i)(?<![a-z0-9.]){token}(?![a-z0-9])", " ", str(title or ""))
        own = [(label, self.entries(value)) for label, value in self.named_facts(page, requirement)]
        singles = [(label, entries[0]) for label, entries in own if len(entries) == 1]
        listed = any(self.fact_entry_matches(entry, requirement, label) for label, entries in own if len(entries) > 1 for entry in entries)
        return self.amount_level(page, options, title, requirement, singles, listed, convert=True) >= 2

    def pick_stand_in(self) -> tuple[str, str] | None:
        """A listing to put in the cart before the choice is made. What the store and the shopper do at the first
        cart - the listing sells out or its price jumps, the shopper names a new limit or another value - falls on
        that listing, and the choice is made knowing it. It is the choice as it stands: the best listing that firmly
        fits everything known and meets the request as first written, kept at the order while it still fits."""
        # (the stand-in is the choice as it stands: the best of the listings that firmly fit everything known now and
        # meet the request as first written - none, when no listing does; the choice then goes in at the commit)
        # (failing that, one that fits everything known, though less plainly, before one that misses a value the
        # shopper has told: that one is swapped out at the order and costs it a share of its worth)
        fitting = [k for k in self.verdicts if self.listing_fits(k) and self.meets_opening(k) and self.affordable(k)]
        # (one that only meets the request as first written misses a value the shopper has told since: swapped out at
        # the order, it costs a share of the order's worth whatever happens, so once such a value is told none goes in)
        return (self.best_of([k for k in self.verdicts if self.stands_firm(k)]) or self.best_of(fitting)
                or (self.as_asked(list(self.verdicts)) if not self.needs_found else None))

    def second_stand_in(self, choice: tuple[str, str]) -> tuple[str, str] | None:
        """A listing to put in the cart between the first cart and the choice: another article than the choice, of
        the kind asked for, on sale within the limit - one that meets the request as first written when there is one
        (a store may count only such a listing as a cart), else the nearest other."""
        counter = self.counter
        title = normalized_lower((counter.pages.get(choice[0]) or {}).get("title"))

        def usable(key: tuple[str, str]) -> bool:
            page = counter.pages.get(key[0]) or {}
            return (key != choice and key[0] != choice[0] and key not in counter.cart and key not in self.gone and self.on_sale(key)
                    and bool((self.verdicts.get(key) or {}).get("kind_ok")) and normalized_lower(page.get("title")) != title)
        keys = [k for k in self.verdicts if usable(k)]
        if not keys or not counter.tool_available("add_to_cart") or not counter.tool_available("remove_from_cart") or counter.turns_remaining() < 3:
            return None
        wanted_now = set(self.ranked())
        # (not one of the listings that could still be the choice: what the store does to it is lost with it)
        return min(keys, key=lambda k: (k in wanted_now, not self.meets_opening(k), -sum(self.plainness_level(k, r) for r in self.requirements),
                                        (counter.stock.get(k) or {}).get("price") or 0.0))

    def cart_stand_in(self) -> None:
        """Put the stand-in in the cart, read its stock, and take in what the store and the shopper do next."""
        counter = self.counter
        if counter.cart or counter.adds or counter.over or counter.turns_remaining() < 9 or self.time_is_short() or not counter.tool_available("add_to_cart"):
            return
        self.mark_listings()
        self.rank_by_priority(every=True)       # (which listing stands last is known only once the figures are read)
        meeting = [k for k in self.verdicts if self.affordable(k) and self.meets_opening(k)]
        if self.name_stand_ins(meeting):
            self.mark_listings()                        # (the reader's word on their colour or material, as cart_early has it)
        key = self.pick_stand_in()
        if key is None:
            return
        title = str((counter.pages.get(key[0]) or {}).get("title") or "")[:50]
        self.record_note(f"stand-in: {key} {title!r} {self.variant_of(key).get('options')!r} turn={counter.turn} fits now={self.listing_fits(key)}")
        group = [make_tool_call("add_to_cart", product_id=key[0], sku=key[1])]
        if counter.tool_available("inspect_stock"):
            group.append(make_tool_call("inspect_stock", product_id=key[0], sku=key[1]))
        counter.send_environment_turn(group)
        if key not in counter.cart:
            return
        self.stand_in = key
        self.stand_in_basis = self.held_basis()
        seen = counter.stock.get(key) or {}
        price = seen.get("price")
        changed, _ = self.absorb_shopper_lines()
        if seen.get("in_stock") is False or (self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9):
            self.gone.add(key)
            self.owed_reaction = True
            self.pending_line = ""          # the shopper's reaction to the change is answered by the choice
        if changed and counter.turns_remaining() > TURN_RESERVE and not self.time_is_short():
            # the values as they now stand are searched before anything else, and what those searches show is
            # opened first
            fresh = self.next_searches(counter.group_size)
            if fresh:
                counter.send_environment_turn(fresh)
                self.absorb_quietly()
            opens = self.listings_to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + 60)
            if opens and not counter.over:
                counter.send_environment_turn(opens[: counter.group_size])
                self.absorb_quietly()
            self.mark_listings()
            self.search_and_read(rounds=4)

    def meets_chosen(self, key: tuple[str, str]) -> bool:
        """The listing meets what the shopper said matters more (the chosen side of a doubted pair), when there is one."""
        met = ((self.verdicts.get(key) or {}).get("met") or [])[: len(self.requirements)]
        marks = [x for x, r in zip(met, self.requirements) if r.get("chosen")]
        return bool(marks) and all(marks)

    def priced_priority(self) -> bool:
        """The shopper's priority is the price itself (the cheapest, the lowest price) - not something else that
        they want the most of for the price or within it."""
        text = normalized_lower(self.priority)
        paid = r"(?:price|cost|expensive|afford\w*|pay\w*|spend\w*)"
        return bool(re.search(r"\bcheap|\bbest (?:price|value|deal)\b|\bmost afford", text)
                    or re.search(rf"\b(?:low\w*|least|less|small\w*|minim\w*)\b(?:\s+\w+){{0,2}}?\s+{paid}", text)
                    or re.search(rf"\b{paid}\s+(?:\w+\s+)?(?:as\s+)?(?:low|little|less|least)\b", text))

    def figure(self, key: tuple[str, str]) -> Any:
        """The listing's figure for the shopper's priority: the one its page writes as a store counts it; where the
        page writes no amount in that unit at all, the one the reader took from it."""
        settled, counted = self.counted_figure(key) if key[0] in self.counter.pages else (False, None)
        if settled:
            return counted
        if key in self.figures:
            return self.figures[key]
        return (self.verdicts.get(key) or {}).get("rank_value")

    def rank_by_priority(self, every: bool = False) -> None:
        """When the shopper said what makes one fitting item better, read each fitting listing's figure for it
        (every: all of them, a batch at a time, even when only one listing fits)."""
        if not self.priority or self.priced_priority():
            return
        for _ in range(4 if every else 1):
            fitting = self.ranked()
            keys = [k for k in fitting if k not in self.figures][:JUDGE_CARDS]
            if not keys or not self.can_think() or len(self.figures) >= self.figure_room or (len(fitting) < 2 and not every):
                return
            self.rank_asks += 1
            text = (f"Item: {self.item}\nThe shopper's priority: {self.priority}\n"
                    + (f"Write every figure in this unit, as for the listings read before: {self.figure_unit}\n" if self.figure_unit else "")
                    + "\nListings:\n" + "\n".join(self.card(i, key) for i, key in enumerate(keys, start=1)))
            answer = self.reader.request_tool_fill(RANK_GUIDE, text, RANK_TOOL, 400) or {}
            if str(answer.get("better") or "").lower() in ("higher", "lower"):
                self.better = str(answer["better"]).lower()
            if not self.figure_unit and isinstance(answer.get("unit"), str):
                self.figure_unit = " ".join(answer["unit"].split())[:20]    # (figures of later batches must be in the same unit)
            values = [] if self.plan_unit_override() else answer.get("values")
            values = re.split(r"[\n;]+", values) if isinstance(values, str) else (values or [])
            entries = [str(entry) for entry in values]
            numbered = [re.match(r"\s*\[?(\d+)\]?\W+(-?\d+(?:\.\d+)?)", entry) for entry in entries]
            if not any(numbered) and len(entries) == len(keys):
                # (one bare figure per listing, in the order shown)
                for key, entry in zip(keys, entries):
                    bare = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*", entry)
                    if bare:
                        self.figures[key] = float(bare.group(1))
            else:
                for found in numbered:
                    if found and 1 <= int(found.group(1)) <= len(keys):
                        self.figures[keys[int(found.group(1)) - 1]] = self.favoured_end(float(found.group(2)), found.string[found.end():])
            for key in keys:
                self.figures.setdefault(key, None)      # read once: not asked again
            self.record_note(f"ranked by {self.priority!r}: " + " ".join(entry[:14] for entry in entries)[:200])

    def favoured_end(self, figure: float, rest: str) -> float:
        """A figure written as one end of a pair or a range (40x60cm, 30-100cm, 10 to 14 inch) counts at the end the
        priority favours; one joined to more than one other figure (90*40*55) stays as read."""
        pair = re.match(r"\s?(?:[x×*~\-–]|to)\s?(\d+(?:\.\d+)?)(?!\d|\.\d|\s?(?:[x×*~\-–]|to)\s?\d)", normalized_lower(rest))
        if not pair:
            return figure
        other = float(pair.group(1))
        return min(figure, other) if self.better == "lower" else max(figure, other)

    def best_figures(self) -> list[tuple[float, tuple[str, str]]]:
        """The fitting listings whose figure for the priority is known, the best first."""
        lower = self.better == "lower"
        known = [(float(self.figure(k)), k) for k in self.ranked() if k not in self.gone
                 and isinstance(self.figure(k), (int, float)) and not isinstance(self.figure(k), bool)]
        return sorted(known, key=lambda pair: pair[0] if lower else -pair[0])

    def hunt_best(self) -> None:
        """The shopper wants the fitting listing that stands best by a priority. Two fitting listings are enough to
        buy one; they are not enough to know the best. So before the first cart the figure of every fitting listing
        is read, and the store is searched for fitting listings with a better figure - as long as a round of that
        still turns up something better."""
        counter = self.counter
        if not self.priority or self.priced_priority() or not counter.tool_available("search"):
            return
        if not any(self.listing_fits(k) for k in self.verdicts):
            return              # (nothing fits yet: the turns go to finding a fit, not a better figure)
        best_so_far: float | None = None

        def apart(query: str) -> str:
            # (an amount closed up - 100ml, 1pcs - written apart, as listings mostly write it; not a figure run on from
            # letters, which is a code's, nor before an ordinal's ending or a single letter, which may be a code's too)
            return re.sub(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)(?!(?:st|nd|rd|th)\b)([a-z]{2,4})\b", r"\1 \2", normalized_lower(" ".join(query.split())))
        for round_number in range(BEST_ROUNDS):
            chosen = [k for k in counter.cart if k != self.stand_in]
            if counter.over or chosen or self.time_is_short() or not self.can_think() or counter.turns_remaining() <= TURN_RESERVE + 4:
                break               # (the stand-in in the cart is not the choice: the best is still looked for)
            self.rank_by_priority(every=True)
            found = self.best_figures()
            best = found[0][0] if found else None
            better = best is not None and (best_so_far is None or (best < best_so_far if self.better == "lower" else best > best_so_far))
            # (while no fitting listing states a figure yet, one more round may reach one that does - one, not every
            # round left)
            if (round_number and not better and (best is not None or best_so_far is not None or round_number > 1)
                    and counter.turns_remaining() <= TURN_RESERVE + 12):
                break               # the last round brought nothing better and few turns are left: the best is likely in hand
                                    # (with many turns left, one more round of other searches still costs nothing the order needs)
            best_so_far = best
            shown = "\n".join(f"- {value:g}: {str((counter.pages.get(key[0]) or {}).get('title') or '')[:90]} | {self.variant_of(key).get('options') or '-'}"
                              for value, key in found[:6]) or (
                "(no fitting listing states a figure yet: write searches that each carry one likely figure for the priority, "
                f"written the way listings write it{' in ' + self.figure_unit if self.figure_unit else ''} - several figures, "
                "joined to the unit and apart - so that listings stating such a figure show up)")
            text = (f"{self.state_text()}\n\nBest figures found so far for the priority, with the listings that carry them:\n{shown}"
                    f"\n\nSearches tried:\n" + "\n".join(sorted(self.sent)[-30:]))
            answer = self.listed_fields(self.reader.request_tool_fill(BEST_GUIDE, text, BEST_TOOL, 300))
            queries = list(dict.fromkeys(apart(q) for q in answer.get("queries") or [] if isinstance(q, str) and q.strip()))[:8]
            # a listing's other variants are sold under its own title: the fitting listings' titles with the required
            # values bring up their variants, the ones with a better figure among them
            every = " ".join(self.values())
            # (and the priority in the words this shelf writes it with, with the best figure so far: in the room of
            # the titles, not of the reader's searches)
            wording = [q for q in map(apart, self.priority_wording(best)) if normalized_lower(" ".join(q.split())[:QUERY_LENGTH]) not in self.sent]
            # (an amount goes out apart, and closed up as well where the turn has room after the titles: a search finds
            # only the spelling it is given, and sellers write both)
            closed = list(dict.fromkeys(join_number_unit(q) for q in wording + queries if join_number_unit(q) != q and join_number_unit(q) not in self.sent))
            siblings = []
            for key in [k for _, k in found] + [k for k in self.ranked() if k not in self.gone]:
                title = normalized_lower((counter.pages.get(key[0]) or {}).get("title"))
                if title and title not in self.kin_swept and len(siblings) < counter.group_size - len(queries) - len(wording):
                    self.kin_swept.add(title)
                    siblings.append(f"{every} {' '.join(letter_digit_tokens(title)[:12])}")
            queries = wording + siblings + queries
            queries += closed[: max(0, counter.group_size - len(queries))]
            self.record_note(f"better by {self.priority!r} than {best}: {queries}")
            if not queries:
                break
            self.search_room = max(self.search_room, len(self.sent) + len(queries))
            self.add_queries(queries, first=True)
            searches = self.next_searches(min(counter.group_size, len(queries)))
            if not searches:
                break
            counter.send_environment_turn(searches)
            self.absorb_quietly()
            self.widened_at = counter.turn      # (the rows this round's searches show are read first, as widen's are)
            opens = self.listings_to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + 60)
            if opens and not counter.over and counter.turns_remaining() > TURN_RESERVE + 2:
                counter.send_environment_turn(opens[: counter.group_size])
                self.absorb_quietly()
            self.mark_listings()
        self.rank_by_priority(every=True)

    def rethink_searches(self) -> bool:
        """Nothing found meets every requirement: the reader writes new searches from the request, the requirements,
        the searches tried and the titles seen, and the agent looks again. True when it did."""
        if self.search_asks >= 2 or not self.can_think() or self.time_is_short() or self.counter.turns_remaining() <= TURN_RESERVE + 2:
            return False
        self.search_asks += 1
        seen = list(dict.fromkeys(str(p.get("title") or "")[:90] for p in self.counter.pages.values()))[-25:]
        text = (f"{self.state_text()}\n\nSearches tried:\n" + "\n".join(sorted(self.sent)[-25:])
                + "\n\nTitles seen (none meets every requirement):\n" + "\n".join(seen))
        answer = self.listed_fields(self.reader.request_tool_fill(SEARCH_GUIDE, text, SEARCH_TOOL, 400))
        queries = [q for q in answer.get("queries") or [] if isinstance(q, str) and q.strip()][:6]
        self.record_note(f"new searches: {queries}")
        if not queries:
            return False
        self.search_room = max(self.search_room, len(self.sent) + len(queries))     # (whatever the looking has used up)
        self.open_room = max(self.open_room, len(self.counter.pages) + 40)          # (and room to read what they bring up)
        self.add_queries(queries, first=True)
        self.search_and_read(rounds=3, enough=1)
        return True

    def lost_queries(self) -> list[str]:
        """The titles of chosen listings that sold out or went over the limit, with the values as they now stand: the
        same article is often listed again, or sold in another variant that is still there."""
        every = " ".join(self.values())
        out = []
        for title in sorted(self.lost_titles)[:3]:
            words = letter_digit_tokens(title)
            if len(words) >= 2:
                out += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}"]
        return out

    def recart(self) -> tuple[str, str] | None:
        """The next fitting listing into the cart, its stock and price read again."""
        counter = self.counter
        for key in [k for k in self.ranked() if k not in self.gone][:2]:
            if counter.over or counter.turns_remaining() < 3:
                break
            group = [make_tool_call("add_to_cart", product_id=key[0], sku=key[1])]
            if counter.tool_available("inspect_stock"):
                group.append(make_tool_call("inspect_stock", product_id=key[0], sku=key[1]))
            counter.send_environment_turn(group)
            self.absorb_shopper_lines()
            seen = counter.stock.get(key) or {}
            if seen.get("in_stock") is not False and (self.budget is None or not isinstance(seen.get("price"), float) or seen["price"] <= self.budget + 1e-9):
                return key
            self.gone.add(key)
        return None

    def reply_before_paying(self, choice: tuple[str, str]) -> None:
        """The lines that changed a requirement get their reply before the order."""
        lines, self.unanswered = self.unanswered[-4:], []
        self.answer("\n".join(f"- {line}" for line in lines))

    def last_resort(self) -> None:
        """After a failure: still order the listing in the cart, or the best one left."""
        counter = self.counter
        if counter.over or counter.turns_remaining() < 1 or not counter.tool_available("place_test_order"):
            return
        alive = [k for k in counter.cart if k not in self.gone]
        key = alive[-1] if alive else next((k for k in self.ranked() if k not in self.gone), None)
        if key is None:
            return
        if key not in counter.cart:
            if counter.turns_remaining() < 2:
                return
            counter.send_environment_turn([make_tool_call("add_to_cart", product_id=key[0], sku=key[1])])
        counter.send_environment_turn([self.final_action(key)])

    def near_misses(self) -> list[tuple[str, str]]:
        def score(key: tuple[str, str]) -> tuple:
            verdict = self.verdicts[key]
            met = (verdict.get("met") or [])[: len(self.requirements)]
            chosen = sum(1 for x, r in zip(met, self.requirements) if x and r.get("chosen"))
            # (of the requirements not met, a material or a fact whose value the listing's own name still states: a store
            # may count a value its name carries though a fact or the reader says otherwise - nearer than one never said)
            title = (self.counter.pages.get(key[0]) or {}).get("title")
            carried = sum(1 for x, r in zip(met, self.requirements)
                          if not x and self.property_place(r) in ("material", "fact") and self.says(title, r, whole=True))
            # what the shopper said matters more outranks the count of other requirements met
            return (-int(bool(verdict.get("kind_ok"))), -chosen, -sum(1 for x in met if x), -carried, -self.said_in_part(key, met),
                    (self.counter.stock.get(key) or {}).get("price") or 0.0)
        return sorted((k for k in self.verdicts if self.affordable(k)), key=score)

    # ---- looking
    def absorb_quietly(self) -> None:
        if self.counter.heard:
            self.absorb_shopper_lines()

    def name_check(self) -> None:
        """For a colour or a material, have the reader say what the words of the likeliest listings name: the
        wanted value alone, the wanted value beside another, or another value. Sellers write a colour in other
        words, and a title that names two materials leaves open which one the item is made of."""
        for requirement in self.requirements:
            kind = self.property_place(requirement)
            if kind not in ("colour", "material") or not self.can_think() or self.name_checks >= 16:
                continue
            others = [r for r in self.requirements if r is not requirement]
            waiting = []
            for key in self.candidates():
                tag = (key, requirement["property"], requirement["value"])
                if tag in self.named_as or any(self.plainness_level(key, r) < 1 for r in others):
                    continue
                page = self.counter.pages.get(key[0]) or {}
                options = str(self.variant_of(key).get("options") or "").strip()
                if kind == "colour":
                    text = f"variant: {options}" if options else f"title: {str(page.get('title') or '')[:150]}"
                else:
                    text = f"title: {str(page.get('title') or '')[:150]}" + (f" | variant: {options}" if options else "")
                    # (and what the store files as its material, in the listing's own words)
                    text += "".join(f" | {label}: {str(value)[:60]}" for label, value in self.named_facts(page, requirement)[:2])
                waiting.append((tag, text))
            # (first the listings whose own words name the value but not plainly - the reader's answer decides them - the
            # better by the priority first; then those that name no value of it; last those that name another)
            higher = self.better != "lower"

            def due(entry: tuple) -> tuple:
                key = entry[0][0]
                options = str(self.variant_of(key).get("options") or "").strip()
                level = self.plainness_level(key, requirement)
                said = self.says(options or (self.counter.pages.get(key[0]) or {}).get("title"), requirement)
                value = self.figure(key) if self.priority and not self.priced_priority() else None
                value = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                # (a value fused to other characters - c-beige - is the reader's to decide too)
                fused = any(form.replace(" ", "") in normalized_lower(options).replace(" ", "") for form in self.forms(requirement)) if options else False
                # (a material fact naming another word - 'abs' for plastic - may name a kind of it: read before the rest)
                tier = 0 if (said or fused) and level < 3 else 1 if level >= 0 or kind == "material" else 2
                return (tier, value is None, -(value or 0.0) if higher else (value or 0.0))
            waiting = sorted(waiting, key=due)[:12]
            if not waiting:
                continue
            self.name_checks += 1
            asked = (f"Item: {self.item}\nProperty: {requirement['property']}\nWanted value: {requirement['value']}\n\nTexts:\n"
                     + "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(waiting, start=1)))
            answer = self.reader.request_tool_fill(NAMES_GUIDE, asked, NAMES_TOOL, 500) or {}
            marks = answer.get("marks")
            marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
            numbered = all(re.match(r"\s*\[?\d+", str(mark)) for mark in marks)
            if numbered or len(marks) == len(waiting):      # (marks without numbers count only when there is one per text)
                for position, mark in enumerate(marks, start=1):
                    word = re.search(r"\b(only|among|other|none)\b", normalized_lower(mark))
                    number = re.match(r"\s*\[?(\d+)", str(mark))
                    at = int(number.group(1)) if numbered else position
                    if word and 1 <= at <= len(waiting):
                        self.named_as[waiting[at - 1][0]] = word.group(1)
            for tag, _ in waiting:
                self.named_as.setdefault(tag, "unread")      # asked once, not asked again
            self.record_note(f"names {requirement['value']}: " + " ".join(f"{i}:{self.named_as[tag]}" for i, (tag, _) in enumerate(waiting, start=1))[:240])

    def seller_words(self, requirement: dict) -> list[str]:
        """The other words sellers use for one required value (the reader's), kept with the requirement."""
        if not self.can_think():
            return []
        answer = self.reader.request_tool_fill(WORDS_GUIDE, f"Item: {self.item}\nProperty: {requirement['property']}\nValue: {requirement['value']}", WORDS_TOOL, 200) or {}
        words = [str(w).strip() for w in answer.get("words") or [] if str(w).strip()][:6]
        name = "written" if re.search(r"\d", requirement["value"]) else "alike"
        requirement[name] = list(dict.fromkeys(list(requirement.get(name) or []) + words))
        return words

    def widen(self) -> None:
        """Too few listings fit by the words known: ask what else sellers call each value, and search for those."""
        self.widened = True
        for requirement in self.requirements:
            if not self.can_think():
                break
            answer = self.reader.request_tool_fill(WORDS_GUIDE, f"Item: {self.item}\nProperty: {requirement['property']}\nValue: {requirement['value']}", WORDS_TOOL, 200) or {}
            words = [str(w).strip() for w in answer.get("words") or [] if str(w).strip()][:6]
            name = "written" if re.search(r"\d", requirement["value"]) else "alike"
            requirement[name] = list(dict.fromkeys(list(requirement.get(name) or []) + words))
            # (the other values as a search writes them - 10pcs, not 10 pcs: a search finds only the spelling it is given)
            others = " ".join(v for r, v in zip(self.requirements, self.values()) if r is not requirement)
            self.add_queries([f"{word} {others} {self.item}" for word in words[:3]], first=True)
        for requirement in self.requirements:
            # a colour is often named by one of its shades, or by a thing of that colour: those names are searched too,
            # and a listing named so counts as that colour only loosely
            if self.property_place(requirement) != "colour" or not self.can_think():
                continue
            shades = self.colour_shades(requirement)        # (asked for once: when the colour was told, else now)
            if shades:
                requirement["alike"] = list(dict.fromkeys(list(requirement.get("alike") or []) + shades))
                others = " ".join(v for r, v in zip(self.requirements, self.values()) if r is not requirement)
                self.add_queries([f"{shade} {others} {self.item}" for shade in shades[:4]], first=True)
        self.record_note(f"widened: {[(r['value'], r.get('written'), r.get('alike')) for r in self.requirements]}")
        # (the other words reach listings that state a value only in them - in the options, which a row does not show:
        # their searches go out now, whatever the looking so far has used up, and what they show is read first)
        counter = self.counter
        if counter.tool_available("search") and not counter.over and counter.turns_remaining() > TURN_RESERVE + 2:
            self.search_room = max(self.search_room, len(self.sent) + counter.group_size)
            searches = self.next_searches(counter.group_size)
            if searches:
                counter.send_environment_turn(searches)
                self.absorb_quietly()
                self.widened_at = counter.turn
                opens = self.listings_to_open(4 * OPEN_BATCH, allowance=len(counter.pages) + 4 * OPEN_BATCH)
                if opens and not counter.over:
                    counter.send_environment_turn(opens[: counter.group_size])
                    self.absorb_quietly()
                self.mark_listings()
                if (not self.ranked() and not counter.over and counter.turns_remaining() > TURN_RESERVE + 2
                        and (not self.time_is_short() or monotonic() - self.started < EPISODE_SECONDS + 60)):
                    # nothing fits even in the other words: the nearest listing of the kind asked for that misses only
                    # what differs between variants (a colour, an amount) is searched by its own title with the values
                    # as they now stand - the same article listed again, or its other variants, may carry what it
                    # misses (a search and a read, no reader; widen runs once)
                    for near in [k for k in self.near_misses() if k not in self.gone and (self.verdicts.get(k) or {}).get("kind_ok")]:
                        missed = [r for r, met in zip(self.requirements, (self.verdicts[near].get("met") or [])) if not met]
                        if missed and all(self.property_place(r) in ("colour", "amount") for r in missed):
                            self.refill(near, wide=False)
                            break

    def refill(self, lost: tuple[str, str], wide: bool = True) -> None:
        """The chosen variant has gone out of stock. A search shows one variant of each listing - the best in
        stock - so the searches that led to the lost one now show that listing's next variant, and its own title
        alone shows the same article listed again. wide: also search the values with it and the shelf at its price."""
        counter = self.counter
        words = letter_digit_tokens((counter.pages.get(lost[0]) or {}).get("title"))
        if len(words) < 2 or not counter.tool_available("search"):
            return
        every = " ".join(self.values())
        own = " ".join(w for w in letter_digit_tokens(self.variant_of(lost).get("options")) if not w.replace(".", "").isdigit())
        queries = [" ".join(words), " ".join(words[:10])]
        if own:
            queries.append(f"{own} {' '.join(words[:10])}")       # the lost variant's own words: its nearest siblings share them
        queries += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.item} {' '.join(words[:3])}"]
        if wide:
            queries += [f"{every} {' '.join(words)}"] + self.shelf_queries(6)
        texts = list(dict.fromkeys(normalized_lower(" ".join(q.split())[:QUERY_LENGTH]) for q in queries))
        calls = [make_tool_call("search", query=text, k=10, in_stock=True) for text in texts]
        label, price = self.store_label(), (counter.stock.get(lost) or {}).get("price")
        if wide and label and isinstance(price, float) and counter.tool_available("filter"):
            # the same shelf at the lost variant's price: variants of one listing share their price
            ceiling = round(min(self.budget or price * 1.1, price * 1.1), 2)
            brand = str((counter.pages.get(lost[0]) or {}).get("brand") or "").strip()
            calls.append(make_tool_call("filter", category=label, max_price=ceiling, in_stock=True, k=10))
            if brand and normalized_lower(brand) not in ("none", "no brand", "oem"):
                calls.append(make_tool_call("filter", category=label, brand=brand, max_price=ceiling, in_stock=True, k=10))
        counter.send_environment_turn(calls[: counter.group_size])
        self.absorb_quietly()
        opens = self.listings_to_open(8 * OPEN_BATCH, allowance=len(counter.pages) + 8 * OPEN_BATCH)
        if opens and not counter.over:
            counter.send_environment_turn(opens)
            self.absorb_quietly()
        self.mark_listings()



    def search_and_read(self, rounds: int, enough: int = 2) -> None:
        """Search and read listings until enough of them fit, or the rounds run out. Every search loads the
        store, so none is sent once enough listings fit firmly."""
        counter = self.counter
        for round_number in range(rounds):
            if counter.over or counter.turns_remaining() <= TURN_RESERVE or self.time_is_short():
                break
            if not counter.cart and not counter.adds and counter.pages and self.first_cart_due(asking=False):
                # (nothing in the cart yet and the first cart is due: what it draws from the shopper and the store is
                # better heard now, with turns left to follow it, than after every round of looking)
                try:
                    self.cart_early(asking=False)
                except Exception as exc:    # the early cart must never cost the order
                    self.record_note(f"early cart while looking stopped: {type(exc).__name__}: {exc}")
            opens = self.listings_to_open(counter.group_size * OPEN_BATCH)
            if opens:
                counter.send_environment_turn(opens[: counter.group_size])
                self.absorb_quietly()
            elif round_number and len(counter.pages) >= self.open_room:
                # nothing more may be read: the searches of one round went out, and more rounds of them would go unread
                # (their turns belong to the stand-in, the search for the best and the order, which read with room of
                # their own)
                break
            self.mark_listings()
            fitting = [k for k in self.ranked() if self.listing_clears(k, "solid")]
            if len([k for k in fitting if self.listing_clears(k, "firm")]) >= enough:
                return
            if len(fitting) >= enough and not self.kin_pending() and round_number >= 2:
                return
            told_new = self.shelf_queries(8) if self.fresh_rounds > 0 and len(fitting) < enough else []
            # (while nothing fits, a value the rows of the searches with every value hardly ever name goes out once
            # more, with more weight)
            weighted = [] if any(self.listing_fits(k) for k in self.verdicts) else self.weighted_queries()
            if told_new:
                # the shopper has just said what they want and too little fits: the new values with the words this
                # shelf's listings use reach the listing a shopper's own search would (a search shows ten rows, and
                # the title of a listing read before brings back that listing's kin, not what is wanted now)
                self.fresh_rounds -= 1
                # (half of the turn's searches: every value now in force with the item's own name, in each spelling)
                named = [q for q in (normalized_lower(" ".join(str(v).split())[:QUERY_LENGTH]) for v in weighted + self.value_queries()) if q and q not in self.sent]
                named = list(dict.fromkeys(named))[: counter.group_size // 2]
                self.add_queries(told_new, first=True)
                self.queries = named + [q for q in self.queries if q not in named]
                searches = self.next_searches(counter.group_size)
            else:
                kin = self.kin_queries()
                if kin:
                    # (a listing that fits brings its kin: room for those searches, whatever was used before)
                    self.search_room = max(self.search_room, len(self.sent) + len(kin))
                self.add_queries(kin, first=True)
                if len(fitting) < enough:
                    self.add_queries(weighted + self.scarce_queries() + self.variant_queries() + self.kind_queries(), first=True)
                if len(self.queries) < counter.group_size // 2:
                    self.add_queries(self.shelf_queries(6))
                searches = self.shelf_filters(counter.group_size // 2)
                searches += self.next_searches(counter.group_size - len(searches))
            if not searches and not opens:
                break
            if searches:
                counter.send_environment_turn(searches)
                self.absorb_quietly()
        opens = self.listings_to_open(counter.group_size * OPEN_BATCH)
        if opens and not counter.over and counter.turns_remaining() > TURN_RESERVE:
            counter.send_environment_turn(opens[: counter.group_size])
            self.absorb_quietly()
        self.mark_listings()
        if len({normalized_lower((counter.pages.get(k[0]) or {}).get("title")) for k in self.ranked()}) < enough and not self.widened and not counter.over:
            self.widen()        # (one article's variants are one listing: they stand and fall together)
            return self.search_and_read(rounds=3, enough=enough)
        for _ in range(2):                  # plain words left too few: the reader may see the value stated in other words
            if len(self.ranked()) >= enough:
                return
            self.ask_reader()
        if len(self.ranked()) < enough and not self.lenient:
            self.lenient = True
            self.mark_listings()

    # ---- the cart and the order
    def places(self, key: tuple[str, str]) -> dict[str, str]:
        """Where a listing states things about itself: its title, the chosen variant's options, each fact by its label."""
        page = self.counter.pages.get(key[0]) or {}
        places = {"title": str(page.get("title") or ""), "options": str(self.variant_of(key).get("options") or "")}
        places.update({str(label): str(value) for label, value in (page.get("facts") or {}).items()})
        return places

    def stated_at(self, key: tuple[str, str], requirement: dict) -> dict | None:
        """The one place on the page that states the required value, with the page's own words for it: the
        property's own fact first, then the variant's options, then the title, then any other fact."""
        places = self.places(key)
        page = self.counter.pages.get(key[0]) or {}
        own = [label for label, _ in self.named_facts(page, requirement)]
        order = own + ["options", "title"] + [label for label in places if label not in own and label not in ("options", "title")]
        forms = sorted(self.forms(requirement) | {" ".join(letter_digit_tokens(join_number_unit(requirement["value"])))}, key=len, reverse=True)
        # (then the amount as a page may write it in another unit, apart or joined)
        forms += sorted({f for c in self.converted(requirement) for f in (c, join_number_unit(c))} - set(forms), key=len, reverse=True)
        for field in order:
            text = places.get(field) or ""
            for reading in (normalized_lower(text), join_number_unit(text)):
                for form in forms:
                    if not form:
                        continue
                    found = re.search(r"(?<![a-z0-9])" + re.escape(form).replace(r"\ ", r"\W+") + r"(?![a-z0-9])", reading)
                    if found and f" {' '.join(letter_digit_tokens(found.group(0)))} " in f" {' '.join(letter_digit_tokens(text))} ":
                        return {"field": field, "value": found.group(0)}
        return None

    def cited(self) -> dict | None:
        """The requirement whose value the shopper asked the order's reasons to show: the one named like the property
        the request asks them to point to (None when it names none, or none of the requirements is it)."""
        wanted = self.word_stem_set(self.cite) - {"item", "listing", "product"}
        if not wanted:
            return None
        scored = [(len(wanted & self.word_stem_set(r["property"])), -i, r) for i, r in enumerate(self.requirements)]
        best = max(scored, key=lambda x: (x[0], x[1]), default=None)
        return best[2] if best and best[0] > 0 else None

    def claims(self, key: tuple[str, str], about: dict | None = None) -> dict | None:
        """The reasons given with an order: its price, that it is in stock, and the one place on its page that
        states what the shopper asked for (about: the requirement to show; else the first one the page states)."""
        seen = self.counter.stock.get(key) or {}
        places = self.places(key)
        claims: list[dict] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        claims.append({"field": "in_stock", "value": seen.get("in_stock") is not False})
        fact = None
        verdict = self.verdicts.get(key) or {}
        ordered = ([about] if about else []) + [r for r in self.requirements if r.get("stated")] + [r for r in self.requirements if not r.get("stated")]
        for requirement in ordered:
            fact = self.stated_at(key, requirement)
            if fact:
                break
        if fact is None:
            # the reader's own proof for a listing it judged, when the plain words name no place
            for where, words in zip(verdict.get("where") or [], verdict.get("words") or []):
                where = re.sub(r"^facts?\s*[:.]\s*", "", str(where or ""))
                field = next((label for label in places if normalized_lower(label) == normalized_lower(where)), None)
                said = " ".join(letter_digit_tokens(words))
                if field and said and f" {said} " in f" {' '.join(letter_digit_tokens(places[field]))} ":
                    fact = {"field": field, "value": str(words).strip()}
                    break
        if fact:
            claims.append(fact)
        return {"claims": claims}

    def shortfall(self, key: tuple[str, str]) -> dict | None:
        """For the closest listing when nothing meets everything: its price, its stock, and the place on its page
        that shows what it has instead of the first requirement it misses."""
        seen = self.counter.stock.get(key) or {}
        page = self.counter.pages.get(key[0]) or {}
        places = self.places(key)
        claims: list[dict] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        claims.append({"field": "in_stock", "value": seen.get("in_stock") is not False})
        missed = sorted((r for r in self.requirements if self.plainness_level(key, r) < 1), key=lambda r: self.plainness_level(key, r)) or list(self.requirements)
        for requirement in missed:
            # the property's own fact, when the page has one, says what the listing is instead
            for label, value in self.named_facts(page, requirement):
                words = " ".join(letter_digit_tokens(value))
                if words:
                    claims.append({"field": label, "value": str(value).strip()})
                    return {"claims": claims}
        for field in ("options", "title"):
            if " ".join(letter_digit_tokens(places.get(field))):
                claims.append({"field": field, "value": places[field].strip()})
                break
        return {"claims": claims}

    def look_again(self, choice: tuple[str, str]) -> None:
        """The shopper asked for a number of fitting options to be opened before the choice. Once everything is
        settled - the answers, the changes, what the market did - that many fitting listings (and a few to spare)
        are opened under the requirements as they now stand, the chosen one among them."""
        counter = self.counter
        stamp = json.dumps([[(r["property"], r["value"]) for r in self.requirements], self.budget], sort_keys=True)
        if not self.look_at or self.looked_under == stamp or counter.over or counter.turns_remaining() < 2 or not (counter.tool_available("compare") or counter.tool_available("view")):
            return
        self.looked_under = stamp
        wanted = self.look_at + 2
        picked: list[str] = []
        titles: set[str] = set()
        for key in [choice] + [k for k in self.ranked() if k not in self.gone] + self.near_misses():
            title = normalized_lower((counter.pages.get(key[0]) or {}).get("title"))
            if key[0] in picked or (title in titles and key != choice) or not self.on_sale(key):
                continue
            picked.append(key[0])
            titles.add(title)
            if len(picked) >= wanted:
                break
        if counter.tool_available("compare"):
            calls = [make_tool_call("compare", product_ids=picked[i:i + OPEN_BATCH]) for i in range(0, len(picked), OPEN_BATCH)]
        else:
            calls = [make_tool_call("view", product_id=i) for i in picked]
        self.record_note(f"opened {len(picked)} fitting options before choosing")
        counter.send_environment_turn(calls)
        self.absorb_quietly()

    def final_action(self, choice: tuple[str, str]) -> dict:
        """The order for the choice. A listing that leaves a value unsaid, or words it another way, may still be what
        was asked for, so whatever is in the cart is ordered. Only when nothing could be put in the cart at all is
        the store told that no listing meets every requirement, the nearest one named with where it falls short."""
        if choice == self.named_nearest and self.can_abstain():
            # (the reasons as for an order: its price, its stock and the one place on its page that states what was
            # asked for - a shopper who asks for the reasons wants them for the listing named here too)
            return make_tool_call("place_test_order", product_id=choice[0], sku=choice[1], abstain=True, justification=self.claims(choice, self.cited()))
        if choice in self.counter.cart or not self.can_abstain():
            return make_tool_call("place_test_order", product_id=choice[0], sku=choice[1], justification=self.claims(choice, self.cited()))
        self.record_note(f"nothing could be put in the cart: abstain, closest {choice}")
        return make_tool_call("place_test_order", product_id=choice[0], sku=choice[1], abstain=True, justification=self.shortfall(choice))

    def nearest_now(self, key: tuple[str, str], read: bool = True) -> tuple[str, str] | None:
        """The listing to name as the nearest, by the store's present word on it (read: its stock and price are read
        again first). When something within the limit fits after all, that is put in the cart and ordered; when the
        named listing has sold out or moved, the next such listing is named; when none is left, the nearest on sale."""
        counter = self.counter
        if read and counter.tool_available("inspect_stock") and counter.turns_remaining() >= 2:
            counter.send_environment_turn([make_tool_call("inspect_stock", product_id=key[0], sku=key[1])])
            if counter.heard and self.absorb_shopper_lines()[0]:
                self.mark_listings()
        left = [k for k in self.ranked() if k not in self.gone]
        best_left = max((self.trust(k) for k in left[:12]), default=0.0)
        none_fits = 1.0
        for k in left[:12]:
            none_fits *= 1.0 - self.trust(k)
        if left and best_left < LIKELY_FIT and any(self.trust(k) >= SURE_FIT or (self.trust(k) >= LIKELY_FIT - 1e-9 and self.trust(k) > best_left + 1e-9)
                                                   or none_fits * self.trust(k) > best_left + 1e-9
                                                   for k in self.only_over()):
            left = []       # (only loose readings fit within the limit: the plainer listing over it still stands)
        if left:
            self.named_nearest = None
            held = self.stand_in_holds(left)
            if held is not None:
                left = [held] + [k for k in left if k != held]
            if left[0] not in counter.cart and counter.turns_remaining() >= 2:
                counter.send_environment_turn([make_tool_call("add_to_cart", product_id=left[0][0], sku=left[0][1])])
                self.absorb_quietly()
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
        """The store lets an order say that no listing meets every requirement."""
        spec = (self.counter.tools.get("place_test_order") or {}).get("parameters") or {}
        return "abstain" in (spec.get("properties") or {})

    def closest(self) -> tuple[str, str] | None:
        """The listing that comes nearest when none meets everything: of the kind asked for, in stock and within the
        budget, meeting what the request itself wrote first, then the most requirements."""
        def nearness(key: tuple[str, str]) -> tuple:
            verdict = self.verdicts.get(key) or {}
            met = (verdict.get("met") or [])[: len(self.requirements)]
            stated = sum(1 for x, r in zip(met, self.requirements) if x and r.get("stated"))
            return (-int(bool(verdict.get("kind_ok"))), -stated, -sum(1 for x in met if x), -self.said_in_part(key, met),
                    -int(key in self.counter.cart), (self.counter.stock.get(key) or {}).get("price") or 0.0)
        open_keys = [k for k in self.verdicts if self.on_sale(k)]
        return min(open_keys, key=nearness) if open_keys else None

    def said_in_part(self, key: tuple[str, str], met: list) -> int:
        """The requirements not met for which the property's own fact on the page gives words of the value and no
        other word: a store may file a value in short, and that is nearer than a page that says nothing of it."""
        page = self.counter.pages.get(key[0]) or {}
        count = 0
        for position, requirement in enumerate(self.requirements):
            if position < len(met) and met[position] or re.search(r"\d", requirement["value"]):
                continue
            wanted = {singular_form(w) for form in self.forms(requirement) for w in form.split()}
            for label, value in self.named_facts(page, requirement):
                named = self.word_stem_set(label)
                words = [w for entry in self.entries(value) for w in entry.split() if not any(same_word_stem(w, n) for n in named)]
                if words and all(any(same_word_stem(w, v) for v in wanted) for w in words):
                    count += 1
                    break
        return count

    def on_sale(self, key: tuple[str, str]) -> bool:
        """In stock and within the budget by what the store last showed (whether or not it was set aside as a choice)."""
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        return seen.get("in_stock") is not False and (self.budget is None or not isinstance(price, float) or price <= self.budget + 1e-9)

    @staticmethod
    def says_value(line: str, value: str) -> bool:
        """The line carries the value: each figure of it stands in the line; a value in words has at least one of
        its telling words there."""
        figures = re.findall(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
        if figures:
            said = {float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(line).replace(",", ""))}
            return all(float(f) in said for f in figures)
        said_words = stem_set(line) | set(letter_digit_tokens(normalized_lower(line)))
        words = [w for w in letter_digit_tokens(normalized_lower(value)) if len(w) >= 3]
        return not words or any(w in said_words or stem_set(w) & said_words for w in words)

    @staticmethod
    def says_amount(text: str, amount: float) -> bool:
        """The amount stands in the text as a figure (however its thousands and cents are written)."""
        figures = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(text).replace(",", ""))]
        return any(abs(figure - amount) < 0.005 for figure in figures)

    def holds_now(self, said: str) -> tuple[list[dict], bool]:
        """Of what the shopper lines being answered newly required: the requirements that still hold, and whether
        one of the lines set the spending limit that holds."""
        lines = [line for line in list(self.fresh) + list(self.fresh_limit) if line and line in said]
        values = {normalized_lower(v) for line in lines for v in self.fresh.get(line, [])}
        return [r for r in self.requirements if normalized_lower(r["value"]) in values], any(line in self.fresh_limit for line in lines)

    def given_up(self, line: str, value: str) -> bool:
        """The value is one the shopper gave up: written in a line only after "instead of", or taken back for another."""
        low = normalized_lower(value)
        if low not in self.took_back and any(normalized_lower(v) == low for v in self.took_back.values()):
            return True
        said = rf"(?<![a-z0-9]){re.escape(low)}(?![a-z0-9])"
        for part in normalized_lower(line).split("\n"):
            for swap in re.finditer(r"\binstead of\b", part):
                # (what stands after "instead of", up to the end of that clause, is the value given up)
                given = re.split(r"[,.;:!?]|\b(?:and|but|while|so|keep\w*|everything)\b", part[swap.end():], maxsplit=1)[0]
                if re.search(said, given) and not re.search(said, part[:swap.start()]):
                    return True
        return False

    def priority_quantity(self, text: str) -> bool:
        """The property is the quantity the priority compares: the measure the plan reader names for it, or wholly
        words of the priority sentence; with no measure read, any word shared with the sentence."""
        words = self.content_words(text)
        if not words or not self.priority:
            return False
        if self.priority_measure:
            return self.word_stem_set(text) == self.word_stem_set(self.priority_measure) or words <= self.content_words(self.priority)
        return bool(words & self.content_words(self.priority))

    def answer(self, line: str, beside: list[dict] | None = None) -> None:
        """Reply to the shopper lines that changed a requirement: one plain statement of what it is now. A line that
        changed none is owed no reply (a new spending limit is met by what is ordered), and a statement that goes
        over several things is easily taken for a question."""
        counter = self.counter
        if counter.over or not counter.tool_available("message") or counter.turns_remaining() <= 2:
            return
        if line in self.unanswered:
            self.unanswered.remove(line)
        news, _ = self.holds_now(line)
        if not news and re.search(r"\bconfirm|\bright\?|\bok(?:ay)?\?", normalized_lower(line)):
            # (the line asks to have its change confirmed, and what it asks for already held - an answer had said it
            # before: the shopper is owed the reply all the same, about the values the line names)
            news = [r for r in self.requirements if self.says_value(line, r["value"]) and not self.given_up(line, r["value"])]
        if not news:
            return
        parts = []
        for requirement in news:
            before = self.took_back.get(normalized_lower(requirement["value"]))
            if before:
                # (the old value as the shopper's line says it - the words of the property's own name aside - or none)
                rest = " ".join(w for w in letter_digit_tokens(line) if singular_form(w) not in self.word_stem_set(requirement["property"]))
                said_old = " ".join(letter_digit_tokens(before))
                if not said_old or not re.search(rf"(?<![a-z0-9]){re.escape(said_old)}(?![a-z0-9])", rest):
                    before = ""
            parts.append(f"the {requirement['property']} is now {requirement['value']}" + (f" instead of {before}" if before else ""))
        # (said so that it reads as the reply it is: yes, the change, and that nothing else moved - a shopper who asks
        # about the change often asks in the same breath whether the budget stands)
        counter.send_environment_turn([make_tool_call("message", content=("Yes, confirmed: " + ", and ".join(parts) + ". Everything else stays the same, your budget included.")[:400])]
                     + list(beside or []))
        self.reply_sent = (counter.turn, line)

    def open_episode(self) -> None:
        """First question, first searches, early cart, and the search for a better listing."""
        counter = self.counter
        question = self.read_opening_request()
        self.opening, self.opening_budget = [dict(r) for r in self.requirements if r.get("stated")], self.budget
        # first turn: the first question (the request nearly always leaves something open) together with the first searches
        first: list[dict] = []
        if question and counter.tool_available("message"):
            first.append(self.shopper_message(question))
        if self.listed and counter.tool_available("compare"):
            first.append(make_tool_call("compare", product_ids=list(dict.fromkeys(k[0] for k in self.listed))[:OPEN_BATCH]))
        filters = [make_tool_call("filter", category=guess, max_price=self.budget, in_stock=True, k=10) for guess in self.label_guesses()[:5]] if counter.tool_available("filter") else []
        first.extend(self.next_searches(counter.group_size - len(first) - len(filters)))
        counter.send_environment_turn(first + filters)
        self.listen()
        if self.pair_again and not self.pair_chosen and counter.tool_available("message") and not counter.over:
            again = [self.shopper_message(self.pair_question)] + self.next_searches(counter.group_size - 1)
            counter.send_environment_turn(again)
            self.listen()
        if self.pair_fallback and not self.pair_chosen:
            # the answer to the pair named nothing: the two were not the open point, so what was set aside as a
            # wish is a requirement again, and the open point is looked for like any other
            self.requirements += self.set_aside
            self.opening += [dict(r) for r in self.set_aside if r.get("stated")]
            self.set_aside, self.pair_fallback = [], ""
            self.plain_pending = self.invited
            self.add_queries(self.value_queries(), first=True)
        # what the request holds back: one property a turn, each question sharing its turn with reading and searching
        try:
            self.find_out()
        except Exception as exc:    # asking must never cost the order: the shopping goes on with what is known
            self.record_note(f"asking stopped: {type(exc).__name__}: {exc}")
            self.awaiting_answer = False
        try:
            self.settle_early()
        except Exception as exc:    # the early cart must never cost the order
            self.record_note(f"early cart stopped: {type(exc).__name__}: {exc}")
        self.search_and_read(rounds=9)
        try:
            self.cart_stand_in()
        except Exception as exc:    # the stand-in must never cost the order
            self.record_note(f"stand-in stopped: {type(exc).__name__}: {exc}")
        try:
            self.hunt_best()
        except Exception as exc:    # looking for a better listing must never cost the order
            self.record_note(f"search for the best stopped: {type(exc).__name__}: {exc}")

    def choose_and_commit(self) -> tuple[str, str] | None:
        """Pick a listing and put it through the cart until one still fits, or none does."""
        counter = self.counter
        commits = 0
        choice: tuple[str, str] | None = None
        while not counter.over and counter.turns_remaining() >= 2 and commits < 5 and self.seconds_remaining() > 12:
            self.rank_by_priority(every=True)
            open_keys = [k for k in self.ranked() if k not in self.gone]
            weak = (bool(open_keys) and not self.search_asks and counter.turns_remaining() >= 14
                    and max(self.trust(k) for k in open_keys[:12]) < LIKELY_FIT)
            if (not open_keys or weak or not counter.cart and not any(self.keeps_request(k) for k in open_keys)) and self.rethink_searches():
                # nothing fits, nothing that fits can be relied on, or nothing for the first cart holds the request as
                # written: look with new searches first
                continue
            if not open_keys and not self.dug and counter.turns_remaining() > TURN_RESERVE + 6 and not self.time_is_short():
                # nothing fits after the new searches either: before settling for a near miss, look once more - the
                # shelf's own words, the titles of what fitted and was lost, and the near misses' own variants, each
                # with the values as they now stand
                self.dug = True
                self.fresh_rounds = max(self.fresh_rounds, 2)
                self.kin_done.clear()
                self.add_queries(self.lost_queries() + self.kin_queries() + self.scarce_queries() + self.variant_queries(), first=True)
                self.search_room = max(self.search_room, len(self.sent) + 48)
                self.open_room = max(self.open_room, len(self.counter.pages) + 40)
                self.search_and_read(rounds=3, enough=1)
                continue
            # (a listing within the limit that fits only in a loose reading does not stand against one that states
            # everything plainly and costs a little more: the plain one is what the request describes)
            loosely = bool(open_keys) and max(self.trust(k) for k in open_keys[:12]) < LIKELY_FIT
            if (not open_keys or loosely) and self.can_abstain():
                # (against fits that hold only in a loose reading, a listing just over the limit that states everything
                # more plainly than any of them is what the request describes, as is one stating it surely)
                best_open = max((self.trust(k) for k in open_keys[:12]), default=0.0)
                # (how likely it is that none of the loose fits within the limit fits after all: naming the nearest then
                # pays as far as that listing is trusted, against the best loose fit's own chance)
                none_fits = 1.0
                for k in open_keys[:12]:
                    none_fits *= 1.0 - self.trust(k)
                over = [k for k in self.only_over() if not open_keys or self.trust(k) >= SURE_FIT
                        or (self.trust(k) >= LIKELY_FIT - 1e-9 and self.trust(k) > best_open + 1e-9)
                        or none_fits * self.trust(k) > best_open + 1e-9]
                if not over and not self.over_noted:
                    self.over_noted = True
                    near = [k for k in self.verdicts if self.just_over(k)]
                    self.record_note(f"nothing fits within {self.budget} and no listing just over it does: moved_on={self.moved_on} brand={self.brand!r} over the limit: {len(near)} "
                              + "; ".join(f"{str((counter.pages.get(k[0]) or {}).get('title'))[:28]}|{self.variant_of(k).get('options')}|{(counter.stock.get(k) or {}).get('price')}|"
                                          f"{(self.verdicts[k].get('met') or [])}|{self.verdicts[k].get('kind_ok')}" for k in near[:8]))
                if over:
                    # nothing within the limit meets every requirement, and a listing that does costs a little more:
                    # the shopper is told that nothing fits, with that listing named as the nearest
                    choice = self.named_nearest = over[0]
                    title = str((counter.pages.get(choice[0]) or {}).get("title") or "")[:50]
                    self.record_note(f"nothing fits within {self.budget}: nearest {choice} {title!r} at {(counter.stock.get(choice) or {}).get('price')} "
                              f"turn={counter.turn} pages={len(counter.pages)}")
                    break
            if (not open_keys and not self.read_before_miss and counter.turns_remaining() > TURN_RESERVE + 2
                    and (not counter.cart or monotonic() - self.started < EPISODE_SECONDS + 90)
                    and counter.tool_available("add_to_cart")):
                # the first cart is held to the request as first written, and a near miss there fails it whatever is
                # ordered after: the rows the searches have shown and nobody read yet are read first (a listing that
                # states a value only in its options looks like nothing from its row)
                self.read_before_miss = True
                opens = self.listings_to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + counter.group_size * OPEN_BATCH)
                if opens:
                    counter.send_environment_turn(opens[: counter.group_size])
                    self.absorb_quietly()
                    self.mark_listings()
                    continue
            order = open_keys or [k for k in self.near_misses() if k not in self.gone]
            if not order:
                break
            if not counter.cart:
                # the first cart: what the shopper said matters more, then the request as written (stable sort)
                order.sort(key=lambda k: (not self.meets_chosen(k), not self.keeps_request(k)))
            held = self.stand_in_holds(order)
            if held is not None and held != order[0]:
                self.record_note(f"the listing in the cart still fits and the top is not clearly better: {held} kept over {order[0]}")
            choice = held or order[0]
            choice = self.whole_check(order, choice)
            commits += 1
            title = str((counter.pages.get(choice[0]) or {}).get("title") or "")[:50]
            fitting = f"{len(self.ranked())}: " + "; ".join(f"{str((counter.pages.get(k[0]) or {}).get('title') or '')[:28]}|{self.variant_of(k).get('options')}|{[self.plainness_level(k, r) for r in self.requirements]}|{(self.verdicts.get(k) or {}).get('by')}" for k in self.ranked()[:4])
            self.record_note(f"commit {commits}: {choice} {title!r} fits={self.listing_fits(choice)} turn={counter.turn} pages={len(counter.pages)} spent={self.reader.spent:.4f} calls={self.reader.calls} fitting: {fitting}")
            group = []
            ahead = None    # (no listing goes into the cart beside the choice: every other listing carted costs the order)
            if ahead is not None:
                # what a store does at the second cart falls on the listing put in then, as what it does at the first
                # falls on the first: one more listing goes in ahead of the choice, and out again with the stand-in
                self.went_ahead = True
                group.append(make_tool_call("add_to_cart", product_id=ahead[0], sku=ahead[1]))
                if counter.tool_available("inspect_stock"):
                    # (read again at once: what the store does to it has to be seen before the order, like any change)
                    group.append(make_tool_call("inspect_stock", product_id=ahead[0], sku=ahead[1]))
                self.record_note(f"ahead of the choice: {ahead} {str((counter.pages.get(ahead[0]) or {}).get('title') or '')[:40]!r}")
            if choice not in counter.cart:
                group.append(make_tool_call("add_to_cart", product_id=choice[0], sku=choice[1]))
            if counter.tool_available("inspect_stock"):
                group.append(make_tool_call("inspect_stock", product_id=choice[0], sku=choice[1]))
            if self.stand_in in counter.cart and self.stand_in != choice and counter.tool_available("remove_from_cart"):
                group.append(make_tool_call("remove_from_cart", product_id=self.stand_in[0], sku=self.stand_in[1]))
                counter.cart.remove(self.stand_in)
            if ahead is not None and counter.tool_available("remove_from_cart"):
                group.append(make_tool_call("remove_from_cart", product_id=ahead[0], sku=ahead[1]))
            counter.send_environment_turn(group)
            if ahead is not None and ahead in counter.cart and counter.tool_available("remove_from_cart"):
                counter.cart.remove(ahead)
            seen = counter.stock.get(choice) or {}
            price = seen.get("price")
            over_budget = self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9
            changed, asks = self.absorb_shopper_lines()
            if seen.get("in_stock") is False or over_budget:
                self.gone.add(choice)
                self.lost_titles.add(normalized_lower((counter.pages.get(choice[0]) or {}).get("title")))
                self.owed_reaction = True
                self.pending_line = ""     # the shopper's reaction to the change is answered by choosing again
                # (what was chosen is gone: room to look again, as when the shopper changes what they want)
                self.search_room = max(self.search_room, len(self.sent) + 32)
                self.open_room = max(self.open_room, len(counter.pages) + 40)
                if counter.turns_remaining() > TURN_RESERVE and (not self.time_is_short() or monotonic() - self.started < EPISODE_SECONDS + 60):
                    # the same article is often listed again, and it shares every fact the lost one was chosen for
                    # (a search and a read, no reader: worth its turns even past the time for looking further)
                    self.refill(choice, wide=not any(self.listing_clears(k, "firm") for k in self.ranked() if k not in self.gone))
                left = [k for k in self.ranked() if k not in self.gone]
                if len([k for k in left if self.listing_clears(k, "firm")]) < 2 and counter.turns_remaining() > TURN_RESERVE and not self.time_is_short():
                    # fewer than two firm listings left: another sell-out would leave only weak ones, so look further now
                    # (the same article is often listed again, and the shelf has more to show)
                    self.kin_done.clear()
                    self.add_queries(self.kin_queries() + self.shelf_queries(6), first=True)
                    self.search_and_read(rounds=3 if left else 5, enough=2 if left else 1)
                choice = None
                continue
            if changed:
                self.mark_listings()
                if not self.listing_fits(choice):
                    self.gone.add(choice)
                    if counter.turns_remaining() > TURN_RESERVE and not self.time_is_short():
                        # the values as they now stand are searched before anything else, and what those searches
                        # show is opened first
                        fresh = self.next_searches(counter.group_size)
                        if fresh:
                            counter.send_environment_turn(fresh)
                            self.absorb_quietly()
                        opens = self.listings_to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + 60)
                        if opens and not counter.over:
                            counter.send_environment_turn(opens[: counter.group_size])
                            self.absorb_quietly()
                        self.mark_listings()
                    if not [k for k in self.ranked() if k not in self.gone] and counter.turns_remaining() > TURN_RESERVE and not self.time_is_short():
                        # what the shopper changed is most often met by another variant of the same article, or by
                        # the article listed again: its own title, with the values as they now stand
                        self.refill(choice, wide=False)
                    if not [k for k in self.ranked() if k not in self.gone] and counter.turns_remaining() > TURN_RESERVE and not self.time_is_short():
                        self.search_and_read(rounds=4, enough=1)
                    choice = None
                    continue
            if asks and self.pending_line:
                self.answer(self.pending_line)
                self.pending_line = ""
                more_changed, _ = self.absorb_shopper_lines()
                if more_changed:
                    continue
            if counter.turn > int((counter.stock.get(choice) or {}).get("turn") or 0) and counter.tool_available("inspect_stock") and counter.turns_remaining() >= 3:
                # a turn went by since the choice was last read: the market may have moved, so read it again before paying
                counter.send_environment_turn([make_tool_call("inspect_stock", product_id=choice[0], sku=choice[1])])
                late_change, _ = self.absorb_shopper_lines()
                seen = counter.stock.get(choice) or {}
                if seen.get("in_stock") is False or (self.budget is not None and isinstance(seen.get("price"), float) and seen["price"] > self.budget + 1e-9):
                    self.gone.add(choice)
                    self.lost_titles.add(normalized_lower((counter.pages.get(choice[0]) or {}).get("title")))
                    self.owed_reaction = True
                    choice = None
                    continue
                if late_change:
                    # the shopper changed what they want while the choice waited: it must still fit
                    self.mark_listings()
                    if not self.listing_fits(choice):
                        self.gone.add(choice)
                        choice = None
                        continue
            if self.unanswered and counter.tool_available("message") and counter.turns_remaining() >= 3:
                self.reply_before_paying(choice)
                if self.absorb_shopper_lines()[0]:
                    self.mark_listings()
                    if not self.listing_fits(choice):
                        self.gone.add(choice)
                        choice = None
                        continue
            break
        return choice

    def place_final_order(self, choice: tuple[str, str] | None) -> list[dict]:
        """Name the nearest when nothing fits, then place the order."""
        counter = self.counter
        if choice is None and not counter.over and self.can_abstain() and not [k for k in self.ranked() if k not in self.gone]:
            over = self.only_over()
            if over:
                choice = self.named_nearest = over[0]
        if choice is None and counter.cart and not counter.over:
            alive = [k for k in counter.cart if k not in self.gone]
            choice = alive[-1] if alive else None
        if choice is None and not counter.over:
            # nothing left to order: the nearest listing still on sale is ordered rather than ending without an order
            # (it may be what was asked for in other words); where the store allows it, a listing that cannot even
            # be put in the cart is named as the closest instead
            self.mark_listings()
            nearest = self.closest()
            if nearest is not None and nearest not in counter.cart and counter.turns_remaining() >= 2:
                counter.send_environment_turn([make_tool_call("add_to_cart", product_id=nearest[0], sku=nearest[1])])
                self.absorb_quietly()
            if nearest is not None and (nearest in counter.cart or self.can_abstain()):
                choice = nearest
        for _ in range(3):
            if choice is None or counter.over or counter.turns_remaining() < 1:
                break
            if self.unanswered and counter.tool_available("message") and counter.turns_remaining() >= 2:
                self.reply_before_paying(choice)
            if choice == self.named_nearest:
                choice = self.nearest_now(choice)
                if choice is None:
                    break
            else:
                self.look_again(choice)
            if counter.heard:
                # the shopper spoke while the options were opened: the line is taken in before the order
                if self.absorb_shopper_lines()[0]:
                    self.mark_listings()
                    if choice == self.named_nearest:
                        choice = self.nearest_now(choice, read=False)
                        if choice is None:
                            break
                    elif not self.listing_fits(choice):
                        self.gone.add(choice)
                        choice = self.recart() or choice
            observations = counter.send_environment_turn([self.final_action(choice)])
            if counter.over:
                break
            # the order was refused (for instance the shopper has a line to read first): take the line in and try again
            changed, asks = self.absorb_shopper_lines()
            if asks and self.pending_line and counter.turns_remaining() >= 2:
                self.answer(self.pending_line)
                self.pending_line = ""
                changed = self.absorb_shopper_lines()[0] or changed
            if changed and counter.turns_remaining() >= 3:
                # what the shopper wants changed while the order waited: the choice must still fit
                self.mark_listings()
                if choice == self.named_nearest:
                    choice = self.nearest_now(choice, read=False)
                elif not self.listing_fits(choice):
                    self.gone.add(choice)
                    choice = self.recart()
            if not observations:
                break
        return counter.dialogue

    def execute_episode(self) -> list[dict]:
        self.open_episode()
        choice = self.choose_and_commit()
        return self.place_final_order(choice)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    errand = ShopOrder(problem_data)
    try:
        return errand.execute_episode()
    except Exception as exc:  # the dialogue so far is still returned for diagnosis, after a last order attempt
        errand.counter.dialogue.append({"role": "assistant", "content": f"agent error: {type(exc).__name__}: {exc}"})
        try:
            errand.last_resort()
        except Exception as late:
            errand.counter.dialogue.append({"role": "assistant", "content": f"agent error at the last order: {type(late).__name__}: {late}"})
        return errand.counter.dialogue


__all__ = ["agent_main"]
