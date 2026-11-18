"""Shopping agent for shopper situations: one shopper, one order.

The shopper states some requirements, keeps others until asked, and the market or the shopper may
change while the agent shops. The agent reads the request once, asks at most the one question the
request leaves open, searches the catalogue, reads the listings it might buy, puts its best match in
the cart, checks what the cart did to price and stock, follows any change, and orders.

A language model reads the request, the shopper's lines and the listing pages; the code keeps the
episode in order (turns, searches, the cart, stock checks, the final order).
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from os import getenv
from time import monotonic
from typing import Any

from src.agent.proxy_client import ProxyClient

# ----------------------------------------------------------------------------- settings
MODELS = {
    "chutes": ["zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE"],
    "openrouter": ["z-ai/glm-5.2", "moonshotai/kimi-k2.6"],
}
SPEND_CAP_USD = 0.05          # per episode; the order is still placed when the cap is reached
CALLS_PER_TURN = 16           # tool calls sent in one turn (at most what the task allows)
OPEN_BATCH = 5                # listings per compare call
OPEN_LIMIT = 250              # listings read in full over the episode
OPEN_AFTER_CHANGE = 100       # more listings that may be read once the shopper changes what they want (the rows found
                              # for the new values come after everything read for the old ones)
SEARCH_LIMIT = 80             # searches and shelf filters sent while looking (every one loads the store)
SHELF_FILTERS = 3             # shelf filters sent with a turn's searches while looking
PRICE_STEPS = (0.7, 0.4, 0.2, 0.9, 0.55, 0.3, 0.1)     # price ceilings, as shares of the budget, the shelf is filtered at
CART_TRIES = 5                # listings carted at most before the order
RELATED_TITLES = 6            # fitting listings whose other variants and listings are searched for
NO_MAKER = {"none", "no brand", "nobrand", "oem", "generic"}    # brand fields that name no maker
JUDGE_CARDS = 18              # listings shown to the reader in one call
JUDGE_LIMIT = 120             # listings the reader is asked about over the episode
EPISODE_SECONDS = 180.0       # after this the agent stops looking and orders what it has
TURN_RESERVE = 6              # turns kept for the cart, the stock checks and the order
QUESTIONS_OVER = 3            # questions allowed beyond one for each requirement the shopper keeps back
QUERY_LENGTH = 200            # a search is cut to this many characters
THINK_TOKENS = 3000           # room for a thinking pass before the tool call is written
MIN_TOKENS = 900              # every call: a provider may think before it writes the tool call, even when asked not to
NOT_A_PROPERTY = re.compile(r"categor|budget|price|\bstock|availab|\bsection\b|department|\bshelf\b|\bcatalog")     # terms of the purchase, not properties of the item
SHELF_NAME = re.compile(r"categor|\bsection\b|department|\bshelf\b|\bcatalog")    # what the store files the item under
PLAIN_QUESTION = "What will you use it for?"
# the quantity a shopper wants the most or least of is what ranks fitting items, never what they keep back
WANTED_MOST = re.compile(r"(?:the\s+)?(?:most|least|fewest|max(?:imum)?|(?:larg|bigg|great|long|high|small|short|low)est)\b", re.I)
WANTED_PHRASE = re.compile(r"\b(?:most|least|fewest|max(?:imum)?|min(?:imum)?|(?:larg|bigg|great|long|high|small|short|low)est|as (?:many|much))\b[^.;]*", re.I)
COLOUR_WORD = re.compile(r"\b(?:colou?r(?:ed)?|red|orange|yellow|green|blue|purple|violet|pink|brown|black|white|gr[ae]y|silver|gold(?:en)?"
                         r"|beige|ivory|navy|maroon|teal|turquoise|khaki|multicolou?r)\b", re.I)
# words many labels share that do not tell one property from another (of the catalogue's 3,522 fact labels: type in
# 722, size in 206, feature in 53, pack in 11)
GENERIC_LABEL_WORDS = {"type", "feature", "size", "pack"}
RARELY_KEPT = re.compile(r"\b(?:pack|packaging|units?|series|width|thickness|age|gender|megapixels?|dimensions?)\b")
# a label that lists several features at once: a question about it asks about several things
LIST_LABEL = re.compile(r"(?:features|benefits|concerns|functions|specifications|details)$")
MORE_IS_BETTER = re.compile(r"\b(?:most|max(?:imum)?|largest|biggest|greatest|highest|longest|heaviest|as many|as much|more)\b")
LESS_IS_BETTER = re.compile(r"\b(?:least|min(?:imum)?|smallest|lowest|shortest|fewest|lightest)\b")
AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]+)")       # a figure and the word written after it


def unit_word(word: str) -> str:
    """A unit in one form: lower case, without a plural s."""
    word = word.lower()
    return word[:-1] if len(word) > 2 and word.endswith("s") and not word.endswith("ss") else word


def amount_set(text: Any) -> set[tuple[str, float]]:
    """The amounts a text gives, each as (unit as written, figure). No unit is known in advance: which units measure
    the same thing, and how they convert, comes from the request's own spellings of a value, or from the reader."""
    return {(unit_word(unit), round(float(figure), 4)) for figure, unit in AMOUNT_RE.findall(lowered(text))}


def ranking_phrase(text: str) -> str:
    """The request's own words for what ranks the fitting items, for when the reader gives none: in the first sentence
    that has one, from its last superlative to the sentence's end. A spending limit or a word of weight (a thing that
    matters more than others) is not a ranking."""
    for sentence in re.split(r"(?<=[.!?;])\s+", str(text or "")):
        cleaned = re.sub(r"\b(?:at (?:most|least)|(?:the )?most (?:i|you|we)\b|(?:matters|counts) (?:the )?most|most (?:important|of all|to me)"
                         r"|the most for|max(?:imum)?\s+(?:budget|price|spend\w*|limit|is|of)\b)", " ", sentence, flags=re.I)
        found = list(WANTED_PHRASE.finditer(cleaned))
        if found:
            return " ".join(cleaned[found[-1].start():].split()).rstrip(".!?;")
    return ""


# a word named to say the item lacks it, and a value that is itself such a lack
NEGATED = re.compile(r"\b(?:no|non|zero|without|free\s+of)[\s-]+[a-z]+|\b0\s*%\s*[a-z]+|\b[a-z]+[\s-]+free\b")
NEGATING = re.compile(r"\b(?:free|no|non|zero|without)\b|0\s*%")
# fact labels about the sale or the maker, not about what the item is
NOT_A_LABEL = re.compile(r"brand|condition|model|warrant|ship|origin|country|stock|sku|seller|deliver|manufactur|made in|place of|date"
                         r"|expir|batch|registration|approval|licen|certif|serial|rating|review|price|discount|categor")

_proxy = ProxyClient(timeout=90, max_retries=2)


# ----------------------------------------------------------------------------- small helpers
def lowered(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def word_list(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", lowered(text))


def singular(word: str) -> str:
    """One form for a word and its plural (also for words written with -ie or -y)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 4 and word.endswith("ie"):
        return word[:-2] + "y"
    return word


def closed_up(text: Any) -> str:
    """Amounts written as listings write them: the number and its unit as one word (3 cm -> 3cm)."""
    return re.sub(r"(\d)\s+(?=[a-z]{1,4}\b)", r"\1", lowered(text))


def stems(text: Any) -> set[str]:
    return {singular(w) for w in word_list(closed_up(text))}


def label_key(text: Any) -> str:
    """A category label reduced to its words in singular form, for comparing two spellings of it."""
    return " ".join(singular(w) for w in word_list(str(text).replace("&", " and ")) if w != "and")


def plural(word: str) -> str:
    """A noun's plural as stores write it on a shelf's label (a word already ending in s is left as it is)."""
    m = re.match(r"([A-Za-z]+)(\W*)$", word)
    if not m or m.group(1).endswith("s"):
        return word
    stem, rest = m.group(1), m.group(2)
    if re.search(r"(?:x|ch|sh)$", stem):
        return f"{stem}es{rest}"
    if re.search(r"[^aeiou]y$", stem):
        return f"{stem[:-1]}ies{rest}"
    return f"{stem}s{rest}"


def line_words(text: Any) -> tuple[set[str], set[str]]:
    """A text's words (singular, one spelling of grey) and its figures, as compared with a line."""
    text = closed_up(text).replace("gray", "grey")
    return {singular(w) for w in word_list(text)}, set(re.findall(r"\d+(?:\.\d+)?", text))


def said_in(line: str, add: dict) -> bool:
    """The value a reading of the line adds is in the line itself - a figure of it, or a word of it or of one of its
    other spellings. A reading can bring in a value the line never names (an acknowledgement read as a change back to
    the first request); such a value is not taken."""
    words, figures = line_words(line)
    for form in [add.get("value"), *(add.get("written") or []), *(add.get("alike") or [])]:
        own_words, own_figures = line_words(form)
        if own_figures:
            if own_figures <= figures:
                return True
        elif any(w in words for w in own_words if len(w) > 2):
            return True
    return False


def drop_said_in(line: str, drop: str, adds: list[dict], old_values: list[str]) -> bool:
    """A requirement a reading withdraws is named in the line (its property or value), or replaced by a value the line
    gives for it: one under the same property name, or an amount in a unit of its old value (old_values:
    the values of the requirements the withdrawal names). A value for another property withdraws nothing."""
    words, figures = line_words(line)
    own_words, own_figures = line_words(drop.replace("=", " "))
    if any(w in words for w in own_words if len(w) > 2 and w != "type") or own_figures & figures:
        return True
    named = lowered(drop.split("=")[0])
    old_units = {unit for value in old_values for unit, _ in amount_set(value)}
    return any(lowered(a.get("property")) == named or old_units & {unit for unit, _ in amount_set(a.get("value"))} for a in adds)


def query_plan(item: str, requirements: list[dict], label: str = "") -> list[str]:
    """Catalogue searches for listings that state every required value, likeliest first. Measured on the store's
    search: every value with the item's last word reaches the most; every value alone, or each beside its
    property's name, reaches listings whose titles call the item something else; then the store's own label for
    the shelf, the item's full name, each value's other spellings, smaller sets of the values."""
    words = word_list(item)
    head = singular(words[-1]) if words else ""
    pairs = [(closed_up(r.get("value")), r) for r in requirements if closed_up(r.get("value"))]
    if not pairs:
        return [lowered(item)] if words else []
    values = [value for value, _ in pairs]
    every = " ".join(values)
    named = " ".join(f"{value} {' '.join(w for w in word_list(r.get('property')) if w not in word_list(value))}" for value, r in pairs)
    out = [f"{every} {head}", every if len(every.split()) > 1 else "", named]
    if label and label_key(label) != label_key(item):
        out.append(f"{every} {label}")
    out.append(f"{every} {item}")
    for n, (value, requirement) in enumerate(pairs):
        others = [closed_up(w) for w in [*(requirement.get("written") or []), *(requirement.get("alike") or [])]]
        for other in [w for w in others if w and w != value][:2]:
            out.append(" ".join(values[:n] + [other] + values[n + 1:] + [head]))
    if len(values) > 2:
        out += [f"{a} {b} {head}" for i, a in enumerate(values) for b in values[i + 1:]]
    out += [f"{value} {head}" for value in reversed(values)]
    return [q for q in dict.fromkeys(lowered(q)[:QUERY_LENGTH].strip() for q in out) if q]


def money(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def tool_spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


def strings(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def act(name: str, **args: Any) -> dict:
    return {"name": name, "args": {k: v for k, v in args.items() if v is not None}}


# ----------------------------------------------------------------------------- what the reader fills in
PLAN_TOOL = tool_spec("plan", "The shopper's request, written down for shopping.", {
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
    "brand": {"type": "string", "description": "A brand the shopper requires; empty when none."},
    "preferred_brand": {"type": "string", "description": "A maker the shopper would choose first but does not require (a first choice, one to take if available or if possible); empty when none."},
    "purposes": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                 "description": "Each use or recipient the request gives for the item, one entry each, in the shopper's words. Empty when it gives none."},
    "priority": {"type": "string", "description": "What the shopper says makes one fitting item better than another (for example the lowest price); empty when they say nothing."},
    "priority_units": {"type": "array", "maxItems": 6, "items": {"type": "object", "properties": {
        "unit": {"type": "string", "description": "A unit as listings write it, abbreviations included."},
        "size": {"type": "number", "description": "How many of the first unit one of this unit is (1 for the first)."}},
        "required": ["unit", "size"]},
        "description": "When the priority is an amount: every unit listings may write its figure in, the commonest first, each with its size in the first unit (for a volume ml 1 and l 1000; for a number of pieces pcs 1, pc 1 and pieces 1; for a warranty month 1 and year 12). Empty for a price, a resolution, a rating or anything else without a unit."},
    "queries": strings("Six short catalogue searches (two to five words) that could return the item: the item words with the required values, with a synonym or another spelling of a value where one exists."),
}, ["item", "requirements", "budget", "purposes", "queries"])

HEAR_TOOL = tool_spec("hear", "What the shopper's new line changes.", {
    "adds": {"type": "array", "description": "Each requirement of the item the line states that was not known before.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string"}, "value": {"type": "string"},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The same value written another way: another spelling of the word, or the same amount in another unit. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value (a synonym; not a part or a wider kind of it). Empty when there is none."}},
        "required": ["property", "value"]}},
    "drops": strings("Properties whose earlier requirement the line withdraws or replaces."),
    "budget": {"type": "number", "description": "A new spending limit the line states; 0 when it states none."},
    "category": {"type": "string", "description": "The store category the line names for the item, as written; empty when it names none."},
    "priority": {"type": "string", "description": "What the line says matters most between fitting items; empty when nothing."},
    "asks": {"type": "boolean", "description": "True when the line asks the assistant something or expects an answer."},
    "about_change": {"type": "boolean", "description": "True when the line only reacts to the stock or the price of the chosen item having changed."},
    "queries": strings("Up to six short catalogue searches that use what the line added (try a synonym or another spelling of the value too); empty when it added nothing."),
}, ["adds", "drops", "budget", "asks", "queries"])

JUDGE_TOOL = tool_spec("judge", "Which listings state every requirement.", {
    "marks": strings("One short entry for every listing shown, in the order shown: the listing's number, then K when it is the kind of thing asked for or X when it is not, "
                     "then one letter per requirement in order - Y when the listing's own words state that value, N when they do not. Example for two requirements: '3 K YN'."),
    "proof": {"type": "array", "description": "Only for listings marked K with every requirement Y: where each requirement is stated.", "items": {
        "type": "object", "properties": {
            "n": {"type": "integer", "description": "The listing's number."},
            "where": strings("For each requirement in order: title, options, or the fact's label exactly as written."),
            "words": strings("For each requirement in order: the listing's exact words that state it."),
            "rank_value": {"type": "number", "description": "The listing's number for the shopper's priority when one is given, else 0."}},
        "required": ["n", "where", "words"]}},
    "queries": strings("When no listing meets every requirement: up to five other short searches worth trying; otherwise empty."),
}, ["marks"])

WORDS_TOOL = tool_spec("words", "Other words sellers use for a value.", {
    "words": {"type": "array", "items": {"type": "string"}, "description": "Up to six, the most usual first; empty when there is none."}},
    ["words"])

RANK_TOOL = tool_spec("rank", "Each listing's figure for the shopper's priority.", {
    "values": {"type": "array", "items": {"type": "string"}, "description": "One entry per listing, in order: '<number> <figure>' with the figure as a plain number in one unit for every listing, or '<number> none' when the listing's words state none."},
    "better": {"type": "string", "enum": ["higher", "lower"], "description": "Whether a higher or a lower figure is better for this priority."}},
    ["values", "better"])

SEARCH_TOOL = tool_spec("search_plan", "New catalogue searches.", {
    "queries": strings("Six new searches of two to five words each, different from those tried."),
}, ["queries"])

REPLY_TOOL = tool_spec("reply", "A short answer to the shopper.", {
    "text": {"type": "string", "description": "One or two plain sentences that answer the shopper's line with the facts given. A statement, never a question."}},
    ["text"])

FEATURE = {"type": "object", "properties": {
    "property": {"type": "string", "description": "A plain name for what the feature is (colour, material, brand, size, voltage...)."},
    "value": {"type": "string", "description": "The feature's value, copied from the request."}},
    "required": ["property", "value"]}
ASK_TOOL = tool_spec("ask", "What the opening question has to be about.", {
    "kind": {"type": "string", "enum": ["pair", "need"]},
    "doubt": {"type": "string", "description": "For a pair: the request's own words voicing the doubt or the hedge, copied exactly. Empty for a need."},
    "pair": {"type": "array", "items": FEATURE, "maxItems": 2, "description": "For a pair: the two wished features, in the request's order. Empty for a need."},
    "properties": strings("The properties the kept-back requirement is most likely about, most likely first: four different ones, each named like a listing's fact label, without a value."),
    "purposes": strings("Each separate reason for buying the request gives (a use, a recipient, an occasion), in its own words."),
}, ["kind", "doubt", "pair", "properties", "purposes"])

# plain properties a shopper may keep back that sellers write in titles and options rather than as fact labels: the
# two kept back most often come first; the reader names others from the amounts the shelf's listings state
FIGURES_FIRST = ("piece count", "material")
FIGURE_WORDS = {word for figure in FIGURES_FIRST for word in figure.split()} | {"capacity"}
LABELS_TOOL = tool_spec("labels", "The likeliest kept-back property among a shelf's fact labels.", {
    "ranked": strings("Up to four of the labels shown, copied exactly, most likely first."),
    "figure": {"type": "string", "description": "Of the plain properties sellers write in titles and options rather than as fact labels (the material, the piece count, or what one of the shelf's amounts measures), the one the request leaves unstated that items of this kind differ by most, named as a shopper names it; none when the request states them all."},
}, ["ranked", "figure"])

PLAN_GUIDE = """You prepare a purchase for a shopping assistant. Read the shopper's request and fill the plan tool from its words only.
- item: the words the shopper uses for the kind of thing, copied completely and exactly as written, with a leading word such as 'other' and any '&' or '/' kept (the store files its listings under such names); do not shorten them or replace them with a broader or narrower kind.
- requirements: every detail the request gives about the item itself is a requirement: its material, its colour, an amount with its unit (how much it holds, how big, heavy or long it is, how many pieces, how strong), its pattern, shape, style, kind or type, a theme it carries, the make or thing the item must fit, and the like. Why it is bought - a use, an occasion, a recipient, the conditions it must cope with - is a purpose, not a requirement. A request for one kind of thing nearly always carries such a detail: read the request again before leaving the list empty. Name each property the way the shopper names it; when the request gives only the value, use the plain name of that kind of detail. Give the value in the fewest words: the word or amount itself (an amount with its unit in letters), without the property's name and without the kind of thing it belongs to. One phrase of the request gives one requirement; do not split it into two. The budget, the store category and the stock status are terms of the purchase, not requirements; neither is why the shopper is buying, which comes in a sentence of its own.
- purposes: every use or recipient the request gives (the job it should do, whom it goes to); each can carry a need the shopper has not stated yet.
- written: the same value written another way - another spelling of the word (grey: gray) or the same amount in another unit or wording (1 kg: 1000g, 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- queries: searches as a shopper types them, two to five words, built from the item words and the required values. Write an amount the way listings do (the number joined to its unit)."""

HEAR_GUIDE = """You follow a shopper's conversation with a shopping assistant. Given what is already known and the shopper's new line, fill the hear tool with what the line changes.
- adds: a requirement of the item the line states (a property with its value), the property named in the shopper's words and the value in the fewest words (the word or amount itself). A purpose alone is not a property; the property the purpose calls for is. Terms of the purchase (the budget, the store category, the stock status) are not properties. written: the same value written another way (another spelling of the word; the same amount in another unit: 1 kg is 1000g or 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- drops: a property whose earlier requirement the line replaces or withdraws.
- A new line can hold several sentences or lines: read them in order; about the same property, a later one overrides an earlier one.
- A line that puts one value in place of another, withdraws one, or adds one is a change of what the shopper wants, even when it is worded as a question or a confirmation (for example ending in 'right?'): drop the old property and add the new value.
- budget: only when the line names a new limit.
- asks: true when the line puts a question to the assistant or expects an answer; a remark that needs no answer is false.
- about_change: true when the line reacts to the chosen item's stock or price having just changed (the assistant answers that by choosing again). A line can do that and also change a requirement: then fill adds and drops too.
- queries: searches of two to five words that combine the item words with the value just added."""

JUDGE_GUIDE = """You check store listings against a shopper's requirements. For every listing shown, say whether it is the kind of thing asked for and, for each requirement in order, whether the listing's own words state that value.
- A requirement is met when the listing's own words state the value: the same word, a plain synonym or another spelling of it, or the same amount written in another unit.
- Where to look: a colour or a size is what the options say, because the options describe the exact variant sold (when the options name another colour or amount, the requirement is unmet whatever the title lists); a material or an amount is stated by the title, the options or a fact; any other property the store keeps as a fact of its own is stated by the fact whose label names that property.
- When a listing shows 'the fact named for requirement N', that fact decides requirement N: met when it states the value, unmet when it states something else - unless the options state the value.
- A fact that states the value settles it even when the title's words suggest otherwise. A value that is absent is unmet; do not assume.
- An amount counts when it is one of several measures written for the variant (a size written as 40*30*5cm states 5 cm; a range such as 4-6cm includes 5 cm). An amount filed under a fact for a different quantity does not state it.
- marks: one entry per listing, in order, as '<number> <K or X> <Y or N for each requirement>'.
- proof: only for the listings that are K with every requirement Y - the place (title, options, or the fact's label exactly as written) and the listing's exact words that state each value; rank_value is the listing's number for the shopper's priority when one is given.
Mark every listing, in order."""

WORDS_GUIDE = """A shopper wants a certain value for a property of an item. List the other words or spellings that sellers use in listing titles and variant names for that same value: synonyms and other spellings of a word, or the same amount in other units and notations. Only words that mean the same value; not a part, a shade or a wider kind of it."""

RANK_GUIDE = """A shopper said what makes one fitting item better than another. For each listing, read from its own words (the options first, then its facts, then its title) the figure that this priority is about, and write it as a plain number in the same unit for every listing (convert where needed). Write none when the listing states no such figure; do not guess."""

SEARCH_GUIDE = """A shopping assistant has not yet found a store listing that meets every requirement of the shopper. Given the request, the requirements, the searches already tried and the titles of the listings seen, write new short catalogue searches that are likely to reach a listing meeting every requirement: combine the required values with each other and with the words such listings carry in their titles, use other words sellers use for a value, and try the values without the item words. Do not repeat a search already tried."""

ASK_GUIDE = """You plan the opening question a shopping assistant puts to a shopper, from the shopper's request, and fill the ask tool.

The shopper keeps back one requirement (sometimes one for each separate reason they give for buying) and reveals it only to a question that names exactly what it is about; a general question, a yes-or-no question or one about several things reveals nothing. Decide which kind of request this is.

pair: the request names two particular features it would like the item to have, each a property with a value (a colour, a material, a size, a type, a brand...), and is not sure one product offers both. The doubt is said outright (it wonders, questions, doubts, is unsure or skeptical, or only hopes the two come together, perhaps in a sentence of its own), or a second wish follows a first one with a hedge (too, also or as well, with if possible, if there is any way, ideally, or nice to have). What is kept back is which of the two the shopper would rather keep.
- The two features are the ones named in the doubt's own sentence; when that sentence names fewer than two, add the one or two named in the sentence just before it.
- Never one of the two: the budget or a price, the stock status, the shelf, a reason for buying, the most or least of something, or a requirement stated earlier as a plain must.
- Give each with the request's own value and a plain name for its property; a name the shopper wants it to carry, come from or simply be is its brand.
- It is no pair when the only wish is one maker or option they would prefer, or when the two things are reasons for buying.

need: every other request. What is kept back is the value of one property of the item that the request never mentions.

For either kind, list the properties the kept-back requirement is most likely about, most likely first, four different ones:
- leave out every property the request gives a value for (a colour word settles the colour, a figure in grams the weight), and the price, stock, shelf, the brand, and the quantity they want the most or least of;
- colour is the property kept back most often: put it first whenever the request names no colour;
- next comes the property sellers use to sort this kind of item into kinds, named '<kind> type' the way its shelf would label it (blade type for knives, lens type for glasses, never 'type' alone); then the material, or the material of the part that matters most (a seat's cover, a tool's head); then the figure its purpose turns on (capacity or volume, weight, size or length, voltage or plug type, piece count);
- name each the way a listing's fact label would, in one to three plain words, never with a value.

purposes: each separate reason for buying the request gives, in its own words."""

LABELS_GUIDE = """A shopper keeps back the value of one property of the item they want and reveals it only to a question naming that property. The properties already asked about revealed nothing. You see the request and the fact labels this shelf's listings carry, each with a few of the values found under it. Choose the labels the kept-back property is most likely to be, most likely first: a property that sorts this kind of item into kinds, whose values differ from listing to listing, and that the request leaves unmentioned. Skip labels about the condition, the maker or model, the warranty, shipping or origin, and anything the request already settles. Among fact labels, the ones kept back most often name the kind or type of the item (a '... type' label, the plug type), the pattern on it, or what one part is made of; labels about packing, units, series, a width or thickness, or the age or gender it suits rarely are. Many kept-back properties are not fact labels at all but plain properties sellers write in titles and options: the material, the piece count, or what one of the amounts the shelf's listings state measures (shown by unit) - name the one this kind of item differs by most, as a shopper would name it, unless the request states it or wants the most or least of it."""

REPLY_GUIDE = """You are a shopping assistant answering the shopper's lines; each one changed what the shopper wants. Answer each line in turn, in plain sentences, as statements: confirm the change in the shopper's own terms - yes, now the new value in place of the old one (or the added or dropped requirement) - and that the rest stays as it was. Name no item and no price, and do not list the shopper's other requirements. Do not ask anything and do not end with a question."""


# ----------------------------------------------------------------------------- inference
class Reader:
    """Language-model calls with a fallback model and a spending guard."""

    def __init__(self) -> None:
        provider = getenv("INFERENCE_PROVIDER", "chutes")
        self.models = MODELS.get(provider, MODELS["chutes"])
        self.spent = 0.0
        self.calls = 0

    def out_of_budget(self) -> bool:
        return self.spent >= SPEND_CAP_USD

    def fill(self, guide: str, text: str, tool: dict, max_tokens: int, second: bool = False, think: bool = False) -> dict | None:
        name = tool["function"]["name"]
        for model in (self.models[1:] if second else self.models):
            body = {"model": model, "temperature": 0, "max_tokens": max(max_tokens, MIN_TOKENS) + (THINK_TOKENS if think else 0),
                    "messages": [{"role": "system", "content": guide}, {"role": "user", "content": text}],
                    "tools": [tool], "tool_choice": {"type": "function", "function": {"name": name}}}
            # a thinking pass spends the completion budget before the tool call is written
            body["reasoning"] = {"enabled": think} if re.search(r"glm|kimi", model, re.I) else {"effort": "low"}
            reply = _proxy.post("/inference/chat/completions", json_data=body)
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
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _closed(raw: str) -> dict | None:
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


# ----------------------------------------------------------------------------- the store session
class Counter:
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
        self.shelf: dict[tuple[str, str], str] = {}      # rows a category filter returned -> the label that returned them
        self.pages: dict[str, dict] = {}                 # product_id -> page from view / compare
        self.stock: dict[tuple[str, str], dict] = {}     # (product_id, sku) -> latest price / stock seen
        self.cart: list[tuple[str, str]] = []
        self.heard: list[str] = []                       # shopper lines not yet handled

    def turns_left(self) -> int:
        return self.turn_limit - self.turn

    def has(self, tool: str) -> bool:
        return tool in self.tools

    def send(self, actions: list[dict]) -> list[dict]:
        """One turn: the actions in order. Returns one observation per action sent."""
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
            for row in observation.get("results") or []:
                if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                    key = (str(row["product_id"]), str(row["sku"]))
                    if key not in self.rows:
                        self.rows[key] = row
                        self.row_order.append(key)
                    if name == "filter" and action["args"].get("category"):
                        self.shelf.setdefault(key, str(action["args"]["category"]))
                    self._note(key[0], key[1], row.get("price"), row.get("in_stock"))
        elif name in ("view", "compare"):
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
                if key not in self.cart:
                    self.cart.append(key)
                self._note(key[0], key[1], added.get("price"), added.get("in_stock"))


# ----------------------------------------------------------------------------- the shopper's order
class Errand:
    def __init__(self, problem: dict) -> None:
        self.counter = Counter(problem)
        self.reader = Reader()
        self.started = monotonic()
        self.item = ""
        self.category = ""
        self.requirements: list[dict] = []     # {"property", "value"}
        self.budget: float | None = None
        self.brand = ""
        self.listed: list[tuple[str, str]] = []
        self.priority = ""
        self.priority_units: dict[str, float] = {}     # the units the reader says the priority's figure is written in, sized
        self.queries: list[str] = []
        self.sent: set[str] = set()            # searches and filters already sent
        self.vocabulary_used: set[str] = set()   # shelf words already searched with the values
        self.questions_asked = 0
        self.verdicts: dict[tuple[str, str], dict] = {}
        self.judged_under = ""
        self.judged_count = 0
        self.search_room = SEARCH_LIMIT        # searches allowed so far; a change by the shopper opens more
        self.open_room = OPEN_LIMIT            # listings that may be read so far
        self.judge_room = JUDGE_LIMIT          # listings the reader may be asked about so far
        self.gone: set[tuple[str, str]] = set()
        self.facts_for_reply: list[str] = []
        self.pending_line = ""
        self.owed_reaction = False             # a market change was just seen: the shopper's next question is about it
        self.reader_asked = False
        self.widened = False
        self.filing_cache: tuple[int, dict] = (-1, {})   # (pages read, the shelf's fact labels with their values)
        self.awaiting_answer = False           # a question was just put to the shopper
        self.lenient = False                   # nothing fits by the firm rules: a value stated anywhere on the page counts
        self.shelf_cache: tuple[int, str] = (-1, "")
        self.related: set[str] = set()         # titles whose other variants and listings were searched for
        self.fresh_rows = 0                    # rows shown from this index on were found for the requirements in force
        self.ask_props: list[str] = []         # properties the kept-back requirement may be about, likeliest first
        self.asked_props: list[str] = []       # properties already asked about
        self.needs_expected = 1                # kept-back requirements to draw out: one for each reason for buying
        self.told = 0                          # requirements the shopper has revealed in answer to a question
        self.labels_ranked = False             # the shelf's fact labels were read for the next property to ask about
        self.told_category = ""                # the catalog category the shopper names in a line
        self.budget_cut = False                # the shopper lowered the budget during the task
        self.delay_first = False               # the first question waits for the shelf's listings
        self.fit_seen = False                  # a listing that fit every requirement was carted
        self.preferred_brand = ""              # a maker the shopper would take first, without requiring it
        self.told_props: list[str] = []        # properties of the requirements questions drew out
        self.reasked: set[str] = set()         # properties asked again after an answer about another one
        self.pair_fallback = ""                # asked when the answer to the pair question names nothing
        self.set_aside: list[dict] = []        # requirements read as the wishes of a doubted pair, until the answer
        self.pending_filters: list[dict] = []  # store filters to send with the next searches
        self.search_asks = 0                   # times the reader wrote new searches because nothing fitted
        self.rank_asks = 0
        self.better = ""                       # the reader's word on the priority: a higher or a lower figure is better
        self.lost_titles: set[str] = set()     # titles of chosen listings that went out of stock or over the budget
        self.unanswered: list[str] = []        # lines that changed what the shopper wants, not yet replied to
        self.changed_values: dict[str, tuple[list[str], list[str]]] = {}   # line -> (values it brought in, values it took out)

    def note(self, text: str) -> None:
        self.counter.dialogue.append({"role": "assistant", "content": f"note: {text}"})

    def late(self) -> bool:
        return monotonic() - self.started > EPISODE_SECONDS

    def can_think(self) -> bool:
        return not self.reader.out_of_budget()

    # ---- the request
    def read_request(self) -> str:
        request = self.counter.request
        plan = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 700, think=True) or {}
        if not [r for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]:
            # a request without any detail of the item is rare: have the other model read it too
            again = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 700, second=True) or {}
            if again.get("requirements"):
                plan = {**plan, **{k: v for k, v in again.items() if v}} if plan else again
        self.item = str(plan.get("item") or "").strip()
        self.category = str(plan.get("category") or "").strip()
        self.requirements = [self.requirement(r) for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        self.requirements = [dict(r, stated=True) for r in self.requirements if not NOT_A_PROPERTY.search(lowered(f"{r['property']} {r['value']}"))]
        self.budget = money(plan.get("budget"))
        if self.budget is None:
            amounts = [float(m.replace(",", "")) for m in re.findall(r"(\d[\d,]*\.\d{2})", request)]
            self.budget = max(amounts) if amounts else None
        self.brand = str(plan.get("brand") or "").strip()
        self.preferred_brand = str(plan.get("preferred_brand") or "").strip()
        if lowered(self.preferred_brand) == lowered(self.brand):
            self.preferred_brand = ""
        self.listed = [(m.group(1), m.group(2)) for m in re.finditer(r"(\d{6,})::([\w-]+)", request)]
        self.priority = str(plan.get("priority") or "").strip() or ranking_phrase(request)
        self.priority_units = self.unit_sizes(plan.get("priority_units")) if str(plan.get("priority") or "").strip() else {}
        self.add_queries(list(plan.get("queries") or []))
        purposes = [" ".join(str(w).split()) for w in plan.get("purposes") or [] if str(w).strip()][:3]
        self.note(f"plan item={self.item!r} requirements={self.requirements} budget={self.budget} purposes={purposes} priority={self.priority!r}")
        if not self.planned() and not self.queries:
            self.add_queries([" ".join(word_list(request)[:5])])
        return self.opening(request, purposes)

    def opening(self, request: str, purposes: list[str]) -> str:
        """The first question, from a reading of what the request keeps back: which of two doubted wishes counts
        for more, or the property the kept-back requirement is most likely about."""
        ask = self.reader.fill(ASK_GUIDE, f"The shopper's request:\n{request}", ASK_TOOL, 700) or {}
        if not self.sound(ask, request):
            # no reading, or a pair whose sides are not two features of the item named where the doubt is voiced:
            # the other model reads it, and its reading is taken when it holds up better
            again = self.reader.fill(ASK_GUIDE, f"The shopper's request:\n{request}", ASK_TOOL, 700, second=True) or {}
            ask = self.better_reading(ask, again, request)
        said = [" ".join(str(w).split()) for w in ask.get("purposes") or [] if str(w).strip()] if isinstance(ask.get("purposes"), list) else []
        reasons = said or purposes
        reasons = [p for i, p in enumerate(reasons) if not any(i != j and self.within(p, q) and (not self.within(q, p) or j < i)
                                                              for j, q in enumerate(reasons))]
        self.needs_expected = max(1, min(2, len(reasons)))
        self.ask_props = self.properties_of(ask, request)
        pair = self.wished_pair(ask) if ask.get("kind") == "pair" else None
        self.note(f"ask kind={ask.get('kind')} pair={pair} properties={self.ask_props} reasons={reasons}")
        if pair:
            self.preferred_brand = ""           # a maker among doubted wishes is one side of the pair, not a first choice
        elif self.preferred_brand:
            # the first pick is that maker's listing when one keeps to the request: look for it from the start
            self.add_queries([f"{self.preferred_brand} {self.item}", self.preferred_brand])
            label = self.category or self.item
            if label:
                self.pending_filters.append(act("filter", category=label, brand=self.preferred_brand, max_price=self.budget, in_stock=True, k=10))
        if pair:
            # the shopper wants one of two things and says which when asked: that is the one open point; until
            # then neither wish is a firm requirement
            wished = lowered(" ".join(value for _, value in pair))
            self.set_aside = [r for r in self.requirements if lowered(r["value"]) in wished]
            self.requirements = [r for r in self.requirements if lowered(r["value"]) not in wished]
            self.pair_fallback = "need"
            self.needs_expected = 1
            return "Which matters more to you: %s or %s?" % (self.named(*pair[0]), self.named(*pair[1]))
        if self.ask_props and self.ask_props[0] != "colour":
            # with the colour settled, the shelf's own fact labels guess the kept-back property better than the request
            # does: the first question waits for the first listings read
            self.delay_first = True
            return ""
        return self.need_question()

    def sound(self, ask: dict, request: str) -> bool:
        """A need reading, or a pair of two features of the item both named where the doubt is voiced."""
        if ask.get("kind") == "need":
            return True
        return ask.get("kind") == "pair" and bool(self.wished_pair(ask)) and self.grounded(ask, request)

    def better_reading(self, first: dict, second: dict, request: str) -> dict:
        if self.sound(second, request):
            return second
        # neither holds up: a usable pair is still worth its question (a request with doubted wishes tells nothing
        # to a question about anything else), and then the property questions follow
        for ask in (first, second):
            if ask.get("kind") == "pair" and self.wished_pair(ask):
                return ask
        return first if first.get("kind") in ("pair", "need") else second

    @staticmethod
    def grounded(ask: dict, request: str) -> bool:
        """Both values of the pair stand in the sentence that voices the doubt, or in the sentence before it."""
        doubt = lowered(" ".join(str(ask.get("doubt") or "").split()))
        sentences = [lowered(x) for x in re.split(r"(?<=[.!?])\s+", " ".join(request.split())) if x.strip()]
        where = next((i for i, x in enumerate(sentences) if doubt and (doubt in x or x in doubt)), None)
        if where is None:
            return False
        window = " ".join(sentences[max(0, where - 1): where + 1])
        values = [lowered(str((w or {}).get("value") or "")) for w in ask.get("pair") or [] if isinstance(w, dict)]
        return len(values) == 2 and all(v and v in window for v in values)

    @staticmethod
    def named(prop: str, value: str) -> str:
        """'the <property> <value>', without the property twice when the value already holds it."""
        if lowered(prop) in lowered(value):
            return value if lowered(value).startswith("the ") else f"the {value}"
        return f"the {prop} {value}"

    def wished_pair(self, ask: dict) -> list[tuple[str, str]] | None:
        """The two wished features of a pair reading, or None when they are not two features of the item."""
        pair = ask.get("pair") if isinstance(ask.get("pair"), list) else []
        out = []
        for wish in pair[:2]:
            prop = " ".join(str((wish or {}).get("property") or "").split()).strip(" .,;:") if isinstance(wish, dict) else ""
            value = " ".join(str((wish or {}).get("value") or "").split()).strip(" .,;:") if isinstance(wish, dict) else ""
            if not prop or not value or NOT_A_PROPERTY.search(lowered(prop)) or (lowered(prop) not in ("brand", "make") and WANTED_MOST.match(value)):
                return None
            out.append((lowered(prop), value))
        if len(out) != 2 or lowered(out[0][1]) == lowered(out[1][1]):
            return None
        return out

    def properties_of(self, ask: dict, request: str) -> list[str]:
        """The reading's properties to ask about, likeliest first: one name each, nothing the request settles,
        and the colour first when the request names no colour."""
        props = ask.get("properties") if isinstance(ask.get("properties"), list) else []
        wanted = lowered(" ".join(m.group(0) for m in WANTED_PHRASE.finditer(request)))
        out: list[str] = []
        for prop in props:
            prop = lowered(" ".join(str(prop or "").replace("_", " ").split())).strip(" .,;:")
            prop = "colour" if re.fullmatch(r"colou?rs?|colou?r family", prop) else prop
            if not prop or len(prop.split()) > 3 or NOT_A_PROPERTY.search(prop) or re.fullmatch(r"type|kind|item|product|purpose|use|brand|maker|manufacturer", prop):
                continue
            if any(word in wanted for word in prop.split() if len(word) >= 4 and word != "type") or prop in out or self.property_settled(prop):
                continue
            out.append(prop)
        named_colour = COLOUR_WORD.search(request)
        out = [p for p in out if p != "colour"]
        if not named_colour:
            out.insert(0, "colour")
        return out[:5]

    def property_settled(self, prop: str) -> bool:
        """A requirement already stands for this property (one name holding every word of the other)."""
        words = self.name_words(prop) - {"type", "kind"}
        for r in self.requirements:
            have = self.name_words(r["property"]) - {"type", "kind"}
            if words and have and (words <= have or have <= words):
                return True
        return False

    def need_question(self) -> str:
        """The question about the next property: the first one also asks about its use, which some
        shoppers answer whatever the property named."""
        prop = self.next_property()
        if not prop:
            return "" if self.asked_props else PLAIN_QUESTION
        first = not self.asked_props and not self.told_props and not self.reasked
        self.asked_props.append(prop)
        return f"Which {prop} does it need to have{', given what you will use it for' if first else ''}?"

    def next_property(self) -> str:
        if (self.asked_props or self.delay_first) and not self.labels_ranked and self.can_think():
            self.rank_labels()
        for prop in self.ask_props:
            if prop not in self.asked_props and not self.property_settled(prop):
                return prop
        return ""

    def shelf_facts(self) -> dict[str, list[str]]:
        """The fact labels of the shelf's listings read so far, each with the values filed under it."""
        pages = [p for p in self.counter.pages.values() if self.in_category(p)] or list(self.counter.pages.values())
        found: dict[str, list[str]] = {}
        for page in pages:
            for label, value in (page.get("facts") or {}).items():
                name = lowered(str(label).replace("_", " "))
                text = " ".join(str(value).split())[:40]
                if not name or NOT_A_LABEL.search(name) or LIST_LABEL.search(name) or not text:
                    continue
                found.setdefault(name, [])
                if text not in found[name]:
                    found[name].append(text)
        # a label filed by one listing only, or with a single value on every listing, sorts nothing into kinds
        return {name: values for name, values in found.items() if len(values) >= 2}

    def shelf_amounts(self) -> list[tuple[str, int]]:
        """The units the shelf's listings state amounts in (titles, options, fact values), each with the number of
        listings stating one, where two sizes or more show over two listings or more: what the shelf differs by."""
        shelf = [p for p in self.counter.pages.values() if self.in_category(p)]
        sizes: dict[str, set[float]] = {}
        listings: dict[str, set[str]] = {}
        for page in (shelf if len(shelf) >= 5 else list(self.counter.pages.values()))[:150]:
            texts = [str(page.get("title") or "")] + [str(v.get("options") or "") for v in page.get("variants") or []]
            texts += [str(value) for value in (page.get("facts") or {}).values()]
            for unit, figure in amount_set(" ".join(texts)):
                sizes.setdefault(unit, set()).add(figure)
                listings.setdefault(unit, set()).add(str(page.get("product_id")))
        return sorted(((u, len(listings[u])) for u in sizes if len(sizes[u]) > 1 and len(listings[u]) > 1), key=lambda x: -x[1])

    def label_core(self, name: str) -> set[str]:
        """The words of a property name that tell it from other properties (units such as ml or g say nothing)."""
        return {w for w in self.name_words(name) if len(w) > 2} - GENERIC_LABEL_WORDS

    def rank_labels(self) -> None:
        """Put the shelf's own fact labels that the reader finds likeliest at the head of the properties left."""
        self.labels_ranked = True
        facts = self.shelf_facts()
        facts = {name: values for name, values in facts.items() if name not in self.asked_props and not self.property_settled(name)}
        shown = sorted(facts.items(), key=lambda kv: -len(kv[1]))[:30]
        text = (f"The shopper's request:\n{self.counter.request}\n\nAlready asked about: {', '.join(self.asked_props)}\n\n"
                "Fact labels on this shelf's listings, with values found under them:\n"
                + "\n".join(f"- {name}: {'; '.join(values[:5])}" for name, values in shown))
        wanted = lowered(" ".join(m.group(0) for m in WANTED_PHRASE.finditer(self.counter.request)))
        open_figures = [f for f in FIGURES_FIRST if f not in self.asked_props and not self.property_settled(f)
                        and not any(word in wanted for word in f.split() if len(word) >= 4)]
        amounts = self.shelf_amounts()
        text += ("\n\nAmounts the shelf's listings state, by unit (listings stating one): "
                 + (", ".join(f"{unit} ({n})" for unit, n in amounts[:10]) or "none"))
        answer = self.reader.fill(LABELS_GUIDE, text, LABELS_TOOL, 400) or {}
        ranked = [lowered(" ".join(str(r).replace("_", " ").split())) for r in answer.get("ranked") or []] if isinstance(answer.get("ranked"), list) else []
        ranked = [r for r in ranked if r in facts]
        # a shelf label that matches a property the request itself made likely is the surer guess: those come first,
        # in the order the request's reading gave them
        agreed: list[str] = []
        for prop in self.ask_props:
            words = self.label_core(prop)
            if prop in self.asked_props or prop == "colour" or not words or words <= FIGURE_WORDS:
                continue                # (a plain property such as the material has a place of its own)
            # the label sharing the most of its words (the first one shown on a tie: the most common)
            scored = [(len(words & self.label_core(name)) / len(words | self.label_core(name)), -i, name)
                      for i, (name, _) in enumerate(shown) if name not in agreed and words & self.label_core(name)]
            if scored:
                agreed.append(max(scored)[2])
        ranked = agreed + [r for r in ranked if r not in agreed]
        # labels about the packing, a measurement or who it suits are seldom what a shopper keeps back: last
        ranked = [r for r in ranked if not RARELY_KEPT.search(r)] + [r for r in ranked if RARELY_KEPT.search(r)]
        # two labels for one property ('volume' and 'volume ml') ask the same thing twice: the first one stands
        kept: list[str] = []
        for label in ranked:
            words = self.label_core(label)
            if not any(words & self.label_core(k) for k in kept + self.asked_props):
                kept.append(label)
        ranked = kept[:3]
        # the plain properties kept back most often when a request leaves them open come first (piece count, then the
        # material, about one task in nine each); the reader's pick of another counts only when neither is open
        picked = lowered(answer.get("figure")).strip(" .")
        figure = next((f for f in open_figures if not any(f in r for r in ranked)), "")
        if not figure and picked and picked != "none" and picked not in self.asked_props and not self.property_settled(picked) \
                and not any(picked in r for r in ranked) and not any(word in wanted for word in picked.split() if len(word) >= 4):
            figure = picked
        self.note(f"shelf labels {list(facts)[:12]} -> {ranked} figure={figure or '-'}")
        # the shelf's two likeliest labels, then the plain property, then the shelf's next label
        head = ranked[:2] + ([figure] if figure else []) + ranked[2:]
        self.ask_props = head + [p for p in self.ask_props if p not in head]

    @staticmethod
    def content_words(text: str) -> set[str]:
        # words of five letters or more carry the meaning; the shorter ones are mostly function words (that, with, from)
        return {singular(w) for w in word_list(text) if len(w) >= 5}

    def within(self, inner: str, outer: str) -> bool:
        """Every content word of one phrase stands in the other."""
        words = self.content_words(inner)
        return bool(words) and words <= self.content_words(outer)

    @staticmethod
    def requirement(raw: dict) -> dict:
        def texts(name: str) -> list[str]:
            return [str(w).strip() for w in raw.get(name) or [] if str(w).strip()][:6] if isinstance(raw.get(name), list) else []
        value = re.sub(r"(\d)\s*(?:\"|\u201d|\u2033|'')", r"\1 inch", str(raw.get("value") or "").strip())      # 5" is 5 inch
        name = lowered(raw.get("property"))
        filed = not re.search(r"\d", value) and not re.search(r"colou?r|material", name)
        return {"filed": filed, "property": str(raw.get("property") or "").strip(), "value": value,
                "written": texts("written"), "alike": texts("alike")}

    def state_text(self) -> str:
        lines = [f"Request: {self.counter.request}", f"Item: {self.item}",
                 "Requirements: " + ("; ".join(f"{r['property']} = {r['value']}" for r in self.requirements) or "(none)"),
                 f"Budget: {self.budget if self.budget is not None else 'none'}"]
        if self.priority:
            lines.append(f"Priority between fitting items: {self.priority}")
        return "\n".join(lines)

    # ---- searches: what to ask the store
    @staticmethod
    def query_text(query: Any) -> str:
        return lowered(query)[:QUERY_LENGTH].strip()

    def add_queries(self, queries: list, first: bool = False) -> None:
        """Searches proposed along the way (by the reader, from a fitting listing, from the shelf's words), queued
        behind the planned ones; first: ahead of the others queued."""
        fresh = [q for q in dict.fromkeys(self.query_text(q) for q in queries) if q and q not in self.sent and q not in self.queries]
        self.queries = fresh + self.queries if first else self.queries + fresh

    def planned(self) -> list[str]:
        """The planned searches not sent yet, worked out afresh from the requirements in force: after a change by the
        shopper the new values' searches come first and the old values' are not sent any more."""
        return [q for q in query_plan(self.category or self.item, self.requirements, self.store_label()) if q not in self.sent]

    def search_values(self) -> str:
        return " ".join(v for v in (closed_up(r["value"]) for r in self.requirements) if v)

    def head_word(self) -> str:
        words = word_list(self.category or self.item)
        return singular(words[-1]) if words else ""

    def shelf_words(self) -> list[str]:
        """The words the shelf's own listings carry most often (rows and pages filed on it), apart from the values and
        the item's words: listings that call the item something else carry them."""
        titles = {k[0]: r.get("title") for k, r in self.counter.rows.items() if k in self.counter.shelf or self.in_category(r)}
        titles.update({pid: p.get("title") for pid, p in self.counter.pages.items() if self.in_category(p)})
        if len(titles) < 3:
            return []
        known = {singular(w) for w in word_list(f"{self.category} {self.item} {self.search_values()}")}
        counts: dict[str, int] = {}
        for title in titles.values():
            for word in {w for w in word_list(title) if len(w) > 3 and w.isalpha() and singular(w) not in known}:
                counts[word] = counts.get(word, 0) + 1
        return [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n > 1]

    def shelf_searches(self, count: int) -> list[str]:
        """The shelf's next most common words, each with every value (half of them with the item's last word too)."""
        if not self.requirements:
            return []
        words = [w for w in self.shelf_words() if w not in self.vocabulary_used][:count]
        self.vocabulary_used.update(words)
        every, head = self.search_values(), self.head_word()
        return [f"{w} {every}" for w in words] + [f"{w} {every} {head}" for w in words[: count // 2]]

    def fitting_titles(self) -> set[str]:
        return {lowered((self.counter.pages.get(k[0]) or {}).get("title")) for k in self.verdicts if self.fits_words(k)} - {""}

    def relatives_waiting(self) -> bool:
        return len(self.related) < RELATED_TITLES and bool(self.fitting_titles() - self.related)

    def own_word(self, words: list[str], title: str) -> str:
        """The longest word of a title that no other row's title carries: it leads to that article's other listings."""
        elsewhere = {w for r in self.counter.rows.values() if lowered(r.get("title")) != title for w in word_list(r.get("title"))}
        return max((w for w in words if len(w) > 4 and w.isalpha() and w not in elsewhere), key=len, default="")

    def off_by_variant(self, key: tuple[str, str], verdict: dict) -> bool:
        """The listing is the kind asked for and misses only on what variants differ by (a colour, an amount)."""
        if not verdict.get("kind_ok") or verdict.get("by") != "words" or not self.requirements:
            return False
        off = [r for r in self.requirements if self.plain(key, r) < 1]
        return bool(off) and all(self.kind(r) in ("colour", "amount") for r in off)

    def relatives(self, near: bool = True) -> list[str]:
        """Searches for the relatives of the listings that fit: other variants of the same article, or the article
        listed again. A search shows one variant of each listing, so a fitting listing's title with the values (or
        with its maker, or a word only its title carries) brings up the rest. near: also a listing that misses only
        on what variants differ by - its title with the wanted value."""
        every, head, out = self.search_values(), self.head_word(), []
        for key, verdict in list(self.verdicts.items()):
            if len(self.related) >= RELATED_TITLES:
                break
            page = self.counter.pages.get(key[0]) or {}
            title = lowered(page.get("title"))
            words = word_list(title)
            if title in self.related or len(words) < 2:
                continue
            if self.fits_words(key):
                self.related.add(title)
                opening = " ".join(words[:8])
                out += [f"{every} {opening}", f"{every} {' '.join(words[:4])}", f"{every} {head} {' '.join(words[:2])}"]
                options = " ".join(word_list(self.variant_of(key).get("options")))
                if options:
                    out.append(f"{options} {opening}")
                maker = lowered(page.get("brand"))
                if maker and maker not in NO_MAKER:
                    out.append(f"{every} {maker} {head}")
                out.append(self.own_word(words, title))
            elif near and self.off_by_variant(key, verdict):
                self.related.add(title)
                out += [f"{closed_up(r['value'])} {' '.join(words[:8])}" for r in self.requirements if self.plain(key, r) < 1]
        return [q for q in out if q]

    def label_guesses(self) -> list[str]:
        """The item's name as the store might file it: as written, with its last word plural, with each word before
        a joining sign plural too (A & B); a describing word in front may not belong to the label, so also without
        the first word or two."""
        words = (self.category or self.item).split()
        guesses: list[str] = []
        for start in range(min(3, len(words))):
            tail = words[start:]
            if start and not re.match(r"[A-Za-z]{3}", tail[0]):
                continue
            last = " ".join(tail[:-1] + [plural(tail[-1])])
            joined = " ".join(plural(w) if i == len(tail) - 1 or re.fullmatch(r"&|/|and", tail[i + 1]) or w.endswith(",") else w
                              for i, w in enumerate(tail))
            guesses += [" ".join(tail), last, joined] if not start else [last]
        return list(dict.fromkeys(guesses))

    def store_label(self) -> str:
        """The label the store files the wanted item under: the shelf the shopper names, else one a filter returned
        rows for, else a label read on a listing that matches the item's name."""
        label = self.named_shelf() or next(iter(self.counter.shelf.values()), "")
        if label:
            return label
        wanted = {label_key(self.category), label_key(self.item)} - {""}
        return next((str(l) for p in self.counter.pages.values() for l in p.get("category_path") or [] if label_key(l) in wanted), "")

    def shelf_filters(self, count: int) -> list[dict]:
        """The shelf under the next price ceilings: each ceiling shows other listings filed there, and the words they
        carry feed the shelf's searches. (Measured: they rarely show a listing the searches miss, so few per turn.)"""
        label = self.store_label()
        if not label or self.budget is None or not self.counter.has("filter"):
            return []
        out = []
        for share in PRICE_STEPS:
            tag = f"filter|{lowered(label)}|{share}"
            if len(out) >= count or len(self.sent) >= self.search_room:
                break
            if tag not in self.sent:
                self.sent.add(tag)
                out.append(act("filter", category=label, max_price=round(self.budget * share, 2), in_stock=True, k=10))
        return out

    def queue_filters(self) -> None:
        """A brand the shopper now requires: the shelf (or the item's likely labels) and the whole store under it."""
        brand = next((r["value"] for r in self.requirements if "brand" in lowered(r["property"])), None)
        label = self.store_label()
        for category in [label] if label else self.label_guesses()[:2]:
            self.pending_filters.append(act("filter", category=category, brand=brand, max_price=self.budget, in_stock=True, k=10))
        if brand:
            self.pending_filters.append(act("filter", brand=brand, max_price=self.budget, in_stock=True, k=10))

    def next_searches(self, room: int) -> list[dict]:
        """The store calls for the next turn: filters waiting, the planned searches not sent yet, then the queue."""
        out: list[dict] = []
        if self.counter.has("filter"):
            out, self.pending_filters = self.pending_filters[:room], self.pending_filters[room:]
        for query in self.planned() + self.queries:
            if len(out) >= room or len(self.sent) >= self.search_room:
                break
            if query not in self.sent:
                self.sent.add(query)
                out.append(act("search", query=query, k=10, in_stock=True))
        self.queries = [q for q in self.queries if q not in self.sent]
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

    def hear(self) -> tuple[bool, bool]:
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
            said = f"{self.state_text()}\n\nThe shopper's new line:\n{line}"
            heard = self.listed_fields(self.reader.fill(HEAR_GUIDE, said, HEAR_TOOL, 1500) if self.can_think() else None)
            if self.awaiting_answer and not [a for a in heard.get("adds") or [] if isinstance(a, dict)] and self.can_think():
                # the line answers the question just asked, and the first reading found no requirement in it
                again = self.listed_fields(self.reader.fill(HEAR_GUIDE, said, HEAR_TOOL, 1500, second=True))
                if [a for a in again.get("adds") or [] if isinstance(a, dict)]:
                    heard = again
            answering, self.awaiting_answer = self.awaiting_answer, False
            # only what the line itself says is taken: a value it never names, or a withdrawal of something it does
            # not mention, is the reading's own
            adds = [a for a in heard.get("adds") or [] if isinstance(a, dict) and said_in(line, a)]
            heard["drops"] = [d for d in heard.get("drops") or []
                              if drop_said_in(line, str(d), adds, [r["value"] for r in self.requirements if lowered(r["property"]) == lowered(str(d).split("=")[0])])]
            heard["adds"] = adds
            if heard.get("about_change") is True:
                # a line reacting to the market change: its figure is a price, not a budget (a requirement it states still counts)
                heard["budget"] = 0
            replaced = {lowered(a.get("property")) for a in heard.get("adds") or [] if isinstance(a, dict)}
            budget_line = money(heard.get("budget")) is not None and money(heard.get("budget")) != self.budget
            for name in heard.get("drops") or []:
                if budget_line and lowered(name) not in replaced:
                    continue                    # a new spending limit restates some requirements; it withdraws none
                kept = [r for r in self.requirements if lowered(r["property"]) != lowered(name)]
                changed = changed or len(kept) != len(self.requirements)
                self.requirements = kept
            for add in heard.get("adds") or []:
                if not isinstance(add, dict) or not str(add.get("value") or "").strip():
                    continue
                new = self.requirement(add)
                if SHELF_NAME.search(lowered(new["property"])) and new["value"] and lowered(new["value"]) != lowered(self.told_category):
                    # the shopper names the store's own category: the shelf to buy from
                    self.told_category = new["value"]
                    self.shelf_cache = (-1, "")
                    self.pending_filters.append(act("filter", category=new["value"], max_price=self.budget, in_stock=True, k=10))
                    changed = True
                    continue
                if NOT_A_PROPERTY.search(lowered(f"{new['property']} {new['value']}")):
                    continue
                same = next((r for r in self.requirements if lowered(r["property"]) == lowered(new["property"])), None)
                if same is not None and lowered(same["value"]) == lowered(new["value"]):
                    continue
                if same is not None:
                    self.requirements.remove(same)
                if answering and self.pair_fallback and self.questions_asked == 1:
                    new["chosen"] = True        # the side of the doubted pair the shopper said matters more
                known_words = {w for r in self.requirements for w in self.name_words(r["property"])} - {"type", "kind"}
                if answering and same is None and not (self.name_words(new["property"]) - {"type", "kind"}) & known_words:
                    self.told += 1              # the question drew out a requirement on a property not known before
                    self.told_props.append(new["property"])
                self.requirements.append(new)
                changed = True
                if "brand" in lowered(new["property"]):
                    # a brand the shopper now requires: ask the store for it directly
                    self.queue_filters()
                    self.add_queries([f"{new['value']} {self.item}", new["value"]], first=True)
            named = " ".join(str(heard.get("category") or "").split()).strip(" .")
            if named and lowered(named) != lowered(self.told_category):
                # the shopper names the store's own category: the shelf to buy from
                self.told_category = named
                self.shelf_cache = (-1, "")
                self.pending_filters.append(act("filter", category=named, max_price=self.budget, in_stock=True, k=10))
                changed = True
            budget = money(heard.get("budget"))
            if budget is not None and any(abs(budget - float((self.counter.stock.get(k) or {}).get("price") or -1.0)) < 0.005 for k in self.counter.cart):
                budget = None               # the price of an item in the cart, not a new limit
            if budget is not None and budget != self.budget:
                lowered_limit = self.budget is None or budget < self.budget
                self.budget_cut = self.budget_cut or (self.budget is not None and budget < self.budget)
                self.budget, changed = budget, True
                if lowered_limit:
                    self.queue_filters()        # the shelf again, under the new limit
            if str(heard.get("priority") or "").strip():
                self.priority, changed = str(heard["priority"]).strip(), True
                self.priority_units = {}        # a new priority: its figures are read by the reader
            after = [(r["property"], r["value"]) for r in self.requirements]
            taken_back = [v for p, v in before if (p, v) not in after]
            if after != before and (not answering or taken_back):
                # the shopper changed what they want (a value replaced or withdrawn is a change even while a question
                # waits for its answer): a plain reply before the order
                self.unanswered.append(line)
                self.changed_values[line] = ([v for p, v in after if (p, v) not in before], taken_back)
            if changed:
                self.add_queries(list(heard.get("queries") or []), first=True)
                # what the shopper wants moved: room to look again, whatever was used before
                self.search_room = max(self.search_room, len(self.sent) + 16)
                self.fresh_rows = len(self.counter.row_order)
                # relatives and shelf words were searched with the old values: they are searched again with the new
                self.related.clear()
                self.vocabulary_used.clear()
                self.open_room = max(self.open_room, len(self.counter.pages) + OPEN_AFTER_CHANGE)
                self.judge_room = max(self.judge_room, self.judged_count + 36)
            self.note(f"heard {line[:80]!r} -> adds={heard.get('adds')} drops={heard.get('drops')} budget={heard.get('budget')} asks={heard.get('asks')}")
            if heard.get("asks") and not heard.get("about_change"):
                if self.owed_reaction:
                    self.owed_reaction = False
                else:
                    asks = True
                    self.pending_line = line
            elif heard.get("about_change"):
                self.owed_reaction = False
        return changed, asks

    # ---- reading listings
    def affordable(self, key: tuple[str, str]) -> bool:
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        if seen.get("in_stock") is False or key in self.gone:
            return False
        return self.budget is None or not isinstance(price, float) or price <= self.budget + 1e-9

    def named_shelf(self) -> str:
        """The store's own label for the shelf the request names: of the labels read on listings, the longest
        whose words stand together in the request."""
        pages = self.counter.pages
        if self.shelf_cache[0] == len(pages):
            return self.shelf_cache[1]
        if self.told_category:
            told = label_key(self.told_category)
            for page in pages.values():
                for label in page.get("category_path") or []:
                    if label_key(label) == told:
                        self.shelf_cache = (len(pages), str(label))
                        return str(label)
        request = f" {label_key(self.counter.request)} "
        seen: dict[str, int] = {}
        for page in pages.values():
            path = [str(label) for label in page.get("category_path") or []]
            if path and label_key(path[-1]) and f" {label_key(path[-1])} " in request:
                seen[path[-1]] = seen.get(path[-1], 0) + 1
        own = {label_key(self.category), label_key(self.item)} - {""}
        best = max(seen, key=lambda label: (len(label_key(label).split()), label_key(label) in own, seen[label]), default="")
        self.shelf_cache = (len(pages), best)
        return best

    def in_category(self, page: dict) -> bool:
        named = self.named_shelf()
        wanted = {label_key(named)} if named else {label_key(self.category), label_key(self.item)} - {""}
        labels = {label_key(label) for label in page.get("category_path") or []}
        return bool(wanted & labels)

    def variant_of(self, key: tuple[str, str]) -> dict:
        page = self.counter.pages.get(key[0]) or {}
        return next((v for v in page.get("variants") or [] if str(v.get("sku")) == key[1]), {})

    @staticmethod
    def name_words(text: Any) -> set[str]:
        words = {singular(w) for w in word_list(str(text).replace("_", " "))} - {"of"}
        return {"color" if w == "colour" else w for w in words}

    # ---- how plainly a listing states a required value (the evidence model)
    @staticmethod
    def value_tokens(text: Any) -> list[str]:
        """Words and figures as compared: lower case, singular, one spelling of grey and colour, amounts joined."""
        text = closed_up(str(text or "")).replace("gray", "grey").replace("colour", "color")
        return [singular(w) for w in re.findall(r"[a-z]+|\d+(?:\.\d+)?[a-z]*", text)]

    def spellings(self, requirement: dict, loose: bool = False) -> list[list[str]]:
        """The value as words, in each way it is written; loose: also the other words sellers use for it. Another
        spelling of a word counts only when it is close to the word itself (grey: gray), or keeps every word of it;
        a shorter or different word ('steel' for stainless steel, 'SS') is another word, read only loosely."""
        value = self.value_tokens(requirement["value"])
        out = [value] if value else []
        for form in requirement.get("written") or []:
            tokens = self.value_tokens(form)
            if not tokens or tokens in out:
                continue
            if amount_set(form) or set(value) <= set(tokens) or \
                    SequenceMatcher(None, " ".join(tokens), " ".join(value)).ratio() >= 0.8:
                out.append(tokens)
            elif loose:
                out.append(tokens)
        if loose:
            out += [t for t in (self.value_tokens(w) for w in requirement.get("alike") or []) if t and t not in out]
        return out

    def kind(self, requirement: dict) -> str:
        """Where a store states a property: an amount; a colour, which the variant's options name; a material;
        anything else, which it keeps as a fact of its own."""
        if self.wanted_amounts(requirement):
            return "amount"
        name = lowered(requirement["property"])
        if re.search(r"colou?r", name):
            return "colour"
        return "material" if re.search(r"material|upholster", name) else "fact"

    def wanted_amounts(self, requirement: dict) -> set[tuple[str, float]]:
        """The required amount(s): the value with its unit, its other spellings, or a bare figure in the unit the
        property's name implies (20 under 'pieces'): the units of the value and of its other spellings are the ones
        that measure it."""
        wanted = amount_set(requirement["value"])
        for form in requirement.get("written") or []:
            wanted |= amount_set(form)
        bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(requirement["value"]))
        if not wanted and bare:
            # a bare figure: in each word of the property's name taken as its unit (20 under 'pieces')
            wanted = {(unit_word(word), float(bare.group(1))) for word in word_list(requirement["property"]) if word.isalpha()}
        return wanted

    def own_facts(self, page: dict, requirement: dict) -> list[tuple[str, str]]:
        """The page's facts filed under this property: labels sharing a telling word with its name, or (with none) the
        labels under which the shelf files this very value on two listings or more."""
        facts = {str(k): str(v) for k, v in (page.get("facts") or {}).items()}
        want = self.label_core(requirement["property"])
        own = [label for label in facts if want & self.label_core(label)]
        if not own:
            value = self.value_tokens(requirement["value"])
            filed = self.shelf_filing()
            own = [label for label in facts if sum(1 for v in filed.get(label, ()) if self.value_tokens(v) == value) >= 2]
        return [(label, facts[label]) for label in own]

    def shelf_filing(self) -> dict[str, list[str]]:
        """Every fact label of the shelf's listings read so far, with the values filed under it."""
        if self.filing_cache[0] != len(self.counter.pages):
            filed: dict[str, list[str]] = {}
            for page in self.counter.pages.values():
                if self.in_category(page):
                    for label, value in (page.get("facts") or {}).items():
                        filed.setdefault(str(label), []).append(str(value))
            self.filing_cache = (len(self.counter.pages), filed)
        return self.filing_cache[1]

    def says(self, text: Any, requirement: dict, loose: bool = False) -> bool:
        """The text carries one of the value's spellings as a run of words. A value named only to say the item
        lacks it ('aluminium-free', 'no BPA', '0% alcohol') is not stated."""
        text = lowered(text)
        if not NEGATING.search(lowered(requirement["value"])):
            text = NEGATED.sub(" ", text)
        have = self.value_tokens(text)
        return any(any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1)) for want in self.spellings(requirement, loose))

    def negates(self, text: Any, requirement: dict) -> bool:
        """The text names the required value in a phrase that says the item lacks it."""
        if NEGATING.search(lowered(requirement["value"])):
            return False
        words = {w for want in self.spellings(requirement) for w in want if len(w) >= 3}
        return any(words & set(self.value_tokens(found.group(0))) for found in NEGATED.finditer(lowered(text)))

    def plain(self, key: tuple[str, str], requirement: dict) -> int:
        """How plainly the variant states one required value: 3 where the store states that kind of property (the
        variant's options, the property's own fact), 2 in the listing's name or beside its own fact, 1 only elsewhere
        or in another word for it, 0 nowhere, -1 when it states something else for it or names it only to deny it."""
        page = self.counter.pages.get(key[0]) or {}
        options, title = str(self.variant_of(key).get("options") or ""), str(page.get("title") or "")
        facts = [str(v) for v in (page.get("facts") or {}).values()]
        kind = self.kind(requirement)
        if self.negates(f"{title} {options}", requirement):
            return -1
        own = self.own_facts(page, requirement)
        if kind == "amount":
            wanted = self.wanted_amounts(requirement)
            units = {unit for unit, _ in wanted}
            in_options = amount_set(options)
            if in_options & wanted:
                # the variant's name gives the amount; beside other amounts of that kind (60cm 60*29*168 cm) it is unclear
                return 1 if {(u, n) for u, n in in_options if u in units} - wanted else 3
            if {unit for unit, _ in in_options} & units:
                return -1               # the variant's own options give another amount in the same unit
            in_own = set().union(*(amount_set(value) for _, value in own)) if own else set()
            if in_own & wanted:
                return 3
            if amount_set(title) & wanted:
                return 2
            figures = {figure for _, figure in wanted}
            if any(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", value) and float(value) in figures for _, value in own):
                return 2                # the property's own fact gives the figure without its unit
            if any(amount_set(value) & wanted for value in facts):
                return 1
            return -1 if {unit for unit, _ in in_own} & units else 0
        if kind == "colour":
            words = set(self.value_tokens(options))
            # the colour stands in the variant's name as a word of its own: joined to a code or a size ('blue-3pcs',
            # '3.5l-green') it is part of another word, and the variant is named otherwise
            parts = [part for part in re.split(r"[\s/,;|()+]+", lowered(options).replace("gray", "grey")) if part]
            standing = {singular(part) for part in parts if re.fullmatch(r"[a-z]+", part)}
            if self.says(options, requirement) and not set(self.value_tokens(requirement["value"])) <= standing:
                return -1
            if self.says(options, requirement):
                named = {w for w in words if COLOUR_WORD.fullmatch(w)} - set(self.value_tokens(requirement["value"])) - {"multicolor", "multicolour"}
                return 1 if named else 3        # 'wine red' is red; 'red / blue' leaves it open
            if any(COLOUR_WORD.fullmatch(w) for w in words):
                return -1                       # the variant is another colour
            if any(self.says(value, requirement) for _, value in own):
                return 2 if options.strip() else 3
            if self.says(title, requirement):
                return 1 if options.strip() else 2
            if any(self.says(value, requirement) for value in facts) or self.says(f"{title} {options}", requirement, loose=True):
                return 1
            return 0
        for _, value in own:
            entries = [entry.strip() for entry in re.split(r"[,;/|]+", value) if entry.strip()]
            if any(self.says(entry, requirement) for entry in entries):
                return 3 if len(entries) == 1 else 1     # one of several entries leaves it open
        if self.says(options, requirement):
            return 3 if kind == "material" else 2
        if own and kind != "material":
            return -1           # the property's own fact states something else: a word for it elsewhere is not the value
        if kind == "material" and self.says(title, requirement):
            return 2
        if any(self.says(value, requirement) for value in facts) or self.says(title, requirement):
            return 1
        if any(self.says(f"{title} {options} {' '.join(facts)}", requirement, loose=True) for _ in (0,)):
            return 1
        return -1 if own else 0                         # its own fact states something else

    def level_needed(self, requirement: dict) -> int:
        """The level at which a value counts as stated where the store states that kind of property."""
        return 2 if self.kind(requirement) in ("amount", "material") else 3

    def settled(self, key: tuple[str, str], requirement: dict) -> bool:
        """The page's own words state the value, plainly or not."""
        return self.plain(key, requirement) >= 1

    def row_score(self, key: tuple[str, str], context: tuple[set, set, set] | None = None) -> float:
        """How likely a search row is to fit, to choose which listings to read: the shopper's own listings first,
        then relatives of fitting ones, rows filed on the shelf, rows whose title states the values, and rows found
        since the requirements last changed (rows found for earlier values tie with them otherwise: search rows
        carry no shelf). context: row_context(), worked out once by a caller that scores many rows."""
        near, siblings, fresh = context or self.row_context()
        row = self.counter.rows.get(key) or {}
        title = stems(row.get("title"))
        score = 50.0 if self.listed and key in self.listed else 0.0
        if lowered(row.get("title")) in near or "-" in key[1] and key[1].rsplit("-", 1)[0] in siblings:
            score += 20
        if key in self.counter.shelf or self.in_category(row):
            score += 4
        if key in fresh:
            score += 3
        for requirement in self.requirements:
            forms = [stems(f) for f in [requirement["value"], *(requirement.get("written") or []), *(requirement.get("alike") or [])]]
            score += 3 if any(f and f <= title for f in forms) else 1 if forms[0] & title else 0
        return score + len({singular(w) for w in word_list(self.category or self.item)} & title)

    def row_context(self) -> tuple[set, set, set]:
        """For scoring rows: the titles of fitting listings and of those whose relatives were searched for, the stems
        of fitting variants' codes (another variant of a listing shares the stem), and the rows found since the
        requirements last changed."""
        siblings = {k[1].rsplit("-", 1)[0] for k in self.verdicts if "-" in k[1] and self.fits_words(k)}
        return self.related | self.fitting_titles(), siblings, set(self.counter.row_order[self.fresh_rows:])

    def to_open(self, limit: int, allowance: int | None = None) -> list[dict]:
        """The next listings to read in full, best rows first, within what may still be read."""
        pages = self.counter.pages
        room = (self.open_room if allowance is None else allowance) - len(pages)
        rows = [k for k in self.counter.row_order if k[0] not in pages and self.affordable(k)]
        if self.listed:
            rows = [k for k in self.listed if k[0] not in pages]
        context = self.row_context()
        rows = sorted(dict.fromkeys(rows), key=lambda k: -self.row_score(k, context))
        ids = list(dict.fromkeys(k[0] for k in rows))[: max(0, min(limit, room))]
        if self.counter.has("compare"):
            return [act("compare", product_ids=ids[i:i + OPEN_BATCH]) for i in range(0, len(ids), OPEN_BATCH)]
        return [act("view", product_id=i) for i in ids]

    def candidates(self) -> list[tuple[str, str]]:
        out = []
        any_shelved = any(self.in_category(p) for p in self.counter.pages.values())
        for product_id, page in self.counter.pages.items():
            if any_shelved and not self.in_category(page):
                continue           # the store files the wanted kind under its own label: only those listings count
            for variant in page.get("variants") or []:
                key = (product_id, str(variant.get("sku")))
                if self.affordable(key) and (not self.listed or key in self.listed):
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
            named = self.own_facts(page, requirement)
            if named:
                facts += f" || the fact named for requirement {position}: " + "; ".join(f"{label} = {value[:110]}" for label, value in named)
        price = (self.counter.stock.get(key) or {}).get("price")
        return (f"[{number}] title: {str(page.get('title') or '')[:150]} | options: {self.variant_of(key).get('options') or '-'} | price: {price} | "
                f"category: {' > '.join(str(x) for x in (page.get('category_path') or [])[-2:])} | brand: {page.get('brand') or '-'} | facts: {facts or '-'}")

    def judge(self) -> None:
        """Settle every unjudged candidate by the page's own words."""
        shelf_words = set(label_key(self.named_shelf()).split())
        self.requirements = [r for r in self.requirements
                             if not (r.get("stated") and shelf_words and set(label_key(r["value"]).split()) <= shelf_words)]
        stamp = json.dumps([self.requirements, self.priority, self.lenient, len(self.counter.pages)], sort_keys=True)
        if stamp != self.judged_under:
            self.verdicts, self.judged_under = {}, stamp
        for key in self.candidates():
            if key not in self.verdicts:
                in_shelf = self.in_category(self.counter.pages[key[0]])
                met = [self.settled(key, r) or (self.lenient and self.plain(key, r) >= 1) for r in self.requirements]
                self.verdicts[key] = {"kind_ok": in_shelf, "met": met, "by": "words"}

    def ask_reader(self) -> None:
        """Have the reader look at the nearest misses: a synonym, another spelling or another unit may state what plain words did not find."""
        open_keys = [k for k in self.verdicts if self.affordable(k) and self.verdicts[k].get("by") == "words" and self.verdicts[k].get("kind_ok")
                     and not self.fits_words(k) and not any(self.plain(k, r) == -1 for r in self.requirements)]
        open_keys.sort(key=lambda k: -sum(self.plain(k, r) for r in self.requirements))
        batch = open_keys[:JUDGE_CARDS]
        if not batch or not self.can_think() or self.judged_count >= self.judge_room:
            return
        self.judged_count += len(batch)
        wanted = "\n".join(f"{i}. {r['property']}: {r['value']}" for i, r in enumerate(self.requirements, start=1)) or "(none beyond the kind of thing)"
        text = (f"The shopper wants: {self.item}\nRequirements, in order:\n{wanted}\n"
                + (f"Priority between fitting items: {self.priority}\n" if self.priority else "")
                + "\nListings:\n" + "\n".join(self.card(i, key) for i, key in enumerate(batch, start=1)))
        answer = self.reader.fill(JUDGE_GUIDE, text, JUDGE_TOOL, 1400) or {}
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
            met = [self.settled(key, r) or (said[i] and self.plain(key, r) != -1 and not re.search(r"\d", r["value"])) for i, r in enumerate(self.requirements)]
            self.verdicts[key] = {"kind_ok": found.group(2) in "Kk", "met": met, "by": "reader"}
        for key in batch:
            if self.verdicts[key].get("by") == "words":
                self.verdicts[key]["by"] = "reader"      # shown once: not shown again
        for proof in answer.get("proof") or []:
            if isinstance(proof, dict) and isinstance(proof.get("n"), int) and 1 <= proof["n"] <= len(batch):
                self.verdicts[batch[proof["n"] - 1]].update({k: proof.get(k) for k in ("where", "words", "rank_value")})
        self.note(f"reader judged {len(batch)}: " + " | ".join(str(m)[:12] for m in marks)[:260])
        self.add_queries(answer.get("queries") or [])

    def fits_words(self, key: tuple[str, str]) -> bool:
        """The listing's words meet every requirement, whatever its stock is now."""
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
        """Fitting listings, best first: those stating every value plainly, then the shopper's priority, then the page
        that states the values most plainly, then price."""
        if self.better in ("higher", "lower"):
            direction = -1.0 if self.better == "higher" else 1.0
        else:
            direction = -1.0 if re.search(r"\b(high|larg|big|long|most|max|best)", lowered(self.priority)) else 1.0
        by_price = self.priced_priority()

        def order(key: tuple[str, str]) -> tuple:
            verdict = self.verdicts[key]
            price = (self.counter.stock.get(key) or {}).get("price") or 0.0
            value = price if by_price else verdict.get("rank_value")
            known = isinstance(value, (int, float)) and not isinstance(value, bool)
            plainly = sum(self.plain(key, r) for r in self.requirements)
            twin = lowered((self.counter.pages.get(key[0]) or {}).get("title")) in self.lost_titles
            # a listing that states every value where it is written plainly (options, own fact, name) comes before one
            # that only hints at a value, whatever the priority says: an order that misses a requirement earns nothing
            weakest = min((self.plain(key, r) for r in self.requirements), default=3)
            tier = 0 if weakest >= 2 else 1 if weakest >= 1 else 2
            return (tier, 0 if known or not self.priority else 1, direction * float(value) if known and self.priority else 0.0,
                    0 if self.firm(key) else 1, 0 if self.solid(key) else 1, 0 if self.verdicts[key].get("by") == "words" else 1, -plainly,
                    0 if twin else 1, price)

        return sorted((k for k in self.verdicts if self.fits(k)), key=order)

    def keeps_request(self, key: tuple[str, str]) -> bool:
        """The listing is the kind asked for and states every requirement the request itself wrote, where the store
        states that kind of property: the first cart is the shopper's first sight of the choice and holds to those."""
        stated = [r for r in self.requirements if r.get("stated")]
        return bool((self.verdicts.get(key) or {}).get("kind_ok")) and all(self.plain(key, r) >= self.level_needed(r) for r in stated)

    def of_preferred(self, key: tuple[str, str]) -> bool:
        """The listing is by the maker the shopper would take first."""
        if not self.preferred_brand:
            return False
        page = self.counter.pages.get(key[0]) or {}
        wanted = set(word_list(self.preferred_brand))
        return bool(wanted) and (wanted <= set(word_list(page.get("brand"))) or wanted <= set(word_list(page.get("title"))))

    def meets_chosen(self, key: tuple[str, str]) -> bool:
        """The listing meets what the shopper said matters more (the chosen side of a doubted pair), when there is one."""
        met = ((self.verdicts.get(key) or {}).get("met") or [])[: len(self.requirements)]
        marks = [x for x, r in zip(met, self.requirements) if r.get("chosen")]
        return bool(marks) and all(marks)

    def priced_priority(self) -> bool:
        """The shopper's priority is about the price itself."""
        return bool(re.search(r"\b(cheap|price|cost|expensive|afford)", lowered(self.priority)))

    @staticmethod
    def unit_sizes(raw: Any) -> dict[str, float]:
        """The reader's units for the priority, each with its size in the first unit."""
        out: dict[str, float] = {}
        for entry in raw if isinstance(raw, list) else []:
            unit = unit_word(re.sub(r"[^a-z]", "", str((entry or {}).get("unit") or "").lower())) if isinstance(entry, dict) else ""
            size = money(entry.get("size")) if unit else None
            if unit and size:
                out.setdefault(unit, size)
        return out

    def figure_for(self, key: tuple[str, str], units: dict[str, float]) -> float | None:
        """The listing's figure for an amount priority, in the first of the priority's units: the variant's own options
        first, then a single figure in the title, then a fact - one whose label shares a word with the priority or names
        one of its units first (a bare number there takes the unit its label names)."""
        page = self.counter.pages.get(key[0]) or {}

        def sized(text: str) -> set[float]:
            return {round(figure * units[unit], 4) for unit, figure in amount_set(text) if unit in units}

        for text in (str(self.variant_of(key).get("options") or ""), str(page.get("title") or "")):
            sizes = sized(text)
            if len(sizes) == 1:
                return sizes.pop()
            if sizes and text is not page.get("title"):
                return max(sizes)
        named = {singular(w) for w in word_list(self.priority) if len(w) > 3} | set(units)
        facts = [(str(label), str(value)) for label, value in (page.get("facts") or {}).items()]
        words = {label: {unit_word(w) for w in word_list(label.replace("_", " "))} for label, _ in facts}
        for label, value in sorted(facts, key=lambda f: not words[f[0]] & named):
            sizes = sized(value)
            if sizes:
                return max(sizes)
            bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", value)
            if bare and words[label] & named:
                return float(bare.group(1)) * next((units[w] for w in words[label] if w in units), 1.0)
        return None

    def rank_by_priority(self) -> None:
        """When the shopper said what makes one fitting item better, read each fitting listing's figure for it: an
        amount from the listing's own words in code; anything else by the reader."""
        if not self.priority or self.priced_priority():
            return
        fitting = self.ranked()
        units = self.priority_units
        if units:
            # one unit for every listing, read the same way each time: the reader is not mixed in
            for key in fitting:
                if "rank_value" not in self.verdicts[key]:
                    self.verdicts[key]["rank_value"] = self.figure_for(key, units)
            if not self.better:
                text = lowered(self.priority)
                self.better = "lower" if LESS_IS_BETTER.search(text) and not MORE_IS_BETTER.search(text) else "higher"
            return
        if not self.can_think() or self.rank_asks >= 3:
            return
        keys = [k for k in fitting if "rank_value" not in self.verdicts[k]][:JUDGE_CARDS]
        if len(fitting) < 2 or not keys:
            return
        self.rank_asks += 1
        text = (f"Item: {self.item}\nThe shopper's priority: {self.priority}\n\nListings:\n"
                + "\n".join(self.card(i, key) for i, key in enumerate(keys, start=1)))
        answer = self.reader.fill(RANK_GUIDE, text, RANK_TOOL, 400) or {}
        if str(answer.get("better") or "").lower() in ("higher", "lower"):
            self.better = str(answer["better"]).lower()
        values = answer.get("values")
        values = re.split(r"[\n;]+", values) if isinstance(values, str) else (values or [])
        for entry in values:
            found = re.match(r"\s*\[?(\d+)\]?\W+(-?\d+(?:\.\d+)?)", str(entry))
            if found and 1 <= int(found.group(1)) <= len(keys):
                self.verdicts[keys[int(found.group(1)) - 1]]["rank_value"] = float(found.group(2))
        for key in keys:
            self.verdicts[key].setdefault("rank_value", None)      # read once: not asked again
        self.note(f"ranked by {self.priority!r}: " + " ".join(str(v)[:14] for v in values)[:200])

    def rethink_searches(self) -> bool:
        """Nothing seen fits: the reader writes new searches from the requirements, the searches tried and the titles
        seen, and the agent looks once more. True when it did."""
        if self.search_asks >= 2 or not self.can_think() or self.late() or self.counter.turns_left() <= TURN_RESERVE + 2:
            return False
        self.search_asks += 1
        tried = sorted(q for q in self.sent if not q.startswith("filter|"))[-25:]
        titles = list(dict.fromkeys(str(p.get("title") or "")[:90] for p in self.counter.pages.values()))[-25:]
        text = "\n\n".join([self.state_text(), "Searches tried:\n" + "\n".join(tried),
                            "Titles seen (none meets every requirement):\n" + "\n".join(titles)])
        answer = self.listed_fields(self.reader.fill(SEARCH_GUIDE, text, SEARCH_TOOL, 400))
        queries = [q for q in answer.get("queries") or [] if isinstance(q, str) and q.strip()][:6]
        self.note(f"new searches: {queries}")
        if not queries:
            return False
        self.add_queries(queries, first=True)
        self.look(rounds=3, enough=1)
        return True

    def recart(self) -> tuple[str, str] | None:
        """The next fitting listing into the cart, its stock and price read again."""
        counter = self.counter
        for key in [k for k in self.ranked() if k not in self.gone][:2]:
            if counter.over or counter.turns_left() < 3:
                break
            group = [act("add_to_cart", product_id=key[0], sku=key[1])]
            if counter.has("inspect_stock"):
                group.append(act("inspect_stock", product_id=key[0], sku=key[1]))
            counter.send(group)
            self.hear()
            if self.holds(key):
                return key
            self.gone.add(key)
        return None

    def reply_before_paying(self, choice: tuple[str, str]) -> None:
        """The lines the shopper said on their own get one plain reply before the order: each line in turn, with the
        item, its price and stock."""
        seen = self.counter.stock.get(choice) or {}
        title = str((self.counter.pages.get(choice[0]) or {}).get("title") or "")[:80]
        stock = "out of stock" if seen.get("in_stock") is False else "in stock"
        self.facts_for_reply.append(f"The item being ordered: {title}, price {seen.get('price')}, {stock}.")
        lines, self.unanswered = self.unanswered[-4:], []
        self.answer("\n".join(f"- {line}" for line in lines))

    def last_resort(self) -> None:
        """After a failure: still order the listing in the cart, or the best one left."""
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
        counter.send([act("place_test_order", product_id=key[0], sku=key[1], justification=self.claims(key))])

    def near_misses(self) -> list[tuple[str, str]]:
        def score(key: tuple[str, str]) -> tuple:
            verdict = self.verdicts[key]
            met = (verdict.get("met") or [])[: len(self.requirements)]
            chosen = sum(1 for x, r in zip(met, self.requirements) if x and r.get("chosen"))
            # what the shopper said matters more outranks the count of other requirements met
            return (-int(bool(verdict.get("kind_ok"))), -chosen, -sum(1 for x in met if x), (self.counter.stock.get(key) or {}).get("price") or 0.0)
        return sorted((k for k in self.verdicts if self.affordable(k)), key=score)

    # ---- looking
    def hear_quietly(self) -> None:
        if self.counter.heard:
            self.hear()

    def widen(self) -> None:
        """Too few listings fit by the words known: the reader names other words sellers write for each value; they
        join the value's spellings (the matcher reads them too) and are searched with the other values first."""
        self.widened = True
        head = self.head_word()
        for requirement in self.requirements:
            if not self.can_think():
                break
            answer = self.reader.fill(WORDS_GUIDE, f"Item: {self.item}\nProperty: {requirement['property']}\nValue: {requirement['value']}", WORDS_TOOL, 200) or {}
            words = [w for w in (" ".join(str(w).split()) for w in answer.get("words") or []) if w][:6]
            field = "written" if re.search(r"\d", requirement["value"]) else "alike"
            requirement[field] = list(dict.fromkeys([*(requirement.get(field) or []), *words]))
            rest = " ".join(closed_up(r["value"]) for r in self.requirements if r is not requirement)
            self.add_queries([f"{rest} {closed_up(w)} {head}" for w in words[:3]], first=True)
        self.note(f"widened: {[(r['value'], r.get('written'), r.get('alike')) for r in self.requirements]}")

    def refill(self, lost: tuple[str, str], wide: bool = True) -> None:
        """The chosen variant went out of stock or over the budget. A search shows each listing's best variant in
        stock, so the lost listing's own title now brings up its next variant and the same article listed again.
        wide: also the values with its title, the shelf's words, and the shelf at its price (variants of one
        listing share their price)."""
        counter = self.counter
        page = counter.pages.get(lost[0]) or {}
        words = word_list(page.get("title"))
        if len(words) < 2 or not counter.has("search"):
            return
        every, opening = self.search_values(), " ".join(words[:8])
        own = " ".join(w for w in word_list(self.variant_of(lost).get("options")) if not w.replace(".", "").isdigit())
        queries = [" ".join(words), opening, f"{own} {opening}" if own else "", f"{every} {opening}", f"{every} {' '.join(words[:4])}"]
        if wide:
            queries += self.shelf_searches(6)
        calls = [act("search", query=q, k=10, in_stock=True) for q in dict.fromkeys(self.query_text(q) for q in queries) if q]
        label, price = self.store_label(), (counter.stock.get(lost) or {}).get("price")
        if wide and label and isinstance(price, float) and counter.has("filter"):
            ceiling = round(min(self.budget or price * 1.1, price * 1.1), 2)
            calls.append(act("filter", category=label, max_price=ceiling, in_stock=True, k=10))
            maker = lowered(page.get("brand"))
            if maker and maker not in NO_MAKER:
                calls.append(act("filter", category=label, brand=str(page.get("brand")).strip(), max_price=ceiling, in_stock=True, k=10))
        counter.send(calls[: counter.group_size])
        self.hear_quietly()
        opens = self.to_open(4 * OPEN_BATCH, allowance=len(counter.pages) + 4 * OPEN_BATCH)
        if opens and not counter.over:
            counter.send(opens)
            self.hear_quietly()
        self.judge()

    def solid(self, key: tuple[str, str]) -> bool:
        """Every value is stated as required - in the options, the property's own fact or the title - not only
        in another word for it or in some other fact."""
        return (self.verdicts.get(key) or {}).get("by") == "words" and all(self.plain(key, r) >= 2 for r in self.requirements)

    def firm(self, key: tuple[str, str]) -> bool:
        """Every value is stated where it counts most: the options or the property's own fact (an amount: the title too)."""
        if (self.verdicts.get(key) or {}).get("by") != "words":
            return False
        return all(self.plain(key, r) >= self.level_needed(r) for r in self.requirements)

    def read_more(self) -> None:
        """Read the best unread rows in full (one turn), then hear any line the turn brought."""
        counter = self.counter
        opens = self.to_open(counter.group_size * OPEN_BATCH)
        if opens and not counter.over and counter.turns_left() > TURN_RESERVE:
            counter.send(opens[: counter.group_size])
            self.hear_quietly()

    def look(self, rounds: int, enough: int = 2) -> None:
        """Search and read listings until enough of them fit, or the rounds run out. Every search loads the store,
        so none is sent once enough listings fit firmly. A round: read the best rows, judge, then the next searches -
        relatives of what fits, the planned searches for the values in force, the shelf's words, a few shelf filters."""
        counter = self.counter
        for round_number in range(rounds):
            if counter.over or counter.turns_left() <= TURN_RESERVE or self.late():
                break
            pages_before = len(counter.pages)
            self.read_more()
            self.judge()
            fitting = [k for k in self.ranked() if self.solid(k)]
            if sum(1 for k in fitting if self.firm(k)) >= enough:
                return
            if len(fitting) >= enough and round_number > 1 and not self.relatives_waiting():
                return
            self.add_queries(self.relatives(near=len(fitting) < enough), first=True)
            if len(self.planned()) + len(self.queries) < counter.group_size // 2:
                self.add_queries(self.shelf_searches(6))
            calls = self.shelf_filters(SHELF_FILTERS)
            calls += self.next_searches(counter.group_size - len(calls))
            if not calls:
                if len(counter.pages) == pages_before:
                    break
                continue
            counter.send(calls)
            self.hear_quietly()
        self.read_more()
        self.judge()
        if len(self.ranked()) < enough and not self.widened and not counter.over:
            self.widen()
            return self.look(rounds=3, enough=enough)
        for _ in range(2):                  # plain words left too few: the reader may see the value stated in other words
            if len(self.ranked()) >= enough:
                return
            self.ask_reader()
        if len(self.ranked()) < enough and not self.lenient:
            self.lenient = True
            self.judge()

    def claims(self, key: tuple[str, str]) -> dict | None:
        seen = self.counter.stock.get(key) or {}
        page = self.counter.pages.get(key[0]) or {}
        places = {"title": str(page.get("title") or ""), "options": str(self.variant_of(key).get("options") or "")}
        places.update({str(label): str(value) for label, value in (page.get("facts") or {}).items()})
        claims: list[dict] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        claims.append({"field": "in_stock", "value": True})
        verdict = self.verdicts.get(key) or {}
        for where, words in zip(verdict.get("where") or [], verdict.get("words") or []):
            where = re.sub(r"^facts?\s*[:.]\s*", "", str(where or ""))
            field = next((label for label in places if lowered(label) == lowered(where)), None)
            said = " ".join(word_list(words))
            if field and said and f" {said} " in f" {' '.join(word_list(places[field]))} " and not any(c["field"] == field for c in claims):
                claims.append({"field": field, "value": str(words).strip()})
        for requirement in self.requirements:        # the plain places that state each value, when the reader named none
            for field, text in places.items():
                value = " ".join(word_list(closed_up(requirement["value"])))
                if value and f" {value} " in f" {' '.join(word_list(closed_up(text)))} " and not any(c["field"] == field for c in claims):
                    found = re.search(re.escape(value).replace(r"\ ", r"\W+"), lowered(text))
                    if found:
                        claims.append({"field": field, "value": found.group(0)})
                        break
        return {"claims": claims}

    def can_abstain(self) -> bool:
        """The order tool takes an abstention: when nothing fits, the order names the closest listing instead."""
        tool = self.counter.tools.get("place_test_order") or {}
        return "abstain" in ((tool.get("parameters") or {}).get("properties") or {})

    def over_budget_match(self) -> tuple[str, str] | None:
        """A listing on the shelf, in stock, that meets every requirement but costs more than the budget: the best
        unread rows over the limit are read first. The cheapest such variant, or None."""
        counter = self.counter
        if self.budget is None:
            return None
        over = [k for k in counter.row_order if k[0] not in counter.pages and k not in self.gone
                and (counter.stock.get(k) or {}).get("in_stock") is not False
                and isinstance((counter.stock.get(k) or {}).get("price"), float) and counter.stock[k]["price"] > self.budget + 1e-9]
        context = self.row_context()
        ids = list(dict.fromkeys(k[0] for k in sorted(dict.fromkeys(over), key=lambda k: -self.row_score(k, context))))[:2 * OPEN_BATCH]
        if ids and counter.has("compare") and counter.turns_left() > 2:
            counter.send([act("compare", product_ids=ids[i:i + OPEN_BATCH]) for i in range(0, len(ids), OPEN_BATCH)])
            self.hear_quietly()
        found = []
        for product_id, page in counter.pages.items():
            if not self.in_category(page):
                continue
            for variant in page.get("variants") or []:
                key = (product_id, str(variant.get("sku")))
                seen = counter.stock.get(key) or {}
                price = seen.get("price")
                if key in self.gone or seen.get("in_stock") is False or not isinstance(price, float) or price <= self.budget + 1e-9:
                    continue
                if self.requirements and all(self.settled(key, r) for r in self.requirements):
                    found.append((price, key))
        return min(found)[1] if found else None

    def shortfall_claims(self, key: tuple[str, str]) -> dict:
        """What the closest listing shows, in its own words: its price, its stock state, and for each requirement it misses
        the fact that states its own value instead."""
        seen = self.counter.stock.get(key) or {}
        page = self.counter.pages.get(key[0]) or {}
        claims: list[dict] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        if isinstance(seen.get("in_stock"), bool):
            claims.append({"field": "in_stock", "value": seen["in_stock"]})
        met = (self.verdicts.get(key) or {}).get("met") or []
        for i, requirement in enumerate(self.requirements):
            if i < len(met) and met[i]:
                continue
            for label, value in self.own_facts(page, requirement)[:1]:
                if not any(c["field"] == label for c in claims):
                    claims.append({"field": label, "value": value})
        return {"claims": claims}

    def answer(self, line: str) -> None:
        """Answer a shopper line that expects one, as a statement."""
        counter = self.counter
        if counter.over or not counter.has("message") or counter.turns_left() <= 2:
            return
        if line in self.unanswered:
            self.unanswered.remove(line)
        facts = "\n".join(self.facts_for_reply[-6:]) or "(nothing new)"
        reply = self.reader.fill(REPLY_GUIDE, f"Facts:\n{facts}\n\nThe shopper's line or lines:\n{line}", REPLY_TOOL, 300) if self.can_think() else None
        text = str((reply or {}).get("text") or "").strip().replace("?", ".")
        new, gone = [], []
        for said, (brought, dropped) in self.changed_values.items():
            if said in line:
                new += [v for v in brought if v not in new]
                gone += [v for v in dropped if v not in gone and v not in new]
        if new and not all(closed_up(v) in closed_up(text) for v in new):
            # the reply must name the new values exactly as the shopper gave them: when it does not, a plain one does
            text = f"Yes, confirmed: it now has to be {' and '.join(new)}" + (f" instead of {' and '.join(gone)}" if gone else "") + \
                   "; the rest of your request is unchanged."
        if text:
            counter.send([act("message", content=text[:400])])

    def reopen(self, asked: str, before: int) -> None:
        """When the answer to a question about `asked` told a requirement on another property (the use the question
        also asked about) and more remain to be told, the property asked about is still open: it is asked again,
        on its own."""
        fresh = self.told_props[before:]
        if asked and fresh and asked in self.asked_props and asked not in self.reasked and self.told < self.needs_expected and \
                not any(self.name_words(asked) & self.name_words(p) - {"type", "kind"} for p in fresh):
            self.reasked.add(asked)
            self.asked_props.remove(asked)
            self.ask_props = [asked] + [p for p in self.ask_props if p != asked]

    def ask_until_told(self) -> None:
        """One property at a time until the shopper reveals what they kept back (one requirement for each reason
        for buying), within a few questions; each question goes with the next searches."""
        counter = self.counter
        while self.told < self.needs_expected and self.questions_asked < self.needs_expected + QUESTIONS_OVER:
            if counter.over or not counter.has("message") or counter.turns_left() <= TURN_RESERVE + 2 or self.late():
                return
            if (self.asked_props or self.delay_first) and not self.labels_ranked:
                # the shelf's own listings show which properties sort it into kinds: read some before choosing
                opens = self.to_open(counter.group_size * OPEN_BATCH)
                if opens:
                    counter.send(opens[: counter.group_size])
                    self.hear_quietly()
            question = self.need_question()
            if not question:
                return
            asked, before = self.asked_props[-1] if self.asked_props else "", len(self.told_props)
            self.awaiting_answer = True
            counter.send([act("message", content=question[:300])] + self.next_searches(counter.group_size - 1))
            self.questions_asked += 1
            self.hear()
            self.reopen(asked, before)

    def run(self) -> list[dict]:
        self.open_episode()
        self.ask_until_told()
        # with a priority to pick by, a third fitting listing is looked for before the first cart: the pick compares them
        self.look(rounds=9, enough=3 if self.priority else 2)
        choice = self.settle()
        abstain_to = self.abstain_target(choice)
        self.place_order(abstain_to or choice, abstain_to)
        return self.counter.dialogue

    def open_episode(self) -> None:
        """The first turn: the opening question (when the request leaves something open), the shopper's own listings,
        the item's likely shelf labels and the first searches, all at once; then the answer is heard."""
        counter = self.counter
        question = self.read_request()
        calls: list[dict] = []
        if question and counter.has("message"):
            calls.append(act("message", content=question[:300]))
            self.questions_asked += 1
        if self.listed and counter.has("compare"):
            calls.append(act("compare", product_ids=list(dict.fromkeys(k[0] for k in self.listed))[:OPEN_BATCH]))
        shelves = [act("filter", category=g, max_price=self.budget, in_stock=True, k=10) for g in self.label_guesses()[:5]] if counter.has("filter") else []
        self.awaiting_answer = bool(question)
        counter.send(calls + self.next_searches(counter.group_size - len(calls) - len(shelves)) + shelves)
        known = len(self.requirements)
        self.hear()
        self.reopen(self.asked_props[-1] if self.asked_props else "", 0)
        if self.pair_fallback and len(self.requirements) == known:
            # the answer to the pair named nothing: the two were not the open point, so what was set aside as a
            # wish is a requirement again, and the properties are asked about
            self.pair_fallback = ""
            self.requirements += self.set_aside

    # ---- the cart and the order
    def holds(self, key: tuple[str, str]) -> bool:
        """The latest stock read leaves the listing buyable: not out of stock, not over the budget."""
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        return seen.get("in_stock") is not False and (self.budget is None or not isinstance(price, float) or price <= self.budget + 1e-9)

    def next_pick(self) -> tuple[str, str] | None:
        """The listing to cart next: the best fitting one not lost; for the first cart, what the shopper said counts
        first, then the request as written, then the maker they would take first. When nothing fits, or nothing
        for the first cart keeps to the request as written, the reader writes new searches first; with nothing
        fitting even then, the nearest miss."""
        while True:
            self.rank_by_priority()
            fitting = [k for k in self.ranked() if k not in self.gone]
            first = not self.counter.cart
            if not (fitting and (not first or any(self.keeps_request(k) for k in fitting))) and self.rethink_searches():
                continue
            break
        pool = fitting or [k for k in self.near_misses() if k not in self.gone]
        if first:
            pool.sort(key=lambda k: (not self.meets_chosen(k), not self.keeps_request(k), not self.of_preferred(k)))
        return pool[0] if pool and not self.counter.over else None

    def settle(self) -> tuple[str, str] | None:
        """Cart the best listing and keep it while it holds: in stock, within the budget, and still what the shopper
        wants after every line they say. A listing that fails gives way to the next pick. The listing to order, or
        the last one still in the cart, or None."""
        counter = self.counter
        for attempt in range(1, CART_TRIES + 1):
            if counter.over or counter.turns_left() < 2:
                break
            key = self.next_pick()
            if key is None:
                break
            if self.keep(key, attempt):
                return key
        alive = [k for k in counter.cart if k not in self.gone]
        return alive[-1] if alive and not counter.over else None

    def keep(self, key: tuple[str, str], attempt: int) -> bool:
        """Cart one listing with a stock read, then hold it against everything that follows before the order: the
        shopper's lines (a question gets its answer), a stale stock read (read again), lines still owed a reply."""
        counter = self.counter
        title = str((counter.pages.get(key[0]) or {}).get("title") or "")[:50]
        shown = "; ".join(f"{str((counter.pages.get(k[0]) or {}).get('title') or '')[:28]}|{self.variant_of(k).get('options')}|{[self.plain(k, r) for r in self.requirements]}|{(self.verdicts.get(k) or {}).get('by')}" for k in self.ranked()[:4])
        self.note(f"commit {attempt}: {key} {title!r} fits={self.fits(key)} turn={counter.turn} pages={len(counter.pages)} spent={self.reader.spent:.4f} calls={self.reader.calls} fitting: {len(self.ranked())}: {shown}")
        if self.fits(key):
            self.fit_seen = True        # something fit: the task is not one where nothing can
        counter.send([act("add_to_cart", product_id=key[0], sku=key[1])] + ([act("inspect_stock", product_id=key[0], sku=key[1])] if counter.has("inspect_stock") else []))
        changed = self.hear()[0]
        if not self.holds(key):
            self.pending_line = ""      # the shopper's reaction to the change is answered by choosing again
            self.lose(key)
            return False
        if changed and not self.still_wanted(key):
            return False
        if self.pending_line:
            # a question still open - asked now, or asked with a change that put an earlier listing out - is answered
            # once a listing holds
            seen = counter.stock.get(key) or {}
            self.facts_for_reply.append(f"The listing now in the cart: {title}, {seen.get('price')}, in stock.")
            self.answer(self.pending_line)
            self.pending_line = ""
            if self.hear()[0] and not self.still_wanted(key):
                return False
        if counter.turn > int((counter.stock.get(key) or {}).get("turn") or 0) and counter.has("inspect_stock") and counter.turns_left() >= 3:
            # a turn went by since the listing was last read: the market may have moved, so read it again before paying
            counter.send([act("inspect_stock", product_id=key[0], sku=key[1])])
            changed = self.hear()[0]
            if not self.holds(key):
                self.lose(key)
                return False
            if changed and not self.still_wanted(key):
                return False
        if self.unanswered and counter.has("message") and counter.turns_left() >= 3:
            self.reply_before_paying(key)
            if self.hear()[0] and not self.still_wanted(key):
                return False
        return True

    def still_wanted(self, key: tuple[str, str]) -> bool:
        """After a line that changed what the shopper wants: the listing is judged again; one that no longer fits is
        out. The listings found so far were looked for under the old values, so unless enough of them state the new
        ones firmly (two to compare with a priority, else one), the agent looks again while there is time."""
        self.judge()
        if self.fits(key):
            return True
        self.gone.add(key)
        wanted = 2 if self.priority else 1
        firm = [k for k in self.ranked() if k not in self.gone and self.firm(k)]
        if len(firm) < wanted and self.counter.turns_left() > TURN_RESERVE and not self.late():
            self.look(rounds=4, enough=wanted)
        return False

    def lose(self, key: tuple[str, str]) -> None:
        """The listing sold out or went over the budget: it is out, the shopper's next line is about that change, and
        while there is time its relatives are searched for (the same article is often listed again); with fewer than
        two firm listings left, the shelf is searched further too."""
        counter = self.counter
        self.gone.add(key)
        self.lost_titles.add(lowered((counter.pages.get(key[0]) or {}).get("title")))
        self.owed_reaction = True
        if counter.turns_left() <= TURN_RESERVE or self.late():
            return
        self.refill(key, wide=not any(self.firm(k) for k in self.ranked() if k not in self.gone))
        left = [k for k in self.ranked() if k not in self.gone]
        if len([k for k in left if self.firm(k)]) < 2 and counter.turns_left() > TURN_RESERVE and not self.late():
            self.related.clear()
            self.add_queries(self.relatives() + self.shelf_searches(6), first=True)
            self.look(rounds=3 if left else 5, enough=2 if left else 1)

    def abstain_target(self, choice: tuple[str, str] | None) -> tuple[str, str] | None:
        """The listing to name while ordering nothing, when nothing within the budget can fit: one that meets every
        requirement but the price. Never after the shopper cut the budget (then the limit, not the task, put the
        fitting listings out of reach, and an order is expected), nor once a fitting listing was carted (then the
        market took the fitting ones away)."""
        if not self.can_abstain() or self.counter.over or self.budget_cut or self.fit_seen or (choice is not None and self.fits(choice)):
            return None
        target = self.over_budget_match()
        if target:
            self.note(f"abstain: nothing within the budget fits; {target} meets everything but the price")
        return target

    def place_order(self, choice: tuple[str, str] | None, abstain_to: tuple[str, str] | None) -> None:
        """The order, tried up to three times: a refused order (the shopper has a line to read first) is followed by
        that line, and a change while the order waited means the choice must still fit, or the next one is carted."""
        counter = self.counter
        for _ in range(3):
            if choice is None or counter.over or counter.turns_left() < 1:
                return
            if self.unanswered and counter.has("message") and counter.turns_left() >= 2:
                self.reply_before_paying(choice)
            abstain = choice == abstain_to
            sent = counter.send([act("place_test_order", product_id=choice[0], sku=choice[1], abstain=True if abstain else None,
                                     justification=self.shortfall_claims(choice) if abstain else self.claims(choice))])
            if counter.over:
                return
            changed, asks = self.hear()
            if asks and self.pending_line and counter.turns_left() >= 2:
                self.answer(self.pending_line)
                self.pending_line = ""
                changed = self.hear()[0] or changed
            if changed and counter.turns_left() >= 3:
                self.judge()
                if not self.fits(choice):
                    self.gone.add(choice)
                    choice = self.recart()
            if not sent:
                return


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    errand = Errand(problem_data)
    try:
        return errand.run()
    except Exception as exc:  # the dialogue so far is still returned for diagnosis, after a last order attempt
        errand.counter.dialogue.append({"role": "assistant", "content": f"agent error: {type(exc).__name__}: {exc}"})
        try:
            errand.last_resort()
        except Exception as late:
            errand.counter.dialogue.append({"role": "assistant", "content": f"agent error at the last order: {type(late).__name__}: {late}"})
        return errand.counter.dialogue


__all__ = ["agent_main"]
