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
SEARCH_LIMIT = 80             # searches and shelf filters sent while looking (every one loads the store)
JUDGE_CARDS = 18              # listings shown to the reader in one call
JUDGE_LIMIT = 120             # listings the reader is asked about over the episode
EPISODE_SECONDS = 150.0       # after this the agent stops looking and orders what it has
TURN_RESERVE = 6              # turns kept for the cart, the stock checks and the order
OVER_REACH = 1.35             # a listing this far over the spending limit is still read: it may be the nearest when nothing fits
OPEN_REACH = 1.6              # a search row this far over the limit is still opened: the row shows one variant, another may cost less
QUERY_LENGTH = 200            # a search is cut to this many characters
THINK_TOKENS = 3000           # room for a thinking pass before the tool call is written
MIN_TOKENS = 900              # every call: a provider may think before it writes the tool call, even when asked not to
SAID_NEEDED = r"(?:(?:a|an|is|be|it|the)\s+)*(?:required|requirement|needed|necessary|essential|must|need|included|present|chosen|selected|provided)"
NOT_A_PROPERTY = re.compile(r"categor|budget|price|\bstock|availab")     # terms of the purchase, not properties of the item
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
ASK_AFTER_PAIR = 2            # questions about properties after the shopper has chosen between two wishes
ASK_AFTER_USE = 1             # questions about properties once a question about a purpose was answered with no requirement
ASK_AFTER_BOTH = 2            # questions about properties once one requirement is out and every purpose was asked about
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

_proxy = ProxyClient(timeout=90, max_retries=2)


# ----------------------------------------------------------------------------- small helpers
def lowered(text: Any) -> str:
    # (the double-prime sign after a figure is a way of writing the inch: 22'' is 22 inch)
    plain = re.sub(r"(?<=\d)\s?(?:''|\u2033|\u201d|\")(?=[\s,;:/)\]x\u00d7*-]|$)", " inch", str(text or "").lower())
    return " ".join(plain.split())


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
    if len(word) > 4 and word.endswith(("ve", "fe")):
        return word[:-2] + "f"      # (a plural in -ves and its singular in -f or -fe are one word; so is any word in -ve)
    return word


def closed_up(text: Any) -> str:
    """Amounts written as listings write them: the number and its unit as one word (3 cm -> 3cm)."""
    return re.sub(r"(\d)\s+(?=[a-z]{1,4}\b)", r"\1", lowered(text))


def same_word(one: str, other: str) -> bool:
    """Two words are the same word when they differ only by an ending (dot / dotted)."""
    if one == other:
        return True
    short, long = sorted((one, other), key=len)
    return len(short) >= 4 and not short[0].isdigit() and long.startswith(short) and len(long) - len(short) <= 2


def stems(text: Any) -> set[str]:
    return {singular(w) for w in word_list(closed_up(text))}


def label_key(text: Any) -> str:
    """A category label reduced to its words in singular form, for comparing two spellings of it."""
    return " ".join(singular(w) for w in word_list(str(text).replace("&", " and ")) if w != "and")


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
    "brand": {"type": "string", "description": "A brand the shopper requires; empty when none, and empty when a brand is only preferred or a first choice."},
    "priority": {"type": "string", "description": "What the shopper wants the most or the least of among the items that fit, in the shopper's words (for example the lowest price). Most requests name one: read the request again before leaving this empty."},
    "queries": strings("Exactly six short catalogue searches (two to five words) that could return the item: the item words with the required values, with a synonym or another spelling of a value where one exists."),
    "priority_queries": strings("When a priority is given: up to four more searches that reach the listings standing best by it; otherwise empty."),
    "doubted_pair": {"type": "array", "items": {"type": "string"}, "maxItems": 2,
                     "description": "When the shopper wishes for two things and doubts one item can have both: the two wishes, each as its value with what it describes (for example 'a leather strap', 'the Acme brand'). Empty otherwise."},
    "purposes": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                 "description": "Each use or recipient the request gives for the item, one entry each, in the shopper's words. Empty when it gives none."},
    "invitation": {"type": "string", "description": "The request's exact words that tell the assistant to put a question to the shopper before deciding (to ask, check or confirm with the shopper whether the purpose changes what the item needs); empty when the request holds no such instruction."},
    "look_at": {"type": "integer", "description": "When the request tells the assistant to open, compare or check at least some number of fitting options before choosing: that number; otherwise 0."},
    "open_properties": {"type": "array", "items": {"type": "string"}, "maxItems": 6,
                        "description": "Properties of this kind of item that the request does not settle and a shopper could still have a requirement about, the likeliest first, each in one to three plain words."},
}, ["item", "requirements", "budget", "priority", "queries", "doubted_pair", "purposes", "invitation", "look_at", "open_properties"])

HEAR_TOOL = tool_spec("hear", "What the shopper's new line changes.", {
    "adds": {"type": "array", "description": "Each requirement of the item the line states that was not known before.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string", "description": "The property's name as the shopper said it, with every word of that name (the material of one part is named with the part)."},
            "value": {"type": "string", "description": "The value alone, as the shopper said it: an amount with its unit; never the property's own name, and never a word that only says it is needed."},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The same value written another way: another spelling of the word, or the same amount in another unit. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value (a synonym; not a part or a wider kind of it). Empty when there is none."}},
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

NAMES_TOOL = tool_spec("names", "What each text names for one property.", {
    "marks": {"type": "array", "items": {"type": "string"}, "description": "One entry per text, in order: '<number> <only|among|other|none>'."}},
    ["marks"])

WORDS_TOOL = tool_spec("words", "Other words sellers use for a value.", {
    "words": {"type": "array", "items": {"type": "string"}, "description": "Up to six, the most usual first; empty when there is none."}},
    ["words"])

RANK_TOOL = tool_spec("rank", "Each listing's figure for the shopper's priority.", {
    "values": {"type": "array", "items": {"type": "string"}, "description": "One entry per listing, in order: '<number> <figure>' with the figure as a plain number in one unit for every listing, or '<number> none' when the listing's words state none."},
    "better": {"type": "string", "enum": ["higher", "lower"], "description": "Whether a higher or a lower figure is better for this priority."},
    "unit": {"type": "string", "description": "The one unit every figure is written in (empty for a plain count)."}},
    ["values", "better", "unit"])

BEST_TOOL = tool_spec("search_better", "Catalogue searches for listings that stand better by the shopper's priority.", {
    "queries": strings("Up to eight new searches of two to six words each, different from those tried."),
}, ["queries"])

SEARCH_TOOL = tool_spec("search_plan", "New catalogue searches.", {
    "queries": strings("Six new searches of two to five words each, different from those tried."),
}, ["queries"])

PLAN_GUIDE = """You prepare a purchase for a shopping assistant. Read the shopper's request and fill the plan tool from its words only.
- item: the words the shopper uses for the kind of thing, copied completely and exactly as written, with a leading word such as 'other' and any '&' or '/' kept (the store files its listings under such names); do not shorten them or replace them with a broader or narrower kind.
- requirements: every detail the request gives about the item itself is a requirement: its material, its colour, an amount with its unit (how much it holds, how big, heavy or long it is, how many pieces, how strong), its pattern, shape, style, kind or type, an occasion or use named together with the item, the make or thing the item must fit, and the like. A kind or type named for the item ('of the ... type') is such a detail too. A request for one kind of thing nearly always carries such a detail: read the request again before leaving the list empty. Name each property the way the shopper names it; when the request gives only the value, use the plain name of that kind of detail. Give the value in the fewest words: the word or amount itself (an amount with its unit in letters), without the property's name and without the kind of thing it belongs to. One phrase of the request gives one requirement; do not split it into two. The budget, being in stock and the category are not requirements, and neither is why the shopper is buying, which comes in a sentence of its own.
- doubted_pair: a request sometimes names two things the shopper would like on top of what the item must have - usually a value of some property and a brand - and doubts, or leaves open, that one item offers both, in any wording (a doubt said outright, or the second wish added as something further that may not be possible); the shopper will say which matters more when asked. Give the two wishes. Leave it empty when the request names fewer than two such wishes: a single thing the shopper would merely like (one brand that is only preferred) is neither a requirement nor a pair, and two words of the item's own name are not two wishes.
- purposes: every use or recipient the request gives (what the item is for, whom it is for); each can carry a need the shopper has not stated yet. A request often gives two - in two sentences, or in one sentence joined by a linking word, the second sometimes only as a further clause (it is for one thing 'and also' has to suit another): give each as its own entry, in the shopper's own words for it and without the linking words, each as a phrase that can follow the word 'for' (a camping trip; a child's room; a friend who is moving house). One purpose is one entry: do not cut a single purpose in two.
- invitation: some requests tell the assistant outright to put a question to the shopper before deciding - to ask, check or confirm with the shopper whether the purpose changes what the item needs. Copy those words exactly as written. A sentence that only gives the purpose, says the purpose may matter, or tells the assistant what the item must be (in stock, within the budget) holds no such instruction: leave it empty.
- priority: what the shopper wants the most or the least of among the items that fit - the most or the least of some quantity the items differ in, or the lowest price - in the shopper's words. Most requests name one, in a sentence of its own or beside the budget: do not leave it out.
- brand: only a brand the shopper requires. A brand that is only preferred, or wanted only when there is one, is not required: leave brand empty.
- look_at: some requests tell the assistant to open, compare, check or go through at least a certain number of fitting options (listings, choices) before choosing. Give that number; 0 when the request sets none.
- open_properties: what the request leaves open about the item itself, in the order to ask about it. A shopper who holds one requirement back holds back something plain, whatever the kind of item - do not judge whether it sounds usual for this item. When a stated purpose itself points to a property of the item (how it has to look, what it has to hold, weigh or work with for that purpose), that property comes first. After it the list always runs in this order, leaving out only what the request already fixes: the colour; then which type of the item it is, written as 'type of' and the item's main word or two (for a kettle: 'type of kettle'; never with 'and' or '&'); then the piece count (how many pieces come together); then the material; then the amounts this kind of item is sold by, each named by the measure itself and never by its unit. Nothing else goes ahead of these. Name each one in one to four plain words; never a loose word on its own that leaves open which measure or which kind is meant (a bare 'size' does). Leave out the quantity named as the priority, the brand, the budget, the category and stock.
- priority_queries: when the shopper names a priority, searches that reach the listings standing best by it (the item words with the kind of amount such listings write).
- written: the same value written another way - another spelling of the word (grey: gray) or the same amount in another unit or wording (1 kg: 1000g, 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- queries: searches as a shopper types them, two to five words, built from the item words and the required values. Write an amount the way listings do (the number joined to its unit)."""

HEAR_GUIDE = """You follow a shopper's conversation with a shopping assistant. Given what is already known and the shopper's new line, fill the hear tool with what the line changes.
- adds: a requirement of the item the line states (a property with its value), the property named in the shopper's words and the value in the fewest words (the word or amount itself). A purpose alone is not a property; the property the purpose calls for is. The budget, the category and being in stock are not properties. written: the same value written another way (another spelling of the word; the same amount in another unit: 1 kg is 1000g or 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- When the line gives a value without naming its property, name the property that value belongs to by what the value itself is (a colour word is a colour, what a thing is made of is its material, an amount is the measure its unit tells). The shopper may answer about another property than the one the assistant had just asked about: use the asked property's name only when the value can be a value of that property.
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
2. the fact labels of the item itself, in the order they are shown (the label most listings carry comes first), up to six, each named after its label in plain words (a label such as 'xyz_type' or 'types_of_xyz' is the 'xyz type'). Go down the shown list and skip only: a label about the seller or the offer rather than the item itself (who makes or sells it, its state, its guarantee, how it ships), a label that only holds the listing's own text or its layout, and a label that says the colour;
3. the material;
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
- only: the words name the wanted value - the same word, or another word sellers use for that same value - and no other value of that property.
- among: the words name the wanted value together with another value of that property (two colours, two materials), so it is unclear which one the item is.
- other: the words name a value of that property, but not the wanted one.
A shade or tint of the wanted colour (dark, light, off-, or a describing word before the colour) is the wanted colour; a name that joins two different colours is another colour.
The variant's words describe the exact item sold: a material they name is the item's material, whatever the title lists.
- none: the words name no value of that property.
Judge only the property asked about; sizes, codes, counts and other details are not values of it. Mark every text, in order."""

WORDS_GUIDE = """A shopper wants a certain value for a property of an item. List the other words or spellings that sellers use in listing titles and variant names for that same value: synonyms and other spellings of a word, or the same amount in other units and notations. Only words that mean the same value; not a part, a shade or a wider kind of it."""

RANK_GUIDE = """A shopper said what makes one fitting item better than another. For each listing, read from its own words (the options first, then its facts, then its title) the figure that this priority is about, and write it as a plain number in the same unit for every listing (convert where needed). The options describe the exact variant sold: when they state the figure, that is the figure, whatever the title lists (a title often lists the amounts of all its variants). Write none when the listing states no such figure; do not guess. One entry per listing, each as the listing's number and then the figure."""

BEST_GUIDE = """A shopper wants, among the store listings that meet every requirement, the one that stands best by a priority of theirs (the most of something, the largest, the longest, ...). A shopping assistant has found some fitting listings. You write catalogue searches that could reach fitting listings that stand better still.
You are given the request, the requirements, the priority, the best figures found so far with the words of the listings that carry them, and the searches already tried.
- Listings state such a figure in their title or variant name. A search that carries a figure written the way a listing writes it brings up the listings that state it - and sellers write it both joined to its unit and apart from it, which the store treats as different words. So write searches made of the item words, the required values and a figure for the priority that is better than the best found so far: several steps beyond it, one figure per search, some written joined and some apart.
- A title that lists several amounts lists the variants sold under it: when such a title shows an amount better than the best found, search that title's main words with that amount and the required values.
- Add one search with the words sellers use for the better end of that priority.
- Keep every search short, and do not repeat a search already tried."""

SEARCH_GUIDE = """A shopping assistant has not yet found a store listing that meets every requirement of the shopper. Given the request, the requirements, the searches already tried and the titles of the listings seen, write new short catalogue searches that are likely to reach a listing meeting every requirement: combine the required values with each other and with the words such listings carry in their titles, use other words sellers use for a value, and try the values without the item words. Do not repeat a search already tried."""

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
        self.shown_at: dict[tuple[str, str], int] = {}     # the turn each row was first shown at
        self.shelf: dict[tuple[str, str], str] = {}      # rows a category filter returned -> the label that returned them
        self.pages: dict[str, dict] = {}                 # product_id -> page from view / compare
        self.stock: dict[tuple[str, str], dict] = {}     # (product_id, sku) -> latest price / stock seen
        self.cart: list[tuple[str, str]] = []
        self.adds = 0                                    # listings put in the cart so far (those taken out again too)
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
                        self.shown_at[key] = self.turn
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
                    self.adds += 1
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
        self.named_as: dict[tuple, str] = {}   # (listing, property, value) -> what the reader found its words to name
        self.name_checks = 0
        self.awaiting_answer = False           # a question was just put to the shopper
        self.probed: set[str] = set()          # titles whose other variants were already searched for
        self.lenient = False                   # nothing fits by the firm rules: a value stated anywhere on the page counts
        self.shelf_cache: tuple[int, str] = (-1, "")
        self.shelf_said = ""                   # the shopper's line that repeats the limit and names the shelf
        self.kin_done: set[str] = set()        # titles whose other listings and variants were already searched for
        self.pair_fallback = ""                # asked when the answer to the pair question names nothing
        self.pair_chosen = False               # the shopper said which of the two wishes matters more
        self.told_to_ask = False               # the request tells the assistant to ask the shopper before deciding
        self.pair_question = ""                # the question that asks which of the two wishes matters more
        self.pair_wishes: list[str] = []       # the two wishes, in the request's words
        self.pair_open = False                 # the question just put is that one
        self.pair_again = False                # it drew a requirement that is neither wish: it is put once more
        self.filed_cache: dict[str, tuple[int, set[str]]] = {}     # per fact label: the entries read under it
        self.use_asked: set[int] = set()       # purposes whose requirement was asked for by the use itself
        self.use_again: set[int] = set()       # purposes to ask about once more, in other words (the answer was another purpose's)
        self.use_twice: set[int] = set()       # purposes asked about twice
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
        self.named_nearest: tuple[str, str] | None = None   # the listing named as the nearest when the order says nothing fits
        self.over_noted = False                # (the note on why no listing over the limit was named, written once)
        self.went_ahead = False                # a second listing went into the cart ahead of the choice
        self.changed_early = False             # the shopper changed a requirement or the limit before anything was in the cart
        self.moved_on = False                  # the shopper changed a requirement or the limit at any point
        self.reply_sent: tuple[int, str] | None = None  # the turn of the last reply to a change, and the lines it answered
        self.replies_again = 0
        self.spoke_of_use = False              # the question now open spoke of the use
        self.look_at = 0                       # fitting options the shopper wants opened before the choice
        self.looked_under = ""                 # the requirements under which those options were last opened

    def note(self, text: str) -> None:
        self.counter.dialogue.append({"role": "assistant", "content": f"note: {text}"})

    def late(self) -> bool:
        return monotonic() - self.started > EPISODE_SECONDS

    def can_think(self) -> bool:
        return not self.reader.out_of_budget()

    # ---- the request
    def read_request(self) -> str:
        request = self.counter.request
        # (no thinking pass: it reads the request no better, and one in ten ran to the token limit, which sent the
        # request to the slower model and used up the episode's time before anything was asked or searched)
        plan = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 900) or {}
        if not [r for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]:
            # a request without any detail of the item is rare: have the other model read it too
            again = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 700, second=True) or {}
            if again.get("requirements"):
                plan = {**plan, **{k: v for k, v in again.items() if v}} if plan else again
        self.item = str(plan.get("item") or "").strip()
        self.category = str(plan.get("category") or "").strip()
        self.requirements = [self.requirement(r) for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        self.requirements = [dict(r, stated=True) for r in self.requirements if not NOT_A_PROPERTY.search(lowered(f"{r['property']} {r['value']}"))]
        # what the item is for is no property of it: a value the request says once, inside a purpose it gives, is the
        # use itself (it can hold a requirement back, and that one is asked for)
        uses = [lowered(" ".join(str(w).split())) for w in plan.get("purposes") or [] if str(w).strip()]
        whole = lowered(" ".join(request.split()))
        rest = whole
        for use in uses:
            rest = rest.replace(use, " ")

        def same_root(a: str, b: str) -> bool:
            """Two forms of one word (paint, painting): they share most of their letters from the start."""
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
        self.listed = [(m.group(1), m.group(2)) for m in re.finditer(r"(\d{6,})::([\w-]+)", request)]
        self.priority = self.grounded_priority(plan.get("priority"), request)
        if self.priority:
            # what the shopper wants the most or least of is the priority between fitting items, not a value an item
            # must state: a "requirement" that only says the priority again would be met by nothing
            most = self.name_words(self.priority)
            self.requirements = [r for r in self.requirements if re.search(r"\d", r["value"])
                                 or not (self.name_words(r["value"]) - {"the", "a", "an"} <= most and self.name_words(r["property"]) & most)]
        self.add_queries(self.value_queries() + list(plan.get("queries") or []))
        if self.priority:
            # the listings that stand best by the shopper's priority are searched for from the start
            self.add_queries([q for q in plan.get("priority_queries") or [] if isinstance(q, str)][:4])
        pair = [" ".join(str(w).split()) for w in plan.get("doubted_pair") or [] if str(w).strip()][:2]
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
        # (an instruction to ask counts only in the request's own words)
        quoted = " ".join(str(plan.get("invitation") or "").split()).strip(" .\"'")
        asking = [w for w in word_list(lowered(quoted)) if len(w) >= 4]
        of_a_purpose = sum(any(same_root(w, u) or w == u for p in purposes for u in word_list(lowered(p))) for w in asking)
        apart = (len(quoted.split()) >= 2 and self.plain_quotes(quoted) in self.plain_quotes(request)
                 and lowered(quoted.split()[0]) not in ("who", "whom", "whose", "which", "that")
                 and not any(self.within(quoted, p) or self.within(p, quoted) for p in purposes)
                 and of_a_purpose * 2 < max(1, len(asking)))
        # (the request itself tells the assistant to ask, check or confirm with the shopper)
        asks_me = r"\b(?:ask|check|confirm|consult|tell)\b[^.?!]{0,24}\bme\b|\blet me know\b|\bif that\b|\bjust say\b"
        self.told_to_ask = bool(re.search(asks_me, self.plain_quotes(request)))
        # (a clause that asks to be asked is an instruction, never a purpose: "if that means the specs change, tell me")
        # (nor is the sentence that names the two wishes and doubts both can be had: "I'd like X and Y, though ...")
        of_the_pair = any(self.content_words(w) and self.content_words(w) <= self.content_words(quoted) for w in pair)
        if apart and len(purposes) < 2 and not re.search(asks_me, self.plain_quotes(quoted)) and not of_the_pair:
            # a clause the reader took for an instruction ("it needs to follow an exact list my partner wrote down")
            # is one more purpose of the request: what it is for, from its first article or -ing word on
            words = quoted.split()
            start = next((i for i, w in enumerate(words) if lowered(w) in ("a", "an", "the", "my", "our") or (lowered(w).endswith("ing") and len(w) > 5)), None)
            if start is not None and len(words) - start >= 3:
                purposes = purposes + [" ".join(words[start:])]
                self.purposes = purposes
                apart = False
        if len(purposes) < 2:
            purposes = purposes + self.unread_purposes(request, purposes, pair)[: 2 - len(purposes)]
            self.purposes = purposes
        self.invited = apart and not purposes     # (a request that gives its purpose is asked what that purpose needs)
        # each purpose the request gives can hold one requirement back
        self.needs_expected = max(1, min(2, len(purposes)))
        # (one property to a question. The colour goes first, whether the reader thought of it or not: it is what a
        # shopper most often holds back, and a request that fixes it takes it out of the line)
        self.queue_properties(["colour"] + [w for w in plan.get("open_properties") or [] if isinstance(w, str)])
        self.to_ask.sort(key=lambda name: not re.search(r"colou?r", name))
        self.first_thoughts = self.to_ask[:4]
        # (both wishes must stand in the request's own words - at least half of each, a telling word among them: a
        # reader's guess is no wish, a reader's word for the property beside the shopper's value is)
        said = set(word_list(request))
        wishes = [[x for x in word_list(w) if x not in ("the", "a", "an", "of", "as", "its", "it", "to", "be", "with")] for w in pair]
        spoken = all(words and sum(x in said for x in words) * 2 >= len(words)
                     and any(x in said and (len(x) >= 3 or x[0].isdigit()) for x in words) for words in wishes)
        if len(pair) == 2 and spoken and len(" ".join(pair).split()) <= 16 and not self.mistaken_pair(pair, purposes):
            # the shopper wants one of two things and says which when asked: that is an open point of its own;
            # until then neither wish is a firm requirement
            wished = lowered(" ".join(pair))
            self.set_aside = [r for r in self.requirements if lowered(r["value"]) in wished]
            self.requirements = [r for r in self.requirements if lowered(r["value"]) not in wished]
            self.pair_fallback = PLAIN_QUESTION
            if self.brand and set(word_list(self.brand)) <= set(word_list(wished)):
                self.brand = ""         # (one of the two wishes: required only once the shopper says it matters more)
            self.pair_wishes = list(pair)
            self.pair_question = f"Which matters more to you: {pair[0]} or {pair[1]}?"
            return self.pair_question
        if self.invited or not self.to_ask:
            # the request asks to be asked about the purpose (or leaves nothing else to ask about)
            return PLAIN_QUESTION
        return self.next_question()

    # ---- finding out what the request leaves open
    def beside_a_purpose(self, request: str, requirement: dict, uses: list[str]) -> str:
        """A value the request says only in the sentence that gives a purpose, tied to a use by an -ing word before it
        ("... and it has to go with brightening a small hallway"): the use itself, in its own words. "" when the
        value is said where the item or an amount is, or with no purpose beside it: then it is a requirement."""
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
        """What the purchase is for, has to match or follow, where the reader of the request left it out: a clause that
        states no requirement, wish, limit or priority and ties the purchase to something by a linking word, from the
        first article or -ing word after that word on ("it has to follow a plan someone made" -> "a plan someone made")."""
        known = [self.content_words(p) for p in purposes]
        taken = [self.content_words(r["value"]) for r in self.requirements] + [self.content_words(w) for w in pair]
        priority = self.content_words(self.priority)
        named = self.content_words(self.item) | self.content_words(self.category) | {singular(w) for w in word_list(self.item) if len(w) >= 3}
        out: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(request).replace("\u2019", "'").split())):
            lead = re.match(r"(?:since|because|as)\s+(?:i'm|i am|we're|we are)\s+([^,]{8,80}),", lowered(sentence))
            if lead and lowered(lead.group(1)).split()[0].endswith("ing"):
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
                        continue        # (the thing wanted itself: "looking for a kettle")
                    if not any(self.within(found, o) for o in out):
                        out.append(found)
                        known.append(self.content_words(found))
                    break
        if out:
            self.note(f"purposes the request gives besides: {out}")
        return out

    def queue_properties(self, names: list[str], first: bool = False, compared: bool = False) -> None:
        """Add properties to ask about: plain names of one to four words, none twice, none already settled or asked
        (compared: the quantity the shopper's priority compares may be among them)."""
        fresh: list[str] = []
        settled = [self.name_words(r["property"]) for r in self.requirements + self.set_aside]
        taken = [self.name_words(p) for p in self.asked + ([] if first else self.to_ask)]
        for name in names:
            raw = lowered(str(name).replace("_", " "))
            if re.search(r"[&/,+]|\b(?:and|or)\b", raw):
                continue            # two things joined: the shopper would hear two questions
            text = " ".join(word_list(raw))
            words = self.name_words(text)
            if not words or len(text.split()) > 4 or NOT_A_PROPERTY.search(text) or "brand" in words:
                continue
            if any(len(self.content_words(text) & self.content_words(purpose)) >= 2 for purpose in self.purposes):
                continue            # words of a purpose the request gives: a clause of the request, not a property
            if not compared and self.content_words(text) & self.content_words(self.priority):
                continue            # the quantity the shopper wants most or least of is seldom a requirement held back
            if any(words == other for other in taken + [self.name_words(f) for f in fresh]):
                continue
            if any(words <= other or other <= words for other in settled if other):
                continue            # the shopper already fixed this property
            # one name inside another (lid type, pot lid type): the shopper may know the property by either, so
            # both are asked about, one beside the other where the first is still in line
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
        """The properties in line that were not asked about and are not known."""
        known = self.asked + [r["property"] for r in self.requirements + self.set_aside]
        return [n for n in self.to_ask if not any(self.name_words(n) == self.name_words(other) for other in known)]

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
        by_use = self.use_question()
        if by_use:
            return by_use
        name = self.next_property()
        self.current_names, self.leading = ([name] if name else []), None
        self.property_asks += bool(name)
        self.current_ask, self.current_other = name, ""
        return ASK_ALONE.format(self.spoken(name)) if name else ""

    def use_question_left(self) -> bool:
        """A purpose of the request has not been asked about yet."""
        return any(i not in self.purposes_met and (i not in self.use_asked or i in self.use_again) and self.said_back(i)
                   for i in range(min(2, len(self.purposes))))

    def use_question(self) -> str:
        """What the item must have for a purpose the request gives: asked once for each purpose (two at most) that
        has no requirement yet, before any property - until such a question is answered with nothing."""
        indices = range(min(2, len(self.purposes)))
        # (each purpose once; then, in other words, one whose question drew another purpose's answer)
        for again in (False, True):
            for index in indices:
                said = self.said_back(index)
                if index in self.purposes_met or not said or (index not in self.use_again if again else index in self.use_asked):
                    continue
                self.use_asked.add(index)
                if again:
                    self.use_again.discard(index)
                    self.use_twice.add(index)
                self.current_names, self.leading = [], index
                self.current_ask = self.current_other = ""
                # (an item named by two things joined would be heard as two things asked about)
                plain = bool(self.item) and len(self.item.split()) <= 3 and not re.search(r"[&/,+]|\b(?:and|or)\b", self.item)
                question = (ASK_BY_USE_AGAIN if again else ASK_BY_USE).format(it=f"the {self.item}" if plain else "it", purpose=said)
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
        """The purpose at that position as it is said back to the shopper: its main clause, in the second person."""
        text = " ".join(str(self.purposes[index]).replace("\u2019", "'").split()).strip(" .\"")
        head = text.split(",")[0]
        text = head if len(head.split()) >= 3 else text
        words = text.split()
        while words and lowered(words[0]) in ("for", "to", "as", "it's", "its"):
            words = words[1:]
        out = [TO_SHOPPER.get(lowered(w), w) for w in words[:16]]
        return " ".join(out) if len(out) >= 2 else ""

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

    def put(self, question: str) -> dict:
        """The message that puts one question to the shopper; its answer is read by the next hear()."""
        self.last_question = question
        self.pair_open = bool(self.pair_question) and question == self.pair_question
        self.plain_asked = self.plain_asked or question == PLAIN_QUESTION
        self.spoke_of_use = question == PLAIN_QUESTION or USE_WORDS in question
        self.questions_asked += 1
        self.awaiting_answer = True
        return act("message", content=question[:400])

    def listen(self) -> tuple[bool, bool]:
        """Take in what the shopper said after a turn that may have put a question; silence answers it with nothing."""
        if self.awaiting_answer and not self.counter.heard:
            self.awaiting_answer = False
            self.answered("", 0, spoke_of_else=False)
        return self.hear()

    def wants_to_ask(self) -> bool:
        """Something may still be held back, and there is room to ask."""
        counter = self.counter
        if self.needs_found and self.questions_asked >= self.first_found_at + ASK_MORE:
            return False            # one requirement came out; a further purpose earns a few more questions, no more
        if self.narrow > ASK_NARROW:
            return False            # the properties asked about widely were gone over, two at a time
        if self.pair_chosen and not self.needs_found and self.questions_asked > ASK_AFTER_PAIR:
            return False            # after a doubted pair often nothing is held back: only the likeliest are asked about
        if (self.use_drew_nothing and self.property_asks >= (0 if self.pair_chosen else ASK_AFTER_USE) and not self.use_question_left()
                and not self.plain_pending):
            return False            # asked what its purpose needs, a shopper who names nothing mostly holds nothing back
                                    # (and one who has just chosen between two wishes, nothing at all)
        if self.needs_found and self.use_asked and self.property_asks >= ASK_AFTER_BOTH and not self.use_question_left():
            return False            # one requirement is out and each purpose was asked what it needs: a couple of properties, no more
        return (self.needs_found < self.needs_expected and self.questions_asked < ASK_LIMIT and counter.has("message")
                and not counter.over and counter.turns_left() > ASK_TURNS and not self.late()
                and not [k for k in counter.cart if k != self.stand_in])

    def shelf_properties(self) -> str:
        """What the store shows about listings of this kind: some of the listings that meet everything known so
        far, and each fact label with the number of listings that carry it and its most frequent values."""
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
        """The units in which the names and options of this shelf's listings state amounts, for those in which the
        listings differ: (unit, listings that state one, different figures), the unit most listings state first.
        A unit the shopper's own amounts are in is left out: that amount is known."""
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
        """A unit as one word, whatever its number: a short unit takes its plural like any word (pcs is pc, kgs is kg)."""
        return word[:-1] if len(word) == 3 and word.endswith("s") else singular(word)

    def read_shelf(self) -> None:
        """Once listings of the wanted kind are read: have the reader choose what to ask about from the request and
        from what those listings differ in."""
        if self.shelf_reads >= 4 or not self.counter.pages or not self.can_think() or (self.shelf_reads and len(self.open_names()) >= ASK_WIDE):
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
            # the reader has seen the listings: its order replaces what was thought of before. The colour stays
            # first (what a shopper most often holds back shows on no shelf); then the amounts in which this
            # shelf's listings differ and the facts most of its listings carry; then the other first thoughts
            first = [n for n in self.to_ask if n in self.first_thoughts]
            colour = [n for n in first if re.search(r"colou?r", n)]
            listed = [n for n in names if isinstance(n, str)]
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
            if self.cart_early() or self.reply_early():
                opening = True
                continue
            if counter.pages:
                self.read_shelf()
            if self.questions_asked and not counter.pages and counter.rows and self.held_back < 2:
                # one question is out and no listing is read yet: the next one is chosen with the shelf in view
                self.held_back += 1
                question = ""
            elif self.plain_pending or (not self.needs_found and not self.plain_asked and not self.use_unheard
                                        and not self.purposes and self.questions_asked >= ASK_LIMIT - 1):
                # (the last question, when no property drew a requirement out, is about the purpose itself)
                question, self.plain_pending = PLAIN_QUESTION, False
            else:
                question = self.next_question()
                if not question and counter.pages and self.shelf_reads < 4:
                    self.read_shelf()   # the list ran out: the reader looks again, knowing what was asked
                    question = self.next_question()
                if not question and counter.pages and self.go_over():
                    question = self.next_question()
            if not question and counter.pages and not self.needs_found and not self.plain_asked and not self.use_unheard and not self.purposes:
                question = PLAIN_QUESTION   # no property drew a requirement out: the purpose itself is asked about once
            if not question:
                if counter.pages or not counter.rows:
                    break
                question = ""           # nothing to ask yet: read listings first, then choose
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

    def cart_early(self) -> bool:
        """As soon as listings are read, one that meets the request as first written goes into the cart, in place of
        the choice. What the shopper changes and what the store does at the first cart come out then, with most of
        the episode still ahead: the questions, the searches and the choice all work on what is wanted in the end.
        True when the turn was spent on it."""
        counter = self.counter
        if (counter.cart or self.early_tries >= 3 or not counter.pages or counter.over or self.late()
                or not counter.has("add_to_cart") or counter.turns_left() < ASK_TURNS + 4):
            return False
        if self.pair_fallback or self.pair_chosen or self.set_aside:
            return False            # (two wishes the shopper doubts between: what the request first asked for is not settled)
        self.early_tries += 1
        self.judge()
        # (the first cart is held to the request as first written, value by value as the store files them: only a
        # listing that states each of those where it counts, with nothing on its page against it, goes in this early)
        meeting = [k for k in self.verdicts if self.affordable(k) and self.meets_opening(k)]
        keys = [k for k in meeting if all(self.reliance(k, r) >= FIRM_FIT for r in self.opening)]
        if not keys and self.early_tries >= 2:
            keys = meeting          # (none states every value that firmly, and more listings have been read since)
        if not keys:
            return False

        shelf = [p for p in counter.pages.values() if self.in_category(p)] or list(counter.pages.values())
        seen_in: dict[str, int] = {}
        for page in shelf:
            for word in self.content_words(str(page.get("title") or "")):
                seen_in[word] = seen_in.get(word, 0) + 1

        def plainly(key: tuple[str, str]) -> tuple:
            # the listing that states the request most plainly; of several, the one most like the rest of the shelf
            # (an odd listing may be filed there without being the kind asked for), then the cheapest (the least to
            # lose: it is taken out again when the choice is made)
            price = (counter.stock.get(key) or {}).get("price") or 0.0
            words = self.content_words(str((counter.pages.get(key[0]) or {}).get("title") or ""))
            usual = sum(seen_in.get(w, 0) for w in words) / max(1, len(words))
            return (-sum(self.plain(key, r) + self.exact(key, r) for r in self.opening), -round(usual, 1), price)
        # (what the store does at the first cart falls on that listing: of two that state the request as plainly, the
        # one that does not meet what the shopper has said since goes in - it could not have been the choice)
        wanted_now = set(self.ranked())
        key = min(keys, key=lambda k: plainly(k)[:1] + (k in wanted_now,) + plainly(k)[1:])
        title = str((counter.pages.get(key[0]) or {}).get("title") or "")[:50]
        self.note(f"early cart: {key} {title!r} {self.variant_of(key).get('options')!r} turn={counter.turn} of {len(keys)} meeting the request as written")
        group = [act("add_to_cart", product_id=key[0], sku=key[1])]
        if counter.has("inspect_stock"):
            group.append(act("inspect_stock", product_id=key[0], sku=key[1]))
        group += self.next_searches(counter.group_size - len(group))
        counter.send(group)
        if key not in counter.cart:
            return True
        self.stand_in = key
        seen = counter.stock.get(key) or {}
        price = seen.get("price")
        self.hear()
        if seen.get("in_stock") is False or (self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9):
            self.gone.add(key)
            self.owed_reaction = True
            self.pending_line = ""          # the shopper's reaction to the change is answered by the choice
        return True

    def settle_early(self) -> None:
        """When the asking ended before anything went into the cart (the shopper answered at once, or nothing was
        left to ask): read the first listings, put the stand-in in the cart and reply to what that draws."""
        counter = self.counter
        for _ in range(3):
            if counter.cart or counter.over or self.early_tries >= 3 or counter.turns_left() < ASK_TURNS + 4 or self.late():
                break
            if not counter.pages or self.early_tries:
                # (no listing read yet, or none of those read meets the request as written: read further first)
                more = self.to_open(counter.group_size * OPEN_BATCH)[: counter.group_size] or self.next_searches(counter.group_size)
                if not more:
                    break
                counter.send(more)
                self.hear_quietly()
            self.cart_early()
        self.reply_early()

    def reply_early(self) -> bool:
        """The lines that changed a requirement while the stand-in went into the cart get their reply at once, in a
        turn that also reads or searches. True when the turn was spent on it."""
        counter = self.counter
        if not self.unanswered or not counter.cart or counter.over or not counter.has("message") or counter.turns_left() <= ASK_TURNS:
            return False
        lines, self.unanswered = self.unanswered[-4:], []
        said = "\n".join(f"- {line}" for line in lines)
        if not self.holds_now(said)[0]:
            return False            # (no requirement changed: nothing is owed)
        room = counter.group_size - 1
        work = self.to_open(room * OPEN_BATCH)[:room] or self.next_searches(room)
        turn = counter.turn
        self.answer(said, beside=work)
        if counter.turn == turn:
            return False
        self.pending_line = ""
        self.hear()
        return True

    @staticmethod
    def content_words(text: str) -> set[str]:
        return {singular(w) for w in word_list(text) if len(w) >= 4} - {"that", "what", "with", "this", "from", "have", "them", "they", "their"}

    def grounded_priority(self, priority: Any, source: str) -> str:
        """What the shopper wants most or least of, kept only when the shopper's own words carry it: a telling
        word of it stands in the text it was read from (every word of it, when it is about the price - the
        cheapest is wanted only when the shopper says so). A reader's guess is no priority."""
        text = " ".join(str(priority or "").split())
        words, have = self.content_words(text), self.content_words(source)
        if not words:
            return ""
        if re.search(r"\b(cheap|price|cost|expensive|afford)", lowered(text)):
            return text if words <= have else ""
        return text if words & have else ""

    def mentions(self, line: str, requirement: dict) -> bool:
        """The line speaks of this requirement: it names the property or carries the value's own words."""
        said = stems(line) | self.name_words(line)
        named = self.name_words(requirement["property"])
        if named and named <= said:
            return True
        return any(set(form.split()) <= said for form in self.forms(requirement) if form)

    @staticmethod
    def plain_quotes(text: Any) -> str:
        """Text with its quotation marks written one way, for finding words copied from the request in the request."""
        return lowered(text).replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')

    def within(self, inner: str, outer: str) -> bool:
        """Every content word of one phrase stands in the other."""
        words = self.content_words(inner)
        return bool(words) and words <= self.content_words(outer)

    def is_wish(self, value: str) -> bool:
        """The value is one of the two wishes the request doubts it can have both of (every word of it stands in one)."""
        words = {singular(w) for w in word_list(value)}
        return bool(words) and any(words <= {singular(w) for w in word_list(wish)} for wish in self.pair_wishes)

    def mistaken_pair(self, pair: list[str], purposes: list[str]) -> bool:
        """A pair entry that shares a word with a purpose, or two with the priority, is not a wish for the item."""
        said = [self.content_words(p) for p in purposes]
        priority = self.content_words(self.priority)
        named = self.content_words(self.item) | self.content_words(self.category)
        if all(self.content_words(w) and self.content_words(w) <= named for w in pair):
            return True             # both stand in the item's own name: one kind of thing, not two wishes
        return any(any(self.content_words(w) & s for s in said) or len(self.content_words(w) & priority) >= 2 for w in pair)

    @staticmethod
    def requirement(raw: dict) -> dict:
        def texts(name: str) -> list[str]:
            return [str(w).strip() for w in raw.get(name) or [] if str(w).strip()][:6] if isinstance(raw.get(name), list) else []
        value = re.sub(r"(\d)\s*(?:\"|\u201d|\u2033|'')", r"\1 inch", str(raw.get("value") or "").strip())      # 5" is 5 inch
        name = lowered(raw.get("property"))
        if re.fullmatch(r"\d+(?:\.\d+)?", value) and re.fullmatch(r"[a-z]{1,3}", name.strip()):
            value = f"{value} {name.strip()}"       # (a short unit read as the property's name - 15 v - is the value's unit)
        filed = not re.search(r"\d", value) and not re.search(r"colou?r|material", name)
        written = texts("written")
        written += [form for form in Errand.turned(value) if closed_up(form) not in {closed_up(w) for w in written}]
        return {"filed": filed, "property": str(raw.get("property") or "").strip(), "value": value,
                "written": written, "alike": texts("alike")}

    @staticmethod
    def turned(value: str) -> list[str]:
        """An amount in a metric unit, written in the unit a thousand times larger or smaller when that gives a plain
        figure (1500 ml is 1.5l, 4 kg is 4000g): a listing states the amount in either."""
        found = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([cmk]?[lgm])", lowered(value).strip())      # (litres, grams and metres only)
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
            text = lowered(" ".join(str(query or "").split())[:QUERY_LENGTH])
            if text and text not in self.sent and text not in self.queries and text not in fresh:
                fresh.append(text)
        self.queries = fresh + self.queries if first else self.queries + fresh

    def values(self) -> list[str]:
        """Each required value as a search word, in the shopper's own spelling (the request and the shopper's lines
        give the store's own values): an amount closed up, without a word that only repeats the property's name."""
        out = []
        for requirement in self.requirements:
            named = self.name_words(requirement["property"])
            words = word_list(closed_up(requirement["value"]))
            if re.search(r"\d[.,]\d", requirement["value"]):
                # (a figure with a decimal point stays one word with its unit, as a listing writes it: 1.5 l is 1.5l)
                words = [w.strip(".,;:()") for w in closed_up(requirement["value"]).split() if w.strip(".,;:()")]
            kept = [w for w in words if singular(w) not in named]
            if kept and not re.search(r"\d", " ".join(kept)) and all(len(w) <= 3 for w in kept):
                kept = []       # ("no lid" under "lid type": the word left over says nothing alone)
            out.append(" ".join(kept or words))
        return out

    def item_words(self) -> list[str]:
        return [w for w in word_list(self.category or self.item) if len(w) >= 3 and w not in ("and", "the", "for", "with", "other")]

    def value_queries(self) -> list[str]:
        """The required values with the item words: every value, the newest alone, and each item word in turn."""
        values, item = self.values(), self.category or self.item
        label = self.store_label()
        if label and values and lowered(label) != lowered(item):
            self.add_queries([f"{' '.join(values)} {label}", f"{values[-1]} {label}"])
        if not values or not item:
            return [item] if item else []
        every, newest = " ".join(values), values[-1]
        out = [f"{every} {item}", f"{newest} {item}", f"{values[0]} {item}"]
        head = (self.item_words() or [""])[-1]
        for requirement in self.requirements:       # the same search in the value's other spelling
            for other in (requirement.get("written") or [])[:3]:
                if closed_up(other) != closed_up(requirement["value"]):
                    turned = " ".join(closed_up(other if r is requirement else r["value"]) for r in self.requirements)
                    out.append(f"{turned} {item}")
                    if head and len(item.split()) > 1 and re.search(r"\d", other) and other in self.turned(requirement["value"]):
                        out.append(f"{turned} {head}")      # (an amount in its other unit, with the item's plain word alone)
        out += [f"{every} {w}" for w in self.item_words()[:3]]
        if len(values) >= 2:
            out.append(every)               # the values alone: listings that state them all, whatever the title calls the item
        out += [f"{newest} {w}" for w in self.item_words()[:2]]
        earlier = " ".join(values[:-1])
        whole = " ".join(word_list(closed_up(self.requirements[-1]["value"])))
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
            if closed_up(other) != closed_up(self.requirements[-1]["value"]):
                out.append(" ".join(f"{earlier} {closed_up(other)} {item}".split()))
        return out

    def shelf_queries(self, count: int) -> list[str]:
        """Words the store's own listings of this category carry most often, each with the required values:
        a search shows ten rows, so other words reach other listings of the same shelf."""
        titles = [self.counter.rows[k].get("title") for k in self.counter.shelf if k in self.counter.rows]
        titles += [p.get("title") for p in self.counter.pages.values() if self.in_category(p)]
        if len(titles) < 3 or not self.requirements:
            return []
        known = set(self.item_words()) | {w for v in self.values() for w in word_list(closed_up(v))}
        counts: dict[str, int] = {}
        for title in titles:
            for word in set(word_list(title)):
                if len(word) >= 4 and not word.isdigit() and word not in known:
                    counts[word] = counts.get(word, 0) + 1
        common = [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2]
        chosen = common[self.shelf_words_used: self.shelf_words_used + count]
        self.shelf_words_used += len(chosen)
        every, newest = " ".join(self.values()), self.values()[-1]
        head = (self.item_words() or [""])[-1]
        # (with the item's own word too for half of them: a listing may name the kind of thing in other words - tab, pad)
        out = ([f"{w} {every}" for w in chosen] + [f"{w} {every} {head}" for w in chosen[: count // 2]]
               + [f"{w} {newest}" for w in chosen[: count // 2]])
        apart = re.sub(r"(\d)([a-z])", r"\1 \2", every)
        if apart != every:
            # a listing writes an amount closed up or with a space (10pcs, 10 pcs), and a search finds the spelling it is given
            out[1:1] = [f"{w} {apart}" for w in chosen[:3]]
        return out

    def kin_titles(self) -> list[str]:
        return list(dict.fromkeys(lowered((self.counter.pages.get(k[0]) or {}).get("title")) for k in self.verdicts if self.fits_words(k)))

    def kin_pending(self) -> bool:
        return len(self.kin_done) < 4 and any(title not in self.kin_done for title in self.kin_titles())

    def kin_queries(self) -> list[str]:
        """The required values with the title of a fitting listing: so specific a search returns that article's
        other variants and its other listings."""
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
                out.append(f"{options} {' '.join(words[:10])}")     # the listing's own words for this variant
            brand = lowered(page.get("brand"))
            if brand and brand not in ("none", "no brand", "oem"):
                out.append(f"{every} {brand} {self.item}")
            out.append(f"{every} {' '.join(words[:2])}")
            elsewhere = {w for row in self.counter.rows.values() if lowered(row.get("title")) != title for w in word_list(row.get("title"))}
            own = sorted((w for w in dict.fromkeys(words) if len(w) >= 5 and not w[0].isdigit() and w not in elsewhere), key=len, reverse=True)
            if own:
                out.append(own[0])          # a word no other listing carries leads straight to this article's variants
        return out

    def variant_queries(self) -> list[str]:
        """A search shows one variant of a listing. When a listing meets everything except what differs between
        its variants (the options shown name another colour or amount), the wanted variant may stand behind the
        one shown: the listing's title with the wanted value brings it up."""
        out = []
        for key, verdict in list(self.verdicts.items()):
            if len(self.probed) >= 4:
                break
            if not verdict.get("kind_ok") or verdict.get("by") != "words" or not self.requirements:
                continue
            levels = [self.plain(key, r) for r in self.requirements]
            off = [r for r, level in zip(self.requirements, levels) if level < 1]
            rest = [level for r, level in zip(self.requirements, levels) if level >= 1]
            if not off or len(rest) != len(self.requirements) - len(off):
                continue
            if any(self.kind(r) not in ("colour", "amount") for r, level in zip(self.requirements, levels) if level < 1):
                continue
            title = lowered((self.counter.pages.get(key[0]) or {}).get("title"))
            if not title or title in self.probed:
                continue
            self.probed.add(title)
            words = " ".join(word_list(title)[:10])
            for requirement in off:
                for word in [closed_up(requirement["value"])] + list(requirement.get("alike") or [])[:2]:
                    out.append(f"{word} {words}")
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
        wanted = {label_key(self.category), label_key(self.item)} - {""}
        for page in self.counter.pages.values():
            for label in page.get("category_path") or []:
                if label_key(label) in wanted:
                    return str(label)
        return ""

    def shelf_filters(self, count: int = 4) -> list[dict]:
        """The category at several price ceilings: each ceiling shows other listings of the shelf."""
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
        """The shelf under the present budget, and under the brand the shopper requires, to send with the next searches."""
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
            if query in self.sent:
                continue
            self.sent.add(query)
            out.append(act("search", query=query, k=10, in_stock=True))
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
                heard = self.listed_fields(self.reader.fill(HEAR_GUIDE, said, HEAR_TOOL, 1500))

                swap = re.search(r"\binstead of\b", lowered(line))

                def real_adds(reading: dict) -> list[dict]:
                    """The requirements a reading found, without those that cannot be: the line's own word for
                    "this is needed" taken as the value, or - on a line that puts one value in place of another -
                    the value given up taken as the new one (it stands only after "instead of")."""
                    kept = []
                    for a in reading.get("adds") or []:
                        value = lowered(a.get("value")) if isinstance(a, dict) else ""
                        if not isinstance(a, dict):
                            continue
                        if re.fullmatch(SAID_NEEDED, value):
                            continue
                        named = {w for n in self.asked[-4:] for w in self.name_words(n)}
                        if self.awaiting_answer and value and (self.name_words(value) <= named or self.name_words(value) == self.name_words(self.item)):
                            continue        # (the property's own name, or the item's whole name, read as its value; one
                                            # word of a longer item name - the primer of "paint, primer & solvent" - is a value)
                        if swap and value and value in lowered(line)[swap.end():] and value not in lowered(line)[:swap.start()]:
                            continue
                        kept.append(a)
                    return kept
                found = [a for a in heard.get("adds") or [] if isinstance(a, dict)]
                if len(real_adds(heard)) < len(found) or (self.awaiting_answer and not found):
                    # the first reading found no requirement in a line that answers the question just asked, or one
                    # that cannot be: the line is read once more
                    again = self.listed_fields(self.reader.fill(HEAR_GUIDE, said, HEAR_TOOL, 1500, second=True))
                    if real_adds(again):
                        heard = again
                if any(isinstance(a, dict) and re.fullmatch(SAID_NEEDED, lowered(a.get("value"))) for a in heard.get("adds") or []):
                    # both readings took the line's word for "this is needed" as the value: the value then stands at the
                    # end of what was read as the property's name ("the handle type folding must be included")
                    mended = []
                    for a in heard["adds"]:
                        words = str(a.get("property") or "").split() if isinstance(a, dict) else []
                        if words and re.fullmatch(SAID_NEEDED, lowered(a.get("value"))) and len(words) >= 2 and self.says_value(line, words[-1]):
                            a = dict(a, property=" ".join(words[:-1]), value=words[-1])
                        mended.append(a)
                    heard["adds"] = mended
                if heard.get("adds") and not real_adds(heard) and self.awaiting_answer:
                    # both readings took the label for the value. A shopper may say the value straight after the
                    # property's name: what is left of that phrase without the name's own words is the value
                    said_so = re.search(r"\bthe ([a-z0-9 \-]+?) (?:is|are) (?:required|needed|necessary|essential|a must)\b", lowered(line))
                    named = self.name_words(self.item) | {w for n in self.asked[-4:] for w in self.name_words(n)}
                    left = [w for w in word_list(said_so.group(1)) if singular(w) not in named and w not in ("of", "a", "an")] if said_so else []
                    asked_for = next((n for n in reversed(self.asked[-4:]) if self.name_words(n) & self.name_words(said_so.group(1))), "") if said_so else ""
                    if left and len(left) <= 3 and asked_for:
                        heard["adds"] = [{"property": asked_for, "value": " ".join(left)}]
                if heard.get("adds"):
                    heard["adds"] = real_adds(heard)
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
            for name in heard.get("drops") or []:
                # (a line withdraws only what it speaks of: a reader's slip on a line about stock drops nothing)
                kept = [r for r in self.requirements if lowered(r["property"]) != lowered(name) or not self.mentions(line, r)]
                changed = changed or len(kept) != len(self.requirements)
                replaced += len(self.requirements) - len(kept)
                withdrawn += [r for r in self.requirements if r not in kept]
                self.requirements = kept
            # (a new spending limit read as an added value is no value put in place of a requirement: a line that names
            # a limit and says what is STILL wanted takes nothing back)
            added = [lowered(a.get("value")) for a in heard.get("adds") or [] if isinstance(a, dict) and str(a.get("value") or "").strip()
                     and not NOT_A_PROPERTY.search(lowered(f"{a.get('property')} {a.get('value')}"))]
            if heard.get("drops") and added and not withdrawn:
                # the line withdraws something under a name no requirement carries (the request was read with another
                # name for the property): what goes is the requirement whose value the line says and does not keep
                said = stems(line)
                gone = [r for r in self.requirements if lowered(r["value"]) not in added and stems(r["value"])
                        and stems(r["value"]) <= said and self.says_value(line, r["value"])
                        and not re.search(r"\bstill\b[^.!?]*" + re.escape(lowered(r["value"])), lowered(line))]
                if len(gone) == 1:
                    self.requirements = [r for r in self.requirements if r is not gone[0]]
                    withdrawn += gone
                    replaced += 1
                    changed = True
            if withdrawn and not added:
                # the line takes an amount back and the reader gave nothing in its place: another amount the line
                # writes in the same unit is the new one ("drop the 2 l size, make it 750ml")
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
                    continue            # a value the line itself does not say is the reader's, not the shopper's
                if not re.search(r"\d", new["value"]) and any(self.within(new["value"], purpose) for purpose in self.purposes):
                    continue            # the use the line gives its requirement for (words of a purpose), not a requirement
                stands_now = next((now for now, before in self.took_back.items() if lowered(before) == lowered(new["value"])), None)
                if stands_now is not None and not self.says_value(line, stands_now):
                    continue            # a line written before the shopper's change, saying the old value "still" holds
                same = next((r for r in self.requirements if lowered(r["property"]) == lowered(new["property"])), None)
                if same is not None and lowered(same["value"]) == lowered(new["value"]):
                    continue
                if same is None and any(lowered(r["value"]) == lowered(new["value"]) for r in self.requirements):
                    continue            # the same value again under another name for the property: nothing new
                if same is not None and answering and not self.mentions(line, same):
                    # the answer to a question names neither that property nor its value: it adds a requirement of
                    # its own and takes nothing back, whatever name the reader gave it
                    new["property"], same = "requirement", None
                if (same is not None and answering and same.get("stated")
                        and not re.search(r"\b(?:instead|chang\w+|switch\w*|swap\w*|forget|no longer|rather than|replac\w+)\b", lowered(line))):
                    # the answer to the question just put, with no word of changing anything, names a value for a
                    # property the request was read to fix already: that reading was the reader's (most often the
                    # item's own name taken for a detail). What the shopper says is what was held back - it takes the
                    # other's place without a word, and counts as the answer it is
                    self.requirements.remove(same)
                    self.opening = [r for r in self.opening if lowered(r["value"]) != lowered(same["value"])]
                    same = None
                if same is not None:
                    self.requirements.remove(same)
                    replaced += 1           # a new value for a property already known: a change, not an answer
                    withdrawn.append(same)
                else:
                    given += 1
                # what the shopper has just withdrawn is no other word for what replaces it
                old = {closed_up(w) for r in withdrawn for w in [r["value"]] + list(r.get("written") or []) + list(r.get("alike") or [])}
                for kind in ("written", "alike"):
                    if isinstance(new.get(kind), list):
                        new[kind] = [w for w in new[kind] if closed_up(str(w)) not in old]
                if answering and self.pair_open and self.is_wish(new["value"]):
                    new["chosen"] = True        # the side of the doubted pair the shopper said matters more
                instead_of = [r for r in withdrawn if lowered(r["property"]) == lowered(new["property"])] or (withdrawn if len(withdrawn) == 1 else [])
                unit_before = re.fullmatch(r"\d+(?:\.\d+)?\s*([a-z][a-z ]*)", lowered(instead_of[0]["value"]).strip()) if instead_of else None
                if unit_before and re.fullmatch(r"\d+(?:\.\d+)?", new["value"].strip()):
                    # ("12 instead of 40": the new figure counts what the old one counted - 12 pcs for 40 pcs)
                    new["value"] = f"{new['value'].strip()} {unit_before.group(1).strip()}"
                    new["filed"] = False
                if instead_of and lowered(instead_of[0]["value"]) != lowered(new["value"]):
                    self.took_back[lowered(new["value"])] = instead_of[0]["value"]
                self.requirements.append(new)
                changed = True
                if "brand" in lowered(new["property"]):
                    # a brand the shopper now requires: ask the store for it directly
                    self.queue_filters()
                    self.add_queries([f"{new['value']} {self.item}", new["value"]], first=True)
            if sorted((lowered(p), lowered(v)) for p, v in before) == sorted((lowered(r["property"]), lowered(r["value"])) for r in self.requirements):
                # the line only repeats what already holds: nothing changed, and nothing new was given
                self.requirements, changed, given, replaced = held, changed_so_far, 0, 0
            else:
                self.fresh[line] = [r["value"] for r in self.requirements if r not in held]
            budget = money(heard.get("budget"))
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
            if self.grounded_priority(heard.get("priority"), line):
                self.priority, changed = self.grounded_priority(heard.get("priority"), line), True
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
                self.answered(line, given, spoke_of_else=spoke_of_else, about=" ".join(word_list(lowered(heard.get("about")))), nothing=nothing)
            if changed:
                self.changed_at = self.counter.turn
                gone_words = [set(word_list(closed_up(r["value"]))) for r in withdrawn if r not in self.requirements]
                kept_words = {w for v in self.values() for w in word_list(v)}
                gone_words = [words - kept_words for words in gone_words if words - kept_words]
                if gone_words:
                    # a search written for a value the shopper has just taken back finds what they no longer want
                    self.queries = [q for q in self.queries if not any(words <= set(q.split()) for words in gone_words)]
                if given or replaced:
                    self.fresh_rounds = 2       # the next looks go for what the shopper wants now, in the shelf's own words
                    self.shelf_words_used = 0   # (from the shelf's most usual words again: they were tried with other values)
                self.add_queries(self.value_queries() + list(heard.get("queries") or []), first=True)
                # what the shopper wants moved: room to look again, whatever was used before
                self.search_room = max(self.search_room, len(self.sent) + 16)
                self.search_asks = 0
                self.open_room = max(self.open_room, len(self.counter.pages) + 40)
                self.figure_room = max(self.figure_room, len(self.figures) + 3 * JUDGE_CARDS)
                self.judge_room = max(self.judge_room, self.judged_count + 36)
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
                # the shopper spoke of the store's change in the very turn of the reply: a shopper taken up with that
                # does not read a message, so the reply is given once more
                self.replies_again += 1
                for said_line in self.reply_sent[1].split("\n"):
                    text = said_line[2:] if said_line.startswith("- ") else said_line
                    if text.strip() and text not in self.unanswered:
                        self.unanswered.append(text)
                self.reply_sent = None
        return changed, asks

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
            said = self.name_words(line)
            told = self.requirements[-given:]
            kinds = {self.kind(r) for r in told}
            units = {self.unit_word(m.group(2)) for r in told for form in self.forms(r) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}

            def about_this(n: str) -> bool:
                """The answer is about this property: it names it (or the shopper's requirements do), the value is of
                its kind - a colour, a material, an amount in the unit it is measured in - or the reader says so."""
                words = self.name_words(n)
                if words <= said or any(words <= self.name_words(r["property"]) or self.name_words(r["property"]) <= words for r in self.requirements):
                    return True
                if ("colour" in kinds and re.search(r"colou?r", n)) or ("material" in kinds and "material" in n):
                    return True
                return bool(self.measure_units.get(n, set()) & units) or " ".join(word_list(n)) == about

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
            asked_for = self.leading if question in self.by_use else None
            if asked_for is not None and asked_for < len(self.purposes):
                mine, spoken = self.content_words(self.purposes[asked_for]), self.content_words(line)
                # (the line gives its reason - "because / since / given that this is for ..." - or carries another
                # purpose of the request nearly whole)
                reasoned = bool(re.search(r"^\s*as\b|\b(?:because|since|given that|given)\b", lowered(line)))
                others = [self.content_words(p) for i, p in enumerate(self.purposes[:2]) if i != asked_for]
                another = any(o and len(o & spoken) * 5 >= len(o) * 3 for o in others)
                if mine and (reasoned or another) and len(mine & spoken) * 5 < len(mine) * 3 and asked_for not in self.use_twice:
                    # the shopper gave the requirement of another purpose (the line names that one, not the one asked
                    # about): what this purpose holds back is still out, and it is asked about once more
                    self.use_again.add(asked_for)
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
                self.first_found_at = self.first_found_at or self.questions_asked
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
            if self.needs_found and self.leading is not None and self.leading not in self.use_twice and self.leading not in self.purposes_met:
                # (another purpose of this request did hold a requirement back: this one is asked about once more, in
                # other words, before any property - a shopper who names nothing may not have taken the question for it)
                self.use_again.add(self.leading)
            if self.told_to_ask and not self.plain_asked and not self.needs_found:
                self.plain_pending = True       # (a request that says to ask: the use itself is asked about once, in the plainest words)
            self.asked_log.append((question, "no requirement"))
            return
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
            if not self.just_over(key) or not self.fits_words(key) or (any_shelved and not self.in_category(page)):
                continue
            if self.brand and not set(word_list(self.brand)) <= set(word_list(page.get("brand"))):
                continue
            out.append(key)
        return sorted(out, key=lambda k: (-self.trust(k), (self.counter.stock.get(k) or {}).get("price") or 0.0))

    def named_shelf(self) -> str:
        """The store's own label for the shelf the request names: of the labels read on listings, the longest
        whose words stand together in the request."""
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
            # the shopper named the shelf in the store's own word: that is the shelf, whatever the request called the item
            best = max(told, key=lambda label: (len(label_key(label).split()), told[label]))
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

    @staticmethod
    def forms(requirement: dict, loose: bool = False) -> set[str]:
        """The required value in each way it may be written, as plain words; loose: the words sellers use alike."""
        spellings = list(requirement.get("written") or [])
        if requirement.get("filed"):        # (a value filed as a fact is filed in the store's own word)
            written = spellings + list(requirement.get("alike") or []) if loose else [requirement["value"]]
        elif re.search(r"\d", requirement["value"]):
            if re.search(r"\d\s*[a-z]", lowered(requirement["value"])):      # a figure alone is no measurement of an amount with a unit
                spellings = [form for form in spellings if not re.fullmatch(r"[\d.,\s]+", str(form).strip())
                             and Errand.same_amount(requirement["value"], form)]
            written = list(requirement.get("alike") or []) if loose else [requirement["value"]] + spellings
        else:                               # a word: another spelling of it is the word, anything else is a seller's other word
            close = [form for form in spellings if SequenceMatcher(None, lowered(form), lowered(requirement["value"])).ratio() >= 0.75]
            written = [form for form in spellings if form not in close] + list(requirement.get("alike") or []) if loose else [requirement["value"]] + close
        return {" ".join(singular(w) for w in word_list(closed_up(form))) for form in written} - {""}

    @staticmethod
    def same_amount(value: str, form: Any) -> bool:
        """Another spelling of an amount is that amount: every figure it gives with a unit is a figure of the value in
        the same unit, or in the same measure with another metric prefix ("4G" is not 4 GB, "14 days" is not 2 weeks)."""
        def figures(text: Any) -> list[tuple[float, str]]:
            return [(float(m.group(1)), Errand.unit_word(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", lowered(text))]
        wanted, found = figures(value), figures(form)
        if not wanted:
            return True
        for number, name in found:
            same = any(number == n and Errand.same_unit(name, u) for n, u in wanted)
            turned = any(Errand.scaled(u, name) and abs(n * (Errand.scaled(u, name) or 0.0) - number) < 1e-6 for n, u in wanted)
            if not same and not turned:
                return False
        return True

    def forms_under(self, requirement: dict, label: str) -> set[str]:
        """Under a fact's label the value may drop the words the label already says (a make is filed by its
        name alone, not by the name and the kind of thing)."""
        named = self.name_words(label) | {"at", "the", "of", "a", "an", "in", "on", "for", "with", "its"}
        out = set(self.forms(requirement))
        if out & self.filed_under(label):
            return out          # (the shelf files the whole value under this label: a shorter entry - "all" beside
                                # "all season" - is another value of the property, not this one said briefly)
        for form in list(out):
            words = form.split()
            kept = [w for w in words if not any(same_word(w, n) for n in named)]
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
        entries = {" ".join(singular(w) for w in word_list(reading(part))) for part in re.split(r"[,;/|]+", lowered(text)) for reading in (lowered, closed_up)}
        if not label:
            return bool(entries & self.forms(requirement))
        named = self.name_words(label)
        for entry in list(entries):         # an entry may repeat the label's own words
            kept = [w for w in entry.split() if not any(same_word(w, n) for n in named)]
            if kept:
                entries.add(" ".join(kept))
        return bool(entries & self.forms_under(requirement, label))

    def kind(self, requirement: dict) -> str:
        """Where a store states a property: an amount; a colour, which the variant's options name; a material,
        which the listing's name carries; anything else, which is kept as a fact of its own."""
        if re.search(r"\d", requirement["value"]):
            return "amount"
        name = lowered(requirement["property"])
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
        """The separate entries of a fact (a fact may list several), as plain words."""
        parts = (" ".join(singular(w) for w in word_list(closed_up(part))) for part in re.split(r"[,;/|]+", lowered(text)))
        return [part for part in parts if part]

    def entry_says(self, entry: str, requirement: dict, label: str, loose: bool = False) -> bool:
        """An entry filed under a label states the value: it carries every word of one of the value's forms.
        Words the label itself says do not have to be repeated, and may be repeated."""
        forms = self.forms(requirement, loose=True) if loose else self.forms_under(requirement, label)
        have = entry.split()
        return any(all(any(same_word(n, h) for h in have) for n in form.split()) for form in forms)

    def entry_is(self, entry: str, requirement: dict, label: str) -> bool:
        """The entry is the value and nothing else."""
        named = self.name_words(label)
        kept = " ".join(w for w in entry.split() if not any(same_word(w, n) for n in named)) or entry
        forms = self.forms_under(requirement, label)
        return entry in forms or kept in forms

    def own_labels(self, requirement: dict) -> tuple[set[str], set[str], set[str]]:
        """The fact labels that are this property's own, and the values the shelf files under them:
        the labels named like the property (one name holding all words of the other); the labels under which
        the shelf's listings state exactly such a value, as their only entry, more than once; and the single
        words that stand as entries under those labels."""
        pages = self.counter.pages
        stamp = (requirement["property"], requirement["value"], len(requirement.get("written") or []), len(pages), self.named_shelf())
        cached = self.own_cache.get(requirement["property"])
        if cached and cached[0] == stamp:
            return cached[1]
        wanted = self.name_words(requirement["property"])
        kind = self.kind(requirement)
        shelf = [page for page in pages.values() if self.in_category(page)] or list(pages.values())
        written = " ".join(word_list(requirement["property"]))
        tiers: list[set[str]] = [set(), set(), set(), set()]     # named exactly so; the same words; one name within the other
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        stated: dict[str, int] = {}
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                have = self.name_words(label)
                if wanted and have:
                    if " ".join(word_list(str(label).replace("_", " "))) == written:
                        tiers[0].add(str(label))
                    elif have == wanted:
                        tiers[1].add(str(label))
                    elif have <= wanted or wanted <= have:
                        # a name that adds only the required amount's unit (power_w for 1200 W) is the property's own;
                        # one that adds another word (suction_power) may name another measure
                        tiers[2 if (have ^ wanted) <= units else 3].add(str(label))
                entries = self.entries(value)
                if len(entries) == 1 and self.entry_is(entries[0], requirement, str(label)):
                    stated[str(label)] = stated.get(str(label), 0) + 1
        related = next((tier for tier in tiers if tier), set())
        if kind != "amount":
            # (a store may keep one property under two labels - a general one and its own department's - and file
            # the value under either: each label named like the property is the property's own)
            related = set().union(*tiers)
            if "material" in lowered(requirement["property"]) and self.part_labels(requirement):
                related = self.part_labels(requirement)     # (the part's own fact, not the item's material)
        most = max(stated.values(), default=0)
        inferred: set[str] = set()
        if kind != "colour" and not any(stated.get(label) for label in related):
            # no fact named like the property carries such a value anywhere on the shelf: the label that does
            inferred = {label for label, count in stated.items() if count >= 2 and 3 * count >= most}
        words: set[str] = set()
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    words |= {entry for entry in self.entries(value) if " " not in entry}
            if kind == "colour":        # a word that stands alone as a variant's name is a value variants are told apart by
                for variant in page.get("variants") or []:
                    words |= {entry for entry in self.entries(variant.get("options")) if re.fullmatch(r"[a-z]{3,}", entry)}
        answer = (related, inferred, words)
        self.own_cache[requirement["property"]] = (stamp, answer)
        return answer

    def named_facts(self, page: dict, requirement: dict) -> list[tuple[str, str]]:
        related, inferred, _ = self.own_labels(requirement)
        return [(str(label), str(value)) for label, value in (page.get("facts") or {}).items() if str(label) in related or str(label) in inferred]

    def says(self, text: Any, requirement: dict, whole: bool = False, loose: bool = False) -> bool:
        """The text carries every word of the required value, in one of the ways it may be written.
        whole: only words standing on their own count (a word joined to another by a hyphen is a different word)."""
        if whole:
            have = {singular(w) for reading in (lowered(text), closed_up(text)) for w in re.split(r"[\s/,;()\[\]|+]+", reading) if w}
            return any(set(form.split()) <= have for form in self.forms(requirement, loose))
        have = {singular(w) for reading in (lowered(text), closed_up(text)) for w in word_list(reading)}
        return any(all(any(same_word(n, h) for h in have) for n in form.split()) for form in self.forms(requirement, loose))

    def amounts(self, text: Any, requirement: dict) -> tuple[set[tuple[float, str]], set[tuple[float, str]]]:
        """The amounts a text states in the units of the required value: (those that are the value, the others)."""
        unit = self.unit_word
        wanted: set[tuple[float, str]] = set()
        for form in self.forms(requirement):
            wanted |= {(float(m.group(1)), unit(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        found = {(float(m.group(1)), unit(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", lowered(text))}
        # one measure written with a larger or smaller metric unit is the same amount (1 l and 1000 ml, 2.5 cm and 25 mm)
        for number, name in list(wanted):
            for seen in {u for _, u in found}:
                scale = self.scaled(name, seen)
                if scale:
                    wanted.add((round(number * scale, 6), seen))
        units = {name for _, name in wanted}
        found = {pair for pair in found if any(self.same_unit(pair[1], name) for name in units)}
        same = {pair for pair in found if any(pair[0] == number and self.same_unit(pair[1], name) for number, name in wanted)}
        return same, found - same

    @staticmethod
    def other_words(tokens: set[str]) -> set[str]:
        """Plain words among option tokens: not a code with digits or signs, and not a size letter code (xl, xxl)."""
        return {w for w in tokens if re.fullmatch(r"[a-z]{3,}", w) and not re.fullmatch(r"x*[sml]|x+l", w)}

    def unit_in_label(self, label: str, requirement: dict) -> bool:
        """The fact's label names the unit of the required amount (capacity_l, display_size_inch)."""
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        return bool(units & self.name_words(label))

    def named_otherwise(self, options: Any, requirement: dict) -> bool:
        """A part of the variant's options gives the property by its name with a figure that is not the required
        amount (however the seller spells the name): that figure is the property's, and the required one, where the
        options also carry it, measures something else."""
        named = [w for w in self.name_words(requirement["property"]) if len(w) >= 5]
        for part in re.split(r"[/,;|]+", lowered(options)):
            if not re.search(r"\d", part) or self.has_amount(part, requirement):
                continue
            if any(SequenceMatcher(None, word, name).ratio() >= 0.8 for word in re.findall(r"[a-z]{4,}", part) for name in named):
                return True
        return False

    def has_amount(self, text: Any, requirement: dict) -> bool:
        """The text states the required amount: the same number with the same unit, however it is spaced. An
        amount required as a bare number takes its unit from the property's name."""
        same, others = self.amounts(text, requirement)
        if same or (not others and self.says(text, requirement)):
            return True         # (where the text gives that unit with another number, the words of the value standing
                                # apart in it are not the amount: "50*23mm (1 piece)" is one piece)
        bare = re.fullmatch(r"\d+(?:\.\d+)?", requirement["value"].strip())
        if not bare:
            return False
        named = [w for w in word_list(requirement["property"]) if len(w) >= 3]
        return any(float(m.group(1)) == float(bare.group(0)) and any(self.same_unit(m.group(2), w) for w in named)
                   for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]+)", lowered(text)))

    @staticmethod
    def same_unit(one: str, other: str) -> bool:
        """One unit written short and long: the short one's letters stand in the long one in order, from its first
        letter. Short units are told apart by every letter."""
        unit = Errand.unit_word
        a, b = sorted((unit(lowered(one)), unit(lowered(other))), key=len)
        if not a or a[0] != b[0] or len(b) <= 3 or (len(a) == 1 and len(b) > 6):
            return bool(a) and a == b       # (a single letter stands for a short word only)
        rest = iter(b)
        return all(letter in rest for letter in a)

    @staticmethod
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
                if size != other_size and (base == other_base or (len(base) > 1 and len(other_base) > 1 and Errand.same_unit(base, other_base))):
                    ratio = size / other_size
                    if any_step or abs(ratio - 1000.0) < 1e-6 or abs(ratio - 0.001) < 1e-9:
                        return ratio
        return None

    def other_amount(self, text: Any, requirement: dict) -> bool:
        """The text states an amount in a unit of the required value, but not that amount."""
        same, others = self.amounts(text, requirement)
        return bool(others) and not same

    def plain(self, key: tuple[str, str], requirement: dict) -> int:
        """How plainly the page states one required value: 3 where the store states that kind of property (the
        variant's options, the property's own fact), 2 in the listing's name, 1 unclearly (beside other values, in
        another fact, in another word for it), 0 nowhere, -1 when the page states something else for it."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        title = page.get("title")
        kind = self.kind(requirement)
        if kind == "amount":
            # (a figure that is the maker's name - a brand written as a number and a letter - is no amount of the item)
            for token in {w for w in word_list(page.get("brand")) if re.fullmatch(r"\d+[a-z]{1,2}", w)}:
                options = re.sub(rf"(?i)(?<![a-z0-9.]){token}(?![a-z0-9])", " ", options)
                title = re.sub(rf"(?i)(?<![a-z0-9.]){token}(?![a-z0-9])", " ", str(title or ""))
        related, inferred, words = self.own_labels(requirement)
        own = [(label, self.entries(value)) for label, value in self.named_facts(page, requirement)]
        singles = [(label, entries[0]) for label, entries in own if len(entries) == 1]
        lists = [(label, entry) for label, entries in own if len(entries) > 1 for entry in entries]
        stated = any(self.entry_says(entry, requirement, label) for label, entry in singles)
        listed = any(self.entry_says(entry, requirement, label) for label, entry in lists)
        value_words = {w for form in self.forms(requirement) for w in form.split()}
        elsewhere = any(self.says(value, requirement) for value in (page.get("facts") or {}).values())

        if kind == "amount":
            if self.other_amount(options, requirement) or self.named_otherwise(options, requirement):
                return -1           # the variant's own options state another amount of that kind
            number = re.search(r"\d+(?:\.\d+)?", requirement["value"])
            if self.has_amount(options, requirement):
                return 3
            # (the property's own fact - named like it, or the one this shelf files such amounts under - gives the figure bare)
            bare = [label for label, entry in singles if number and re.fullmatch(r"\d+(?:\.\d+)?", entry) and float(entry) == float(number.group(0))]
            if any(self.unit_in_label(label, requirement) for label in bare) or any(self.has_amount(entry, requirement) for _, entry in singles):
                return 3
            if bare:
                # the property's own fact gives the figure without its unit: the same number, but no measurement on its own
                return 2 if self.has_amount(title, requirement) else 1
            if any(re.search(r"\d", entry) for _, entry in singles):
                return -1           # the property's own fact gives another figure
            if self.has_amount(title, requirement):
                return 1 if self.amounts(title, requirement)[1] else 2      # (a title with several such amounts lists its variants)
            if self.other_amount(title, requirement) and not self.amounts(options, requirement)[0]:
                return -1           # the listing's name gives another amount of that kind ("trial 2ml"): a fact that
                                    # says the required one is the seller's form filled in, not what is sold
            for value in (page.get("facts") or {}).values():
                if self.amounts(value, requirement)[1]:
                    continue        # (a fact with several amounts of that kind lists what is on offer, not this variant)
                for entry in self.entries(value):
                    # one entry of a fact that is this amount and nothing else (not one side of a 30x20x5cm)
                    if self.amounts(entry, requirement)[0] and len(re.findall(r"\d+(?:\.\d+)?", entry)) == 1:
                        return 2
            return 1 if listed or any(self.has_amount(value, requirement) for value in (page.get("facts") or {}).values()) else 0

        named_as = self.named_as.get((key, requirement["property"], requirement["value"]))
        if kind == "colour":
            wanted = self.forms(requirement)
            parts = [{singular(w) for reading in (lowered(part), closed_up(part)) for w in re.split(r"[\s()\[\]+]+", reading) if w}
                     for part in re.split(r"[/,;|]+", options)]
            # (a colour joined by a hyphen to a code with a digit in it - green-a09 - is that colour, with the seller's code)
            parts = [part | {singular(piece) for w in part if "-" in w and re.search(r"\d", w) for piece in w.split("-") if piece} for part in parts]
            named = [part for part in parts if any(set(form.split()) <= part for form in wanted)]
            if named:
                # the part of the options that names the colour: another colour word beside it makes it another colour
                alone = any(not ((part & words) - value_words) and not self.other_words(part - value_words) for part in named)
                return 3 if alone and named_as not in ("among", "other") else 1
            # (a sign that is not a letter, a digit, a point or a hyphen parts words too - cap*red: the colour is
            # named, but joined to something else by the seller, so less plainly than a word standing on its own)
            joined = [{singular(w) for reading in (lowered(part), closed_up(part)) for w in re.split(r"[^a-z0-9.\-]+", reading) if w}
                      for part in re.split(r"[/,;|]+", options)]
            if any(set(form.split()) <= part for form in wanted for part in joined):
                return 1
            if any(form.replace(" ", "") in lowered(options).replace(" ", "") for form in wanted):
                return -1           # the word joined to other characters is a code or another option's name, not this colour
            if stated:
                return 1 if options.strip() else 3      # (options that name the variant otherwise leave it unclear)
            before = self.took_back.get(lowered(requirement["value"]))
            if before and any(set(form.split()) <= part for form in self.forms(dict(requirement, value=before, written=[], alike=[])) for part in parts):
                return -1           # the variant is named by the colour the shopper took back for this one
            alike = any(set(form.split()) <= part for form in self.forms(requirement, loose=True) for part in parts)
            if alike or named_as in ("only", "among"):
                return 1            # another word for the colour: the reader's or the seller's
            if options.strip():
                return -1           # the options name the variant: when they say something else, the variant is something else
            return 1 if listed else 0

        if kind == "material":
            # (the property's own fact has to BE the material: a near word - silicon for silicone - or a blend of two
            # - wood+silicone - is another entry of the store's list, and such listings failed where the exact one passed)
            exactly = any(self.entry_is(entry, requirement, label) and not re.search(r"[+&]", str(raw))
                          for (label, entry), raw in zip(singles, [value for label, value in self.named_facts(page, requirement)
                                                                 if len(self.entries(value)) == 1]))
            if stated and exactly:
                return 3
            if stated:
                return 1
            if named_as in ("among", "other"):
                return 1 if named_as == "among" else -1
            contrary = bool(singles)        # its own fact names another material
            if named_as == "only":
                return 2            # (the name says what the item is made of; a filed material is often that of a part)
            if self.says(title, requirement):
                named_too = ({singular(w) for w in word_list(title)} & words) - value_words
                return 1 if named_too else 2    # (another material beside it: unclear what the item is made of)
            if listed or self.says(title, requirement, loose=True):
                return 1
            if contrary:
                return -1
            return 1 if elsewhere else 0

        # a property the store keeps as a fact of its own: the fact has to be the value, not only hold its words
        # (a "table lamp shade" is filed as another kind of thing than a "table lamp")
        if stated:
            return 3 if any(self.entry_is(entry, requirement, label) for label, entry in singles) else 1
        if listed or any(self.entry_says(entry, requirement, label, loose=True) for label, entry in singles):
            return 1
        if own:
            if self.filed_alike(page, requirement):
                return 1            # (the store keeps the property under a second label, its department's own, and that one is the value)
            return -1               # its own fact states something else
        if self.filed_elsewhere(page, requirement):
            return 1                # (another fact of the listing is the value and nothing else: the store has it on file)
        if related:
            return -1               # the store files this property as a fact, and this listing has none
        if inferred:
            return 0
        return 1 if elsewhere or self.says(title, requirement) else 0

    def exact(self, key: tuple[str, str], requirement: dict) -> int:
        """How firmly the page states the value: each of the property's own facts whose only entry is exactly the
        value counts for the listing, each that states something else counts against it; options that are
        exactly the value count for it."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        score = 0
        for label, value in self.named_facts(page, requirement):
            entries = self.entries(value)
            if len(entries) == 1:
                score += 1 if self.entry_is(entries[0], requirement, label) else (0 if self.entry_says(entries[0], requirement, label) else -1)
        return score + int(self.states(options, requirement))

    def filed_elsewhere(self, page: dict, requirement: dict) -> bool:
        """A fact that is not the property's own states the value as its only entry."""
        mine = {label for label, _ in self.named_facts(page, requirement)}
        for label, value in (page.get("facts") or {}).items():
            entries = self.entries(value)
            if str(label) not in mine and len(entries) == 1 and self.entry_is(entries[0], requirement, str(label)):
                return True
        return False

    def filed_alike(self, page: dict, requirement: dict) -> bool:
        """A fact that shares a word of its name with the property (beyond "type") states the value as its only entry."""
        mine = {label for label, _ in self.named_facts(page, requirement)}
        wanted = self.name_words(requirement["property"]) - {"type"}
        for label, value in (page.get("facts") or {}).items():
            entries = self.entries(value)
            if (str(label) not in mine and wanted & self.name_words(label) and len(entries) == 1
                    and self.entry_is(entries[0], requirement, str(label))):
                return True
        return False

    def label_nearness(self, label: str, requirement: dict) -> tuple[int, int, int]:
        """How near a fact's label is to the property's name: the words they share less the label's other words;
        then whether its other words are the shelf's own name (a store keeps a property under a general label and
        under its department's - "garden_tools_type" on the garden tools shelf - and goes by the
        department's); then the words they share."""
        wanted, have = self.name_words(requirement["property"]), self.name_words(label)
        shelf = self.name_words(self.named_shelf() or self.category or self.item)
        other = have - wanted
        departments = bool(other) and other <= shelf
        return (len(wanted & have) - (0 if departments else len(other)), int(departments), len(wanted & have))

    def nearest_label(self, key: tuple[str, str], requirement: dict) -> int:
        """Of the facts on the page that bear on the property - those named like it, and any whose only entry is the
        value: 1 when the one named nearest to the property is the value and nothing else, -1 when the nearest says
        something else or lists the value among others, 0 when two as near disagree or no fact bears on it. A store
        keeps one property under several labels and goes by the one that is the property's own name."""
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
            # (a label that carries two words of the property's name bears on it too, whatever the shelf files under it)
            if not is_value and label not in mine and len(wanted & self.name_words(label)) < 2:
                continue
            near = self.label_nearness(label, requirement)
            if not is_value and len(entries) == 1 and self.entry_says(entries[0], requirement, label):
                is_value = True     # (one longer entry that holds the value - "indoor ceiling lights" - is still that value)
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
            if len(entries) == 1 and not self.entry_says(entries[0], requirement, label):
                return True
        return False

    def level_needed(self, requirement: dict) -> int:
        """The level at which a value counts as stated where the store states that kind of property."""
        return 2 if self.kind(requirement) in ("amount", "material") else 3

    def settled(self, key: tuple[str, str], requirement: dict) -> bool:
        """The page's own words state the value, plainly or not."""
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
        elif "-" in key[1] and key[1].rsplit("-", 1)[0] in {k[1].rsplit("-", 1)[0] for k in self.verdicts if "-" in k[1] and self.fits_words(k)}:
            score += 20             # another variant of a fitting listing
        return score + (50 if self.listed and key in self.listed else 0)

    def to_open(self, limit: int, allowance: int | None = None) -> list[dict]:
        allowance = self.open_room if allowance is None else allowance
        unopened = [k for k in self.counter.row_order if k[0] not in self.counter.pages and (self.affordable(k) or self.just_over(k, OPEN_REACH))]
        if self.listed:
            unopened = [k for k in self.listed if k[0] not in self.counter.pages] + [k for k in unopened if k in self.listed]
        # (rows first shown since the shopper last changed something go first: the searches sent since then carry the
        # values as they now stand, and a search matches a variant's options, which a row does not show)
        fresh = (lambda k: self.counter.shown_at.get(k, 0) >= self.changed_at) if self.changed_at else (lambda k: False)
        # (a row a little over the limit is read after the rows within it that look as good: it can only be named, not bought)
        ranked = sorted(dict.fromkeys(unopened), key=lambda k: (not fresh(k), -(self.row_score(k) - (0 if self.affordable(k) else 2))))
        ids = list(dict.fromkeys(k[0] for k in ranked))[: max(0, min(limit, allowance - len(self.counter.pages)))]
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
        """Settle every unjudged candidate by the page's own words."""
        shelf_words = set(label_key(self.named_shelf()).split())

        def the_shelf_itself(r: dict) -> bool:
            return bool(r.get("stated") and shelf_words and set(label_key(r["value"]).split()) <= shelf_words)
        self.requirements = [r for r in self.requirements if not the_shelf_itself(r)]
        # (nor is it a value the first cart has to state: the request as first written is held to the same reading)
        self.opening = [r for r in self.opening if not the_shelf_itself(r)]
        self.name_check()
        stamp = json.dumps([self.requirements, self.priority, self.lenient, len(self.named_as)], sort_keys=True)
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
            # (nor a colour: where no word of the page names one for the variant, the store files none for it)
            met = [self.settled(key, r) or (said[i] and self.plain(key, r) != -1 and not re.search(r"\d", r["value"]) and self.kind(r) != "colour")
                   for i, r in enumerate(self.requirements)]
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
        """Fitting listings, best first. Without a priority: the page that states the values most plainly, then
        price. With one: by what ordering the listing is expected to bring - how well it stands by the priority
        among the fitting listings, weighed by how sure it is that it fits at all (a listing that states every
        requirement plainly against one that states it unclearly)."""
        if self.better in ("higher", "lower"):
            higher = self.better == "higher"
        else:
            # (until the reader has said which way the priority runs: more is better, unless its words say less)
            higher = not re.search(r"\b(low|least|small|short|few|min|light)", lowered(self.priority))
        by_price = self.priced_priority()
        by_figure = bool(self.priority) and not by_price
        fitting = [k for k in self.verdicts if self.fits(k)]

        def number(key: tuple[str, str]) -> float | None:
            value = self.figure(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        figures = {k: number(k) for k in fitting} if by_figure else {}
        trust = {k: self.trust(k) for k in fitting}
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
            title = lowered((self.counter.pages.get(k[0]) or {}).get("title")) or k[0]
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
            own = lowered((self.counter.pages.get(key[0]) or {}).get("title")) or key[0]
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
            plainly = sum(self.plain(key, r) for r in self.requirements) + sum(self.exact(key, r) for r in self.requirements)
            twin = lowered((self.counter.pages.get(key[0]) or {}).get("title")) in self.lost_titles
            rest = (0 if self.firm(key) else 1, 0 if self.solid(key) else 1, 0 if self.verdicts[key].get("by") == "words" else 1, -plainly,
                    0 if twin else 1, price)
            if by_price:
                return (0, -trust[key], price) + rest
            if not by_figure:
                return (0, -trust[key]) + rest
            value = figures[key]
            worth = trust[key] * standing(key)
            # equal worth: the surer fit first (a wrong order is worth nothing, a fitting one mostly its full worth),
            # then the listing whose own options state the figure (they describe the exact variant sold)
            in_options = value is not None and any(abs(float(n) - value) < 1e-9 for n in re.findall(r"\d+(?:\.\d+)?", str(self.variant_of(key).get("options") or "")))
            # (a surely fitting listing that states no figure does not stand before the rest on that account: where a
            # fitting listing states one, a store pays nothing for it - it is weighed by its worth like the others)
            ahead = key in sure and (value is not None or not rivals)
            return (0, 0 if ahead else 1, -round(worth, 6), -trust[key], 1 if value is None else 0,
                    -((value if higher else -value) if value is not None else 0.0), 0 if in_options else 1) + rest

        return sorted(fitting, key=order)

    def counted_unit(self) -> str:
        """The unit the priority is counted in: the one the shopper's own words name (a count of pieces), else the one
        the reader gave the figures in."""
        if re.search(r"\bpieces?\b", lowered(self.priority)):
            return "pc"
        return self.unit_word((re.findall(r"[a-z]+", lowered(self.figure_unit)) or [""])[0])

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
            for m in re.finditer(r"(?<![a-z0-9.,])(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", lowered(text)):
                word = self.unit_word(m.group(2))
                if self.same_unit(word, unit):
                    out.add(float(m.group(1)))
                elif self.scaled(word, unit, any_step=True):
                    out.add(round(float(m.group(1)) * (self.scaled(word, unit, any_step=True) or 0.0), 6))
            return out
        facts: set[float] = set()
        for label, fact in (page.get("facts") or {}).items():
            facts |= written(fact)
            bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(fact))
            if bare and unit != "pc" and any(self.same_unit(word, unit) for word in word_list(str(label).replace("_", " "))):
                facts.add(float(bare.group(1)))     # (a bare number under a label that names the unit: content_ml)
        answer: tuple[bool, float | None] = (False, None)
        for found in (written(self.variant_of(key).get("options")), facts, written(page.get("title"))):
            if found:
                answer = (True, next(iter(found)) if len(found) == 1 else None)
                break
        self.counted_cache[stamp] = answer
        return answer

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
            found = [(float(m.group(1)), m.group(2)) for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", lowered(text))]
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
        # when the fact writes it with its unit, or bare under a label that names the unit; a figure inferred from
        # another kind of fact (a single item for one piece) is not on the page at all
        for label, fact in ((self.counter.pages.get(key[0]) or {}).get("facts") or {}).items():
            if stands_in(fact) is True:
                return True
            bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(fact))
            if bare and float(bare.group(1)) == float(value) and any(self.same_unit(word, unit) for word in word_list(str(label).replace("_", " "))):
                return True
        return False

    def fixed_figure(self) -> float | None:
        """The amount a requirement fixes in the unit the figures for the priority are read in, when there is one."""
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
        level, kind = self.plain(key, requirement), self.kind(requirement)
        if kind == "fact":
            nearest = self.nearest_label(key, requirement)
            if level >= 3:
                # (a store may keep the property under several labels and grades by the one that is the property's
                # own name: where that one says something else, or lists the value among others, the listing is not
                # filed under the value, whatever a label named less like the property says)
                if nearest < 0:
                    return FAINT_FIT
                if nearest > 0 and self.exact(key, requirement) > 0 and not self.gainsaid(key, requirement):
                    return FIRM_FIT
                return SURE_FIT if nearest > 0 else LIKELY_FIT
            if level >= 1 and nearest < 0:
                return FAINT_FIT
            if level >= 1:
                # the property's own fact carries the value inside a longer entry, or another fact of the listing is
                # exactly the value: the store has the value on file for this listing, and such listings are nearly
                # always what was asked for; a value only the name carries is not
                page = self.counter.pages.get(key[0]) or {}
                within = any(len(self.entries(value)) == 1 and self.entry_says(self.entries(value)[0], requirement, label)
                             for label, value in self.named_facts(page, requirement))
                if within or self.filed_elsewhere(page, requirement):
                    return SURE_FIT
            return FAINT_FIT
        if kind == "amount":
            return FIRM_FIT if level >= 2 else LIKELY_FIT if level == 1 else FAINT_FIT
        if kind == "colour":
            if level >= 3:
                # the variant named by the colour's own word (or filed under it) is that colour; a shortened or
                # other spelling of the word in its place is nearly as sure, not as sure
                own_word = dict(requirement, written=[], alike=[])
                if self.second_colour(key, requirement) or (self.fused_colour(key, requirement) and self.exact(key, requirement) <= 0):
                    return LIKELY_FIT       # (another part of the variant's name carries another colour, or the colour
                                            # is fused into a longer word by a hyphen and no fact of its own files it)
                # (firmly so where a part of the options is the colour and nothing else, or its own fact is: beside a
                # seller's code - kx12-white - the store may file the variant under the code)
                return FIRM_FIT if self.states(self.variant_of(key).get("options"), own_word) or self.exact(key, requirement) > 0 else SURE_FIT
            # beside other words the variant's own options still mostly mean that colour, and so does the colour the
            # store files for the listing; a colour only the name carries, or only another word for it, seldom is
            own = self.says(self.variant_of(key).get("options"), requirement) or self.exact(key, requirement) > 0
            if level >= 1 and own and (self.second_colour(key, requirement) or self.fused_colour(key, requirement)) and self.exact(key, requirement) <= 0:
                return LIKELY_FIT       # (a variant named by two colours is as often filed under the other one)
            return SURE_FIT if level >= 1 and own else UNSURE_FIT if level >= 1 else FAINT_FIT
        return FIRM_FIT if level >= 3 else SURE_FIT if level >= 1 else UNSURE_FIT

    def fused_colour(self, key: tuple[str, str], requirement: dict) -> bool:
        """The variant's options carry the colour only as one piece of a word joined by a hyphen (to a seller's code, an
        amount, another word): such a name is that colour in some stores' filing and no colour at all in others."""
        options = lowered(self.variant_of(key).get("options"))
        wanted = self.forms(requirement)
        alone = [{singular(w) for w in re.split(r"[\s()\[\]+]+", part) if w} for part in re.split(r"[/,;|]+", options)]
        if any(set(form.split()) <= part for form in wanted for part in alone):
            return False
        pieces = {singular(piece) for word in re.split(r"[\s/,;|()\[\]+]+", options) if "-" in word for piece in word.split("-") if piece}
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
        parts = [{singular(w) for w in re.split(r"[^a-z0-9]+", lowered(part)) if w} for part in re.split(r"[/,;|]+", options)]
        named = any(set(form.split()) <= words for form in self.forms(requirement) for words in parts)
        return named and any(words & known for words in parts)

    def keeps_request(self, key: tuple[str, str]) -> bool:
        """The listing is the kind asked for and states every requirement the request itself wrote, where the store
        states that kind of property: the first cart is the shopper's first sight of the choice and holds to those."""
        stated = [r for r in self.requirements if r.get("stated")]
        return bool((self.verdicts.get(key) or {}).get("kind_ok")) and all(self.plain(key, r) >= self.level_needed(r) for r in stated)

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
        if self.brand and not set(word_list(self.brand)) <= set(word_list(page.get("brand"))):
            return False
        return all(self.plain(key, r) >= self.level_needed(r) for r in self.opening)

    def pick_stand_in(self) -> tuple[str, str] | None:
        """A listing to put in the cart ahead of the choice. What the store and the shopper do at the first cart -
        the listing sells out or its price jumps, the shopper names a new limit or another value - falls on that
        listing, and the choice is made knowing it. The first cart is held to the request as first written, and a
        listing is surest to meet that when it meets everything known: so the stand-in is another of the fitting
        listings, one as plainly fitting as any and the least to lose by the priority. When no other listing fits, it
        is a listing that meets the request as first written."""
        ranked = [k for k in self.ranked() if k not in self.gone]
        best = ranked[0] if ranked else None
        best_title = lowered((self.counter.pages.get(best[0]) or {}).get("title")) if best else ""

        def other(key: tuple[str, str]) -> bool:
            return key != best and lowered((self.counter.pages.get(key[0]) or {}).get("title")) != best_title
        fitting = [k for k in ranked if other(k) and self.meets_opening(k)]
        if fitting:
            more_is_better = self.better != "lower" and not re.search(r"\b(low|least|small|short|few|min|light|cheap)", lowered(self.priority))

            def worth(key: tuple[str, str]) -> tuple:
                # what would be lost with it: least for a listing whose figure is read and stands lowest by the
                # priority; a listing whose figure could not be read may be the best, so among those the cheaper one
                # (more of anything costs more)
                value, price = self.figure(key), (self.counter.stock.get(key) or {}).get("price") or 0.0
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return (0, float(value) if more_is_better else -float(value), price)
                return (1, price if more_is_better else -price, 0.0)
            surest = max(self.trust(k) for k in fitting)
            return min([k for k in fitting if self.trust(k) >= surest], key=worth)
        if best is not None and self.meets_opening(best) and not self.changed_early:
            return None             # the choice itself meets the request as written, and nothing else fitting does
        # the choice no longer meets the request as written - or it alone meets everything with a shopper who changes
        # things as the shopping goes, where the store's event at the first cart is the rule and would leave nothing:
        # a listing that meets the request as first written (and not what the shopper has said since) goes in first
        spare = [k for k in self.verdicts if other(k) and k not in ranked and self.affordable(k) and self.meets_opening(k)]

        def plainness(key: tuple[str, str]) -> tuple:
            price = (self.counter.stock.get(key) or {}).get("price") or 0.0
            return (sum(self.plain(key, r) + self.exact(key, r) for r in self.opening), -price)
        return max(spare, key=plainness) if spare else None

    def second_stand_in(self, choice: tuple[str, str]) -> tuple[str, str] | None:
        """A listing to put in the cart between the first cart and the choice: another article than the choice, of
        the kind asked for, on sale within the limit - one that meets the request as first written when there is one
        (a store may count only such a listing as a cart), else the nearest other."""
        counter = self.counter
        title = lowered((counter.pages.get(choice[0]) or {}).get("title"))

        def usable(key: tuple[str, str]) -> bool:
            page = counter.pages.get(key[0]) or {}
            return (key != choice and key[0] != choice[0] and key not in counter.cart and key not in self.gone and self.on_sale(key)
                    and bool((self.verdicts.get(key) or {}).get("kind_ok")) and lowered(page.get("title")) != title)
        keys = [k for k in self.verdicts if usable(k)]
        if not keys or not counter.has("add_to_cart") or not counter.has("remove_from_cart") or counter.turns_left() < 3:
            return None
        wanted_now = set(self.ranked())
        # (not one of the listings that could still be the choice: what the store does to it is lost with it)
        return min(keys, key=lambda k: (k in wanted_now, not self.meets_opening(k), -sum(self.plain(k, r) for r in self.requirements),
                                        (counter.stock.get(k) or {}).get("price") or 0.0))

    def cart_stand_in(self) -> None:
        """Put the stand-in in the cart, read its stock, and take in what the store and the shopper do next."""
        counter = self.counter
        if counter.cart or counter.over or counter.turns_left() < 9 or self.late() or not counter.has("add_to_cart"):
            return
        self.judge()
        self.rank_by_priority(every=True)       # (which listing stands last is known only once the figures are read)
        key = self.pick_stand_in()
        if key is None:
            return
        title = str((counter.pages.get(key[0]) or {}).get("title") or "")[:50]
        self.note(f"stand-in: {key} {title!r} {self.variant_of(key).get('options')!r} turn={counter.turn} fits now={self.fits(key)}")
        group = [act("add_to_cart", product_id=key[0], sku=key[1])]
        if counter.has("inspect_stock"):
            group.append(act("inspect_stock", product_id=key[0], sku=key[1]))
        counter.send(group)
        if key not in counter.cart:
            return
        self.stand_in = key
        seen = counter.stock.get(key) or {}
        price = seen.get("price")
        changed, _ = self.hear()
        if seen.get("in_stock") is False or (self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9):
            self.gone.add(key)
            self.owed_reaction = True
            self.pending_line = ""          # the shopper's reaction to the change is answered by the choice
        if changed and counter.turns_left() > TURN_RESERVE and not self.late():
            # the values as they now stand are searched before anything else, and what those searches show is
            # opened first
            fresh = self.next_searches(counter.group_size)
            if fresh:
                counter.send(fresh)
                self.hear_quietly()
            opens = self.to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + 60)
            if opens and not counter.over:
                counter.send(opens[: counter.group_size])
                self.hear_quietly()
            self.judge()
            self.look(rounds=4)

    def meets_chosen(self, key: tuple[str, str]) -> bool:
        """The listing meets what the shopper said matters more (the chosen side of a doubted pair), when there is one."""
        met = ((self.verdicts.get(key) or {}).get("met") or [])[: len(self.requirements)]
        marks = [x for x, r in zip(met, self.requirements) if r.get("chosen")]
        return bool(marks) and all(marks)

    def priced_priority(self) -> bool:
        """The shopper's priority is the price itself (the cheapest, the lowest price) - not something else that
        they want the most of for the price or within it."""
        text = lowered(self.priority)
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
            answer = self.reader.fill(RANK_GUIDE, text, RANK_TOOL, 400) or {}
            if str(answer.get("better") or "").lower() in ("higher", "lower"):
                self.better = str(answer["better"]).lower()
            if not self.figure_unit and isinstance(answer.get("unit"), str):
                self.figure_unit = " ".join(answer["unit"].split())[:20]    # (figures of later batches must be in the same unit)
            values = answer.get("values")
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
                        self.figures[keys[int(found.group(1)) - 1]] = float(found.group(2))
            for key in keys:
                self.figures.setdefault(key, None)      # read once: not asked again
            self.note(f"ranked by {self.priority!r}: " + " ".join(entry[:14] for entry in entries)[:200])

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
        if not self.priority or self.priced_priority() or not counter.has("search"):
            return
        best_so_far: float | None = None
        for round_number in range(BEST_ROUNDS):
            chosen = [k for k in counter.cart if k != self.stand_in]
            if counter.over or chosen or self.late() or not self.can_think() or counter.turns_left() <= TURN_RESERVE + 4:
                break               # (the stand-in in the cart is not the choice: the best is still looked for)
            self.rank_by_priority(every=True)
            found = self.best_figures()
            best = found[0][0] if found else None
            if round_number and (best is None or best_so_far is not None and
                                 not (best < best_so_far if self.better == "lower" else best > best_so_far)):
                break               # the last round brought nothing better: the best is likely in hand
            best_so_far = best
            shown = "\n".join(f"- {value:g}: {str((counter.pages.get(key[0]) or {}).get('title') or '')[:90]} | {self.variant_of(key).get('options') or '-'}"
                              for value, key in found[:6]) or "(no fitting listing states a figure yet)"
            text = (f"{self.state_text()}\n\nBest figures found so far for the priority, with the listings that carry them:\n{shown}"
                    f"\n\nSearches tried:\n" + "\n".join(sorted(self.sent)[-30:]))
            answer = self.listed_fields(self.reader.fill(BEST_GUIDE, text, BEST_TOOL, 300))
            queries = [q for q in answer.get("queries") or [] if isinstance(q, str) and q.strip()][:8]
            # a listing's other variants are sold under its own title: the fitting listings' titles with the required
            # values bring up their variants, the ones with a better figure among them
            every = " ".join(self.values())
            siblings = []
            for key in [k for _, k in found] + [k for k in self.ranked() if k not in self.gone]:
                title = lowered((counter.pages.get(key[0]) or {}).get("title"))
                if title and title not in self.kin_swept and len(siblings) < counter.group_size - len(queries):
                    self.kin_swept.add(title)
                    siblings.append(f"{every} {' '.join(word_list(title)[:12])}")
            queries = siblings + queries
            self.note(f"better by {self.priority!r} than {best}: {queries}")
            if not queries:
                break
            self.search_room = max(self.search_room, len(self.sent) + len(queries))
            self.add_queries(queries, first=True)
            searches = self.next_searches(min(counter.group_size, len(queries)))
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
        """Nothing found meets every requirement: the reader writes new searches from the request, the requirements,
        the searches tried and the titles seen, and the agent looks again. True when it did."""
        if self.search_asks >= 2 or not self.can_think() or self.late() or self.counter.turns_left() <= TURN_RESERVE + 2:
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
            # what the shopper said matters more outranks the count of other requirements met
            return (-int(bool(verdict.get("kind_ok"))), -chosen, -sum(1 for x in met if x), -self.said_in_part(key, met),
                    (self.counter.stock.get(key) or {}).get("price") or 0.0)
        return sorted((k for k in self.verdicts if self.affordable(k)), key=score)

    # ---- looking
    def hear_quietly(self) -> None:
        if self.counter.heard:
            self.hear()

    def name_check(self) -> None:
        """For a colour or a material, have the reader say what the words of the likeliest listings name: the
        wanted value alone, the wanted value beside another, or another value. Sellers write a colour in other
        words, and a title that names two materials leaves open which one the item is made of."""
        for requirement in self.requirements:
            kind = self.kind(requirement)
            if kind not in ("colour", "material") or not self.can_think() or self.name_checks >= 10:
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
            numbered = all(re.match(r"\s*\[?\d+", str(mark)) for mark in marks)
            if numbered or len(marks) == len(waiting):      # (marks without numbers count only when there is one per text)
                for position, mark in enumerate(marks, start=1):
                    word = re.search(r"\b(only|among|other|none)\b", lowered(mark))
                    number = re.match(r"\s*\[?(\d+)", str(mark))
                    at = int(number.group(1)) if numbered else position
                    if word and 1 <= at <= len(waiting):
                        self.named_as[waiting[at - 1][0]] = word.group(1)
            for tag, _ in waiting:
                self.named_as.setdefault(tag, "unread")      # asked once, not asked again
            self.note(f"names {requirement['value']}: " + " ".join(f"{i}:{self.named_as[tag]}" for i, (tag, _) in enumerate(waiting, start=1))[:240])

    def widen(self) -> None:
        """Too few listings fit by the words known: ask what else sellers call each value, and search for those."""
        self.widened = True
        for requirement in self.requirements:
            if not self.can_think():
                break
            answer = self.reader.fill(WORDS_GUIDE, f"Item: {self.item}\nProperty: {requirement['property']}\nValue: {requirement['value']}", WORDS_TOOL, 200) or {}
            words = [str(w).strip() for w in answer.get("words") or [] if str(w).strip()][:6]
            name = "written" if re.search(r"\d", requirement["value"]) else "alike"
            requirement[name] = list(dict.fromkeys(list(requirement.get(name) or []) + words))
            others = " ".join(r["value"] for r in self.requirements if r is not requirement)
            self.add_queries([f"{word} {others} {self.item}" for word in words[:3]], first=True)
        self.note(f"widened: {[(r['value'], r.get('written'), r.get('alike')) for r in self.requirements]}")

    def refill(self, lost: tuple[str, str], wide: bool = True) -> None:
        """The chosen variant has gone out of stock. A search shows one variant of each listing - the best in
        stock - so the searches that led to the lost one now show that listing's next variant, and its own title
        alone shows the same article listed again. wide: also search the values with it and the shelf at its price."""
        counter = self.counter
        words = word_list((counter.pages.get(lost[0]) or {}).get("title"))
        if len(words) < 2 or not counter.has("search"):
            return
        every = " ".join(self.values())
        own = " ".join(w for w in word_list(self.variant_of(lost).get("options")) if not w.replace(".", "").isdigit())
        queries = [" ".join(words), " ".join(words[:10])]
        if own:
            queries.append(f"{own} {' '.join(words[:10])}")       # the lost variant's own words: its nearest siblings share them
        queries += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.item} {' '.join(words[:3])}"]
        if wide:
            queries += [f"{every} {' '.join(words)}"] + self.shelf_queries(6)
        texts = list(dict.fromkeys(lowered(" ".join(q.split())[:QUERY_LENGTH]) for q in queries))
        calls = [act("search", query=text, k=10, in_stock=True) for text in texts]
        label, price = self.store_label(), (counter.stock.get(lost) or {}).get("price")
        if wide and label and isinstance(price, float) and counter.has("filter"):
            # the same shelf at the lost variant's price: variants of one listing share their price
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
        """Every value is stated as required - in the options, the property's own fact or the title - not only
        in another word for it or in some other fact."""
        return (self.verdicts.get(key) or {}).get("by") == "words" and all(self.plain(key, r) >= 2 for r in self.requirements)

    def firm(self, key: tuple[str, str]) -> bool:
        """Every value is stated where it counts most: the options or the property's own fact (an amount: the title too)."""
        if (self.verdicts.get(key) or {}).get("by") != "words":
            return False
        return all(self.plain(key, r) >= self.level_needed(r) for r in self.requirements)

    def look(self, rounds: int, enough: int = 2) -> None:
        """Search and read listings until enough of them fit, or the rounds run out. Every search loads the
        store, so none is sent once enough listings fit firmly."""
        counter = self.counter
        for round_number in range(rounds):
            if counter.over or counter.turns_left() <= TURN_RESERVE or self.late():
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
            told_new = self.shelf_queries(8) if self.fresh_rounds > 0 and len(fitting) < enough else []
            if told_new:
                # the shopper has just said what they want and too little fits: the new values with the words this
                # shelf's listings use reach the listing a shopper's own search would (a search shows ten rows, and
                # the title of a listing read before brings back that listing's kin, not what is wanted now)
                self.fresh_rounds -= 1
                # (half of the turn's searches: every value now in force with the item's own name, in each spelling)
                named = [q for q in (lowered(" ".join(str(v).split())[:QUERY_LENGTH]) for v in self.value_queries()) if q and q not in self.sent]
                named = list(dict.fromkeys(named))[: counter.group_size // 2]
                self.add_queries(told_new, first=True)
                self.queries = named + [q for q in self.queries if q not in named]
                searches = self.next_searches(counter.group_size)
            else:
                self.add_queries(self.kin_queries(), first=True)
                if len(fitting) < enough:
                    self.add_queries(self.variant_queries(), first=True)
                if len(self.queries) < counter.group_size // 2:
                    self.add_queries(self.shelf_queries(6))
                searches = self.shelf_filters(counter.group_size // 2)
                searches += self.next_searches(counter.group_size - len(searches))
            if not searches and not opens:
                break
            if searches:
                counter.send(searches)
                self.hear_quietly()
        opens = self.to_open(counter.group_size * OPEN_BATCH)
        if opens and not counter.over and counter.turns_left() > TURN_RESERVE:
            counter.send(opens[: counter.group_size])
            self.hear_quietly()
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
        forms = sorted(self.forms(requirement) | {" ".join(word_list(closed_up(requirement["value"])))}, key=len, reverse=True)
        for field in order:
            text = places.get(field) or ""
            for reading in (lowered(text), closed_up(text)):
                for form in forms:
                    if not form:
                        continue
                    found = re.search(r"(?<![a-z0-9])" + re.escape(form).replace(r"\ ", r"\W+") + r"(?![a-z0-9])", reading)
                    if found and f" {' '.join(word_list(found.group(0)))} " in f" {' '.join(word_list(text))} ":
                        return {"field": field, "value": found.group(0)}
        return None

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
                field = next((label for label in places if lowered(label) == lowered(where)), None)
                said = " ".join(word_list(words))
                if field and said and f" {said} " in f" {' '.join(word_list(places[field]))} ":
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
        missed = sorted((r for r in self.requirements if self.plain(key, r) < 1), key=lambda r: self.plain(key, r)) or list(self.requirements)
        for requirement in missed:
            # the property's own fact, when the page has one, says what the listing is instead
            for label, value in self.named_facts(page, requirement):
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
        """The shopper asked for a number of fitting options to be opened before the choice. Once everything is
        settled - the answers, the changes, what the market did - that many fitting listings (and a few to spare)
        are opened under the requirements as they now stand, the chosen one among them."""
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
        """The order for the choice. A listing that leaves a value unsaid, or words it another way, may still be what
        was asked for, so whatever is in the cart is ordered. Only when nothing could be put in the cart at all is
        the store told that no listing meets every requirement, the nearest one named with where it falls short."""
        if choice == self.named_nearest and self.can_abstain():
            # (the reasons as for an order: its price, its stock and the one place on its page that states what was
            # asked for - a shopper who asks for the reasons wants them for the listing named here too)
            return act("place_test_order", product_id=choice[0], sku=choice[1], abstain=True, justification=self.claims(choice))
        if choice in self.counter.cart or not self.can_abstain():
            return act("place_test_order", product_id=choice[0], sku=choice[1], justification=self.claims(choice))
        self.note(f"nothing could be put in the cart: abstain, closest {choice}")
        return act("place_test_order", product_id=choice[0], sku=choice[1], abstain=True, justification=self.shortfall(choice))

    def nearest_now(self, key: tuple[str, str], read: bool = True) -> tuple[str, str] | None:
        """The listing to name as the nearest, by the store's present word on it (read: its stock and price are read
        again first). When something within the limit fits after all, that is put in the cart and ordered; when the
        named listing has sold out or moved, the next such listing is named; when none is left, the nearest on sale."""
        counter = self.counter
        if read and counter.has("inspect_stock") and counter.turns_left() >= 2:
            counter.send([act("inspect_stock", product_id=key[0], sku=key[1])])
            if counter.heard and self.hear()[0]:
                self.judge()
        left = [k for k in self.ranked() if k not in self.gone]
        if left and max(self.trust(k) for k in left[:12]) < LIKELY_FIT and any(self.trust(k) >= SURE_FIT for k in self.only_over()):
            left = []       # (only loose readings fit within the limit: the plainly stated listing over it still stands)
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
            wanted = {singular(w) for form in self.forms(requirement) for w in form.split()}
            for label, value in self.named_facts(page, requirement):
                named = self.name_words(label)
                words = [w for entry in self.entries(value) for w in entry.split() if not any(same_word(w, n) for n in named)]
                if words and all(any(same_word(w, v) for v in wanted) for w in words):
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
        said_words = stems(line) | set(word_list(lowered(line)))
        words = [w for w in word_list(lowered(value)) if len(w) >= 3]
        return not words or any(w in said_words or stems(w) & said_words for w in words)

    @staticmethod
    def says_amount(text: str, amount: float) -> bool:
        """The amount stands in the text as a figure (however its thousands and cents are written)."""
        figures = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(text).replace(",", ""))]
        return any(abs(figure - amount) < 0.005 for figure in figures)

    def holds_now(self, said: str) -> tuple[list[dict], bool]:
        """Of what the shopper lines being answered newly required: the requirements that still hold, and whether
        one of the lines set the spending limit that holds."""
        lines = [line for line in list(self.fresh) + list(self.fresh_limit) if line and line in said]
        values = {lowered(v) for line in lines for v in self.fresh.get(line, [])}
        return [r for r in self.requirements if lowered(r["value"]) in values], any(line in self.fresh_limit for line in lines)

    def answer(self, line: str, beside: list[dict] | None = None) -> None:
        """Reply to the shopper lines that changed a requirement: one plain statement of what it is now. A line that
        changed none is owed no reply (a new spending limit is met by what is ordered), and a statement that goes
        over several things is easily taken for a question."""
        counter = self.counter
        if counter.over or not counter.has("message") or counter.turns_left() <= 2:
            return
        if line in self.unanswered:
            self.unanswered.remove(line)
        news, _ = self.holds_now(line)
        if not news and re.search(r"\bconfirm|\bright\?|\bok(?:ay)?\?", lowered(line)):
            # (the line asks to have its change confirmed, and what it asks for already held - an answer had said it
            # before: the shopper is owed the reply all the same, about the values the line names)
            news = [r for r in self.requirements if self.says_value(line, r["value"])]
        if not news:
            return
        parts = []
        for requirement in news:
            before = self.took_back.get(lowered(requirement["value"]))
            parts.append(f"the {requirement['property']} is now {requirement['value']}" + (f" instead of {before}" if before else ""))
        # (said so that it reads as the reply it is: yes, the change, and that nothing else moved - a shopper who asks
        # about the change often asks in the same breath whether the budget stands)
        counter.send([act("message", content=("Yes, confirmed: " + ", and ".join(parts) + ". Everything else stays the same, your budget included.")[:400])]
                     + list(beside or []))
        self.reply_sent = (counter.turn, line)

    def run(self) -> list[dict]:
        counter = self.counter
        question = self.read_request()
        self.opening, self.opening_budget = [dict(r) for r in self.requirements if r.get("stated")], self.budget
        # first turn: the first question (the request nearly always leaves something open) together with the first searches
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
            self.note(f"asking stopped: {type(exc).__name__}: {exc}")
            self.awaiting_answer = False
        try:
            self.settle_early()
        except Exception as exc:    # the early cart must never cost the order
            self.note(f"early cart stopped: {type(exc).__name__}: {exc}")
        self.look(rounds=9)
        try:
            self.cart_stand_in()
        except Exception as exc:    # the stand-in must never cost the order
            self.note(f"stand-in stopped: {type(exc).__name__}: {exc}")
        try:
            self.hunt_best()
        except Exception as exc:    # looking for a better listing must never cost the order
            self.note(f"search for the best stopped: {type(exc).__name__}: {exc}")
        commits = 0
        choice: tuple[str, str] | None = None
        while not counter.over and counter.turns_left() >= 2 and commits < 5:
            self.rank_by_priority(every=True)
            open_keys = [k for k in self.ranked() if k not in self.gone]
            weak = (bool(open_keys) and not self.search_asks and counter.turns_left() >= 14
                    and max(self.trust(k) for k in open_keys[:12]) < LIKELY_FIT)
            if (not open_keys or weak or not counter.cart and not any(self.keeps_request(k) for k in open_keys)) and self.rethink_searches():
                # nothing fits, nothing that fits can be relied on, or nothing for the first cart holds the request as
                # written: look with new searches first
                continue
            # (a listing within the limit that fits only in a loose reading does not stand against one that states
            # everything plainly and costs a little more: the plain one is what the request describes)
            loosely = bool(open_keys) and max(self.trust(k) for k in open_keys[:12]) < LIKELY_FIT
            if (not open_keys or loosely) and self.can_abstain():
                over = [k for k in self.only_over() if not open_keys or self.trust(k) >= SURE_FIT]
                if not over and not self.over_noted:
                    self.over_noted = True
                    near = [k for k in self.verdicts if self.just_over(k)]
                    self.note(f"nothing fits within {self.budget} and no listing just over it does: moved_on={self.moved_on} brand={self.brand!r} over the limit: {len(near)} "
                              + "; ".join(f"{str((counter.pages.get(k[0]) or {}).get('title'))[:28]}|{self.variant_of(k).get('options')}|{(counter.stock.get(k) or {}).get('price')}|"
                                          f"{(self.verdicts[k].get('met') or [])}|{self.verdicts[k].get('kind_ok')}" for k in near[:8]))
                if over:
                    # nothing within the limit meets every requirement, and a listing that does costs a little more:
                    # the shopper is told that nothing fits, with that listing named as the nearest
                    choice = self.named_nearest = over[0]
                    title = str((counter.pages.get(choice[0]) or {}).get("title") or "")[:50]
                    self.note(f"nothing fits within {self.budget}: nearest {choice} {title!r} at {(counter.stock.get(choice) or {}).get('price')} "
                              f"turn={counter.turn} pages={len(counter.pages)}")
                    break
            order = open_keys or [k for k in self.near_misses() if k not in self.gone]
            if not order:
                break
            if not counter.cart:
                # the first cart: what the shopper said matters more, then the request as written (stable sort)
                order.sort(key=lambda k: (not self.meets_chosen(k), not self.keeps_request(k)))
            choice = order[0]
            commits += 1
            title = str((counter.pages.get(choice[0]) or {}).get("title") or "")[:50]
            fitting = f"{len(self.ranked())}: " + "; ".join(f"{str((counter.pages.get(k[0]) or {}).get('title') or '')[:28]}|{self.variant_of(k).get('options')}|{[self.plain(k, r) for r in self.requirements]}|{(self.verdicts.get(k) or {}).get('by')}" for k in self.ranked()[:4])
            self.note(f"commit {commits}: {choice} {title!r} fits={self.fits(choice)} turn={counter.turn} pages={len(counter.pages)} spent={self.reader.spent:.4f} calls={self.reader.calls} fitting: {fitting}")
            group = []
            ahead = self.second_stand_in(choice) if choice not in counter.cart and counter.adds == 1 and not self.went_ahead else None
            if ahead is not None:
                # what a store does at the second cart falls on the listing put in then, as what it does at the first
                # falls on the first: one more listing goes in ahead of the choice, and out again with the stand-in
                self.went_ahead = True
                group.append(act("add_to_cart", product_id=ahead[0], sku=ahead[1]))
                if counter.has("inspect_stock"):
                    # (read again at once: what the store does to it has to be seen before the order, like any change)
                    group.append(act("inspect_stock", product_id=ahead[0], sku=ahead[1]))
                self.note(f"ahead of the choice: {ahead} {str((counter.pages.get(ahead[0]) or {}).get('title') or '')[:40]!r}")
            if choice not in counter.cart:
                group.append(act("add_to_cart", product_id=choice[0], sku=choice[1]))
            if counter.has("inspect_stock"):
                group.append(act("inspect_stock", product_id=choice[0], sku=choice[1]))
            if self.stand_in in counter.cart and self.stand_in != choice and counter.has("remove_from_cart"):
                group.append(act("remove_from_cart", product_id=self.stand_in[0], sku=self.stand_in[1]))
                counter.cart.remove(self.stand_in)
            if ahead is not None and counter.has("remove_from_cart"):
                group.append(act("remove_from_cart", product_id=ahead[0], sku=ahead[1]))
            counter.send(group)
            if ahead is not None and ahead in counter.cart and counter.has("remove_from_cart"):
                counter.cart.remove(ahead)
            seen = counter.stock.get(choice) or {}
            price = seen.get("price")
            over_budget = self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9
            changed, asks = self.hear()
            if seen.get("in_stock") is False or over_budget:
                self.gone.add(choice)
                self.lost_titles.add(lowered((counter.pages.get(choice[0]) or {}).get("title")))
                self.owed_reaction = True
                self.pending_line = ""     # the shopper's reaction to the change is answered by choosing again
                if counter.turns_left() > TURN_RESERVE and not self.late():
                    # the same article is often listed again, and it shares every fact the lost one was chosen for
                    self.refill(choice, wide=not any(self.firm(k) for k in self.ranked() if k not in self.gone))
                left = [k for k in self.ranked() if k not in self.gone]
                if len([k for k in left if self.firm(k)]) < 2 and counter.turns_left() > TURN_RESERVE and not self.late():
                    # fewer than two firm listings left: another sell-out would leave only weak ones, so look further now
                    # (the same article is often listed again, and the shelf has more to show)
                    self.kin_done.clear()
                    self.add_queries(self.kin_queries() + self.shelf_queries(6), first=True)
                    self.look(rounds=3 if left else 5, enough=2 if left else 1)
                choice = None
                continue
            if changed:
                self.judge()
                if not self.fits(choice):
                    self.gone.add(choice)
                    if counter.turns_left() > TURN_RESERVE and not self.late():
                        # the values as they now stand are searched before anything else, and what those searches
                        # show is opened first
                        fresh = self.next_searches(counter.group_size)
                        if fresh:
                            counter.send(fresh)
                            self.hear_quietly()
                        opens = self.to_open(counter.group_size * OPEN_BATCH, allowance=len(counter.pages) + 60)
                        if opens and not counter.over:
                            counter.send(opens[: counter.group_size])
                            self.hear_quietly()
                        self.judge()
                    if not [k for k in self.ranked() if k not in self.gone] and counter.turns_left() > TURN_RESERVE and not self.late():
                        # what the shopper changed is most often met by another variant of the same article, or by
                        # the article listed again: its own title, with the values as they now stand
                        self.refill(choice, wide=False)
                    if not [k for k in self.ranked() if k not in self.gone] and counter.turns_left() > TURN_RESERVE and not self.late():
                        self.look(rounds=4, enough=1)
                    choice = None
                    continue
            if asks and self.pending_line:
                self.answer(self.pending_line)
                self.pending_line = ""
                more_changed, _ = self.hear()
                if more_changed:
                    continue
            if counter.turn > int((counter.stock.get(choice) or {}).get("turn") or 0) and counter.has("inspect_stock") and counter.turns_left() >= 3:
                # a turn went by since the choice was last read: the market may have moved, so read it again before paying
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
                    # the shopper changed what they want while the choice waited: it must still fit
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
            # nothing left to order: the nearest listing still on sale is ordered rather than ending without an order
            # (it may be what was asked for in other words); where the store allows it, a listing that cannot even
            # be put in the cart is named as the closest instead
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
                # the shopper spoke while the options were opened: the line is taken in before the order
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
            # the order was refused (for instance the shopper has a line to read first): take the line in and try again
            changed, asks = self.hear()
            if asks and self.pending_line and counter.turns_left() >= 2:
                self.answer(self.pending_line)
                self.pending_line = ""
                changed = self.hear()[0] or changed
            if changed and counter.turns_left() >= 3:
                # what the shopper wants changed while the order waited: the choice must still fit
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
    except Exception as exc:  # the dialogue so far is still returned for diagnosis, after a last order attempt
        errand.counter.dialogue.append({"role": "assistant", "content": f"agent error: {type(exc).__name__}: {exc}"})
        try:
            errand.last_resort()
        except Exception as late:
            errand.counter.dialogue.append({"role": "assistant", "content": f"agent error at the last order: {type(late).__name__}: {late}"})
        return errand.counter.dialogue


__all__ = ["agent_main"]
