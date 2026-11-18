from __future__ import annotations

from os import getenv
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agent.proxy_client import ProxyClient

SEARCH, FILTER, VIEW, COMPARE = "search", "filter", "view", "compare"
INSPECT_STOCK, INSPECT_CART = "inspect_stock", "inspect_cart"
ADD, REMOVE, MESSAGE, ORDER = "add_to_cart", "remove_from_cart", "message", "place_test_order"
BRIEF, SHORTLIST, EXTRACT, JUDGE, PICK, CLAIMS, RESOLVE = "brief", "shortlist", "extract", "judge", "pick", "claims", "resolve"

GLM = ("z-ai/glm-5.2", "zai-org/GLM-5.2-TEE")
KIMI = ("moonshotai/kimi-k2.6", "moonshotai/Kimi-K2.6-TEE")
SEATS = {BRIEF: (GLM, GLM, KIMI), SHORTLIST: (GLM, GLM, KIMI), EXTRACT: (GLM, KIMI), JUDGE: (GLM, GLM, KIMI), PICK: (GLM, KIMI), CLAIMS: (GLM, KIMI), RESOLVE: (GLM, GLM, KIMI)}
OUTPUT_TOKENS = {BRIEF: 2048, SHORTLIST: 1024, EXTRACT: 3072, JUDGE: 2560, PICK: 256, CLAIMS: 512, RESOLVE: 512}
WAIT_SECONDS = {BRIEF: 30, SHORTLIST: 30, EXTRACT: 45, JUDGE: 40, PICK: 15, CLAIMS: 15, RESOLVE: 20}
TEMPERATURES = {BRIEF: 0.0, SHORTLIST: 0.1, EXTRACT: 0.0, JUDGE: 0.1, PICK: 0.1, CLAIMS: 0.0, RESOLVE: 0.0}
JUDGE_VOTES = 3
HEDGE_AT = 0.75
IN_FLIGHT = 3

TIME_LIMIT_SECONDS = 270
WRAP_UP_SECONDS = 235
HURRY_SECONDS = 170
EPISODE_DOLLARS = 0.06
EPISODE_DOLLARS_LAST = 0.08

SHORTLIST_ROWS = 10
MAKER_FILTERS = 2
MAKER_ROWS = 3
MODEL_ROWS = 3
FIGURE_ROWS = 2
EXTRACT_PAGES = 4
EXTRACT_IN_FLIGHT = 3
TABLE_ROWS = 20
PAGE_ROWS = 6
REPLACEMENT_OPENS = 6
SISTER_PRICE_ROOM = 1.1
RATE_LIMIT_RETRIES = 3

_proxy = ProxyClient(timeout=90, max_retries=1)

BRIEF_GUIDE = """You write down one shopper's order conditions as a single record, from their own words: the request and everything they said after it. A shopping assistant will shop against your record alone, so every clause the shopper wrote must land in the right field, and nothing they did not say may appear. Every field is required: write an empty string, an empty list, false or 0 where the shopper said nothing for it.
- product: the kind of thing, in plain words. category: the store category only when the shopper named one ('from the <name> category', 'in the <name> category'), spelled as they spelled it - otherwise empty. A brand they would pick is a maker preference, never a category.
- musts: each firm requirement the item itself must have, one per entry: text in the shopper's words; value, the requirement's figure as a bare number when it has one, null when it has none (a named standard such as a resolution name is a must without a figure); unit, written exactly as next to that figure; relation 'exact' unless the shopper's own words make a range (at least, at most, up to, or more, minimum, maximum) - then copy those words with the figure into range_words; a size, capacity, warranty or rating stated plainly is exact. Values and units exactly as written - never converted, rounded or completed. The money is not a must (it is max_price), being in stock is not a must, and buying one item is not a must.
- max_price: the most they will pay now, as a bare number - the newest amount when they changed it; null only when they never named an amount.
- exclude: what they ruled out.
- prefer: what they would like but can do without, one property per entry with its kind: maker (a brand or company), model (a model name or code, letter for letter, even when it reads like a quantity), colour, or other. A preference is never a must and never a priority.
- fall_back: the order in which they would give preferences up, in their words; empty when they gave none.
- priorities: only when the shopper gave a rule for ranking the listings against one another, its steps in their order, each with the direction that wins: higher when a larger figure wins, lower when a smaller figure wins, cheaper when the step is the price itself, none when it ranks no figure (a clause such as "never one another beats on all of them" is such a step). A wish for a brand, or "the firm requirements come first", is not a ranking rule.
- candidates: the listings they named to choose among, exactly as written - a product_id::sku key, or a name.
- phrase: only when the shopper asks for listings that match a phrase ('matching the phrase <words>', 'that matches "<words>"'): those words, letter for letter including odd spellings. Words describing the item are never a phrase: 'an in-stock <item> with <value>' asks for that item with that value, in stock. survey_count: how many different options they say to survey or look at, only when they give that number.
- unstated: a requirement they say they have but have not stated (the sentence telling you to ask); leave it empty once they have stated it.
- wants_reason: true only when they ask why, or for a reason or justification they can trust. update_promised: true only when they say they will change a requirement after the first cart choice.
- searches: three store searches of three or four words each, worded differently. For a requirement with a value: the item with the value written as pages write it, figure and unit joined (8GB, 500ml, 1080p); the item with the value written apart (8 GB, 500 ml); and, when they named a maker, the maker with the item and the value, otherwise the item and value in other words. For a named model: its code with the colour, the code with the item, and the code alone - the store's search reads whole pages, and a model code often stands only on a page's model line, not in its title. For a phrase request: the phrase itself, the phrase with the category name, and its words in another order.
- changes: when you rewrite after a new message, each field that message changed, with the old value, the new value and the shopper's words that changed it.
The newest message wins: a new amount or value replaces the old one from that moment, and everything the shopper did not change stays exactly as it was - their preferences included. An answer to a question you asked adds what it states: the category they name is their category, and a requirement they state is a must. A message that only asks about availability or a price, repeats the conditions or reacts to the store changes nothing: write the record as it was, with no changes."""

BRIEF_TOOL = {"type": "function", "function": {
    "name": BRIEF,
    "description": "The shopper's order conditions as they stand now, complete.",
    "parameters": {"type": "object", "properties": {
        "product": {"type": "string", "description": "the kind of thing"},
        "category": {"type": "string", "description": "only when the shopper named one, as they spelled it"},
        "musts": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the requirement in the shopper's words"},
            "value": {"type": ["number", "null"], "description": "its figure as a bare number; null when it has none"},
            "unit": {"type": "string", "description": "the unit written with the figure, as written; empty when none"},
            "relation": {"type": "string", "enum": ["exact", "at least", "at most"]},
            "range_words": {"type": "string", "description": "only for a range: the shopper's own words that make it one, with the figure, copied exactly; empty when exact"}},
            "required": ["text", "value", "unit", "relation"]}},
        "max_price": {"type": ["number", "null"], "description": "the most they will pay now; null only when they named no amount"},
        "exclude": {"type": "array", "items": {"type": "string"}},
        "prefer": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "kind": {"type": "string", "enum": ["maker", "model", "colour", "other"]}},
            "required": ["text", "kind"]}},
        "fall_back": {"type": "string"},
        "priorities": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "direction": {"type": "string", "enum": ["higher", "lower", "cheaper", "none"]}},
            "required": ["text", "direction"]}},
        "candidates": {"type": "array", "items": {"type": "string"}},
        "phrase": {"type": "string"},
        "survey_count": {"type": "integer"},
        "unstated": {"type": "string"},
        "wants_reason": {"type": "boolean"},
        "update_promised": {"type": "boolean"},
        "searches": {"type": "array", "items": {"type": "string"}},
        "changes": {"type": "array", "items": {"type": "object", "properties": {
            "field": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}, "said": {"type": "string"}},
            "required": ["field", "new"]}}},
        "required": ["product", "category", "musts", "max_price", "exclude", "prefer", "fall_back", "priorities", "candidates", "phrase",
                     "survey_count", "unstated", "wants_reason", "update_promised", "searches"]}}}

SHORTLIST_GUIDE = """You pick which search rows a careful personal shopper opens. A row shows a listing's title, its brand field, one variant's price and stock, and the searches or filters that returned it - it proves nothing else. Pick up to ten rows, the likeliest first: rows that could be the very item asked for. Rows carrying the shopper's own words (the model, the maker, the phrase, the requirement's figure with its unit) come first, then rows whose title names the kind of thing asked for. A maker the shopper prefers is only a preference: open the rows of other makers whose titles write the required figure too, since one of them serves when none of that maker's listings meets the requirement. The store's search reads whole pages, not only titles: a row that a search for the shopper's model or value returned may state it on its page though its title does not show it, so open such a row when it could be the kind of thing asked for - and when the shopper named a model, open every row of that kind the model's searches returned. Do not open accessories, parts, cases, refills or other kinds of things, nor a title spelling a brand or model with look-alike characters (a zero for an O). When the shopper named candidates, open every row that could be one of them. When the shopper wants a reason they can trust, open a title offering a second-hand or reconditioned unit last. When no row could be the item, say what to search for instead, in three or four words each."""

SHORTLIST_TOOL = {"type": "function", "function": {
    "name": SHORTLIST,
    "description": "The rows to open, likeliest first, with three or four words each on why; or what to search for instead.",
    "parameters": {"type": "object", "properties": {
        "open": {"type": "array", "maxItems": SHORTLIST_ROWS, "items": {"type": "object", "properties": {
            "product_id": {"type": "string"}, "why": {"type": "string"}}, "required": ["product_id", "why"]}},
        "search_instead": {"type": "array", "items": {"type": "string"}}},
        "required": ["open", "search_instead"]}}}

EXTRACT_GUIDE = """You copy what shop pages state, for a colleague who decides. Below are one shopper's requirements and the pages opened. For every variant of those pages that is in stock and within the shopper's money, write one row: the listing, the variant, any words on the page of a form the shopper ruled out (only when there are some), and - for each requirement - what the page states for it in each of three places, copied exactly as written with its unit: the facts (as 'label = value', only under a label that names that property: a figure under another property's label is not this requirement's), this variant's own options, and the title. Copy, never judge: a value that looks absurd is still what the page states, and a value in the page's own notation is copied as the page has it. A fact that lists several values the seller sells ('capacity = 350ml, 650ml', 'size = 12.5 & 14 inches') states none of them for one variant: leave the facts place empty for it - this variant's options say which one it is - while a value written with two parts ('1920x1080', '16GB+512GB') is one value. From the options copy only the part that states this requirement (14.0inch out of '840g3【14.0inch】i7 / 16g ram+512g ssd'), never the whole options string. Fill every place of every requirement, an empty string where that place states nothing - and look at the facts before you leave them empty: a label like ram, size or display is the facts speaking; from the title copy only the words that state the requirement. Answer with the extract tool alone."""

EXTRACT_TOOL = {"type": "function", "function": {
    "name": EXTRACT,
    "description": "One row per variant in stock within the shopper's money: what its page states for each requirement, as written.",
    "parameters": {"type": "object", "properties": {
        "rows": {"type": "array", "items": {"type": "object", "properties": {
            "product_id": {"type": "string"}, "sku": {"type": "string"},
            "excluded_words": {"type": "string", "description": "words of a ruled-out form found on the page, quoted; left out when none"},
            "values": {"type": "array", "items": {"type": "object", "properties": {
                "requirement": {"type": "string"},
                "facts": {"type": "string", "description": "'label = value' copied exactly; empty when no fact states it"},
                "options": {"type": "string", "description": "what this variant's options state for it; empty when silent"},
                "title": {"type": "string", "description": "the title's words that state it; empty when silent"}},
                "required": ["requirement", "facts", "options", "title"]}}},
            "required": ["product_id", "sku", "values"]}}},
        "required": ["rows"]}}}

JUDGE_GUIDE = """You are the careful personal shopper who decides which variant to buy for one person. Below are their conditions and a table of candidate variants the store showed, each in stock and within their money: the page's brand field, shelf, model line, facts, this variant's own options and title, and, for each requirement, what the page states for it, copied as written, with a note where the person's figure appears in those copies. The person is served only when the variant's own page states every firm requirement, so treat every candidate as wrong until its page states each must with the person's value.

READING THE EVIDENCE
- A requirement is stated by the page's own words: this variant's options, which say which variant arrives, a fact under that property's own label, or the title. When this variant's options state the property, they decide it, whatever the facts or the title say. The title is the seller's advertising - weaker than the options or facts, but still the seller's statement - so a value only the title states meets the requirement: mark the must met, quote the title, and count no page doubt for it. It gives way only to a candidate that serves the person's preferences as well and states the value in its options or facts: when the person prefers a maker and none of that maker's listings states the value more firmly, that maker's listing whose title states it fits and comes before every other maker's listing, and among listings that serve their preferences equally the one whose options or facts state the value comes first. A value the person has just changed must be stated in the options or facts, and when the person wants a reason only the facts count, since a reason quotes a fact. A fact listing several values the seller sells (350ml, 650ml; 12.5 & 14 inches), or a title listing several sizes or models, states none of them for this variant unless its options pin one. A figure inside a model name or under another property's label proves nothing about the requirement, and a bare number without its unit proves no measurement.
- Some statements are plainer than others. A fact under the label of that very property (screen size = 2.4 inches) or an option written in the person's own unit (40 inch) states the value plainly; the figure inside a label that describes something broader together with other values (display = 320*240; 2.4 inch tft), or written with a clipped or misspelt unit (40 inc), states it weakly. Among candidates that fit and serve the person's preferences equally, the plainer statement comes first - a preferred maker's listing still comes before a plainer listing of another maker.
- The value the person names is exactly the value they want: a nearer, larger, longer, higher or 'better' value is another value unless their own words make a range - a longer warranty, a bigger screen or a larger capacity does not meet a shorter, smaller or lesser one they named. The same amount written in the page's own notation or another unit is the same value: ask whether the page states the very amount they named, not whether it is enough. What no place states is unknown - never met, never "probably".
- The department the person named is proven by a shelf label that is that department's name; a neighbouring shelf that shares a word is another department. A listing on the named department is that kind of thing whatever its title says. When the person names the model they want, the page that states that model is what they asked for, and their description of it is not a second test. A candidate the person names in a list is one listing, named the way its page spells it: its brand field followed by its model line, character for character (a hyphen or a space is part of it), or, when the page has no model line, the first words of its title. Among several listings of the same product only the one whose page spells the name is that candidate: rank it on its own page's values, and when it is gone the candidate is gone - the next named candidate comes next.
- The brand is the brand field; the title counts only when the field is empty, and a title advertising a maker over a brand field naming another maker is the other maker's listing. The model the person named is found on the facts' model line - a code in a title or an option alone is advertising. A colour they named is met by an option spelled as they spelled it. A look-alike spelling of a brand or model (a zero for an O, a changed letter) is another product and the gravest doubt: a listing carrying one loses to any listing without one. When the person named no maker, the brand field is no reason for doubt - do not expect the maker a model name suggests.
- When the person wants a reason they can trust, choose a listing whose facts state the requested specification and whose brand field is filled; a title offering a refurbished or otherwise restored unit does not fit then, however cheap - never choose it while a new unit fits.
- The price is the price: a low price never disqualifies a listing that proves every must, and it is no reason to prefer one either - the budget is a ceiling, never a wish for the cheapest, so among fitting candidates the price decides only when the person ranked by it.

HOW TO DECIDE
1. For each candidate that could win - your choice and its closest rivals, four at most, or every candidate the person named - look first at its brand field and its shelf as the table writes them: the brand field, not the title, says whose listing it is, and the shelf says what kind of thing it is. Then write the strongest reason it may be the wrong item, in twelve words at most; then each of their musts, with met, unmet or unknown and the page's words you rely on, quoted exactly (ten words at most; empty when nothing states it) - the money and the stock need no line, since every candidate here is in stock within their money; then the preferences it meets; then its doubts, each in the page's own words and empty when there is none: another kind - words showing it is another kind of thing than asked, or a form the person ruled out; look-alike - a brand or model spelled with changed characters (a zero for an O); other model - only when the person named a model, a model line naming another; restored - a title offering a refurbished or otherwise restored unit; page - what makes the page less trustworthy (places contradicting each other, a fact that cannot be true). A brand field that differs from the title is none of these unless the person named that maker. Write under musts only the person's musts: the steps of their ranking rule are not musts, and a page that states nothing for a step still fits - it only ranks lower there. Say fits only when every must is met by its own page and it carries no another-kind, look-alike or other-model doubt that another candidate does not share, nor a restored doubt when the person wants a reason; a candidate the person named fits when it is in stock within their money and its page contradicts none of their musts.
2. Rank the fitting candidates. First the person's ranking rule, when they gave one: for each of its steps write the value this variant's options or facts state for that property - as written, even when it looks implausible, since you rank what the sellers state - and that value as a number in one unit for all candidates; a candidate whose options and facts state nothing for a step ranks after those that do, and a title figure only breaks a tie. The best on the first step wins and later steps only break ties; price counts only where they put it. Then what they prefer - the named model, then its colour, then the maker - giving preferences up only in the order they gave. Then the fewest doubts of the first four kinds, then the fewest page doubts, then the plainest statement of the musts, then the page that states the most requirements in its options or facts - never the lower price, unless the person ranked by price.
3. Choose the first of that ranking. When the person named candidates, only they are in the running, whatever they look like. When the person asked for listings matching a phrase, a listing matches when its title carries every word of the phrase as written, and every matching listing fits equally: choose the first. When nothing fits, say so, give three searches of three or four words for what to look for, and still choose the closest candidate.
4. When the person asked a question, write a one-line reply in plain words. When a reason is required, write claims about your choice, copied from its page: the requested specification under the facts' own label with the value exactly as written, the price as a bare number, in_stock true, and the brand field."""

JUDGE_TOOL = {"type": "function", "function": {
    "name": JUDGE,
    "description": "Your decision among the candidate variants, with the evidence it rests on.",
    "parameters": {"type": "object", "properties": {
        "candidates": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "key": {"type": "string", "description": "product_id::sku as the table writes it"},
            "against": {"type": "string", "description": "the strongest reason this may be the wrong item, twelve words at most"},
            "musts": {"type": "array", "items": {"type": "object", "properties": {
                "must": {"type": "string"},
                "verdict": {"type": "string", "enum": ["met", "unmet", "unknown"]},
                "quote": {"type": "string", "description": "the page's words you rely on, copied exactly, ten words at most; empty when nothing states it"}},
                "required": ["must", "verdict", "quote"]}},
            "ranks": {"type": "array", "items": {"type": "object", "properties": {
                "step": {"type": "string"}, "quote": {"type": "string"}, "number": {"type": "number"}},
                "required": ["step", "quote", "number"]},
                "description": "only when the person gave a ranking rule: each step's value as the options or facts state it (the title only when they state nothing), and as a number"},
            "preferences_met": {"type": "array", "items": {"type": "string"}},
            "doubts": {"type": "object", "properties": {
                "another_kind": {"type": "string", "description": "the page's words showing another kind of thing or a form the person ruled out; empty when none"},
                "look_alike": {"type": "string", "description": "a brand or model spelled with changed characters, as the page spells it; empty when none"},
                "other_model": {"type": "string", "description": "only when the person named a model: the model line naming another; empty otherwise"},
                "restored": {"type": "string", "description": "the title's words offering a refurbished or otherwise restored unit; empty when none"},
                "page": {"type": "string", "description": "what makes this page less trustworthy, a few words; empty when none"}},
                "required": ["another_kind", "look_alike", "other_model", "restored", "page"]},
            "fits": {"type": "boolean"}},
            "required": ["key", "against", "musts", "doubts", "fits"]}},
        "ranking": {"type": "array", "items": {"type": "string"}, "description": "the fitting keys, best first"},
        "choice": {"type": "string", "description": "product_id::sku of the variant to buy"},
        "why": {"type": "string", "description": "why this variant and no other, one sentence"},
        "none_fit": {"type": "boolean"},
        "searches": {"type": "array", "items": {"type": "string"}},
        "reply": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "field": {"type": "string"}, "value": {"type": "string"}}, "required": ["field", "value"]}}},
        "required": ["candidates", "choice", "why"]}}}

PICK_TOOL = {"type": "function", "function": {
    "name": PICK,
    "description": "The variant to buy.",
    "parameters": {"type": "object", "properties": {
        "choice": {"type": "string", "description": "product_id::sku of the variant to buy, as the table writes it"},
        "why": {"type": "string", "description": "why, twelve words at most"}},
        "required": ["choice", "why"]}}}
PICK_NOTE = "Answer now with the pick tool alone: the key of the variant to buy by the rules above, and why in a few words."

CLAIMS_GUIDE = """You write the justification of one order: short claims about the ordered variant, each copied from its page. A claim's field is price, in_stock, brand, or a label of the page's facts; its value is exactly as the page or the latest reading writes it - the requested specification under the facts' own label with its value as written, the latest price as a bare number, in_stock true, and the brand field. One false claim fails them all, so write only what the page below shows."""

CLAIMS_TOOL = {"type": "function", "function": {
    "name": CLAIMS,
    "description": "The claims of the order, each copied from the ordered variant's page.",
    "parameters": {"type": "object", "properties": {"claims": {"type": "array", "items": {"type": "object", "properties": {
        "field": {"type": "string"}, "value": {"type": "string"}}, "required": ["field", "value"]}}},
        "required": ["claims"]}}}

RESOLVE_GUIDE = """The person named the candidates they will choose among by name. For each name, find on the list below the one listing that is that candidate: the listing whose page spells the name - its brand field followed by its model line, character for character (a hyphen or a space is part of it), or, when the page has no model line, the first words of its title. Several listings may sell the same product under other brand fields, model lines or titles: only the one whose page spells the name is the candidate, and the store's other listings of it are not. When no listing spells a name, give that name no key. Answer with the resolve tool alone."""

RESOLVE_TOOL = {"type": "function", "function": {
    "name": RESOLVE,
    "description": "The listing each named candidate is.",
    "parameters": {"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string", "description": "The name as the person wrote it."},
        "key": {"type": "string", "description": "The key of the listing whose page spells that name, or an empty string."}},
        "required": ["name", "key"]}}}, "required": ["candidates"]}}}

QUESTION = ("Before I choose: you said you have a firm requirement you have not stated yet. What exactly is that requirement, "
            "with its exact value?")


def _openrouter() -> bool:
    return getenv("INFERENCE_PROVIDER", "openrouter") != "chutes"


def _native(model: Tuple[str, str]) -> str:
    return model[0] if _openrouter() else model[1]


def reasoning_off() -> Dict[str, Any]:
    return {"reasoning": {"enabled": False}} if _openrouter() else {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _post(path: str, body: Dict[str, Any], client: Any = _proxy) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        result = client.post_verbose(path, json_data=body)
        error = result.error or {}
        if error.get("status") == 429 and "capacity" in str(error.get("body") or "").lower():
            return None, dict(error, kind="capacity")
        if result.data is not None or error.get("status") != 429 or attempt == RATE_LIMIT_RETRIES:
            return result.data, error
        _pause(2 ** attempt * (1 + sum(map(ord, dumps(body.get("messages") or body.get("calls"))[-512:])) % 100 / 100))
    return None, {}


def _pause(seconds: float) -> None:
    box: Queue = Queue()
    try:
        box.get(timeout=max(0.0, seconds))
    except Empty:
        pass


def squash(text: Any) -> str:
    return " ".join(str(text).casefold().split())


_TYPOGRAPHY = str.maketrans({"“": '"', "”": '"', "″": '"', "‘": "'", "’": "'", "′": "'", "–": "-", "—": "-"})


def plain(text: Any) -> str:
    return squash(text).translate(_TYPOGRAPHY)


def words(text: Any) -> set:
    text = re.sub(r"[()\[\]{}\"/【】「」『』（）〈〉《》〔〕]", " ", str(text))
    return {word.strip("'.,;:!?") for word in squash(text).split()} - {""}


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def figures(text: Any) -> List[Tuple[float, str]]:
    found = []
    for match in re.finditer(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*-?\s*([^\W\d_]*[\"']?)", str(text)):
        value = number(match.group(1))
        if value is not None:
            found.append((value, match.group(2).lower()))
    return [(value, unit or next((later for figure, later in found[index + 1:] if figure == value and later), "")) for index, (value, unit) in enumerate(found)]


def unit_like(one: str, other: str) -> bool:
    return not one or not other or one[0] == other[0]


def states(text: Any, value: float, unit: str, relation: str) -> bool:
    for found, written in figures(text):
        if not unit_like(written, unit) or (unit and not written and not re.search(r"\d", str(unit))):
            continue
        if (relation == "at least" and found >= value) or (relation == "at most" and found <= value) or found == value:
            return True
    return False


def another_figure(text: Any, value: float, unit: str) -> Optional[float]:
    for found, written in figures(text):
        if unit and written and unit_like(written, unit) and found != value:
            return found
    return None


def money(text: str) -> List[float]:
    return [value for value in (number(figure.replace(",", "")) for figure in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*PHP\b", str(text))) if value is not None]


def keys_in(text: Any) -> List[str]:
    return [key for key in (piece.strip(".,;:()[]\"'") for piece in str(text).split() if "::" in piece) if key]


def over(price: Any, budget: Optional[float]) -> bool:
    return budget is not None and number(price) is not None and number(price) > budget


def fit(value: Any, schema: Dict[str, Any]) -> Tuple[bool, Any]:
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((option for option in kind if option != "null"), None)
    if kind == "object":
        if not isinstance(value, dict):
            return False, value
        declared = schema.get("properties") or {}
        fitted = {}
        for name, given in value.items():
            if name in declared and given is not None:
                ok, fitted[name] = fit(given, declared[name])
                if not ok:
                    fitted.pop(name)
        return all(name in fitted for name in schema.get("required") or []), fitted
    if kind == "array":
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if not isinstance(value, list):
            return False, value
        items = [fit(item, schema.get("items") or {}) for item in value]
        return True, [item for ok, item in items if ok]
    if kind == "string":
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value = "%g" % value
        options = schema.get("enum")
        if isinstance(value, str) and isinstance(options, list) and value not in options:
            same = [option for option in options if squash(option) == squash(value)]
            return bool(same), same[0] if same else value
        return isinstance(value, str), value
    if kind in ("number", "integer"):
        found = number(value)
        if found is None and isinstance(value, str):
            written = figures(value)
            found = written[0][0] if written else None
        if found is None:
            return False, value
        return True, int(found) if kind == "integer" else found
    if kind == "boolean":
        if isinstance(value, str) and squash(value) in ("true", "false"):
            return True, squash(value) == "true"
        return isinstance(value, bool), value
    return True, value


def cut_short(reply: Any) -> bool:
    first = (reply.get("choices") or [None])[0] if isinstance(reply, dict) else None
    return isinstance(first, dict) and "length" in (first.get("finish_reason"), first.get("native_finish_reason"))


def tool_args(reply: Any, name: str) -> Optional[Dict[str, Any]]:
    first = (reply.get("choices") or [None])[0] if isinstance(reply, dict) else None
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict) or cut_short(reply):
        return None
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                continue
        if isinstance(args, dict):
            return args
    return None


def stock_word(in_stock: Any) -> str:
    return "in stock" if in_stock else "SOLD OUT"


def render_rows(rows: List[Dict[str, Any]], budget: Optional[float], keep: set, phrase: set, found: Dict[str, List[str]]) -> str:
    shown = [row for row in rows if not over(row.get("price"), budget) or str(row.get("product_id")) in keep]
    if phrase:
        carrying = [row for row in shown if phrase <= words(row.get("title"))]
        if len(carrying) >= 3:
            shown = carrying
    lines = ["product_id=%s | %s PHP | %s | brand field: %s | %s | found by: %s" % (
        row.get("product_id"), row.get("price"), stock_word(row.get("in_stock")), row.get("brand") or "(empty)", str(row.get("title") or "")[:160],
        "; ".join((found.get(str(row.get("product_id"))) or [])[:3]) or "-") for row in shown]
    return "\n".join(lines) or "(no rows within the budget)"


def render_page(page: Dict[str, Any]) -> str:
    lines = ["product_id=%s | brand field: %s | title: %s" % (page.get("product_id"), page.get("brand") or "(empty)", page.get("title")),
             "  category_path: %s" % " > ".join(str(label) for label in page.get("category_path") or []),
             "  facts: %s" % ("; ".join("%s = %s" % (label, value) for label, value in (page.get("facts") or {}).items()) or "(none)")]
    for variant in page.get("variants") or []:
        if isinstance(variant, dict):
            lines.append("  variant sku=%s | options: %s | %s PHP | %s" % (variant.get("sku"), variant.get("options") or "(none)",
                                                                         variant.get("price"), stock_word(variant.get("in_stock"))))
    return "\n".join(lines)


def stem(sku: Any) -> str:
    return str(sku).rsplit("-", 1)[0]


class Session:
    def __init__(self, binding: Dict[str, Any], problem_id: str, max_steps: int) -> None:
        self.binding = binding
        self.problem_id = problem_id
        self.max_steps = max_steps
        self.turn = 1
        self._generation = 0
        self.dead = False
        self.done = False
        self.dialogue: List[Dict[str, Any]] = []

    def turns_left(self) -> int:
        return self.max_steps - self.turn + 1

    def send(self, actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for attempt, turn in enumerate((self.turn, self.turn, self.turn + 1)):
            if self.dead:
                return None
            base = "%s-t%d-g%d" % (self.problem_id, turn, self._generation)
            envelope = {**self.binding, "call_id": base, "idempotency_key": base, "turn": turn,
                        "calls": [{"call_id": "%s-c%d" % (base, i), "action": {"name": a["name"], "args": a["args"]}} for i, a in enumerate(actions, 1)]}
            try:
                result, error = _post("/environment/call", envelope)
            except Exception:
                result, error = None, {}
            status = error.get("status")
            if status == 402 or (status in (409, 500, 504) and "quarantined" in str(error.get("body") or "").lower()):
                self.dead = True
                return None
            if isinstance(result, dict) and isinstance(result.get("calls"), list):
                count = result.get("solver_turn_count")
                self.turn = (count if isinstance(count, int) and not isinstance(count, bool) else turn) + 1
                return result
            if attempt:
                self._generation += 1
        return None


class Clock:
    def __init__(self) -> None:
        self.started = monotonic()

    def elapsed(self) -> float:
        return monotonic() - self.started


class Models:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.dollars = 0.0
        self.flying = 0
        self.lock = Lock()

    def spend(self, reply: Any) -> None:
        usage = reply.get("usage") if isinstance(reply, dict) else None
        if not isinstance(usage, dict):
            return
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            cost = (number(usage.get("prompt_tokens")) or 0) * 0.643e-6 + (number(usage.get("completion_tokens")) or 0) * 2.02e-6
        with self.lock:
            self.dollars += float(cost)

    def attempt(self, model: Tuple[str, str], messages: List[Dict[str, Any]], tool: Dict[str, Any], seat: str, wait: float) -> Optional[Dict[str, Any]]:
        name = tool["function"]["name"]
        body = {"model": _native(model), "messages": messages, "tools": [tool], "tool_choice": {"type": "function", "function": {"name": name}},
                "temperature": TEMPERATURES[seat], "max_tokens": OUTPUT_TOKENS[seat], **reasoning_off()}
        client = ProxyClient(timeout=max(3, int(wait)), max_retries=1)
        try:
            reply, error = _post("/inference/chat/completions", body, client)
            if reply is None and error.get("status") in (400, 422) and "tool_choice" in str(error.get("body") or ""):
                reply, error = _post("/inference/chat/completions", dict(body, tool_choice="required"), client)
        except Exception:
            reply = None
        self.spend(reply)
        args = tool_args(reply, name)
        return args

    def ask(self, seat: str, messages: List[Dict[str, Any]], tool: Dict[str, Any], deadline: Optional[float] = None, inline: bool = False) -> Optional[Dict[str, Any]]:
        chain = list(SEATS[seat])
        limit = deadline if deadline is not None else TIME_LIMIT_SECONDS - 10
        while chain:
            left = limit - self.clock.elapsed()
            if left < 5:
                return None
            wait = min(WAIT_SECONDS[seat], left - 2)
            first = chain.pop(0)
            if inline:
                answer = self.attempt(first, messages, tool, seat, wait)
                if answer is not None:
                    return answer
                continue
            partner = chain[0] if chain and self.dollars < EPISODE_DOLLARS else None
            box: Queue = Queue()
            started = 0

            def launch(model: Tuple[str, str], seconds: float) -> bool:
                with self.lock:
                    if self.flying >= IN_FLIGHT:
                        return False
                    self.flying += 1

                def run() -> None:
                    try:
                        box.put(self.attempt(model, messages, tool, seat, seconds))
                    except Exception:
                        box.put(None)
                    finally:
                        with self.lock:
                            self.flying -= 1
                try:
                    Thread(target=run, daemon=True).start()
                except RuntimeError:
                    with self.lock:
                        self.flying -= 1
                    return False
                return True

            if launch(first, wait):
                started = 1
            else:
                answer = self.attempt(first, messages, tool, seat, wait)
                if answer is not None:
                    return answer
                continue
            begun, answers, until = monotonic(), 0, wait
            while answers < started:
                now = monotonic() - begun
                if partner is not None and started == 1 and now >= wait * HEDGE_AT:
                    spare = min(wait, limit - self.clock.elapsed() - 2)
                    if spare > 3 and launch(partner, spare):
                        started = 2
                        until = now + spare
                    partner = None
                    continue
                timeout = (wait * HEDGE_AT if partner is not None and started == 1 else until) - now + 0.5
                try:
                    answer = box.get(timeout=max(0.05, timeout))
                except Empty:
                    if partner is not None:
                        continue
                    break
                answers += 1
                if answer is not None:
                    return answer
                if partner is not None and started == 1:
                    break
            if started == 2 and chain:
                chain.pop(0)
        return None

    def votes(self, seat: str, messages: List[Dict[str, Any]], tool: Dict[str, Any], count: int, deadline: Optional[float] = None) -> List[Dict[str, Any]]:
        limit = deadline if deadline is not None else TIME_LIMIT_SECONDS - 10
        left = limit - self.clock.elapsed()
        if left < 5:
            return []
        wait, model, box, started = min(WAIT_SECONDS[seat], left - 2), SEATS[seat][0], Queue(), 0
        for _ in range(count):
            with self.lock:
                if self.flying >= IN_FLIGHT:
                    break
                self.flying += 1

            def run() -> None:
                try:
                    box.put(self.attempt(model, messages, tool, seat, wait))
                except Exception:
                    box.put(None)
                finally:
                    with self.lock:
                        self.flying -= 1
            try:
                Thread(target=run, daemon=True).start()
                started += 1
            except RuntimeError:
                with self.lock:
                    self.flying -= 1
                break
        answers: List[Dict[str, Any]] = []
        begun = monotonic()
        for _ in range(started):
            try:
                answer = box.get(timeout=max(0.05, wait + 1 - (monotonic() - begun)))
            except Empty:
                break
            if answer is not None:
                answers.append(answer)
                choices = [squash(item.get("choice")) for item in answers]
                if any(choices.count(choice) * 2 > count for choice in choices):
                    break
        return answers


class Episode:
    def __init__(self, problem_data: Dict[str, Any]) -> None:
        environment = problem_data["environment"]
        view = environment["policy_view"]
        self.problem_id = str(problem_data.get("problem_id") or "problem")
        self.clock = Clock()
        self.models = Models(self.clock)
        self.request = str(view["query"])
        self.session = Session(dict(environment["binding"]), self.problem_id, int(view["max_steps"]))
        self.schemas: Dict[str, Dict[str, Any]] = {}
        for tool in view["tools"]:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                self.schemas[function["name"]] = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        self.per_turn = max(1, min(int(view.get("max_calls_per_turn") or 1), 16))
        self.claims_required = "justification" in (self.schemas.get(ORDER, {}).get("required") or [])
        self.record: Dict[str, Any] = {}
        self.request_keys = {squash(key) for key in keys_in(self.request)}
        self.said: List[str] = []
        self.heard = 0
        self.question_open = False
        self.asked = False
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.opened_order: List[str] = []
        self.live: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.row_order: List[str] = []
        self.search_rows: List[Tuple[str, List[str]]] = []
        self.maker_rows: List[str] = []
        self.figure_rows: List[str] = []
        self.searches = 0
        self.evidence: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.extracted: set = set()
        self.cart: List[Tuple[str, str]] = []
        self.first_add: Optional[Tuple[str, str]] = None
        self.lost: set = set()
        self.last_judge: Optional[Dict[str, Any]] = None
        self.backups: List[Tuple[str, str]] = []
        self.reply = ""
        self.claims: Dict[Tuple[str, str], List[Any]] = {}
        self.changed_musts: List[Dict[str, Any]] = []
        self.budget_dropped = False
        self.old_price: Optional[float] = None
        self.dropped_musts: List[Dict[str, Any]] = []
        self.found_by: Dict[str, List[str]] = {}
        self.model_rows: List[str] = []
        self.instead: List[str] = []
        self.opened_before_add = 0
        self.rank_memory: Dict[Tuple[Tuple[str, str], str], float] = {}

    def declared(self, name: str) -> Dict[str, Any]:
        return self.schemas.get(name, {}).get("properties") or {}

    def budget(self) -> Optional[float]:
        return number(self.record.get("max_price"))

    def musts(self) -> List[Dict[str, Any]]:
        return [must for must in self.record.get("musts") or [] if isinstance(must, dict) and must.get("text")]

    def figure_musts(self) -> List[Dict[str, Any]]:
        return [must for must in self.musts() if number(must.get("value")) is not None]

    def named_keys(self) -> set:
        return {squash(item) for item in self.record.get("candidates") or [] if "::" in str(item)} | self.request_keys

    def named_names(self) -> List[str]:
        return [str(item) for item in self.record.get("candidates") or [] if "::" not in str(item)]

    def makers(self) -> List[str]:
        return [str(item.get("text")) for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("kind") == "maker" and item.get("text")]

    def phrase_words(self) -> set:
        return words(self.record.get("phrase")) if self.record.get("phrase") else set()

    def phrase_holds(self, record: Dict[str, Any]) -> bool:
        phrase = words(record.get("phrase")) if record.get("phrase") else set()
        return bool(phrase) and sum(1 for row in self.rows.values() if phrase <= words(row.get("title"))) >= 3

    def models_named(self) -> List[str]:
        return [str(item.get("text")) for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("kind") == "model" and item.get("text")]

    def orderable(self, key: Tuple[str, str]) -> bool:
        in_stock, price = self.live.get(key, (None, None))
        return in_stock is not False and not over(price, self.budget()) and (not self.named_keys() or "%s::%s" % key in self.named_keys())

    def late(self) -> bool:
        return (self.clock.elapsed() > WRAP_UP_SECONDS or self.session.turns_left() <= 3 or self.session.dead
                or self.models.dollars > EPISODE_DOLLARS_LAST)

    def constrained(self, name: str, args: Dict[str, Any], cap: Optional[float] = None) -> Dict[str, Any]:
        declared = self.declared(name)
        args = {key: value for key, value in args.items() if key in declared and value not in (None, "")}
        if "in_stock" in declared:
            args["in_stock"] = True
        if "k" in declared:
            args["k"] = 10
        if name == FILTER and "max_price" in declared:
            ceiling = [value for value in (cap, self.budget()) if value is not None]
            if ceiling:
                args["max_price"] = round(min(ceiling), 2)
        return {"name": name, "args": args}

    def send(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        calls = [call for call in calls if call["name"] in self.schemas][:self.per_turn]
        if not calls or self.session.done or self.session.turns_left() <= 0:
            return []
        turn = self.session.turn
        result = self.session.send(calls)
        if result is None:
            raise RuntimeError("the store did not answer")
        self.session.dialogue.append({"role": "environment", "turn": turn, "request": calls, "result": result})
        observations = []
        for index, call in enumerate(calls):
            outcome = result["calls"][index] if index < len(result["calls"]) and isinstance(result["calls"][index], dict) else {}
            step = outcome.get("observation") if isinstance(outcome.get("observation"), dict) else {}
            if step.get("done") is True:
                self.session.done = True
            obs = step.get("observation") if isinstance(step.get("observation"), dict) else {}
            if not step.get("error") and not obs.get("error"):
                self.learn(call, obs)
            observations.append(obs)
        said = result.get("user_message")
        text = str(said.get("content") or "").strip() if isinstance(said, dict) else ""
        if text:
            self.said.append(text)
            self.question_open = "?" in text
        return observations

    def learn(self, call: Dict[str, Any], obs: Dict[str, Any]) -> None:
        name, args = call["name"], call["args"]
        key = (str(args.get("product_id") or ""), str(args.get("sku") or ""))
        if name in (SEARCH, FILTER):
            rows = [row for row in obs.get("results") or [] if isinstance(row, dict)]
            source = "search %s" % dumps(str(args.get("query") or "")) if name == SEARCH else "filter %s" % dumps({k: args[k] for k in ("category", "brand") if args.get(k)})
            for row in rows:
                product_id = str(row.get("product_id"))
                if product_id not in self.rows:
                    self.rows[product_id] = row
                    self.row_order.append(product_id)
                if source not in self.found_by.setdefault(product_id, []):
                    self.found_by[product_id].append(source)
                self.note(product_id, row.get("sku"), row.get("in_stock"), row.get("price"))
            if name == SEARCH:
                self.searches += 1
                self.search_rows.append((str(args.get("query") or ""), [str(row.get("product_id")) for row in rows]))
        elif name in (VIEW, COMPARE):
            for page in ([obs] if name == VIEW else obs.get("compared") or []):
                if isinstance(page, dict) and page.get("product_id"):
                    product_id = str(page["product_id"])
                    if product_id not in self.pages:
                        self.opened_order.append(product_id)
                    self.pages[product_id] = page
                    for variant in page.get("variants") or []:
                        if isinstance(variant, dict):
                            self.note(product_id, variant.get("sku"), variant.get("in_stock"), variant.get("price"))
        elif name == INSPECT_STOCK:
            self.note(key[0], key[1], obs.get("in_stock"), obs.get("price"))
        elif name == ADD:
            added = obs.get("added") if isinstance(obs.get("added"), dict) else {}
            if key not in self.cart:
                self.cart.append(key)
            if self.first_add is None:
                self.opened_before_add = len(self.opened_order)
            self.first_add = self.first_add or key
            self.note(key[0], key[1], added.get("in_stock"), added.get("price"))
        elif name == MESSAGE:
            self.question_open = False

    def note(self, product_id: Any, sku: Any, in_stock: Any, price: Any) -> None:
        if product_id and sku and isinstance(in_stock, bool):
            self.live[(str(product_id), str(sku))] = (in_stock, price)

    def write_conditions(self) -> None:
        newest = "\n".join(self.said[self.heard:])
        said = "\n".join("- " + text for text in self.said) or "(nothing yet)"
        before = "\n\nThe conditions as you wrote them before the newest message: %s" % dumps(self.record) if self.record else ""
        messages = [{"role": "system", "content": BRIEF_GUIDE},
                    {"role": "user", "content": "The shopper's request:\n%s\n\nWhat the shopper said since, oldest first:\n%s%s\n\nWrite the order conditions as they stand now." % (self.request, said, before)}]
        args = self.models.ask(BRIEF, messages, BRIEF_TOOL)
        heard_words = words(self.request + " " + " ".join(self.said))
        ungrounded = lambda record: [text for text in [str(item.get("text")) for item in record.get("prefer") or [] if isinstance(item, dict) and item.get("text")]
                                     + [str(record.get("category") or ""), str(record.get("phrase") or "")]
                                     if text and not any(len(word) >= 3 and word in heard_words for word in words(text))]
        if args is not None and ungrounded(args):
            again = self.models.ask(BRIEF, messages, BRIEF_TOOL)
            if again is not None and not ungrounded(again):
                args = again
        if args is not None and ungrounded(args):
            bad = set(ungrounded(args))
            args["prefer"] = [item for item in args.get("prefer") or [] if not (isinstance(item, dict) and str(item.get("text")) in bad)]
            for name in ("category", "phrase"):
                if str(args.get(name) or "") in bad:
                    args[name] = ""
        old = dict(self.record)
        if args is not None:
            ok, record = fit(args, BRIEF_TOOL["function"]["parameters"])
            record = {key: value for key, value in record.items() if value not in (None, "", []) and value is not False}
        else:
            record = dict(old) if old else {"product": self.request[:120], "musts": [], "searches": [self.request[:60]]}
            if not old:
                amounts = money(self.request)
                record["max_price"] = amounts[0] if len(amounts) == 1 else None
        if old:
            for name in ("product", "category", "musts", "max_price", "exclude", "prefer", "fall_back", "priorities", "candidates", "phrase", "survey_count", "wants_reason", "update_promised"):
                if record.get(name) in (None, "", []) and old.get(name) not in (None, "", []) and not (words(dumps(old.get(name))) & words(newest)):
                    record[name] = old[name]
            if self.asked and self.said:
                record.pop("unstated", None)
            cut = self.budget_cut(newest, number(old.get("max_price")))
            if cut is not None:
                record["max_price"] = cut
        elif record.get("max_price") is None:
            amounts = money(self.request)
            if len(amounts) == 1:
                record["max_price"] = amounts[0]
        same = lambda one, other: squash(one.get("text")) == squash(other.get("text")) or (
            number(one.get("value")) is not None and number(one.get("value")) == number(other.get("value")) and unit_like(squash(one.get("unit")), squash(other.get("unit"))))
        if isinstance(record.get("musts"), list):
            record["musts"] = [must for must in record["musts"] if not (isinstance(must, dict) and squash(must.get("unit")) == "php")]
        musts_now = [must for must in record.get("musts") or [] if isinstance(must, dict)]
        musts_before = [must for must in old.get("musts") or [] if isinstance(must, dict)] if old else []
        if old and newest:
            self.changed_musts = [must for must in musts_now if not any(same(must, other) for other in musts_before)]
            dropped = [must for must in musts_before if not any(same(must, other) for other in musts_now)]
            self.dropped_musts = dropped
            if len(dropped) == 1 and len(self.changed_musts) == 1 and unit_like(squash(dropped[0].get("unit")), squash(self.changed_musts[0].get("unit"))):
                for row in self.evidence.values():
                    for cell in row["values"]:
                        if squash(cell["requirement"]) == squash(dropped[0].get("text")):
                            cell["requirement"] = str(self.changed_musts[0].get("text"))
            elif self.changed_musts:
                self.extracted.clear()
        heard = plain(self.request + " " + " ".join(self.said))
        for must in record.get("musts") or []:
            quote = plain(must.get("range_words") or "") if isinstance(must, dict) else ""
            if isinstance(must, dict) and must.get("relation") not in (None, "exact") and not (quote and quote in heard and states(quote, float(number(must.get("value")) or 0), squash(must.get("unit")), "exact")):
                must["relation"] = "exact"
        if self.rows and record.get("phrase") and not self.phrase_holds(record):
            record.pop("phrase")
        self.record = record
        self.heard = len(self.said)

    def budget_cut(self, text: str, current: Optional[float]) -> Optional[float]:
        prices = {number(self.live.get(key, (None, None))[1]) for key in self.cart}
        below = {amount for amount in money(text) if current is not None and amount < current and amount not in prices}
        return below.pop() if len(below) == 1 else None

    def hear(self) -> bool:
        if self.heard >= len(self.said) or self.clock.elapsed() > WRAP_UP_SECONDS:
            return False
        before = dumps({key: self.record.get(key) for key in ("musts", "max_price", "category", "prefer", "candidates", "phrase")})
        self.write_conditions()
        return before != dumps({key: self.record.get(key) for key in ("musts", "max_price", "category", "prefer", "candidates", "phrase")})

    def first_look(self) -> None:
        calls: List[Dict[str, Any]] = []
        budget = self.budget()
        keyed = sorted({key.split("::")[0] for key in self.named_keys()})
        if keyed and COMPARE in self.schemas:
            calls += [{"name": COMPARE, "args": {"product_ids": keyed[i:i + 5]}} for i in range(0, len(keyed), 5)]
        queries = [str(query) for query in self.record.get("searches") or [] if str(query).strip()][:3]
        if not keyed:
            queries += [name for name in self.named_names() if name not in queries][:4]
            queries += [code for code in self.models_named()[:1] if squash(code) not in {squash(query) for query in queries}]
            if not queries:
                queries = [str(self.record.get("product") or self.request)[:60]]
            calls += [self.constrained(SEARCH, {"query": query}) for query in queries]
            calls += [dict(self.constrained(SEARCH, {"query": query}), figure=True) for query in self.maker_figure_queries(queries)]
            shelf = str(self.record.get("category") or self.record.get("product") or "")
            if shelf:
                calls.append(self.constrained(FILTER, {"category": shelf}))
            for maker in self.makers()[:MAKER_FILTERS]:
                calls.append(dict(self.constrained(FILTER, {"category": shelf, "brand": maker}), maker=True))
                calls.append(dict(self.constrained(FILTER, {"brand": maker}), maker=True))
        if self.record.get("unstated") and MESSAGE in self.schemas and not self.asked:
            calls.append({"name": MESSAGE, "args": {"content": QUESTION}})
            self.asked = True
        calls = calls[:self.per_turn]
        before = set(self.rows)
        observations = self.send([{"name": call["name"], "args": call["args"]} for call in calls])
        codes = {squash(code) for code in self.models_named()}
        for call, obs in zip(calls, observations):
            if call["name"] == SEARCH and squash(call["args"].get("query")) in codes:
                self.model_rows += [str(row.get("product_id")) for row in obs.get("results") or [] if isinstance(row, dict) and row.get("in_stock") is not False
                                    and not over(row.get("price"), budget) and str(row.get("product_id")) not in self.model_rows]
            if call.get("figure"):
                self.figure_rows += [str(row.get("product_id")) for row in obs.get("results") or [] if isinstance(row, dict) and row.get("in_stock") is not False
                                     and not over(row.get("price"), budget) and str(row.get("product_id")) not in self.figure_rows]
            if call.get("maker"):
                self.maker_rows += [str(row.get("product_id")) for row in obs.get("results") or [] if isinstance(row, dict) and row.get("in_stock") is not False
                                    and not over(row.get("price"), budget) and str(row.get("product_id")) not in self.maker_rows]

    def maker_figure_queries(self, done: List[str]) -> List[str]:
        if self.phrase_words() or not self.makers() or not self.figure_musts():
            return []
        must = self.figure_musts()[0]
        base = "%s %s %g" % (self.makers()[0], str(self.record.get("product") or "").strip(), float(number(must.get("value"))))
        unit = str(must.get("unit") or "").strip()
        seen = {squash(query) for query in done}
        return [query for query in dict.fromkeys((base + unit, base + " " + unit)) if squash(query) not in seen]

    def titled_rows(self) -> List[str]:
        if not self.makers():
            return []
        budget = self.budget()
        return [product_id for product_id in self.row_order if self.rows[product_id].get("in_stock") is not False and not over(self.rows[product_id].get("price"), budget)
                and any(states(self.rows[product_id].get("title"), float(number(must.get("value"))), squash(must.get("unit")), str(must.get("relation") or "exact"))
                        for must in self.figure_musts())][:FIGURE_ROWS]

    def shortlist(self, only: Optional[List[str]] = None, lead: Optional[List[str]] = None, limit: int = 0, note: str = "") -> List[str]:
        budget = self.budget()
        rows = [self.rows[product_id] for product_id in (self.row_order if only is None else only) if product_id not in self.pages and self.rows[product_id].get("in_stock") is not False]
        if not rows:
            return []
        keep = {key.split("::")[0] for key in self.named_keys()}
        phrase = self.phrase_words()
        want = limit or max(SHORTLIST_ROWS, int(number(self.record.get("survey_count")) or 0))
        named_rows = [product_id for query, found in self.search_rows if query in self.named_names() for product_id in found[:3]] if only is None else []
        messages = [{"role": "system", "content": SHORTLIST_GUIDE},
                    {"role": "user", "content": "Order conditions: %s\n\n%sThe rows:\n%s" % (dumps(self.record), note + "\n\n" if note else "", render_rows(rows, budget, keep, phrase, self.found_by))}]
        args = self.models.ask(SHORTLIST, messages, SHORTLIST_TOOL, deadline=HURRY_SECONDS + 20)
        picked = [str(item.get("product_id")) for item in (args or {}).get("open") or [] if isinstance(item, dict)]
        if args is None:
            picked = [str(row.get("product_id")) for row in rows if not over(row.get("price"), budget)][:want]
        self.instead = [str(query) for query in (args or {}).get("search_instead") or [] if str(query).strip()][:3]
        if only is None:
            model_rows = sorted(self.model_rows, key=lambda product_id: (not any(source.startswith("filter") for source in self.found_by.get(product_id, [])),
                                                                      -sum(source.startswith("search") for source in self.found_by.get(product_id, []))))
            front = self.figure_rows[:FIGURE_ROWS] + self.titled_rows() + self.maker_rows[:MAKER_ROWS] + model_rows[:MODEL_ROWS] + named_rows
        else:
            front = list(lead or [])
        ids = [product_id for product_id in dict.fromkeys(front + [product_id for product_id in picked if product_id in self.rows]) if product_id not in self.pages]
        if phrase:
            ids = [product_id for product_id in ids if phrase <= words(self.rows.get(product_id, {}).get("title"))] or ids
        return ids[:max(want, SHORTLIST_ROWS + MAKER_ROWS + MODEL_ROWS + 2 * FIGURE_ROWS)] if only is None else ids[:want]

    def gather(self, calls: List[Dict[str, Any]]) -> List[str]:
        sent = [call for call in calls if call["name"] in self.schemas][:self.per_turn]
        observations = self.send(sent)
        returned = [row for call, obs in zip(sent, observations) if call["name"] in (SEARCH, FILTER) for row in obs.get("results") or [] if isinstance(row, dict)]
        phrase = self.phrase_words()
        return [product_id for product_id in dict.fromkeys(str(row.get("product_id")) for row in returned if row.get("in_stock") is not False and not over(row.get("price"), self.budget())
                                                            and (not phrase or phrase <= words(row.get("title"))))
                if product_id not in self.pages]

    def look_again(self) -> None:
        done = {squash(query) for query, _ in self.search_rows}
        calls = [self.constrained(SEARCH, {"query": str(query)}) for query in self.record.get("searches") or [] if str(query).strip() and squash(query) not in done][:3]
        if self.record.get("category"):
            calls.append(self.constrained(FILTER, {"category": str(self.record["category"])}))
        if calls:
            self.send(calls)

    def open(self, ids: List[str]) -> List[str]:
        ids = [product_id for product_id in dict.fromkeys(ids) if product_id not in self.pages][:5 * min(3, self.per_turn)]
        if not ids or COMPARE not in self.schemas or self.session.turns_left() <= 3:
            return []
        self.send([{"name": COMPARE, "args": {"product_ids": ids[i:i + 5]}} for i in range(0, len(ids), 5)])
        return [product_id for product_id in ids if product_id in self.pages]

    def requirement_names(self) -> List[str]:
        items = [str(self.record.get("category") or ""), str(self.record.get("product") or "")]
        items += [str(must.get("text")) for must in self.musts()]
        items += [str(item.get("text")) for item in self.record.get("priorities") or [] if isinstance(item, dict)]
        items += [str(item.get("text")) for item in self.record.get("prefer") or [] if isinstance(item, dict)]
        return [item for index, item in enumerate(items) if item and squash(item) not in {squash(other) for other in items[:index]}]

    def extract(self, ids: List[str]) -> None:
        fresh = [product_id for product_id in ids if product_id in self.pages and product_id not in self.extracted]
        if not fresh:
            return
        self.extracted |= set(fresh)
        names = self.requirement_names()
        chunks = [fresh[i:i + EXTRACT_PAGES] for i in range(0, len(fresh), EXTRACT_PAGES)][:EXTRACT_IN_FLIGHT * 2]
        tool = json.loads(json.dumps(EXTRACT_TOOL))
        if names:
            tool["function"]["parameters"]["properties"]["rows"]["items"]["properties"]["values"]["items"]["properties"]["requirement"]["enum"] = names
        budget = self.budget()

        def job(chunk: List[str]) -> Optional[Dict[str, Any]]:
            pages = "\n\n".join(render_page(self.pages[product_id]) for product_id in chunk)
            messages = [{"role": "system", "content": EXTRACT_GUIDE},
                        {"role": "user", "content": "The shopper's money: at most %s PHP. Requirements to copy: %s\n\nThe pages:\n\n%s" % (budget, "; ".join(names), pages)}]
            return self.models.ask(EXTRACT, messages, tool, deadline=WRAP_UP_SECONDS - 25, inline=True)

        results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
        for start in range(0, len(chunks), EXTRACT_IN_FLIGHT):
            batch = list(range(start, min(start + EXTRACT_IN_FLIGHT, len(chunks))))
            threads = []
            for index in batch:
                def run(index: int = index) -> None:
                    try:
                        results[index] = job(chunks[index])
                    except Exception:
                        results[index] = None
                thread = Thread(target=run, daemon=True)
                try:
                    thread.start()
                    threads.append(thread)
                except RuntimeError:
                    run()
            for thread in threads:
                thread.join()
        for chunk, answer in zip(chunks, results):
            for row in (answer or {}).get("rows") or []:
                if not isinstance(row, dict):
                    continue
                key = (str(row.get("product_id")), str(row.get("sku")))
                if key[0] not in chunk or key not in self.live:
                    continue
                given = {squash(value.get("requirement")): value for value in row.get("values") or [] if isinstance(value, dict)}
                self.evidence[key] = {"excluded_words": str(row.get("excluded_words") or ""), "values": [
                    {"requirement": name, **{place: str((given.get(squash(name)) or {}).get(place) or "") for place in ("facts", "options", "title")}} for name in names]}

    def candidates(self) -> List[Tuple[str, str]]:
        named = self.named_keys()
        changed = self.first_add is not None and (bool(self.changed_musts) or bool(self.budget_dropped))
        order = self.opened_order[self.opened_before_add:] + self.opened_order[:self.opened_before_add] if changed else self.opened_order
        keys = []
        for product_id in order:
            own = []
            for variant in self.pages[product_id].get("variants") or []:
                if not isinstance(variant, dict):
                    continue
                key = (product_id, str(variant.get("sku")))
                if key in self.lost or not self.orderable(key) or (named and "%s::%s" % key not in named):
                    continue
                own.append(key)
            keys += own[:PAGE_ROWS]
        return keys

    def cell(self, key: Tuple[str, str], must: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return next((value for value in (self.evidence.get(key) or {}).get("values", []) if squash(value.get("requirement")) == squash(must.get("text"))), None)

    def copied(self, key: Tuple[str, str], must: Dict[str, Any]) -> Tuple[List[str], Optional[float], Optional[float]]:
        cell = self.cell(key, must)
        if cell is None:
            return [], None, None
        value, unit, relation = float(number(must.get("value")) or 0), squash(must.get("unit")), str(must.get("relation") or "exact")
        present = [place for place in ("options", "facts", "title") if cell.get(place) and states(str(cell[place]).split("=", 1)[-1], value, unit, relation)]
        clash = None if "options" in present or not cell.get("options") else another_figure(str(cell["options"]).split("=", 1)[-1], value, unit)
        titles = (str(cell.get("title") or ""), str((self.pages.get(key[0]) or {}).get("title") or ""))
        listed = any(len({found for found, written in figures(text) if written and unit_like(written, unit)}) > 1 or re.search(r"\d\s*[/&]\s*\d", text) for text in titles)
        title_clash = another_figure(titles[0], value, unit) if present and "title" not in present and titles[0] and not listed else None
        return present, clash, title_clash

    def figure_note(self, key: Tuple[str, str], must: Dict[str, Any]) -> str:
        cell = self.cell(key, must)
        if cell is None:
            return ""
        value, unit = float(number(must.get("value")) or 0), squash(must.get("unit"))
        present, clash, title_clash = self.copied(key, must)
        if present:
            return "the person's %g %s appears in: %s%s%s" % (value, must.get("unit") or "", ", ".join(present), " - but this variant's options write %g" % clash if clash is not None else "",
                                                           " - but its title writes %g" % title_clash if title_clash is not None else "")
        other = next(((place, found) for place in ("options", "facts", "title") if cell.get(place)
                      for found in [another_figure(str(cell[place]).split("=", 1)[-1], value, unit)] if found is not None), None)
        return "the copies write %g, not the person's %g %s (in %s)" % (other[1], value, must.get("unit") or "", other[0]) if other else "the person's %g %s appears in none of the copies" % (value, must.get("unit") or "")

    def table(self, keys: List[Tuple[str, str]]) -> str:
        lost_titles = {squash((self.pages.get(key[0]) or {}).get("title")) for key in self.lost}
        lost_stems = {stem(key[1]) for key in self.lost}
        lines = []
        for index, key in enumerate(keys[:TABLE_ROWS], 1):
            page = self.pages[key[0]]
            variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), {})
            facts = page.get("facts") or {}
            marks = []
            if squash(page.get("title")) in lost_titles:
                marks.append("SAME TITLE AS THE LOST LISTING")
            if stem(key[1]) in lost_stems:
                marks.append("SAME SKU FAMILY AS THE LOST VARIANT")
            lines.append("[%d] key=%s::%s | %s PHP, in stock | brand field: %s | shelf: %s | model line: %s | this variant's options: %s%s\n    title: %s\n    facts: %s" % (
                index, key[0], key[1], self.live.get(key, (None, None))[1], page.get("brand") or "(empty)", " > ".join(str(label) for label in page.get("category_path") or []),
                facts.get("model") or "(none)", variant.get("options") or "(none)", "".join(" | %s" % mark for mark in marks),
                str(page.get("title") or "")[:200], ("; ".join("%s = %s" % item for item in facts.items()) or "(none)")[:500]))
            evidence = self.evidence.get(key)
            if evidence:
                for cell in evidence["values"]:
                    places = "; ".join("%s: %s" % (place, cell[place][:100]) for place in ("options", "facts", "title") if cell.get(place)) or "nothing stated"
                    must = next((must for must in self.figure_musts() if squash(must.get("text")) == squash(cell["requirement"])), None)
                    lines.append("    %s -> %s%s" % (cell["requirement"], places, " -> " + self.figure_note(key, must) if must else ""))
                if evidence["excluded_words"]:
                    lines.append("    words of a ruled-out form on the page: %s" % evidence["excluded_words"])
        more = len(keys) - TABLE_ROWS
        return "\n".join(lines) + ("\n(%d more variants of these pages not listed)" % more if more > 0 else "")

    def situation(self) -> str:
        notes = []
        for must in self.changed_musts:
            if self.dropped_musts:
                notes.append("The person changed a requirement: it is now %s. That value is exactly what they want now: a page stating the value they dropped, or any other value, does not meet it, a title alone does not state it, and a preference counts only among variants whose options or facts state the new value." % dumps(must.get("text")))
            else:
                notes.append("The person's requirements now include %s: judge every candidate against it." % dumps(must.get("text")))
        for key in self.lost:
            in_stock, price = self.live.get(key, (None, None))
            notes.append("The variant carted before (%s::%s, %s) is %s and is not a candidate; rows marked SAME TITLE or SAME SKU FAMILY are the same product in another listing%s." % (
                key[0], key[1], str((self.pages.get(key[0]) or {}).get("title") or "")[:80], "sold out" if in_stock is False else "now %s PHP, above the budget" % price,
                " - but they carry the old value of a changed requirement unless their page states the new one" if self.changed_musts and self.dropped_musts else ""))
        if self.question_open and self.said:
            notes.append("The person asked: %s - write your one-line reply." % dumps(self.said[-1][:200]))
        if self.claims_required or self.record.get("wants_reason"):
            notes.append("A reason is required with the order: write the claims about your choice.")
        return "\n".join(notes)

    def judge(self, keys: List[Tuple[str, str]]) -> Tuple[Optional[Tuple[str, str]], Optional[Dict[str, Any]]]:
        table = self.table(keys)
        messages = [{"role": "system", "content": JUDGE_GUIDE},
                    {"role": "user", "content": "The person's request:\n%s\n\nWhat they said since:\n%s\n\nOrder conditions: %s\n\n%s\n\nCandidates:\n%s" % (
                        self.request, "\n".join("- " + text for text in self.said) or "(nothing)", dumps(self.record), self.situation(), table)}]
        answers = self.models.votes(JUDGE, messages, JUDGE_TOOL, JUDGE_VOTES, deadline=WRAP_UP_SECONDS + 10) if self.models.dollars < EPISODE_DOLLARS else []
        answer = self.majority(answers, keys) if answers else self.models.ask(JUDGE, messages, JUDGE_TOOL, deadline=WRAP_UP_SECONDS + 10)
        steps = [item for item in self.record.get("priorities") or [] if isinstance(item, dict)]
        for other in answers:
            for entry in other.get("candidates") or []:
                found = self.key_of(entry.get("key"), keys) if isinstance(entry, dict) else None
                for step in steps if found else []:
                    self.rank_value(found, entry, step)
        if answer is None:
            picked = self.models.ask(PICK, messages + [{"role": "user", "content": PICK_NOTE}], PICK_TOOL, deadline=WRAP_UP_SECONDS + 10)
            key = self.key_of((picked or {}).get("choice"), keys)
            return (key, {"choice": picked.get("choice"), "why": picked.get("why")}) if key is not None and picked else (None, None)
        key, problem = self.verify(answer, keys)
        if problem and self.clock.elapsed() < WRAP_UP_SECONDS - 20:
            messages.append({"role": "user", "content": "Your answer was: %s\n\n%s Answer again with the judge tool." % (dumps(answer)[:6000], problem)})
            seconds = self.models.votes(JUDGE, messages, JUDGE_TOOL, JUDGE_VOTES, deadline=WRAP_UP_SECONDS + 10) if self.models.dollars < EPISODE_DOLLARS else []
            second = self.majority(seconds, keys) if seconds else self.models.ask(JUDGE, messages, JUDGE_TOOL, deadline=WRAP_UP_SECONDS + 10)
            if second is not None:
                answer = second
                key, _ = self.verify(second, keys, final=True)
        return key, answer

    def majority(self, answers: List[Dict[str, Any]], keys: List[Tuple[str, str]]) -> Dict[str, Any]:
        chosen = [self.key_of(answer.get("choice"), keys) for answer in answers]
        counts = {key: chosen.count(key) for key in chosen if key is not None}
        best = max(counts.values()) if counts else 0
        return next((answer for answer, key in zip(answers, chosen) if key is not None and counts[key] == best), answers[0])

    def key_of(self, text: Any, keys: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
        text = squash(text).strip(" .")
        if "::" in text:
            product_id, _, sku = text.partition("::")
            key = (product_id.strip(), sku.strip())
            return next((candidate for candidate in keys if squash(candidate[0]) == key[0] and squash(candidate[1]) == key[1]), None)
        matches = [candidate for candidate in keys if squash(candidate[0]) == text]
        return matches[0] if len(matches) == 1 else None

    def line_for(self, entry: Dict[str, Any], must: Dict[str, Any]) -> Dict[str, Any]:
        for line in entry.get("musts") or []:
            if isinstance(line, dict) and (squash(line.get("must")) == squash(must.get("text")) or words(must.get("text")) <= words(line.get("must"))):
                return line
        return {}

    def proof(self, key: Tuple[str, str], entry: Dict[str, Any], must: Dict[str, Any]) -> str:
        line = self.line_for(entry, must)
        quote = plain(line.get("quote") or "")
        if line.get("verdict") != "met" or not quote:
            return ""
        if not states(quote.split("=", 1)[-1], float(number(must.get("value")) or 0), squash(must.get("unit")), str(must.get("relation") or "exact")):
            return ""
        page = self.pages.get(key[0]) or {}
        options = next((str(v.get("options") or "") for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), "")
        places = (("options", options), ("facts", "; ".join("%s = %s" % item for item in (page.get("facts") or {}).items())), ("title", str(page.get("title") or "")))
        place = next((name for name, text in places if quote in plain(text)), "")
        return "" if place != "options" and self.copied(key, must)[1] is not None else place

    def copies_prove(self, key: Tuple[str, str], must: Dict[str, Any], places: Tuple[str, ...] = ("options", "facts")) -> bool:
        present, clash, _ = self.copied(key, must)
        return clash is None and bool(set(places) & set(present))

    def identity(self, entry: Dict[str, Any]) -> List[str]:
        doubts = entry.get("doubts") if isinstance(entry.get("doubts"), dict) else {}
        kinds = ["another_kind", "look_alike"] + (["other_model"] if self.models_named() else []) + (["restored"] if self.claims_required or self.record.get("wants_reason") else [])
        return ["%s: %s" % (kind.replace("_", " "), doubts[kind]) for kind in kinds if squash(doubts.get(kind)) not in ("", "none", "no", "n/a", "-")]

    def sound(self, entry: Dict[str, Any]) -> bool:
        return bool(entry.get("fits")) and not self.identity(entry) and all(self.line_for(entry, must).get("verdict", "met") == "met" for must in self.musts())

    def verify(self, answer: Dict[str, Any], keys: List[Tuple[str, str]], final: bool = False) -> Tuple[Optional[Tuple[str, str]], str]:
        entries = {}
        for entry in answer.get("candidates") or []:
            if isinstance(entry, dict):
                found = self.key_of(entry.get("key"), keys)
                if found:
                    entries[found] = entry
        order = [self.key_of(item, keys) for item in [answer.get("choice")] + list(answer.get("ranking") or [])]
        order = [key for key in dict.fromkeys(order) if key is not None]
        key = order[0] if order else None
        if key is None:
            fitting = [found for found, entry in entries.items() if self.sound(entry)]
            key = fitting[0] if fitting else None
        self.backups = [other for other in [other for other in order[1:] if other not in entries or self.sound(entries[other])]
                        + [found for found, entry in entries.items() if self.sound(entry)] if other != key]
        if key is None or final:
            return key, ""
        entry = entries.get(key) or {}
        good = ("facts",) if self.claims_required or self.record.get("wants_reason") else ("options", "facts")
        changed = {squash(must.get("text")) for must in self.changed_musts} if self.dropped_musts else set()
        mine = len(entry.get("preferences_met") or [])
        for must in self.figure_musts():
            place = self.proof(key, entry, must)
            omitted = place not in good and not plain(self.line_for(entry, must).get("quote") or "") and self.copies_prove(key, must, good)
            if place in good or omitted:
                place = place if place in good else " or ".join(good)
                title_clash = self.copied(key, must)[2]
                steady = [other for other, their in entries.items() if other != key and self.sound(their) and self.proof(other, their, must) in good
                          and self.copied(other, must)[2] is None and len(their.get("preferences_met") or []) >= mine] if title_clash is not None else []
                if title_clash is not None and not steady:
                    brand = squash((self.pages.get(key[0]) or {}).get("brand"))
                    steady = [other for other in keys if other != key and other not in entries and self.copies_prove(other, must, good) and self.copied(other, must)[2] is None
                              and (not self.record.get("prefer") or squash((self.pages.get(other[0]) or {}).get("brand")) == brand)]
                if steady:
                    return key, ("Your choice %s::%s writes %g in its title but %g %s in its %s: the page contradicts itself. %s::%s states %g %s without a contradiction. Choose it, or say in why your page is right." % (
                        key[0], key[1], title_clash, number(must.get("value")), must.get("unit") or "", place, steady[0][0], steady[0][1], number(must.get("value")), must.get("unit") or ""))
                continue
            strict = good == ("facts",) or squash(must.get("text")) in changed or not self.record.get("prefer")
            rivals = [other for other, their in entries.items() if other != key and self.sound(their) and self.proof(other, their, must) in good
                      and (strict or len(their.get("preferences_met") or []) >= mine)]
            if strict:
                rivals += [other for other in keys if other != key and other not in rivals and self.copies_prove(other, must, good)]
            if rivals:
                quote = self.line_for(entry, must).get("quote")
                return key, ("Your choice %s::%s is not shown to meet %s in its own options or facts: %s. %s::%s states %g %s there (%s). Choose that one, or quote the words of your choice's options or facts that state that very value." % (
                    key[0], key[1], dumps(must.get("text")), "your quote %s stands only in its title" % dumps(quote) if place == "title" else "your quote %s does not write that value on its page" % dumps(quote) if quote else "you quoted nothing for it",
                    rivals[0][0], rivals[0][1], number(must.get("value")), must.get("unit") or "", self.figure_note(rivals[0], must) or "its quote"))
        if self.lost and not self.named_keys() and not self.named_names() and not (self.changed_musts and self.dropped_musts):
            lost_stems = {stem(lost[1]) for lost in self.lost}
            lost_titles = {squash((self.pages.get(lost[0]) or {}).get("title")) for lost in self.lost}
            lost_shelves = {tuple((self.pages.get(lost[0]) or {}).get("category_path") or []) for lost in self.lost}
            same = lambda other: stem(other[1]) in lost_stems or (squash((self.pages.get(other[0]) or {}).get("title")) in lost_titles
                                                                   and tuple((self.pages.get(other[0]) or {}).get("category_path") or []) in lost_shelves)
            sisters = [other for other, their in entries.items() if other != key and same(other) and self.sound(their)]
            sisters += [other for other in keys if other != key and other not in entries and same(other) and all(self.copies_prove(other, must) for must in self.figure_musts())]
            if sisters and not same(key):
                return key, ("%s::%s is the same product as the variant carted before - the same sku family or title - and fits. After a loss the same product in another listing comes before another product. Choose it, or keep yours and say in why yours is the same product." % sisters[0])
        maker_brands = {squash(self.rows[other].get("brand")) for other in self.maker_rows if other in self.rows and self.rows[other].get("brand")}
        makers_own = lambda other: other[0] in self.maker_rows or squash((self.pages.get(other[0]) or {}).get("brand")) in maker_brands
        maker_keys = [other for other in keys if other != key and makers_own(other)] if self.makers() and self.maker_rows else []
        if maker_keys and not makers_own(key) and not entry.get("preferences_met") and not self.named_keys() and not self.named_names():
            return key, ("The person prefers %s, and firm requirements come first: the store's %s listings on the table are %s. If one of them meets every must on its own page (a value written in another unit of the same amount counts), it comes before any other maker's listing - choose it; if none does, keep yours and say in why which must each of them fails." % (
                " / ".join(self.makers()[:2]), " / ".join(self.makers()[:2]), ", ".join("%s::%s" % other for other in maker_keys[:4])))
        if self.makers() and self.maker_rows and not makers_own(key) and entry.get("preferences_met") and not self.named_keys() and not self.named_names():
            theirs = [other for other, their in entries.items() if other != key and makers_own(other) and self.sound(their)]
            if theirs:
                return key, ("The store's filter for %s did not return your choice %s::%s, whose brand field is %s, while %s::%s, which that filter returned, fits. The brand field says whose listing it is: choose it, or keep yours and say in why yours is that maker's." % (
                    " / ".join(self.makers()[:2]), key[0], key[1], (self.pages.get(key[0]) or {}).get("brand") or "(empty)", theirs[0][0], theirs[0][1]))
        doubts = self.identity(entry)
        if doubts and not self.named_keys():
            clean = [other for other, their in entries.items() if other != key and self.sound(their) and len(their.get("preferences_met") or []) >= mine]
            if clean:
                return key, ("You wrote an identity doubt about your choice (%s) while %s::%s fits with none and meets as many preferences. Identity doubts decide between candidates: choose that one, or keep yours and say in why the doubt does not hold." % (
                    "; ".join(doubts)[:200], clean[0][0], clean[0][1]))
        steps = [item for item in self.record.get("priorities") or [] if isinstance(item, dict)]
        if steps and str(steps[0].get("direction")) in ("higher", "lower", "cheaper"):
            ranked = self.rank(entries, steps, everyone=bool(self.named_keys()))
            if ranked and ranked[0] != key and key in entries:
                shown = lambda other: ", ".join("%s %s" % (str(step.get("text"))[:30], "not stated" if found is None else "%g" % found)
                                                 for step in steps if str(step.get("direction")) in ("higher", "lower", "cheaper") for found in [self.rank_value(other, entries[other], step)])
                return key, ("By the person's ranking and the values you wrote from the pages, %s::%s comes before your choice - it has %s; yours has %s. The first step where they differ decides, and a later step only breaks a tie. Choose it, or correct the value you wrote." % (
                    ranked[0][0], ranked[0][1], shown(ranked[0]), shown(key)))
        return key, ""

    def rank_value(self, key: Tuple[str, str], entry: Dict[str, Any], step: Dict[str, Any]) -> Optional[float]:
        if step.get("direction") == "cheaper":
            return number(self.live.get(key, (None, None))[1])
        page = self.pages.get(key[0]) or {}
        page_text = plain(render_page(page))
        line = next((line for line in entry.get("ranks") or [] if isinstance(line, dict) and (squash(line.get("step")) == squash(step.get("text")) or words(line.get("step")) & words(step.get("text")))), None)
        memo = (key, squash(step.get("text")))
        quote = plain((line or {}).get("quote") or "")
        if line and quote and quote in page_text and number(line.get("number")) is not None:
            found = float(number(line.get("number")))
            written = figures(quote.split("=", 1)[-1])
            if written and found <= 0 < written[0][0]:
                found = written[0][0]
            options = next((str(v.get("options") or "") for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), "")
            body = plain(options + " " + "; ".join("%s = %s" % item for item in (page.get("facts") or {}).items()))
            cell = self.cell(key, {"text": step.get("text")}) or {}
            said, stated = figures(quote), figures(str(cell.get("options") or cell.get("facts") or "").split("=", 1)[-1])
            if quote in plain(page.get("title")) and quote not in body and said and stated and said[0][0] != stated[0][0] and unit_like(said[0][1], stated[0][1]):
                found = stated[0][0]
            self.rank_memory[memo] = found
        return self.rank_memory.get(memo)

    def rank(self, entries: Dict[Tuple[str, str], Dict[str, Any]], steps: List[Dict[str, Any]], everyone: bool = False) -> List[Tuple[str, str]]:
        def value(key: Tuple[str, str], entry: Dict[str, Any], index: int) -> Optional[float]:
            return self.rank_value(key, entry, steps[index])

        fitting = [(key, entry) for key, entry in entries.items() if everyone or self.sound(entry)]
        def sort_key(item: Tuple[Tuple[str, str], Dict[str, Any]]) -> Tuple:
            parts = []
            for index, step in enumerate(steps):
                direction = str(step.get("direction"))
                if direction not in ("higher", "lower", "cheaper"):
                    break
                found = value(item[0], item[1], index)
                parts.append((found is None, -(found or 0) if direction == "higher" else (found or 0)))
            return tuple(parts)
        return [key for key, _ in sorted(fitting, key=sort_key)]

    def fallback_key(self, keys: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
        for key in self.backups:
            if key in keys:
                return key
        if not keys:
            return None
        recent = [key for key in keys if key[0] in self.opened_order[self.opened_before_add:]] if self.first_add else []
        order = recent + [key for key in keys if key not in recent]
        return max(order, key=lambda key: (sum(self.copies_prove(key, must) for must in self.figure_musts()), -order.index(key)))

    def resolve_names(self) -> None:
        keys = self.candidates()
        if not keys:
            return
        lines = []
        for key in keys[:TABLE_ROWS * 2]:
            page = self.pages[key[0]]
            variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), {})
            lines.append("key=%s::%s | brand field: %s | model line: %s | options: %s | title: %s" % (key[0], key[1], page.get("brand") or "(empty)", (page.get("facts") or {}).get("model") or "(none)",
                                                                                                    variant.get("options") or "(none)", str(page.get("title") or "")[:160]))
        messages = [{"role": "system", "content": RESOLVE_GUIDE},
                    {"role": "user", "content": "The names: %s\n\nThe listings:\n%s" % ("; ".join(self.named_names()), "\n".join(lines))}]
        answer = self.models.ask(RESOLVE, messages, RESOLVE_TOOL, deadline=HURRY_SECONDS)
        found = [self.key_of(item.get("key"), keys) for item in (answer or {}).get("candidates") or [] if isinstance(item, dict) and item.get("key")]
        found = list(dict.fromkeys(key for key in found if key is not None))
        if len(found) >= 2:
            self.request_keys |= {"%s::%s" % key for key in found} | {squash("%s::%s" % key) for key in found}

    def choose(self) -> Optional[Tuple[str, str]]:
        if self.clock.elapsed() < HURRY_SECONDS:
            self.extract([product_id for product_id in self.opened_order if any(key[0] == product_id for key in self.candidates())])
        if self.named_names() and not self.named_keys():
            self.resolve_names()
        keys = self.candidates()
        key, answer = self.judge(keys) if keys else (None, None)
        self.last_judge = answer or self.last_judge
        if answer and answer.get("reply"):
            self.reply = str(answer["reply"])[:300]
        if key is not None and answer is not None:
            self.claims[key] = list(answer.get("claims") or [])
        second_round = (keys == [] or (answer is not None and answer.get("none_fit"))) and not self.phrase_words() and not self.named_keys()
        if second_round and self.clock.elapsed() < HURRY_SECONDS and self.session.turns_left() > 8 and self.models.dollars < EPISODE_DOLLARS:
            done = {squash(query) for query, _ in self.search_rows}
            offered = [str(query) for query in list((answer or {}).get("searches") or []) + self.instead if str(query).strip() and squash(query) not in done]
            queries = list(dict.fromkeys(offered))[:3] or [str(self.record.get("product") or "")]
            fresh = self.gather([self.constrained(SEARCH, {"query": query}) for query in queries])
            opened = self.open(self.shortlist(only=fresh, limit=REPLACEMENT_OPENS))
            self.extract(opened)
            again = self.candidates()
            if again:
                key2, answer2 = self.judge(again)
                if key2 is not None:
                    key, answer = key2, answer2
                    self.last_judge = answer2
                    self.claims[key] = list(answer2.get("claims") or []) if answer2 else []
                    if answer2 and answer2.get("reply"):
                        self.reply = str(answer2["reply"])[:300]
            keys = again or keys
        if key is None:
            key = self.fallback_key(self.candidates())
        return key

    def add(self, key: Tuple[str, str]) -> bool:
        if key in self.cart:
            return self.orderable(key)
        before = number(self.live.get(key, (None, None))[1])
        calls = [{"name": ADD, "args": {"product_id": key[0], "sku": key[1]}}]
        if INSPECT_STOCK in self.schemas:
            calls.append({"name": INSPECT_STOCK, "args": {"product_id": key[0], "sku": key[1]}})
        self.send(calls)
        self.old_price = before
        if not self.orderable(key):
            self.lost.add(key)
            return False
        return True

    def reads_note(self) -> str:
        if self.changed_musts and self.dropped_musts:
            return "The person changed a requirement: it is now %s. Open rows that could state the new value." % "; ".join(dumps(must.get("text")) for must in self.changed_musts)
        titles = [str((self.pages.get(key[0]) or {}).get("title") or "")[:100] for key in self.lost]
        return "The listing carted before is gone: %s. Rows carrying its title are the same product in another listing." % "; ".join(dumps(title) for title in titles) if titles else ""

    def replacement_reads(self) -> None:
        if self.named_keys() or self.named_names() or self.clock.elapsed() > HURRY_SECONDS + 15 or self.session.turns_left() <= 5:
            return
        calls: List[Dict[str, Any]] = []
        product = " ".join(squash(self.record.get("product") or "").split()[:4])
        shelf = str(self.record.get("category") or "")
        phrase = bool(self.phrase_words())
        if self.changed_musts and not phrase:
            for must in self.changed_musts[:2]:
                plain_figure = "%g %s" % (number(must.get("value")), str(must.get("unit") or "").strip("- ")) if number(must.get("value")) is not None and must.get("unit") else ""
                for written in [text for text in (plain_figure, squash(must.get("text"))) if text]:
                    calls.append(self.constrained(SEARCH, {"query": " ".join(dict.fromkeys(("%s %s" % (product, written)).split()[:10]))}))
                for maker in self.makers()[:1]:
                    calls.append(self.constrained(SEARCH, {"query": " ".join(dict.fromkeys(("%s %s %s" % (squash(maker), product, plain_figure or squash(must.get("text")))).split()[:10]))}))
            if shelf or product:
                calls.append(self.constrained(FILTER, {"category": shelf or product}))
        lost = [key for key in self.lost if key in self.cart]
        if lost and not self.changed_musts:
            key = lost[-1]
            page = self.pages.get(key[0]) or {}
            path = [str(label) for label in page.get("category_path") or [] if label]
            if path:
                args: Dict[str, Any] = {"category": path[-1]}
                if page.get("brand"):
                    args["brand"] = str(page["brand"])
                cap = round(self.old_price * SISTER_PRICE_ROOM, 2) if self.old_price else None
                calls.append(self.constrained(FILTER, args, cap))
            if not phrase and page.get("title"):
                title = [word for word in dict.fromkeys(piece.strip(".,;:!?()[]\"'") for piece in squash(page.get("title")).split()) if word][:8]
                variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), {})
                for must in self.figure_musts()[:2]:
                    written = next((piece for piece in ("%s %s" % (variant.get("options") or "", page.get("title") or "")).replace("/", " ").split()
                                    if re.search(r"\d", piece) and states(piece, float(number(must.get("value")) or 0), squash(must.get("unit")), "exact") and re.search(r"[a-zA-Z]", piece)), None)
                    if written and squash(written) not in title:
                        title.append(squash(written).strip(".,;:()[]"))
                    elif not written and number(must.get("value")) is not None and must.get("unit"):
                        title += [word for word in ("%g %s" % (number(must.get("value")), str(must.get("unit")).strip("- "))).split() if word not in title]
                calls.append(self.constrained(SEARCH, {"query": " ".join(title)}))
        if self.budget_dropped and (shelf or product) and not any(call["name"] == FILTER and call["args"].get("category") == (shelf or product) for call in calls):
            calls.append(self.constrained(FILTER, {"category": shelf or product}))
            for maker in self.makers()[:MAKER_FILTERS]:
                calls.append(self.constrained(FILTER, {"category": shelf or product, "brand": maker}))
        backups = [key for key in self.backups if key not in self.cart][:3]
        if INSPECT_STOCK in self.schemas:
            calls += [{"name": INSPECT_STOCK, "args": {"product_id": key[0], "sku": key[1]}} for key in backups]
        if not calls:
            return
        fresh = self.gather(calls)
        lost_titles = {squash((self.pages.get(key[0]) or {}).get("title")) for key in self.lost}
        lead = [product_id for product_id in fresh if squash(self.rows[product_id].get("title")) in lost_titles]
        opened = self.open(self.shortlist(only=fresh, lead=lead, limit=REPLACEMENT_OPENS, note=self.reads_note()))
        self.extract(opened)

    def claims_for(self, key: Tuple[str, str]) -> List[Dict[str, Any]]:
        kept = self.shown_claims(key, self.claims.get(key) or [])
        page = self.pages.get(key[0]) or {}
        labels = {squash(label) for label in (page.get("facts") or {})}
        if any(squash(claim["field"]) in labels for claim in kept) or self.clock.elapsed() > TIME_LIMIT_SECONDS - 25:
            return kept
        in_stock, price = self.live.get(key, (None, None))
        messages = [{"role": "system", "content": CLAIMS_GUIDE},
                    {"role": "user", "content": "The request: %s\n\nThe ordered variant: sku=%s, latest price %s, in_stock %s\n%s" % (self.request, key[1], price, dumps(in_stock), render_page(page))}]
        answer = self.models.ask(CLAIMS, messages, CLAIMS_TOOL, deadline=TIME_LIMIT_SECONDS - 8)
        written = self.shown_claims(key, list((answer or {}).get("claims") or []))
        fields = {squash(claim["field"]) for claim in written}
        return written + [claim for claim in kept if squash(claim["field"]) not in fields]

    def shown_claims(self, key: Tuple[str, str], claims: List[Any]) -> List[Dict[str, Any]]:
        page = self.pages.get(key[0]) or {}
        facts = page.get("facts") or {}
        in_stock, price = self.live.get(key, (None, None))
        kept = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            field, value = squash(claim.get("field") or ""), squash(claim.get("value") or "")
            if field == "price":
                amount = number("".join(ch for ch in value if ch.isdigit() or ch == "."))
                if amount is not None and amount == number(price):
                    kept.append({"field": "price", "value": str(price)})
            elif field == "in_stock":
                if value == squash(in_stock) and in_stock is True:
                    kept.append({"field": "in_stock", "value": "true"})
            elif field == "brand":
                if value and plain(value) == plain(page.get("brand") or ""):
                    kept.append({"field": "brand", "value": str(page.get("brand"))})
            else:
                fact = next((str(fact) for label, fact in facts.items() if squash(label) == field and value and plain(value) in plain(fact)), None)
                if fact is not None:
                    kept.append({"field": str(next(label for label in facts if squash(label) == field)), "value": fact})
        return kept

    def order(self, key: Tuple[str, str]) -> None:
        args: Dict[str, Any] = {"product_id": key[0], "sku": key[1]}
        if self.claims_required or self.record.get("wants_reason"):
            claims = self.claims_for(key)
            if claims or self.claims_required:
                args["justification"] = {"claims": claims}
        calls = []
        if self.question_open and MESSAGE in self.schemas:
            calls.append({"name": MESSAGE, "args": {"content": self.reply or "Yes, I re-checked the latest price and stock and chose the variant that fits your requirements now; ordering it."}})
        calls.append({"name": ORDER, "args": args})
        self.send(calls)

    def run(self) -> None:
        self.write_conditions()
        self.first_look()
        if self.record.get("phrase") and not self.phrase_holds(self.record):
            self.record.pop("phrase")
        before = (dumps(self.musts()), self.record.get("category"))
        self.track_budget(self.hear)
        if (dumps(self.musts()), self.record.get("category")) != before and not self.cart:
            self.look_again()
        self.open(self.shortlist())
        self.track_budget(self.hear)
        self.extract(self.opened_order)
        if self.budget_dropped and not self.candidates():
            self.replacement_reads()
        choice = self.choose()
        self.changed_musts, self.dropped_musts = [], []
        rounds = 0
        while choice is not None and not self.session.done and not self.late():
            heard = len(self.said)
            self.add(choice)
            self.track_budget(self.hear)
            if self.record.get("update_promised") and rounds == 0 and len(self.said) == heard and INSPECT_CART in self.schemas and self.session.turns_left() > 5:
                self.send([{"name": INSPECT_CART, "args": {}}])
                self.track_budget(self.hear)
            if (self.orderable(choice) and choice in self.cart and not self.changed_musts) or rounds >= 2:
                break
            if not self.orderable(choice):
                self.lost.add(choice)
            rounds += 1
            self.replacement_reads()
            self.track_budget(self.hear)
            nxt = self.choose()
            self.changed_musts, self.dropped_musts = [], []
            if nxt is None or nxt == choice:
                break
            choice = nxt
        self.close()

    def track_budget(self, step: Any) -> bool:
        before = self.budget()
        changed = step()
        after = self.budget()
        if before is not None and after is not None and after < before:
            self.budget_dropped = True
        return bool(changed)

    def catch_up(self) -> None:
        if self.heard >= len(self.said):
            return
        if self.clock.elapsed() < WRAP_UP_SECONDS:
            self.track_budget(self.hear)
            return
        cut = self.budget_cut("\n".join(self.said[self.heard:]), self.budget())
        if cut is not None:
            self.record["max_price"] = cut
            self.budget_dropped = True
        self.heard = len(self.said)

    def close(self) -> None:
        if self.session.done or self.session.dead:
            return
        if not self.rows and not self.pages and SEARCH in self.schemas and self.session.turns_left() >= 3:
            self.send([self.constrained(SEARCH, {"query": str(self.record.get("product") or self.request)[:120]})])
        self.catch_up()
        target = next((key for key in reversed(self.cart) if self.orderable(key)), None)
        tried: set = set()
        while target is None and not self.session.done and self.session.turns_left() >= 2:
            keys = [key for key in self.candidates() if key not in tried and key not in self.cart]
            if not keys:
                keys = [key for key in self.live if key not in tried and key not in self.cart and self.live[key][0] is True and not over(self.live[key][1], self.budget())
                        and (not self.named_keys() or "%s::%s" % key in self.named_keys())]
            pick = next((key for key in self.backups if key in keys), None) or (self.fallback_key(keys) if keys else None)
            if pick is None:
                break
            tried.add(pick)
            view = [{"name": VIEW, "args": {"product_id": pick[0]}}] if pick[0] not in self.pages and VIEW in self.schemas else []
            self.send(view + [{"name": ADD, "args": {"product_id": pick[0], "sku": pick[1]}}] + ([{"name": INSPECT_STOCK, "args": {"product_id": pick[0], "sku": pick[1]}}] if INSPECT_STOCK in self.schemas else []))
            self.catch_up()
            if pick in self.cart and self.orderable(pick):
                target = pick
        if target is None and self.cart:
            target = self.cart[-1]
        if target is not None and not self.session.done and self.session.turns_left() > 0:
            self.order(target)


def agent_main(problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        episode.run()
    except Exception as exc:
        episode.session.dialogue.append({"role": "assistant", "content": "stopped: %s" % type(exc).__name__})
        try:
            if not episode.session.done:
                episode.close()
        except Exception:
            pass
    return episode.session.dialogue
