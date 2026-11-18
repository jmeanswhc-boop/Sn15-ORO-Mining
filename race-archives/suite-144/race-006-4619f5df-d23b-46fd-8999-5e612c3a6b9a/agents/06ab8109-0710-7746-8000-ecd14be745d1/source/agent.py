from __future__ import annotations

from os import getenv
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agent.proxy_client import ProxyClient

SEARCH, FILTER, VIEW, COMPARE = "search", "filter", "view", "compare"
INSPECT_STOCK, INSPECT_CART = "inspect_stock", "inspect_cart"
ADD, REMOVE, MESSAGE, ORDER = "add_to_cart", "remove_from_cart", "message", "place_test_order"
BRIEF, SHORTLIST, EXTRACT, JUDGE, PICK, CLAIMS, RESOLVE, SPLIT = "brief", "shortlist", "extract", "judge", "pick", "claims", "resolve", "split"

GLM = ("z-ai/glm-5.2", "zai-org/GLM-5.2-TEE")
KIMI = ("moonshotai/kimi-k2.6", "moonshotai/Kimi-K2.6-TEE")
MODELS = (GLM, GLM, KIMI)
SEATS = {seat: MODELS for seat in (BRIEF, SHORTLIST, EXTRACT, JUDGE, PICK, CLAIMS, RESOLVE, SPLIT)}
OUTPUT_TOKENS = {BRIEF: 1536, SHORTLIST: 768, EXTRACT: 3584, JUDGE: 2560, PICK: 384, CLAIMS: 640, RESOLVE: 640, SPLIT: 256}
WAIT_SECONDS = {BRIEF: 35, SHORTLIST: 45, EXTRACT: 50, JUDGE: 48, PICK: 20, CLAIMS: 20, RESOLVE: 25, SPLIT: 15}
TEMPERATURES = {BRIEF: 0.0, SHORTLIST: 0.1, EXTRACT: 0.0, JUDGE: 0.1, PICK: 0.1, CLAIMS: 0.0, RESOLVE: 0.1, SPLIT: 0.0}
HEDGE_AT = 0.75
IN_FLIGHT = 3
CLIENT_SECONDS = 60

TIME_LIMIT_SECONDS = 270
WRAP_UP_SECONDS = 235
HURRY_SECONDS = 170
EPISODE_DOLLARS_LAST = 0.12

SHORTLIST_ROWS = 10
MAKER_FILTERS = 2
MAKER_ROWS = 3
MODEL_ROWS = 3
FIGURE_ROWS = 2
EXTRACT_PAGES = 6
EXTRACT_IN_FLIGHT = 3
TABLE_ROWS = 24
PAGE_ROWS = 6
REPLACEMENT_OPENS = 8
SISTER_PRICE_ROOM = 1.1
RATE_LIMIT_RETRIES = 3

_proxy = ProxyClient(timeout=90, max_retries=1)

ROLE = "- You are a careful personal shopper buying one item for a person in an online store, with the store's own tools. They may send more messages while you shop - a lower budget, a changed requirement, a question - and you follow their newest word and answer any question before you order."

BRIEF_GUIDE = "# Personal shopper - this step: write down the order conditions\n" + ROLE + """
- Here you write their order conditions as one record, from their own words: the request and everything they said after it. The rest of the shopping reads this record alone, so every clause they wrote lands in its field and nothing they did not say appears.
- Every field is required: an empty string, an empty list, false or 0 where they said nothing for it.

# The fields
- product: the kind of thing, in plain words. For a request for any listing that matches a phrase, the phrase's words with the category as they wrote it - never a narrower kind guessed from the category's name.
- category: the store category, only when they named one ('from the <name> category'), spelled as they spelled it. A brand is never a category.
- musts: each firm requirement of the item itself, one per entry, in their words; a requirement they state again is the entry it already has, never a second one.
  - value: its figure as a bare number; null when it has none (a named standard such as a resolution name). unit: as written next to the figure.
  - relation: exact, unless their own words make a range (at least, at most, up to, or more, minimum, maximum) - then copy those words with the figure into range_words. A size, capacity, warranty or rating stated plainly is exact.
  - Values and units exactly as written - never converted, rounded or completed. The money, being in stock and buying one item are not musts.
- max_price: the most they will pay now, as a bare number - the newest amount when they changed it; null only when they never named one.
- exclude: what they ruled out.
- prefer: what they would like but can do without, one property per entry with its kind: maker (a brand or company they would like it to come from, even when the word is also an ordinary noun), model (a model name or code, letter for letter, even when it reads like a quantity), colour, or other. A preference is never a must or a priority. The thing they name with a colour and would take in another colour of the same model is that model - a model preference, never a must, however much it reads like an amount.
- fall_back: the order in which they would give preferences up, in their words; empty when they gave none.
- priorities: only when they gave a rule for ranking the listings against one another: its steps in their order, each with the direction that wins - higher or lower for a figure, cheaper when the step is the price itself, none when it ranks no figure (such as "never one another beats on all of them"). A wish for a brand, or a reminder that the firm requirements matter most, is not a ranking rule.
- candidates: the listings they named to choose among, exactly as written - a product_id::sku key, or a name - only when they limit the choice to those listings; a model they prefer is a preference. A name written two ways with a bar or a slash between ('Maker | AB-12') is one candidate.
- phrase: only when they ask for listings that match a phrase ('matching the phrase <words>'): those words, letter for letter, odd spellings included. Words describing the item are never a phrase.
- unstated: a requirement they say they have but have not stated (they tell you to ask for it); empty once they have stated it.
- survey_count: how many options they say to look over, 0 when they gave no number - a way of shopping, not a hidden requirement.
- wants_reason: true only when they ask why, or for a reason or justification they can trust.
- update_promised: true only when they say they will change a requirement later, once an item is in the cart.
- searches: three store searches of three or four words. The store ignores word order, letter case and plural endings, so each search adds words the others lack.
  - For a requirement with a value: the item with the value as pages write it, figure and unit joined (8GB, 500ml); the item with them apart (8 GB); and the preferred maker with the item and the value, or else the item and the value in other words.
  - For a named model: its code with the colour, the code with the item, and the code alone - a page often writes the code only on its model line.
  - For a phrase: the phrase itself; the phrase with the category's name when that name adds a word (never the word 'category'); and the phrase with another word sellers write beside it in the product's own name.
- changes: after a new message, each field that message changed, with the old value, the new one and their words that changed it.

# Rewriting after a new message
- The newest message wins: a new amount or value replaces the old one from then on, and everything they did not change stays exactly as it was, preferences included.
- An answer to your question adds what it states: the category they name is their category, and a requirement they state is a must.
- A message that only asks about availability or a price, repeats the conditions or reacts to the store changes nothing: write the record as it was, with no changes."""

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
        "unstated": {"type": "string"},
        "survey_count": {"type": "integer", "description": "how many options they say to look over; 0 when none"},
        "wants_reason": {"type": "boolean"},
        "update_promised": {"type": "boolean"},
        "searches": {"type": "array", "items": {"type": "string"}},
        "changes": {"type": "array", "items": {"type": "object", "properties": {
            "field": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}, "said": {"type": "string"}},
            "required": ["field", "new"]}}},
        "required": ["product", "category", "musts", "max_price", "exclude", "prefer", "fall_back", "priorities", "candidates", "phrase",
                     "unstated", "survey_count", "wants_reason", "update_promised", "searches"]}}}

SHORTLIST_GUIDE = "# Personal shopper - this step: pick the rows to open\n" + ROLE + """
- Here you pick which search rows to open. A row shows a listing's title, its brand field, one variant's price and stock, and the searches or filters that returned it - nothing more is proven yet.
- The rows are the options you look over; open the few worth reading in full, up to ten, likeliest first.

# Which rows
- First the rows carrying the person's own words (the model, the maker, the phrase, the requirement's figure with its unit), then rows whose title names the kind of thing asked for.
- A maker they prefer is only a preference: open other makers' rows whose titles write the required figure too, since one of them serves when none of that maker's listings meets the requirement.
- The store's search reads whole pages: a row that a search for their model or value returned may state it on its page though its title does not, so open it when it could be the kind of thing asked for - and when they named a model, every row of that kind the model's searches returned.
- Never accessories, parts, cases, refills or other kinds of things than the one asked for, nor a title spelling a brand or model with look-alike characters (a zero for an O).
- When they named candidates, every row that could be one of them. When they asked for any listing that matches a phrase and named no kind of thing, a part, a set or a pack filed on the named department is a listing they asked for.
- When they want a reason they can trust, a title offering a second-hand or reconditioned unit last.
- When no row could be the item, say what to search for instead, three or four words each; for a model code no row carries, the code's words apart with the item (SUNRISEKETTLE12: sunrise kettle 12) and the code with a space before its digits, since pages spell a code their own way."""

SHORTLIST_TOOL = {"type": "function", "function": {
    "name": SHORTLIST,
    "description": "The rows to open, likeliest first, with three or four words each on why; or what to search for instead.",
    "parameters": {"type": "object", "properties": {
        "open": {"type": "array", "maxItems": SHORTLIST_ROWS, "items": {"type": "object", "properties": {
            "product_id": {"type": "string"}, "why": {"type": "string"}}, "required": ["product_id", "why"]}},
        "search_instead": {"type": "array", "items": {"type": "string"}}},
        "required": ["open", "search_instead"]}}}

EXTRACT_GUIDE = "# Personal shopper - this step: copy what the pages state\n" + ROLE + """
- Here you copy what the opened pages state, for the colleague who decides. For every variant that is in stock within the person's money, write one row: the listing, the variant, any words of a form they ruled out (only when there are some), and for each requirement what the page states for it in three places, copied exactly as written with its unit - the facts (as 'label = value', only under a label that names that property: a figure under another property's label is not this requirement's), this variant's own options, and the title.

# How to copy
- Copy, never judge: an odd value is still what the page states, in the page's own notation.
- A fact listing several values the seller sells ('volume = 180ml, 360ml') states none of them for one variant: leave the facts place empty - this variant's options say which one it is. A value written in two parts ('1366x912', '10GB+256GB') is one value.
- From the options copy only the part that states the requirement ('13.6 inch' out of 'model c / 13.6 inch / grey'), and from the title only the words that state it.
- Fill every place of every requirement, an empty string where it states nothing - and read the facts before leaving them empty: labels such as size, display, ram, capacity, wattage or warranty_duration are the facts speaking.

# Forms pages use (examples)
- A screen's size: a fact 'size = 6.45 inches', an option '6.45 inch', a title's '6.45" HD+'; inside 'display = 1366 x 912; 6.45 inch lcd' it stands among other values.
- A warranty: a fact 'warranty_duration = 21 months', an option '21 months warranty'; a warranty's kind alone ('warranty_type = shop warranty') states no length.
- A power, a capacity, a memory: 'wattage = 1175 w' or an option '1175W'; 'capacity = 485 ml' or '485ml'; 'ram = 10gb' or an option '10GB+256GB'.
- A resolution or a refresh rate: mostly inside the display fact ('display = 1366 x 912', 'display = 100hz, lcd') or an option such as '100Hz'.

# Answer
- With the extract tool alone."""

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

JUDGE_GUIDE = "# Personal shopper - this step: decide which variant to buy\n" + ROLE + """
- Here you decide which variant to buy. Below are the person's words, their order conditions and a table of candidate variants the store showed, each in stock within their money: the brand field, shelf, model line, facts, this variant's own options and title, and for each requirement what the page states, copied as written, with a note on where the person's figure appears in those copies.
- The person is served only when the variant's own page states every firm requirement: treat each candidate as wrong until its page states each must with their value.

# Reading a page
- A requirement is stated in three places. This variant's options say which variant arrives, so they decide it over the facts and the title. A fact under that property's own label states it for the product. The title is the seller's advertising - a value only the title states still meets the requirement, but weakly.
- Plainly stated: a fact under the label of that very property with its unit, or an option in the person's own unit. Weakly stated: the title alone, a figure inside a broader label among other values ('display = 1366 x 912; 6.45 inch lcd'), or a clipped or misspelt unit.
- Nothing stated: a figure inside a model name or under another property's label; a bare number without its unit, unless its label names the unit; a fact, option or title listing several values the seller sells, unless this variant's options pin one; a warranty's kind without its length.
- The value they named is exactly the value they want: a nearer, larger, longer, higher or 'better' value is another value unless their own words make a range (a 6.9-inch screen is not 6.5 inches, 30 months is not 18, 1450 W is not 1250 W). The same amount written in the page's own notation or another unit is the same value: ask whether the page states the very amount they named, not whether it is enough. What no place states is unknown, never met.
- When they want a reason, the specification must stand in the facts: the order tool takes specification claims from the product's raw facts, and a figure on the model line names a model, not a property.

# Who made it and what it is
- The department they named is proven by a shelf label that is that department's name; a neighbouring shelf sharing a word is another department. A listing on the named department is that kind of thing whatever its title says.
- The brand is the brand field; the title counts only when the field is empty. A title advertising a maker over a brand field naming another maker is that other maker's listing, and when they prefer a maker, a listing titled with that maker's model under another company's brand field is a look-alike listing, not their maker's. A generic seller word in the brand field (OEM, no brand, imported, a shop's own name) is no doubt, and when they named no maker an empty brand field is no doubt either.
- A model they named is the page whose model line spells the code as they wrote it (spaces and letter case aside), whatever else its title says; a code in a title or an option alone is advertising, and a model line spelling another code is another model (a kettle titled Aqua is not the aqua K200 when its model line says k100). A colour they named is met by an option spelled as they spelled it.
- A look-alike spelling of a brand or model (a zero for an O, a changed letter; a change of case alone is no look-alike) is another product and the gravest doubt.
- A candidate they named in a list is the listing whose page spells the name: its brand field followed by its model line, character for character, or the first words of its title when the page has no model line. Other listings of the same product are not that candidate: rank it on its own page, and when it is gone the next named candidate comes next.
- When they want a reason they can trust, prefer a listing whose brand field is filled, and never choose a title offering a refurbished or otherwise restored unit while a new one fits.
- The price is the price: a low price neither disqualifies nor recommends a listing, and the budget is a ceiling, not a wish for the cheapest - the price decides only when they ranked by it. A price that seems low for the kind of thing is no doubt, since the table is all you know of this store's prices; only the same model line's other listings on the table can show a price is off, and one at a small fraction of theirs (a 20 PHP phone beside the same model at 2,400 PHP) deserves less trust.

# Step 1 - assess the candidates that could win
- Your choice first, then its closest rivals, four at most - or every candidate the person named. The table lists first the variants whose copies state every must in their options or facts (in their facts when a reason is wanted): read those before settling.
- For each, look first at its brand field and shelf as the table writes them. Then write the strongest reason it may be the wrong item (twelve words at most); each of their musts with met, unmet or unknown and the page's words you rely on, quoted exactly (ten words at most; empty when nothing states it) - the money and the stock need no line - and, when they named the department, one line for it, met only when the shelf label is that department's name, quoted from the shelf; the preferences it meets; and its doubts in the page's own words, empty when there is none:
  - another kind: words showing another kind of thing than asked, or a form they ruled out - for their preferred maker's listing on the named department only its facts can show it;
  - look-alike: a brand or model spelled with changed characters;
  - other model: only when they named a model, a model line naming another;
  - restored: a title offering a refurbished or otherwise restored unit;
  - page: what makes the page less trustworthy (places contradicting each other, a fact that cannot be true).
- A brand field that differs from the title is none of these unless they named that maker, and a ranking step is not a must.
- Fits: every must met by its own page and no another-kind, look-alike or other-model doubt (nor a restored one when they want a reason) - a doubt every candidate shares separates none of them. A candidate they named fits when its page contradicts none of their musts.

# Step 2 - rank the fitting candidates
- Their ranking rule first, when they gave one: for each step, the value this variant's options or facts state for that property - as written, even when it looks odd, since you rank what the sellers state - and that value as a number in one unit for all candidates. A candidate stating nothing for a step ranks after those that do; a range or a list of figures ('10-40', '7/8/9') is no value; a title figure only breaks a tie. The first step decides and later steps only break ties.
- Then their preferences - the named model, then its colour, then the maker - given up only in the order they gave. When the colour they named is gone and they gave no second one, the same model in a plain, common colour is the safest substitute.
- Then the fewest doubts, then the plainest statement of the musts - a preference still comes before a plainer statement, so their maker's listing whose title states the value comes before another maker's plainer listing - then, among equally plain ones, a value the title and the options or facts both state before one a single place states, since two places of the page that agree are surer, then the listing earlier on the table. Never the lower price, unless they ranked by price.

# Step 3 - choose
- The first of that ranking. When they named candidates, only those are in the running, whatever they look like.
- When they asked for listings matching a phrase, the phrase and the department are their whole description: a listing on that department's shelf whose title carries every word of the phrase is the kind of thing they asked for - a part, a set, an accessory or a pack included - so it carries no another-kind doubt. Every candidate here carries the phrase; when they are equal, the first on the table, as the store ranked them.
- When nothing fits, say so, give three searches of three or four words for what to look for, and still choose the closest candidate.

# Step 4 - reply and claims
- When the person asked a question, a one-line reply in plain words.
- When a reason is required, claims about your choice copied from its page: the requested specification under the facts' own label with its value as written, the price as a bare number, in_stock true, and one more supported fact - the brand field exactly as the page writes it, whatever it says (a generic seller word too), or, only when the brand field is empty, another fact whose value plainly states its own property. The order tool asks for these, each true, and nothing more."""

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
                "description": "only when the person gave a ranking rule: each step's value as this variant's options or facts state it, quoted, and as a number; empty when they state none"},
            "preferences_met": {"type": "array", "items": {"type": "string"}},
            "doubts": {"type": "object", "properties": {
                "another_kind": {"type": "string", "description": "the page's words showing another kind of thing or a form the person ruled out; empty when none - for the preferred maker's listing on the named department, only words of its facts"},
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

JUDGE_TOOL_PLAIN = {"type": "function", "function": {**JUDGE_TOOL["function"], "parameters": {**JUDGE_TOOL["function"]["parameters"], "properties": {
    name: shape for name, shape in JUDGE_TOOL["function"]["parameters"]["properties"].items() if name != "claims"}}}}

PICK_TOOL = {"type": "function", "function": {
    "name": PICK,
    "description": "The variant to buy.",
    "parameters": {"type": "object", "properties": {
        "choice": {"type": "string", "description": "product_id::sku of the variant to buy, as the table writes it"},
        "why": {"type": "string", "description": "why, twelve words at most"}},
        "required": ["choice", "why"]}}}
PICK_NOTE = "Answer now with the pick tool alone: the key of the variant to buy by the rules above, and why in a few words."

CLAIMS_GUIDE = "# Personal shopper - this step: write the order's claims\n" + ROLE + """
- Here you write the justification of one order: short claims about the ordered variant, each copied from its page or the latest reading.
- A claim's field is price, in_stock, brand, or a label of the page's facts; its value exactly as the page or the reading writes it.
- The order tool asks for the requested specification under the facts' own label with its value as written, the latest price as a bare number, in_stock true, and at least one more supported fact - the brand field exactly as the page writes it, whatever it says (a generic seller word too), or, only when the brand field is empty, another fact whose value plainly states its own property (an amount with its unit for a measure, never a label whose value names something else). Write those, each true to the page, and nothing more."""

CLAIMS_TOOL = {"type": "function", "function": {
    "name": CLAIMS,
    "description": "The claims of the order, each copied from the ordered variant's page.",
    "parameters": {"type": "object", "properties": {"claims": {"type": "array", "items": {"type": "object", "properties": {
        "field": {"type": "string"}, "value": {"type": "string"}}, "required": ["field", "value"]}}},
        "required": ["claims"]}}}

SPELL_GUIDE = "# Personal shopper - the situation: a named model\n" + ROLE + """
- The person named the model they want, and perhaps its colour. Below are the listings the store showed, each with its brand field, model line, this variant's options and facts, copied from the pages.

# The two questions
- Read the model lines on their own: which listings' model line spells the code the person wrote, character for character apart from spaces and letter case (a model line ending in k200 spells K200)? A title or an option that merely resembles the code does not make a listing that model, a model line spelling another code makes it another model, and a model line naming several codes ('k200/k310') is not that one model when another listing's model line writes the code alone.
- Then, when they named a colour, which variants' options or colour fact write that colour as they wrote it - on its own or joined to other words ('k200-blue' writes blue)?

# Answer
- With the spell tool alone, with keys; none when no page spells it."""
SPELL_TOOL = {"type": "function", "function": {"name": "spell", "description": "The listings whose model line spells the named code, and the variants in the named colour.",
    "parameters": {"type": "object", "properties": {
        "model_keys": {"type": "array", "items": {"type": "string"}, "description": "product_id::sku of every variant whose page's model line writes the code, letter case and spaces aside"},
        "colour_keys": {"type": "array", "items": {"type": "string"}, "description": "product_id::sku of every variant whose options or colour fact write the named colour"}},
        "required": ["model_keys", "colour_keys"]}}}
RESOLVE_GUIDE = "# Personal shopper - the situation: candidates named by name\n" + ROLE + """
- The person named by name the candidates they will choose among.

# How a page spells a name
- For each name, find on the list below the one listing that is that candidate: the listing whose page spells the name - its brand field followed by its model line, character for character (a hyphen or a space is part of it), or the first words of its title when the page has no model line.
- A listing whose model line spells the name comes before one carrying the name only in its title; a model line saying something else is not overruled by the title.
- The store may sell the same product in other listings, under other brand fields, model lines or titles: only the one whose page spells the name is the candidate.
- When no listing spells a name, give that name no key.

# Answer
- With the resolve tool alone."""

RESOLVE_TOOL = {"type": "function", "function": {
    "name": RESOLVE,
    "description": "The listing each named candidate is.",
    "parameters": {"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string", "description": "The name as the person wrote it."},
        "key": {"type": "string", "description": "The key of the listing whose page spells that name, or an empty string."}},
        "required": ["name", "key"]}}}, "required": ["candidates"]}}}

SPLIT_GUIDE = "# Personal shopper - the situation: a model code the search did not find\n" + ROLE + """
- The person named a model code, and the store's search returned no listing for it written as they wrote it. Sellers spell a code their own way: words run together or apart, a space or a hyphen before the digits, the maker's name in front.

# What to write
- Three short store searches, each the item with the code spelled another way: the code's words apart (SUNRISEKETTLE12: sunrise kettle 12), the code split before its digits, and the code's distinctive part alone with the item.

# Answer
- With the tool alone."""
SPLIT_TOOL = {"type": "function", "function": {"name": "split", "description": "Three searches for a model code spelled as sellers write it.",
    "parameters": {"type": "object", "properties": {"searches": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}, "required": ["searches"]}}}

QUESTION = ("Before I choose: you said you have a firm requirement you have not stated yet. What exactly is that requirement, "
            "with its exact value?")
SHELF_QUESTION = ("Before I choose: which store category should I look in for this, and is there any other firm requirement I should "
                  "keep to?")


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


_TYPOGRAPHY = str.maketrans({"“": '"', "”": '"', "″": '"', "‘": "'", "’": "'", "′": "'", "–": "-", "—": "-", "\\": None, "\u200b": None, "\u200c": None, "\u200d": None, "\ufeff": None})


_ESCAPED = re.compile(r"\\u([0-9a-f]{4})")


def plain(text: Any) -> str:
    return _ESCAPED.sub(lambda match: chr(int(match.group(1), 16)), squash(text)).translate(_TYPOGRAPHY)


def tag(text: Any) -> str:
    return "".join(re.findall(r"[^\W_]+", squash(text)))


def words(text: Any) -> set:
    text = re.sub(r"[()\[\]{}\"/【】「」『』（）〈〉《》〔〕]", " ", str(text))
    return ({word.strip("'.,;:!?") for word in squash(text).split()} | set(re.findall(r"[^\W_]+", squash(text)))) - {""}


def tokens(text: Any) -> set:
    return set(re.findall(r"[^\W_]+", squash(text)))


def bag(text: Any) -> frozenset:
    return frozenset(word[:-1] if word.endswith("s") and len(word) > 3 else word for word in tokens(text))


def within(needle: Any, haystack: Any) -> bool:
    one, other = plain(needle), plain(haystack)
    return bool(one) and (one in other or one.replace(" ", "") in other.replace(" ", ""))


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*[^\d]*", text)
        return float(match.group(1)) if match else None


def figures(text: Any) -> List[Tuple[float, str]]:
    found = []
    for match in re.finditer(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*-?\s*([°]?[^\W\d_]*[\"']?)", plain(text)):
        value = number(match.group(1))
        if value is not None:
            found.append((value, match.group(2).lower()))
    return [(value, unit or next((later for figure, later in found[index + 1:] if figure == value and later), "")) for index, (value, unit) in enumerate(found)]


def unit_like(one: str, other: str) -> bool:
    first = lambda unit: next((char for char in unit if char.isalnum()), "")
    return not first(one) or not first(other) or first(one) == first(other)


def unit_stated(written: str, wanted: str) -> bool:
    return bool(written) or not wanted


def states(text: Any, value: float, unit: str, relation: str) -> bool:
    for found, written in figures(text):
        if not unit_like(written, unit) or not unit_stated(written, unit):
            continue
        if (relation == "at least" and found >= value) or (relation == "at most" and found <= value) or found == value:
            return True
    return False


def another_figure(text: Any, value: float, unit: str) -> Optional[float]:
    for found, written in figures(text):
        if unit and written and unit_like(written, unit) and found != value:
            return found
    return None


def spells(text: Any, code: Any) -> bool:
    parts, want = re.findall(r"[^\W_]+", squash(text)), "".join(re.findall(r"[^\W_]+", squash(code)))
    return bool(want) and any("".join(parts[i:j]) == want for i in range(len(parts)) for j in range(i + 1, len(parts) + 1))


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
    if not isinstance(message, dict):
        return None
    if cut_short(reply):
        if name != SHORTLIST:
            return None
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and function.get("name") == name:
                ids = list(dict.fromkeys(re.findall(r'"product_id"\s*:\s*"(\d+)"', str(function.get("arguments") or ""))))
                if ids:
                    return {"open": [{"product_id": product_id, "why": "named before the answer was cut"} for product_id in ids[:SHORTLIST_ROWS]], "search_instead": []}
        return None
    found = []
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
            if name != EXTRACT:
                return args
            found.append(args)
    return {"rows": [row for args in found for row in args.get("rows") or []]} if found else None


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
            if attempt < 2:
                Event().wait(1.0)
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
        self.awaited: set = set()

    def release(self, tokens: List[Any]) -> None:
        with self.lock:
            for token in tokens:
                if token in self.awaited:
                    self.awaited.discard(token)
                    self.flying -= 1

    def spend(self, reply: Any) -> None:
        usage = reply.get("usage") if isinstance(reply, dict) else None
        if not isinstance(usage, dict):
            return
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            cost = (number(usage.get("prompt_tokens")) or 0) * 0.643e-6 + (number(usage.get("completion_tokens")) or 0) * 2.02e-6
        with self.lock:
            self.dollars += float(cost)

    def attempt(self, model: Tuple[str, str], messages: List[Dict[str, Any]], tool: Dict[str, Any], seat: str, wait: float, bound: bool = False) -> Optional[Dict[str, Any]]:
        name = tool["function"]["name"]
        body = {"model": _native(model), "messages": messages, "tools": [tool], "tool_choice": {"type": "function", "function": {"name": name}},
                "temperature": TEMPERATURES[seat], "max_tokens": OUTPUT_TOKENS[seat], **reasoning_off()}
        client = ProxyClient(timeout=max(3, int(wait) + 1 if bound else int(max(wait, CLIENT_SECONDS))), max_retries=1)
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
                answer = self.attempt(first, messages, tool, seat, wait, bound=True)
                if answer is not None:
                    return answer
                continue
            partner = chain[0] if chain else None
            box: Queue = Queue()
            started = 0

            outstanding: List[Any] = []

            def launch(model: Tuple[str, str], seconds: float) -> bool:
                token = object()
                with self.lock:
                    if self.flying >= IN_FLIGHT:
                        return False
                    self.flying += 1
                    self.awaited.add(token)

                def run() -> None:
                    try:
                        box.put(self.attempt(model, messages, tool, seat, seconds))
                    except Exception:
                        box.put(None)
                    finally:
                        self.release([token])
                try:
                    Thread(target=run, daemon=True).start()
                except RuntimeError:
                    self.release([token])
                    return False
                outstanding.append(token)
                return True

            if launch(first, wait):
                started = 1
            else:
                answer = self.attempt(first, messages, tool, seat, wait, bound=True)
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
                    self.release(outstanding)
                    return answer
                if partner is not None and started == 1:
                    break
            self.release(outstanding)
            if started == 2 and chain:
                chain.pop(0)
        return None

    def second_opinion(self, seat: str, messages: List[Dict[str, Any]], tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        left = TIME_LIMIT_SECONDS - 10 - self.clock.elapsed()
        if left < 8:
            return None
        return self.attempt(SEATS[seat][-1], messages, tool, seat, min(WAIT_SECONDS[seat], left - 2), bound=True)


class Episode:
    def __init__(self, problem_data: Dict[str, Any]) -> None:
        environment = problem_data["environment"]
        view = environment["policy_view"]
        self.problem_id = str(problem_data.get("problem_id") or "problem")
        self.spelled_keys: set = set()
        self.spelled_colour: set = set()
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
        self.maybe_makers: List[str] = []
        self.maker_filtered: set = set()
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
        self.rank_units: Dict[Tuple[Tuple[str, str], str], str] = {}
        self.rank_converted: set = set()
        self.list_open = False

    def declared(self, name: str) -> Dict[str, Any]:
        return self.schemas.get(name, {}).get("properties") or {}

    def budget(self) -> Optional[float]:
        return number(self.record.get("max_price"))

    def musts(self) -> List[Dict[str, Any]]:
        return [must for must in self.record.get("musts") or [] if isinstance(must, dict) and must.get("text")]

    def figure_musts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for must in self.musts():
            if number(must.get("value")) is not None and not any(number(other.get("value")) == number(must.get("value")) and squash(other.get("unit")) == squash(must.get("unit")) for other in out):
                out.append(must)
        return out

    def named_keys(self) -> set:
        return {squash(item) for item in self.record.get("candidates") or [] if "::" in str(item) and str(item).split("::")[0].strip().isdigit()} | self.request_keys

    def named_names(self) -> List[str]:
        return [str(item) for item in self.record.get("candidates") or [] if not ("::" in str(item) and str(item).split("::")[0].strip().isdigit())]

    def reason(self) -> bool:
        return bool(self.claims_required or self.record.get("wants_reason"))

    def makers(self) -> List[str]:
        prefer = [item for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("text")]
        named = [str(item.get("text")) for item in prefer if item.get("kind") == "maker"]
        return named or [word for word in self.maybe_makers if any(squash(item.get("text")) == squash(word) for item in prefer)]

    def phrase_words(self) -> set:
        return tokens(self.record.get("phrase")) if self.record.get("phrase") else set()

    def phrase_holds(self, record: Dict[str, Any]) -> bool:
        phrase = tokens(record.get("phrase")) if record.get("phrase") else set()
        return bool(phrase) and sum(1 for row in self.rows.values() if phrase <= words(row.get("title"))) >= 3

    def models_named(self) -> List[str]:
        return [str(item.get("text")) for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("kind") == "model" and item.get("text")]

    def orderable(self, key: Tuple[str, str]) -> bool:
        in_stock, price = self.live.get(key, (None, None))
        return in_stock is not False and not over(price, self.budget()) and (not self.named_keys() or "%s::%s" % key in self.named_keys() or self.list_open)

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
        if args is not None and isinstance(args.get("candidates"), list):
            args["candidates"] = [item for item in args["candidates"] if ("::" in str(item) and str(item).split("::")[0].strip().isdigit()) or any(len(word) >= 3 and word in heard_words for word in words(item))]
            args["candidates"] = [str(item).replace("::", " ") if not str(item).split("::")[0].strip().isdigit() else item for item in args["candidates"]]
            if len(args["candidates"]) < 2:
                args["candidates"] = []
        old = dict(self.record)
        if args is not None:
            ok, record = fit(args, BRIEF_TOOL["function"]["parameters"])
            record = {key: value for key, value in record.items() if value not in (None, "", []) and value is not False}
        else:
            record = dict(old) if old else {"product": " ".join(self.request.split()[:8]), "musts": [], "searches": [" ".join(self.request.split()[:8])]}
            if not old:
                amounts = money(self.request)
                record["max_price"] = amounts[0] if len(amounts) == 1 else None
        if old:
            for name in ("product", "category", "musts", "max_price", "exclude", "prefer", "fall_back", "priorities", "candidates", "phrase", "wants_reason", "update_promised"):
                if record.get(name) in (None, "", []) and old.get(name) not in (None, "", []) and not (words(dumps(old.get(name))) & words(newest)):
                    record[name] = old[name]
            if self.asked and self.said:
                record.pop("unstated", None)
            record.pop("wants_reason", None)
            if old.get("wants_reason"):
                record["wants_reason"] = True
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
            for must in record["musts"]:
                if isinstance(must, dict):
                    must["unit"] = str(must.get("unit") or "").strip(" -–—/.,:;")
        musts_now = [must for must in record.get("musts") or [] if isinstance(must, dict)]
        musts_before = [must for must in old.get("musts") or [] if isinstance(must, dict)] if old else []
        if old and newest:
            changed = [must for must in musts_now if not any(same(must, other) for other in musts_before)]
            dropped = [must for must in musts_before if not any(same(must, other) for other in musts_now)]
            if changed or dropped:
                self.changed_musts, self.dropped_musts = changed, dropped
                if len(dropped) == 1 and len(changed) == 1 and unit_like(squash(dropped[0].get("unit")), squash(changed[0].get("unit"))):
                    for row in self.evidence.values():
                        for cell in row["values"]:
                            if squash(cell["requirement"]) == squash(dropped[0].get("text")):
                                cell["requirement"] = str(changed[0].get("text"))
                elif changed:
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
        phrase = [str(self.record.get("phrase"))] if self.phrase_words() else []
        queries = phrase + self.code_queries() + [str(query) for query in self.record.get("searches") or [] if str(query).strip()][:3]
        queries = [query for index, query in enumerate(queries) if squash(query) and bag(query) not in {bag(other) for other in queries[:index]}][:5 if not self.phrase_words() else 3]
        if not keyed:
            names = [name for name in self.named_names() if name not in queries][:4]
            product = str(self.record.get("product") or "").strip()
            queries += names + [("%s %s" % (name, product)).strip() for name in names if product and squash(product) not in squash(name)]
            queries += [code for code in self.models_named()[:1] if squash(code) not in {squash(query) for query in queries}]
            queries += [("%s %s" % (colour, product)).strip() for colour in self.colours_named()[:1] if product and squash("%s %s" % (colour, product)) not in {squash(query) for query in queries}]
            if not queries:
                queries = [str(self.record.get("product") or self.request)[:60]]
            queries = [query for index, query in enumerate(queries) if bag(query) not in {bag(other) for other in queries[:index]}]
            calls += [self.constrained(SEARCH, {"query": query}) for query in queries]
            calls += [dict(self.constrained(SEARCH, {"query": query}), code=True) for query in self.split_queries(queries)]
            calls += [dict(self.constrained(SEARCH, {"query": query}), figure=True) for query in self.maker_figure_queries(queries)]
            shelf = str(self.record.get("category") or self.record.get("product") or "")
            if shelf:
                calls.append(self.constrained(FILTER, {"category": shelf}))
            for maker in self.makers()[:MAKER_FILTERS]:
                calls.append(dict(self.constrained(FILTER, {"category": shelf, "brand": maker}), maker=True))
                calls.append(dict(self.constrained(FILTER, {"brand": maker}), maker=True))
            if not self.makers():
                maybe = [str(item.get("text")) for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("kind") == "other" and len(str(item.get("text") or "").split()) == 1 and squash(item.get("text")) not in squash(product)]
                calls += [dict(self.constrained(FILTER, {"brand": maker}), maybe=maker) for maker in maybe[:1]]
                calls += [dict(self.constrained(SEARCH, {"query": query}), maybe_figure=True) for maker in maybe[:1] for query in self.maker_figure_queries(queries, maker)]
        listed = bool(self.named_keys() or self.named_names())
        if MESSAGE in self.schemas and not self.asked and (self.record.get("unstated") or not (self.record.get("category") or self.record.get("phrase") or listed)):
            calls.insert(0, {"name": MESSAGE, "args": {"content": QUESTION if self.record.get("unstated") else SHELF_QUESTION}})
            self.asked = True
        calls = calls[:self.per_turn]
        before = set(self.rows)
        self.collect(calls, self.send([{"name": call["name"], "args": call["args"]} for call in calls]))

    def collect(self, calls: List[Dict[str, Any]], observations: List[Dict[str, Any]]) -> None:
        budget = self.budget()
        codes = {squash(code) for code in self.models_named()}

        def keep(obs: Dict[str, Any], into: List[str]) -> None:
            for row in obs.get("results") or []:
                if isinstance(row, dict) and row.get("in_stock") is not False and not over(row.get("price"), budget) and str(row.get("product_id")) not in into:
                    into.append(str(row.get("product_id")))
        for call, obs in zip(calls, observations):
            if call["name"] == FILTER and call["args"].get("brand"):
                self.maker_filtered.add(squash(call["args"]["brand"]))
            if call["name"] == SEARCH and (squash(call["args"].get("query")) in codes or call.get("code")):
                keep(obs, self.model_rows)
            if call.get("figure"):
                keep(obs, self.figure_rows)
            if call.get("maker") or (call.get("maybe") and obs.get("results")):
                keep(obs, self.maker_rows)
                if call.get("maybe") and call["maybe"] not in self.maybe_makers:
                    self.maybe_makers.append(call["maybe"])
        if self.maybe_makers:
            for call, obs in zip(calls, observations):
                if call.get("maybe_figure"):
                    keep(obs, self.figure_rows)
        musts = self.figure_musts()
        if musts:
            self.maker_rows.sort(key=lambda product_id: not any(states(self.rows.get(product_id, {}).get("title"), float(number(must.get("value"))), squash(must.get("unit")),
                                                                        str(must.get("relation") or "exact")) for must in musts))

    def colours_named(self) -> List[str]:
        return [str(item.get("text")) for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("kind") in ("colour", "color") and item.get("text")]

    def split_queries(self, done: List[str]) -> List[str]:
        if not self.models_named() or SEARCH not in self.schemas:
            return []
        answer = self.models.ask(SPLIT, [{"role": "system", "content": SPLIT_GUIDE}, {"role": "user", "content": "The item: %s. The code: %s." % (
            self.record.get("product") or self.request[:60], self.models_named()[0])}], SPLIT_TOOL)
        queries = [str(query) for query in (answer or {}).get("searches") or [] if str(query).strip() and squash(query) not in {squash(query) for query in done}][:3]
        return queries

    def carrying_rows(self) -> List[str]:
        budget = self.budget()
        musts = self.figure_musts()
        if not musts:
            return []
        return [product_id for product_id in self.row_order if product_id not in self.pages and self.rows[product_id].get("in_stock") is not False and not over(self.rows[product_id].get("price"), budget)
                and any(states(self.rows[product_id].get("title"), float(number(must.get("value"))), squash(must.get("unit")), str(must.get("relation") or "exact")) for must in musts)][:4]

    def code_queries(self) -> List[str]:
        if self.phrase_words() or self.named_keys():
            return []
        product = " ".join(squash(self.record.get("product") or "").split()[:4])
        out: List[str] = []
        must = next(iter(self.figure_musts()), None)
        if product and must is not None and must.get("unit"):
            figure, unit = "%g" % float(number(must.get("value"))), str(must.get("unit")).strip("- ")
            out += ["%s %s%s" % (product, figure, unit), "%s %s %s" % (product, figure, unit)]
            words_of = [word for word in product.split() if len(word) > 2]
            if len(words_of) > 1:
                out += ["%s %s%s" % (words_of[0], figure, unit), "%s %s%s" % (words_of[-1], figure, unit)]
        elif product:
            out.append(product)
        category = squash(self.record.get("category")) if self.record.get("category") else ""
        if category and category != product:
            out.append(category)
        return out[:3]

    def maker_figure_queries(self, done: List[str], maker: str = "") -> List[str]:
        maker = maker or next(iter(self.makers()), "")
        if self.phrase_words() or not maker or not self.figure_musts():
            return []
        must = self.figure_musts()[0]
        base = "%s %s %g" % (maker, str(self.record.get("product") or "").strip(), float(number(must.get("value"))))
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
        want = limit or SHORTLIST_ROWS
        searched: List[str] = []
        if phrase and only is None:
            want = 3
            budget_ok = lambda product_id: not over(self.rows.get(product_id, {}).get("price"), budget) and self.rows.get(product_id, {}).get("in_stock") is not False
            verbatim = squash(self.record.get("phrase"))
            order = sorted(self.search_rows, key=lambda item: squash(item[0]) != verbatim)
            searched = [product_id for product_id in dict.fromkeys(product_id for _, found in order for product_id in found)
                        if product_id in self.rows and product_id not in self.pages and phrase <= words(self.rows[product_id].get("title")) and budget_ok(product_id)]
            if searched:
                rows = [self.rows[product_id] for product_id in searched]
        names = [squash(name) for name in self.named_names()]
        named_rows = [product_id for query, found in self.search_rows if any(squash(query).startswith(name) for name in names) for product_id in found[:3]] if only is None else []
        messages = [{"role": "system", "content": SHORTLIST_GUIDE},
                    {"role": "user", "content": "Order conditions: %s\n\n%sThe rows:\n%s" % (dumps(self.record), note + "\n\n" if note else "", render_rows(rows, budget, keep, phrase, self.found_by))}]
        args = self.models.ask(SHORTLIST, messages, SHORTLIST_TOOL, deadline=HURRY_SECONDS + 20)
        picked = [str(item.get("product_id")) for item in (args or {}).get("open") or [] if isinstance(item, dict)]
        if args is None:
            picked = [str(row.get("product_id")) for row in rows if not over(row.get("price"), budget)][:want]
        self.instead = [str(query) for query in (args or {}).get("search_instead") or [] if str(query).strip()][:3]
        if only is None:
            code_alone = {squash("search %s" % dumps(code)) for code in self.models_named()}
            model_rows = sorted(self.model_rows, key=lambda product_id: (not any(squash(source) in code_alone for source in self.found_by.get(product_id, [])),
                                                                      not any(source.startswith("filter") for source in self.found_by.get(product_id, [])),
                                                                      -sum(source.startswith("search") for source in self.found_by.get(product_id, []))))
            front = self.figure_rows[:FIGURE_ROWS] + self.titled_rows() + self.maker_rows[:MAKER_ROWS] + model_rows[:MODEL_ROWS] + named_rows
        else:
            front = list(lead or [])
        ids = [product_id for product_id in dict.fromkeys(front + [product_id for product_id in picked if product_id in self.rows]) if product_id not in self.pages]
        if phrase and only is None:
            chosen = set(ids)
            ids = [product_id for product_id in searched if product_id in chosen] + [product_id for product_id in searched if product_id not in chosen] if searched else [
                product_id for product_id in dict.fromkeys(list(ids) + [row_id for row_id in self.row_order if row_id not in self.pages])
                if phrase <= words(self.rows.get(product_id, {}).get("title")) and not over(self.rows.get(product_id, {}).get("price"), budget)] or ids
            ids = list(dict.fromkeys(ids))
        if not ids and only is None:
            pool = [row.get("product_id") for row in rows if not over(row.get("price"), budget)]
            pool.sort(key=lambda product_id: (not any(source.startswith("filter") for source in self.found_by.get(product_id, [])), -len(self.found_by.get(product_id, []))))
            ids = pool[:want]
        return ids[:want if phrase else max(want, SHORTLIST_ROWS + MAKER_ROWS + MODEL_ROWS + 2 * FIGURE_ROWS)] if only is None else ids[:want]

    def phrase_note(self) -> str:
        if not self.phrase_words():
            return ""
        shelf = "on the %s shelf" % dumps(self.record.get("category")) if self.record.get("category") else "on the shelf they named"
        return ("The person wants any listing %s whose title carries the words %s, and every row below carries them, in the store's order for them. Pick every row that could sit on that shelf - a part, a set or a pack of that department's goods counts, since they named no narrower kind of thing - and leave out only rows whose titles show another department's goods; price, brand and pack size are no reason to pass a row over." % (
            shelf, dumps(self.record.get("phrase"))))

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
        calls = [] if self.phrase_words() else [self.constrained(SEARCH, {"query": str(query)}) for query in self.record.get("searches") or [] if str(query).strip() and squash(query) not in done][:3]
        must = next(iter(self.figure_musts()), None)
        if must is not None and must.get("unit") and number(must.get("value")) is not None and not self.phrase_words():
            joined = "%g%s" % (number(must.get("value")), str(must.get("unit")).strip("- "))
            words_of = [word for word in squash(self.record.get("product") or "").split() if len(word) > 2]
            for word in ([words_of[0], words_of[-1]] if len(words_of) > 1 else words_of):
                if squash("%s %s" % (word, joined)) not in done:
                    calls.append(self.constrained(SEARCH, {"query": "%s %s" % (word, joined)}))
        calls += [self.constrained(SEARCH, {"query": query}) for query in self.feedback_queries() if squash(query) not in done]
        if self.record.get("category"):
            calls.append(self.constrained(FILTER, {"category": str(self.record["category"])}))
        if not self.phrase_words():
            shelf = str(self.record.get("category") or self.record.get("product") or "")
            for maker in [maker for maker in self.makers()[:MAKER_FILTERS] if squash(maker) not in self.maker_filtered]:
                calls += [dict(self.constrained(FILTER, {"category": shelf, "brand": maker}), maker=True), dict(self.constrained(FILTER, {"brand": maker}), maker=True)]
                calls += [dict(self.constrained(SEARCH, {"query": query}), figure=True) for query in self.maker_figure_queries([query for query, _ in self.search_rows], maker)]
        calls = [call for call in {(call["name"], dumps(call["args"])): call for call in calls}.values() if call["name"] in self.schemas][:self.per_turn]
        if calls:
            self.collect(calls, self.send([{"name": call["name"], "args": call["args"]} for call in calls]))

    def open(self, ids: List[str]) -> List[str]:
        ids = [product_id for product_id in dict.fromkeys(ids) if product_id not in self.pages][:5 * min(4, self.per_turn)]
        if not ids or COMPARE not in self.schemas or self.session.turns_left() <= 3:
            return []
        self.send([{"name": COMPARE, "args": {"product_ids": ids[i:i + 5]}} for i in range(0, len(ids), 5)])
        return [product_id for product_id in ids if product_id in self.pages]

    def feedback_queries(self, limit: int = 2) -> List[str]:
        if self.phrase_words() or self.named_keys():
            return []
        said = words(self.request) | words(" ".join(self.said)) | {word for query, _ in self.search_rows for word in words(query)}
        counts: Dict[str, int] = {}
        for row in self.rows.values():
            for word in set(tokens(row.get("title"))):
                if len(word) >= 3 and not word.isdigit() and word not in said:
                    counts[word] = counts.get(word, 0) + 1
        top = [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit] if counts[word] >= 3]
        must = next(iter(self.figure_musts()), None)
        tail = "%g%s" % (float(number(must.get("value"))), str(must.get("unit") or "").strip("- ")) if must is not None and must.get("unit") else " ".join(squash(self.record.get("product") or "").split()[:2])
        return ["%s %s" % (word, tail) for word in top if tail]

    def requirement_names(self) -> List[str]:
        items = [str(must.get("text")) for must in self.musts()]
        items += [str(item.get("text")) for item in self.record.get("priorities") or [] if isinstance(item, dict) and item.get("direction") in ("higher", "lower")]
        items += [str(item.get("text")) for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("kind") in ("model", "colour", "color")]
        return [item for index, item in enumerate(items) if item and squash(item) not in {squash(other) for other in items[:index]}]

    def extract(self, ids: List[str]) -> None:
        fresh = [product_id for product_id in ids if product_id in self.pages and product_id not in self.extracted]
        if not fresh:
            return
        self.extracted |= set(fresh)
        names = self.requirement_names()
        if not names:
            return
        chunks = [fresh[i:i + EXTRACT_PAGES] for i in range(0, len(fresh), EXTRACT_PAGES)][:EXTRACT_IN_FLIGHT * 2]
        tool = json.loads(json.dumps(EXTRACT_TOOL))
        shape = tool["function"]["parameters"]["properties"]["rows"]["items"]
        shape["properties"]["values"]["items"]["properties"]["requirement"]["enum"] = names
        excluded = [str(item) for item in self.record.get("exclude") or [] if str(item).strip()]
        if not excluded:
            shape["properties"].pop("excluded_words", None)
        budget = self.budget()

        def job(chunk: List[str]) -> Optional[Dict[str, Any]]:
            pages = "\n\n".join(render_page(self.pages[product_id]) for product_id in chunk)
            messages = [{"role": "system", "content": EXTRACT_GUIDE},
                        {"role": "user", "content": "The shopper's money: at most %s PHP.%s Requirements to copy: %s\n\nThe pages:\n\n%s" % (
                            budget, " Forms they ruled out: %s." % "; ".join(excluded) if excluded else "", "; ".join(names), pages)}]
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
                thread.join(timeout=max(1.0, min(60.0, TIME_LIMIT_SECONDS - 5 - self.clock.elapsed())))
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
            own.sort(key=lambda key: not self.copies_prefer(key))
            keys += own[:PAGE_ROWS]
        musts = self.figure_musts()
        if musts:
            good = ("facts",) if self.reason() else ("options", "facts")
            keys.sort(key=lambda key: (not all(self.copies_prove(key, must, good) for must in musts), not all(self.copies_prove(key, must, ("options", "facts", "title")) for must in musts)))
        if self.spelled_keys and not self.lost:
            keys.sort(key=lambda key: (key not in (self.spelled_colour or self.spelled_keys), key not in self.spelled_keys))
        return keys

    def copies_prefer(self, key: Tuple[str, str]) -> bool:
        prefs = [item.get("text") for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("kind") in ("colour", "color") and item.get("text")]
        return any(within(pref, cell.get("options")) for pref in prefs for cell in (self.evidence.get(key) or {}).get("values", []) if cell.get("options"))

    def cell(self, key: Tuple[str, str], must: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return next((value for value in (self.evidence.get(key) or {}).get("values", []) if squash(value.get("requirement")) == squash(must.get("text"))), None)

    def copied(self, key: Tuple[str, str], must: Dict[str, Any]) -> Tuple[List[str], Optional[float], Optional[float]]:
        cell = self.cell(key, must)
        if cell is None:
            return [], None, None
        value, unit, relation = float(number(must.get("value")) or 0), squash(must.get("unit")), str(must.get("relation") or "exact")
        present = [place for place in ("options", "facts", "title") if cell.get(place) and states(str(cell[place]).split("=", 1)[-1], value, unit, relation)]
        if self.reason() and "facts" in present and tag(str(cell["facts"]).split("=", 1)[0]) == "model":
            present = ["model line" if place == "facts" else place for place in present]
        raw = next((str(v.get("options") or "") for v in (self.pages.get(key[0]) or {}).get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), "")
        clash = next((found for text in (str(cell.get("options") or "").split("=", 1)[-1], raw) if text for found in [another_figure(text, value, unit)] if found is not None), None)
        page_title = str((self.pages.get(key[0]) or {}).get("title") or "")
        titles = (str(cell.get("title") or page_title), page_title)
        listed = any(len({found for found, written in figures(text) if written and unit_like(written, unit)}) > 1 or re.search(r"\d\s*[/&\-–~]\s*\d", text) for text in titles)
        title_clash = another_figure(titles[0], value, unit) if present and "title" not in present and titles[0] and not listed else None
        return present, clash, title_clash

    def other_figure(self, key: Tuple[str, str], must: Dict[str, Any]) -> bool:
        page = self.pages.get(key[0]) or {}
        value, unit = float(number(must.get("value")) or 0), squash(must.get("unit"))
        if not unit:
            return False
        text = " ".join([str(page.get("title") or "")] + ["%s" % fact for fact in (page.get("facts") or {}).values()])
        return any(found != value and written and unit_like(written, unit) and found > 0 for found, written in figures(text))

    def bundled(self, key: Tuple[str, str], must: Dict[str, Any]) -> bool:
        text = str((self.cell(key, must) or {}).get("facts") or "").split("=", 1)[-1]
        pair = r"\d+(?:\.\d+)?\s*[x×*]\s*\d+(?:\.\d+)?"
        return len(set(re.findall(pair, text)) | {found for found, _ in figures(re.sub(pair, " ", text))}) > 1

    def stated(self, key: Tuple[str, str], must: Dict[str, Any]) -> Tuple[str, str]:
        cell = self.cell(key, must)
        if cell is None:
            return "none", ""
        value, unit = float(number(must.get("value")) or 0), squash(must.get("unit"))
        present, clash, title_clash = self.copied(key, must)
        good = ("facts",) if self.reason() else ("options", "facts")
        if set(present) & set(good):
            return ("contradiction" if clash is not None or title_clash is not None else "plain"), ", ".join(present)
        tail = lambda place: str(cell.get(place) or "").split("=", 1)[-1]
        bare = next((place for place in ("facts", "options") if cell.get(place) and any(found == value and not written for found, written in figures(tail(place)))), None)
        if bare:
            return "bare", bare
        if present:
            return "weak", ", ".join(present)
        other = next((place for place in ("options", "facts", "title") if cell.get(place) and another_figure(tail(place), value, unit) is not None), None)
        if other:
            return "other", other
        foreign = next((place for place in ("facts", "options", "title") if cell.get(place) and any(written and not unit_like(written, unit) for _, written in figures(tail(place)))), None)
        return ("foreign", foreign) if foreign else ("none", "")

    def plain_row(self, key: Tuple[str, str]) -> bool:
        musts = self.figure_musts()
        return bool(musts) and all(self.stated(key, must)[0] == "plain" for must in musts) and (not self.reason() or bool((self.pages.get(key[0]) or {}).get("brand")))

    def figure_note(self, key: Tuple[str, str], must: Dict[str, Any]) -> str:
        cell = self.cell(key, must)
        if cell is None:
            return ""
        value, unit, said = float(number(must.get("value")) or 0), squash(must.get("unit")), "%g %s" % (float(number(must.get("value")) or 0), must.get("unit") or "")
        present, clash, title_clash = self.copied(key, must)
        page_title = str((self.pages.get(key[0]) or {}).get("title") or "")
        several = len({found for found, written in figures(page_title) if written and unit_like(written, unit)}) > 1 or bool(re.search(r"\d\s*[/&\-–~]\s*\d", page_title))
        tail = lambda place: str(cell.get(place) or "").split("=", 1)[-1]
        bare = next((place for place in ("facts", "options") if cell.get(place) and any(found == value and not written for found, written in figures(tail(place)))), None)
        if present:
            return "the person's %s appears in: %s%s%s%s%s" % (said.strip(), ", ".join(present), " - but this variant's options write %g" % clash if clash is not None else "",
                                                           " - but its title writes %g" % title_clash if title_clash is not None else "",
                                                           " - the title lists several figures, so only the options or the facts pin this variant's" if several and "title" in present and len(present) == 1 else "",
                                                           "; the %s write the number %g with no unit after it" % (bare, value) if bare and bare not in present else "")
        if bare:
            return "the %s write the number %g with no unit after it (%s) - read whether its label names the unit" % (bare, value, str(cell.get(bare))[:80])
        other = next(((place, found) for place in ("options", "facts", "title") if cell.get(place) for found in [another_figure(tail(place), value, unit)] if found is not None), None)
        if other:
            return "the copies write %g, not the person's %s (in %s)" % (other[1], said.strip(), other[0])
        tier, place = self.stated(key, must)
        if tier == "foreign":
            return "no copy writes %s as such; the %s write %s - whether that is the very amount they named is yours to read" % (said.strip(), place, dumps(str(cell.get(place))[:80]))
        return "the person's %s appears in none of the copies" % said.strip()

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
                notes.append("The person changed a requirement: it is now %s - exactly that value, which they have just corrected, so be surest of it: only a variant whose options or facts state it meets it (a title alone, the value they dropped or any other value does not), and their preferences count only among those." % dumps(must.get("text")))
            else:
                notes.append("The person's requirements now include %s: judge every candidate against it." % dumps(must.get("text")))
        for key in self.lost:
            in_stock, price = self.live.get(key, (None, None))
            notes.append("The variant carted before (%s::%s, %s) is %s and is not a candidate; rows marked SAME TITLE or SAME SKU FAMILY are the same product in another listing%s." % (
                key[0], key[1], str((self.pages.get(key[0]) or {}).get("title") or "")[:80], "sold out" if in_stock is False else "now %s PHP, above the budget" % price,
                " - but they carry the old value of a changed requirement unless their page states the new one" if self.changed_musts and self.dropped_musts else ""))
        if self.question_open and self.said:
            notes.append("The person asked: %s - write your one-line reply." % dumps(self.said[-1][:200]))
        if self.reason():
            notes.append("A reason is required with the order: write the claims about your choice.")
        if self.spelled_keys and not self.lost:
            notes.append("The model lines were read on their own: the code the person wrote is spelled by the model line of %s%s." % (
                ", ".join("%s::%s" % key for key in sorted(self.spelled_keys)[:4]), "; of these, %s is in the colour they named" % ", ".join("%s::%s" % key for key in sorted(self.spelled_colour)[:4]) if self.spelled_colour else ""))
        return "\n".join(notes)

    def judge(self, keys: List[Tuple[str, str]]) -> Tuple[Optional[Tuple[str, str]], Optional[Dict[str, Any]]]:
        table = self.table(keys)
        messages = [{"role": "system", "content": JUDGE_GUIDE},
                    {"role": "user", "content": "The person's request:\n%s\n\nWhat they said since:\n%s\n\nOrder conditions: %s\n\n%s\n\nCandidates:\n%s" % (
                        self.request, "\n".join("- " + text for text in self.said) or "(nothing)", dumps(self.record), self.situation(), table)}]
        tool = self.judge_tool()
        answer = self.models.ask(JUDGE, messages, tool, deadline=WRAP_UP_SECONDS + 10)
        answers = [answer] if answer is not None else []
        steps = [item for item in self.record.get("priorities") or [] if isinstance(item, dict)]
        for other in answers:
            for entry in other.get("candidates") or []:
                found = self.key_of(entry.get("key"), keys) if isinstance(entry, dict) else None
                for step in steps if found else []:
                    self.rank_value(found, entry, step)
        if answer is None:
            picked = self.models.ask(PICK, messages + [{"role": "user", "content": PICK_NOTE}], PICK_TOOL, deadline=WRAP_UP_SECONDS + 10)
            if picked is None and self.clock.elapsed() < WRAP_UP_SECONDS:
                picked = self.models.second_opinion(PICK, messages + [{"role": "user", "content": PICK_NOTE}], PICK_TOOL)
            key = self.key_of((picked or {}).get("choice"), keys)
            return (key, {"choice": picked.get("choice"), "why": picked.get("why")}) if key is not None and picked else (None, None)
        key, problem = self.verify(answer, keys)
        if problem and self.clock.elapsed() < WRAP_UP_SECONDS - 20:
            messages.append({"role": "user", "content": "Your answer was: %s\n\n%s Answer again with the judge tool." % (dumps(answer)[:6000], problem)})
            second = self.models.ask(JUDGE, messages, tool, deadline=WRAP_UP_SECONDS + 10)
            if second is not None:
                answer = second
                key, _ = self.verify(second, keys, final=True)
        return key, answer

    def judge_tool(self) -> Dict[str, Any]:
        tool = JUDGE_TOOL if self.reason() else JUDGE_TOOL_PLAIN
        if not any(isinstance(step, dict) and step.get("direction") in ("higher", "lower") for step in self.record.get("priorities") or []):
            return tool
        tool = json.loads(json.dumps(tool))
        item = tool["function"]["parameters"]["properties"]["candidates"]["items"]
        item["required"] = item["required"] + ["ranks"]
        return tool

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
        facts = page.get("facts") or {}
        places = (("options", options), ("facts", "; ".join("%s = %s" % item for item in facts.items() if tag(item[0]) != "model")),
                  ("model line" if self.reason() else "facts", "; ".join("%s = %s" % item for item in facts.items() if tag(item[0]) == "model")), ("title", str(page.get("title") or "")))
        place = next((name for name, text in places if text and within(quote, text)), "")
        return "" if place != "options" and self.copied(key, must)[1] is not None else place

    def copies_prove(self, key: Tuple[str, str], must: Dict[str, Any], places: Tuple[str, ...] = ("options", "facts")) -> bool:
        present, clash, _ = self.copied(key, must)
        return clash is None and bool(set(places) & set(present))

    def identity(self, entry: Dict[str, Any]) -> List[str]:
        doubts = entry.get("doubts") if isinstance(entry.get("doubts"), dict) else {}
        kinds = ["another_kind", "look_alike"] + (["other_model"] if self.models_named() and not self.lost else []) + (["restored"] if self.reason() else [])
        return ["%s: %s" % (kind.replace("_", " "), doubts[kind]) for kind in kinds if squash(doubts.get(kind)) not in ("", "none", "no", "n/a", "-")]

    def sound(self, entry: Dict[str, Any]) -> bool:
        return bool(entry.get("fits")) and not self.identity(entry) and self.shelved(entry) and all(self.line_for(entry, must).get("verdict", "met") == "met" for must in self.musts())

    def shelved(self, entry: Dict[str, Any]) -> bool:
        shelf = str(self.record.get("category") or "")
        if not shelf or self.named_keys() or self.named_names():
            return True
        names = lambda quote: {plain(piece) for piece in re.split(r"[>/|]", str(quote or ""))}
        return any(isinstance(line, dict) and str(line.get("verdict")) == "met" and plain(shelf) in names(line.get("quote")) for line in entry.get("musts") or [])

    def prefs_met(self, entry: Dict[str, Any]) -> int:
        wanted = [words(item.get("text")) for item in self.record.get("prefer") or [] if isinstance(item, dict) and item.get("text")]
        return sum(1 for said in entry.get("preferences_met") or [] if any(want & words(said) for want in wanted)) if wanted else 0

    def model_met(self, entry: Dict[str, Any]) -> bool:
        return any(words(model) & words(said) for model in self.models_named() for said in entry.get("preferences_met") or [])

    def makers_own(self, key: Tuple[str, str]) -> bool:
        brands = {squash(self.rows[other].get("brand")) for other in self.maker_rows if other in self.rows and self.rows[other].get("brand")}
        return key[0] in self.maker_rows or squash((self.pages.get(key[0]) or {}).get("brand")) in brands

    def maker_note(self, key: Tuple[str, str], entry: Dict[str, Any], keys: List[Tuple[str, str]]) -> str:
        if not self.makers() or not self.maker_rows or self.named_keys() or self.named_names() or self.makers_own(key) or entry.get("preferences_met"):
            return ""
        theirs = [other for other in keys if other != key and self.makers_own(other)]
        if self.changed_musts and self.dropped_musts:
            theirs = [other for other in theirs if all(self.copies_prove(other, must) for must in self.changed_musts if number(must.get("value")) is not None)]
        if not theirs:
            return ""
        shelf = lambda other: " > ".join(str(label) for label in (self.pages.get(other[0]) or {}).get("category_path") or []) or "not shown"
        return ("The person prefers %s, and their firm requirements matter most: the store's %s listings on the table are %s. Each of them on the department the person named is that kind of thing in this store, whatever its title calls it. If one of them meets every must on its own page (a value written in another unit of the same amount counts), it comes before any other maker's listing - choose it; if none does, keep yours and say in why which must each of them fails." % (
            " / ".join(self.makers()[:2]), " / ".join(self.makers()[:2]), ", ".join("%s::%s (shelf: %s)" % (other[0], other[1], shelf(other)) for other in theirs[:4])))

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
        self.backups = list(dict.fromkeys(other for other in [other for other in order[1:] if other not in entries or self.sound(entries[other])]
                                          + [found for found, entry in entries.items() if self.sound(entry)] if other != key))
        if key is None or final:
            return key, ""
        entry = entries.get(key) or {}
        if key not in entries and entries:
            note = self.maker_note(key, {}, keys)
            troubled = ("bare", "contradiction", "other", "weak") if self.reason() else ("bare", "contradiction", "other")
            weak = next((must for must in self.figure_musts() if self.stated(key, must)[0] in troubled), None)
            plainer = next((other for other in keys if other != key and self.plain_row(other)), None)
            evidence = " Its copies: %s; %s::%s states it plainly." % (self.figure_note(key, weak), plainer[0], plainer[1]) if weak is not None and plainer is not None else ""
            pool = {other for other in self.spelled_colour if other not in self.lost} or {other for other in self.spelled_keys if other not in self.lost}
            spelled = [other for other in keys if other != key and other in pool and self.orderable(other)] if self.spelled_keys and key not in self.spelled_keys else []
            model_note = " The model lines were read on their own: %s::%s's model line spells %s as the person wrote it, and a code in a title or an option alone is advertising." % (
                spelled[0][0], spelled[0][1], dumps(self.models_named()[0])) if spelled else ""
            return key, ("Your choice %s::%s is not among the candidates you assessed. Assess it too - its musts with the page's words quoted, its doubts - or choose one you assessed." % key) + evidence + model_note + (" " + note if note else "")
        if self.spelled_keys and key not in self.spelled_keys:
            pool = {other for other in self.spelled_colour if other not in self.lost} or {other for other in self.spelled_keys if other not in self.lost}
            better = [other for other in keys if other != key and other in pool and self.orderable(other) and (other not in entries or self.sound(entries[other]))]
            if better:
                return key, ("The model lines were read on their own: %s::%s's model line spells %s as the person wrote it, while your choice's reads %s. The named model is the page whose model line spells it, whatever a title or an option says. Choose it, or keep yours and say in why yours is that model." % (
                    better[0][0], better[0][1], dumps(self.models_named()[0]), dumps(((self.pages.get(key[0]) or {}).get("facts") or {}).get("model") or "(none)")))
        good = ("facts",) if self.reason() else ("options", "facts")
        changed = {squash(must.get("text")) for must in self.changed_musts} if self.dropped_musts else set()
        mine = self.prefs_met(entry)
        index = {other: position for position, other in enumerate(keys)}
        doubted = lambda their: squash(((their or {}).get("doubts") or {}).get("page")) not in ("", "none", "no", "n/a", "-")
        standing = lambda other, must=None: (not self.plain_row(other), must is not None and self.stated(other, must)[0] == "contradiction", doubted(entries.get(other)),
                                             self.prefs_met(entries.get(other) or {}) < mine, bool(self.reason() and must is not None and self.bundled(other, must)),
                                             bool(self.reason() and not (self.pages.get(other[0]) or {}).get("brand")), index.get(other, len(keys)))
        for must in self.figure_musts():
            place = self.proof(key, entry, must)
            omitted = place not in good and (place == "model line" or not plain(self.line_for(entry, must).get("quote") or "")) and self.copies_prove(key, must, good)
            if place in good or omitted:
                place = place if place in good else " or ".join(good)
                title_clash = self.copied(key, must)[2]
                steady = sorted([other for other, their in entries.items() if other != key and self.sound(their) and self.proof(other, their, must) in good
                                 and self.copied(other, must)[2] is None and self.prefs_met(their) >= mine and not doubted(their)
                                 and (not self.reason() or (self.pages.get(other[0]) or {}).get("brand") or not (self.pages.get(key[0]) or {}).get("brand"))] if title_clash is not None else [],
                                key=lambda other: standing(other, must))
                if steady:
                    return key, ("Your choice %s::%s writes %g in its title but %g %s in its %s: the page contradicts itself. %s::%s states %g %s without a contradiction. Choose it, or say in why your page is right." % (
                        key[0], key[1], title_clash, number(must.get("value")), must.get("unit") or "", place, steady[0][0], steady[0][1], number(must.get("value")), must.get("unit") or ""))
                continue
            strict = good == ("facts",) or squash(must.get("text")) in changed or not self.record.get("prefer")
            rivals = [other for other, their in entries.items() if other != key and self.sound(their) and self.proof(other, their, must) in good
                      and (strict or self.prefs_met(their) >= mine)]
            rivals.sort(key=lambda other: standing(other, must))
            quote = self.line_for(entry, must).get("quote")
            where = {"title": "stands only in its title", "options": "stands only in its options, and a reason quotes a fact", "model line": "stands only on its model line, which names a model"}
            if rivals:
                return key, ("Your choice %s::%s is not shown to meet %s in its own options or facts: %s. %s::%s states %g %s there (%s). Choose that one, or quote the words of your choice's options or facts that state that very value." % (
                    key[0], key[1], dumps(must.get("text")), "your quote %s %s" % (dumps(quote), where[place]) if quote and place in where else "your quote %s does not write that value on its page" % dumps(quote) if quote else "you quoted nothing for it",
                    rivals[0][0], rivals[0][1], number(must.get("value")), must.get("unit") or "", self.figure_note(rivals[0], must) or "its quote"))
            if self.reason() and not self.copies_prove(key, must, good) and self.stated(key, must)[0] != "foreign":
                return key, ("Your choice %s::%s states %s %s; a reason the person can trust quotes a fact written under that property's own label. Choose a candidate whose facts state it, or say none_fit with three searches for listings whose facts do, and still choose the closest." % (
                    key[0], key[1], dumps(must.get("text")), {"title": "only in its title", "options": "only in its options", "model line": "only on its model line, which names a model"}.get(place, "in none of its facts")))
        shelf = str(self.record.get("category") or "")
        if shelf and not self.named_keys() and not self.named_names() and not self.shelved(entry):
            shelved = [other for other, their in entries.items() if other != key and self.sound(their)]
            path = " > ".join(str(label) for label in (self.pages.get(key[0]) or {}).get("category_path") or []) or "(not shown)"
            return key, ("The person's required department is %s, and your choice %s::%s (category path: %s) has no met line quoting that shelf label from its category path: a listing on another shelf is not what they asked for, whatever its title or facts state. %sAnswer again with a line for that department on every candidate - met only when the shelf label the table writes is that department's name, quoted - and choose one whose shelf is that department, or say none_fit when none is." % (
                dumps(shelf), key[0], key[1], path, "%s::%s quotes it and fits. " % shelved[0] if shelved else ""))
        sold_out = [lost for lost in self.lost if self.live.get(lost, (None, None))[0] is False]
        if sold_out and not self.named_keys() and not self.named_names() and not (self.changed_musts and self.dropped_musts):
            lost_stems = {stem(lost[1]) for lost in sold_out}
            same = lambda other: stem(other[1]) in lost_stems
            clean = lambda other: all(self.copied(other, must)[1] is None for must in self.figure_musts())
            proves = lambda other: all(self.copies_prove(other, must) for must in self.figure_musts())
            sisters = [other for other, their in entries.items() if other != key and same(other) and self.sound(their) and clean(other) and proves(other) and self.prefs_met(their) >= mine]
            if not self.model_met(entry):
                sisters += [other for other in keys if other != key and other not in entries and same(other) and self.orderable(other) and clean(other) and proves(other)]
            if sisters and not same(key) and key not in self.spelled_keys:
                return key, ("%s::%s is the same product as the variant carted before - the same sku family - and fits. After a loss the same product in another listing comes before another product, in another colour too. Choose it, or keep yours and say in why yours is the same product." % sisters[0])
        note = self.maker_note(key, entry, keys)
        if note:
            return key, note
        if self.makers() and self.maker_rows and not self.makers_own(key) and entry.get("preferences_met") and not self.named_keys() and not self.named_names():
            theirs = [other for other, their in entries.items() if other != key and self.makers_own(other) and self.sound(their)]
            if theirs:
                return key, ("The store's filter for %s did not return your choice %s::%s, whose brand field is %s, while %s::%s, which that filter returned, fits. The brand field says whose listing it is: choose it, or keep yours and say in why yours is that maker's." % (
                    " / ".join(self.makers()[:2]), key[0], key[1], (self.pages.get(key[0]) or {}).get("brand") or "(empty)", theirs[0][0], theirs[0][1]))
        if self.reason() and not (self.pages.get(key[0]) or {}).get("brand") and not self.named_keys() and not self.named_names():
            branded = sorted([other for other, their in entries.items() if other != key and self.sound(their) and (self.pages.get(other[0]) or {}).get("brand")
                              and all(self.copies_prove(other, must, good) for must in self.figure_musts())], key=standing)
            if branded:
                return key, ("The person wants a reason they can trust, and your choice %s::%s has an empty brand field - no maker stands behind it - while %s::%s fits with its brand field filled. Choose it, or keep yours and say in why a claim about it can be trusted." % (
                    key[0], key[1], branded[0][0], branded[0][1]))
            listed = [other for other in keys if other != key and other not in entries and (self.pages.get(other[0]) or {}).get("brand") and self.orderable(other)]
            if listed:
                return key, ("The person wants a reason they can trust, and your choice %s::%s has an empty brand field - no maker stands behind it. The table also holds candidates whose brand field is filled: %s. If one of them meets every requirement on its own page, choose it; otherwise keep yours and say in why a claim about it can be trusted." % (
                    key[0], key[1], ", ".join("%s::%s" % other for other in listed[:4])))
        if self.reason() and not self.named_keys() and not self.named_names():
            for must in self.figure_musts():
                if not self.other_figure(key, must):
                    continue
                clean = sorted([other for other, their in entries.items() if other != key and self.sound(their) and not self.other_figure(other, must)
                                and self.proof(other, their, must) in good and self.prefs_met(their) >= mine], key=lambda other: standing(other, must))
                if clean:
                    return key, ("Your choice %s::%s's page also writes another figure in %s than the person's %g %s, so what it states for this variant is in doubt, while %s::%s states %g %s with no other figure in that unit. Choose it, or keep yours and say in why your page states the person's value for this variant." % (
                        key[0], key[1], must.get("unit") or "that unit", number(must.get("value")), must.get("unit") or "", clean[0][0], clean[0][1], number(must.get("value")), must.get("unit") or ""))
                break
        steps = [item for item in self.record.get("priorities") or [] if isinstance(item, dict)]
        if steps and str(steps[0].get("direction")) in ("higher", "lower", "cheaper"):
            ranked = self.rank(entries, steps, everyone=bool(self.named_keys()))
            if ranked and ranked[0] != key and key in entries and self.comparable(ranked[0], key, entries, steps):
                unit = lambda other, step: self.rank_units.get((other, squash(step.get("text"))), "") if step.get("direction") != "cheaper" else "PHP"
                shown = lambda other: ", ".join("%s %s" % (str(step.get("text"))[:30], "not stated" if found is None else ("%g %s" % (found, unit(other, step))).strip())
                                                 for step in steps if str(step.get("direction")) in ("higher", "lower", "cheaper") for found in [self.rank_value(other, entries[other], step)])
                return key, ("By the person's ranking and the values you wrote from the pages, %s::%s comes before your choice - it has %s; yours has %s. The first step where they differ decides, and a later step only breaks a tie; every candidate on the table is in stock within their %s PHP. Choose it, or correct the value you wrote (in one unit for every candidate)." % (
                    ranked[0][0], ranked[0][1], shown(ranked[0]), shown(key), "%g" % self.budget() if self.budget() is not None else "stated"))
        return key, ""

    def rank_value(self, key: Tuple[str, str], entry: Dict[str, Any], step: Dict[str, Any]) -> Optional[float]:
        if step.get("direction") == "cheaper":
            return number(self.live.get(key, (None, None))[1])
        page = self.pages.get(key[0]) or {}
        page_text = plain(render_page(page))
        others = [item for item in self.record.get("priorities") or [] if isinstance(item, dict) and squash(item.get("text")) != squash(step.get("text"))]
        own = words(step.get("text")) - set().union(*[words(item.get("text")) for item in others]) if others else words(step.get("text"))
        line = next((line for line in entry.get("ranks") or [] if isinstance(line, dict) and (squash(line.get("step")) == squash(step.get("text")) or (words(line.get("step")) & own))), None)
        memo = (key, squash(step.get("text")))
        quote = plain((line or {}).get("quote") or "")
        if quote and re.search(r"\d\s*(?:-|–|~|to)\s*\d", quote.split("=", 1)[-1]):
            apart = re.sub(r"\d+(?:\.\d+)?\s*(?:-|–|~|to)\s*\d+(?:\.\d+)?", " ", quote)
            single = {found for found, _ in figures(apart)}
            if not (line and number(line.get("number")) is not None and float(number(line.get("number"))) in single):
                self.rank_memory[memo] = None
                return None
        if line and quote and within(quote, page_text) and number(line.get("number")) is not None:
            found = float(number(line.get("number")))
            written = figures(quote.split("=", 1)[-1])
            if written and found <= 0 < written[0][0]:
                found = written[0][0]
            options = next((str(v.get("options") or "") for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), "")
            body = plain(options + " " + "; ".join("%s = %s" % item for item in (page.get("facts") or {}).items()))
            cell = self.cell(key, {"text": step.get("text")}) or {}
            said, stated = figures(quote), figures(str(cell.get("options") or cell.get("facts") or "").split("=", 1)[-1])
            if within(quote, page.get("title")) and not within(quote, body):
                found = stated[0][0] if said and stated and unit_like(said[0][1], stated[0][1]) else None
            self.rank_memory[memo] = found
            self.rank_units[memo] = next((unit for figure, unit in written if figure == found), written[0][1] if written else "")
            if written and found is not None and found not in {figure for figure, _ in written}:
                self.rank_converted.add(memo)
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

    def comparable(self, one: Tuple[str, str], other: Tuple[str, str], entries: Dict[Tuple[str, str], Dict[str, Any]], steps: List[Dict[str, Any]]) -> bool:
        for step in steps:
            if str(step.get("direction")) not in ("higher", "lower", "cheaper"):
                break
            first, second = self.rank_value(one, entries[one], step), self.rank_value(other, entries[other], step)
            if first is None and second is None:
                return False
            if first is not None and second is not None and step.get("direction") != "cheaper":
                units = [self.rank_units.get((key, squash(step.get("text"))), "") for key in (one, other)]
                if not (bool(units[0]) == bool(units[1]) and unit_like(units[0], units[1])) and not any((key, squash(step.get("text"))) in self.rank_converted for key in (one, other)):
                    return False
            if first != second:
                return True
        return True

    def fallback_key(self, keys: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
        for key in self.backups:
            if key in keys:
                return key
        if not keys:
            return None
        recent = [key for key in keys if key[0] in self.opened_order[self.opened_before_add:]] if self.first_add else []
        order = recent + [key for key in keys if key not in recent]
        return max(order, key=lambda key: (sum(self.copies_prove(key, must) for must in self.figure_musts()), -order.index(key)))

    def spelled(self, name: str, key: Tuple[str, str]) -> bool:
        page = self.pages.get(key[0]) or {}
        variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), {})
        text = squash("%s %s %s %s" % (page.get("brand") or "", (page.get("facts") or {}).get("model") or "", page.get("title") or "", variant.get("options") or ""))
        return all(word in text for word in squash(name).split())

    def resolve_round(self, names: List[str], keys: List[Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
        lines = []
        for key in keys[:TABLE_ROWS * 2]:
            page = self.pages[key[0]]
            variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), {})
            lines.append("key=%s::%s | brand field: %s | model line: %s | options: %s | title: %s" % (key[0], key[1], page.get("brand") or "(empty)", (page.get("facts") or {}).get("model") or "(none)",
                                                                                                    variant.get("options") or "(none)", str(page.get("title") or "")[:160]))
        messages = [{"role": "system", "content": RESOLVE_GUIDE},
                    {"role": "user", "content": "The names: %s\n\nThe listings:\n%s" % ("; ".join(names), "\n".join(lines))}]
        answer = self.models.ask(RESOLVE, messages, RESOLVE_TOOL, deadline=HURRY_SECONDS)
        answers = [answer] if answer else []
        resolved: Dict[str, Tuple[str, str]] = {}
        for name in names:
            given = [self.key_of(item.get("key"), keys) for answer in answers for item in answer.get("candidates") or []
                     if isinstance(item, dict) and item.get("key") and squash(item.get("name")) == squash(name)]
            given = [key for key in given if key is not None]
            if not given:
                continue
            best = max(dict.fromkeys(given), key=lambda key: (given.count(key), -given.index(key)))
            if self.spelled(name, best):
                resolved[name] = best
        return resolved

    def spell_named(self) -> None:
        if not self.models_named() or self.named_keys() or self.named_names() or self.clock.elapsed() > HURRY_SECONDS:
            return
        keys = [key for key in self.candidates() if self.orderable(key)][:TABLE_ROWS * 2]
        if len(keys) < 2:
            return
        lines = []
        for key in keys:
            page = self.pages[key[0]]
            variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), {})
            lines.append("key=%s::%s | brand field: %s | model line: %s | options: %s | facts: %s" % (key[0], key[1], page.get("brand") or "(empty)", (page.get("facts") or {}).get("model") or "(none)",
                                                                                                  variant.get("options") or "(none)", ("; ".join("%s = %s" % item for item in (page.get("facts") or {}).items()) or "(none)")[:300]))
        colour = (self.colours_named() or [""])[0]
        answer = self.models.ask(RESOLVE, [{"role": "system", "content": SPELL_GUIDE}, {"role": "user", "content": "The model the person wrote: %s%s\n\nThe listings:\n%s" % (
            self.models_named()[0], "; the colour they wrote: %s" % colour if colour else "", "\n".join(lines))}], SPELL_TOOL, deadline=HURRY_SECONDS)
        if not answer:
            return
        found = lambda items: {key for key in (self.key_of(item, keys) for item in items or [] if isinstance(item, str)) if key is not None}
        line = lambda key: str((self.pages[key[0]].get("facts") or {}).get("model") or "")
        worn = lambda key: " ".join([next((str(v.get("options") or "") for v in self.pages[key[0]].get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), "")]
                                    + [str(value) for label, value in (self.pages[key[0]].get("facts") or {}).items() if tag(label) in ("colour", "color")])
        code = self.models_named()[0]
        self.spelled_keys = {key for key in found(answer.get("model_keys")) if spells(line(key), code)}
        coloured = {key for key in found(answer.get("colour_keys")) if spells(worn(key), colour)} if colour else set()
        if coloured and not self.spelled_keys & coloured and self.clock.elapsed() < HURRY_SECONDS:
            lines = ["key=%s::%s | model line: %s" % (key[0], key[1], (self.pages[key[0]].get("facts") or {}).get("model") or "(none)") for key in sorted(coloured)]
            again = self.models.ask(RESOLVE, [{"role": "system", "content": SPELL_GUIDE}, {"role": "user", "content": "The model the person wrote: %s\n\nThe listings in the colour they wrote, with their model lines alone:\n%s" % (
                self.models_named()[0], "\n".join(lines))}], SPELL_TOOL, deadline=HURRY_SECONDS)
            self.spelled_keys |= {key for key in found((again or {}).get("model_keys")) if spells(line(key), code)} & coloured
        self.spelled_colour = self.spelled_keys & coloured

    def resolve_names(self) -> None:
        names = self.named_names()
        keys = self.candidates()
        if not names or not keys:
            return
        resolved = self.resolve_round(names, keys)
        missing = [name for name in names if name not in resolved]
        if missing and self.clock.elapsed() < HURRY_SECONDS - 30 and self.session.turns_left() > 8 and SEARCH in self.schemas:
            done = {squash(query) for query, _ in self.search_rows}
            queries = []
            for name in missing[:3]:
                parts = squash(name).split()
                code = next((part for part in reversed(parts) if re.search(r"\d", part)), parts[-1])
                queries += [query for query in (code, " ".join(parts[-2:])) if len(query) >= 3 and squash(query) not in done and query not in queries]
            fresh = self.gather([self.constrained(SEARCH, {"query": query}) for query in queries[:4]])
            if fresh:
                self.open(fresh[:10])
                later = self.resolve_round(missing, [key for key in self.candidates() if key not in keys] + keys)
                resolved.update(later)
        found = list(dict.fromkeys(resolved.values()))
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
        if second_round and self.clock.elapsed() < HURRY_SECONDS and self.session.turns_left() > 8:
            done = {squash(query) for query, _ in self.search_rows}
            offered = [str(query) for query in list((answer or {}).get("searches") or []) + self.instead if str(query).strip() and squash(query) not in done]
            queries = list(dict.fromkeys(offered + [query for query in self.feedback_queries() if squash(query) not in done]))[:4] or [str(self.record.get("product") or "")]
            fresh = self.gather([self.constrained(SEARCH, {"query": query}) for query in queries])
            opened = self.open(self.shortlist(only=fresh + [product_id for product_id in self.carrying_rows() if product_id not in fresh], limit=REPLACEMENT_OPENS))
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
        if phrase and any(self.phrase_words() <= words((self.pages.get(key[0]) or {}).get("title")) for key in self.candidates()):
            return
        if self.changed_musts and not phrase:
            for must in self.changed_musts[:2]:
                unit = str(must.get("unit") or "").strip("- ")
                plain_figure = "%g %s" % (number(must.get("value")), unit) if number(must.get("value")) is not None and unit else ""
                for written in [text for text in (plain_figure, squash(must.get("text"))) if text]:
                    calls.append(self.constrained(SEARCH, {"query": " ".join(dict.fromkeys(("%s %s" % (product, written)).split()[:10]))}))
                joined = plain_figure.replace(" ", "")
                words_of = [word for word in product.split() if len(word) > 2]
                if joined and len(words_of) > 1 and not self.dropped_musts:
                    calls += [self.constrained(SEARCH, {"query": "%s %s" % (words_of[0], joined)}), self.constrained(SEARCH, {"query": "%s %s" % (words_of[-1], joined)})]
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
            if not phrase and page.get("brand") and self.figure_musts():
                must = self.figure_musts()[0]
                joined = "%g%s" % (number(must.get("value")), str(must.get("unit") or "").strip("- "))
                calls.append(self.constrained(SEARCH, {"query": " ".join(dict.fromkeys(("%s %s %s" % (squash(page["brand"]), product, joined)).split()[:10]))}))
            if not phrase and page.get("title"):
                title = [word for word in dict.fromkeys(piece.strip(".,;:!?()[]\"'") for piece in squash(page.get("title")).split()) if word][:16]
                variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and str(v.get("sku")) == key[1]), {})
                for must in self.figure_musts()[:2]:
                    written = next((piece for piece in ("%s %s" % (variant.get("options") or "", page.get("title") or "")).replace("/", " ").split()
                                    if re.search(r"\d", piece) and states(piece, float(number(must.get("value")) or 0), squash(must.get("unit")), "exact") and re.search(r"[a-zA-Z]", piece)), None)
                    if written and squash(written) not in title:
                        title.append(squash(written).strip(".,;:()[]"))
                    elif not written and number(must.get("value")) is not None and must.get("unit"):
                        title += [word for word in ("%g %s" % (number(must.get("value")), str(must.get("unit")).strip("- "))).split() if word not in title]
                calls.append(self.constrained(SEARCH, {"query": " ".join(title)}))
        done = {squash(call["args"].get("query")) for call in calls if call["name"] == SEARCH} | {squash(query) for query, _ in self.search_rows}
        if not self.dropped_musts:
            calls += [self.constrained(SEARCH, {"query": query}) for query in self.feedback_queries() if squash(query) not in done]
        if self.budget_dropped and (shelf or product) and not any(call["name"] == FILTER and call["args"].get("category") == (shelf or product) for call in calls):
            calls.append(self.constrained(FILTER, {"category": shelf or product}))
            for maker in self.makers()[:MAKER_FILTERS]:
                calls.append(self.constrained(FILTER, {"category": shelf or product, "brand": maker}))
        backups = [key for key in self.backups if key not in self.cart][:3]
        if INSPECT_STOCK in self.schemas:
            calls += [{"name": INSPECT_STOCK, "args": {"product_id": key[0], "sku": key[1]}} for key in backups]
        if not calls:
            return
        maker_queries = {squash(call["args"].get("query")) for call in calls if call["name"] == SEARCH and self.makers() and squash(call["args"].get("query")).startswith(squash(self.makers()[0]))}
        fresh = self.gather(calls)
        if maker_queries:
            maker_fresh = [product_id for query, found in self.search_rows if squash(query) in maker_queries for product_id in found[:MAKER_ROWS] if product_id in fresh]
            self.maker_rows += [product_id for product_id in maker_fresh if product_id not in self.maker_rows]
        else:
            maker_fresh = []
        lost_titles = {squash((self.pages.get(key[0]) or {}).get("title")) for key in self.lost}
        lead = [product_id for product_id in fresh if squash(self.rows[product_id].get("title")) in lost_titles] + maker_fresh
        now = [must for must in self.changed_musts if number(must.get("value")) is not None] if self.dropped_musts and self.makers() else []
        if now:
            reserve = [product_id for product_id in self.maker_rows if product_id in self.rows and product_id not in self.pages
                       and all(states(self.rows[product_id].get("title"), float(number(must.get("value"))), squash(must.get("unit")), str(must.get("relation") or "exact")) for must in now)][:2]
            lead = reserve + [product_id for product_id in lead if product_id not in reserve]
            fresh = fresh + [product_id for product_id in reserve if product_id not in fresh]
        opened = self.open(self.shortlist(only=fresh + [product_id for product_id in self.carrying_rows() if product_id not in fresh], lead=lead, limit=REPLACEMENT_OPENS + (2 if self.changed_musts or self.budget_dropped else 0), note=self.reads_note()))
        self.extract(opened)

    def claims_for(self, key: Tuple[str, str]) -> List[Dict[str, Any]]:
        page = self.pages.get(key[0]) or {}
        labels = {tag(label) for label in (page.get("facts") or {})}
        in_stock, price = self.live.get(key, (None, None))
        tied = {}
        for must in self.musts():
            cell = self.cell(key, must)
            if cell and cell.get("facts"):
                tied[squash(must.get("text"))] = tag(str(cell.get("facts")).split("=", 1)[0])
        figured = lambda claim, must: number(must.get("value")) is not None and tag(claim["field"]) in labels and states(
            str(claim["value"]).split("=", 1)[-1], float(number(must.get("value")) or 0), squash(must.get("unit")), str(must.get("relation") or "exact"))
        needed = lambda claim: squash(claim["field"]) in ("price", "in_stock", "brand") or tag(claim["field"]) in tied.values() or any(figured(claim, must) for must in self.musts())

        def trimmed(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            core = [claim for claim in claims if needed(claim)]
            fields = {squash(claim["field"]) for claim in core}
            coming = int(price is not None and "price" not in fields) + int(in_stock is True and "in_stock" not in fields)
            spare = [claim for claim in claims if claim not in core and tag(claim["field"]) in labels]
            return core + spare[:max(0, 4 - len(core) - coming)]

        def covered(claims: List[Dict[str, Any]]) -> bool:
            fields = {squash(claim["field"]) for claim in claims}
            if len(claims) + int(price is not None and "price" not in fields) + int(in_stock is True and "in_stock" not in fields) < 4 or (page.get("brand") and "brand" not in fields):
                return False
            return {"price", "in_stock"} <= fields and all(
                any((tied.get(squash(must.get("text"))) == tag(claim["field"])) or figured(claim, must) for claim in claims)
                for must in self.musts() if squash(must.get("text")) in tied or number(must.get("value")) is not None)
        kept = trimmed(self.shown_claims(key, self.claims.get(key) or []))
        if covered(kept) or self.clock.elapsed() > TIME_LIMIT_SECONDS - 25:
            return kept
        messages = [{"role": "system", "content": CLAIMS_GUIDE},
                    {"role": "user", "content": "The request: %s\nThe firm requirements to cite: %s\n\nThe ordered variant: sku=%s, latest price %s, in_stock %s\n%s" % (
                        self.request, "; ".join(str(must.get("text")) for must in self.musts()) or "(none)", key[1], price, dumps(in_stock), render_page(page))}]
        answer = self.models.ask(CLAIMS, messages, CLAIMS_TOOL, deadline=TIME_LIMIT_SECONDS - 8)
        written = self.shown_claims(key, list((answer or {}).get("claims") or []))
        merged = written + [claim for claim in kept if squash(claim["field"]) not in {squash(c["field"]) for c in written}]
        if not covered(merged) and self.clock.elapsed() < TIME_LIMIT_SECONDS - 25:
            more = self.shown_claims(key, list((self.models.second_opinion(CLAIMS, messages, CLAIMS_TOOL) or {}).get("claims") or []))
            merged = more + [claim for claim in merged if squash(claim["field"]) not in {squash(c["field"]) for c in more}]
        return trimmed(merged) or merged[:4]

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
                fact = next((str(fact) for label, fact in facts.items() if tag(label) == tag(field) and within(value, fact)), None)
                if fact is not None:
                    kept.append({"field": str(next(label for label in facts if tag(label) == tag(field))), "value": fact})
        return kept

    def order(self, key: Tuple[str, str]) -> None:
        args: Dict[str, Any] = {"product_id": key[0], "sku": key[1]}
        if self.claims_required:
            claims = self.claims_for(key)
            in_stock, price = self.live.get(key, (None, None))
            if price is not None and not any(squash(claim.get("field")) == "price" for claim in claims):
                claims.append({"field": "price", "value": str(price)})
            if in_stock is True and not any(squash(claim.get("field")) == "in_stock" for claim in claims):
                claims.append({"field": "in_stock", "value": "true"})
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
        self.open(self.shortlist(note=self.phrase_note()))
        self.track_budget(self.hear)
        if self.named_names() and not self.named_keys():
            self.resolve_names()
        pages = [product_id for product_id in self.opened_order if any(key[0] == product_id for key in self.candidates())] if self.named_keys() else self.opened_order
        self.extract(pages)
        if self.budget_dropped and not self.candidates():
            self.replacement_reads()
        self.spell_named()
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

    def pick_rows(self, keys: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if self.clock.elapsed() > TIME_LIMIT_SECONDS - 30 or self.models.dollars > EPISODE_DOLLARS_LAST:
            return []
        rows = [self.rows[key[0]] for key in keys if key[0] in self.rows][:30]
        if not rows:
            return []
        messages = [{"role": "system", "content": SHORTLIST_GUIDE},
                    {"role": "user", "content": "Order conditions: %s\n\nNothing opened so far can be bought. Pick the rows likeliest to be the item asked for, likeliest first; one of them will be bought.\n\nThe rows:\n%s" % (
                        dumps(self.record), render_rows(rows, self.budget(), {key.split("::")[0] for key in self.named_keys()}, self.phrase_words(), self.found_by))}]
        answer = self.models.ask(SHORTLIST, messages, SHORTLIST_TOOL, deadline=TIME_LIMIT_SECONDS - 12)
        ids = [str(item.get("product_id")) for item in (answer or {}).get("open") or [] if isinstance(item, dict)]
        return [key for product_id in ids for key in keys if key[0] == product_id]

    def last_resort(self) -> None:
        if self.session.done or self.session.dead or SEARCH not in self.schemas or self.session.turns_left() < 4:
            return
        if any(self.orderable(key) for key in self.cart) or any(in_stock is True and not over(price, self.budget()) for in_stock, price in self.live.values()):
            return
        words_of = self.request.split()
        calls = [self.constrained(SEARCH, {"query": " ".join(words_of[:8])}), self.constrained(SEARCH, {"query": str(self.record.get("product") or " ".join(words_of[3:9]))})]
        if self.record.get("category"):
            calls.append(self.constrained(FILTER, {"category": str(self.record["category"])}))
        self.send(calls)

    def close(self) -> None:
        if self.session.done or self.session.dead:
            return
        if not self.rows and not self.pages and SEARCH in self.schemas and self.session.turns_left() >= 3:
            self.send([self.constrained(SEARCH, {"query": str(self.record.get("product") or self.request)[:120]})])
        self.catch_up()
        self.last_resort()
        target = next((key for key in reversed(self.cart) if self.orderable(key)), None)
        tried: set = set()
        picked_rows: List[Tuple[str, str]] = []
        while target is None and not self.session.done and self.session.turns_left() >= 2:
            keys = [key for key in self.candidates() if key not in tried and key not in self.cart]
            if not keys:
                keys = [key for key in self.live if key not in tried and key not in self.cart and self.live[key][0] is True and not over(self.live[key][1], self.budget())
                        and (not self.named_keys() or "%s::%s" % key in self.named_keys())]
            if not keys and self.named_keys() and not self.list_open:
                self.list_open = True
                continue
            keys.sort(key=lambda key: (not any(source.startswith("filter") for source in self.found_by.get(key[0], [])), -len(self.found_by.get(key[0], []))))
            if keys and not picked_rows:
                picked_rows = self.pick_rows(keys)
            keys = [key for key in picked_rows if key in keys] + [key for key in keys if key not in picked_rows]
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
