"""Evidence Agent v3.113: a model-led shopping agent with a thin controller.

The model reads the shopper's request and every later line, searches the store, reads the listings the controller
fetches for it, and names the SKU to order with its claims about that SKU's page. The controller has the model read
the request into a plan (the kind of product, each required attribute with its value, price limit, ranking,
preferences, search queries, and two wishes the shopper doubts can be met together), asks the shopper one question
(which of two wishes matters more when the plan names two the shopper doubts or lists two it would only like - two
uses of the purchase that the message names are no such pair; else
what the use of the purchase that the first message names requires of one attribute, in a question the model writes
that names that use, or what the purchase is for when the message names no use; the question about the use when the
answer about the attribute only restated the terms of the request, the other of the two questions when the answer
about the wishes named nothing, and the question about the two wishes once more in other words when both answers only
restated the terms of the request; once the shelf was read, one attribute once more when every answer so far restated
the terms, about another use the message names when it names several, and a second one when that answer restated the
terms too), has the
plan read again after each
shopper line (each line of a shopper message is a line of its own), confirms in one statement a requirement the shopper
changed or added of their own accord (not in the store turn after the store reported a change with a cart addition: the
shopper remarks on that change then and reads no message, so the controller reads the cart in that turn and speaks in the
next), runs the plan's searches in one store turn, filters the shelf the plan names (also
by brand and by price), fetches the pages of listed results (the model is shown those filed under the shelf the shopper
named, when the store has that shelf, and can look values up among the pages read; the titles of the listings whose
pages carry the most of the plan's values are searched with those values, since a store's search may show one variant
of each listing), and carries out the purchase: it first adds another SKU whose page bears out what the model claims about its choice from
the shopper's first message (one whose page also writes what the shopper added later before one that does not, a
variant of the chosen listing when the store has one) and reads it again, because a
store can change the stock or the price of the SKU that enters the cart first (the addition and that read are one
store turn, and the first SKU leaves the cart in the turn that brings the choice in), and a shopper may lower the
limit or change a requirement once something is in the cart, which takes turns to answer: when nothing has entered
the cart yet and few store turns remain, the controller tells the model to commit its best SKU in view now; then it adds the model's choice, reads
it again and places the order with the model's claims about what the shopper asked for of their own accord, the
price and the stock. When the choice has become unavailable the controller asks the model for another. Before any
cart addition or order the controller checks only what the store itself reported: that the SKU's page was read, that
it is filed under the shelf the shopper named, that its latest stock reading is positive, that its latest price does
not exceed the limit the model read from the shopper, and that the page shows what the model claims. When a claim
shows a value of the page that does not write the requirement in the plan's words, the controller says so once; the
same commit made again stands, because whether another word or another unit states the requirement is the model's
reading. A requirement of the plan that only restates a use the first message names, and that no field of the
chosen SKU's page writes, is demanded no claim: what the purchase is for is no attribute of the SKU. A reading of
the plan that brings no plan is asked once more; when none comes of that either, the plan in force is stale and a
purchase the shopper interrupted is not resumed on it: the model decides anew. A question about an attribute that
names one asked about before is read once more, with that attribute named. A requirement
of the plan is stated plainly by a page when one of its facts is that value, or when the SKU's option
text carries it, and no fact under that attribute's label reads otherwise: when the choice's page writes it only in its
title or inside a longer text while other SKUs in view state it plainly, the controller says so once and the model
decides again. A claim that rests a requirement of words on the title alone does not stand when a fact of the page
whose label names that attribute states another value: the page's own fact says what the attribute is. When the
pages of the shelf state facts whose
whole value the shopper's answer names and the plan requires nowhere, the model reads the answer once more with them;
a rewording that loses a word of a required value which the shopper said and the shelf's facts write as a value does
not replace the plan.
A claim that rests a requirement of words on a fact that lists several values,
the required value one of them, is sent back once: a page whose fact is that value alone states it more plainly; the
same commit made again stands.
Lines of the shopper's own that arrived in several store turns since the plan was last read are read turn by turn:
the later ones are kept out of the model's view while the earlier ones are read and confirmed, then put back and
read in their turn.
A requirement that a line of the shopper named stays in force until the shopper changes it: when a later reading of
the plan drops one that the lines read since do not speak of, the controller puts it back; the wish the shopper chose
of two doubted ones stays a requirement, worded as the request words it. A value the shopper replaced stays replaced
when a later line that lowers the price limit repeats the earlier value, under whatever wording of its attribute the
new reading uses. An entry of a reading that only names the kind of product again (also as the shelf), or that words
the ranking as a requirement without a number, is no requirement of its own.
A request that names two uses of the purchase: after the shopper's answer about the use and the first look at the
shelf, the model lists the uses the first message names and, when it names two, writes one open question about the one
attribute the other use depends on; the controller asks it once, before any cart addition. Every question about an
attribute is one question that names the use and one attribute: one the model wrote otherwise (several attributes, a
list of examples) is asked in the plain form "For <use>, which <attribute> should it have?".

Single-file ORO submission with one strategy for every request, driven by the public task input, the tool
observations and the shopper's messages.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from os import getenv
from time import sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# The two models of each inference provider: the first decides; the second takes over a decision the first failed on.
MODELS = {"chutes": ("Qwen/Qwen3.5-397B-A17B-TEE", "moonshotai/Kimi-K2.6-TEE"),
          "openrouter": ("qwen/qwen3.5-397b-a17b", "moonshotai/kimi-k2.6")}
MODEL_SETTINGS = {
    "Qwen/Qwen3.5-397B-A17B-TEE": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                   "chat_template_kwargs": {"enable_thinking": False}},
    "moonshotai/Kimi-K2.6-TEE": {"temperature": 0.6, "top_p": 0.95, "chat_template_kwargs": {"thinking": False}},
    "qwen/qwen3.5-397b-a17b": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "reasoning": {"enabled": False}},
    "moonshotai/kimi-k2.6": {"temperature": 0.6, "top_p": 0.95, "reasoning": {"enabled": False}},
}
OUTPUT_TOKENS = 3000
INFERENCE_TIMEOUT = 60.0
STORE_TIMEOUT = 30.0
RATE_LIMIT_WAITS = (2.0, 4.0, 8.0, 16.0)  # a 429 waits and is tried again; a run starts every task at once

DECISION_LIMIT = 26
STALLS_TOLERATED = 2  # decisions without a usable tool call before the episode ends
PURCHASE_TURNS = 4  # the alternative with its re-read; its removal when the shopper spoke; the choice with its re-read; the order
CLOSING_TURNS = 3  # store turns, beyond the purchase, in which the model is told to commit and new listings are not fetched
FIRST_ADD_TURNS_LEFT = 14  # store turns left at which the model is told to make its first commit, when nothing is in the cart yet
BLOCK_AFTER_REFUSALS = 3  # commits of one SKU refused for its claims before the SKU is no longer accepted
VARIANT_TURNS = 2  # a store turn that lists the other variants of a listing and one that reads their pages
VARIANT_QUERY_WORDS = 12  # words of a listing's title that the search for its variants carries
SHELF_WORDS = 10  # the most frequent title words of the shelf, each searched with the values the plan asks for
LISTING_TITLES = 6  # listings whose title is searched with the plan's values
KIND_WORDS = 3  # words of the kind of product that are each searched with the plan's values
BRAND_TRIES = 4  # values of the plan, and then brands seen on the shelf, that the shelf is filtered by as a brand
CLOSING_DECISIONS = 4  # decisions left at which the controller orders the model's choice with the claims its page shows
CHOICE_TURNS = 2  # the choice with its re-read, the order
PLAIN_POINTERS = 2  # commits of one episode that are answered with the SKUs that state an added requirement plainly
PLAIN_SHOWN = 5  # SKUs such an answer lists
USE_QUESTION_CHARS = 200  # a question about what a use requires, at most
USE_QUESTIONS_AFTER_SHELF = 2  # attribute questions after the shelf was read, when every answer so far restated the terms
STORE_SPELLINGS = 2  # other forms of a unit, as the pages read write it, that a value is searched in
READ_PAGES_PER_CALL = 5
AUTO_READ_PAGES = 60  # pages the controller fetches after one decision
AUTO_READ_TOTAL = 240  # pages the controller fetches in one episode
SWEEP_SHARES = (0.9, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)  # price caps of the shelf sweep, as shares of the ceiling
RESULT_ROWS = 10
FACT_LABELS = 18
VALUE_CHARS = 48
TITLE_CHARS = 110
VARIANTS_SHOWN = 14

OPENING_QUESTION = "What is this purchase going to be used for?"
USE_TEMPLATE = "For {use}, which {attribute} should it have?"  # when the model's question does not name the use
USE_TEMPLATE_NONE = "Which {attribute} should it have?"  # the same, for a request that names no use
OWN_FIELDS = ("title", "options", "price", "in_stock", "category")

SYSTEM = (
    "You shop in an online store for one shopper, through the store's tools, and order one SKU that meets every "
    "requirement the shopper stated: in the first message and in every later line; a later line overrides an earlier "
    "one (a lower budget, a changed value). A stated price limit is a ceiling. When the shopper ranks by a quantity "
    "(wants the most or the least of something), order, among the SKUs that "
    "meet every requirement, the one with the best value of that quantity as its page writes it, so read the whole "
    "shelf before choosing; when no page in view states that quantity, order a SKU that meets every requirement all "
    "the same, and order before the decisions run out. A wish the shopper only prefers and does not require "
    "is not a requirement: never let it exclude a SKU that meets the requirements; among SKUs "
    "equal on everything else, prefer the one that meets the wish. When the shopper names two wishes and is unsure "
    "that the two go together, your plan names the two and the controller asks which matters more; the answer makes "
    "that wish a requirement and the other wish a preference. The controller has you "
    "read the request into a plan, and again after each later shopper line: the latest plan is the one in force. "
    "Work in this order. Find the shelf the shopper names: filter by category, and try the plural or the singular "
    "spelling when a filter finds nothing. Search with the shopper's own words and values, then with the words a "
    "listing might use instead: a synonym of the kind of product, another form of an attribute word, a value with its "
    "unit written differently. Read the pages the controller fetches for you: facts, option texts, prices and stock; "
    "when the store has the shelf the shopper named, you are shown the pages filed under it whose SKU can be ordered, "
    "and the others are counted. find_pages lists, without a store turn, the SKUs among the pages read whose page "
    "carries the values you give, those whose option text or facts carry them before those where only the title "
    "does, each with the facts of its page that name the quantity your plan ranks by: before your first commit, call it with the values your plan requires and judge the SKUs it lists; call it "
    "again when more pages were read. "
    "Judge a SKU by its own page: a fact whose whole value is the requested value, or an option text that writes the "
    "requested word as a word of its own, is firm evidence; a fact that lists several values, or a word inside a "
    "code, is weaker, so prefer the firmer SKU. A title describes a whole listing and may name several colours, "
    "sizes or models at once: what tells one SKU apart is its own option text and the facts of its page, so rest a "
    "requirement on those where the page has them, and on the title only for what holds for the whole listing. When "
    "pages state an attribute under different labels, prefer the SKU whose fact label names that attribute. "
    "Order only a SKU whose page is filed under the shelf the shopper named (the category shown on the page); a "
    "listing filed elsewhere is not that kind of product, whatever its title says, and a SKU filed under that shelf "
    "is that kind of product whatever else its title names (a part, an accessory, a set): do not rule it out for its "
    "title. When the store has a shelf whose "
    "label is the shopper's kind of product itself, that shelf is the one meant, not a shelf whose label adds "
    "another word to it. Each decision either calls store "
    "tools, several at once, or calls commit; a call made before is not run again, so vary the query or read the "
    "listed pages instead. As soon as pages show a SKU on the requested shelf whose own page states every "
    "requirement, call commit at once. In commit, claims must cover every requirement of your plan with the field of "
    "the page that states it and the value as the page writes it, each naming the requirement it meets as your plan "
    "words it (the kind of product by the field category, stock by in_stock, each attribute by its fact label or by "
    "options or title); the controller refuses a claim the page does not bear out and a SKU that leaves a requirement "
    "of your plan without a claim. Before your choice goes into the cart the controller adds another SKU on the same "
    "shelf whose page bears out the claims that the first message states, a variant of your choice's listing when the "
    "store has one, so that a change the store makes to the first SKU in the cart does not meet your choice. If the "
    "controller reports that your choice is no longer available, call commit again with another SKU that meets every "
    "requirement. The controller asks the shopper its one question (which of two doubted wishes matters more, else "
    "what the purchase is for) and confirms a requirement the shopper changes: you send no messages. A shopper question about the listing in the cart (whether it "
    "can still be had, whether its price rose) needs no reply: go on with the purchase. When a shopper line with "
    "something new arrives while your purchase is under way (a lower budget, a changed requirement), the controller "
    "stops and shows it to you: read it and call commit again with a SKU that meets the request as it now stands."
)
COMMIT_TOOL = {"type": "function", "function": {
    "name": "commit",
    "description": (
        "Order this SKU. The controller first puts another SKU into the cart and reads it again, because a store "
        "can change the stock or the price of the SKU that enters the cart first; then it adds your choice, reads it "
        "again and places the order. That other SKU is one the controller picks among the read pages: on the same "
        "shelf, orderable within the price limit, and bearing out the claims of your choice that the first message states. max_price "
        "is the price ceiling in force: the lowest limit the shopper has stated so far (a later line overrides an "
        "earlier one), as a number, or null when none was stated; the controller refuses a SKU priced above it. "
        "claims are field and value pairs about the ordered SKU, one per requirement of your plan, each true and "
        "shown on its page: a fact label as the page shows it, or options, title, category or in_stock; each names "
        "the requirement it meets, worded as in your plan. The price limit needs no claim: max_price covers it."),
    "parameters": {"type": "object", "additionalProperties": False,
                   "required": ["product_id", "sku", "max_price", "claims"],
                   "properties": {
                       "product_id": {"type": "string"}, "sku": {"type": "string"},
                       "max_price": {"type": ["number", "null"]},
                       "claims": {"type": "array", "items": {
                           "type": "object", "additionalProperties": False, "required": ["field", "value", "requirement"],
                           "properties": {"field": {"type": "string"},
                                          "value": {"type": ["string", "number", "boolean"]},
                                          "requirement": {"type": "string"}}}}}}}}
FIND_TOOL = {"type": "function", "function": {
    "name": "find_pages",
    "description": (
        "List the SKUs among the pages already read whose page carries every one of the given values. A value is a "
        "word or a short phrase as a listing may write it (a colour, a material, a number with its unit); it counts "
        "when a fact, the option text or the title of the page writes it. Only SKUs that can be ordered are listed "
        "(stocked, priced within the limit of your plan, on the shelf the shopper named when the store has it), each with "
        "the fields that carry each value; the SKUs whose option text or facts carry the values come first, those for "
        "which a value stands only inside a longer fact text last. No store turn is used."),
    "parameters": {"type": "object", "additionalProperties": False, "required": ["values"],
                   "properties": {"values": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6}}}}}
FOUND_SHOWN = 20  # SKUs one find_pages call lists
PLAN_TOOL = {"type": "function", "function": {
    "name": "plan",
    "description": (
        "Your reading of the shopper's request and lines so far; a later line overrides an earlier one, and a value the "
        "shopper has replaced by another stays replaced: a later line about something else (a new price limit, say) that "
        "still speaks of the earlier value does not bring it back; only a line that changes that attribute again does. "
        "kind: the kind "
        "of product alone, in the shopper's words. requirements: each attribute the ordered SKU must have, one entry "
        "per attribute: the attribute named as a store labels it (as in colour, material, width) and its value in the "
        "shopper's own words (as in navy, bamboo, 40 cm); a stated amount or size is a requirement, not a ranking; "
        "when the shopper asks for it, stock is one too (attribute stock, value in stock). No price limit is "
        "listed there, and what the purchase is for is no attribute of the SKU. price_limit: the ceiling in force as a number, or null. ranking: the quantity the shopper wants "
        "the most or the least of among the SKUs that meet the requirements, with 'max' or 'min', or null. "
        "preferences: wishes the shopper would like met but does not require, in the same form; of two wishes the "
        "shopper was unsure about, the one the shopper then said matters less belongs here. "
        "doubted_pair: the two wishes the shopper is unsure go together and has not ranked yet, "
        "exactly two short phrases, or null when there is no such doubt; doubt_quote: the shopper's words that express "
        "that doubt, or null. queries: three to eight short store searches that would list fitting products, each with "
        "the kind of product and one or two of the requested values in the shopper's own words; the controller runs "
        "them all at once."),
    "parameters": {"type": "object", "additionalProperties": False,
                   "required": ["kind", "requirements", "price_limit", "ranking", "preferences", "doubt_quote", "doubted_pair", "queries"],
                   "properties": {
                       "kind": {"type": "string"},
                       "requirements": {"type": "array", "items": {
                           "type": "object", "additionalProperties": False, "required": ["attribute", "value"],
                           "properties": {"attribute": {"type": "string"}, "value": {"type": "string"}}}},
                       "doubt_quote": {"type": ["string", "null"]},
                       "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
                       "price_limit": {"type": ["number", "null"]},
                       "ranking": {"type": ["object", "null"], "additionalProperties": False, "required": ["quantity", "direction"],
                                   "properties": {"quantity": {"type": "string"}, "direction": {"type": "string", "enum": ["max", "min"]}}},
                       "preferences": {"type": "array", "items": {
                           "type": "object", "additionalProperties": False, "required": ["attribute", "value"],
                           "properties": {"attribute": {"type": "string"}, "value": {"type": "string"}}}},
                       "doubted_pair": {"type": ["array", "null"], "items": {"type": "string"}, "minItems": 2, "maxItems": 2}}}}}
PRIORITY_QUESTION = "Which matters more to you: {a} or {b}?"
PRIORITY_AGAIN = "Of the two wishes you named, {a} and {b}, which one matters more to you?"
PAIR_AGAIN = (" Your reading names the doubted pair {pair}, and these are not two wishes about the item that your plan "
              "lists. When the shopper is unsure that one item, or the price limit, allows for two of their wishes at "
              "once, doubted_pair names those two wishes, each by the value your plan gives it, and both stand among "
              "the preferences until the shopper says which matters more; when the shopper voices no such doubt about "
              "two wishes for the item, doubted_pair is null: what the purchase is for is no wish about the item.")
PAIR_QUOTE = (" Your reading quotes the doubt as {quote}: the pair is the two wishes those words speak of, the ones the "
              "shopper named together with them, not a requirement stated elsewhere in the request.")
UNIT_AGAIN = (" Your plan words a requirement as a number without a unit ({entries}). Read the lines once more: a number "
              "of items the shopper needs is a count of pieces, worded as that number of pcs; any other quantity "
              "carries the unit the shopper gave it.")
CONFIRMATION = "Confirmed: {changes}. The rest of your request stands as it was."
USES_TOOL = {"type": "function", "function": {
    "name": "uses_named",
    "description": "List the uses of the purchase that the shopper's first message names.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["uses", "answered_use"],
                   "properties": {
                       "uses": {"type": "array", "items": {"type": "string"},
                                "description": "Each use of the purchase that the first message names, in the message's own words."},
                       "answered_use": {"type": ["string", "null"],
                                        "description": "The use that the shopper's answer speaks of, in the message's own words."}}}}}
USES_NAMED = ("Controller: read the shopper's first message once more and call uses_named. List the uses of the purchase it "
              "names (the things the item is for), each in the message's own words, and name the one that the shopper's answer "
              "speaks of. A message that tells one thing the item is for names one use.")
USES_FIRST_TOOL = {"type": "function", "function": {
    "name": "uses_named",
    "description": "List the uses of the purchase that the shopper's first message names.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["uses"],
                   "properties": {
                       "uses": {"type": "array", "items": {"type": "string"},
                                "description": "Each use of the purchase that the first message names, in the message's own words."}}}}}
USES_FIRST = ("Controller: read the shopper's first message once more and call uses_named. List the uses of the purchase it "
              "names, each in the message's own words: whatever the shopper says the item is for or has to suit (a purpose, a "
              "recipient, an occasion, a setting, or something the item has to match, correspond to or stand up to), wherever "
              "and however the sentence puts it, also as a phrase with 'for' that opens the message or follows what the "
              "shopper needs. A property of the item itself (its colour, size, count, price, stock, the kind of product) is no "
              "use. A message that tells one such purpose names one use; one that tells none lists none.")
USE_QUESTION_TOOL = {"type": "function", "function": {
    "name": "question",
    "description": "Write one short question to the shopper about one attribute.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["attribute", "question"],
                   "properties": {"attribute": {"type": "string"}, "question": {"type": "string"}}}}}
USE_QUESTION = ("Controller: the shopper's first message names the use of the purchase {use}. A shopper who has told the use "
                "answers a question about what that use requires of one attribute of the product, and says nothing new to a "
                "question about the use itself. Call question with one short question that names this use in the message's "
                "own words and asks which value one attribute of the product should have (the attribute goes into the field "
                "attribute): the attribute of this kind of product "
                "that the use most plainly depends on (a property a listing states as a fact or in its option text: a material, a "
                "colour, a type, a size, a count, ...), not one the plan already holds or ranks by ({held}){asked}. Name the "
                "attribute and no value: no yes-or-no question, no example values, one attribute only, one question mark. "
                "Call the product \"it\", not by the name of its kind.")
USE_SHELF = (" The SKUs read are in view: take the attribute in which they differ (a fact label of their pages, or what their "
             "option texts vary in) that this use most plainly depends on.")
USE_ATTRIBUTE_AGAIN = (" Your question asks about {attribute}, which the shopper was asked about already and answered with the "
                       "terms of the request alone: name another attribute of this kind of product, one the use may depend on.")
USE_QUESTION_NONE = ("Controller: the shopper's first message names no use of the purchase, and the answer to the question about "
                     "the use only restated the terms of the request. Such a shopper may still answer a question about one "
                     "attribute of the product. Call question with one short question that asks which value one attribute of "
                     "the product should have (the attribute goes into the field attribute): a property a listing states as a "
                     "fact or in its option text (a material, a colour, a type, a size, a count, ...), not one the plan already "
                     "holds or ranks by ({held}){asked}. Name the attribute and no value: no yes-or-no question, no example "
                     "values, one attribute only, one question mark. Call the product \"it\", not by the name of its kind.")
SECOND_USE = ("Controller: the first message names a second use of the purchase, {use}, of which the shopper's answer said "
              "nothing; it may carry a requirement of its own that the shopper has not told. Call question with one short, "
              "open question to the shopper that names this use and asks which value one attribute of the product should "
              "have (the attribute goes into the field attribute). The attribute must not be one your plan already holds or ranks by ({held}). Look at the SKUs read "
              "that meet everything known so far: take the attribute in which they differ (a fact label of their pages, or "
              "what their option texts vary in) that this use most plainly depends on; when the use gives no lead, the one in "
              "which most of them differ. Name the attribute and no value: no yes-or-no question, no example values, and one "
              "attribute only. Call the product \"it\", not by the name of its kind.")
QUANTITY_AGAIN = (" The request writes {found}, and the plan you read carries it nowhere. Read the request once more: a "
                  "quantity it states about the item is a requirement of the plan, or a preference when the shopper only "
                  "wishes for it.")
KIND_AGAIN = (" Your first reading named the kind of product {first}; this reading names {named}. The shopper's first "
              "message names the kind of product and a later line does not change it: keep {first} as the kind, and read "
              "the later line once more for what it requires of that product (a type, a value, a count of pieces).")
ANSWER_FACTS = (" The shopper answered: {answer}. Pages read on this shelf state facts whose value that answer names: "
                "{facts}. Read the answer once more: when it says the item has to be or have one of them, the plan "
                "requires it, with the attribute as that label names it and the value as the pages write it; when the "
                "answer only mentions it (what the purchase is for, whom it is for), keep the plan.")
FACT_WORDS = 4  # words of a fact value, at most, for the answer to name it whole
FACTS_SHOWN = 6
ANSWER_AGAIN = (" The shopper has answered your question about the use, and the plan you read after it is the plan as it "
                "was before. Read the answer once more: what it says the item has to have or be (a value, a number with "
                "its unit, a count of pieces) is a requirement of the plan; a number of items the shopper says they need "
                "is a count of pieces the item has to have, worded as that number of pcs; when it says nothing of that "
                "kind, keep the plan.")
MODEL_STORE_TOOLS = ("search", "filter", "view", "compare", "inspect_stock", "inspect_cart")
OTHER_WORDING = "this value does not write the requirement"  # why a claim that the page shows is sent back
LIST_WORDING = "this fact lists several values and names the requirement among them"  # a claim on a list fact, sent back once


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _models() -> tuple[str, str]:
    return MODELS.get(getenv("INFERENCE_PROVIDER", "chutes"), MODELS["chutes"])


def _cut(text: Any, limit: int) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _words(text: Any) -> set[str]:
    # The letter-and-digit words of a text, lower-cased, a final s dropped from long words so that a plural and its
    # singular read the same.
    out = set()
    for raw in "".join(ch if ch.isalnum() else " " for ch in str(text or "").casefold()).split():
        out.add(raw[:-1] if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss") else raw)
    return out


def _tokens(text: Any) -> set[str]:
    # The words of an option text or a title as its own parts: split at spaces and punctuation, but a hyphenated
    # compound or a code ("ns-168-red", "sk007-green") and a decimal stay one word, so a word inside a code does not
    # count as the word itself; a final s is dropped from long words.
    out = set()
    for raw in "".join(ch if ch.isalnum() or ch in "-." else " " for ch in str(text or "").casefold()).split():
        raw = raw.strip("-.")
        if raw:
            out.add(raw[:-1] if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss") else raw)
    return out


def _measures(text: Any) -> list[tuple[float, str]]:
    # The numbers of a text with the letters right after each (its unit, lower-cased, possibly empty); an inch mark
    # right after a number is the unit inch.
    out = []
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?:([\"”″])|\s?([a-zA-Z]{0,8}))(?![\w])", str(text or "")):
        try:
            out.append((float(m.group(1)), "inch" if m.group(2) else m.group(3).casefold()))
        except ValueError:
            continue
    return out


def _related_units(a: str, b: str) -> bool:
    # Two spellings of one unit as texts write it after a number: one is a prefix of the other, or an abbreviation of
    # it (its letters stand in the other in the same order, from the same first letter); an empty unit relates to
    # any. Only the letters the two texts write are compared: no unit is known beforehand.
    a, b = a.casefold(), b.casefold()
    if not a or not b or a.startswith(b) or b.startswith(a):
        return True
    short, long = sorted((a, b), key=len)
    rest = iter(long)
    return short[0] == long[0] and all(letter in rest for letter in short)


def _same_measure(want: tuple[float, str], have: tuple[float, str]) -> bool:
    # The same number with the same unit in one of its spellings. Nothing is converted: a quantity that a page states
    # in another unit is another text, and reading it is the model's part.
    return _related_units(want[1], have[1]) and abs(want[0] - have[0]) < 1e-9


def _stem(word: str) -> str:
    # One spelling for the forms of a word: plural (-s, -es, -ies), past or adjective (-ed), a final e and -our for
    # -or, so that "striped" and "stripe", "boxes" and "box", "colour" and "color" compare equal.
    w = word
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ie"):  # a plural in -ies whose s was dropped: supplie(s) is supply
        w = w[:-2] + "y"
    elif len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("our"):  # one spelling for colour and color
        w = w[:-2] + "r"
    return w


def _split(requirement: Any) -> tuple[str, str]:
    # A requirement worded "attribute: value" as its two parts; one without a colon is all value.
    head, colon, tail = str(requirement or "").partition(":")
    if colon and head.strip() and tail.strip():
        return head.strip(), tail.strip()
    return "", str(requirement or "").strip()



def _without_measures(text: str) -> str:
    # The text without its numbers and their units: a shelf label carries no quantity.
    return re.sub(r"(?<![\w.])\d+(?:\.\d+)?(?:\s?[a-zA-Z]{1,8})?(?![\w])", " ", text)


def _shelf_names_kind(path_text: Any, kind: str) -> bool:
    # A category path carries the kind of product the model named: every word of the kind (its quantities aside; a
    # final s aside) stands in the path, or all but one when the kind has three words or more.
    wanted = {_stem(w) for w in _words(_without_measures(kind)) if len(w) >= 2}
    have = {_stem(w) for w in _words(path_text)}
    if not wanted:
        return True
    missing = len(wanted - have)
    if missing == 0 or (len(wanted) >= 3 and missing <= 1):
        return True
    # the model may have worded attributes into the kind (a colour or a shape before the noun): the page's own shelf label
    # (the last element of the path) standing whole in the kind names it too
    own = {_stem(w) for w in _words(str(path_text).split(">")[-1]) if len(w) >= 2}
    return bool(own) and own <= wanted


def _label_words(label: Any) -> tuple[str, ...]:
    # A shelf label as its words in one spelling (singular, -ie written -y), so that two spellings of one label
    # compare equal: "Hoodies & Jackets" and "hoodie & jacket". The "&" of a label is a word of it ("and" reads the
    # same): the shelf "A & B" is another shelf than "A B" filed under it.
    out = []
    for w in "".join(ch if ch.isalnum() else " & " if ch == "&" else " " for ch in str(label or "").casefold()).split():
        if w == "and":
            w = "&"
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = _other_number(w)
        if w.endswith("ie"):
            w = w[:-2] + "y"
        out.append(w)
    return tuple(out)


def _markup(value: Any) -> bool:
    # A value that carries a tag of page markup is a leftover of the page's layout, not a statement about the product.
    return bool(re.search(r"</?[a-zA-Z][^<>]*>", str(value)))


def _compact(value: Any) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _query_variants(query: str) -> list[str]:
    # The same query with each number written the other way round its unit: "60 grams" as "60grams" and "60g" as
    # "60 g", since a store's index may hold either spelling as one word.
    glued = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?) ([a-zA-Z]{1,8})(?![\w])", r"\1\2", query)
    split = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)([a-zA-Z]{1,8})(?![\w])", r"\1 \2", query)
    return [v for v in (glued, split) if v != query]


def _other_number(word: str) -> str:
    # The plural of a singular English noun, or the singular of a plural, by the common endings.
    low = word.casefold()
    if low.endswith("ies") and len(low) > 4:
        return word[:-3] + "y"
    if low.endswith(("ches", "shes", "sses", "xes", "zes")) and len(low) > 4:
        return word[:-2]
    if low.endswith(("ch", "sh", "ss", "x", "z")):
        return word + "es"
    if low.endswith("s") and len(low) > 3:
        return word[:-1]
    if low.endswith("y") and len(low) > 2 and low[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _argument_error(value: Any, schema: dict, path: str = "args") -> str | None:
    # Common tool-schema constraints; the store remains authoritative.
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: value is outside the advertised enum"
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds] if kinds else []
    if kinds:
        ok = False
        for kind in kinds:
            if kind == "null":
                ok = ok or value is None
            elif kind == "boolean":
                ok = ok or isinstance(value, bool)
            elif kind == "integer":
                ok = ok or (isinstance(value, int) and not isinstance(value, bool))
            elif kind == "number":
                ok = ok or (isinstance(value, (int, float)) and not isinstance(value, bool))
            elif kind == "string":
                ok = ok or isinstance(value, str)
            elif kind == "array":
                ok = ok or isinstance(value, list)
            elif kind == "object":
                ok = ok or isinstance(value, dict)
            else:
                ok = True
        if not ok:
            return f"{path}: expected {' or '.join(map(str, kinds))}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below the minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above the maximum"
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{path}: shorter than minLength"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{path}: longer than maxLength"
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return f"{path}: fewer than minItems"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{path}: more than maxItems"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _argument_error(item, item_schema, f"{path}[{index}]")
                if error:
                    return error
    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for name in schema.get("required") or []:
            if name not in value:
                return f"{path}: missing {name}"
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    return f"{path}: unexpected field {name}"
        for name, item in value.items():
            if name in properties and isinstance(properties[name], dict):
                error = _argument_error(item, properties[name], f"{path}.{name}")
                if error:
                    return error
    return None


def _observation(step: dict) -> dict:
    # Unwrap the public step envelope without losing terminal and error fields.
    if isinstance(step.get("observation"), dict):
        observation = deepcopy(step["observation"])
        if step.get("done") is True:
            observation["done"] = True
        if step.get("error"):
            observation["error"] = step["error"]
        return observation
    return deepcopy(step)


class EvidenceAgent:
    def __init__(self, problem: dict, proxy: Any):
        environment = problem["environment"]
        self.binding = deepcopy(environment["binding"])
        policy = deepcopy(environment["policy_view"])
        self.query = str(policy.get("query") or "")
        self.tools = policy["tools"]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in self.tools}
        self.max_steps = int(policy["max_steps"])
        self.max_calls = int(policy.get("max_calls_per_turn", 1))
        if self.max_steps < 1 or self.max_calls < 1:
            raise ValueError("public tool limits must be positive")
        self.proxy = proxy
        self.models = _models()
        self.model = self.models[0]
        self.turn = 0
        self.decisions = 0
        self.done = False
        self.stop_reason: str | None = None
        self.dialogue: list[dict] = []
        self.paths: Counter = Counter()
        # What the store has shown.
        self.pages: dict[str, dict] = {}  # product_id -> the page as last read
        self.listed: dict[str, dict] = {}  # product_id -> a result row of search or filter
        self.market: dict[tuple[str, str], dict] = {}  # (product_id, sku) -> latest price and stock
        self.cart: list[dict] = []
        self.events: list[dict] = []
        self.lines: list[dict] = []  # the shopper's lines, with the turn they came in
        self.notes: list[str] = []  # what the controller tells the model in the next decision
        self.sweeps: set[str] = set()  # categories swept already
        self.swept_limit: float | None = None  # the price limit the shelf was last filtered and swept within
        self.pending: dict | None = None  # the commit the shopper interrupted, until the model commits again
        self.plan_stale = False  # the latest plan reading brought no plan: the lines since the last plan are not in it
        self.latest: tuple[dict, tuple] | None = None  # the model's latest commit, as it made it, and the plan it was made under
        self.looked_for: tuple | None = None  # what the plan asked, and its price limit, when the model last looked values up
        self.pair_unanswered = False  # the shopper's answer to the question about two doubted wishes named neither
        self.opening_answer = False  # the answer to the question about the use has come and no plan has read it yet
        self.use_unanswered = False  # the plan gained no requirement from the answer to the question about the use
        self.uses: list[str] | None = None  # the uses of the purchase the first message names, as the model listed them
        self.waived_uses: set[str] = set()  # requirements that only restate a use and were demanded no claim
        self.use_questions: list[tuple[str, str]] = []  # (use, attribute) of each question about what a use requires
        self.kind_named = ""  # the kind of product as the first reading of the request named it
        self.grounded = False  # the answer was read once more with the facts of the shelf whose value it names
        self.asked_pair: tuple[list[str], str, int] | None = None  # the two doubted wishes asked about, the answer, its turn
        self.pair_entries: list[str] = []  # those two wishes as the plan worded them when the question went out
        self.pair_read = False  # a plan was read after the answer about the two wishes
        self.read_lines = 0  # the shopper's lines that the plan in force has read
        self.units_asked: set[str] = set()  # the entries worded as a bare number that a reading was asked about
        self.replaced: list[tuple[str, str]] = []  # (the requirement as it was, as the shopper changed it), each confirmed
        self.pointed: set[tuple[str, str]] = set()  # SKUs whose commit was answered once with other SKUs (closing phase; a plainer statement)
        self.auto_read = 0
        self.held: tuple[str, str] | None = None  # the alternative the controller added first
        self.added: list[tuple[str, str]] = []
        self.asked_turns: list[int] = []  # the store turns in which the controller asked the shopper a question
        self.ordered_claims: list[dict] = []
        self.calls_made: dict[str, str] = {}  # a store call as the model made it -> what it returned, once
        self.filtered_shelves: list[str] = []  # the categories the model filtered by and the store listed
        self.unavailable: list[dict] = []  # SKUs the store reported out of stock or over the limit after a commit
        # The conversation with the model: the request, its tool calls, their results, the shopper's lines and the
        # controller's reads; a status line is added for each decision and not kept.
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": "Shopper's request: " + self.query}]
        self.fetched: list[str] = []  # product ids whose pages the controller read since the last decision
        self.plan_read: dict | None = None  # the model's reading of the request (plan)
        self.refused: set[str] = set()  # commits refused for their claims, as the model made them
        self.reread: set[str] = set()  # commits sent back once only because a claim's value words the requirement otherwise
        self.readings: set[tuple] = set()  # (product, SKU, requirement) of the readings the model stood by
        self.affirmed: dict[tuple, bool] = {}  # (requirement, lines heard, ranked quantity) -> its value only affirms its attribute
        self.blocked: dict[tuple[str, str], int] = {}  # SKU -> commits refused for claims or coverage
        self.planned_lines = 0  # shopper lines the latest plan has read
        self.looked = False  # the variants of a chosen listing were looked up once
        self.widened = False  # the view of the shelf was widened once (brands, title words)
        self.going_on = False  # an interrupted purchase is being carried on with the commit taken before
        self.doubted: list[str] = []  # the wishes a plan named as doubted: requirements by a later answer only

    # ----- the store -----
    def call(self, calls: list[dict]) -> list[dict]:
        # One store turn with these calls; returns the observations in order and absorbs what they show.
        self.turn += 1
        group = f"turn-{self.turn}"
        envelope = {**self.binding, "call_id": group, "idempotency_key": group, "turn": self.turn,
                    "calls": [{"call_id": f"{group}-{i}", "action": {"name": c["name"], "args": c["args"]}}
                              for i, c in enumerate(calls)]}
        result = self.request(envelope, "/environment/call")
        if result is None:
            raise RuntimeError("Environment outcome unknown after idempotent retry; stopped without a new action.")
        self.dialogue.append({"role": "assistant", "content": "Environment actions", "environment_request": envelope,
                              "environment_result": deepcopy(result)})
        rows = result.get("calls")
        if not isinstance(rows, list) or len(rows) != len(calls):
            raise RuntimeError("Invalid environment response; stopped without assuming execution.")
        by_id = {row.get("call_id"): row for row in rows if isinstance(row, dict)}
        observations = []
        for wire, c in zip(envelope["calls"], calls):
            row = by_id.get(wire["call_id"])
            if row is None or not isinstance(row.get("observation"), dict):
                raise RuntimeError("Environment response IDs do not match the submitted actions.")
            observation = _observation(row["observation"])
            self.absorb(c["name"], c["args"], observation)
            observations.append(observation)
        line = result.get("user_message")
        content = line.get("content") if isinstance(line, dict) else line
        if isinstance(content, str) and content.strip():
            # each line of the message is a line of its own (a later line overrides an earlier one), numbered within
            # its message
            self.dialogue.append({"role": "user", "content": content.strip()})
            for index, part in enumerate(part.strip() for part in content.strip().split("\n") if part.strip()):
                self.lines.append({"turn": self.turn, "content": part, "part": index, "whole": content.strip()})
                self.messages.append({"role": "user", "content": "Shopper: " + part})
        return observations

    def answer_line(self, line: dict) -> bool:
        # The shopper's answer to a message of the controller: the first line of the message that came in that turn.
        # A further line of the same message is one the shopper said of their own accord.
        return line["turn"] in self.asked_turns and not line.get("part")

    def absorb(self, name: str, args: dict, observation: dict) -> None:
        self.done = self.done or observation.get("done") is True
        if isinstance(observation.get("market_event"), dict):
            self.events.append({"turn": self.turn, "event": deepcopy(observation["market_event"])})
        if name in ("search", "filter") and isinstance(observation.get("results"), list):
            if name == "filter" and observation["results"] and isinstance(args.get("category"), str):
                if args["category"].casefold() not in self.filtered_shelves:
                    self.filtered_shelves.append(args["category"].casefold())
            for row in observation["results"]:
                if isinstance(row, dict) and isinstance(row.get("product_id"), str):
                    entry = self.listed.setdefault(row["product_id"], deepcopy(row))
                    entry["filtered"] = entry.get("filtered", False) or name == "filter"
                    self.note_market(row.get("product_id"), row.get("sku"), row)
        pages = [observation] if name == "view" else observation.get("compared") if name == "compare" else []
        for page in pages if isinstance(pages, list) else []:
            if isinstance(page, dict) and isinstance(page.get("product_id"), str):
                self.pages[page["product_id"]] = deepcopy(page)
                self.fetched.append(page["product_id"])
                for variant in page.get("variants") or []:
                    if isinstance(variant, dict):
                        self.note_market(page["product_id"], variant.get("sku"), variant)
        if name == "inspect_stock" and isinstance(observation.get("product_id"), str):
            self.note_market(observation["product_id"], observation.get("sku"), observation)
        if isinstance(observation.get("cart"), list):
            self.cart = deepcopy(observation["cart"])
            for item in self.cart:
                if isinstance(item, dict):
                    fields = {k: item[k] for k in ("in_stock", "currency") if k in item}
                    if "current_price" in item:
                        fields["price"] = item["current_price"]
                    self.note_market(item.get("product_id"), item.get("sku"), fields)
            if name == "add_to_cart" and isinstance(observation.get("added"), dict):
                key = (args.get("product_id"), args.get("sku"))
                if key not in self.added:
                    self.added.append(key)

    def note_market(self, pid: Any, sku: Any, fields: dict) -> None:
        if not isinstance(pid, str) or not isinstance(sku, str):
            return
        entry = self.market.setdefault((pid, sku), {})
        for field in ("price", "in_stock", "currency"):
            if field in fields and fields[field] is not None:
                entry[field] = fields[field]
        entry["turn"] = self.turn

    def state_of(self, pid: str, sku: str) -> dict:
        return self.market.get((pid, sku), {})

    def turns_left(self) -> int:
        return self.max_steps - self.turn

    # ----- the model -----
    def request(self, payload: dict, path: str) -> dict | None:
        # One store call with up to two attempts, or one model request: the model the request names, once more when
        # its server faulted (a 5xx is often gone a moment later), then the other model; a model that refuses the
        # request as it is (a 4xx) is not asked again. A 429 waits (RATE_LIMIT_WAITS) and asks the same model again.
        request = deepcopy(payload)
        inference = path == "/inference/chat/completions"
        models: list = [request.get("model")] + [m for m in self.models if m != request.get("model")] if inference else [None, None]
        waits, tries, again = 0, 0, False
        while models:
            tries += 1
            if inference:
                request["model"] = models[0]
                for settings in MODEL_SETTINGS.values():  # the other model's settings leave the request
                    for key in settings:
                        request.pop(key, None)
                request.update(deepcopy(MODEL_SETTINGS[request["model"]]))
                request["max_tokens"] = OUTPUT_TOKENS
            self.proxy.timeout = INFERENCE_TIMEOUT if inference else STORE_TIMEOUT
            error: Any = None
            try:
                verbose = getattr(self.proxy, "post_verbose", None)
                if callable(verbose):
                    response = verbose(path, json_data=deepcopy(request))
                    result, error = response.data, response.error
                else:
                    result = self.proxy.post(path, json_data=deepcopy(request))
            except Exception as exc:  # noqa: BLE001 - a transport failure is one failed attempt
                result, error = None, {"kind": type(exc).__name__, "status": None}
            if isinstance(result, dict) and (not inference or isinstance(result.get("choices"), list)):
                return result
            status = error.get("status") if isinstance(error, dict) else None
            self.dialogue.append({"role": "assistant", "content": "Request failed.",
                                  "request_failure": {"path": path, "model": request.get("model"), "attempt": tries,
                                                      "status": status, "kind": (error or {}).get("kind") if isinstance(error, dict) else str(error)}})
            if status == 429 and waits < len(RATE_LIMIT_WAITS):
                sleep(RATE_LIMIT_WAITS[waits])
                waits += 1
                continue
            if status in (400, 401, 402, 403, 422) and not inference:
                self.stop_reason = "non_retryable_request_error"
                return None
            if inference and isinstance(status, int) and status >= 500 and not again:
                again = True
                sleep(RATE_LIMIT_WAITS[0])
                continue
            models.pop(0)
        return None

    def model_tools(self) -> list[dict]:
        tools = [deepcopy(t) for t in self.tools if t["function"]["name"] in MODEL_STORE_TOOLS]
        tools.append(deepcopy(FIND_TOOL))
        tools.append(deepcopy(COMMIT_TOOL))
        return tools

    def page_view(self, pid: str) -> dict:
        page = self.pages[pid]
        facts = {k: v for k, v in (page.get("facts") if isinstance(page.get("facts"), dict) else {}).items() if not _markup(v)}
        shown_facts = {str(k): _cut(v, VALUE_CHARS) for k, v in list(facts.items())[:FACT_LABELS]}
        variants = []
        for variant in (page.get("variants") or [])[:VARIANTS_SHOWN]:
            if not isinstance(variant, dict):
                continue
            state = self.state_of(pid, str(variant.get("sku")))
            variants.append({"sku": variant.get("sku"), "options": _cut(variant.get("options"), VALUE_CHARS),
                             "price": state.get("price", variant.get("price")),
                             "in_stock": state.get("in_stock", variant.get("in_stock"))})
        path = page.get("category_path") if isinstance(page.get("category_path"), list) else []
        return {"product_id": pid, "title": _cut(page.get("title"), TITLE_CHARS), "brand": page.get("brand"),
                "category": " > ".join(str(p) for p in path[-2:]), "facts": shown_facts, "variants": variants,
                **({"more_facts": len(facts) - FACT_LABELS} if len(facts) > FACT_LABELS else {}),
                **({"more_variants": len(page.get("variants") or []) - VARIANTS_SHOWN}
                   if len(page.get("variants") or []) > VARIANTS_SHOWN else {})}

    def status(self) -> str:
        unread = sum(1 for pid in self.listed if pid not in self.pages)
        parts = [f"Store turns left: {self.turns_left()}. Decisions left: {DECISION_LIMIT - self.decisions}.",
                 f"Pages read: {len(self.pages)}. Listed but unread: {unread}."]
        shelves = self.shelves_read()
        bound = self.kind_shelf()
        if bound is not None:
            parts.append(f"The shelf that is the kind of product the shopper named: {bound!r}; pages read on it: "
                         f"{sum(1 for pid in self.pages if self.under(pid, bound))}.")
        elif shelves:
            parts.append("Shelves of the pages read (exact filter labels): " + _json(shelves[:4]) + ".")
        if self.closing() and not self.done:
            parts.append("Few store turns or decisions are left: call commit now with the best SKU in view; the controller "
                         "orders it with the claims its page bears out.")
        elif (not self.done and not self.added and self.held is None and self.turns_left() <= FIRST_ADD_TURNS_LEFT
              and self.paths["page_lookups"]):
            # the purchase has not begun and turns are running short: a shopper may lower the limit or change a
            # requirement once something is in the cart, and answering that needs turns of its own
            parts.append("Nothing has entered the cart yet and store turns are running short: call commit now with the best SKU "
                         "in view that meets every requirement (the shopper may still change the request or the limit once "
                         "something is in the cart, and answering that takes turns).")
        if self.cart:
            parts.append("Cart: " + _json([{k: i.get(k) for k in ("product_id", "sku", "in_stock", "current_price")}
                                            for i in self.cart if isinstance(i, dict)]))
        if self.unavailable:
            parts.append("Reported unavailable after your commit: " + _json(self.unavailable))
        if self.notes:
            parts.append("Controller notes: " + " | ".join(self.notes[-6:]))
        self.notes = []
        return " ".join(parts)

    def decide(self) -> list[dict] | None:
        # One model decision over the conversation so far: its tool calls (id, name, args), [] when it called none,
        # None when no model answered.
        self.decisions += 1
        payload = {"model": self.model, "messages": self.messages + [{"role": "user", "content": self.status()}],
                   "tools": self.model_tools(), "tool_choice": "required"}
        result = self.request(payload, "/inference/chat/completions")
        if result is None:
            return None
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            return None
        tool_calls = [item for item in message.get("tool_calls") or [] if isinstance(item, dict)
                      and isinstance(item.get("function"), dict)]
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        if tool_calls:
            assistant["tool_calls"] = [{"id": str(item.get("id") or f"call-{self.decisions}-{i}"), "type": "function",
                                        "function": {"name": item["function"].get("name"),
                                                     "arguments": item["function"].get("arguments")
                                                     if isinstance(item["function"].get("arguments"), str)
                                                     else _json(item["function"].get("arguments"))}}
                                       for i, item in enumerate(tool_calls)]
        self.messages.append(assistant)
        self.dialogue.append(deepcopy(assistant))
        calls = []
        for item in assistant.get("tool_calls") or []:
            try:
                args = json.loads(item["function"]["arguments"])
            except (ValueError, TypeError):
                args = None
            calls.append({"id": item["id"], "name": item["function"]["name"],
                          "args": args if isinstance(args, dict) else {},
                          "result": None if isinstance(args, dict) else {"not_run": "the arguments are not a JSON object"}})
        return calls

    def answer(self, calls: list[dict]) -> None:
        # The result of each tool call, as a tool message, then what the controller fetched on its own.
        for c in calls:
            if c.get("variant_of"):
                continue
            message = {"role": "tool", "tool_call_id": c["id"],
                       "content": _json(c["result"] if c["result"] is not None else {"not_run": "no result"})}
            self.messages.append(message)
            self.dialogue.append(deepcopy(message))
        self.show_pages()

    def under(self, pid: str, shelf: str) -> bool:
        # Whether a read page is filed under a shelf: an element of its category path carries the same label.
        path = (self.pages.get(pid) or {}).get("category_path")
        return isinstance(path, list) and _label_words(shelf) in [_label_words(element) for element in path]

    def show_pages(self) -> None:
        # What the controller fetched since the model last looked. When the store listed the shelf that is the kind
        # of product the shopper named, the pages filed under it whose SKU can be ordered (in stock, within the
        # plan's price limit) are shown and the others counted; else every page is shown.
        pids = [pid for pid in dict.fromkeys(self.fetched) if pid in self.pages]
        self.fetched = []
        if not pids:
            return
        bound = self.kind_shelf()
        if bound is None:
            self.messages.append({"role": "user", "content": "Controller: pages read for you.\n"
                                  + _json([self.page_view(pid) for pid in pids])})
            return
        limit = _number(self.plan_read.get("price_limit")) if isinstance(self.plan_read, dict) else None
        shown, elsewhere, unavailable = [], 0, 0
        for pid in pids:
            if not self.under(pid, bound):
                elsewhere += 1
            elif all(self.availability(pid, str(v.get("sku")), limit) for v in self.pages[pid].get("variants") or [] if isinstance(v, dict)):
                unavailable += 1
            else:
                shown.append(self.page_view(pid))
        self.messages.append({"role": "user", "content": f"Controller: pages read for you, those filed under the shelf {bound!r} whose SKU "
                              f"can be ordered ({elsewhere} pages filed under other shelves and {unavailable} out of stock or over the "
                              "price limit are not shown).\n" + _json(shown)})

    # ----- the controller -----
    def ask(self) -> None:
        # The question the request calls for, each asked once at most: which of two doubted wishes matters more when
        # the plan names such a pair (a shopper who doubts two wishes answers that question and no other); else what
        # the purchase is for. When the answer about the two wishes names neither, the use is asked about as well.
        schema = self.schemas.get("message")
        field = next((f for f in ("content", "text") if isinstance((schema or {}).get("properties"), dict)
                      and f in schema["properties"]), None)
        if field is None or self.done or self.stop_reason is not None or self.turns_left() <= PURCHASE_TURNS:
            return
        pair = self.two_wishes()
        asked_use = self.paths["use_questions"] + self.paths["opening_questions"]  # questions about the use, either kind
        if (pair is not None and not self.paths["priority_questions"]
                and (not asked_use or self.use_unanswered)):
            question, path = PRIORITY_QUESTION.format(a=pair[0], b=pair[1]), "priority_questions"
        elif (self.asked_pair is not None and self.paths["priority_questions"] and asked_use
              and not self.paths["priority_questions_again"] and self.pair_unanswered and self.terms_only()):
            # both questions were answered with the terms of the request alone: the one about the two wishes was
            # not understood, and it is the one this shopper answers (asked about the same two wishes: a plan read
            # after the first question lists no pair any more)
            pair = self.asked_pair[0]
            question, path = PRIORITY_AGAIN.format(a=pair[0], b=pair[1]), "priority_questions_again"
        elif self.paths["priority_questions"] and not self.pair_unanswered:
            return
        elif not asked_use:
            # What the use the first message names requires of one attribute: a shopper who has told the use answers
            # that, and one who answers the question about the use answers it too. A message that names no use is
            # asked what the purchase is for.
            question = self.use_question()
            path = "use_questions" if question is not None else "opening_questions"
            question = question if question is not None else OPENING_QUESTION
        elif (self.paths["use_questions"] and not self.paths["opening_questions"] and self.use_unanswered
              and self.restates_terms(self.latest_answer())):
            # the question about the attribute was answered with the terms of the request alone: this shopper is
            # asked what the purchase is for
            question, path = OPENING_QUESTION, "opening_questions"
        else:
            return
        entries = (self.pair_entries if path == "priority_questions_again"
                   else self.held_wishes(pair) if pair is not None and path not in ("opening_questions", "use_questions") else [])
        self.before_speaking()
        self.call([{"name": "message", "args": {field: question}}])
        self.asked_turns.append(self.turn)
        self.paths[path] += 1
        self.opening_answer = path in ("opening_questions", "use_questions")
        if path not in ("opening_questions", "use_questions"):
            answer = next((line["content"] for line in self.lines if line["turn"] == self.turn and not line.get("part")), "")
            self.pair_unanswered = not self.names_a_wish(answer, pair)
            self.asked_pair = (list(pair), answer, self.turn)
            self.pair_entries, self.pair_read = entries, False
            self.doubted += [wish for wish in pair if wish not in self.doubted]  # requirements by the answer alone

    def latest_answer(self) -> str:
        # The shopper's latest answer to a question of the controller ("" when none came).
        answers = [line["content"] for line in self.lines if self.answer_line(line)]
        return answers[-1] if answers else ""

    def restates_terms(self, text: str) -> bool:
        # Whether a shopper's line only restates the terms of the request: it carries the price limit (a shopper who
        # is asked something they have no answer to repeats the limit and the shelf).
        limit = self.ceiling(None)
        return bool(text) and limit is not None and any(abs(n - limit) < 1e-9 for n, _ in _measures(text.replace(",", "")))

    def uses_named(self) -> list[str]:
        # The uses of the purchase that the first message names, as the model lists them in the message's own words
        # (every word of each stands in the message, and none is a part of another), read once.
        if self.uses is None:
            args = self.consult(USES_FIRST, USES_FIRST_TOOL) or {}
            written = _words(self.query)
            uses = [" ".join(use.split()) for use in args.get("uses") or []
                    if isinstance(use, str) and _words(use) and _words(use) <= written]
            self.uses = [use for use in dict.fromkeys(uses) if not any(other != use and _words(use) <= _words(other) for other in uses)]
            self.dialogue.append({"role": "assistant", "content": "Controller uses named", "controller_uses": list(self.uses)})
        return list(self.uses)

    def use_question(self, shelf: bool = False) -> str | None:
        # One question about what a use the first message names requires of one attribute, written by the model: it
        # names the use in the message's own words and one attribute (not one the plan holds or ranks by, nor one asked
        # about before); when the model's question does not name the use, the question is worded from its attribute.
        # None when the message names no use or the model wrote no attribute.
        uses = self.uses_named()
        # the use asked about: the first the message names; after an answer that restated the terms, one not asked
        # about yet (each use the shopper names has a need of its own), else the first again with another attribute
        asked_uses = [named for named, _ in self.use_questions]
        use = next((u for u in uses if u not in asked_uses), uses[0]) if uses else None
        if use is None and not shelf:
            return None  # before the shelf is read, a request that names no use is asked what the purchase is for
        plan = self.plan_read if isinstance(self.plan_read, dict) else {}
        ranking = plan.get("ranking") if isinstance(plan.get("ranking"), dict) else {}
        held = [entry for entry in self.plan_attributes() if isinstance(entry, str)]
        held += [str(ranking["quantity"])] if str(ranking.get("quantity") or "").strip() else []
        asked = [attribute for named, attribute in self.use_questions if named == (use or "")]
        before_text = (", and not one asked about before (" + _json(asked) + ")") if asked else ""
        instruction = (USE_QUESTION.format(use=_json(use), held=_json(held), asked=before_text) if use
                       else USE_QUESTION_NONE.format(held=_json(held), asked=before_text))
        args = self.consult(instruction + (USE_SHELF if shelf else ""), USE_QUESTION_TOOL) or {}
        attribute = " ".join(str(args.get("attribute") or "").split())
        before = {_stem(x) for x in _words(" ".join(asked))}
        repeated = lambda named: bool(_words(named)) and {_stem(w) for w in _words(named)} <= before
        if attribute and repeated(attribute):
            # the attribute was asked about before: read once more, with it named (the first answer is in view)
            self.paths["use_attributes_read_again"] += 1
            args = self.consult(instruction + (USE_SHELF if shelf else "") + USE_ATTRIBUTE_AGAIN.format(attribute=_json(attribute)),
                                USE_QUESTION_TOOL) or {}
            attribute = " ".join(str(args.get("attribute") or "").split())
        if not attribute or len(attribute) > VALUE_CHARS or not _words(attribute) or repeated(attribute):
            return None  # (an attribute asked about before, in whatever wording, is no new question)
        question = self.attribute_question(args.get("question"), use, attribute)
        self.use_questions.append((use or "", attribute))
        return question

    def attribute_question(self, written: Any, use: str | None, attribute: str) -> str:
        # The model's question when it is one short question that names the use (when there is one) and asks about
        # one attribute without listing alternatives or examples (a shopper asked several things at once answers none);
        # else the plain form of the same question.
        question = " ".join(str(written or "").split())
        named = {w for w in _words(use or "") if len(w) >= 4}
        names_use = not named or len(named & _words(question)) * 2 >= len(named)
        several = bool(re.search(r"\bsuch as\b|\be\.g\.|\bfor example\b|\([^)]*,[^)]*\)|(?:,\s*[^,?]+){2,}(?:,|\s+or\b)", question, re.I))
        if question and question.count("?") == 1 and len(question) <= USE_QUESTION_CHARS and names_use and not several:
            return question
        self.paths["use_questions_templated"] += 1
        return USE_TEMPLATE.format(use=use, attribute=attribute) if use else USE_TEMPLATE_NONE.format(attribute=attribute)

    def use_again(self) -> None:
        # After the shelf was read: when the question about the use and the question about what the use requires
        # (if the request named a use) were both answered with the terms of the request alone, one attribute is asked
        # about, with the SKUs read in view, before any cart addition; once more when that answer restated the terms
        # too (another use the message names, or another attribute).
        schema = self.schemas.get("message")
        field = next((f for f in ("content", "text") if isinstance((schema or {}).get("properties"), dict)
                      and f in schema["properties"]), None)
        if (field is None or self.done or self.stop_reason is not None or self.added
                or self.paths["use_questions"] > USE_QUESTIONS_AFTER_SHELF
                or not self.paths["opening_questions"] or not self.use_unanswered
                or self.paths["use_questions_again"] >= USE_QUESTIONS_AFTER_SHELF
                or not self.restates_terms(self.latest_answer())
                or self.turns_left() <= PURCHASE_TURNS + CLOSING_TURNS + 2 * VARIANT_TURNS):
            return
        self.paths["use_questions_again"] += 1
        fetched, kept = list(self.fetched), len(self.messages)  # the listings read are put in view for this reading only
        self.show_pages()
        question = self.use_question(shelf=True)
        del self.messages[kept:]
        self.fetched = fetched
        if question is None:
            return
        self.before_speaking()
        self.call([{"name": "message", "args": {field: question}}])
        self.asked_turns.append(self.turn)  # what the shopper says to it is an answer
        self.paths["use_questions"] += 1
        self.opening_answer = True

    def before_speaking(self) -> None:
        # A store that reported a change with a cart addition has the shopper remark on it in the store turn after
        # that one, and a message sent in that turn is not read. When the last store turn was such an addition, one
        # store turn reads the cart (else the stock of the SKU that went in) and the message goes out in the next.
        if not any(event["turn"] == self.turn for event in self.events) or self.turns_left() <= CHOICE_TURNS:
            return
        calls = []
        if "inspect_cart" in self.schemas and _argument_error({}, self.schemas["inspect_cart"]) is None:
            calls.append({"name": "inspect_cart", "args": {}})
        elif self.added:
            calls = self.rereads(*self.added[-1])[:1]
        if calls:
            self.call(calls)
            self.paths["turns_given_to_a_remark"] += 1

    def terms_only(self) -> bool:
        # Whether every answer the shopper gave so far only restated the terms of the request: each carries the
        # price limit (a shopper who is asked something they have no answer to repeats those terms).
        limit = self.ceiling(None)
        answers = [line["content"] for line in self.lines if self.answer_line(line)]
        return (bool(answers) and limit is not None
                and all(any(abs(n - limit) < 1e-9 for n, _ in _measures(answer.replace(",", ""))) for answer in answers))

    def held_wishes(self, pair: list[str]) -> list[str]:
        # The entries of the plan (a requirement or a preference, as the plan words it) that the two wishes name, in
        # the pair's order; fewer than two when the plan does not hold both apart.
        entries = self.plan_attributes() + self.plan_preferences()
        held = [next((entry for entry in entries if self.names_requirement(wish, entry)), None) for wish in pair]
        return [entry for entry in dict.fromkeys(held) if entry is not None]

    def two_wishes(self) -> list[str] | None:
        # The two wishes to ask about: the pair the plan names as doubted, when both are wishes the plan lists; else,
        # when the plan lists exactly two preferences (what the shopper would like and does not require), those two,
        # each worded as the plan words it. None when the plan holds neither.
        plan = self.plan_read if isinstance(self.plan_read, dict) else {}
        pair = plan.get("doubted_pair")
        if (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) and x.strip() for x in pair)
                and self.wishes(pair) and not self.names_uses(pair)):
            return [x.strip() for x in pair]
        preferences = [p for p in self.plan_preferences() if p.strip()]
        if len(preferences) == 2 and not self.names_uses(preferences):
            return [" ".join(p.replace(":", " ").split()) for p in preferences]
        return None

    def names_uses(self, pair: list[str]) -> bool:
        # Whether each of two wishes names a use of the purchase that the first message names (as the model listed
        # them): a wish shares at least half of a use's words of four letters or more. Two uses are no pair of wishes
        # about the item: each has a need of its own, asked about in turn.
        uses = [{w for w in _words(use) if len(w) >= 4} for use in self.uses_named()]
        uses = [named for named in uses if named]
        if len(uses) < 2:
            return False
        matched = [any(len(named & _words(wish)) * 2 >= len(named) for named in uses) for wish in pair]
        if all(matched):
            self.paths["pairs_of_uses_dropped"] += 1
            return True
        return False

    def settle_pair(self, plan: dict) -> str:
        # After the shopper answered which of two doubted wishes matters more, the wish the answer names is a
        # requirement and the other a preference, in the plan read right after the answer and in every later one:
        # the shopper doubted that the two can be had together and chose. A wish that a later line of the shopper's
        # own names stays as the plan has it. Returns the note for the model.
        if self.asked_pair is None or self.pair_unanswered or not isinstance(plan.get("requirements"), list):
            return ""
        pair, answer, turn = self.asked_pair
        required = [entry for entry in plan["requirements"][1:] if isinstance(entry, str)]
        wished = [entry for entry in plan.get("preferences") or [] if isinstance(entry, str)]
        held = [next((entry for entry in required + wished if self.names_requirement(wish, entry)), None) for wish in pair]
        if None in held or held[0] == held[1]:
            return ""
        chosen = [self.answer_names(answer, held[index], [held[1 - index]]) for index in (0, 1)]
        if chosen.count(True) != 1:
            return ""
        first, other = held[chosen.index(True)], held[chosen.index(False)]
        later = " ".join(line["content"] for line in self.lines if line["turn"] > turn and not self.answer_line(line))
        raised = first not in required
        lowered = other in required and not (later.strip() and self.names_requirement(later, other))
        if not raised and not lowered:
            return ""
        if raised:
            plan["preferences"] = [entry for entry in plan.get("preferences") or [] if entry != first]
            plan["requirements"] = plan["requirements"] + [first]
            plan["doubted_pair"] = None
        if lowered:
            plan["requirements"] = [entry for entry in plan["requirements"] if entry != other]
            plan["preferences"] = [entry for entry in plan.get("preferences") or [] if entry != other] + [other]
        self.paths["wishes_settled"] += 1
        if not raised:
            return f" (the shopper's answer names {first!r}: of the two doubted wishes {other!r} is a preference)"
        return (f" (the shopper's answer names {first!r} of the two doubted wishes: it is a requirement"
                + (f" and {other!r} a preference)" if lowered else ")"))

    def keep_wishes(self, plan: dict) -> str:
        # The answer about two wishes chooses between them as the request words them and changes neither: when a
        # plan read after it holds, under the attribute of one of the two, another value than the wish's own, the
        # wish's own value stands, unless a line the shopper said of their own accord after the answer names the new
        # value. Returns the note for the model.
        if self.asked_pair is None or len(self.pair_entries) != 2:
            return ""
        turn = self.asked_pair[2]
        later = " ".join(line["content"] for line in self.lines if line["turn"] > turn and not self.answer_line(line))
        stems = lambda value: {_stem(w) for w in _words(value)}
        notes = []
        for name in ("requirements", "preferences"):
            rows = plan.get(name)
            for index, entry in enumerate(rows if isinstance(rows, list) else []):
                if not isinstance(entry, str) or (name == "requirements" and index == 0):
                    continue
                attribute = _split(entry)[0]
                for wish in self.pair_entries:
                    if (attribute and stems(attribute) == stems(_split(wish)[0])
                            and not self.names_requirement(entry, wish) and not self.names_requirement(wish, entry)
                            and not (later.strip() and self.names_requirement(later, entry))):
                        rows[index] = wish
                        notes.append(f"{wish!r} stands as the request words it, not {entry!r}")
                        self.paths["wishes_kept_as_worded"] += 1
                        break
        return (" (" + "; ".join(notes) + ")") if notes else ""

    def keep_requirements(self, plan: dict, since: int) -> str:
        # A requirement that a line of the shopper named (an answer, a line said unasked) stays in force until the
        # shopper changes it. When the plan, read again after a later line, no longer requires it (no requirement of
        # the new reading writes that value or stands under that attribute) and the lines read since name neither
        # that value (nor its number) nor that attribute, it is put back as a requirement (and leaves the preferences
        # when the reading moved it there). What the model read out of the first message alone and then dropped is
        # its own correction and stands; so does its reading of two doubted wishes, which the shopper's answer
        # settles. Returns the note for the model.
        entries = plan.get("requirements")
        if not isinstance(entries, list) or not entries:
            return ""
        fold = lambda value: " ".join(str(value).casefold().split())
        stems = lambda value: {_stem(w) for w in _words(value)}
        said = " ".join(line["content"] for line in self.lines[since:])
        earlier = " ".join(line["content"] for line in self.lines[:since])
        spoken, numbers = stems(said), [n for n, _ in _measures(said)]
        now = [entry for entry in entries[1:] if isinstance(entry, str)]
        back = []
        for was in self.plan_attributes():
            attribute = _split(was)[0]
            if any(self.names_requirement(wish, was) for wish in self.doubted):
                continue  # one of two doubted wishes
            if not (earlier.strip() and self.names_requirement(earlier, was)):
                continue  # read out of the first message alone
            if any(fold(entry) == fold(was) or self.names_requirement(entry, was)
                   or (attribute and stems(attribute) == stems(_split(entry)[0])) for entry in now):
                continue  # still required, or replaced under the same attribute
            if (self.names_requirement(said, was) or (attribute and stems(attribute) <= spoken)
                    or any(abs(n - m) < 1e-9 for n, _ in _measures(_split(was)[1]) for m in numbers)):
                continue  # the shopper spoke of it (its value, its attribute, or its number in whatever unit): the reading stands
            back.append(was)
        for was in back:
            attribute = _split(was)[0]
            plan["preferences"] = [entry for entry in plan.get("preferences") or []
                                   if not (isinstance(entry, str) and (fold(entry) == fold(was) or (
                                       attribute and stems(attribute) == stems(_split(entry)[0])
                                       and self.names_requirement(entry, was))))]
            entries.append(was)
            self.paths["requirements_kept"] += 1
        if not back:
            return ""
        return (" (the plan before this line required " + ", ".join(repr(was) for was in back)
                + ", and the line does not speak of it: it stays required)")

    def keep_changes(self, plan: dict, limit: float | None) -> str:
        # A line that lowers the price limit and repeats a value the shopper had replaced before does not bring that
        # value back: when the plan, read again after such a line, holds the earlier requirement in place of the
        # changed one, the changed one is put back, and the note for the model is returned. The earlier requirement
        # is the entry as the plan had it; else, when no entry carries the changed value, the entry whose value is
        # the earlier one under another wording of its attribute. Only a line that leaves the limit alone can change
        # that requirement again.
        ceiling = _number(plan.get("price_limit"))
        entries = plan.get("requirements")
        if not self.replaced or limit is None or ceiling is None or ceiling >= limit or not isinstance(entries, list):
            return ""
        fold = lambda v: " ".join(str(v).casefold().split())
        notes = []
        for was, changed in self.replaced:
            have = [fold(r) for r in entries]
            if fold(changed) in have:
                continue
            if fold(was) in have:
                at = have.index(fold(was))
            else:
                old, new = _split(was)[1], _split(changed)[1]
                at = next((i for i, r in enumerate(entries) if i and isinstance(r, str)
                           and self.carries(_split(r)[1], old) and self.carries(old, _split(r)[1])), None)
                if at is None or any(isinstance(r, str) and self.carries(_split(r)[1], new) for r in entries[1:]):
                    continue
            entries[at] = changed
            self.paths["changes_kept"] += 1
            notes.append(f"the shopper's new line lowers the price limit and repeats {was!r}, which the shopper had replaced "
                         f"before: the change stands and the plan requires {changed!r}")
        return (" (" + "; ".join(notes) + ")") if notes else ""

    def confirm(self, before: list[str], index: int, limit: float | None = None, wished: list[str] | None = None) -> None:
        # A line the shopper said of their own accord that set a requirement of the plan (a value replaced, a
        # requirement added) asks to be confirmed: one statement names each requirement the line set, with the value
        # it replaces (and a price limit lowered in the same breath), and says that nothing else changes. A
        # requirement counts as set by the line when the plan did not hold it before, the line names its value (or
        # its number) and nothing said before carries that value, unless the plan held it as a preference (wished):
        # a line that repeats what was asked all along (a lower price limit with the old requirements) gets no
        # answer, and neither does the shopper's answer to the controller's own question. A value that takes the
        # place of another under the same attribute is set by the line also when words said before carry it (the
        # kind of product may share its words), unless the shopper had replaced that value before. A wish the plan
        # held as a preference and now requires is set by the lines read, whatever words they use for it, when all of
        # them are the shopper's own (no answer among them).
        schema = self.schemas.get("message")
        field = next((f for f in ("content", "text") if isinstance((schema or {}).get("properties"), dict)
                      and f in schema["properties"]), None)
        if field is None or self.done or self.stop_reason is not None or self.turns_left() <= CHOICE_TURNS:
            return
        said = " ".join(line["content"] for line in self.lines[index:] if not self.answer_line(line))
        if not said.strip():
            return
        fold = lambda v: " ".join(str(v).casefold().split())
        preferred = {fold(_split(p)[1]) for p in wished or [] if isinstance(p, str)}
        now = self.plan_attributes()
        earlier, current = {fold(e) for e in before}, {fold(e) for e in now}
        heard = self.query + " " + " ".join(line["content"] for line in self.lines[:index])
        numbers = [n for n, _ in _measures(said)]
        gone = [e for e in before if fold(e) not in current]
        taken_back = {fold(_split(was)[1]) for was, _ in self.replaced}
        alone = not any(self.answer_line(line) for line in self.lines[index:])

        def spoken(entry: str) -> bool:
            return (self.names_requirement(said, entry)
                    or any(abs(n - m) < 1e-9 for n, _ in _measures(_split(entry)[1]) for m in numbers))

        def in_place(entry: str) -> bool:
            attribute, value = _split(entry)
            return (bool(attribute) and fold(value) not in taken_back
                    and any(fold(_split(g)[0]) == fold(attribute) and not self.carries(_split(g)[1], value)
                            and not self.carries(value, _split(g)[1]) for g in gone))

        fresh = [e for e in now if fold(e) not in earlier
                 and (fold(_split(e)[1]) in preferred or not self.carries(heard, _split(e)[1]) or in_place(e))
                 and (spoken(e) or (alone and fold(_split(e)[1]) in preferred))]
        if not fresh:
            return
        parts = []
        for entry in fresh:
            attribute, value = _split(entry)
            old = next((_split(g)[1] for g in gone if attribute and fold(_split(g)[0]) == fold(attribute)), None)
            if old is None and len(gone) == 1 and len(fresh) == 1 and fold(value) not in preferred:
                old = _split(gone[0])[1]  # (a wish made a requirement takes no other entry's place)
            named = f"{attribute} {value}".strip()
            parts.append(f"{named} in place of {old}" if old else f"{named} as well")
            was = next((g for g in gone if _split(g)[1] == old), None) if old else None
            if was is not None:
                self.replaced.append((was, entry))
        ceiling = self.ceiling(None)
        if limit is not None and ceiling is not None and ceiling < limit:
            parts.append(f"price limit {ceiling:g}")
        self.before_speaking()
        self.call([{"name": "message", "args": {field: CONFIRMATION.format(changes="; ".join(parts))}}])
        self.asked_turns.append(self.turn)  # what the shopper says to it is an answer, no line of their own accord
        self.paths["changes_confirmed"] += 1

    def names_a_wish(self, answer: str, pair: list[str]) -> bool:
        # Whether the shopper's answer names one of two doubted wishes: it writes the value of the plan's requirement
        # or preference that the wish carries, every word of that entry's attribute, or a number of its value that
        # the other wish does not carry.
        preferences = self.plan_read.get("preferences") if isinstance(self.plan_read, dict) else None
        entries = self.plan_attributes() + [e for e in (preferences if isinstance(preferences, list) else []) if isinstance(e, str)]
        named = [entry for entry in entries if any(self.names_requirement(wish, entry) for wish in pair)]
        return any(self.answer_names(answer, entry, [other for other in named if other != entry]) for entry in named)

    def answer_names(self, answer: str, entry: str, others: list[str]) -> bool:
        # Whether an answer about two wishes names the wish a plan entry carries: it writes the entry's value or
        # every word of its attribute; or it writes a number of that value which no other wish carries and which is
        # not the price limit (the answer chooses between the two: "the 10-pack" for 10 pcs).
        said = {_stem(w) for w in _words(answer)}
        attribute = {_stem(w) for w in _words(_split(entry)[0])}
        if self.names_requirement(answer, entry) or bool(attribute and attribute <= said):
            return True
        limit = self.ceiling(None)
        own = ({n for n, _ in _measures(_split(entry)[1])}
               - {n for other in others for n, _ in _measures(_split(other)[1])})
        heard = {n for n, _ in _measures(str(answer).replace(",", "")) if limit is None or abs(n - limit) > 1e-9}
        return bool(own & heard)

    def wishes(self, pair: list[str]) -> bool:
        # Whether both phrases of a doubted pair are wishes about the product that the plan lists: each writes the
        # value of one of its required attributes or preferences. A pair read out of other words of the request (what
        # the purchase is for, whom it is for) is no such pair: the question about the use is the one to ask then.
        preferences = self.plan_read.get("preferences") if isinstance(self.plan_read, dict) else None
        entries = self.plan_attributes() + [e for e in (preferences if isinstance(preferences, list) else []) if isinstance(e, str)]
        return all(any(self.names_requirement(wish, entry) for entry in entries) for wish in pair)

    def run_reads(self, calls: list[dict]) -> None:
        # The model's store calls, validated against the store's own schemas and run in as few turns as they need;
        # each call gets its result.
        accepted = []
        for c in calls:
            if c["result"] is not None:
                continue
            name = c["name"]
            if name not in self.schemas or name not in MODEL_STORE_TOOLS:
                c["result"] = {"not_run": f"the tool {name} is not offered"}
                continue
            error = _argument_error(c["args"], self.schemas[name])
            if error:
                c["result"] = {"not_run": error}
                continue
            key = f"{name} {_json(c['args'])}"
            if key in self.calls_made and name in ("search", "filter"):
                c["result"] = {"not_run": f"the same call was made before and returned {self.calls_made[key]}"}
                self.paths["repeats_skipped"] += 1
                continue
            if name == "view" and c["args"].get("product_id") in self.pages:
                c["result"] = {**self.page_view(c["args"]["product_id"]), "note": "as read before; prices and stock are "
                               "re-read when you commit"}
                self.paths["views_from_memory"] += 1
                continue
            if name == "compare" and isinstance(c["args"].get("product_ids"), list) and c["args"]["product_ids"] and all(
                    pid in self.pages for pid in c["args"]["product_ids"]):
                c["result"] = {"pages": [self.page_view(pid) for pid in c["args"]["product_ids"]], "note": "as read before"}
                self.paths["views_from_memory"] += 1
                continue
            if name in ("search", "filter") and self.turns_left() <= self.spare():
                c["result"] = {"not_run": "no store turns are left for new listings: call commit now with the best SKU in "
                                          "view that meets every requirement"}
                self.paths["late_reads_refused"] += 1
                continue
            if any(f"{a['name']} {_json(a['args'])}" == key for a in accepted):
                c["result"] = {"not_run": "the same call is in this decision already"}
                continue
            accepted.append(c)
        extra = []
        for c in accepted:
            if c["name"] == "search" and isinstance(c["args"].get("query"), str):
                for variant in _query_variants(c["args"]["query"]):
                    args = {**c["args"], "query": variant}
                    key = f"search {_json(args)}"
                    if key not in self.calls_made and not any(f"{a['name']} {_json(a['args'])}" == key for a in accepted + extra):
                        extra.append({"name": "search", "args": args, "result": None, "variant_of": c["id"]})
        for c in extra[: max(0, self.max_calls - len(accepted))]:
            accepted.append(c)
        for start in range(0, len(accepted), self.max_calls):
            if self.turns_left() <= CHOICE_TURNS:
                for c in accepted[start:]:
                    c["result"] = {"not_run": "few store turns are left"}
                break
            batch = accepted[start:start + self.max_calls]
            observations = self.call(batch)
            for c, observation in zip(batch, observations):
                c["result"] = self.result_view(c, observation)
            self.retry_filters(batch, observations)
            self.sweep(batch, observations)
        self.auto_reads()

    def result_view(self, c: dict, observation: dict) -> dict:
        name = c["name"]
        key = f"{name} {_json(c['args'])}"
        if observation.get("error"):
            self.calls_made[key] = "an error"
            return {"error": _cut(str(observation["error"]), 200)}
        if name in ("search", "filter"):
            rows = observation.get("results") if isinstance(observation.get("results"), list) else []
            shown = [{"product_id": r.get("product_id"), "title": _cut(r.get("title"), 90), "price": r.get("price"),
                      "in_stock": r.get("in_stock")} for r in rows[:RESULT_ROWS] if isinstance(r, dict)]
            self.calls_made[key] = f"{len(rows)} results"
            return {"results": shown, "note": "the controller reads the pages of new results and shows them below"}
        if name == "view" and isinstance(observation.get("product_id"), str):
            self.calls_made[key] = "read"
            return self.page_view(observation["product_id"]) if observation["product_id"] in self.pages else {"page": None}
        if name == "compare":
            pages = observation.get("compared") if isinstance(observation.get("compared"), list) else []
            self.calls_made[key] = "read"
            return {"pages": [self.page_view(p["product_id"]) for p in pages if isinstance(p, dict) and p.get("product_id") in self.pages]}
        self.calls_made[key] = "read"
        return {k: v for k, v in observation.items() if k != "done"}

    def shelves_read(self) -> list[str]:
        # The shelves of the pages read (the last element of each category path), the most frequent first.
        counts: Counter = Counter()
        for page in self.pages.values():
            path = page.get("category_path") if isinstance(page.get("category_path"), list) else []
            if path:
                counts[str(path[-1])] += 1
        return [shelf for shelf, _ in counts.most_common()]

    def retry_filters(self, batch: list[dict], observations: list[dict]) -> None:
        # A filter by category that listed nothing is tried again, in one turn, with the last element of a path, with
        # the other number of its last word, and with the shelves of pages read that share a word with it: the store
        # matches one category label exactly.
        calls = []
        for c, observation in zip(batch, observations):
            category = c["args"].get("category") if c["name"] == "filter" else None
            rows = observation.get("results") if isinstance(observation.get("results"), list) else []
            if not isinstance(category, str) or rows or observation.get("error"):
                continue
            label = category.replace("/", ">").split(">")[-1].strip()
            variants = [label] if label.casefold() != category.casefold() else []
            words = label.split()
            if words:
                variants.append(" ".join(words[:-1] + [_other_number(words[-1])]))
            asked = _words(label)
            variants += [shelf for shelf in self.shelves_read() if _words(shelf) & asked][:3]
            for variant in variants:
                args = {**c["args"], "category": variant}
                key = f"filter {_json(args)}"
                if key not in self.calls_made and _argument_error(args, self.schemas["filter"]) is None:
                    calls.append({"name": "filter", "args": args})
        if calls and self.turns_left() > PURCHASE_TURNS + 1:
            before = len(self.listed)
            observations = self.call(calls[: self.max_calls])
            for c, observation in zip(calls, observations):
                self.result_view(c, observation)
            self.paths["filter_retries"] += len(calls)
            self.notes.append(f"The controller retried the filter with {len(calls)} other spellings of the category "
                              f"and listed {len(self.listed) - before} more products.")

    def sweep(self, batch: list[dict], observations: list[dict]) -> None:
        # A filter by category that fills a whole page is repeated once with lower price caps, in one turn, so that the
        # cheaper part of the shelf is listed too.
        for c, observation in zip(batch, observations):
            category = c["args"].get("category") if c["name"] == "filter" else None
            rows = observation.get("results") if isinstance(observation.get("results"), list) else []
            if not isinstance(category, str) or category.casefold() in self.sweeps or len(rows) < RESULT_ROWS:
                continue
            prices = [p for p in (_number(r.get("price")) for r in rows if isinstance(r, dict)) if p is not None]
            ceiling = _number(c["args"].get("max_price")) or (max(prices) if prices else None)
            if ceiling is None or self.turns_left() <= PURCHASE_TURNS + 1:
                continue
            self.sweeps.add(category.casefold())
            calls = []
            schema = self.schemas.get("filter", {})
            for share in SWEEP_SHARES[: self.max_calls]:
                args = {**c["args"], "max_price": round(ceiling * share, 2)}
                if _argument_error(args, schema) is None:
                    calls.append({"name": "filter", "args": args})
            if calls:
                before = len(self.listed)
                self.call(calls)
                self.paths["sweep_filters"] += len(calls)
                self.notes.append(f"The controller filtered the shelf {category!r} with lower price caps and listed "
                                  f"{len(self.listed) - before} more products.")

    def auto_reads(self) -> None:
        # The pages of listed products not read yet, up to AUTO_READ_PAGES in one turn: in-stock results first.
        reader = "compare" if "compare" in self.schemas else "view" if "view" in self.schemas else None
        if reader is None or self.turns_left() <= CHOICE_TURNS + 1:
            return
        room = min(AUTO_READ_PAGES, AUTO_READ_TOTAL - self.auto_read,
                   self.max_calls * (READ_PAGES_PER_CALL if reader == "compare" else 1))
        unread = [pid for pid, row in self.listed.items() if pid not in self.pages]
        unread.sort(key=lambda pid: (self.listed[pid].get("in_stock") is not True, not self.listed[pid].get("filtered"),
                                     _number(self.listed[pid].get("price")) or 0))
        unread = unread[:max(0, room)]
        if unread:
            self.read(unread)
            self.auto_read += len(unread)
            self.paths["auto_read_pages"] += len(unread)

    def read(self, pids: list[str]) -> None:
        # The pages of these products in one store turn, as many as the turn's calls hold.
        reader = "compare" if "compare" in self.schemas else "view" if "view" in self.schemas else None
        pids = [pid for pid in dict.fromkeys(pids) if pid not in self.pages]
        if reader is None or not pids or self.turns_left() <= CHOICE_TURNS + 1:
            return
        calls = []
        if reader == "compare":
            for start in range(0, len(pids), READ_PAGES_PER_CALL):
                args = {"product_ids": pids[start:start + READ_PAGES_PER_CALL]}
                if _argument_error(args, self.schemas["compare"]) is None:
                    calls.append({"name": "compare", "args": args})
        else:
            calls = [{"name": "view", "args": {"product_id": pid}} for pid in pids
                     if _argument_error({"product_id": pid}, self.schemas["view"]) is None]
        if calls:
            self.call(calls[: self.max_calls])

    def availability(self, pid: str, sku: str, max_price: float | None) -> str | None:
        # What the store reported about this SKU that rules it out: not read, not in stock, or over the price limit.
        page = self.pages.get(pid)
        if page is None:
            return "its page was not read"
        if not any(isinstance(v, dict) and v.get("sku") == sku for v in page.get("variants") or []):
            return "the page shows no such SKU"
        state = self.state_of(pid, sku)
        if state.get("in_stock") is not True:
            return "it was last reported out of stock"
        price = _number(state.get("price"))
        if max_price is not None and price is not None and price > max_price:
            return f"its price {price} is above the limit {max_price}"
        return None

    def unlike(self, pid: str, alt_pid: str) -> str | None:
        # What keeps the alternative from standing in for the choice: another shelf (the last element of the category
        # path).
        shelf = lambda p: str(((self.pages.get(p) or {}).get("category_path") or [None])[-1]).casefold()
        return None if shelf(pid) == shelf(alt_pid) else "its page is filed under another shelf than your choice"

    def rereads(self, pid: str, sku: str) -> list[dict]:
        # The calls that read a SKU again: its stock and its page.
        calls = []
        if "inspect_stock" in self.schemas and _argument_error({"product_id": pid, "sku": sku}, self.schemas["inspect_stock"]) is None:
            calls.append({"name": "inspect_stock", "args": {"product_id": pid, "sku": sku}})
        if "view" in self.schemas and _argument_error({"product_id": pid}, self.schemas["view"]) is None:
            calls.append({"name": "view", "args": {"product_id": pid}})
        return calls

    def read_again(self, pid: str, sku: str) -> None:
        calls = self.rereads(pid, sku)
        if calls:
            self.call(calls[: self.max_calls])

    def add(self, pid: str, sku: str, first: list[dict] | None = None) -> bool:
        # One store turn: the calls to make first (the other SKU leaving the cart), the addition, and the SKU read
        # again right after it. A store applies what it changes at the addition, so a read later in the same turn
        # shows the stock and the price as they stand after it.
        args = {"product_id": pid, "sku": sku}
        if "add_to_cart" not in self.schemas or _argument_error(args, self.schemas["add_to_cart"]):
            return False
        before = list(first or [])
        if len(before) + 1 > self.max_calls:  # a store that takes fewer calls in one turn: those calls get a turn of their own
            for start in range(0, len(before), self.max_calls):
                self.call(before[start:start + self.max_calls])
            before = []
        calls = before + [{"name": "add_to_cart", "args": args}] + self.rereads(pid, sku)
        observation = self.call(calls[: self.max_calls])[len(before)]
        if calls[self.max_calls:]:  # the reads that did not fit into that turn
            self.call(calls[self.max_calls:][: self.max_calls])
        return isinstance(observation.get("added"), dict) and not observation.get("error")

    def off_shelf(self, pid: str) -> str | None:
        # The page's own shelf (the last element of its category path) when no element of the path shares a word with
        # a shelf the model filtered by (a filtered shelf lists its sub-shelves too); None when the page is on a
        # filtered shelf or nothing was filtered.
        page = self.pages.get(pid) or {}
        path = page.get("category_path") if isinstance(page.get("category_path"), list) else []
        if not self.filtered_shelves or not path:
            return None
        elements = [_label_words(element) for element in path]
        for shelf in self.filtered_shelves:
            if _label_words(shelf) in elements:
                return None  # the filtered shelf or one of its sub-shelves
        return str(path[-1])

    def whole_in_kind(self, shelf: Any) -> bool:
        # A shelf label that stands whole in the kind of product the plan names: each of its words is a word of the
        # kind ("Boots" in "boot"; not "Rain Boots").
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        kind = set(_label_words(_without_measures(str(plan[0])))) if isinstance(plan, list) and plan else set()
        label = set(_label_words(shelf))
        return bool(label) and label <= kind

    def kind_shelf(self) -> str | None:
        # The shelf the store listed products for whose label stands whole in the kind of product the plan names, the
        # longest such label: the shelf the shopper named. None when the store listed none.
        found = [shelf for shelf in self.filtered_shelves if self.whole_in_kind(shelf)]
        return max(found, key=lambda shelf: len(_label_words(shelf))) if found else None

    def plan_requirement(self, named: Any) -> str | None:
        # The requirement of the plan that a claim's requirement names: the same phrase, else the one whose words
        # agree most (at least half of them, numbers carried), or None.
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        if not isinstance(plan, list) or not isinstance(named, str):
            return None
        fold = lambda v: " ".join(str(v).casefold().split())
        for requirement in plan:
            if isinstance(requirement, str) and fold(requirement) == fold(named):
                return requirement
        ws, best = _words(named), None
        for requirement in plan:
            if not isinstance(requirement, str):
                continue
            words = {w for w in _words(requirement) if len(w) >= 3 or any(ch.isdigit() for ch in w)}
            numbers = [_compact(n) for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:\s?[a-zA-Z]{1,8})?(?![\w])", requirement)]
            if words and (words <= ws or len(words & ws) * 2 >= len(words)) and all(n in _compact(named) for n in numbers):
                share = len(words & ws) / len(words)
                if best is None or share > best[0]:
                    best = (share, requirement)
        return best[1] if best else None

    def unproven(self, pid: str, sku: str, claims: Any) -> list[dict]:
        # The model's claims that the page does not bear out; a claim about a fact, the options, the title or the
        # category that names a requirement of the plan also when its value does not write that requirement (every
        # number of it, else a word of it as a part of its own: a word inside a code is not the word).
        out = []
        for c in claims if isinstance(claims, list) else []:
            if not isinstance(c, dict):
                continue
            kept = self.claims_for(pid, sku, [c])
            named = self.plan_requirement(c.get("requirement"))
            if not kept:
                out.append({**c, "why": "the page does not show this value under this field"
                                        + ("; " + self.where_written(pid, sku, named) if named else "")})
            elif named is None or c.get("field") in ("price", "in_stock"):
                continue
            elif c.get("field") == "category":
                # the page's own shelf names the kind; or, when the store listed no shelf whose label is the kind, the
                # page is filed under a shelf it listed for a filter
                listed = self.kind_shelf() is None and bool(self.filtered_shelves) and self.off_shelf(pid) is None
                if not _shelf_names_kind(str(kept[0]["value"]).split(" > ")[-1], _split(named)[1]) and not listed:
                    out.append({**c, "why": "the shelf this page is filed under is not the kind of product named; a kind of product is "
                                            "shown by the shelf, not by a title, so choose a SKU filed under that shelf"})
            elif not self.writes(kept[0]["value"], named, c["field"]):
                out.append({**c, "why": OTHER_WORDING + "; " + self.where_written(pid, sku, named)})
            elif c["field"] == "title" and not set(self.fields_writing(pid, sku, named)) - {"title"}:
                # the title alone writes it: when a fact of this page whose label names the attribute states another
                # value, the page's own fact says what the attribute is (a title also names uses, parts and sets)
                other = self.named_otherwise(pid, named)
                if other:
                    out.append({**c, "why": "the title names it, but a fact of this page states that attribute otherwise ("
                                            + other + "): choose a SKU whose facts or option text write it"})
            elif c["field"] not in OWN_FIELDS and c["field"] != "brand" and self.lists_values(kept[0]["value"], named):
                # a fact that lists several values names the requirement among others: a page whose fact is that
                # value alone states it more plainly (sent back once; the same commit again stands)
                out.append({**c, "why": LIST_WORDING + " (" + str(_cut(str(kept[0]["value"]), VALUE_CHARS))
                                        + "); a page whose fact is that value alone states it more plainly: look for one among "
                                          "the pages read or in the store, else make the same commit again"})
        return out

    def lists_values(self, written: Any, requirement: str) -> bool:
        # Whether a fact's text is a list of several values (parts set off by commas, semicolons or slashes) that
        # carries the requirement's value without being that value: a requirement of words only.
        value = _split(requirement)[1]
        if not value or _measures(value):
            return False
        fold = lambda v: " ".join(str(v).casefold().split())
        parts = [p.strip() for p in re.split(r"[,;/]", fold(written)) if p.strip()]
        return len(parts) >= 2 and fold(written) != fold(value) and self.carries(written, value)

    def fact_values(self, page: dict) -> dict:
        # The labelled values of a page: its facts (one whose value carries page markup aside), and its brand when no
        # fact holds one.
        facts = page.get("facts") if isinstance(page.get("facts"), dict) else {}
        values = {str(k): v for k, v in facts.items()
                  if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip() and not _markup(v)}
        if isinstance(page.get("brand"), str) and page["brand"].strip():
            values.setdefault("brand", page["brand"])
        return values

    def fields_writing(self, pid: str, sku: str, requirement: str) -> list[str]:
        # The fields of a read page whose values write a requirement: the SKU's option text, the facts, the title.
        page = self.pages.get(pid) or {}
        variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and v.get("sku") == sku), {})
        fields = {"options": variant.get("options"), **self.fact_values(page), "title": page.get("title")}
        return [name for name, value in fields.items()
                if isinstance(value, (str, int, float)) and self.writes(value, requirement, name)]

    def where_written(self, pid: str, sku: str, requirement: str) -> str:
        # Where the page writes the requirement, for the model's next claim.
        found = self.fields_writing(pid, sku, requirement)
        return ("the page writes it under: " + ", ".join(found[:6])) if found else (
            "no field of this page writes it: choose another SKU (find_pages lists the read SKUs whose pages carry a value)")

    def affirms(self, requirement: str) -> bool:
        # A requirement whose value only affirms its attribute: the shopper wrote every word of the attribute (a
        # property the item has to have) and no word of the value, which is the reading's own word for having it and
        # no number. The quantity the plan ranks by is no such property.
        attribute, value = _split(requirement)
        if not attribute or _measures(value):
            return False
        ranking = self.plan_read.get("ranking") if isinstance(self.plan_read, dict) else None
        quantity = str(ranking.get("quantity") or "") if isinstance(ranking, dict) else ""
        known = (requirement, len(self.lines), quantity)
        if known not in self.affirmed:
            said = {_stem(w) for w in _words(self.query + " " + " ".join(line["content"] for line in self.lines))}
            named, worded = {_stem(w) for w in _words(attribute)}, {_stem(w) for w in _words(value)}
            ranked = {_stem(w) for w in _words(quantity)}
            self.affirmed[known] = bool(named) and bool(worded) and named <= said and not worded & said and named != ranked
        return self.affirmed[known]

    def writes(self, text: Any, requirement: str, field: str) -> bool:
        # A field of a page (its option text, a fact, the title) writes a requirement: its value, as any text does
        # (names_requirement); or, for a requirement whose value only affirms its attribute, the attribute's words.
        return self.names_requirement(text, requirement, field) or (
            self.affirms(requirement) and self.carries(text, _split(requirement)[0], field))

    def names_requirement(self, text: Any, requirement: str, field: str = "") -> bool:
        # The text writes the value of a requirement ("attribute: value", or all value): every number of it (the same
        # number, whatever the spelling of its unit; a number written without a unit only under a label that carries
        # the unit); without a number, the
        # value as a part of the text, one of its words as a part of its own (a word inside a hyphenated code is not
        # the word), or two of its words in a row with hyphens read as spaces. Words of the value that name the
        # attribute, or that the fact's label carries, are left aside when others remain.
        attribute, value = _split(requirement)
        if field == "brand":  # a brand is a name: it writes a requirement only as a whole
            return self.whole(text, value, field)
        numbers = _measures(value)
        if numbers:
            written = [(n, unit or next((u for u in (want[1] for want in numbers) if u and u in str(field).casefold()), ""))
                       for n, unit in _measures(text)]
            return all(any(_same_measure(want, have) and (have[1] or not want[1]) for have in written) for want in numbers)
        fold = lambda v: " ".join(str(v).casefold().split())
        if fold(value) and re.search(r"(?<![\w.-])" + re.escape(fold(value)) + r"(?![\w-])", fold(text)):
            return True
        wanted = [_stem(w) for w in re.findall(r"[a-z0-9]+", value.casefold())]
        aside = {_stem(w) for w in _words(attribute if field in OWN_FIELDS or not field else field)}
        wanted = [w for w in wanted if w not in aside] or wanted
        parts = {_stem(w) for w in _tokens(text)}
        if any(w in parts for w in wanted):
            return True
        flat = [_stem(w) for w in re.findall(r"[a-z0-9]+", str(text).casefold())]
        return any(flat[i:i + 2] == wanted[j:j + 2] for j in range(len(wanted) - 1) for i in range(len(flat) - 1))

    def first_stated(self, requirement: str) -> bool:
        # Whether the shopper states a requirement of their own accord: the words and numbers of its value stand in
        # the first message and the lines the shopper said unasked (a changed requirement) at least as often as in
        # the answers to the controller's question, and it is not one of two wishes the shopper doubted. A store
        # judges the first SKU in the cart by the request as it stands, not by what an answer added to it.
        if any(self.names_requirement(wish, requirement) for wish in self.doubted):
            return False  # one of two doubted wishes: a requirement only by the shopper's later answer
        value = _split(requirement)[1]
        words = {_stem(w) for w in _words(value) if not any(ch.isdigit() for ch in w)}
        numbers = _measures(value)

        def carried(text: str) -> int:
            have = {_stem(w) for w in _words(text)}
            found = _measures(text)
            return len(words & have) + sum(1 for n in numbers if any(_same_measure(n, h) for h in found))

        answers = " ".join(line["content"] for line in self.lines if self.answer_line(line))
        unasked = " ".join(line["content"] for line in self.lines if not self.answer_line(line))
        return carried(self.query + " " + unasked) >= carried(answers)

    def first_claims(self, pid: str, sku: str, claims: Any) -> tuple[list[dict], list[str]]:
        # What an alternative must bear out, so that it meets the request as the shopper first stated it: the claims
        # about the choice that name a requirement the first message states, and, apart, those requirements whose
        # claim the choice's page does not bear out or does not write (a commit of the closing phase) or that have no
        # claim: the alternative's own page has to write these. A fact claim as it is; an option or title claim by
        # the words and numbers of that requirement which its value writes (a title shares other words with a
        # request). Price, stock and category claims are left out, and so is a claim about the kind of product on
        # any other field: the alternative is checked for those itself (it stands on the choice's shelf). A claim
        # that names a preference of the plan counts as one that names a requirement (a plan may hold as a
        # preference what the request states as a requirement); one that names neither (the ranked quantity, say)
        # asks nothing of the alternative.
        first, open_requirements, named_requirements = [], [], []
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        kind = plan[0] if isinstance(plan, list) and plan else None
        for c in claims if isinstance(claims, list) else []:
            if not isinstance(c, dict) or not isinstance(c.get("field"), str) or c["field"] in ("price", "in_stock", "category"):
                continue
            requirement = self.plan_requirement(c.get("requirement"))
            if requirement is None and isinstance(c.get("requirement"), str):
                fold = " ".join(c["requirement"].casefold().split())
                requirement = next((p for p in self.plan_preferences() if " ".join(p.casefold().split()) == fold), None)
            if not isinstance(requirement, str) or not requirement.strip() or requirement == kind:
                continue
            named_requirements.append(requirement)
            if not self.first_stated(requirement):
                continue
            kept = self.claims_for(pid, sku, [c])
            # (a reading the model stood by rests on the value its claim shows: the alternative has to show it too)
            if not kept or not (self.writes(kept[0]["value"], requirement, c["field"]) or (pid, sku, requirement) in self.readings):
                open_requirements.append(requirement)
                continue
            if c["field"] not in ("options", "title"):
                first.append({"field": c["field"], "value": c.get("value")})
                continue
            value = _split(requirement)[1]
            named, numbers = {_stem(w) for w in _words(value)}, _measures(value)
            parts = [w for w in _tokens(c.get("value")) if (_stem(w) in named and not any(ch.isdigit() for ch in w))
                     or any(_same_measure(n, m) for n in numbers for m in _measures(w))]
            # a value that does not write the requirement's own words as parts of its own is kept whole
            first.append({"field": c["field"], "value": " ".join(sorted(set(parts))) if parts else c.get("value")})
        open_requirements += [r for r in self.plan_attributes() if r not in named_requirements and self.first_stated(r)]
        return first, list(dict.fromkeys(open_requirements))

    def same_listing(self, pid: str, other: str) -> bool:
        # Two product pages that are variants of one listing: they carry the same title.
        fold = lambda p: " ".join(str((self.pages.get(p) or {}).get("title") or "").casefold().split())
        return bool(fold(pid)) and fold(pid) == fold(other)

    def pick_alternative(self, pid: str, sku: str, claims: Any, max_price: float | None) -> tuple[str, str] | None:
        # Another read SKU that stands in for the choice: in stock, within the price limit, on the same shelf, and
        # bearing out the claims the model made about the choice that the first message states (first_claims).
        # First the SKUs whose page also writes what the plan requires beyond the first message (the fewest of those
        # requirements left unwritten); then the SKUs whose option text or facts write every requirement the shopper stated of their own accord (a
        # title names all the variants of a listing); among them those whose own option text writes the stated values
        # (a fact holds for the whole listing and may not hold for the variant), the plainer statement first: the
        # part of the option text that writes a value with the fewest other words; then a variant of the same
        # listing, then the same brand, then the SKU whose facts share the most values with the choice's, then the
        # one read first.
        claims, open_requirements = self.first_claims(pid, sku, claims)
        stated = [r for r in self.plan_attributes() if self.first_stated(r)]
        # what the plan requires beyond the first message (an answer of the shopper, a later line): a SKU whose own
        # page writes these as well meets the whole request and comes before one that does not
        later = [r for r in self.plan_attributes() if r not in stated]
        page = self.pages.get(pid) or {}
        fold = lambda v: " ".join(str(v).casefold().split())
        values = {fold(v) for v in self.fact_values(page).values()}
        rows = []
        for other_pid, other in self.pages.items():
            if self.unlike(pid, other_pid):
                continue
            shared = len(values & {fold(v) for v in self.fact_values(other).values()})
            for variant in other.get("variants") or []:
                if not isinstance(variant, dict) or not isinstance(variant.get("sku"), str):
                    continue
                key = (other_pid, variant["sku"])
                if key == (pid, sku) or self.availability(*key, max_price):
                    continue
                # (what the choice's title writes may stand in the alternative's option text instead)
                if not all(self.claims_for(*key, [c]) or (c["field"] == "title" and self.claims_for(*key, [{**c, "field": "options"}]))
                           for c in claims):
                    continue
                if not all(self.fields_writing(*key, requirement) for requirement in open_requirements):
                    continue
                by_title = sum(1 for r in stated if set(self.fields_writing(*key, r)) <= {"title"})
                beside = [self.words_beside(variant.get("options"), _split(r)[1]) for r in stated]
                rows.append((sum(1 for r in later if not self.fields_writing(*key, r)),
                             by_title, sum(1 for n in beside if n is None), sum(n for n in beside if n is not None),
                             not self.same_listing(pid, other_pid), fold(other.get("brand")) != fold(page.get("brand")),
                             -shared, len(rows), key))
        return min(rows)[8] if rows else None

    def consult(self, instruction: str, tool: dict, messages: list[dict] | None = None) -> dict | None:
        # One model reading beside the decisions: the conversation with one instruction, answered through one tool.
        # Its arguments, or None when no usable answer came; nothing of it enters the conversation.
        name = tool["function"]["name"]
        payload = {"model": self.model,
                   "messages": (self.messages if messages is None else messages) + [{"role": "user", "content": instruction}],
                   "tools": [deepcopy(tool)], "tool_choice": {"type": "function", "function": {"name": name}}}
        result = self.request(payload, "/inference/chat/completions")
        choice = (result.get("choices") or [{}])[0] if isinstance(result, dict) else {}
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = [item for item in (message or {}).get("tool_calls") or [] if isinstance(item, dict)
                 and isinstance(item.get("function"), dict) and item["function"].get("name") == name]
        if not calls:
            return None
        try:
            args = json.loads(calls[0]["function"].get("arguments") or "{}")
        except (ValueError, TypeError):
            return None
        return args if isinstance(args, dict) and _argument_error(args, tool["function"]["parameters"]) is None else None

    def second_use(self) -> None:
        # A request may name two uses of the purchase, and a shopper who does may hold a requirement for each; the
        # answer to the question about the use speaks of one. The model lists the uses the first message names; when
        # two of them stand in the message itself (every word of each, and neither a part of the other) it writes,
        # with the listings in view, one question about the one attribute the other use depends on, and the
        # controller asks it, once and before any cart addition. The use the answer speaks of is the one that shares
        # the most words with it, else the first-named one.
        schema = self.schemas.get("message")
        field = next((f for f in ("content", "text") if isinstance((schema or {}).get("properties"), dict)
                      and f in schema["properties"]), None)
        if (field is None or self.done or self.stop_reason is not None or self.added or self.paths["second_use_readings"]
                or not (self.paths["opening_questions"] or self.paths["use_questions"]) or self.paths["priority_questions"]
                or self.use_unanswered or not any(self.answer_line(line) for line in self.lines)
                or self.turns_left() <= PURCHASE_TURNS + CLOSING_TURNS + 2 * VARIANT_TURNS):
            return
        self.paths["second_use_readings"] += 1
        args = self.consult(USES_NAMED, USES_TOOL) or {}
        written = _words(self.query)
        uses = [use for use in args.get("uses") or [] if isinstance(use, str) and _words(use) and _words(use) <= written]
        uses += [use for use in (self.uses or []) if use not in uses]  # (the uses read for the attribute question count too)
        apart = [use for use in uses if not any(other != use and _words(use) <= _words(other) for other in uses)]
        # the use the shopper's answer speaks of: the one that shares the most words (of four letters or more) with the
        # answer and with the model's naming of it; when none stands out, the first-named one
        heard = {word for word in _words(args.get("answered_use")) | _words(" ".join(line["content"] for line in self.lines
                                                                                    if self.answer_line(line))) if len(word) >= 4}
        shared = [len(_words(use) & heard) for use in apart]
        spoken = apart[shared.index(max(shared))] if apart and shared.count(max(shared)) == 1 else (apart[0] if apart else None)
        others = [use for use in apart if use != spoken]
        record = {"uses": args.get("uses"), "answered_use": args.get("answered_use"), "spoken_of": spoken, "question": None, "asked": False}
        self.dialogue.append({"role": "assistant", "content": "Controller second use", "controller_second_use": record})
        if len(apart) < 2 or not others:
            return  # one use
        plan = self.plan_read if isinstance(self.plan_read, dict) else {}
        ranking = plan.get("ranking") if isinstance(plan.get("ranking"), dict) else {}
        held = [entry for entry in self.plan_attributes() if isinstance(entry, str)]
        held += [str(ranking["quantity"])] if str(ranking.get("quantity") or "").strip() else []
        # the listings read are put in view for this reading only: the model's first decision sees them as before
        fetched, kept = list(self.fetched), len(self.messages)
        self.show_pages()
        seen = list(self.messages)
        del self.messages[kept:]
        self.fetched = fetched
        args = self.consult(SECOND_USE.format(use=_json(others[-1]), held=_json(held)), USE_QUESTION_TOOL, seen) or {}
        attribute = " ".join(str(args.get("attribute") or "").split())
        record["question"] = args.get("question")
        if not attribute or len(attribute) > VALUE_CHARS or not _words(attribute):
            return
        question = self.attribute_question(args.get("question"), others[-1], attribute)
        record["question"] = question
        record["asked"] = True
        self.use_questions.append((others[-1], attribute))
        self.before_speaking()
        self.call([{"name": "message", "args": {field: " ".join(question.split())}}])
        self.asked_turns.append(self.turn)  # what the shopper says to it is an answer
        self.paths["second_use_questions"] += 1

    def words_beside(self, options: Any, value: str) -> int | None:
        # How plainly an option text writes a value: the fewest other words in a part of it (the parts a store
        # separates by a slash) that carries the value; None when no part carries it.
        counts = [len(re.findall(r"[a-z0-9]+", part.casefold())) - len(re.findall(r"[a-z0-9]+", value.casefold()))
                  for part in str(options or "").split("/") if self.carries(part, value, "options")]
        return max(0, min(counts)) if counts else None

    def carries(self, text: Any, value: str, field: str = "") -> bool:
        # The text carries a value: every number of it with its unit; else every word of it as a part of the text's
        # own, or its words in a row with hyphens read as spaces.
        if field == "brand":  # a brand is a name: it carries a value only as a whole
            return self.whole(text, value, field)
        numbers = _measures(value)
        if numbers:
            written = [(n, unit or next((u for u in (want[1] for want in numbers) if u and u in str(field).casefold()), ""))
                       for n, unit in _measures(text)]
            return all(any(_same_measure(want, have) and (have[1] or not want[1]) for have in written) for want in numbers)
        wanted = [_stem(w) for w in re.findall(r"[a-z0-9]+", str(value).casefold())]
        if field and field not in OWN_FIELDS:  # a word that the fact's own label carries is left aside when others remain
            label = {_stem(w) for w in _words(field)}
            wanted = [w for w in wanted if w not in label] or wanted
        if not wanted:
            return False
        if all(w in {_stem(part) for part in _tokens(text)} for w in wanted):
            return True
        flat = [_stem(w) for w in re.findall(r"[a-z0-9]+", str(text).casefold())]
        return len(wanted) >= 2 and any(flat[i:i + len(wanted)] == wanted for i in range(len(flat) - len(wanted) + 1))

    def whole(self, written: Any, value: str, field: str) -> bool:
        # The field's text is the value and nothing else: the same words in the same order (those of the fact's own
        # label aside), or, for a number, that one number with its unit.
        text = [_stem(w) for w in re.findall(r"[a-z0-9]+", str(written).casefold())]
        wanted = [_stem(w) for w in re.findall(r"[a-z0-9]+", str(value).casefold())]
        label = {_stem(w) for w in _words(field)}
        trim = lambda words: [w for w in words if w not in label] or words
        if trim(text) == trim(wanted):
            return True
        numbers, found = _measures(value), _measures(written)
        return (len(numbers) == 1 and len(found) == 1 and len(str(written).split()) <= 2
                and _same_measure(numbers[0], (found[0][0], found[0][1] or numbers[0][1])))

    def weighed(self) -> list[str]:
        # The requirements of the plan whose statement by a page is weighed before a purchase: each one worded
        # "attribute: value" (a value in words or a number with its unit), whether the first message states it, an
        # answer added it or the shopper chose it of two wishes; not the kind of product, stock or the price limit.
        return [requirement for requirement in self.plan_attributes() if all(_split(requirement))]

    def stated_otherwise(self, pid: str, requirement: str) -> str | None:
        # A fact of a read page whose label carries every word of the requirement's attribute and whose text does
        # not carry the value, as "label: text"; None when the page has no such fact. A page that reads the value
        # under one such label and another value under a second one (a jug of glass, a jar of plastic) states the
        # attribute both ways: the fact that reads otherwise is returned all the same.
        attribute, value = _split(requirement)
        named = {_stem(w) for w in _words(attribute)}
        against = [(name, written) for name, written in self.fact_values(self.pages.get(pid) or {}).items()
                   if named and named <= {_stem(w) for w in _words(name)} and not self.carries(written, value, name)]
        if not against:
            return None
        return f"{against[0][0]}: {_cut(str(against[0][1]), VALUE_CHARS)}"

    def named_otherwise(self, pid: str, requirement: str) -> str | None:
        # stated_otherwise for a requirement of words, held to the labels that name the attribute beyond doubt: an
        # attribute of one word is named by a label whose other words (of three letters or more) are words of the kind
        # of product (the type of an electric toothbrush by "electric_toothbrush_type", not by "battery_type"). A
        # requirement with a number is left to the reading of numbers.
        attribute, value = _split(requirement)
        named = {_stem(w) for w in _words(attribute)}
        if not named or _measures(value):
            return None
        plan = self.plan_read if isinstance(self.plan_read, dict) else {}
        entries = plan.get("requirements")
        kind = {_stem(w) for w in _words(entries[0])} if isinstance(entries, list) and entries else set()
        for name, written in self.fact_values(self.pages.get(pid) or {}).items():
            label = {_stem(w) for w in _words(name)}
            if not named <= label or self.carries(written, value, name):
                continue
            if len(named) == 1 and not {w for w in label - named if len(w) >= 3} <= kind:
                continue
            return f"{name}: {_cut(str(written), VALUE_CHARS)}"
        return None

    def plain_field(self, pid: str, sku: str, requirement: str) -> str | None:
        # The field of a read page that states a requirement as a value of its own: a fact whose whole text is the
        # value, unless a fact under the attribute's label reads otherwise; else the SKU's option text when it
        # carries the value and no fact under the attribute's label reads otherwise (a page that reads "material:
        # plastic" is not shown to be gold by a gold in its option text). A value that stands only in the title or
        # inside a longer fact text is not stated plainly: titles and long texts also name uses, accessories and
        # other products.
        value = _split(requirement)[1]
        page = self.pages.get(pid) or {}
        if self.stated_otherwise(pid, requirement):
            return None
        for name, written in self.fact_values(page).items():
            if self.whole(written, value, name):
                return name
        variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and v.get("sku") == sku), {})
        options = variant.get("options")
        return "options" if isinstance(options, str) and self.carries(options, value, "options") else None

    def plainer(self, pid: str, sku: str, max_price: float | None) -> dict:
        # For a commit whose page does not state plainly a requirement of the plan: those
        # requirements with what the page shows instead, and the SKUs in view that carry every value of the plan
        # and state each of them plainly, with the field that does; those whose page also carries more of the
        # plan's preferences first (each row names them), then in the order of a lookup. Empty when the choice
        # states them plainly or no such SKU is in view.
        faint = [r for r in self.weighed() if self.plain_field(pid, sku, r) is None]
        if not faint:
            return {}
        rows = []
        wishes = self.plan_preferences()
        for other, other_sku in self.carrying_all(max_price):
            fields = {r: self.plain_field(other, other_sku, r) for r in faint}
            if (other, other_sku) != (pid, sku) and all(fields.values()):
                variant = next((v for v in (self.pages.get(other) or {}).get("variants") or []
                                if isinstance(v, dict) and v.get("sku") == other_sku), {})
                liked = [wish for wish in wishes if self.fields_writing(other, other_sku, wish)]
                rows.append((-len(liked), len(rows), {
                    "product_id": other, "sku": other_sku, "price": self.state_of(other, other_sku).get("price"),
                    "options": _cut(variant.get("options"), VALUE_CHARS), "states_it_under": fields,
                    **({"also_carries": liked} if liked else {})}))
        if not rows:
            return {}
        rows.sort(key=lambda row: row[:2])
        shown = {r: self.stated_otherwise(pid, r) or "only in its title or inside a longer text, or not at all" for r in faint}
        return {"requirements": shown, "skus": [row[2] for row in rows[:PLAIN_SHOWN]]}

    def lookup_values(self, values: Any) -> list[str]:
        # The values of a lookup as the pages are searched for them: each without its attribute, once; a value that
        # names only the kind of product or the ranked quantity is left out (every listed SKU stands on the kind's
        # shelf, and a ranking is no value a page carries), and so is stock (every listed SKU can be ordered).
        plan = self.plan_read if isinstance(self.plan_read, dict) else {}
        kind = {_stem(w) for w in _words(plan["requirements"][0])} if isinstance(plan.get("requirements"), list) and plan["requirements"] else set()
        ranked = ({_stem(w) for w in _words(plan["ranking"].get("quantity"))} if isinstance(plan.get("ranking"), dict) else set())
        aside = lambda v: any(group and {_stem(w) for w in _words(v)} <= group for group in (kind, ranked))
        return list(dict.fromkeys(" ".join(_split(v)[1].split()) for v in values if isinstance(values, list)
                                  and isinstance(v, str) and v.strip() and not _words("in_stock") <= _words(v) and not aside(v)))

    def carrying(self, values: list[str], limit: float | None) -> tuple[list[dict], int]:
        # The SKUs among the pages read that can be ordered (in stock, within the limit, under the shelf that is the
        # kind of product when the store listed it) and whose page carries every one of the values, each with the
        # fields that carry each value: the SKUs whose option text or facts carry the values before those where only
        # the title carries one, the SKUs with more such fields first, then in the order read; last the SKUs for
        # which a value stands only inside a longer fact text (a fact that lists several values says less about the
        # SKU than its option text, a fact of its own or the title). With them, the number of pages looked at.
        bound = self.kind_shelf()
        rows, searched = [], 0
        ranking = self.plan_read.get("ranking") if isinstance(self.plan_read, dict) else None
        quantity = {_stem(w) for w in _words(ranking.get("quantity"))} if isinstance(ranking, dict) and isinstance(ranking.get("quantity"), str) else set()
        for pid, page in self.pages.items():
            if bound is not None and not self.under(pid, bound):
                continue
            searched += 1
            path = page.get("category_path") if isinstance(page.get("category_path"), list) else []
            # the facts of the page whose label names the quantity the plan ranks by, as the page writes them
            ranked = {name: _cut(written, VALUE_CHARS) for name, written in self.fact_values(page).items()
                      if quantity and quantity & {_stem(w) for w in _words(name)}} if quantity else {}
            for variant in page.get("variants") or []:
                sku = variant.get("sku") if isinstance(variant, dict) else None
                if not isinstance(sku, str) or self.availability(pid, sku, limit):
                    continue
                fields = {"options": variant.get("options"), **self.fact_values(page), "title": page.get("title")}
                where = {value: [name for name, written in fields.items() if isinstance(written, (str, int, float))
                                 and self.carries(written, value, name)][:3] for value in values}
                if values and all(where.values()):
                    by_title = sum(1 for names in where.values() if names == ["title"])
                    own = sum(1 for names in where.values() for name in names if name != "title")
                    apart = all(any(name == "options" or (name != "title" and self.whole(fields[name], value, name))
                                    for name, written in fields.items() if isinstance(written, (str, int, float))
                                    and self.carries(written, value, name)) for value in values)
                    rows.append((by_title == 0 and not apart, by_title, -own, len(rows), {
                        "product_id": pid, "sku": sku, "price": self.state_of(pid, sku).get("price"),
                        "options": _cut(variant.get("options"), VALUE_CHARS), "title": _cut(page.get("title"), 70),
                        **({} if bound is not None else {"shelf": str(path[-1]) if path else None}), "carries": where,
                        **({"ranked_by": ranked} if ranked else {})}))
        rows.sort(key=lambda row: row[:4])
        return [row[4] for row in rows], searched

    def carrying_all(self, max_price: float | None) -> list[tuple[str, str]]:
        # The controller's own lookup with the values the plan requires: the SKUs among the pages read that carry
        # every one of them and can be ordered within the limit, on the shelf that is the kind of product (else on a
        # shelf the store listed for a filter), in the order a lookup lists them.
        values = self.lookup_values(self.plan_values())
        if not values:
            return []
        rows, _ = self.carrying(values, max_price)
        return [(row["product_id"], row["sku"]) for row in rows
                if not self.off_shelf(row["product_id"])
                and not any(u["product_id"] == row["product_id"] and u["sku"] == row["sku"] for u in self.unavailable)]

    def store_units(self, unit: str, shorter: bool = False) -> list[str]:
        # Other forms of a unit as the pages read write it after a number, the most frequent first: the shorter forms
        # of a word (pieces: pcs, pc; watts: w) and, unless only shorter forms are asked for, its plural or singular
        # (pc, pcs). A short symbol is not spelled out (m is no month), and two symbols are two units (m, mm).
        unit = unit.casefold()
        seen: Counter = Counter()
        for page in self.pages.values():
            texts = [page.get("title")] + [v.get("options") for v in page.get("variants") or [] if isinstance(v, dict)]
            for text in texts + list(self.fact_values(page).values()):
                seen.update(other for _, other in _measures(text) if other and other != unit)

        def short_for(short: str, word: str) -> bool:  # its letters stand in the word, in order, from the first
            rest = iter(word)
            return len(word) >= 4 and len(short) < len(word) and short[0] == word[0] and all(letter in rest for letter in short)

        return [other for other, _ in seen.most_common()
                if short_for(other, unit) or (not shorter and other.rstrip("s") == unit.rstrip("s"))][:STORE_SPELLINGS]

    def store_spellings(self, text: str, shorter: bool = False) -> list[str]:
        # The text with a number's unit in the forms the store's pages use, joined to the number ("3 pieces" as
        # "3pcs"): one text for each form.
        out = []
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s?([a-zA-Z]{1,12})(?![\w])", text):
            out += [text[:m.start()] + m.group(1) + other + text[m.end():] for other in self.store_units(m.group(2), shorter)]
        return list(dict.fromkeys(out))

    def joined(self, text: str) -> str:
        # The text with each number joined to the unit after it ("1 pcs 80 w" as "1pcs 80w") where the pages read
        # write that unit joined to a number: listings write a quantity as one word, and a store's search matches
        # words as they are written. A word no page writes so stays apart ("3 seater", "12 and").
        glued: set[str] = set()
        for page in self.pages.values():
            texts = [page.get("title")] + [v.get("options") for v in page.get("variants") or [] if isinstance(v, dict)]
            for written in texts + list(self.fact_values(page).values()):
                glued.update(m.group(1).casefold() for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?([a-zA-Z]{1,8})(?![\w])", str(written or "")))
        return re.sub(r"(?<![\w.])(\d+(?:\.\d+)?) ([a-zA-Z]{1,8})(?![\w])",
                      lambda m: m.group(1) + m.group(2) if m.group(2).casefold() in glued else m.group(0), text)

    def listing_queries(self) -> list[str]:
        # A store's search may show one variant of each listing: the variant that carries the plan's values can stand
        # behind the one shown, and other sellers may list the same product. The titles of the listings in view
        # whose pages carry the most of the plan's values (at least one), each with those values in both spellings
        # of a number and its unit, bring such SKUs to the front: the listings that carry the most values first,
        # then in the order read.
        values = self.lookup_values(self.plan_values())
        if not values:
            return []
        bound = self.kind_shelf()
        listings: dict[str, list] = {}  # the first words of a title -> [the values its pages carry, those words]
        for pid, page in self.pages.items():
            if (bound is not None and not self.under(pid, bound)) or (bound is None and self.off_shelf(pid)):
                continue
            title = " ".join(str(page.get("title") or "").split()[:VARIANT_QUERY_WORDS])
            if not title:
                continue
            entry = listings.setdefault(title.casefold(), [set(), title])
            for variant in page.get("variants") or []:
                fields = {"options": variant.get("options") if isinstance(variant, dict) else None,
                          **self.fact_values(page), "title": page.get("title")}
                entry[0].update(value for value in values if any(isinstance(written, (str, int, float))
                                                                 and self.carries(written, value, name)
                                                                 for name, written in fields.items()))
        ranked = sorted((entry for entry in listings.values() if entry[0]), key=lambda entry: -len(entry[0]))
        joined = " ".join(values)
        spelled = " ".join(dict.fromkeys(" ".join([joined] + _query_variants(joined) + self.store_spellings(joined)).split()))
        return [f"{entry[1]} {spelled}" for entry in ranked[:LISTING_TITLES]]

    def others_carrying(self, pid: str, sku: str, max_price: float | None) -> dict:
        # For a commit that is sent back: the SKUs whose own pages carry every value the plan requires, when the
        # choice is not one of them.
        rows = self.carrying_all(max_price)
        if not rows or (pid, sku) in rows:
            return {}
        return {"skus_that_carry_every_value_of_your_plan": [{"product_id": p, "sku": s} for p, s in rows[:3]]}

    def find_pages(self, args: dict) -> dict:
        # The model's own lookup among the pages already read (carrying): no store turn is used.
        error = _argument_error(args, FIND_TOOL["function"]["parameters"])
        if error:
            return {"not_run": error}
        limit = _number(self.plan_read.get("price_limit")) if isinstance(self.plan_read, dict) else None
        rows, searched = self.carrying(self.lookup_values(args["values"]), limit)
        self.paths["page_lookups"] += 1
        self.looked_for = (self.plan_key(), self.ceiling(None))
        ranking = self.plan_read.get("ranking") if isinstance(self.plan_read, dict) else None
        ranked = ({"ranking": f"your plan ranks by {ranking['quantity']} ({ranking.get('direction')}): compare the listed SKUs "
                              f"that meet every requirement by it before you commit; a SKU whose page states no {ranking['quantity']} "
                              "ranks behind every SKU that states one"}
                  if len(rows) > 1 and isinstance(ranking, dict) and isinstance(ranking.get("quantity"), str) and ranking["quantity"].strip()
                  else {})
        return {"found": rows[:FOUND_SHOWN], **({"more": len(rows) - FOUND_SHOWN} if len(rows) > FOUND_SHOWN else {}),
                "pages_searched": searched, **ranked,
                **({} if rows else {"note": "no read page carries all of these values: try one value less or another spelling of "
                                            "a value, or fetch more listings"})}

    def spare(self) -> int:
        # The store turns kept back for the purchase: the whole of it (another SKU first, then the choice) with the
        # turns of the closing phase; once a SKU has gone into the cart first, the choice alone (its addition, its
        # re-read, the order) and one turn more.
        return CHOICE_TURNS + 1 if (self.held is not None or self.added) else PURCHASE_TURNS + CLOSING_TURNS

    def closing(self) -> bool:
        # Few store turns or decisions are left: the model is told to commit, and its choice is ordered with the
        # claims its page bears out.
        return self.turns_left() <= self.spare() + 2 or DECISION_LIMIT - self.decisions <= CLOSING_DECISIONS

    def look_further(self) -> None:
        # Once, before the model's first decision: one store turn that widens the view of the shelf
        # and one that reads what it lists. The titles of the listings that carry the most of the plan's values,
        # searched with those values (listing_queries); the shelf filtered by each of the brands most frequent on it
        # (a filter lists only the first products of a shelf; a brand names another part of it); and searches of
        # the values the plan asks for with each of the words most frequent in the titles on the shelf (a listing
        # that states a value as a fact may carry none of it in its title; a value whose unit the pages read write
        # in another form goes out in that form).
        if (self.widened or self.decisions or self.added or self.held is not None or not self.filtered_shelves
                or "search" not in self.schemas or self.turns_left() <= PURCHASE_TURNS + CLOSING_TURNS + VARIANT_TURNS):
            return
        self.widened = True
        shelf = self.kind_shelf() or self.filtered_shelves[0]
        limit = _number(self.plan_read.get("price_limit")) if isinstance(self.plan_read, dict) else None
        values = self.plan_values()
        listings = {}  # each listing of the shelf once: its title -> its page
        for page in self.pages.values():
            path = page.get("category_path") if isinstance(page.get("category_path"), list) else []
            if path and _label_words(path[-1]) == _label_words(shelf):
                listings.setdefault(" ".join(str(page.get("title") or "").casefold().split()), page)
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        used = {_stem(w) for w in _words(" ".join(values + [str(plan[0]) if isinstance(plan, list) and plan else ""]))}
        words: Counter = Counter()
        for title in listings:
            words.update({w for w in re.findall(r"[a-z]+", title) if len(w) >= 4 and _stem(w) not in used})
        brands = Counter(str(page["brand"]) for page in listings.values() if isinstance(page.get("brand"), str) and page["brand"].strip())
        calls = []
        known = self.schemas["search"].get("properties") if isinstance(self.schemas["search"].get("properties"), dict) else {}
        for query in self.listing_queries():
            args = {k: v for k, v in {"query": query, "in_stock": True, "k": RESULT_ROWS}.items() if k in known}
            if _argument_error(args, self.schemas["search"]) is None and f"search {_json(args)}" not in self.calls_made:
                calls.append({"name": "search", "args": args})
        self.paths["listing_searches"] += len(calls)
        sieve = self.schemas.get("filter") if isinstance(self.schemas.get("filter"), dict) else None
        if sieve and isinstance(sieve.get("properties"), dict) and "brand" in sieve["properties"]:
            for brand, _ in brands.most_common(BRAND_TRIES):
                args = {k: v for k, v in {"category": shelf, "brand": brand, "in_stock": True, "k": RESULT_ROWS,
                                          "max_price": limit}.items() if k in sieve["properties"] and v is not None}
                if _argument_error(args, sieve) is None and f"filter {_json(args)}" not in self.calls_made:
                    calls.append({"name": "filter", "args": args})
        # (a unit the plan spells out goes out in the shorter form the store's pages use, and every number joined to
        # its unit, as listings write a quantity; when the store's form is another than the plan's, the kind of product
        # is searched with the values in that form, and those values alone)
        stores = self.store_spellings(" ".join(values), True)
        written = self.joined((stores or [" ".join(values)])[0])
        kind = " ".join(_without_measures(str(plan[0])).split()) if isinstance(plan, list) and plan else ""
        wanted = ([f"{kind} {written}".strip(), written] if stores else []) + [f"{written} {word}" for word, _ in words.most_common(SHELF_WORDS)]
        for query in wanted if values else []:
            args = {k: v for k, v in {"query": query, "in_stock": True, "k": RESULT_ROWS}.items() if k in known}
            if (_argument_error(args, self.schemas["search"]) is None and f"search {_json(args)}" not in self.calls_made
                    and not any(c["name"] == "search" and c["args"] == args for c in calls)):
                calls.append({"name": "search", "args": args})
        if not calls:
            return
        calls = calls[: self.max_calls]
        observations = self.call(calls)
        listed = []
        for c, observation in zip(calls, observations):
            self.result_view(c, observation)
            listed += [row.get("product_id") for row in observation.get("results") or [] if isinstance(row, dict)]
        self.read([p for p in listed if isinstance(p, str)])
        self.paths["wider_views"] += 1
        self.notes.append(f"The controller widened the view of the shelf {shelf!r} (the titles of the listings that carry "
                          "your plan's values, its brands, its title words with those values) and read the pages listed.")

    def look_for_variants(self, pid: str, max_price: float | None) -> None:
        # One store turn that lists the other variants of a listing (a search with its title) and its brand's listings
        # on its shelf, and one that reads their pages: a store may list the variants of one product as pages of their
        # own.
        page = self.pages.get(pid) or {}
        title, brand = page.get("title"), page.get("brand")
        path = page.get("category_path") if isinstance(page.get("category_path"), list) else []
        calls = []
        if isinstance(title, str) and title.strip():
            calls.append({"name": "search", "args": {"query": " ".join(title.split()[:VARIANT_QUERY_WORDS]), "in_stock": True,
                                                     "k": RESULT_ROWS}})
        if isinstance(brand, str) and brand.strip() and path:
            calls.append({"name": "filter", "args": {"category": str(path[-1]), "brand": brand, "in_stock": True, "k": RESULT_ROWS,
                                                     **({"max_price": max_price} if max_price is not None else {})}})
        ready = []
        for c in calls:
            schema = self.schemas.get(c["name"])
            known = schema.get("properties") if isinstance(schema, dict) and isinstance(schema.get("properties"), dict) else {}
            c["args"] = {k: v for k, v in c["args"].items() if k in known}
            if isinstance(schema, dict) and _argument_error(c["args"], schema) is None \
                    and f"{c['name']} {_json(c['args'])}" not in self.calls_made:
                ready.append(c)
        if not ready or self.turns_left() <= CHOICE_TURNS + VARIANT_TURNS:
            return
        observations = self.call(ready[: self.max_calls])
        listed = []
        for c, observation in zip(ready, observations):
            self.result_view(c, observation)
            listed += [row.get("product_id") for row in observation.get("results") or [] if isinstance(row, dict)]
        self.read([p for p in listed if isinstance(p, str)])
        self.paths["variant_lookups"] += 1

    def claims_for(self, pid: str, sku: str, claims: Any) -> list[dict]:
        # The model's claims that the page and the latest reads bear out, as the order's justification.
        page = self.pages.get(pid) or {}
        facts = page.get("facts") if isinstance(page.get("facts"), dict) else {}
        variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and v.get("sku") == sku), {})
        state = self.state_of(pid, sku)
        path = page.get("category_path") if isinstance(page.get("category_path"), list) else []
        shown = {"title": page.get("title"), "options": variant.get("options"), "price": state.get("price"),
                 "in_stock": state.get("in_stock"), "category": " > ".join(str(p) for p in path) if path else None}
        kept = []
        fold = lambda v: " ".join(str(v).casefold().split())
        for claim in claims if isinstance(claims, list) else []:
            if not isinstance(claim, dict):
                continue
            field, value = claim.get("field"), claim.get("value")
            actual = shown[field] if field in OWN_FIELDS else self.fact_values(page).get(field) if isinstance(field, str) else None
            if actual is None:
                continue  # a label the page lacks bears nothing out: the claim has to name a field of the page
            displayed = _cut(actual, TITLE_CHARS if field == "title" else VALUE_CHARS)  # as the model saw it
            if field == "category":
                displayed = " > ".join(str(p) for p in path[-2:])  # the page view shows the last two elements
            same = (actual == value or (isinstance(value, str) and fold(value) in (fold(actual), fold(displayed)))
                    or (_number(actual) is not None and _number(value) is not None and _number(actual) == _number(value)))
            if same and field == "options" and isinstance(value, str) and isinstance(actual, str) and fold(value) != fold(actual):
                same = False  # a part of an option text must stand as whole words of it, below
            if not same and field in ("options", "title", "category") and isinstance(value, str) and isinstance(actual, str):
                # a part of an option text, a title or a category path, as whole words of it (a word inside a hyphenated
                # code does not count); a title also as a run of its text; a value with a digit also as written
                # without spaces ("2 pcs" and "2pcs")
                same = bool(_tokens(value)) and _tokens(value) <= _tokens(actual)
                if not same and field == "title" and fold(value) and fold(value) in fold(actual):
                    same = True
                if not same and any(ch.isdigit() for ch in value) and _compact(value):
                    same = _compact(value) in _compact(actual)
            if same:
                kept.append({"field": field, "value": actual})
        return kept

    def uncovered(self, claims: Any) -> list[str]:
        # The requirements of the latest plan that no claim names: each claim names the requirement it meets in the
        # plan's words; a requirement counts as named when the words agree.
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        if not isinstance(plan, list):
            return []
        claim_rows = [c for c in claims if isinstance(c, dict) and isinstance(c.get("requirement"), str)] if isinstance(claims, list) else []
        claim_names = [c["requirement"] for c in claim_rows]
        numbered_names = [c["requirement"] for c in claim_rows if c.get("field") != "category"]  # a shelf carries no quantity
        limit = _number(self.plan_read.get("price_limit"))
        wanted = [r for r in plan if isinstance(r, str) and r.strip()]
        out = []
        for requirement in wanted:
            if limit is not None and any(abs(m[0] - limit) < 1e-9 for m in _measures(requirement)):
                continue  # the price limit is covered by max_price
            words = {w for w in _words(requirement) if len(w) >= 3 or any(ch.isdigit() for ch in w)}
            if not words:
                continue
            # the words that set this requirement apart from the others (its value rather than the attribute both name)
            others = {w for other in wanted if other != requirement for w in _words(other)}
            must = (words - others) or words
            numbers = [_compact(n) for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:\s?[a-zA-Z]{1,8})?(?![\w])", requirement)]
            covered = False
            for n in numbered_names if numbers else claim_names:
                ws = set(_words(n))
                agree = must <= ws or (ws and ws <= set(_words(requirement)) and ws & must)
                if agree and all(number in _compact(n) for number in numbers):
                    covered = True  # a requirement with a number is named only by a claim that carries the number
                    break
            if not covered:
                out.append(requirement)
        return out

    def restates_a_use(self, requirement: str) -> bool:
        # A requirement of the plan that only restates a use the first message names (as the model listed the uses
        # when asked; none listed, none restated): the words of its value all stand in that use, or the words of its
        # attribute do and its value merely affirms it. What the purchase is for is no attribute of the SKU.
        attribute, value = _split(requirement)
        if _measures(value):
            return False
        named, worded = _words(attribute), _words(value)
        for use in self.uses or []:
            spoken = _words(use)
            if worded and worded <= spoken and any(len(w) >= 4 for w in worded):
                return True
            if named and named <= spoken and any(len(w) >= 4 for w in named) and self.affirms(requirement):
                return True
        return False

    def waived_use(self, pid: str, sku: str, requirement: str) -> bool:
        # Whether a requirement of the plan without a claim is waived at this commit: it only restates a use the
        # first message names and no field of the chosen SKU's page writes it (a page that writes it is claimed as
        # any other). Counted once per requirement.
        if not self.restates_a_use(requirement) or self.fields_writing(pid, sku, requirement):
            return False
        if requirement not in self.waived_uses:
            self.waived_uses.add(requirement)
            self.paths["use_requirements_waived"] += 1
        return True

    def ceiling(self, stated: Any) -> float | None:
        # The price ceiling in force: the lower of the one the model names at commit and the one of its latest plan.
        planned = _number(self.plan_read.get("price_limit")) if isinstance(self.plan_read, dict) else None
        limits = [x for x in (_number(stated), planned) if x is not None]
        return min(limits) if limits else None

    def news_since(self, index: int) -> list[str]:
        # The shopper's lines from this one on that say something new. A line that is nothing but questions and comes
        # one or two turns after a cart addition for which the store reported a market event is the shopper's remark
        # on that event (whether the listing can still be had, whether its price rose): nothing new. Every other
        # line is news, a statement that ends in a question too.
        marked = [event["turn"] for event in self.events]

        def remark(line: dict) -> bool:  # (judged by the whole message the line came in)
            sentences = [part for part in re.split(r"(?<=[.!?])\s+", str(line.get("whole") or line["content"]).strip()) if part]
            return (bool(sentences) and all(part.endswith("?") for part in sentences)
                    and any(1 <= line["turn"] - turn <= 2 for turn in marked))

        return [line["content"] for line in self.lines[index:] if not remark(line)]

    def cited(self, claims: Any) -> Any:
        # The claims an order is justified with: the price, the stock, the brand, and each claim about a requirement
        # the shopper stated of their own accord (in the first message or in a line said unasked). A requirement that
        # an answer to the controller's question added is met by the order and not cited, nor is the shelf: a shopper
        # who wants the order justified names what to cite. Without a claim about a requirement of the shopper's own,
        # every claim stands as the model made it.
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        kind = plan[0] if isinstance(plan, list) and plan else None
        own, beside = [], []
        for c in claims if isinstance(claims, list) else []:
            if not isinstance(c, dict):
                continue
            if c.get("field") in ("price", "in_stock", "brand"):
                beside.append(c)
                continue
            requirement = self.plan_requirement(c.get("requirement"))
            if isinstance(requirement, str) and requirement != kind and self.first_stated(requirement):
                own.append(c)
        return own + beside if own else claims

    def order(self, pid: str, sku: str, claims: Any) -> bool:
        args: dict = {"product_id": pid, "sku": sku}
        properties = self.schemas.get("place_test_order", {}).get("properties")
        kept = self.claims_for(pid, sku, self.cited(claims))
        path = (self.pages.get(pid) or {}).get("category_path")
        for claim in kept:  # the category as one label: a store checks a category claim against the labels of the path one by one
            if claim["field"] == "category" and isinstance(path, list) and path:
                claim["value"] = str(path[-1])
        state = self.state_of(pid, sku)
        for field in ("price", "in_stock"):  # the controller's own latest readings stand beside the model's claims
            if state.get(field) is not None and not any(c["field"] == field for c in kept):
                kept.append({"field": field, "value": state[field]})
        if isinstance(properties, dict) and "justification" in properties and kept:
            args["justification"] = {"claims": kept}
        if "place_test_order" not in self.schemas or _argument_error(args, self.schemas["place_test_order"]):
            args.pop("justification", None)
            if _argument_error(args, self.schemas.get("place_test_order", {})):
                return False
        self.ordered_claims = kept
        self.call([{"name": "place_test_order", "args": args}])
        return self.done

    def commit(self, args: dict) -> dict:
        # The purchase the model asked for: another SKU that bears out the claims of the first request goes into the
        # cart first, then the choice, then the order. Returns the result for the model.
        error = _argument_error(args, COMMIT_TOOL["function"]["parameters"])
        if error:
            return {"not_run": error}
        pid, sku = args["product_id"], args["sku"]
        if not self.going_on:
            self.latest = (deepcopy(args), self.plan_key())
        key = f"{pid} {sku} {_json(args.get('claims'))} {args.get('max_price')}"
        repeated = key in self.refused
        self.refused.add(key)
        again = "This same commit was refused before: change it or choose another SKU. " if repeated else ""
        max_price = self.ceiling(args.get("max_price"))
        if any(u["product_id"] == pid and u["sku"] == sku for u in self.unavailable):
            return {"not_run": f"{again}{pid} {sku} was reported unavailable; choose another SKU"}
        problem = self.availability(pid, sku, max_price)
        if problem:
            return {"not_run": f"{again}your choice {pid} {sku} cannot be ordered: {problem}"}
        bound = self.kind_shelf()
        if bound is not None and not self.under(pid, bound):
            path = (self.pages.get(pid) or {}).get("category_path")
            self.paths["off_shelf_refusals"] += 1
            return {"not_run": f"{again}this page is filed under {str(path[-1]) if isinstance(path, list) and path else None!r}, not under "
                               f"{bound!r}: the store has that shelf, whose label is the kind of product the shopper named. Choose a SKU "
                               "among the pages shown to you: they are the ones filed under it"}
        claims = args.get("claims")
        # few turns or decisions are left, or the commit was taken before: the choice is ordered with the claims its page bears out
        closing = self.closing() or self.going_on
        report_claims: dict = {}
        if not closing and bound is not None and self.plan_attributes() and self.looked_for != (self.plan_key(), self.ceiling(None)):
            self.paths["commits_before_lookup"] += 1
            return {"not_run": ("look before you commit: " if self.looked_for is None else
                                "your plan or its price limit changed since your last lookup: look again before you commit; ")
                               + "call find_pages with the values your plan requires; it lists the SKUs among the pages read "
                                 "whose page carries them and that can be ordered as the request now stands, those that state "
                                 "them in their option text or facts first"}
        if not closing and self.blocked.get((pid, sku), 0) >= BLOCK_AFTER_REFUSALS:
            return {"not_run": f"{pid} {sku} was refused {BLOCK_AFTER_REFUSALS} times for its claims and is no longer accepted in this "
                               "episode: choose another SKU"}
        unproven = self.unproven(pid, sku, claims)
        # A commit sent back only because a claim's value does not write the requirement in the plan's words stands
        # when the model makes it again: the page shows that value under that field, and whether it states the
        # requirement (by another word, in another unit) is the model's reading. A claim the page does not show,
        # another shelf, and a requirement without a claim are refused as before.
        wording = bool(unproven) and all(str(c.get("why") or "").startswith((OTHER_WORDING, LIST_WORDING)) for c in unproven)
        # (a title names every variant of a listing: when the SKU's own option text writes a number, a quantity read
        # off the title does not tell this variant apart, and such a reading does not stand)
        chosen = next((v for v in (self.pages.get(pid) or {}).get("variants") or [] if isinstance(v, dict) and v.get("sku") == sku), {})
        by_title = any(c.get("field") == "title" and _measures(_split(self.plan_requirement(c.get("requirement")) or "")[1])
                       for c in unproven) and bool(_measures(chosen.get("options")))
        stands = wording and key in self.reread and not by_title
        if stands:
            self.paths["readings_taken"] += 1
            self.readings.update((pid, sku, self.plan_requirement(c.get("requirement"))) for c in unproven)
        if unproven and not stands and (not closing or any(c.get("field") == "category" for c in unproven)):  # the shelf is never waived
            if wording:
                self.reread.add(key)
            self.paths["claim_refusals"] += 1
            self.blocked[(pid, sku)] = self.blocked.get((pid, sku), 0) + 1
            return {"not_run": again + "these claims are not borne out: " + _json(unproven)
                               + ". A claim's value must be what the page writes under that field, and a claim for a requirement of your "
                                 "plan must write that requirement (its number, or one of its words as a word of its own).",
                    **self.others_carrying(pid, sku, max_price)}
        missing = [m for m in self.uncovered(claims) if not self.waived_use(pid, sku, m)]
        if missing and not closing:
            self.blocked[(pid, sku)] = self.blocked.get((pid, sku), 0) + 1
            return {"where_this_page_writes_them": {m: self.where_written(pid, sku, m) for m in missing},
                    "not_run": again + "no claim names these requirements of your plan: " + _json(missing)
                               + "; add a claim the page bears out for each, or choose another SKU",
                    **self.others_carrying(pid, sku, max_price)}
        if unproven or missing:
            others = self.carrying_all(max_price)
            if others and (pid, sku) not in others and (pid, sku) not in self.pointed and DECISION_LIMIT - self.decisions >= 2:
                self.pointed.add((pid, sku))
                self.paths["commits_pointed_to_carrying_skus"] += 1
                return {"not_run": "this page does not bear out: " + _json([c.get("requirement") for c in unproven] + missing)
                                   + ". These SKUs among the pages read carry every value your plan requires on their own page "
                                     "and can be ordered: " + _json([{"product_id": row[0], "sku": row[1]} for row in others[:3]])
                                   + ". Commit one of them when it meets the request; else make the same commit again"}
            report_claims = {"claims_not_borne_out": [c.get("requirement") for c in unproven], "requirements_without_a_claim": missing}
        if not closing and (pid, sku) not in self.pointed and self.paths["commits_pointed_to_plain_statements"] < PLAIN_POINTERS:
            plainer = self.plainer(pid, sku, max_price)
            if plainer:
                self.pointed.add((pid, sku))
                self.paths["commits_pointed_to_plain_statements"] += 1
                return {"not_run": "your plan requires these, and this page does not state them as a value of "
                                   "its own (what it shows instead): " + _json(plainer["requirements"])
                                   + ". These SKUs among the pages read state them in a fact that is that value or in their option "
                                     "text, carry every value your plan requires and can be ordered: " + _json(plainer["skus"])
                                   + ". Those that also carry what the shopper would like come first. Commit the one that meets "
                                     "the request (the best by your plan's ranking when it has one); else make the same commit again"}
        kind_claimed = any(isinstance(c, dict) and c.get("field") == "category" and self.plan_requirement(c.get("requirement")) is not None
                           for c in claims or [])
        shelf = None if kind_claimed else self.off_shelf(pid)  # a category claim that carries the kind's words was checked above
        if shelf:
            self.paths["off_shelf_refusals"] += 1
            return {"not_run": f"{again}this page is filed under {shelf!r}, not under a shelf you filtered "
                               f"({', '.join(self.filtered_shelves)}). Choose a SKU on the requested shelf; if this "
                               "is the shelf the shopper meant, filter it first"}
        self.refused.discard(key)
        report: dict = {}
        leaving: list[dict] = []  # the call that takes the SKU that went in first out of the cart again
        lines_before = len(self.lines)
        if self.held is None and not self.added and self.turns_left() >= PURCHASE_TURNS:
            alt = self.pick_alternative(pid, sku, claims, max_price)
            if ((alt is None or not self.same_listing(pid, alt[0])) and not self.looked
                    and self.turns_left() >= PURCHASE_TURNS + VARIANT_TURNS):
                self.looked = True
                self.look_for_variants(pid, max_price)
                alt = self.pick_alternative(pid, sku, claims, max_price)
            if self.news_since(lines_before):
                alt = None  # the shopper spoke while the controller looked: nothing enters the cart before the model has read it
            elif alt is None:
                report["alternative"] = "no other SKU in view bears out the claims the first message states; your choice went in directly"
            if alt is not None:
                self.held = alt
                self.paths["alternatives_added"] += 1
                self.dialogue.append({"role": "assistant", "content": "Controller first cart addition",
                                      "controller_first_hold": {"product_id": alt[0], "sku": alt[1],
                                                                "choice": {"product_id": pid, "sku": sku}}})
                if self.add(alt[0], alt[1]):  # (read again in the same store turn)
                    state = self.state_of(*alt)
                    report["alternative"] = (f"the SKU that went in first, {alt[0]} {alt[1]} (not your choice), was read again: "
                                             f"in_stock {state.get('in_stock')}, price {state.get('price')}")
                    if (state.get("in_stock") is True and "remove_from_cart" in self.schemas
                            and _argument_error({"product_id": alt[0], "sku": alt[1]}, self.schemas["remove_from_cart"]) is None):
                        # the store left it as it was: it is not the choice, so it leaves the cart again, in the store
                        # turn that brings the choice in (in a turn of its own when the shopper spoke meanwhile)
                        leaving = [{"name": "remove_from_cart", "args": {"product_id": alt[0], "sku": alt[1]}}]
                else:
                    report["alternative"] = "the store did not add it"
        if self.turns_left() < CHOICE_TURNS:
            self.stop_reason = "turns"
            return {**report, "not_run": "too few store turns are left to add, re-read and order"}
        news = self.news_since(lines_before)
        if news:
            if leaving and self.turns_left() > CHOICE_TURNS:
                self.call(leaving)
                self.paths["alternatives_removed"] += 1
                report["alternative"] += "; taken out of the cart again"
            self.paths["commits_interrupted"] += 1
            self.pending = deepcopy(args)
            return {**report, "shopper_said": news,
                    "choice": f"your choice {pid} {sku} is not in the cart yet: read the shopper's new line and call commit again, "
                              "with your choice if it still meets the request as it now stands, else with another SKU"}
        if not any(isinstance(i, dict) and (i.get("product_id"), i.get("sku")) == (pid, sku) for i in self.cart):
            # (a SKU that was added first and taken out again is added anew: an order needs it in the cart)
            entered = self.add(pid, sku, leaving)  # (read again in the same store turn)
            if leaving:
                self.paths["alternatives_removed"] += 1
                report["alternative"] += "; taken out of the cart again"
            if not entered:
                return {**report, "choice": f"the store did not add {pid} {sku}; choose another SKU"}
        else:
            self.read_again(pid, sku)
        problem = self.availability(pid, sku, max_price)
        if problem:
            self.paths["choices_changed"] += 1
            self.unavailable.append({"product_id": pid, "sku": sku, "why": problem})
            if not self.looked and self.turns_left() >= CHOICE_TURNS + VARIANT_TURNS + 1:
                self.looked = True
                self.look_for_variants(pid, max_price)  # the same product listed again may still be had
            return {**report, "choice": f"{pid} {sku} is no longer available: {problem}. Call commit with another SKU "
                                        "that meets every requirement; the controller adds it, reads it again and orders it"}
        news = self.news_since(lines_before)
        if news and self.turns_left() > CHOICE_TURNS:  # the shopper spoke while the choice went in: no order before the model has read it
            self.paths["commits_interrupted"] += 1
            self.pending = deepcopy(args)
            return {**report, "shopper_said": news,
                    "choice": f"your choice {pid} {sku} is in the cart and was read again, not ordered yet: read the shopper's new "
                              "line and call commit again, with your choice if it still meets the request as it now stands, else "
                              "with another SKU"}
        if self.order(pid, sku, claims):
            self.paths["orders"] += 1
            return {**report, **report_claims, "ordered": {"product_id": pid, "sku": sku, "claims": self.ordered_claims}}
        return {**report, "choice": "the order was not accepted; read the latest state and decide again"}

    def read_plan(self, instruction: str) -> dict | None:
        # One model reading of the request and the shopper's lines so far: the plan with the kind first and each
        # requirement and preference as a phrase ("attribute: value"); None when no usable plan came.
        payload = {"model": self.model, "messages": self.messages + [{"role": "user", "content": instruction}],
                   "tools": [deepcopy(PLAN_TOOL)], "tool_choice": {"type": "function", "function": {"name": "plan"}}}
        result = self.request(payload, "/inference/chat/completions")
        choice = (result.get("choices") or [{}])[0] if isinstance(result, dict) else {}
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = [item for item in (message or {}).get("tool_calls") or [] if isinstance(item, dict)
                 and isinstance(item.get("function"), dict) and item["function"].get("name") == "plan"]
        if not calls:
            return None
        try:
            args = json.loads(calls[0]["function"].get("arguments") or "{}")
        except (ValueError, TypeError):
            return None
        if not isinstance(args, dict):
            return None
        for name in ("requirements", "preferences"):  # an entry written as one phrase is all value
            if isinstance(args.get(name), list):
                args[name] = [{"attribute": "", "value": row} if isinstance(row, str) else row for row in args[name]]
        if _argument_error(args, PLAN_TOOL["function"]["parameters"]):
            args = {**args, "doubted_pair": None}  # a malformed pair does not lose the plan
            if _argument_error(args, PLAN_TOOL["function"]["parameters"]):
                return None
        # kept as lists of phrases: the kind first, then each requirement as "attribute: value". An entry that only
        # names the kind again (under no attribute, under the kind's own words, under the plan's name for it or under
        # the page's name for its shelf) is no requirement of its own: the kind stands first. Neither is an entry
        # under the quantity the plan ranks by that states no number: it words the ranking again.
        phrase = lambda row: (f"{row['attribute'].strip()}: " if row["attribute"].strip() else "") + row["value"].strip()
        kind = {_stem(w) for w in _words(args["kind"])}
        names = kind | {_stem(w) for w in _words("kind")} | {_stem(w) for w in _words("category")}
        again = lambda row: (bool(kind) and {_stem(w) for w in _words(row["value"])} == kind
                             and {_stem(w) for w in _words(row["attribute"])} <= names)
        ranking = args.get("ranking") if isinstance(args.get("ranking"), dict) else {}
        ranked = {_stem(w) for w in _words(ranking.get("quantity"))}
        ranks = lambda row: (bool(ranked) and {_stem(w) for w in _words(row["attribute"])} == ranked
                             and not _measures(row["value"]))
        return {**args, "requirements": [args["kind"].strip()] + [phrase(r) for r in args["requirements"]
                                                                  if r["value"].strip() and not again(r) and not ranks(r)],
                "preferences": [phrase(r) for r in args["preferences"] if r["value"].strip()]}

    def plan_in_order(self, first_new: int, required: list[str], ceiling: float | None, preferred: list[str]) -> None:
        # The lines that arrived since the last plan read into the plan and confirmed; when lines of the shopper's own
        # came in more than one store turn, turn by turn: the later turns' lines leave the model's view (self.lines
        # and their messages) while the earlier ones are read and confirmed, and come back for their own reading.
        turns = sorted({line["turn"] for line in self.lines[first_new:] if not self.answer_line(line)})
        for cut in turns[:-1]:
            later = [line for line in self.lines[first_new:] if line["turn"] > cut]
            if not later:
                continue
            self.lines = [line for line in self.lines if line not in later]
            texts = ["Shopper: " + line["content"] for line in later]
            kept: list[dict] = []
            for message in reversed(self.messages):  # the latest copies of those lines
                if texts and message.get("role") == "user" and message.get("content") == texts[-1]:
                    texts.pop()
                    continue
                kept.append(message)
            self.messages = kept[::-1]
            self.paths["lines_read_in_turn"] += 1
            self.plan()
            self.confirm(required, first_new, ceiling, preferred)
            required, ceiling, preferred = self.plan_attributes(), self.ceiling(None), self.plan_preferences()
            first_new = len(self.lines)
            self.lines.extend(later)
            self.messages.extend({"role": "user", "content": "Shopper: " + line["content"]} for line in later)
        self.plan()
        self.confirm(required, first_new, ceiling, preferred)  # a requirement the shopper changed or added is confirmed

    def plan(self) -> None:
        # The request and the shopper's lines so far read into the plan, kept in the conversation. When the plan read
        # after the answer to the question about the use is the plan as it was before the answer, the model reads the
        # answer once more: an answer that names what the item has to have adds a requirement. A first reading whose
        # doubted pair is not two wishes of the plan, and a reading that words a requirement as a bare number, are
        # read once more. What a later reading drops without the shopper having spoken of it stays required.
        again = self.plan_read is not None
        answered, self.opening_answer = self.opening_answer, False
        since, heard = self.read_lines, len(self.lines)
        self.planned_lines = len(self.lines)
        instruction = "Read the request and the shopper's lines so far and call plan."
        if self.paths["priority_questions"]:
            instruction += (" The shopper was asked which of the two doubted wishes matters more: the one the answer names is a "
                            "requirement, the other a preference, and doubted_pair is null."
                            + (" When the answer names neither of the two, both stay preferences." if self.pair_unanswered else ""))
        args = self.read_plan(instruction)
        if args is None:  # a reply that carried no plan: read once more
            self.paths["plans_read_again"] += 1
            args = self.read_plan(instruction + " Call plan with every field filled in.")
        if args is None:
            self.plan_stale = True  # the lines since the last plan are not in it: no purchase goes on by itself
            return
        self.plan_stale = False
        if not again:
            # a quantity the request writes as one word (a number joined to its unit) that the plan carries nowhere
            have = _measures(" ".join(args["requirements"] + args["preferences"]
                                      + [str((args.get("ranking") or {}).get("quantity") or "") if isinstance(args.get("ranking"), dict) else ""]))
            limit = _number(args.get("price_limit"))
            missing = [m.group(0) for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)([a-zA-Z]{1,8})(?![\w])", self.query)
                       if not (limit is not None and abs(float(m.group(1)) - limit) < 1e-9)
                       and not any(_same_measure((float(m.group(1)), m.group(2).casefold()), other) for other in have)]
            if missing:
                self.paths["quantities_read_again"] += 1
                args = self.read_plan(instruction + QUANTITY_AGAIN.format(found=", ".join(dict.fromkeys(missing)))) or args
            args = self.read_pair(args, instruction)
        if again and answered:
            # (the requirements, the ranking and the limit: a preference the answer added is no requirement yet)
            before, held = (self.plan_key()[:3], self.ceiling(None)), self.plan_read
            self.plan_read = args
            unchanged = (self.plan_key()[:3], self.ceiling(None)) == before
            self.plan_read = held
            if unchanged:
                self.paths["answers_read_again"] += 1
                args = self.read_plan(instruction + ANSWER_AGAIN) or args
        args = self.read_units(args, instruction)
        if again:
            args, kind_note = self.keep_kind(args, instruction)
            if answered:
                # whether the answer set a requirement: one the plan did not hold before, other than the kind of product
                # worded again (an answer that repeats the shelf and the price limit names nothing the item has to have)
                fold = lambda v: " ".join(str(v).casefold().split())
                earlier = {fold(r) for r in self.plan_attributes()}
                before_answer, self.plan_read = self.plan_read, args
                kind = {_stem(w) for w in _words(args["requirements"][0])} if args.get("requirements") else set()
                gained = [r for r in self.plan_attributes() if fold(r) not in earlier
                          and not {_stem(w) for w in _words(_split(r)[1])} <= kind]
                self.plan_read = before_answer
                self.use_unanswered = not gained
        else:
            self.kind_named, kind_note = args["requirements"][0], ""
        if isinstance(args.get("doubted_pair"), list):
            self.doubted += [x for x in args["doubted_pair"] if isinstance(x, str) and x.strip()]
        kept = self.keep_changes(args, self.ceiling(None)) if again else ""  # (the limit of the plan before this one)
        settling = again and self.asked_pair is not None and not self.pair_read
        if settling:
            self.pair_read = True
            if self.pair_unanswered and self.model_chose(args):
                self.pair_unanswered = False  # the model read the answer as naming one of the two
                self.paths["wishes_read_by_the_model"] += 1
        if again:
            kept += self.keep_wishes(args)
        kept += self.settle_pair(args)
        if again:
            kept += self.keep_requirements(args, since)
        kept += kind_note
        self.plan_read, self.read_lines = args, heard
        self.paths["plans"] += 1
        self.messages.append({"role": "user", "content": ("Your plan, read again after the shopper's new line" + kept + ": " if again
                                                          else "Your plan: ") + _json(args)})
        self.dialogue.append({"role": "assistant", "content": "Plan", "plan": deepcopy(args)})

    def read_pair(self, args: dict, instruction: str) -> dict:
        # A first reading that names a doubted pair which is not two wishes the plan lists (the two things the
        # purchase is for, words of the doubt itself) is read once more; the new reading stands when its pair is
        # two wishes of its own plan.
        def listed(plan: dict) -> bool:
            pair = plan.get("doubted_pair")
            held, self.plan_read = self.plan_read, plan
            try:
                return (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) and x.strip() for x in pair)
                        and self.wishes(pair))
            finally:
                self.plan_read = held

        if not isinstance(args.get("doubted_pair"), list) or listed(args):
            return args
        self.paths["pairs_read_again"] += 1
        quote = args.get("doubt_quote")
        read = self.read_plan(instruction + PAIR_AGAIN.format(pair=_json(args["doubted_pair"]))
                              + (PAIR_QUOTE.format(quote=_json(quote)) if isinstance(quote, str) and quote.strip() else ""))
        return read if read is not None and listed(read) else args

    def read_units(self, args: dict, instruction: str) -> dict:
        # A reading that words a requirement or a preference as a number without a unit is read once more: a count
        # of items is worded as that number of pcs, any other quantity with its unit. The new reading stands when it
        # words no entry so; an entry is asked about once.
        bare = lambda plan: [entry for name in ("requirements", "preferences") for entry in plan.get(name) or []
                             if isinstance(entry, str) and re.fullmatch(r"\d+(?:\.\d+)?", _split(entry)[1])]
        found = [entry for entry in bare(args) if entry not in self.units_asked]
        if not found:
            return args
        self.units_asked.update(found)  # each such entry is asked about once
        self.paths["units_read_again"] += 1
        read = self.read_plan(instruction + UNIT_AGAIN.format(entries=", ".join(found)))
        return read if read is not None and not bare(read) else args

    def model_chose(self, plan: dict) -> bool:
        # Whether a reading of the plan made one of the two wishes asked about a requirement that the plan before it
        # did not require, and left the other out of the requirements.
        pair = self.asked_pair[0] if self.asked_pair is not None else []
        was = [any(self.names_requirement(wish, entry) for entry in self.plan_attributes()) for wish in pair]
        held, self.plan_read = self.plan_read, plan
        try:
            now = [any(self.names_requirement(wish, entry) for entry in self.plan_attributes()) for wish in pair]
        finally:
            self.plan_read = held
        return now.count(True) == 1 and not was[now.index(True)]

    def keep_kind(self, args: dict, instruction: str) -> tuple[dict, str]:
        # The kind of product is what the first message names. When a later reading of the plan names another kind
        # (neither wording holds every word of the other: a shorter or a fuller wording of the same kind passes), the
        # model reads the lines once more with the first kind before it; a reading that still names another kind has
        # the first kind put back. Returns the plan and the note for the model.
        first = self.kind_named
        stems = lambda v: {_stem(w) for w in _words(v)}

        def another(plan: dict) -> bool:
            named = stems(plan["requirements"][0]) if plan.get("requirements") else set()
            return bool(first) and bool(named) and not (named <= stems(first) or stems(first) <= named)

        if not another(args):
            return args, ""
        self.paths["kinds_read_again"] += 1
        read = self.read_plan(instruction + KIND_AGAIN.format(first=_json(first), named=_json(args["requirements"][0])))
        if read is not None:
            args = read
        if another(args):
            args["requirements"][0] = first
            if "kind" in args:
                args["kind"] = first
            self.paths["kinds_put_back"] += 1
        return args, f" (the kind of product stays as the shopper first named it: {first})"

    def answer_facts(self) -> dict[str, list[str]]:
        # The facts of the pages read on the shopper's shelf whose whole value an answer of the shopper names (every
        # word of a value of a few words, no number) and that the plan holds nowhere: {label: values}. An answer
        # that restates the terms of the request (it carries the price limit) names no such value.
        limit = _number(self.plan_read.get("price_limit")) if isinstance(self.plan_read, dict) else None
        answers = [line["content"] for line in self.lines if self.answer_line(line)
                   and not (limit is not None and any(abs(n - limit) < 1e-9 for n, _ in _measures(line["content"].replace(",", ""))))]
        if not answers or not isinstance(self.plan_read, dict):
            return {}
        said = {_stem(w) for answer in answers for w in re.findall(r"[a-z0-9]+", answer.casefold())}
        held = self.plan_attributes() + self.plan_preferences()
        bound = self.kind_shelf()
        found: dict[str, list[str]] = {}
        for pid, page in self.pages.items():
            if (bound is not None and not self.under(pid, bound)) or (bound is None and self.off_shelf(pid)):
                continue
            for label, written in self.fact_values(page).items():
                value = " ".join(str(written).split())
                words = [_stem(w) for w in re.findall(r"[a-z0-9]+", value.casefold())]
                missing = sum(1 for w in words if w not in said)
                if (not words or len(words) > FACT_WORDS or any(ch.isdigit() for ch in value)
                        or missing > (1 if len(words) >= 3 else 0) or any(len(w) < 3 for w in words)):
                    continue
                if any(self.names_requirement(value, entry, label)
                       or set(words) <= {_stem(w) for w in re.findall(r"[a-z0-9]+", _split(entry)[1].casefold())} for entry in held):
                    continue  # held already, also as a part of a longer value (a brand of two words, a type with its noun)
                if value.casefold() not in [v.casefold() for v in found.get(label, [])]:
                    found.setdefault(label, []).append(value)
        return found

    def shelf_fact_words(self) -> set[str]:
        # The words of the fact values that the pages read on the shopper's shelf state.
        bound = self.kind_shelf()
        out: set[str] = set()
        for pid, page in self.pages.items():
            if (bound is not None and not self.under(pid, bound)) or (bound is None and self.off_shelf(pid)):
                continue
            for written in self.fact_values(page).values():
                out |= {_stem(w) for w in re.findall(r"[a-z0-9]+", str(written).casefold())}
        return out

    def ground(self) -> None:
        # Once, after the shelf was read: when its pages state facts whose value the shopper's answer names and the
        # plan requires nowhere, the model reads the answer again with those facts beside it and the plan it then
        # reads is the one in force.
        if self.grounded or self.done or self.stop_reason is not None or not isinstance(self.plan_read, dict):
            return
        facts = self.answer_facts()
        if not facts:
            return
        self.grounded = True
        answers = [line["content"] for line in self.lines if self.answer_line(line)]
        shown = dict(list(facts.items())[:FACTS_SHOWN])
        # (the answer shown is the one that names the first of those values)
        naming = [a for a in answers if any(_stem(w) in {_stem(x) for x in re.findall(r"[a-z0-9]+", a.casefold())}
                                            for values in shown.values() for w in re.findall(r"[a-z0-9]+", values[0].casefold()))]
        answers = naming or answers
        instruction = ("Read the request and the shopper's lines so far and call plan."
                       + ANSWER_FACTS.format(answer=_json(answers[-1]), facts=_json(shown)))
        args = self.read_plan(instruction)
        self.paths["answers_read_with_facts"] += 1
        if args is None:
            return
        args, _ = self.keep_kind(args, instruction)
        # the reading may add what the answer names, or word it as the store does. When a word that the shopper
        # said themselves, of a value the plan requires, stands nowhere in the new reading, the reading went astray
        # and the plan stays, unless the new reading words that requirement again: an entry of its own whose value
        # is drawn from the words the plan had under the attribute (the plan had split attribute and value otherwise),
        # not a shorter form of the value itself.
        stems = lambda value: {_stem(w) for w in re.findall(r"[a-z0-9]+", str(value).casefold())}
        fold = lambda value: " ".join(str(value).casefold().split())
        said = stems(self.query + " " + " ".join(line["content"] for line in self.lines))
        reading = stems(" ".join(str(r) for r in args["requirements"][1:]))
        held_before = {fold(r) for r in self.plan_attributes()}
        fresh = [r for r in args["requirements"][1:] if isinstance(r, str) and fold(r) not in held_before]
        written = None  # the words of the fact values the shelf's pages state, read when a rewording is weighed
        for requirement in self.plan_attributes():
            value = stems(_split(requirement)[1])
            own = value & said
            if not own or own <= reading:
                continue
            if any(stems(_split(entry)[1]) <= stems(requirement) and not stems(_split(entry)[1]) <= value for entry in fresh):
                # (a word of the value that the shopper said and that the shelf's facts write as a value is the
                # store's own word for it: a rewording that loses it went astray all the same)
                if written is None:
                    written = self.shelf_fact_words()
                if not (own - reading) & written:
                    continue
            self.paths["plans_kept_against_facts"] += 1
            return
        before = self.plan_key()
        held, self.plan_read = self.plan_read, args
        if self.plan_key() == before:
            self.plan_read = held
            return
        self.paths["plans_changed_by_facts"] += 1
        self.paths["plans"] += 1
        self.messages.append({"role": "user", "content": "Your plan, read again with the facts of the shelf that the shopper's "
                                                          "answer names: " + _json(args)})
        self.dialogue.append({"role": "assistant", "content": "Plan", "plan": deepcopy(args)})
        self.plan_searches(args.get("queries"))

    def explore(self) -> None:
        # The plan's searches and the survey of the shelf it names.
        self.plan_searches(self.plan_read.get("queries") if isinstance(self.plan_read, dict) else None)
        self.survey()

    def plan_searches(self, queries: Any) -> None:
        # The plan's searches in one store turn, then the pages of what they listed: the kind of product with the
        # values of all required attributes and with each alone (a query that carries every value ranks listings by
        # all of them and can miss the ones that state one in another word), then the model's queries, then the
        # values alone and with each word of the kind (a listing may name the kind in other words); on a later
        # reading the titles of the listings in view that carry the values (listing_queries); last each query with
        # every number written the other way round its unit.
        if "search" not in self.schemas or self.turns_left() <= (CHOICE_TURNS if (self.held is not None or self.added) else PURCHASE_TURNS) + 1:
            return
        calls = []
        wanted = []
        kind, values = "", []
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        limit = _number(self.plan_read.get("price_limit")) if isinstance(self.plan_read, dict) else None
        if isinstance(plan, list) and plan and isinstance(plan[0], str) and plan[0].strip():
            kind = " ".join(_without_measures(plan[0]).split()) or plan[0].strip()
            for requirement in plan[1:]:
                if not isinstance(requirement, str) or not requirement.strip() or _words("in_stock") <= _words(requirement):
                    continue
                if limit is not None and any(abs(m[0] - limit) < 1e-9 for m in _measures(requirement)):
                    continue  # the price limit is no search word
                values.append(" ".join(_split(requirement)[1].split()))
            if len(values) > 1:
                wanted.append(" ".join([kind] + values))
            wanted += [f"{kind} {value}" for value in values]
            if _measures(plan[0]):
                wanted.append(" ".join(plan[0].split()))
        for query in queries if isinstance(queries, list) else []:
            if isinstance(query, str) and query.strip():
                wanted += [" ".join(query.split())]
        if len(values) > 1:
            joined = " ".join(values)
            words = re.findall(r"[A-Za-z]{3,}", kind)
            wanted += [joined] + ([f"{word} {joined}" for word in words[:KIND_WORDS]] if len(words) > 1 else [])
        wanted = wanted + self.listing_queries() + [v for q in wanted for v in _query_variants(q)]
        for query in wanted:
            args = {"query": query, "in_stock": True, "k": RESULT_ROWS}
            if "in_stock" not in self.schemas["search"].get("properties", {}):
                args.pop("in_stock")
            if "k" not in self.schemas["search"].get("properties", {}):
                args.pop("k")
            key = f"search {_json(args)}"
            if _argument_error(args, self.schemas["search"]) is None and key not in self.calls_made and not any(
                    f"search {_json(c['args'])}" == key for c in calls):
                calls.append({"name": "search", "args": args})
        if not calls:
            return
        before = len(self.listed)
        batch = calls[: self.max_calls]
        observations = self.call(batch)
        for c, observation in zip(batch, observations):
            self.result_view(c, observation)
        self.paths["plan_searches"] += len(batch)
        self.notes.append(f"The controller ran {len(batch)} searches of your plan and listed {len(self.listed) - before} products.")
        self.auto_reads()

    def survey(self) -> None:
        # The shelf the plan's kind of product names, filtered within the price limit, swept by price, and its pages
        # read: among the shelves of the pages read so far, one whose label carries the kind's words (the label that
        # is the kind itself first); else the kind as the label, with the spellings a missed filter is retried with.
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        schema = self.schemas.get("filter")
        if (not isinstance(schema, dict) or not isinstance(plan, list) or not plan or not isinstance(plan[0], str)
                or not plan[0].strip() or self.filtered_shelves or self.turns_left() <= PURCHASE_TURNS + CLOSING_TURNS + 2):
            return
        kind = " ".join(_without_measures(plan[0]).split()) or plan[0].strip()
        shelves = [shelf for shelf in self.shelves_read() if _shelf_names_kind(shelf, kind)]
        # the label that is the kind itself first, then the longest label that stands whole in the kind
        shelves.sort(key=lambda shelf: (_label_words(shelf) != _label_words(kind),
                                        -len(_label_words(shelf)) if self.whole_in_kind(shelf) else 1))
        limit = _number(self.plan_read.get("price_limit"))
        args = {"category": shelves[0] if shelves else kind, "in_stock": True, "k": RESULT_ROWS,
                **({"max_price": limit} if limit is not None else {})}
        known = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        args = {k: v for k, v in args.items() if k in known}
        if _argument_error(args, schema) is not None or f"filter {_json(args)}" in self.calls_made:
            return
        batch = [{"name": "filter", "args": args}]
        if "brand" in known:  # the store's filter takes a brand: a value of the plan may be one
            for value in self.plan_values()[:BRAND_TRIES]:
                branded = {**args, "brand": value}
                if _argument_error(branded, schema) is None:
                    batch.append({"name": "filter", "args": branded})
        self.swept_limit = limit
        observations = self.call(batch[: self.max_calls])
        for c, observation in zip(batch, observations):
            self.result_view(c, observation)
        self.retry_filters(batch[:1], observations[:1])
        self.sweep(batch[:1], observations[:1])
        self.paths["surveys"] += 1
        self.auto_reads()

    def sweep_again(self) -> None:
        # After the shopper lowered the price limit: the shelf is filtered again within the new limit and with lower
        # caps, in one store turn, and the pages it lists are read in the next. The first sweep listed the shelf up
        # to the earlier limit, so its cheaper part may be unread.
        limit = _number(self.plan_read.get("price_limit")) if isinstance(self.plan_read, dict) else None
        schema = self.schemas.get("filter")
        shelf = self.kind_shelf() or (self.filtered_shelves[0] if self.filtered_shelves else None)
        if (limit is None or self.swept_limit is None or limit >= self.swept_limit or shelf is None
                or not isinstance(schema, dict) or self.turns_left() <= self.spare() + VARIANT_TURNS):
            return
        self.swept_limit = limit
        known = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        calls = []
        for share in (1.0,) + SWEEP_SHARES:
            args = {k: v for k, v in {"category": shelf, "in_stock": True, "k": RESULT_ROWS,
                                      "max_price": round(limit * share, 2)}.items() if k in known}
            if (_argument_error(args, schema) is None and f"filter {_json(args)}" not in self.calls_made
                    and not any(c["args"] == args for c in calls)):
                calls.append({"name": "filter", "args": args})
        if not calls:
            return
        before = len(self.listed)
        batch = calls[: self.max_calls]
        observations = self.call(batch)
        for c, observation in zip(batch, observations):
            self.result_view(c, observation)
        self.paths["sweeps_again"] += 1
        self.notes.append(f"The controller filtered the shelf {shelf!r} again within the new price limit and listed "
                          f"{len(self.listed) - before} more products.")
        self.auto_reads()

    def own_claims(self, pid: str, sku: str) -> list[dict]:
        # Claims the controller reads off a page: the shelf for the kind of product and, for each requirement of the
        # plan, the first field that writes it (the option text, a fact, the title) with what that field shows.
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        page = self.pages.get(pid) or {}
        variant = next((v for v in page.get("variants") or [] if isinstance(v, dict) and v.get("sku") == sku), {})
        path = page.get("category_path") if isinstance(page.get("category_path"), list) else []
        fields = {"options": variant.get("options"), **self.fact_values(page), "title": page.get("title")}
        claims = []
        if isinstance(plan, list) and plan and isinstance(plan[0], str) and path:
            claims.append({"field": "category", "value": " > ".join(str(p) for p in path[-2:]), "requirement": plan[0]})
        for requirement in self.plan_attributes():
            name = next(iter(self.fields_writing(pid, sku, requirement)), None)
            if name is not None and isinstance(fields.get(name), (str, int, float)):
                claims.append({"field": name, "value": fields[name], "requirement": requirement})
        return claims

    def last_resort(self) -> None:
        # The episode would end without an order (no decision is left, no model answered, or too few store turns
        # remain for another decision), and an episode without an order earns nothing: the controller orders what it
        # can stand behind. In turn: the commit a shopper line interrupted and the model's latest commit under the
        # plan in force, when that SKU stands on the shelf and can be ordered within the limit; the SKUs in view that carry every value of the
        # plan; the SKU on the shelf whose page carries the most of those values. The purchase runs as a commit of
        # the closing phase does: with two turns, the SKU goes into the cart with its read, then the order.
        if self.done or "place_test_order" not in self.schemas or "add_to_cart" not in self.schemas or self.turns_left() < 2:
            return
        limit, bound = self.ceiling(None), self.kind_shelf()
        choices: list[tuple[str, str, Any]] = []
        made = [self.pending] + ([self.latest[0]] if self.latest is not None and self.latest[1] == self.plan_key() else [])
        for args in made:  # (a commit made under another plan than the one in force is not taken up again)
            if isinstance(args, dict) and isinstance(args.get("product_id"), str) and isinstance(args.get("sku"), str):
                choices.append((args["product_id"], args["sku"], args.get("claims")))
        choices += [(pid, sku, None) for pid, sku in self.carrying_all(limit)]
        values, ranked = self.lookup_values(self.plan_values()), []
        for pid, page in self.pages.items():
            for variant in page.get("variants") or []:
                sku = variant.get("sku") if isinstance(variant, dict) else None
                if not isinstance(sku, str):
                    continue
                fields = {"options": variant.get("options"), **self.fact_values(page), "title": page.get("title")}
                carried = sum(1 for value in values if any(isinstance(written, (str, int, float)) and self.carries(written, value, name)
                                                           for name, written in fields.items()))
                ranked.append((-carried, len(ranked), pid, sku))
        choices += [(pid, sku, None) for _, _, pid, sku in sorted(ranked)]
        seen: set[tuple[str, str]] = set()
        attempts = 0
        for pid, sku, claims in choices:
            if (pid, sku) in seen or pid not in self.pages or self.turns_left() < 2:
                continue
            seen.add((pid, sku))
            if (self.availability(pid, sku, limit) or any(u["product_id"] == pid and u["sku"] == sku for u in self.unavailable)
                    or (bound is not None and not self.under(pid, bound)) or (bound is None and self.off_shelf(pid))):
                continue
            self.paths["last_resort_orders"] += 1
            attempts += 1
            stopped, self.stop_reason = self.stop_reason, None
            claims = claims if isinstance(claims, list) else self.own_claims(pid, sku)
            self.going_on = True
            try:
                result: Any = self.commit({"product_id": pid, "sku": sku, "max_price": limit, "claims": claims})
            finally:
                self.going_on = False
            self.dialogue.append({"role": "assistant", "content": "Controller last order",
                                  "last_resort": {"product_id": pid, "sku": sku, "done": self.done, "result": _cut(_json(result), 300)}})
            if self.done:
                return
            self.stop_reason = self.stop_reason or stopped
            if attempts >= 3:
                return  # three SKUs tried: the store keeps refusing

    def plan_key(self) -> tuple:
        # What the plan asks of the SKU, apart from the price limit: the kind, the required attributes, the ranking
        # and the preferences, each in one spelling. Two readings with the same key differ by the limit at most.
        plan = self.plan_read if isinstance(self.plan_read, dict) else {}
        fold = lambda v: " ".join(str(v).casefold().split())
        requirements = plan.get("requirements") if isinstance(plan.get("requirements"), list) else []
        ranking = plan.get("ranking") if isinstance(plan.get("ranking"), dict) else {}
        preferences = plan.get("preferences") if isinstance(plan.get("preferences"), list) else []
        return (fold(requirements[0]) if requirements else "", tuple(sorted(fold(r) for r in self.plan_attributes())),
                (fold(ranking.get("quantity") or ""), ranking.get("direction")), tuple(sorted(fold(p) for p in preferences)))

    def resume(self, before: tuple) -> bool:
        # The purchase the shopper interrupted goes on without another decision when the plan, read again, asks the
        # same of the SKU as before (the price limit may have changed) and the model's choice can still be ordered
        # within the limit in force: the line changed nothing about that choice. The model is told what happened.
        pending, self.pending = self.pending, None
        if pending is None or self.done or self.stop_reason is not None or self.plan_key() != before or self.plan_stale:
            return False  # (a stale plan does not say what the shopper's line changed: the model decides)
        limit = self.ceiling(None)
        if self.availability(pending["product_id"], pending["sku"], limit) or self.turns_left() < CHOICE_TURNS:
            return False
        self.paths["purchases_resumed"] += 1
        self.looked_for = (self.plan_key(), limit)  # the plan asks the same: the earlier lookup stands
        self.going_on = True
        try:
            result = self.commit({**pending, "max_price": limit})
        finally:
            self.going_on = False
        note = {"role": "user", "content": "Controller: the shopper's new line changed no requirement of your plan and your choice "
                                           f"{pending['product_id']} {pending['sku']} still met it, so the purchase went on. " + _json(result)}
        self.messages.append(note)
        self.dialogue.append(deepcopy(note))
        return True

    def plan_attributes(self) -> list[str]:
        # The attributes the plan requires, as it words them ("attribute: value"): not the kind, not stock, not the
        # price limit.
        plan = self.plan_read.get("requirements") if isinstance(self.plan_read, dict) else None
        limit = _number(self.plan_read.get("price_limit")) if isinstance(self.plan_read, dict) else None
        return [" ".join(r.split()) for r in (plan[1:] if isinstance(plan, list) else [])
                if isinstance(r, str) and r.strip() and not _words("in_stock") <= _words(r)
                and not (limit is not None and any(abs(m[0] - limit) < 1e-9 for m in _measures(r)))]

    def plan_values(self) -> list[str]:
        # The values of the attributes the plan requires.
        return [_split(r)[1] for r in self.plan_attributes()]

    def plan_preferences(self) -> list[str]:
        # The wishes the plan lists as preferences ("attribute: value").
        preferences = self.plan_read.get("preferences") if isinstance(self.plan_read, dict) else None
        return [p for p in (preferences if isinstance(preferences, list) else []) if isinstance(p, str)]

    def run(self) -> list[dict]:
        stalls = 0
        unanswered = False
        try:
            self.plan()  # the request alone: which question it calls for
            for _ in range(4):  # the question, its answer read into the plan; the other question when the answer named
                # nothing; the question about two wishes once more when both answers only restated the terms
                asked = len(self.lines)
                required, preferred, ceiling = self.plan_attributes(), self.plan_preferences(), self.ceiling(None)
                self.ask()
                if self.done or self.stop_reason is not None or len(self.lines) == asked:
                    break
                self.plan()
                self.confirm(required, asked, ceiling, preferred)  # a line of the shopper's own that came with the answer
            if not self.done and self.stop_reason is None:
                self.explore()
                self.ground()  # the answer read against the facts the shelf states
                self.use_again()  # every answer restated the terms: another attribute is asked about, once
                self.second_use()  # a request that names two uses: what the other one requires is asked once
            while not self.done and self.stop_reason is None and self.decisions < DECISION_LIMIT and self.turns_left() >= CHOICE_TURNS:
                if self.news_since(self.planned_lines):
                    first_new = self.planned_lines
                    self.planned_lines = len(self.lines)
                    before, required, ceiling = self.plan_key(), self.plan_attributes(), self.ceiling(None)
                    preferred = self.plan_preferences()
                    self.plan_in_order(first_new, required, ceiling, preferred)  # a later line overrides an earlier one: the request is read again, turn by turn
                    self.ask()  # a doubt that only this reading names is asked about, once
                    self.use_again()  # the attribute question after the shelf restated the terms too: once more
                    if self.resume(before):
                        continue  # the interrupted purchase went on: nothing else changed
                    lower = self.ceiling(None)
                    if (ceiling is not None and lower is not None and lower < ceiling
                            and self.turns_left() <= self.spare() + 2 * VARIANT_TURNS):
                        self.sweep_again()  # turns for one step only: the cheaper part of the shelf comes first
                    if self.turns_left() > self.spare() + VARIANT_TURNS:
                        self.plan_searches(self.plan_read.get("queries") if isinstance(self.plan_read, dict) else None)
                    self.sweep_again()  # a lower price limit: the cheaper part of the shelf
                    continue  # a line that arrived meanwhile is read into the next plan
                self.planned_lines = len(self.lines)
                self.look_further()
                self.show_pages()  # what the controller fetched on its own is in view before the model decides
                calls = self.decide()
                if calls is None and self.stop_reason is None and not unanswered:
                    unanswered = True  # no model answered: the decision is asked for once more
                    self.decisions -= 1
                    continue
                if calls is None:
                    self.stop_reason = self.stop_reason or "inference_unavailable"
                    break
                if not calls:
                    stalls += 1
                    self.messages.append({"role": "user", "content": "No tool was called. Call store tools or commit."})
                    if stalls > STALLS_TOLERATED:
                        self.stop_reason = "no_decision"
                        break
                    continue
                stalls = 0
                for c in calls:
                    if c["name"] == "find_pages" and c["result"] is None:
                        c["result"] = self.find_pages(c["args"])  # answered from the pages read: no store turn
                reads = [c for c in calls if c["name"] != "commit"]
                commits = [c for c in calls if c["name"] == "commit"]
                if reads:
                    self.run_reads(reads)
                    for c in reads:
                        if c.get("variant_of"):
                            calls.append(c)
                for index, c in enumerate(commits):
                    if index or self.done or self.stop_reason is not None:
                        c["result"] = {"not_run": "one commit per decision"}
                    else:
                        self.pending = None  # the model decides anew
                        c["result"] = self.commit(c["args"])
                self.answer(calls)
            if not self.done and self.stop_reason in (None, "inference_unavailable", "no_decision", "turns"):
                self.last_resort()  # no episode ends without an order while a SKU can be ordered
            if not self.done and self.stop_reason is None:
                self.stop_reason = "decision_limit" if self.decisions >= DECISION_LIMIT else "turn_limit"
        except RuntimeError as error:
            self.stop_reason = self.stop_reason or "environment_outcome_unknown"
            self.dialogue.append({"role": "assistant", "content": str(error)})
        self.dialogue.append({"role": "assistant", "content": "Controller status", "controller_status": {
            "done": self.done, "stop_reason": self.stop_reason, "environment_turns": self.turn,
            "model_decisions": self.decisions, "model": self.model, "paths": dict(self.paths),
            "first_hold": self.held, "ordered_claims": self.ordered_claims}})
        return self.dialogue


def agent_main(problem_data: dict) -> list[dict]:
    proxy = ProxyClient(timeout=120, max_retries=1)
    agent = EvidenceAgent(problem_data, proxy)
    try:
        return agent.run()
    except Exception as error:  # noqa: BLE001 - what the episode already did is returned, never discarded
        agent.dialogue.append({"role": "assistant", "content": f"Agent fault: {type(error).__name__}: {error}"[:300]})
        return agent.dialogue


__all__ = ["agent_main"]
