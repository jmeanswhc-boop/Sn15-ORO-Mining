
from __future__ import annotations

import html
import itertools
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from xml.etree import ElementTree

from src.agent.proxy_client import ProxyClient

HAIKU, SONNET = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"
PRICE = {HAIKU: (0.1, 0.5, 0.01, 0.125), SONNET: (2.0, 10.0, 0.1, 2.5)}
WALK_MONEY, TASK_MONEY = 0.040, 0.055
STEPS = [(SONNET, "low"), (HAIKU, "high"), (HAIKU, "medium")]
WRITES = {"low": 700, "medium": 1600, "high": 3000}
WALK_S, TASK_S, SEAT_S = 240.0, 290.0, 20.0
RATE, BURST = 1.0, 5
TRACE_T0 = [time.monotonic()]


def trace_row(trace: list, kind: str, **fields: Any) -> None:
    return None

"""Every model seat in one place: the seat table (model, reasoning effort, and whether it decides past the money cap), then each seat's prompt and its output schema, in the order the walk calls them. Code elsewhere only fills the inputs."""

SEATS = {
    "MAP": (HAIKU, "low", "walk"),
    "INTAKE": (HAIKU, "high", "walk"),
    "INTAKE_TIE": (SONNET, "low", "walk"),
    "RECORD": (HAIKU, "medium", "decide"),
    "STORY": (SONNET, "low", "walk"),
    "READ": (HAIKU, "low", "change"),
    "READ_ALT": (SONNET, "low", "walk"),
    "NOTICE": (HAIKU, "low", "change"),
    "CHAIN_PARSE": (HAIKU, "low", "walk"),
    "CHAIN_READ": (HAIKU, "medium", "walk"),
    "FIT": (HAIKU, "medium", "decide"),
    "JUDGE": (SONNET, "low", "final"),
    "AUDIT": (SONNET, "low", "final"),
    "SEARCH": (HAIKU, "medium", "walk"),
    "ACTOR": (HAIKU, "medium", "walk"),
    "ROWS": (HAIKU, "low", "walk"),
    "MAP_ALT": (SONNET, "low", "walk"),
    "INTAKE_ALT": (SONNET, "low", "walk"),
}

MAP_TOOLS = """You map a shop's tools to roles for a shopping agent. You read only the tool list below and answer with one JSON object.

Roles, each null when no tool fits:
- search: free-text product search. query_arg = the free-text argument. page_arg = the argument that asks for further results (a token, a page number or a row offset), page_kind = "token", "page" or "offset". fixed = other required arguments with the value that selects plain text search (an enum value copied exactly), else {}. price_arg = an argument that caps the price, category_arg = an argument that limits results to one shelf or category. order = "price" only when the tool's description says its results always come cheapest first, else null (an argument that chooses the sort order is not such a statement).
- browse: a tool that lists a shelf or price range without free text (same fields as search; query_arg null).
- facts: a tool that returns the attribute table of one product. id_arg = its product id argument.
- page: a tool that returns the product page (description, small print, remarks) of one product. id_arg as above. The same tool may be both facts and page.
- compare: a tool that shows several products at once. ids_arg = its list argument, most = the largest number of ids it takes (null if not stated).
- cart_write: the tool that changes the basket. mode = "per_item" when one call adds or removes units of one product, "replace" when one call sets the whole basket from a list. For per_item: id_arg, qty_arg, op_arg (null if none), add_value and remove_value (enum values copied exactly, null if none). For replace: list_arg, item_id_key, item_qty_key. fixed = other required arguments with their value, else {}.
- cart_view: a tool that shows the basket.
- order: the call that buys the basket: tool and args (fixed values copied exactly, e.g. an enum value meaning order or commit).
- decline: the call that refuses the request: tool and args.
- message: the tool that sends a message to the shopper: tool and text_arg.

Copy tool names, argument names and enum values exactly as written. Example of the form (not of any shop): {"search": {"tool": "find", "query_arg": "q", "page_arg": "after", "page_kind": "token", "fixed": {}, "price_arg": null, "category_arg": null, "order": null}, "browse": null, "facts": {"tool": "specs", "id_arg": "pid"}, "page": {"tool": "specs", "id_arg": "pid"}, "compare": null, "cart_write": {"tool": "basket", "mode": "per_item", "id_arg": "pid", "qty_arg": "n", "op_arg": "op", "add_value": "add", "remove_value": "drop", "list_arg": null, "item_id_key": null, "item_qty_key": null, "fixed": {}}, "cart_view": {"tool": "show_basket"}, "order": {"tool": "finish", "args": {"choice": "buy"}}, "decline": {"tool": "finish", "args": {"choice": "refuse"}}, "message": {"tool": "tell", "text_arg": "text"}}"""

MAP_AGAIN = """TOOLS:
{listing}

A FIRST MAP OF THESE TOOLS:
{map}

That map leaves unused: {gaps}. An unused part is a reason to read its description again, not proof that it serves a role. Give it a role the first map left null only when its description says it does that role's job: a value of a search tool's mode argument described as a filter, shelf or range search is the browse role (the same tool with that value fixed, its price and shelf arguments as the description gives them); a tool that takes one product id and opens, shows or returns that product (its listing, page, description or small print) is the page role, or the facts role when it is described as returning attributes or specifications. A tool about something other than the product itself (its reviews or questions, help, account status, anything the roles do not name) stays unused and the role stays null. Keep every role the first map filled as it is. Answer with the whole map as one JSON object."""

MAP_FIELDS = """You read one search result and one product record of a shop and name what each field holds. Answer with one JSON object.

row = keys of a search result row: id (the product id used by other tools), title, price, stock (in-stock flag), category (shelf name), seller, brand, text (description shown in the row). result = top-level keys of the search result: rows (the list of products), count (the total number of matches, null if none), next (the value to send back for further results, null if none). record = keys of the product record: facts (the attribute table: a key holding key-value pairs, or null when the record is one text), small_print (fine print, remarks or notes), description. Use null for anything not present. When a row is one text of key: value pairs, it is also shown as PAIRS: name the row's keys from those pairs exactly as for a row of fields, and set text_rows true; give null row keys only when no PAIRS are shown.

When the input is STOCK VALUES instead (texts the shop wrote in its stock field), answer only {"stock": {"<each text exactly as given>": true, false or null}}: true when the text says the product can be bought now, false when it says it cannot be bought now, null when it does not say.

Example of the form: {"row": {"id": "code", "title": "name", "price": "cost", "stock": "available", "category": "shelf", "seller": "shop", "brand": null, "text": "about"}, "result": {"rows": "items", "count": "total", "next": "cursor"}, "record": {"facts": "specs", "small_print": "notes", "description": "about"}, "text_rows": false}"""

ROWS = """You read one search result of a shop whose rows the code could not split, and list the products it shows. Answer with one JSON object only.

Input: RESULT, the search result exactly as the shop returned it; a long result comes in parts, each cut at a row or sentence end with the count of characters left out stated, and a later part starts where the one before it was cut. A task scores 1 only when the exact products are bought, so every id is copied character for character.
1. Find every product the result lists, in the order shown.
2. id: the product's identifier exactly as written, the value other tools take to open or buy it (not its title, and not a model number inside its text).
3. title: the product's name as written. price: its price as the number written, null when none is shown. in_stock: true or false when the result says whether it can be bought now, else null.
4. next: the value the result gives for asking for further results (a cursor, token or page value), copied exactly; null when none is shown.
A value not written in RESULT is null; never invent one.
Form (not any shop's data): {"rows": [{"id": "<id as written>", "title": "<name as written>", "price": 1250, "in_stock": true}], "next": null}"""

ROWS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rows", "next"],
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "title", "price", "in_stock"],
                "properties": {"id": {"type": "string"}, "title": {"type": "string"}, "price": {"type": ["number", "null"]}, "in_stock": {"type": ["boolean", "null"]}},
            },
        },
        "next": {"type": ["string", "null"]},
    },
}

INTAKE = """You read one shopping request and write the brief a shop assistant works from. Answer with one JSON object only.

1. lines: one line per product the shopper wants bought. Count lines by quantities and requirements: products joined by "and" in one phrase with one quantity are one line; a product gets its own line only when it has its own quantity or its own requirements. Never add a product the request does not ask to buy. "a pair" of a thing sold in pairs is quantity 1.
2. For each line write item (the product phrase the line's requirements are said of, in the words that ask the shop for it, as the request names it, never a noun only a story sentence uses; when the ask names the product only by a range or shelf, as in "something from the <shelf> range", item is that shelf's words), category (the shelf or category name when the request names one, else null), qty (a whole number; 1 when the request gives none), and wants: every requirement the request states for that product. A want has key (the attribute words as written), op (is, one_of, not, at_least, at_most, between, ref), values (the request's value words exactly: every alternative for one_of, the number and its unit as written for numbers) and quote (the request's exact words for it).
   - "up to X" about a value or a quantity is at_most X; "goes up to X" or "supports up to X" about what the product can do is at_least X.
   - A requirement said twice (restated in other words, or explained after a colon, a dash or "so") keeps every value it says: the same value said again is one want, taking the words the request states most exactly; a restatement that turns away values it names is a second want ("<A>, so nothing <B> or <C>" gives op is <A> and op not <B>, <C>), one that turns away only "any other" or "anything else" adds no want; op one_of only when it offers alternatives the shopper accepts.
   - When or how the order reaches the shopper (its arrival day, one parcel or several, the courier, tracking) is story, not a want; the product's own packaging, contents or condition is a want.
   - A requirement taken from another listing ("the same as the one sold by …", "whatever the page of … says") is op ref with values [] and relation: how the value stands to that listing's: same, different ("different from …" is ref, never not), at_least, at_most, within (values = [the distance as written]) or partner (the piece that goes with it); quote the whole pointer. Words that describe that other listing (its seller, its kind, the values that tell it apart, in a clause or brackets before or after the pointer) belong to the pointer and are never a want of the line: in "<attribute> the same as on <seller>'s <kind> listing, with <key> <value>, says", "with <key> <value>" tells that listing apart, so it is one ref want whose quote holds all of it.
   - When the request says how the listing writes a value ("the listing says …", "spelled …"), the want is one_of with both: the request's words and the same words in that spelling.
   - A requirement belongs to the line whose item words its sentence names, and to that line only. Read the product phrase whole first: when it holds every content word of one line's item (singular or plural, "and" or "or" between them), it names that line alone, even when a part of it is another line's item; with lines "<A>" and "<A> or <B>", "the <A>s and <B>s" names the second line only. The same holds for the line of each side of a relation (rule 5).
3. objective per line: "cheapest" only when the request asks for the lowest price, "best" (with best_key and best_dir max or min) when it asks for the most or least of an attribute, else "any". A sentence about budget is a cap, not an objective. "Lowest in total" makes total_objective "cheapest" and every line "cheapest".
4. cap_each per line when a price limit is for one unit of that product; total_cap when the limit is for the whole order. Write the number only (160, not "160 cents"), in the request's unit.
5. relations only when the request states them, each in one of these forms with quote (the request's exact words): fits, line a must fit, suit or be compatible with line b or with a product, model or part the request names: {"type": "fits", "line": 2, "with": 1, "quote": "…"} or {"type": "fits", "line": 2, "with": "<name or code as written>", "quote": "…"}; same or different, two lines share or differ in an attribute: {"type": "same", "lines": [1, 2], "key": "<attribute>", "quote": "…"}; if_then: {"type": "if_then", "if": {"line": 1, "key": "<attribute>", "op": "is", "values": ["<value>"]}, "then": {"line": 2, "key": "<attribute>", "op": "is", "values": ["<value>"]}, "quote": "…"}. A sentence about who or where the product is for is story, not a fits relation. An exception ("<K> must be <A>, except when <the other line's attribute> is <C>", or "unless …") is one if_then whose if is that attribute op not <C> and whose then is <K> is <A>: never a negated then, and never a separate unconditional want beside it.
6. in_stock: true when the request rules out sold-out items. messages and steps: the numbers the request allows, null if not stated.
7. procedures: every instruction about how to act (ask before buying, report first, wait for approval, put every question in one message, explain a decline), each once as a string of the request's exact words. decline_rule: what the request says to do when nothing fits, quoted, else null. nearest: when the request asks to be told or offered the listing that comes closest if nothing fits, its exact words for that; else null, also when it asks for no alternative.
8. story: every sentence about the shopper's situation, use, people or places rather than a requirement, copied whole, each with the line number it is about (null when about no single line).
9. checks: attributes the shopper says they will check or confirm themselves, with the value when one is given (null otherwise), each with its line.

Copy words exactly and never invent a value. Form (not any real request): {"lines": [{"item": "<item>", "category": null, "qty": 2, "objective": "cheapest", "best_key": null, "best_dir": null, "cap_each": null, "wants": [{"key": "<attribute>", "op": "is", "values": ["<value>"], "quote": "<request words>"}]}], "total_objective": "cheapest", "total_cap": 500, "in_stock": true, "messages": 3, "steps": 40, "relations": [], "procedures": ["<request words>"], "decline_rule": null, "nearest": null, "story": [{"text": "<sentence>", "line": 1}], "checks": [{"line": 1, "key": "<attribute>", "value": null}]}"""

INTAKE_TIE = """Two readings of one shopping request disagree, or a reading contradicts itself, on the POINTS TO DECIDE below them. For each point read the request's own words and decide what it says under the rules above (fix both readings when neither is right); keep every field the readings agree on. Answer with the whole corrected brief as one JSON object in the same form as the readings."""

RECORD = """You judge shop records against a shopper's requirements. A task scores 1 only when the finish is exactly right, so a requirement passes only when the record's own words show that this product meets it.

INPUTS:
- ITEM and REQUIREMENTS: "<n>. <attribute> <op> <values>", marked [disclosed by the shopper], [from a store notice] or [value read from another listing] when the request did not state it, then STORE USE: how the other opened records of this shelf write the attribute (the store's vocabulary for it).
- RECORDS: each record whole: id, title, shelf, price, stock, seller, brand, every fact, then every text field under its source label (page.* from the listing's page, record.* from its record, row.* from its search row); then the requirement numbers to judge on it, and any point another check raised on one of them.

PROCEDURE, for each record and each requirement number to judge on it:
1. Find every piece of evidence the record holds on the requirement: the key named as its attribute (own key), keys that hold the same attribute under another name (sibling keys), an options list, the title, the shelf, the page's fine print and memos, the description. The title is the store's own words for this product: when it names the required value and no other evidence on the record gives another value, the record says it (PASS, where title), also when the word stands inside a product name. A raised point says where to look again; it proves nothing by itself.
2. Judge by meaning what the evidence says this product is: PASS when it shows the product has the required value, FAIL when it shows the product has another value that does not meet the requirement, UNKNOWN when nothing in the record says it. When pieces of evidence disagree, decide by the one that describes this product: a memo or fine print that corrects a fact, else the product's own key, which an options field, the variant a listing shows or the title never overrides; when nothing settles it, UNKNOWN.
3. Words: the store may write the required value in another spelling, form, order, case or abbreviation (STORE USE shows its vocabulary); that is the value. A fact that lists several values (items split by commas, slashes or "and") says the product has each of them, also when they are the sizes or variants it is sold in. Words that name something else (a wider class, a related material, a near value, one of several variants offered as choices) are not the value.
4. Numbers: read each number with its unit and its role (one size, a range, a capacity, a limit) and convert units before comparing. A bound passes only when the product's value meets it; a range meets a bound only when the whole range lies inside it.
5. "within" (values = [center, distance]): convert the record's number for the attribute, the center and the distance to one unit; PASS when the number lies no farther from the center than the distance, FAIL when farther, UNKNOWN with no number for the attribute.
6. "not X": PASS when the evidence shows a value that is not X, FAIL when the value is X or a kind of X ("pu leather" is leather), UNKNOWN when nothing says the attribute.
7. A requirement with no attribute named is a feature the product has: PASS when the record says the product has it, FAIL only when it says the product lacks it, else UNKNOWN.
8. A requirement [disclosed by the shopper] is the shopper's own words for a value: the store's own spelling of that same value passes; another value, a near value or a wider class does not.
9. "fits" (values = identifiers of the product it must fit): PASS when the record's own words name one of those identifiers, or a family that covers it ("every <code>xx model" covers <code>10 unless an exception names it), as what it fits; FAIL when the record names what it fits and none covers them; UNKNOWN when it says nothing about fit. A line "Part <X> also fits <M>" proves this record only when X is this record's own code.
10. FAIL only when the record shows another value; a record you cannot read for the attribute is UNKNOWN, never FAIL.
11. support, on each PASS: direct when the evidence states the required value itself (in any spelling, form, abbreviation or notation of it, rule 3) or a number that meets the bound; inferred when the PASS rests on your reasoning rather than on the record's statement: a narrower kind or a member of the value written in its own words, the value with one of its words missing, or descriptive wording in the record's text that does not state the attribute. An explicit statement in the text ("<attribute>: <value>") and fine print that corrects a fact are direct. FAIL and UNKNOWN are direct.

OUTPUT FORMAT:
One JSON object: for every record and every requirement number to judge on it, label, where (the evidence that decided: own_key, sibling, options, title, shelf, small_print for fine print or a memo, description, none), quote (that evidence's exact words, copied from the record; empty only when UNKNOWN) and support (rule 11).

EXAMPLE (not any shop's data):
{"verdicts": [{"id": "<record id>", "want": 1, "label": "PASS", "where": "own_key", "quote": "<exact words>", "support": "direct"}]}"""

RECORD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "want", "label", "where", "quote", "support"],
                "properties": {
                    "id": {"type": "string"},
                    "want": {"type": "integer"},
                    "label": {"type": "string", "enum": ["PASS", "FAIL", "UNKNOWN"]},
                    "where": {"type": "string", "enum": ["own_key", "sibling", "options", "title", "shelf", "small_print", "description", "none"]},
                    "quote": {"type": "string"},
                    "support": {"type": "string", "enum": ["direct", "inferred"]},
                },
            },
        }
    },
}

CHAIN_PARSE = """You read one pointer sentence: a shopper's requirement takes its value from another listing, and the SENTENCE describes that listing. The words that point at it are marked «…»; with no mark the whole sentence is the pointer. Say what the sentence tells about that pointed listing, in the sentence's own words only. Answer with one JSON object only.
- party: the seller, brand or shop the sentence names for the pointed listing, else null.
- kind: the kind of product the pointed listing is, as written, else null.
- described: every value the sentence says of the pointed listing to tell it apart, inside the mark or outside it (a clause or brackets about that listing), as {"key": "<the attribute the value is of>", "value": "<value words>"}; a number keeps its unit. The seller and the kind are not described values, and neither is a requirement of a product the shopper buys (what that product must be or have, such as "<attribute> is at least <number>"), even beside the pointer.
- attribute: the attribute the shopper's product takes from that listing.
- relation: how the shopper's value stands to the listing's: "same" (the same, identical, matching), "different" (different from, not the one), "at_least", "at_most", "within" (within a distance either way; distance = that distance as written), "partner" (the partner piece, what goes with it).
Every word you write must stand in the sentence. Form (not any shop's data): {"party": "<seller>", "kind": "<product>", "described": [{"key": "<attribute>", "value": "<value>"}], "attribute": "<attribute>", "relation": "within", "distance": "<number and unit>"}"""

CHAIN_PARSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["party", "kind", "described", "attribute", "relation", "distance"],
    "properties": {
        "party": {"type": ["string", "null"]},
        "kind": {"type": ["string", "null"]},
        "described": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["key", "value"], "properties": {"key": {"type": "string"}, "value": {"type": "string"}}}},
        "attribute": {"type": ["string", "null"]},
        "relation": {"type": "string", "enum": ["same", "different", "at_least", "at_most", "within", "partner"]},
        "distance": {"type": ["string", "null"]},
    },
}

CHAIN_READ = """A shopper's requirement takes its value from another listing. You read one PAGE the code found for the POINTER and say what it gives. Answer with one JSON object only.

INPUTS:
- POINTER: the sentence describing the listing.
- ATTRIBUTE: what the shopper's product takes from that listing.
- PAGE: the listing's whole record: title, shelf, seller, facts and page text.

PROCEDURE:
1. described: true when the PAGE is the listing the POINTER describes: its seller and kind of product match and every described value holds in the same sense (convert units before comparing; a value the page states with "not", "no" or "without" does not hold); false only when the seller, the kind or a described value clearly differs. Words that send the ATTRIBUTE elsewhere ("rather than this item", "not to this one", "check a different product") never mean the page is not the described listing: they are its next.
2. next: when the page sends the ATTRIBUTE to a further listing ("for <attribute>, see …", "look to …", "go by …", "not this one"), copy that whole pointer sentence; value is then null, even when the page states its own value.
3. value: otherwise the page's own value of the ATTRIBUTE as written, with its unit: a fact first, then the page text.
4. pairs: when the page states a pairing rule ("<A> goes with <B>", "the rule pairs <A> with <B>"), every pair of the rule as {"from": "<A>", "to": "<B>"}, else [].
5. quote: the page's exact words that give next, value or the rule.

EXAMPLE (not any shop's data):
{"described": true, "next": null, "value": "<value>", "pairs": [{"from": "<A>", "to": "<B>"}], "quote": "<page words>"}"""

CHAIN_READ_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["described", "next", "value", "pairs", "quote"],
    "properties": {
        "described": {"type": "boolean"},
        "next": {"type": ["string", "null"]},
        "value": {"type": ["string", "null"]},
        "pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["from", "to"], "properties": {"from": {"type": "string"}, "to": {"type": "string"}}}},
        "quote": {"type": "string"},
    },
}

FIT = """A shopper buys a main product and parts that must fit it. You read the MAIN record and the PART records, copy the identifiers each record carries, and say which parts fit the main product. A task scores 1 only when every part bought fits the main product bought. Answer with one JSON object only.

INPUTS:
- MAIN: the main product's whole record.
- PARTS: the candidate parts' whole records, each beginning "ID <id>".

PROCEDURE:
1. For MAIN and for every part, copy exactly from that record's own words:
   - own: the identifiers of the record itself: its model, number, series or code, a code in its title, names it is also sold as.
   - fits: the identifiers it says it fits, takes, goes with or is compatible with ("takes part series <X>", "compatible with <M>", "fits <M>"), and a family statement as written ("every <code>xx model"). An identifier it says it does not fit, or names as an exception, never goes in fits.
   A word that only gives a size, a color or a material, and a sentence, are no identifier. A line "Part <X> also fits <M>" puts <M> in the fits of record <X> only.
2. For every part a label: PASS when the part's fits name MAIN's own identifier or a family that covers it ("every <code>xx model" covers <code>10 unless an exception names it), or MAIN's fits name the part's own identifier or its series; FAIL when the part names fit identifiers or a series and none covers MAIN; UNKNOWN when neither record says anything about fit.
3. A line "Part <X> also fits <M>" says that part X fits M: it proves the record it stands on only when X is that record's own code.
4. quote: the exact words (of the part, or of MAIN) that decide; empty when UNKNOWN.

OUTPUT FORMAT:
One JSON object: records (MAIN and every part, each with id, own and fits) and verdicts (one per part).

EXAMPLE (not any shop's data):
{"records": [{"id": "<main id>", "own": ["<model code>"], "fits": ["<part series>"]}, {"id": "<part id>", "own": ["<part code>"], "fits": []}], "verdicts": [{"id": "<part id>", "label": "PASS", "quote": "<exact words>"}]}"""

FIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["records", "verdicts"],
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "own", "fits"],
                "properties": {"id": {"type": "string"}, "own": {"type": "array", "items": {"type": "string"}}, "fits": {"type": "array", "items": {"type": "string"}}},
            },
        },
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "quote"],
                "properties": {"id": {"type": "string"}, "label": {"type": "string", "enum": ["PASS", "FAIL", "UNKNOWN"]}, "quote": {"type": "string"}},
            },
        },
    },
}

STORY = """A shopper's request has story sentences that may hint at needs the shopper did not state, and the shopper tells such a need only when a message asks about that attribute by name, one attribute per message. You decide, for each line of the order, which of its candidate attributes to ask about first, and, for each of the request's procedures, whether it needs a message of its own and how it fixes the questions. A task scores 1 only when the finish is exactly right, unstated needs included.

INPUTS:
- LINES: per line "line <n>: <item>", then STATED (the line's stated requirements as key op value) and CANDIDATES, one row per key: "- <key> | <opened records carrying it> | <values seen>".
- SENTENCES: "<k>. (line <n>) <sentence>", the request's story sentences; n is the line the sentence is about (None: no single line). There may be none.
- PROCEDURES: "<k>. <instruction>", the request's instructions about how to act. There may be none.

PROCEDURE:
1. For each sentence, decide which property of the product it describes, if any, by reading what the sentence says about the product and how this item is chosen.
2. Put that property in "pointed" of the sentence's line (a sentence marked None belongs to the line whose product it describes, if any), at most two per sentence, best first: a CANDIDATES key of that line copied exactly when one is that property (branch false); only when none is, the property's own plain name (branch true). In quote, copy the sentence's words that describe the property exactly.
3. A sentence saying the shopper has not decided, does not mind, or wants to see the choices on some aspect of the product points at that aspect.
4. A sentence points at nothing when it describes no property of the product (only when, where or how often it is used), only says why something new is needed, or restates or explains a STATED requirement (it names the stated value, or gives the reason that value is wanted); for such a sentence give neither the stated key nor another key for the same property.
5. In "order", give the line's eight CANDIDATES keys most likely to be an unstated need, most likely first (all of them when there are fewer): first the attributes on which a buyer of this item would hold a firm wish before buying it, judged for this item and this story, then the rest. Among keys alike, a key whose values differ across the records goes before a key with one value, since only then does the wish change which record is right; then more records first.
6. For each procedure, say in "message" whether following it takes a message to the shopper of its own and when: "order" (a message before ordering, such as a confirmation, a report or a wait for approval), "decline" (a message before declining), else "none". The order call itself buys and the decline call itself tells the shopper that nothing fits, so a procedure that only says to buy, to decline or to say so when nothing fits takes none, and so does a limit on messages or steps.
7. For each procedure, also say how it fixes the questions themselves: "together" true when it says to put every question into one message; "questions", the number of questions it allows or asks for (null when it names none: a limit on messages or steps is no number of questions); "ask", each question it requires, with the line it is about, the attribute as the procedure words it, and quote, the procedure's words that require it. A procedure that only invites questions, naming no attribute, number or form, requires none. A procedure that says nothing about the questions gives false, null and [].

OUTPUT FORMAT:
One JSON object with one entry per line, candidate keys copied exactly as that line's CANDIDATES writes them, and one entry per procedure, as in the example.

EXAMPLE (not any real request):
{"lines": [{"line": 1, "pointed": [{"sentence": 2, "key": "<candidate key>", "branch": false, "quote": "<the sentence's words>"}], "order": ["<candidate key>", "<candidate key>", "<candidate key>"]}], "procedures": [{"procedure": 1, "message": "none", "together": false, "questions": null, "ask": []}]}"""

STORY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lines", "procedures"],
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["line", "pointed", "order"],
                "properties": {
                    "line": {"type": "integer"},
                    "pointed": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["sentence", "key", "branch", "quote"],
                            "properties": {"sentence": {"type": "integer"}, "key": {"type": "string"}, "branch": {"type": "boolean"}, "quote": {"type": "string"}},
                        },
                    },
                    "order": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "procedures": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["procedure", "message", "together", "questions", "ask"],
                "properties": {
                    "procedure": {"type": "integer"},
                    "message": {"type": "string", "enum": ["order", "decline", "none"]},
                    "together": {"type": "boolean"},
                    "questions": {"type": ["integer", "null"]},
                    "ask": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": False, "required": ["line", "key", "quote"], "properties": {"line": {"type": "integer"}, "key": {"type": "string"}, "quote": {"type": "string"}}},
                    },
                },
            },
        },
    },
}

READ = """You read one shopper reply to the assistant's message and say what it tells. A task scores 1 only when the finish is exactly right, every requirement the shopper states included. Answer with one JSON object only.

INPUTS:
- LINES: "line <n>: <item>", the products of the order.
- ASKED (line <n>): an attribute the message asked about, one row each (several when the message asked several, none when it asked none), then MESSAGE: the message itself.
- SEEN, only on a second reading: the values the store's records give for the asked attribute.
- REPLY: the shopper's reply.
- REQUIREMENTS: per line the requirements as they stand, numbered "<k>. <attribute> <op> <values>", which a change may refer to.

PROCEDURE:
1. kind: "told" (the shopper states a requirement), "no_preference" (they do not mind), "not_understood" (they did not follow, ask to repeat or say it more simply), "one_at_a_time" (they ask for one question at a time or refuse several at once), "other".
2. For told give every requirement stated: line (the number of the line whose product the reply names; when it names none, the line the message asked that attribute for), key (the attribute in the reply's own words; an ASKED attribute only when the reply names none), op (is, one_of, not, at_least, at_most, between, ref), values (the reply's value words exactly; numbers with their unit), quote (the reply's exact words), relation ("none" except for ref).
3. When the reply names an attribute and then a word or code that the shopper holds to for it, that word or code is the value and the kind is told, even when it is a short common word such as a pronoun, a preposition or a two-letter code.
4. Read the polarity from what the shopper turns away. A reply that refuses every option other than a value ("every <attribute> other than <value> is turned away", "nothing apart from <value> will do") wants that value: write it with its own operator (is, at_least, at_most, between, ref), never op not. Only a reply that refuses the value itself ("any <attribute> but <value> is fine", "no <value>") is op not.
5. op ref only when the reply names another product's page, listing, seller or record as the source of the value ("whatever is on the page of <seller>'s listing of <item>"); quote that whole pointer, values [] and relation: how the value stands to that listing's: same, different, at_least, at_most, within (values = [the distance as written]) or partner (the piece that goes with it). A value "different from" what another listing gives is op ref with relation different, never op not.
6. A value said directly is never ref, whatever the verb: "<attribute> has got to match <value>" is op is with value <value>.
7. SEEN only helps to recognise a value the reply itself says; never write a value from SEEN that the reply does not say.
8. changes: what the reply changes in the order beyond the asked attribute, one entry per change, each with quote, the reply's exact words that state it: "cap" (a new budget of the order as a whole; value the amount as the reply writes it), "cap_each" (the same for one line), "drop" (the shopper gives up one of the REQUIREMENTS, naming its attribute or value in words: line and want are its numbers, and quote holds those words; words that name no requirement drop nothing), "stock" (a listing can no longer be bought). A reply that only says the shopper does not mind about the asked attribute changes nothing else: give no change for it. Fields a change does not use are null; none when the reply changes nothing else.

OUTPUT FORMAT:
One JSON object, as in the example (not any real reply).

EXAMPLE:
{"kind": "told", "needs": [{"line": 1, "key": "<attribute>", "op": "is", "values": ["<value>"], "quote": "<reply words>", "relation": "none"}], "changes": [{"change": "drop", "line": 2, "want": 1, "value": null, "quote": "<reply words>"}]}"""

READ_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "needs", "changes"],
    "properties": {
        "kind": {"type": "string", "enum": ["told", "no_preference", "not_understood", "one_at_a_time", "other"]},
        "needs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["line", "key", "op", "values", "quote", "relation"],
                "properties": {
                    "line": {"type": "integer"},
                    "key": {"type": "string"},
                    "op": {"type": "string", "enum": ["is", "one_of", "not", "at_least", "at_most", "between", "ref"]},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                    "relation": {"type": "string", "enum": ["none", "same", "different", "at_least", "at_most", "within", "partner"]},
                },
            },
        },
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["change", "line", "want", "value", "quote"],
                "properties": {
                    "change": {"type": "string", "enum": ["cap", "cap_each", "drop", "stock"]},
                    "line": {"type": ["integer", "null"]},
                    "want": {"type": ["integer", "null"]},
                    "value": {"type": ["number", "null"]},
                    "quote": {"type": "string"},
                },
            },
        },
    },
}

NOTICE = """You read the notices a store sent during a shopping order (sentences riding on its results, the reasons a checkout or a decline was refused) and say what each changes in the order now. A task scores 1 only when the finish is exactly right under the order as it stands after every change. Answer with one JSON object only.

INPUTS:
- LINES: per line "line <n>: <item>", then its requirements as they stand, numbered "<k>. <attribute> <op> <values>".
- NOTICES: the store's sentences, one per row.

PROCEDURE:
1. Give one entry per notice, in the order of NOTICES. A notice often apologises or explains first: read what it then states. Its change is one of these, or "none" when it states no change to the order:
   - "cap": the order's budget, its price ceiling as a whole, is now a new amount; "cap_each": the same for one line (line). value is the amount as a number, as the notice writes it.
   - "price": the price of a listing has changed (its number moved, it costs differently now).
   - "stock": a listing can no longer be bought (gone, sold out, taken).
   - "drop": a requirement no longer applies: the shopper gives it up or relaxes it. line and want are that requirement's numbers in LINES. A notice drops only a requirement whose attribute or value it names in words; a notice that names none drops nothing, however it is worded. When the notice could mean requirements of several lines, give line null.
   - "add": a new requirement: line, key, op (is, one_of, not, at_least, at_most, between) and values.
2. id only when the notice writes the listing's id, else null.
3. quote: the notice's own words that say the change, copied exactly; for a drop or an add, the words that name the requirement.
4. Fields a change does not use are null, values [].

OUTPUT FORMAT:
One JSON object with one entry per notice, as in the example (not any real order).

EXAMPLE:
{"changes": [{"line": 1, "want": 2, "change": "drop", "key": "<attribute>", "op": null, "values": [], "value": null, "id": null, "quote": "<the notice's words>"}, {"line": null, "want": null, "change": "price", "key": null, "op": null, "values": [], "value": null, "id": null, "quote": "<the notice's words>"}]}"""

NOTICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["changes"],
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["line", "want", "change", "key", "op", "values", "value", "id", "quote"],
                "properties": {
                    "line": {"type": ["integer", "null"]},
                    "want": {"type": ["integer", "null"]},
                    "change": {"type": "string", "enum": ["drop", "add", "cap", "cap_each", "stock", "price", "none"]},
                    "key": {"type": ["string", "null"]},
                    "op": {"type": ["string", "null"]},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "value": {"type": ["number", "null"]},
                    "id": {"type": ["string", "null"]},
                    "quote": {"type": "string"},
                },
            },
        }
    },
}

JUDGE = """You make the final check of a basket before it is written to the cart and ordered. A task scores 1 only when the finish is exactly right: every stated and disclosed requirement met, the cheapest or best option where asked, quantities and caps right, every stated procedure followed. Answer with one JSON object only.

INPUTS:
- REQUEST, TOTAL CAP, STORE NOTICES (a notice overrides the request), SHOPPER DIALOGUE (every question sent and the shopper's reply, word for word) and UNHANDLED (relations and conflicts of the request that no code acted on, with their sentences).
- Per line: quantity, item, objective and cap; EVIDENCE, each requirement as it stands now with where it came from (the request's words, the shopper's reply, a notice, or the pages of a chain) and the code's verdict on the pick with the record's words it rests on; the PICK's whole record; up to three OTHER CANDIDATES (records the code's checks pass or leave unproven; for a part that must fit, only parts that fit the main line's pick); the FIT pair chosen with the words that join it; each CHAIN NOT FINISHED.

PROCEDURE:
1. Read the request, the dialogue and the notices yourself. A requirement that EVIDENCE lists wrongly or misses, or that a later reply or notice changed, is a problem of its line.
2. Read the pick's whole record against every requirement. Where your reading differs from the code's verdict, name the requirement and quote the record's words that decide it; the code has the line's records judged again on your point, so a reading you cannot ground in the record's words is no problem.
3. Check each UNHANDLED item and each CHAIN NOT FINISHED against the picks it concerns.
4. For each line say ok, or the problem. In use, give the OTHER CANDIDATE to take instead (its id) only when it meets every requirement where the pick does not; null when none does.
5. pointers: per line, identifiers of other products the pick's own words point to ("see part …", "refer to …", "fits: …", "compatible part …"), so the code opens and judges them.

OUTPUT FORMAT:
One JSON object as in the example (not any real basket).

EXAMPLE:
{"order": true, "lines": [{"line": 1, "ok": true, "problem": "", "use": null}, {"line": 2, "ok": false, "problem": "<the requirement the pick fails, with the record's words>", "use": "<id>"}], "pointers": [{"line": 2, "text": "<identifier>"}]}"""

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["order", "lines", "pointers"],
    "properties": {
        "order": {"type": "boolean"},
        "pointers": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False, "required": ["line", "text"], "properties": {"line": {"type": "integer"}, "text": {"type": "string"}}},
        },
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["line", "ok", "problem", "use"],
                "properties": {"line": {"type": "integer"}, "ok": {"type": "boolean"}, "problem": {"type": "string"}, "use": {"type": ["string", "null"]}},
            },
        },
    },
}

AUDIT = """The assistant is about to decline a shopping request because some lines have no record proven to meet their requirements. You check that evidence once more. A task scores 1 only when the finish is exactly right, and a decline is right only when the store holds no product meeting every requirement of a line. Answer with one JSON object only.

INPUTS:
- REQUEST.
- Per line without a pick: its REQUIREMENTS numbered, its OPENED RECORDS that no requirement fails for certain, each whole with OUR VERDICTS (PASS, FAIL or UNKNOWN per requirement number), and its ROWS NOT OPENED (id, title, shelf, price, stock).

PROCEDURE:
1. For each opened record and each requirement OUR VERDICTS hold UNKNOWN, read the record whole: when its own words prove the requirement, wherever in the record they stand, give a fit with the record id, the requirement number and the record's exact words. The code has the record judged again with your words beside it.
2. Never give a fit on a guess, a near value, a wider category or a value the record only offers as one choice: an UNKNOWN that no words of the record settle stays UNKNOWN.
3. In open, name up to four ROWS NOT OPENED whose title, shelf or price make them the likeliest to meet every requirement of the line, likeliest first; none when no row is likely.

OUTPUT FORMAT:
One JSON object with one entry per line, as in the example (not any real request).

EXAMPLE:
{"lines": [{"line": 2, "fits": [{"id": "<record id>", "want": 1, "quote": "<the record's words>"}], "open": ["<row id>"]}]}"""

AUDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lines"],
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["line", "fits", "open"],
                "properties": {
                    "line": {"type": "integer"},
                    "fits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "want", "quote"],
                            "properties": {"id": {"type": "string"}, "want": {"type": "integer"}, "quote": {"type": "string"}},
                        },
                    },
                    "open": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

SEARCH = """You write the next searches for one line of a shopping order in the store's own words. A task scores 1 only when the finish is exactly right, so the line needs a listing proven to meet every requirement, and every search spends one step. Answer with one JSON object only.

INPUTS:
- LINE: the product, its objective and price cap.
- REQUIREMENTS: each requirement numbered, then STORE USE: how the opened listings of this shelf write that attribute (their key and the values seen).
- DROPPED: requirements the shopper gave up; never search for them.
- SEARCHES: the searches sent so far, each with the rows it found, the pages read and whether more pages remain.
- OPENED: listings read, each with its title and its verdict on each requirement number; ROWS: listings found but not read yet, with title, shelf and price.
- LIMIT: the most characters one search takes.

PROCEDURE:
1. For each requirement, find how this store writes it: the key and value words in STORE USE, the titles and the facts (an abbreviation, a unit written another way, another word order, the store's own term for the same thing). Write them in terms.
2. Write up to two new searches that should find listings meeting the requirements no opened listing proves yet: the store's value words first, then the word the store's titles use for the product. The store returns only listings holding every word of a search, so use few words, only words the requirements, titles or facts give (a search that found nothing is retried with the words exactly as the rows write them), never a search of SEARCHES again in any word order, and stay within LIMIT.
3. Give no searches when no new search can find such listings.

OUTPUT FORMAT:
One JSON object as in the example (not any real store).

EXAMPLE:
{"terms": [{"want": 1, "store_words": "<key>: <value as the store writes it>"}], "searches": [{"text": "<store words> <product word>", "why": "<the requirement it serves and why these words>"}]}"""

SEARCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["terms", "searches"],
    "properties": {
        "terms": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["want", "store_words"], "properties": {"want": {"type": "integer"}, "store_words": {"type": "string"}}}},
        "searches": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["text", "why"], "properties": {"text": {"type": "string"}, "why": {"type": "string"}}}},
    },
}

ACTOR = """You choose the next store calls for one line of a shopping order that the code's own search ladder has not settled. A task scores 1 only when the finish is exactly right, so the line needs a record proven to meet every requirement, and every call you give spends one of the steps left. Answer with one JSON object only.

INPUTS:
- WHY: what brought the line here: the ladder of searches is spent, a result could not be read, a round of opening brought no new passing record, or the final check rejected the pick without naming another.
- LINE: the item, quantity, objective and cap, then each requirement with how many opened records pass, fail or leave it unproven.
- TOOLS: the store's tools for reading (searching, listing, opening, comparing) with their argument schemas.
- SEARCHES: the line's searches so far, each with its arguments, the rows it brought and whether it ended.
- CANDIDATES: the line's best records with the code's verdict on each requirement.
- SET ASIDE: what the code set aside and why (rows, values, searches); something set aside may still hold the answer.
- RECENT: the last store calls with their raw results and errors.
- RELATIONS: how the line relates to the other lines, and what no code acted on.
- LEFT: steps and money left.

PROCEDURE:
1. Find what blocks the line: a value the store spells or indexes differently, a shelf or listing not tried, a record not opened, an argument the store refused, something set aside that may hold the answer, a relation not acted on.
2. Give at most two actions, best first, each one call of TOOLS: tool, args (each argument's name and its value written as the schema types it; a list as JSON) and why. Use only ids written in CANDIDATES, SET ASIDE or RECENT. Never repeat a call of SEARCHES or RECENT with the same arguments.
3. stop: true, with no actions, when no call of TOOLS can bring new evidence for the line.

OUTPUT FORMAT:
One JSON object as in the example (not any real order).

EXAMPLE:
{"actions": [{"tool": "<tool name>", "args": [{"name": "<argument>", "value": "<value>"}], "why": "<what the call can show>"}], "stop": false}"""

ACTOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["actions", "stop"],
    "properties": {
        "stop": {"type": "boolean"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tool", "args", "why"],
                "properties": {
                    "tool": {"type": "string"},
                    "why": {"type": "string"},
                    "args": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": False, "required": ["name", "value"], "properties": {"name": {"type": "string"}, "value": {"type": "string"}}},
                    },
                },
            },
        },
    },
}

class Pace:

    def __init__(self) -> None:
        self.lock, self.tokens, self.at = threading.Lock(), 1.0, time.monotonic()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(float(BURST), self.tokens + (now - self.at) * RATE)
                self.at = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                need = (1.0 - self.tokens) / RATE
            time.sleep(need)


class Env:

    def __init__(self, problem_data: dict, pace: Pace, trace: list) -> None:
        environment = problem_data["environment"]
        self.binding, self.view = dict(environment["binding"]), environment["policy_view"]
        self.step, self.finished, self.over = 0, bool(self.view.get("finished")), False
        self.shut: set = set()
        self.durations: list[float] = []
        self.recent: list[dict] = []
        self.max_steps = int(self.view.get("max_steps") or 0)
        self.pace, self.trace, self.client = pace, trace, ProxyClient(timeout=60, max_retries=2, retry_delay=1)
        self.t0 = time.monotonic()

    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def per_call(self) -> float:
        recent = self.durations[-10:]
        return sum(recent) / len(recent) if recent else 2.0

    def call(self, name: str, args: dict) -> dict:
        if self.finished or self.over or self.left() <= 0:
            return {"observation": None, "finished": self.finished, "error": "no step left" if not self.finished else "finished"}
        if name in self.shut:
            return {"observation": None, "finished": False, "error": "tool shut: its budget is spent"}
        payload = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
        conflicts = 0
        for attempt in range(5):
            if TASK_S + 8 - self.elapsed() < 3:
                trace_row(self.trace, "call_fail", step=payload["step"], tool=name, status="time")
                break
            self.client.timeout = max(3, min(60, int(TASK_S + 8 - self.elapsed())))
            self.pace.wait()
            sent = time.monotonic()
            result = self.client.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                self.durations.append(time.monotonic() - sent)
                self.step = data["step"] if isinstance(data.get("step"), int) else payload["step"]
                self.finished = bool(data.get("finished"))
                obs = data.get("observation")
                error = obs.get("error") if isinstance(obs, dict) and isinstance(obs.get("error"), str) else None
                if error and error.split()[0].strip(" :.,;").lower() == "budget_exceeded":
                    if self.left() <= 0:
                        self.over = True
                    else:
                        self.shut.add(name)
                        trace_row(self.trace, "meter_spent", tool=name, step=self.step)
                trace_row(self.trace, "call", step=self.step, tool=name, args=args, obs=obs, finished=self.finished)
                self.recent = self.recent[-7:] + [{"step": self.step, "tool": name, "args": args, "error": error, "result": json.dumps(obs, ensure_ascii=False, default=str) if not isinstance(obs, str) else obs}]
                refusal = data.get("refusal")
                return {"observation": obs, "finished": self.finished, "error": error, "refusal": refusal if isinstance(refusal, list) else [refusal] if isinstance(refusal, str) else []}
            failure = result.error or {}
            status, body = failure.get("status"), str(failure.get("body") or "")
            trace_row(self.trace, "call_fail", step=payload["step"], tool=name, status=status, body=body[:300])
            if status == 409:
                conflicts += 1
                named = [int(x) for x in re.findall(r"\d+", body) if int(x) != payload["step"] and 0 < int(x) <= max(1, self.max_steps) + 1]
                if conflicts > 2:
                    self.over = True
                    self.recent = self.recent[-7:] + [{"step": payload["step"], "tool": name, "args": args, "error": "conflict", "result": body[:600]}]
                    trace_row(self.trace, "handoff", reason="step conflict the server's replies do not resolve: no further call", body=body[:600], step=payload["step"])
                    break
                self.step = named[0] - 1 if len(set(named)) == 1 else payload["step"]
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                break
            time.sleep(min(8.0, 0.5 * 2**attempt) * (0.8 + 0.4 * random.random()))
        return {"observation": None, "finished": self.finished, "error": "transport"}

def parse_json(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                return None
    return None


class Brain:

    def __init__(self, env: Env, pace: Pace, trace: list) -> None:
        self.env, self.pace, self.trace = env, pace, trace
        self.spent, self.kept, self.clients, self.lock = 0.0, 0.0, {}, threading.Lock()
        self.wrote: dict[str, int] = {}
        self.adapting = False
        self.judged = False

    def client(self) -> ProxyClient:
        limit = int(max(8, min(60, TASK_S - self.env.elapsed() - 8)))
        if limit not in self.clients:
            self.clients[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
        return self.clients[limit]

    def room(self, seat: str, chars: int) -> list[tuple[str, str]]:
        model, effort, kind = SEATS[seat]
        steps = [(model, effort)] + (STEPS[STEPS.index((model, effort)) + 1 :] if (model, effort) in STEPS else [])
        money = TASK_MONEY if kind in ("final", "change") or self.adapting or self.judged else WALK_MONEY
        fits = lambda setting: self.spent + self.estimate(seat, setting, chars) <= money
        chosen = [s for s in steps if fits(s)] or (steps[-1:] if kind != "walk" else [])
        if not chosen:
            return []
        other = {HAIKU: (SONNET, "low"), SONNET: (HAIKU, "medium")}[chosen[0][0]]
        return [chosen[0], other if fits(other) else chosen[-1]]

    def estimate(self, seat: str, setting: tuple[str, str], chars: int) -> float:
        fresh, out, _, _ = PRICE[setting[0]]
        return (chars / 3.5 * fresh + max(WRITES[setting[1]], self.wrote.get(seat, 0)) * out) / 1e6

    def price(self, model: str, usage: dict) -> float:
        upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost")
        if isinstance(usage.get("cost"), (int, float)):
            return float(usage["cost"]) + (float(upstream) if isinstance(upstream, (int, float)) else 0.0)
        fresh, out, read, write = PRICE[model]
        details = usage.get("prompt_tokens_details") or {}
        cached, written = int(details.get("cached_tokens") or 0), int(usage.get("cache_write_tokens") or details.get("cache_write_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
        return ((prompt - cached - written) * fresh + cached * read + written * write + int(usage.get("completion_tokens") or 0) * out) / 1e6

    def send(self, model: str, effort: str, messages: list, schema: dict | None, name: str) -> tuple[str | None, str]:
        body: dict[str, Any] = {"model": model, "messages": messages, "reasoning": {"effort": effort}}
        if schema and schema.get("properties"):
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        for attempt in range(3):
            if TASK_S - self.env.elapsed() < 10:
                return None, "time"
            self.pace.wait()
            result = self.client().post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                cost = self.price(model, usage)
                with self.lock:
                    self.spent += cost
                    self.kept += cost if SEATS[name.upper()][2] in ("final", "change") or self.adapting or self.judged else 0.0
                    self.wrote[name.upper()] = max(self.wrote.get(name.upper(), 0), int(usage.get("completion_tokens") or 0))
                choice = data["choices"][0]
                message = choice.get("message") or {}
                trace_row(self.trace, "model", seat=name, model=model, effort=effort, cost=round(cost, 5), usage=usage, finish=choice.get("finish_reason"), reply=message.get("content"))
                return message.get("content"), str(choice.get("finish_reason") or "")
            failure = result.error or {}
            status = failure.get("status")
            trace_row(self.trace, "model_fail", seat=name, model=model, status=status, body=str(failure.get("body") or "")[:300])
            if status not in (401, 408, 425, 429, 500, 502, 503, 504, 529, None):
                return None, f"http {status}"
            time.sleep(min(6.0, 1.0 * 2**attempt) * (0.8 + 0.4 * random.random()))
        return None, "failed"

    def ask(self, seat: str, fixed: str, task: str, now: str, schema: dict, check=None) -> dict | None:
        chars = len(fixed) + len(task) + len(now)
        self.judged = self.judged or SEATS[seat][2] == "final"
        tries = self.room(seat, chars)
        if not tries:
            trace_row(self.trace, "money_stop", seat=seat, spent=round(self.spent, 4), estimate=round(self.estimate(seat, SEATS[seat][:2], chars), 4))
            return None
        if tries[0] != SEATS[seat][:2]:
            trace_row(self.trace, "money_step", seat=seat, setting=tries[0], spent=round(self.spent, 4), estimate=round(self.estimate(seat, tries[0], chars), 4))
        user = ([{"type": "text", "text": task + "\n\n", "cache_control": {"type": "ephemeral"}}] if task else []) + [{"type": "text", "text": now}]
        messages = [{"role": "system", "content": [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": user}]
        for turn in range(2):
            text, finish = self.send(*tries[0], messages, schema, seat.lower())
            if text is None and turn == 0 and finish not in ("time",):
                text, finish = self.send(*tries[1], messages, schema, seat.lower())
            answer = parse_json(text)
            problem = "the reply was not one JSON object" if not isinstance(answer, dict) else (check(answer) if check else None)
            if problem is None:
                return answer
            trace_row(self.trace, "reask", seat=seat, problem=problem)
            chars += len(text or "") + len(problem) + 60
            tries = self.room(seat, chars) if text is not None else []
            if not tries:
                return None
            messages = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": f"Your answer cannot be used: {problem}. Answer again with only the JSON object."}]
        return None

READ_ROLES, HEAD_KEYS = ("search", "browse", "facts", "page", "compare"), ("title", "price", "stock", "category", "seller", "brand")


ROLE_ARGS = {"search": ("query_arg",), "facts": ("id_arg",), "page": ("id_arg",), "compare": ("ids_arg",), "cart_write": ("id_arg", "list_arg"), "message": ("text_arg",)}


def tool_props(tool: dict) -> dict:
    fn = tool.get("function", tool)
    return ((fn.get("parameters") or fn.get("input_schema") or {}).get("properties")) or {}


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", html.unescape(str(text)).lower())


def scalar(text: Any) -> Any:
    t = str(text or "").strip()
    return t.lower() == "true" if t.lower() in ("true", "false") else (float(t) if "." in t else int(t)) if re.fullmatch(r"-?\d+(?:\.\d+)?", t) else t


def xml_value(node: Any) -> Any:
    kids = list(node)
    if not kids and not node.attrib:
        return scalar(node.text)
    if not node.attrib and len({kid.tag for kid in kids}) == 1 and all(len(kid) >= 2 for kid in kids):
        return [xml_value(kid) for kid in kids]
    out: dict = {k: scalar(v) for k, v in node.attrib.items()}
    tags = [kid.tag for kid in kids]
    for kid in kids:
        if tags.count(kid.tag) > 1:
            out.setdefault(kid.tag, []).append(xml_value(kid))
        else:
            out[kid.tag] = xml_value(kid)
    if (node.text or "").strip():
        out["text"] = node.text.strip()
    return out


def decode(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: decode(v) for k, v in node.items()}
    if isinstance(node, list):
        return [decode(v) for v in node]
    if not isinstance(node, str):
        return node
    text = node.strip()
    if text[:1] in "{[" and text[-1:] in "}]":
        try:
            return decode(json.loads(text))
        except ValueError:
            return node
    if text[:1] == "<" and text[-1:] == ">":
        try:
            return xml_value(ElementTree.fromstring(text))
        except ElementTree.ParseError:
            return node
    return node


def kv_text(text: str) -> dict:
    found: dict = {}
    for part in re.split(r"\s*\|\s*|\n+", text):
        colon, equals = part.find(":"), part.find("=")
        key, sep, value = part.partition("=" if equals > 0 and (colon < 0 or equals < colon) else ":")
        if sep and 0 < len(key.strip()) <= 40 and value.strip():
            inner = [pair.partition("=") for pair in value.split(";")]
            if sum(1 for k, eq, _ in inner if eq and k.strip()) >= 2:
                found.setdefault(key.strip(), {k.strip(): v.strip() for k, eq, v in inner if eq and k.strip()})
            else:
                found.setdefault(key.strip(), value.strip())
    return found


def cut_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    for pattern in (r"\n|\},(?=\s?\{)|\",(?=\s?\")", r"[.!?](?=\s)", r"\s\|\s|,\s", r"\s"):
        ends = [m.end() for m in re.finditer(pattern, text[:limit]) if m.end() > limit // 2]
        if ends:
            break
    end = ends[-1] if ends else limit
    return text[:end].rstrip() + f" …({len(text) - end} more characters left out)"


def clip(text: Any, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    head = text[: limit + 1]
    return (head.rsplit(" ", 1)[0] if " " in head else text[:limit]).strip()


def raw_of(obs: Any) -> str:
    return obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False, default=str)


def text_of(value: Any) -> str:
    if isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value):
        return "; ".join(str(v) for v in value if v not in (None, ""))
    return value if isinstance(value, str) else raw_of(value) if isinstance(value, (dict, list)) else str(value)


def keep_text(card: dict, label: str, value: Any) -> None:
    text = text_of(value).strip() if value is not None else ""
    if text and text not in [t for k, t in card["texts"].items() if k != label]:
        card["texts"][label] = text


def flag_of(text: str) -> bool | None:
    ws = words(text)
    nums = [float(w) for w in ws if re.fullmatch(r"\d+(?:\.\d+)?", w)]
    neg, yes = [w for w in ws if w in ("no", "not", "none", "false", "out", "zero")], {"yes", "true", "in"} & set(ws)
    if len(neg) > 1 or (neg and yes):
        return None
    return False if neg else nums[0] > 0 if nums else True if yes else None


class Store:

    def __init__(self, env: Env, brain: Brain, trace: list) -> None:
        self.env, self.brain, self.trace = env, brain, trace
        self.tools = {fn["name"]: t for t in env.view.get("tools") or [] if isinstance(t, dict) for fn in [t.get("function", t)] if isinstance(fn, dict) and fn.get("name")}
        self.roles: dict = {}
        self.fields: dict = {}
        self.pager_keys: dict = {}
        self.first_result = ""
        self.record_asked = False
        self.cards: dict[str, dict] = {}
        self.records: dict[str, dict] = {}
        self.pages: dict[str, Any] = {}
        self.notices: list[str] = []
        self.tried: set = set()
        self.memo: dict = {}
        self.shows: dict = {}
        self.calls: dict = {}
        self.first_page: dict = {}
        self.thin: set = set()
        self.stale: set = set()
        self.stock_texts: dict[str, bool | None] = {}

    def learn_tools(self, seat: str = "MAP") -> None:
        listing = "\n".join(f"- {name}: {t.get('function', t).get('description', '')} | arguments: {json.dumps(tool_props(t), sort_keys=True)}" for name, t in sorted(self.tools.items()))

        def check(answer: dict) -> str | None:
            for role, spec in answer.items():
                if isinstance(spec, dict) and spec.get("tool") and spec["tool"] not in self.tools:
                    return f"{role}.tool {spec['tool']!r} is not a tool name"
            mapped = [(role, spec) for role, spec in answer.items() if isinstance(spec, dict) and spec.get("tool") in self.tools]
            refused = [f"{role}.{why}" for role, spec in mapped for kind, why in self.strays(json.loads(json.dumps(spec)), tool_props(self.tools[spec["tool"]])) if kind == "value"]
            return "; ".join(refused) if refused else None if isinstance(answer.get("search"), dict) else "no search tool named"

        answer = self.brain.ask(seat, MAP_TOOLS, "", "TOOLS:\n" + listing, {"type": "object"}, check) or {}
        gaps = self.unmapped(answer) if answer else []
        if gaps:
            again = self.brain.ask(seat, MAP_TOOLS, "", MAP_AGAIN.format(listing=listing, map=json.dumps(answer, ensure_ascii=False), gaps="; ".join(gaps)), {"type": "object"}, check) or {}
            filled = [role for role, spec in again.items() if isinstance(spec, dict) and spec.get("tool") in self.tools and not (isinstance(answer.get(role), dict) and answer[role].get("tool") in self.tools)
                      and not self.strays(json.loads(json.dumps(spec)), tool_props(self.tools[spec["tool"]])) and (role not in ROLE_ARGS or any(spec.get(arg) for arg in ROLE_ARGS[role]))]
            answer.update({role: again[role] for role in filled})
            trace_row(self.trace, "map_gaps", gaps=gaps, filled=filled, left=self.unmapped(answer))
        for role, spec in answer.items():
            if isinstance(spec, dict) and spec.get("tool") in self.tools:
                props = tool_props(self.tools[spec["tool"]])
                for what, why in self.strays(spec, props):
                    trace_row(self.trace, "map_stray", role=role, tool=spec["tool"], what=what, why=why)
                if role == "cart_write" and spec.get("mode") != "replace" and not spec.get("qty_arg"):
                    ints = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "integer" and k != spec.get("id_arg")]
                    spec["qty_arg"] = ints[0] if len(ints) == 1 else None
                    trace_row(self.trace, "qty_arg_guess", tool=spec["tool"], qty_arg=spec["qty_arg"], integers=ints)
                if role == "compare":
                    ceiling = (props.get(spec.get("ids_arg") or "") or {}).get("maxItems")
                    counts = [c for c in (spec.get("most"), ceiling) if isinstance(c, int) and not isinstance(c, bool) and c >= 1]
                    spec["most"] = min(counts) if counts else 4
                self.roles[role] = spec
        trace_row(self.trace, "roles", roles=self.roles)

    def unmapped(self, answer: dict) -> list[str]:
        used, fixed = set(), {}
        for spec in answer.values():
            if isinstance(spec, dict) and spec.get("tool") in self.tools:
                used.add(spec["tool"])
                for field in ("fixed", "args"):
                    for arg, value in (spec.get(field) if isinstance(spec.get(field), dict) else {}).items():
                        fixed.setdefault((spec["tool"], arg), set()).add(str(value).lower())
        out = [f"tool {name} (arguments {', '.join(sorted(tool_props(t)))})" for name, t in sorted(self.tools.items()) if tool_props(t) and name not in used]
        for (tool, arg), values in sorted(fixed.items()):
            enum = (tool_props(self.tools[tool]).get(arg) or {}).get("enum")
            out += [f"value {e!r} of {tool}.{arg}" for e in enum or [] if str(e).lower() not in values]
        return out

    def strays(self, spec: dict, props: dict) -> list[tuple[str, str]]:
        out = []
        allowed = lambda arg, value: not isinstance((props.get(arg) or {}).get("enum"), list) or any(str(e).lower() == str(value).lower() for e in props[arg]["enum"])
        for key in ("query_arg", "page_arg", "price_arg", "category_arg", "id_arg", "ids_arg", "qty_arg", "op_arg", "list_arg", "text_arg"):
            if spec.get(key) and spec[key] not in props:
                out.append(("name", f"{key} {spec[key]!r} is no argument of {spec['tool']}"))
                spec[key] = None
        for field in ("fixed", "args"):
            values = spec.get(field) if isinstance(spec.get(field), dict) else {}
            for arg, value in list(values.items()):
                if arg not in props or not allowed(arg, value):
                    out.append(("value", f"{field}.{arg} {value!r} is " + (f"no argument of {spec['tool']}" if arg not in props else f"not one of {props[arg]['enum']}")))
                    values.pop(arg)
        for key in ("add_value", "remove_value"):
            if spec.get(key) is not None and spec.get("op_arg") and not allowed(spec["op_arg"], spec[key]):
                out.append(("value", f"{key} {spec[key]!r} is not one of {props[spec['op_arg']]['enum']}"))
                spec[key] = None
        items = ((props.get(spec.get("list_arg") or "") or {}).get("items") or {}).get("properties")
        for key in ("item_id_key", "item_qty_key"):
            if isinstance(items, dict) and spec.get(key) and spec[key] not in items:
                out.append(("value", f"{key} {spec[key]!r} is no key of the items of {spec['list_arg']}"))
                spec[key] = None
        return out

    def fit_args(self, tool: str, args: dict) -> dict:
        props, fitted = tool_props(self.tools.get(tool, {})), {}
        for key, value in args.items():
            spec = props.get(key)
            if spec is None or value is None:
                continue
            kind = spec.get("type")
            kind = next((k for k in kind if k != "null"), None) if isinstance(kind, list) else kind
            if spec.get("enum") and isinstance(value, str):
                match = [e for e in spec["enum"] if str(e).lower() == value.lower()]
                value = match[0] if match else value
            if kind == "string" and isinstance(value, (bool, int, float)):
                value = str(value).lower() if isinstance(value, bool) else str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
            if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            if kind == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
            if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()):
                value = int(float(value)) if kind == "integer" else float(value)
            if kind == "string" and isinstance(value, str) and spec.get("maxLength") and len(value) > int(spec["maxLength"]):
                value = clip(value, int(spec["maxLength"]))
            fitted[key] = value
        return fitted

    def act(self, tool: str, args: dict) -> dict:
        fitted = self.fit_args(tool, args)
        read = any(isinstance(spec, dict) and spec.get("tool") == tool for role, spec in self.roles.items() if role in READ_ROLES)
        key = self.call_key(tool, args)
        if read and key in self.memo:
            trace_row(self.trace, "memo", tool=tool, args=fitted)
            return self.memo[key]
        before = self.env.step
        result = self.env.call(tool, fitted)
        if self.env.step > before:
            self.calls.setdefault(key, self.env.step)
        heard = len(self.notices)
        result["observation"], said = self.split(decode(result.get("observation")), fitted)
        for text in said + [t for t in result.get("refusal") or [] if isinstance(t, str) and t.strip()]:
            if text in self.notices:
                trace_row(self.trace, "notice_again", tool=tool, step=self.env.step, text=text)
            else:
                self.notices.append(text)
        if result.get("refusal"):
            trace_row(self.trace, "refusal", tool=tool, refusal=result["refusal"])
        if not read or len(self.notices) > heard:
            self.memo.clear()
        if read and not result.get("error") and result.get("observation") is not None:
            self.memo[key] = result
            self.learn_reads(tool, fitted, result["observation"])
        self.watch(tool, fitted, result)
        return result

    def call_key(self, tool: str, args: dict) -> tuple[str, str]:
        return tool, json.dumps(self.fit_args(tool, args), sort_keys=True, default=str)

    def seen_call(self, tool: str, args: dict) -> bool:
        return self.call_key(tool, args) in self.calls

    def split(self, obs: Any, sent: dict) -> tuple[Any, list[str]]:
        idkey = ((self.fields.get("row") or {}) if isinstance(self.fields.get("row"), dict) else {}).get("id")
        echoes = {t for v in sent.values() for t in (v if isinstance(v, list) else [v]) if isinstance(t, str)}
        said: list[str] = []

        def read(node: Any) -> Any:
            if not isinstance(node, dict) or (idkey and idkey in node):
                return node
            notes = [k for k, v in node.items() if isinstance(v, list) and v and all(isinstance(t, str) for t in v) and any(len(t.split()) >= 2 for t in v) and not echoes & set(v) and not self.looks_rows(v)]
            said.extend(t for k in notes for t in node[k] if t.strip())
            rest = {k: read(v) for k, v in node.items() if k not in notes}
            if notes and len(rest) == 1 and isinstance(next(iter(rest.values())), (dict, list, str)):
                return next(iter(rest.values()))
            if not notes and len(node) == 2 and any(isinstance(v, str) and len(v.split()) >= 2 for v in node.values()) and any(isinstance(v, (dict, list)) for v in node.values()):
                trace_row(self.trace, "notice_shape", keys=sorted(map(str, node)), why="a sentence alone beside one result: not read as a notice")
            return {k: rest.get(k, v) for k, v in node.items()}

        return read(obs), said

    def learn_reads(self, tool: str, args: dict, obs: Any) -> None:
        compare = self.roles.get("compare") or {}
        is_compare = tool == compare.get("tool")
        if not is_compare and not any(tool == (self.roles.get(r) or {}).get("tool") for r in ("facts", "page")):
            return
        sent = {str(i) for i in (args.get(compare.get("ids_arg") or "") or [])} if is_compare else {str(v) for v in args.values()}
        rows = self.rows_list(obs)[0] if is_compare else [obs]
        head = {v for v in ((self.fields.get("row") or {}) if isinstance(self.fields.get("row"), dict) else {}).values() if isinstance(v, str)}
        mine = []
        for row in rows:
            data = row if isinstance(row, dict) else kv_text(row) if isinstance(row, str) else {}
            card = self.card_of({k: v for k, v in data.items() if not isinstance(v, dict)}) if data else None
            if card and (card["id"] in sent or not is_compare):
                mine.append((card, data))
        was = self.shows.get(tool, (False, False))
        self.shows[tool] = (was[0] or any(c["price"] is not None for c, _ in mine), was[1] or any(c["stock"] is not None for c, _ in mine))
        if is_compare and mine and not compare.get("facts"):
            table = lambda d: any(isinstance(v, dict) and len(v) >= 2 for v in d.values()) or sum(1 for k, v in d.items() if k not in head and v not in (None, "", [], {})) >= 2
            thin = [c["id"] for c, d in mine if not table(d)]
            bare = [i for i in thin if (self.cards.get(i) or {}).get("facts")]
            verdict = True if len(thin) < len(mine) else False if bare else None
            self.thin.update(thin)
            if verdict is not None:
                compare["facts"], compare["facts_seen"] = verdict, {"step": self.env.step, "ids": sorted(c["id"] for c, _ in mine), "without a table": bare, "keys": sorted({str(k) for _, d in mine for k in d})[:12]}
            trace_row(self.trace, "compare_facts", facts=verdict, tool=tool, ids=sorted(c["id"] for c, _ in mine))

    def watch(self, tool: str, args: dict, result: dict) -> None:
        roles = [r for r, spec in self.roles.items() if isinstance(spec, dict) and spec.get("tool") == tool]
        error = str(result.get("error") or "")
        if any(r not in self.tried for r in roles) and re.search(r"invalid|unknown|schema|argument", error.lower()):
            trace_row(self.trace, "handoff", reason="schema error on the first call of a role", roles=roles, tool=tool, args=args, error=error)
        self.tried.update(roles)

    def looks_rows(self, items: list) -> bool:
        return sum(1 for v in items if len(kv_text(v)) >= 2) >= max(1, len(items) // 2)

    def rows_list(self, obs: Any) -> tuple[list, dict]:
        if isinstance(obs, list):
            return obs, {}
        lists: dict[str, list] = {}
        scalars: dict[str, Any] = {}

        def walk(node: dict, path: str) -> None:
            for key, value in node.items():
                at = f"{path}.{key}" if path else str(key)
                if isinstance(value, dict):
                    walk(value, at)
                elif isinstance(value, list):
                    if value and (all(isinstance(v, dict) for v in value) or (all(isinstance(v, str) for v in value) and self.looks_rows(value))):
                        lists[at] = value
                else:
                    scalars[at] = value

        if isinstance(obs, dict):
            walk(obs, "")
        named = (self.fields.get("result") or {}).get("rows") if isinstance(self.fields.get("result"), dict) else None
        if named not in lists and len(lists) > 1:
            trace_row(self.trace, "rows_ambiguous", lists={k: len(v) for k, v in lists.items()})
        return (lists[named] if named in lists else max(lists.values(), key=len, default=[])), scalars

    def pager(self, top: dict, shown: int, query: str, spec: dict) -> tuple[Any, Any, bool]:
        res = self.fields.get("result") if isinstance(self.fields.get("result"), dict) else {}
        confirmed = self.pager_keys.setdefault(spec.get("tool"), {})
        is_count = lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= shown
        is_next = lambda v: v is None or (isinstance(v, (str, int)) and not isinstance(v, bool))
        found = {}
        for role, fits in (("count", is_count), ("next", is_next)):
            for name in (confirmed.get(role), res.get(role)):
                if isinstance(name, str) and name in top and fits(top[name]) and name not in found.values():
                    found[role] = name
                    break
        if "count" not in found and res.get("count"):
            ints = [k for k, v in top.items() if is_count(v) and k != found.get("next")]
            if len(ints) == 1:
                found["count"] = ints[0]
        if "next" not in found and spec.get("page_kind") == "token" and spec.get("page_arg") and not confirmed.get("next"):
            others = [k for k, v in top.items() if k != found.get("count") and is_next(v) and (v is None or (str(v).strip() and str(v).strip() != query.strip()))]
            if len(others) > 1:
                others = [k for k in others if not is_count(top[k])] or others
            if len(others) == 1:
                found["next"] = others[0]
        missed = {role: res.get(role) for role in ("count", "next") if res.get(role) and res.get(role) != found.get(role) and not confirmed.get(role)}
        if missed:
            trace_row(self.trace, "pager", named=missed, read=found, scalars=sorted(top))
        marked = "next" in found or bool(confirmed.get("next"))
        confirmed.update(found)
        return (top[found["count"]] if "count" in found else None), (top[found["next"]] if "next" in found else None), marked

    def learn_fields(self, obs: Any) -> None:
        rows, _ = self.rows_list(obs)
        if not rows or self.fields:
            return

        def shrink(node: Any) -> Any:
            if node is rows:
                return [shrink(rows[0])]
            if isinstance(node, dict):
                return {k: shrink(v) for k, v in node.items()}
            if isinstance(node, list):
                return [shrink(v) for v in node[:3]]
            return cut_text(node, 1500 if node is rows[0] else 300) if isinstance(node, str) else node

        self.first_result = f"SEARCH RESULT (its list of {len(rows)} rows cut to the first row, long texts cut):\n" + json.dumps(shrink(obs), ensure_ascii=False, default=str)
        pairs = kv_text(rows[0]) if isinstance(rows[0], str) else {}
        if len(pairs) >= 2:
            self.first_result += "\n\nPAIRS (the first row is one text; its key: value pairs as the code reads them):\n" + json.dumps({k: cut_text(v, 200) if isinstance(v, str) else v for k, v in pairs.items()}, ensure_ascii=False, default=str)
        answer = self.brain.ask("MAP", MAP_FIELDS, "", self.first_result, {"type": "object"}) or {}
        self.fields = answer if isinstance(answer, dict) else {}
        trace_row(self.trace, "fields", fields=self.fields)

    def learn_record(self, data: dict) -> None:
        self.fields["record"] = {}
        shown = json.dumps({k: cut_text(v, 600) if isinstance(v, str) else v for k, v in data.items()}, ensure_ascii=False, default=str)
        answer = self.brain.ask("MAP", MAP_FIELDS, "", f"{self.first_result}\n\nPRODUCT RECORD:\n{shown}", {"type": "object"}) or {}
        rec = answer.get("record") if isinstance(answer.get("record"), dict) else {}
        self.fields["record"] = {k: v for k, v in rec.items() if isinstance(v, str) and v in data}
        trace_row(self.trace, "fields", record=self.fields["record"], asked=rec)

    def head_of(self, data: dict) -> tuple[dict, dict]:
        f = (self.fields.get("row") or {}) if isinstance(self.fields.get("row"), dict) else {}
        if isinstance(data.get(f.get("id")), int) and not isinstance(data[f["id"]], bool):
            data = {**data, f["id"]: str(data[f["id"]])}
        kinds = {"title": (str,), "price": (int, float, str), "stock": (bool, str), "category": (str,), "seller": (str,), "brand": (str,), "id": (str,)}
        shaped = {"price": lambda v: (isinstance(v, (int, float)) and not isinstance(v, bool)) or (isinstance(v, str) and bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", v))),
                  "stock": lambda v: isinstance(v, bool) or (isinstance(v, str) and v.strip().lower() in ("true", "false")),
                  "id": lambda v: isinstance(v, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,40}", v) and re.search(r"\d", v)) and not re.fullmatch(r"[\d.,]+", v)}
        values: dict = {}
        keys: dict = {}
        for role, kind in kinds.items():
            fits = lambda v: isinstance(v, kind) and (role == "stock" or not isinstance(v, bool))
            named = [f[role]] if f.get(role) in data else [k for k, v in data.items() if fits(v) and role in str(k).lower()]
            if role == "id" and len(named) == 1 and (not isinstance(data[named[0]], str) or " " in data[named[0]].strip()):
                named = []
            found = named if len(named) == 1 else [k for k, v in data.items() if role in shaped and k not in keys.values() and shaped[role](v)]
            value = data[found[0]] if len(found) == 1 else None
            if isinstance(value, str) and role == "price":
                nums = re.findall(r"\d+(?:\.\d+)?", value.replace(",", ""))
                value = float(nums[0]) if len(nums) == 1 else None
            elif isinstance(value, str) and role == "stock":
                value = self.stock_of(value)
            values[role] = value
            if value not in (None, ""):
                keys[role] = found[0]
        return values, keys

    def stock_of(self, text: str) -> bool | None:
        key = " ".join(text.split()).lower()
        if key not in self.stock_texts:
            bare = lambda k: not k or re.fullmatch(r"true|false|yes|no|\d+", k)
            check = lambda a: None if isinstance(a.get("stock"), dict) and text in a["stock"] and (a["stock"][text] is None or isinstance(a["stock"][text], bool)) else "stock must map the given text, exactly as given, to true, false or null"
            ask = not bare(key) and sum(1 for k in self.stock_texts if not bare(k)) < 4
            answer = self.brain.ask("MAP", MAP_FIELDS, "", "STOCK VALUES: " + json.dumps([text], ensure_ascii=False), {"type": "object"}, check) if ask else None
            self.stock_texts[key] = answer["stock"][text] if answer else flag_of(text)
            if not bare(key):
                trace_row(self.trace, "stock_text", text=text, reading=self.stock_texts[key], by="MAP" if answer else "flag_of", asked=ask)
        return self.stock_texts[key]

    def card_of(self, row: Any) -> dict | None:
        data = row if isinstance(row, dict) else kv_text(row) if isinstance(row, str) else {}
        values, keys = self.head_of(data) if data else ({}, {})
        if not isinstance(values.get("id"), str):
            return None
        text = " | ".join(f"{k}: {v}" for k, v in data.items() if isinstance(v, (str, int, float)) and v != values["id"]) if isinstance(row, dict) else str(row)
        card = {"id": values["id"].strip(), **{r: str(values[r] or "") for r in ("title", "category", "seller", "brand")}, "price": values["price"], "stock": values["stock"], "text": text, "facts": {}, "texts": {}, "small": "",
                "raw": row, "heads": keys}
        if isinstance(row, dict):
            for key, value in row.items():
                if key not in keys.values():
                    keep_text(card, f"row.{key}", value)
        else:
            card["texts"]["row"] = str(row)
        return card

    def join_row(self, card: dict, row_card: dict) -> None:
        card.update(text=row_card["text"], raw=row_card["raw"])
        for key in HEAD_KEYS:
            if card.get(key) in (None, ""):
                card[key] = row_card[key]
        for label, text in row_card["texts"].items():
            keep_text(card, label, text)

    def query_args(self, role: str, query: str, extra: dict | None = None) -> tuple[str | None, dict]:
        spec = self.roles.get(role) or {}
        args = dict(spec.get("fixed") or {})
        if spec.get("query_arg") and query:
            args[spec["query_arg"]] = clip(query, self.query_limit(role))
        args.update(extra or {})
        return spec.get("tool"), args

    def search(self, query: str, more: Any = None, role: str = "search", extra: dict | None = None) -> dict:
        spec = self.roles.get(role) or {}
        tool, args = self.query_args(role, query, extra)
        if not tool:
            return {"cards": [], "count": None, "next": None, "error": "no tool", "end": "no tool"}
        arg, kind, low, limit = self.page_bounds(role)
        number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        if more is None and low is not None and self.first_page.get(tool, (None,))[0] is False:
            more = low
        if more is not None and limit is not None and number(more) and more > limit:
            trace_row(self.trace, "page_end", reason="page value above the schema maximum", query=query, page=more, maximum=limit)
            return {"cards": [], "count": None, "next": None, "error": None, "end": "the schema allows no further page"}
        if more is not None and arg:
            args[arg] = more
        page = self.page_of(tool, args, query, spec, more, low, limit)
        bare = {k: v for k, v in args.items() if k != arg}
        read_low = low is None or self.seen_call(tool, {**bare, arg: low}) or (more is not None and self.first_page.get(tool, (None,))[0] is not False and self.seen_call(tool, bare))
        if page["cards"] and page["next"] is None and isinstance(page["count"], int) and page["count"] > len(page["cards"]) and more != low and not read_low and (more is not None or tool not in self.first_page):
            probe = self.page_of(tool, {**bare, arg: low}, query, spec, low, low, limit)
            other = [c for c in probe["cards"] if c["id"] not in {x["id"] for x in page["cards"]}]
            evidence = f"{tool} {query!r} at page {more if more is not None else 'default'}: {len(page['cards'])} of {page['count']} rows and no next value; at {low}: {len(other)} other rows"
            if more is None:
                self.first_page[tool] = (not other if probe["cards"] else None, evidence)
            trace_row(self.trace, "first_page", tool=tool, query=query, page=more, lowest=low, other=len(other), count=page["count"], rows=len(page["cards"]))
            if other:
                page = dict(probe, cards=page["cards"] + other)
        return page

    def page_bounds(self, role: str) -> tuple[str | None, str | None, Any, Any]:
        spec = self.roles.get(role) or {}
        arg = spec.get("page_arg")
        prop = (tool_props(self.tools.get(spec.get("tool"), {})).get(arg or "") or {}) if arg else {}
        number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        kind = spec.get("page_kind") if arg else None
        low = (prop["minimum"] if number(prop.get("minimum")) else 1 if kind == "page" else 0) if kind in ("page", "offset") else None
        return arg, kind, low, prop.get("maximum") if number(prop.get("maximum")) else None

    def rows_reach(self, role: str, per_page: int) -> int | None:
        _, kind, low, limit = self.page_bounds(role)
        if limit is None or kind not in ("page", "offset"):
            return None
        return int(limit) + per_page if kind == "offset" else int(limit - low + 1) * per_page

    def page_of(self, tool: str, args: dict, query: str, spec: dict, more: Any, low: Any, limit: Any) -> dict:
        kind = spec.get("page_kind") if spec.get("page_arg") else None
        result = self.act(tool, args)
        obs = result.get("observation")
        self.learn_fields(obs)
        rows, top = self.rows_list(obs)
        cards = [c for c in (self.card_of(r) for r in rows) if c]
        read_next = None
        if not cards and not result.get("error") and obs is not None and (rows or len(raw_of(obs)) > 200):
            trace_row(self.trace, "handoff", reason="search result not empty but no card read", query=query, chars=len(raw_of(obs)))
            read, read_next = self.read_rows(obs)
            cards = [c for c in (self.card_of(r) for r in read) if c]
        for card in cards:
            have = self.cards.setdefault(card["id"], card)
            if have is not card and have.get("raw") is None:
                self.join_row(have, card)
        count, token, marked = self.pager(top, len(cards), query, spec)
        if token is None and read_next is not None:
            token, marked = (int(read_next) if kind in ("page", "offset") and read_next.isdigit() else read_next), True
        if more is None and token is None and cards and kind == "token" and isinstance(count, int) and count > len(cards):
            trace_row(self.trace, "handoff", reason="count above the rows shown but no next value read", query=query, count=count, rows=len(cards), scalars=sorted(top))
        at = more if more is not None else low
        end = "the store's next value is empty" if marked and token is None else None
        if not marked and token is None and cards and kind in ("page", "offset") and isinstance(at, (int, float)):
            token = at + 1 if kind == "page" else at + len(cards)
        if kind in ("page", "offset") and isinstance(token, (int, float)) and isinstance(at, (int, float)) and token <= at:
            token, end = None, "the next value does not move past the page read"
        if kind == "offset" and isinstance(token, (int, float)) and isinstance(count, (int, float)) and token >= count:
            token, end = None, "the count says this was the last page"
        if limit is not None and isinstance(token, (int, float)) and not isinstance(token, bool) and token > limit:
            token, end = None, "the schema allows no further page"
        return {"cards": cards, "count": count if isinstance(count, (int, float)) else None, "next": token, "error": result.get("error"), "end": end or ("no rows" if not cards else None if token is not None else "no next value")}

    def read_rows(self, obs: Any) -> tuple[list[dict], str | None]:
        raw = raw_of(obs)

        def check(answer: dict) -> str | None:
            if not isinstance(answer.get("rows"), list):
                return "rows is not a list"
            lost = [r.get("id") for r in answer["rows"] if isinstance(r, dict) and str(r.get("id") or "").strip() not in raw]
            return f"these ids are not written in RESULT: {lost[:5]}" if lost else None

        numbers, rows, token, at, answered = set(numbers_in(raw)), [], None, 0, 0
        for _ in range(3):
            part = cut_text(raw[at:], 12000)
            answer = self.brain.ask("ROWS", ROWS, "", "RESULT:\n" + part, ROWS_SCHEMA, check) or {}
            answered += len(answer.get("rows") or [])
            for r in answer.get("rows") or []:
                ident = str(r.get("id") or "").strip() if isinstance(r, dict) else ""
                if ident and ident in raw and all(x["id"] != ident for x in rows):
                    price = r.get("price") if isinstance(r.get("price"), (int, float)) and float(r["price"]) in numbers else None
                    rows.append({"id": ident, "title": str(r.get("title") or ""), "price": price, "in_stock": r["in_stock"] if isinstance(r.get("in_stock"), bool) else None})
            told = answer.get("next")
            token = told if isinstance(told, str) and told.strip() and told in raw else token
            left = re.search(r" …\((\d+) more characters left out\)$", part)
            at = len(raw) - int(left[1]) if left else len(raw)
            if at >= len(raw):
                break
        starts = sorted((raw.find(r["id"]), k) for k, r in enumerate(rows))
        for (begin, k), until in zip(starts, [s for s, _ in starts[1:]] + [len(raw)]):
            rows[k]["text"] = cut_text(raw[begin + len(rows[k]["id"]) : until], 800)
        trace_row(self.trace, "rows", answered=answered, kept=[r["id"] for r in rows], next=token, unread_chars=len(raw) - at)
        return rows, token

    def card(self, ident: str) -> dict:
        return self.cards.setdefault(ident, {"id": ident, "title": "", "price": None, "stock": None, "category": "", "seller": "", "brand": "", "text": "", "facts": {}, "texts": {}, "small": "", "raw": None})

    def query_limit(self, role: str = "search") -> int:
        spec = self.roles.get(role) or {}
        return int((tool_props(self.tools.get(spec.get("tool"), {})).get(spec.get("query_arg") or "") or {}).get("maxLength") or 60)

    def read_card(self, ident: str, want_page: bool = False) -> dict | None:
        card = self.card(ident)
        for role in ("facts", "page") if want_page else ("facts",):
            spec = self.roles.get(role) or self.roles.get("page" if role == "facts" else "facts")
            if not spec or not spec.get("tool") or (role == "page" and spec.get("tool") == (self.roles.get("facts") or {}).get("tool") and ident in self.records):
                continue
            result = self.act(spec["tool"], {spec.get("id_arg") or "id": ident})
            obs = result.get("observation")
            if result.get("error") or obs is None:
                continue
            self.absorb(card, obs, role)
            if role == "facts" and (self.roles.get("page") or {}).get("tool") == spec["tool"]:
                self.pages[card["id"]] = obs
        return card

    def opened(self, ident: str) -> bool:
        return ident in self.records or ident in self.pages

    def reread(self, ident: str) -> dict:
        card = self.card(ident)
        tools = [(role, spec) for role in ("page", "facts", "compare") for spec in [self.roles.get(role) or {}] if spec.get("tool") and (role != "compare" or spec.get("ids_arg"))]
        shown = lambda spec: self.shows.get(spec["tool"], (False, False))
        role, spec = max(tools, key=lambda rs: (all(shown(rs[1])), shown(rs[1])[1], shown(rs[1])[0], rs[1]["tool"] not in self.shows), default=(None, {}))
        if spec and (any(shown(spec)) or spec["tool"] not in self.shows):
            args = {spec["ids_arg"]: [ident]} if role == "compare" else {spec.get("id_arg") or "id": ident}
            self.memo.pop(self.call_key(spec["tool"], args), None)
            result = self.act(spec["tool"], args)
            obs = result.get("observation")
            fresh = False
            if not result.get("error") and obs is not None:
                for row in self.rows_list(obs)[0] if role == "compare" else [obs]:
                    if role != "compare" or (self.card_of(row) or {}).get("id") == ident:
                        fresh |= self.absorb(card, row, "page" if role == "page" and spec["tool"] != (self.roles.get("facts") or {}).get("tool") else "facts")
            (self.stale.discard if fresh else self.stale.add)(ident)
            trace_row(self.trace, "reread", id=ident, tool=spec["tool"], price=card.get("price"), stock=card.get("stock"), shows=shown(spec), fresh=fresh)
            if not fresh:
                trace_row(self.trace, "handoff", reason="re-read gave no price or stock of this listing: the card keeps an earlier read's", id=ident, tool=spec["tool"], obs=raw_of(obs)[:300])
            return card
        title = clip(card.get("title") or "", self.query_limit())
        row = next((c for c in (self.search(title)["cards"] if title else []) if c["id"] == ident), None)
        if row:
            card.update({k: row[k] for k in ("price", "stock") if row.get(k) is not None})
        trace_row(self.trace, "reread", id=ident, tool="search row" if row else None, price=card.get("price"), stock=card.get("stock"), why="no read tool shows price or stock" + ("" if row else "; the cart view decides"))
        return card

    def absorb(self, card: dict, obs: Any, role: str) -> bool:
        data = obs if isinstance(obs, dict) else kv_text(obs) if isinstance(obs, str) else {}
        if isinstance(obs, dict) and isinstance(obs.get("observation"), (dict, str)):
            data = obs["observation"] if isinstance(obs["observation"], dict) else kv_text(obs["observation"])
        tables = [k for k, v in data.items() if isinstance(v, dict) and len(v) >= 2]
        if isinstance(obs, dict) and data and len(tables) != 1 and self.fields and not self.record_asked:
            self.record_asked = True
            self.learn_record(data)
        rec = self.fields.get("record") if isinstance(self.fields.get("record"), dict) else {}
        named = data.get(rec["facts"]) if isinstance(rec.get("facts"), str) else None
        pairs = kv_text(named) if isinstance(named, str) else {}
        facts = named if isinstance(named, dict) else pairs if len(pairs) >= 2 else data[tables[0]] if tables else None
        if isinstance(facts, dict):
            card["facts"].update({str(k): str(v) for k, v in facts.items() if v is not None})
            compare = self.roles.get("compare") or {}
            if card["facts"] and card["id"] in self.thin and compare.get("facts") is None:
                compare["facts"], compare["facts_seen"] = False, {"step": self.env.step, "id": card["id"], "why": "its record shows facts its compare row did not"}
                trace_row(self.trace, "compare_facts", facts=False, id=card["id"], why="its record shows facts its compare row did not")
        elif isinstance(obs, str):
            card["facts"].update(data)
        small = data.get(rec["small_print"]) if isinstance(rec.get("small_print"), str) else None
        if isinstance(small, str) and small.strip():
            card["small"] = small
        fresh = self.card_of({k: v for k, v in data.items() if not isinstance(v, dict)}) if data else None
        fresh = fresh if fresh and fresh["id"] == card["id"] else None
        if fresh:
            for key in HEAD_KEYS:
                if fresh.get(key) not in (None, ""):
                    card[key] = fresh[key]
            if card.get("raw") is None and not card.get("text"):
                card["text"] = fresh["text"]
        heads = set(fresh["heads"].values()) if fresh else set()
        for key, value in data.items():
            if key not in heads and value is not facts and not (facts is pairs and value is named):
                keep_text(card, f"{'page' if role == 'page' else 'record'}.{key}", value)
        raw = raw_of(obs)
        if not card["facts"] and len(raw) > 200:
            if not card.get("raw_text"):
                trace_row(self.trace, "handoff", reason="record without readable facts", id=card["id"], role=role, chars=len(raw))
            if raw[:200] not in card.get("raw_text", ""):
                card["raw_text"] = cut_text((card.get("raw_text", "") + "\n" + raw).strip(), 6000)
        if role == "page":
            self.pages[card["id"]] = obs
        else:
            self.records[card["id"]] = obs
        return bool(fresh) and (fresh["price"] is not None or fresh["stock"] is not None)

def norm_text(text: Any) -> str:
    return " " + " ".join(words(text)) + " "


def stands(quote: Any, text: Any) -> bool:
    pieces = [p for p in map(norm_text, re.split(r"[;|\n]|(?<=[.!?])\s", "" if quote is None else str(quote))) if p.strip()]
    whole = norm_text(text)
    glued = whole.replace(" ", "")
    return bool(pieces) and all(p.replace(" ", "") in glued if len(p.split()) >= 2 and len(p.replace(" ", "")) >= 6 else p in whole for p in pieces)


def numbers_in(text: Any) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text))]


def cap_written(value: Any, text: Any) -> bool:
    return float(value) in numbers_in(text)


def want_text(want: dict) -> str:
    return f"{want['key'] or '(no attribute named)'} {want['op']} {' | '.join(map(str, want.get('values') or []))}"


def line_of(lines: list, named: Any) -> int | None:
    if (isinstance(named, int) or (isinstance(named, str) and named.strip().isdigit())) and 1 <= int(named) <= len(lines):
        return int(named)
    return next((k for k, l in enumerate(lines, 1) if isinstance(l, dict) and isinstance(named, str) and spelled(l.get("item")) == spelled(named)), None)


class Ledger:

    def __init__(self, request: str) -> None:
        self.request = request
        self.lines: list[dict] = []
        self.total_cap: float | None = None
        self.total_objective = "any"
        self.in_stock = True
        self.messages_left: int | None = None
        self.relations: list[dict] = []
        self.fits: list[dict] = []
        self.chain_pages: list[str] = []
        self.unhandled: list[dict] = []
        self.procedures: list[str] = []
        self.decline_rule: str | None = None
        self.nearest: str | None = None
        self.story: list[dict] = []
        self.refusals = 0

    def line(self, n: int) -> dict:
        if not (isinstance(n, int) and 1 <= n <= len(self.lines)):
            raise IndexError(f"Ledger.line({n!r}) outside lines 1 to {len(self.lines)}")
        return self.lines[n - 1]

    def wants(self, n: int) -> list[dict]:
        return [w for w in self.line(n)["wants"] if not w.get("dropped")]

    def listing(self, items: bool = False) -> str:
        return "\n".join(f"line {n}:" + (f" {l['item']}" if items else "") + "".join(f"\n  {k}. {want_text(w)}" for k, w in enumerate(self.wants(n), 1)) for n, l in enumerate(self.lines, 1))

    def state(self, env: Env) -> str:
        rows = [f"steps left {env.left()}, messages left {self.messages_left}, total cap {self.total_cap}"]
        for n, line in enumerate(self.lines, 1):
            wants = "; ".join(f"{w['key']} {w['op']} {'/'.join(map(str, w['values']))}" for w in self.wants(n)) or "-"
            rows.append(f"line {n}: {line['qty']} x {line['item']} ({line['objective']}) wants: {wants} | pick {line.get('pick')} | standby {line.get('standby')}")
        return "\n".join(rows)

def brief_problem(brief: dict, request: str) -> str | None:
    lines = brief.get("lines")
    if not isinstance(lines, list) or not lines:
        return "no lines"
    for n, line in enumerate(lines, 1):
        if not isinstance(line, dict) or not isinstance(line.get("qty"), int) or line["qty"] < 1:
            return f"line {n} has no whole quantity"
        for want in (w for w in line.get("wants") or [] if isinstance(w, dict)):
            if norm_text(want.get("quote") or "").strip() and not stands(want["quote"], request):
                return f"line {n} quote {want.get('quote')!r} is not the request's words"
    nearest = brief.get("nearest")
    if nearest not in (None, "") and not (isinstance(nearest, str) and stands(nearest, request)):
        return f"nearest {nearest!r} is not the request's words (null when the request does not ask for the closest listing)"
    for cap in [brief.get("total_cap")] + [l.get("cap_each") for l in lines]:
        if isinstance(cap, (int, float)) and not cap_written(cap, request):
            return f"cap {cap} is not a number the request writes"
    return None


def procedure_texts(brief: dict, request: str) -> list[str]:
    out: list[str] = []
    for p in brief.get("procedures") or []:
        text = p if isinstance(p, str) else max((v for v in p.values() if isinstance(v, str)), key=lambda v: (stands(v, request), len(v)), default="") if isinstance(p, dict) else ""
        if norm_text(text).strip() and norm_text(text) not in map(norm_text, out):
            out.append(text)
    return out


def quote_span(quote: Any, text: str) -> tuple[Any, list[tuple[int, int]]]:
    tokens = words(quote)
    found = re.search(r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(map(re.escape, tokens)) + r"(?![a-z0-9])", text, re.I) if tokens else None
    spans, depth, start = [], 0, 0
    for k, ch in enumerate(text):
        depth = max(0, depth + (ch in "([{") - (ch in ")]}"))
        if ch in ".!?" and not depth and (k + 1 == len(text) or text[k + 1].isspace()):
            spans.append((start, k + 1))
            start = k + 1
    return found, [(a, b) for a, b in spans + [(start, len(text))] if text[a:b].strip()]


def pointer_context(quote: str, text: str, key: str = "") -> str:
    text = str(text)
    found, spans = quote_span(quote, text)
    if not found:
        return str(quote)
    mine = [j for j, (a, b) in enumerate(spans) if a < found.end() and b > found.start()]
    own = lambda t: {w for w in words(t) if len(w) > 2 and w not in FUNCTION_WORDS and not w.replace(".", "").isdigit()}
    pointer, last = own(quote) - set(words(key)), mine[-1]
    while last + 1 < len(spans) and last < mine[-1] + 2:
        others = set().union(*(own(text[a:b]) for j, (a, b) in enumerate(spans) if not mine[0] <= j <= last + 1))
        if not (pointer & own(text[slice(*spans[last + 1])])) - others:
            break
        last += 1
    a, b = spans[mine[0]][0], spans[last][1]
    return (text[a : found.start()] + "«" + found.group(0) + "»" + text[found.end() : b]).strip()


def content(text: Any, loose: bool = True) -> set:
    return {key_form([w]) for w in words(split_joins(unpossessed(text))) if w not in FUNCTION_WORDS and (loose or len(w) > 1 and not w.replace(".", "").isdigit())}


def meaning(brief: dict) -> set:
    lines = [l for l in brief.get("lines") or [] if isinstance(l, dict)]
    number = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    named = lambda v: line_of(lines, v) or spelled(v)

    def form(x: Any) -> Any:
        if isinstance(x, dict):
            return tuple(sorted((k, named(v) if k == "line" else form(v)) for k, v in x.items() if k != "quote"))
        if isinstance(x, list):
            return tuple(sorted({repr(form(v)) for v in x}))
        return number(x) if number(x) is not None else spelled(x) if isinstance(x, str) else x

    caps = [number(brief.get("total_cap"))] + ([number(lines[0].get("cap_each")) * lines[0]["qty"]] if len(lines) == 1 and number(lines[0].get("cap_each")) is not None else [])
    out = {("order", min([c for c in caps if c is not None], default=None), brief.get("total_objective") or "any", brief.get("in_stock") is not False, brief.get("messages") if isinstance(brief.get("messages"), int) else None)}
    for k, line in enumerate(lines, 1):
        objective, item = line.get("objective") if line.get("objective") in ("cheapest", "best") else "any", content(line.get("item"))
        shelf = None if not line.get("category") or content(line["category"]) <= item else form(line["category"])
        out.add(("line", k, shelf, line.get("qty"), objective, (form(line.get("best_key")), line.get("best_dir")) if objective == "best" else None, number(line.get("cap_each")) if len(lines) > 1 else None))
        for w in (w for w in line.get("wants") or [] if isinstance(w, dict)):
            values = [str(v) for v in w.get("values") or []]
            if not (values and w.get("op") in ("is", "one_of") and set().union(*map(content, values)) <= item):
                out.add(("want", k, form({x: y for x, y in dict(w, values=values).items() if x != "key"})))
    for rel in (r for r in brief.get("relations") or [] if isinstance(r, dict)):
        if rel.get("type") == "fits":
            on, target = fits_parts(rel)
            out.add(("relation", "fits", tuple(sorted({repr(named(v)) for v in on})), named(target) if target is not None else None))
        else:
            out.add(("relation", form(rel)))
    budget = {float(v) for v in (brief.get("messages"), brief.get("steps")) if number(v) is not None}
    out |= {("procedure", spelled(t)) for t in (brief.get("procedures") or []) + [brief.get("decline_rule")] if isinstance(t, str) and t.strip() and not set(numbers_in(t)) & budget}
    return out


def differences(a: dict, b: dict) -> list[str]:
    ma, mb = meaning(a), meaning(b)
    within = lambda x, other: x[0] == "procedure" and any(y[0] == "procedure" and (x[1] in y[1] or y[1] in x[1]) for y in other)
    out = [f"only reading {name} says: {x}" for name, mine, other in (("A", ma, mb), ("B", mb, ma)) for x in sorted(mine - other, key=repr) if not within(x, other)]
    for k, (x, y) in enumerate(zip(*[[content(l.get("item")) for l in r.get("lines") or [] if isinstance(l, dict)] for r in (a, b)]), 1):
        if not (x <= y or y <= x):
            out.append(f"line {k} item: reading A {sorted(x)}, reading B {sorted(y)}")
    return out


def conflicts(brief: dict) -> list[str]:
    lines = [l for l in brief.get("lines") or [] if isinstance(l, dict)]
    out = []
    for rel in (r for r in brief.get("relations") or [] if isinstance(r, dict) and r.get("type") == "if_then" and isinstance(r.get("then"), dict)):
        then, m = rel["then"], line_of(lines, rel["then"].get("line")) or 0
        for want in (lines[m - 1].get("wants") or []) if m else []:
            if isinstance(want, dict) and clash({"key": then.get("key"), "op": then.get("op") or "is", "values": then.get("values")}, want):
                out.append(f"line {m}: the if_then {rel.get('quote')!r} gives {then.get('key')} {then.get('op')} {then.get('values')}, the stated requirement {want.get('quote')!r} gives {want.get('key')} {want.get('op')} {want.get('values')}")
    return out


def read_request(brain: Brain, request: str, trace: list) -> dict | None:
    check = lambda answer: brief_problem(answer, request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        readings = [r for r in pool.map(lambda _: brain.ask("INTAKE", INTAKE, "", "REQUEST:\n" + request, {"type": "object"}, check), range(2)) if r]
    if not readings:
        return None
    points = (differences(*readings) if len(readings) == 2 else []) + [f"reading {name}, {c}" for name, r in zip("AB", readings) for c in conflicts(r)]
    trace_row(trace, "intake_compare", readings=len(readings), points=points)
    if not [x for x in points if not x.startswith("only reading") or "'procedure'" not in x[:40]]:
        first = readings[0]
        said = [spelled(t) for t in procedure_texts(first, request)]
        extra = [t for r in readings[1:] for t in procedure_texts(r, request) if not any(spelled(t) in x or x in spelled(t) for x in said)]
        nearest = next((r["nearest"] for r in readings if isinstance(r.get("nearest"), str) and r["nearest"].strip()), None)
        return dict(first, procedures=procedure_texts(first, request) + extra, nearest=nearest)
    shown = "".join(f"\n\nREADING {name}:\n{json.dumps(r, ensure_ascii=False)}" for name, r in zip("AB", readings))
    tie = brain.ask("INTAKE_TIE", INTAKE + "\n\n" + INTAKE_TIE, "", f"REQUEST:\n{request}{shown}\n\nPOINTS TO DECIDE:\n" + "\n".join(points), {"type": "object"}, check)
    trace_row(trace, "intake_tie", points=points, chosen=tie, kept=None if tie else "the first reading: INTAKE_TIE gave no usable brief")
    return tie or readings[0]


def load_brief(ledger: Ledger, brief: dict, trace: list | None = None) -> None:
    for line in brief.get("lines") or []:
        wants = [dict(w, key=str(w.get("key") or ""), source="stated", values=[str(v) for v in (w.get("values") or [])]) for w in line.get("wants") or [] if isinstance(w, dict)]
        for want in (w for w in wants if w.get("op") == "ref"):
            want.update(followed=False, context=pointer_context(want.get("quote") or "", ledger.request, want["key"]))
        ledger.lines.append(
            {
                "item": str(line.get("item") or ""),
                "category": line.get("category"),
                "qty": int(line.get("qty") or 1),
                "objective": line.get("objective") if line.get("objective") in ("any", "cheapest", "best") else "any",
                "best_key": line.get("best_key"),
                "best_dir": line.get("best_dir"),
                "cap_each": line.get("cap_each"),
                "wants": wants,
                "verdicts": {},
                "pick": None,
                "standby": None,
                "seen": [],
                "asked": [],
                "told": [],
            }
        )
    ledger.total_cap = brief.get("total_cap") if isinstance(brief.get("total_cap"), (int, float)) else None
    ledger.total_objective = brief.get("total_objective") or "any"
    ledger.in_stock = brief.get("in_stock") is not False
    written = brief.get("messages") if isinstance(brief.get("messages"), int) and not isinstance(brief.get("messages"), bool) else None
    ledger.messages_left = len(ledger.lines) if written is None else written
    if written is None and trace is not None:
        trace_row(trace, "messages_unknown", messages=ledger.messages_left, reason="the request writes no message count: one message per line, and the runtime's budget_exceeded (Env.shut) ends them sooner; an unknown count is never zero")
    ledger.relations, ledger.fits, ledger.chain_pages, ledger.unhandled = [], [], [], []
    for rel in brief.get("relations") or []:
        if isinstance(rel, dict):
            use = file_relation(ledger, rel)
            ledger.relations.append(dict(rel, use=use))
            if use.startswith(("dropped", "unhandled")):
                ledger.unhandled.append({"kind": "relation", "relation": rel, "quote": rel.get("quote"), "why": use})
                if trace is not None:
                    trace_row(trace, "relation_unhandled", relation=rel, why=use, reason="no code path acts on it: the final check reads its sentence")
    ledger.procedures = procedure_texts(brief, ledger.request)
    ledger.decline_rule = brief.get("decline_rule")
    ledger.nearest = brief["nearest"] if isinstance(brief.get("nearest"), str) and brief["nearest"].strip() else None
    ledger.story = [s for s in brief.get("story") or [] if isinstance(s, dict) and s.get("text")]
    ledger.checks = [c for c in brief.get("checks") or [] if isinstance(c, dict)]


def fits_parts(rel: dict) -> tuple[list, Any]:
    lines = rel["lines"] if isinstance(rel.get("lines"), list) else [rel.get(k) for k in ("line", "a", "line_a") if rel.get(k) is not None][:1]
    target = next((rel[k] for k in ("with", "with_line", "identifier", "b", "line_b") if rel.get(k) not in (None, "")), None)
    return (lines[:1], lines[1]) if target is None and len(lines) == 2 else (lines, target)


def file_relation(ledger: Ledger, rel: dict) -> str:
    count = len(ledger.lines)
    as_line = lambda v: line_of(ledger.lines, v)
    if rel.get("type") == "if_then":
        cond, then = rel.get("if"), rel.get("then")
        if not (isinstance(cond, dict) and isinstance(then, dict) and as_line(then.get("line")) in range(1, count + 1) and then.get("key") and then.get("values")):
            return "dropped: if_then without a then-line requirement"
        want = {"key": str(then["key"]), "op": then.get("op") or "is", "values": [str(v) for v in then["values"]], "quote": rel.get("quote") or "", "source": "stated", "when": dict(cond, line=as_line(cond.get("line")))}
        file_want(ledger, as_line(then["line"]), want)
        return f"want on line {as_line(then['line'])}"
    if rel.get("type") != "fits":
        return f"unhandled: no code acts on a {rel.get('type')!r} relation"
    lines, target = fits_parts(rel)
    lines = [m for m in map(as_line, lines) if m in range(1, count + 1)]
    other = as_line(target)
    if other in range(1, count + 1):
        for m in (m for m in lines if m != other):
            ledger.fits.append({"line": m, "with": other, "quote": rel.get("quote") or ""})
        return f"fitter: lines {lines} with line {other}"
    code = str(target or "").strip()
    if not (code and stands(code, ledger.request) and lines):
        return f"dropped: {code!r} is not written in the request"
    for m in lines:
        file_want(ledger, m, {"key": "fits", "op": "fits", "values": [code], "quote": rel.get("quote") or code, "source": "stated"})
    return f"fits want on lines {lines}"

def content_words(text: Any) -> list[str]:
    return [w for w in words(text) if w not in FUNCTION_WORDS]


class Engine:

    def __init__(self, trace: list) -> None:
        self.trace = trace
        self.seen: list[dict] = []
        self.tier_votes: list[bool] = []
        self.caps: dict[str, dict] = {}
        self.facts: dict[str, Any] = {"or": None, "tiers": None, "hidden": None, "numbers": None, "first_words": None, "capped": None, "price_sorted": None, "price_cap": None}
        self.evidence: dict[str, Any] = {}

    def note(self, query: str, cards: list[dict], count: Any, page: int) -> None:
        qwords = content_words(query)
        rows = [norm_text(c.get("title", "") + " " + c.get("text", "") + " " + c.get("category", "")) for c in cards]
        matched = [sum(1 for w in qwords if any(f" {v} " in r for v in variants(w))) for r in rows]
        prices = [c["price"] for c in cards if isinstance(c.get("price"), (int, float))]
        partial = sum(1 for m in matched if 0 < m < len(qwords)) if len(qwords) >= 2 else 0
        ids = [c["id"] for c in cards]
        self.seen.append({"q": " ".join(qwords), "words": qwords, "ids": ids, "matched": matched, "count": count, "page": page, "prices": prices, "partial": partial,
                          "whole": len(qwords) >= 2 and bool(rows) and page == 0 and all(m in (0, len(qwords)) for m in matched)})
        f, ev = self.facts, self.evidence
        alike = lambda s: s["ids"] == ids and (s["count"] == count or not (isinstance(s["count"], (int, float)) and isinstance(count, (int, float))))
        if qwords and 0 in matched and not f["hidden"]:
            f["hidden"], ev["hidden"] = True, f"{query!r} page {page} returned {ids[matched.index(0)]}, which shows none of its words"
        f["or"], ev["or"] = self.or_matching()
        if len(matched) >= 4 and partial:
            half = len(matched) // 2
            self.tier_votes.append(sum(matched[:half]) / half > sum(matched[half:]) / (len(matched) - half))
            if len(self.tier_votes) >= 3:
                share = sum(self.tier_votes) / len(self.tier_votes)
                f["tiers"], ev["tiers"] = (True if share >= 0.7 else False if share <= 0.3 else None), f"{sum(self.tier_votes)} of {len(self.tier_votes)} pages ranked rows holding more words first"
        if page == 0 and cards:
            bare = " ".join(w for w in qwords if not w.replace(".", "").isdigit())
            numeric = bare != " ".join(qwords)
            for s in self.seen[:-1]:
                s_bare = " ".join(w for w in s["words"] if not w.replace(".", "").isdigit())
                if s["page"] == 0 and s["ids"] and s_bare == bare and (s_bare != s["q"]) != numeric:
                    f["numbers"], ev["numbers"] = not alike(s), f"{s['q']!r} and {query!r} differ only by numbers and returned {'the same' if alike(s) else 'different'} rows"
                    break
        if page == 0 and len(qwords) >= 2:
            same = lambda a, b: bool(variants(a) & variants(b))
            for s in self.seen[:-1]:
                if s["page"] != 0 or not s["ids"] or (len(s["words"]) == len(qwords) and all(map(same, s["words"], qwords))):
                    continue
                k = next((i for i, (a, b) in enumerate(zip(s["words"], qwords)) if not same(a, b)), min(len(s["words"]), len(qwords)))
                extra = (qwords if len(qwords) > len(s["words"]) else s["words"])[k:]
                if alike(s) and 1 <= k and [w for w in extra if sum(1 for r in rows if any(f" {v} " in r for v in variants(w))) * 2 < len(rows)]:
                    f["first_words"], ev["first_words"] = min(k, f["first_words"]) if isinstance(f["first_words"], int) else k, f"{s['q']!r} and {query!r} share their first {k} words and returned the same rows"
                elif not alike(s) and isinstance(f["first_words"], int) and k >= f["first_words"] and extra:
                    f["first_words"], ev["first_words"] = None, f"refuted: {s['q']!r} and {query!r} share their first {k} words and returned different rows"
        firsts = {s["q"]: s["count"] for s in self.seen if s["page"] == 0 and isinstance(s["count"], (int, float))}
        if len(firsts) >= 3:
            f["capped"], ev["capped"] = len(set(firsts.values())) == 1, f"first-page counts {sorted(set(firsts.values()))[:5]} over {len(firsts)} searches"
        if len(prices) >= 4 and f["price_sorted"] is not False:
            rising = all(a <= b for a, b in zip(prices, prices[1:]))
            f["price_sorted"], ev["price_sorted"] = rising, f"{query!r} page {page}: prices {'rose' if rising else 'did not rise'} ({prices[:6]})"
        trace_row(self.trace, "engine", query=query, page=page, matched=matched[:12], facts=f)

    def or_matching(self) -> tuple[bool | None, str]:
        firsts = [s for s in self.seen if s["page"] == 0 and isinstance(s["count"], int) and not isinstance(s["count"], bool) and s["words"]]
        forms = lambda s: [variants(w) for w in s["words"]]
        within = lambda a, b: len(a["words"]) < len(b["words"]) and all(any(fa & fb for fb in forms(b)) for fa in forms(a))
        pairs = [(a, b) for a in firsts for b in firsts if within(a, b) and b["count"] != a["count"]]
        if len({b["count"] > a["count"] for a, b in pairs}) == 1:
            a, b = pairs[0]
            return b["count"] > a["count"], f"{b['q']!r} adds words to {a['q']!r} and returned {b['count']} matches against {a['count']}"
        part, whole = next((s for s in self.seen if s["partial"]), None), next((s for s in self.seen if s["whole"]), None)
        if whole and (self.facts["hidden"] or not part):
            return False, f"{whole['q']!r}: every row of its first page shows all its words or none"
        return (True, f"{part['q']!r}: a row shows some but not all of its words") if part and not self.facts["hidden"] else (None, "")

    def tier_dropped(self, query: str, cards: list[dict]) -> bool:
        qwords = content_words(query)
        if not (self.facts["or"] and self.facts["tiers"]) or self.facts["hidden"] or len(qwords) < 2 or not cards:
            return False
        last = norm_text(cards[-1].get("title", "") + " " + cards[-1].get("text", ""))
        return sum(1 for w in qwords if any(f" {v} " in last for v in variants(w))) < len(qwords)

    def indexed(self, value: str) -> bool | None:
        own = content_words(value)
        if not own:
            return None
        holds = lambda s: all(any(variants(w) & variants(x) for x in s["words"]) for w in own)
        verdict = None
        for s in (s for s in self.seen if s["page"] == 0 and holds(s)):
            alone = len(s["words"]) == len(own)
            if alone and not s["ids"]:
                verdict = False
            elif s["ids"] and (alone or self.facts["or"] is False):
                verdict = True
        return verdict

    def note_cap(self, listing: str, cap: float, cards: list[dict], count: Any, page: int) -> None:
        seen = self.evidence.setdefault("price_cap", {})
        over = next((c for c in cards if isinstance(c.get("price"), (int, float)) and c["price"] > cap), None)
        if over:
            seen["ignored"] = f"listing {listing} under {cap} showed {over['id']} at {over['price']}"
        elif page == 0 and isinstance(count, (int, float)):
            counts = self.caps.setdefault(listing, {})
            counts[cap] = count
            if len(set(counts.values())) > 1:
                seen["filters"] = f"listing {listing}: first-page counts {counts} under different caps"
        self.facts["price_cap"] = None if len(seen) == 2 else (False if "ignored" in seen else True if "filters" in seen else None)
        trace_row(self.trace, "engine_cap", listing=listing, cap=cap, count=count, page=page, price_cap=self.facts["price_cap"])

    def searchable(self, value: str) -> str:
        if self.facts["numbers"] is False:
            return " ".join(p for p in value.split() if not re.fullmatch(r"[\d.,]+", p))
        return value

FUNCTION_WORDS = {"a", "an", "the", "and", "or", "of", "for", "with", "in", "on", "to", "by", "x"}
BOUNDS, KIND_RULE = ("at_least", "at_most", "between"), "no word naming the line's kind"


def variants(word: str) -> set[str]:
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return {w, w[:-3] + "y"}
    if w.endswith("es") and len(w) > 4 and (w[-3] in "sxz" or w[-4:-2] in ("ch", "sh")):
        return {w, w[:-2], w[:-1]}
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return {w, w[:-1]}
    return {w, w + "es" if w.endswith(("s", "x", "z", "ch", "sh")) else w[:-1] + "ies" if len(w) > 2 and w[-1] == "y" and w[-2] not in "aeiou" else w + "s"}


def one_form(word: str) -> str:
    return min(x for v in variants(word) for x in variants(v))


def unpossessed(text: Any) -> str:
    return re.sub(r"['’]s\b", "", str(text))


def surface(text: Any) -> list[tuple[str, str, int]]:
    return [(p, raw, k) for k, raw in enumerate(words(re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text)))) for p in (re.findall(r"\d+(?:\.\d+)?|[a-z]+", raw) if re.search(r"[a-z]", raw) and re.search(r"\d", raw) else [raw])]


def unit_like(a: str, b: str) -> bool:
    short, long_ = sorted((a, b), key=len)
    rest = iter(long_)
    return bool(variants(a) & variants(b)) or (len(short) >= 2 and short.isalpha() and long_.isalpha() and short[0] == long_[0] and all(c in rest for c in short))


def frames(toks: list[tuple[str, str, int]], value: list[str]) -> list[tuple[int, list[float]]]:
    num = lambda t: bool(re.fullmatch(r"\d+(?:\.\d+)?", t))
    fits = lambda v, t, j: num(t) if num(v) else (unit_like(v, t) if j and num(value[j - 1]) else bool(variants(v) & variants(t)))
    whole = lambda i, end: (i == 0 or toks[i - 1][2] != toks[i][2]) and (end == len(toks) or toks[end][2] != toks[end - 1][2])
    return [(i, [float(t) for v, (t, _, _) in zip(value, toks[i:]) if num(v)]) for i in range(len(toks) - len(value) + 1) if whole(i, i + len(value)) and all(fits(v, t, j) for j, (v, (t, _, _)) in enumerate(zip(value, toks[i : i + len(value)])))]


def item_words(line: dict) -> list[str]:
    found = []
    for text in (line.get("item") or "", line.get("category") or ""):
        ws = [w for w in words(text) if w not in FUNCTION_WORDS and not w.replace(".", "").isdigit()]
        found += [w for w in dict.fromkeys([w for w in ws if len(w) > 2] or ws) if w not in found]
    return found


def kind_words(line: dict) -> list[str]:
    held = {x for want in line.get("wants") or [] if not want.get("dropped") and want.get("op") in ("is", "one_of") and want.get("source") != "admit" for v in want.get("values") or [] for w in words(unpossessed(v)) for x in variants(w)}
    item = item_words({"item": line.get("item")})
    named = [w for w in item_words({"category": line.get("category")}) if w not in item and not variants(w) & held]
    return ([w for w in item if not variants(w) & held] or item) + named


def value_words(line: dict, wants: list[dict]) -> list[tuple[int, str]]:
    order = lambda kw: 0 if kw[1].get("op") == "fits" else 1 if kw[1].get("source") in ("disclosed", "notice") else 2
    found = []
    for k, want in sorted(enumerate(wants), key=order):
        op = want.get("op")
        if op not in ("fits", "is", "one_of") + BOUNDS or want.get("source") == "admit":
            continue
        for value in want.get("values") or []:
            text = " ".join(str(value).split())
            if op in BOUNDS:
                text = " ".join(w for w in words(re.sub(r"\d+(?:[.,]\d+)*", " ", text)) if len(w) > 1 and w not in FUNCTION_WORDS)
            elif text and op != "fits" and all(w.replace(".", "").isdigit() for w in words(text)):
                unit = " ".join(w for w in key_words(want.get("key", "")).split() if w not in FUNCTION_WORDS)
                text = f"{text} {unit}".strip()
            if text and all(text.lower() != v.lower() for _, v in found):
                found.append((k, text))
    return found


class Gather:

    def __init__(self, store: Store, ledger: Ledger, trace: list, engine: Engine | None = None, plan: StepPlan | None = None) -> None:
        self.store, self.ledger, self.trace = store, ledger, trace
        self.engine = engine or Engine(trace)
        self.plan = plan or StepPlan(store.env, ledger, store)
        self.cursors: dict[tuple, dict] = {}
        self.suggested: dict[int, list[str]] = {}
        self.consulted: dict[int, tuple] = {}
        self.asked: dict[int, list] = {}
        self.hits: dict[int, dict[str, int]] = {}
        self.cover: dict[int, dict[str, set]] = {}
        self.capped: set = set()
        self.admitted: dict[int, dict[str, str]] = {}
        self.stop_at = 10**9
        self.memo: dict[tuple, Any] = {}

    def key(self, text: str, extra: dict | None = None, role: str = "search") -> tuple[str, str]:
        tool, args = self.store.query_args(role, text, extra)
        return self.store.call_key(str(tool), args)

    def cursor_of(self, text: str, extra: dict | None = None, role: str = "search") -> dict | None:
        cur = self.cursors.get(self.key(text, extra, role))
        if cur or role != "search":
            return cur
        forms = lambda t: [one_form(w) for w in str(t).lower().split()]
        return next((c for c in self.cursors.values() if c["role"] == "search" and c["extra"] == dict(extra or {}) and c["rows"] and forms(c["text"]) == forms(text)), None)

    def sent(self, text: str, extra: dict | None = None, role: str = "search") -> bool:
        return self.cursor_of(text, extra, role) is not None

    def run(self, n: int, pages: int, resume: bool = False, steps: int | None = None) -> list[str]:
        line = self.ledger.line(n)
        if steps is None:
            steps = max(3, self.plan.walk_left() // (1 + sum(1 for k in range(n + 1, len(self.ledger.lines) + 1) if not self.ledger.line(k)["seen"])))
        self.stop_at = self.store.env.step + steps
        tried: set = set()
        more = self.plan.rounds("Gather.run")
        while not self.out_of_steps() and more(n, sorted(tried)):
            if line["seen"] and self.file_category(n, line):
                continue
            extra = self.listing(n)
            if extra and self.over_cap(n) and not self.sent("", extra, "browse") and "listing" not in tried:
                tried.add("listing")
                self.page(n, "", "listing", extra, "browse")
                continue
            rung = next((r for r in self.ladder(n) if not self.sent(r[0]) and r[0] not in tried), None)
            if rung:
                tried.add(rung[0])
                self.read(n, rung[0], rung[1] != "phrase", value_words(line, self.ledger.wants(n)), pages)
                continue
            if extra and self.listing_due(n) and not self.sent("", extra, "browse") and "listing" not in tried:
                tried.add("listing")
                self.page(n, "", "listing", extra, "browse")
                continue
            break
        if self.cheapest(line) and line.get("pick") and self.bands(n):
            self.band(n, bound=True)
        for text, kind in self.ladder(n):
            cur = self.cursor_of(text)
            if kind != "phrase" and cur and not cur["ended"] and not cur["stop"] and cur["page"] < pages and not self.out_of_steps():
                self.read(n, cur["text"], True, value_words(line, self.ledger.wants(n)), pages - cur["page"])
        self.widen(n, line)
        if resume:
            self.resume(n, within_share=True)
        if self.engine.facts.get("capped"):
            self.capped.add(n)
        order, doubt = self.ordered(n), self.kind_doubt(n)
        trace_row(self.trace, "gather", line=n, values=value_words(line, self.ledger.wants(n)), seen=len(line["seen"]), kept=len(order), capped=n in self.capped, order=order[:12])
        if doubt and not any(self.shown(n, self.store.cards[i], value_words(line, self.ledger.wants(n))) for i in order):
            trace_row(self.trace, "kind_doubt", line=n, rows=doubt[:8], kind_words=kind_words(line), reason="no kept row shows a value of the line while rows the kind rule set aside do: the kind words may not be the store's (Gather.admit brings a row back)")
        return order

    def ladder(self, n: int) -> list[tuple[str, str]]:
        line = self.ledger.line(n)
        wants = self.ledger.wants(n)
        values = [(k, self.engine.searchable(v)) for k, v in value_words(line, wants) if self.engine.indexed(self.engine.searchable(v)) is not False]
        values = [(k, v) for k, v in values if v.strip()]
        heads = self.heads(line)
        ahead = [(t, "value") for t in self.suggested.get(n, [])]
        k_words = self.engine.facts.get("first_words")
        if isinstance(k_words, int) and k_words >= 1:
            return ahead + self.short_queries(line, values, heads[0], k_words)
        out = []
        by_want: dict[int, list[str]] = {}
        for k, v in values:
            by_want.setdefault(k, []).append(v)
        alts = next((vs for k, vs in by_want.items() if wants[k].get("op") == "one_of" and len(vs) > 1), [None])[:4]
        for alt in alts if len(by_want) >= 2 else []:
            pick = [alt if alt is not None and alt in vs else vs[0] for vs in by_want.values()]
            out += [(self.fit_query(pick, h), "value") for h in heads]
        for k, v in values:
            out += [(self.fit_query([v], h), "bound" if wants[k].get("op") in BOUNDS else "value") for h in heads]
        out += [(self.fit_query([f], heads[0]), "value") for k, v in value_words(line, wants) if wants[k].get("op") not in BOUNDS for f in self.forms(n, v)]
        for k, v in values:
            if wants[k].get("op") not in BOUNDS and self.missed(n, [self.fit_query([v], h) for h in heads]):
                out.append((clip(v, self.store.query_limit()), "value"))
        if not values or self.missed(n, [t for t, kind in out]):
            out.append((self.phrase(line) or heads[0], "phrase"))
        return [r for r in dict.fromkeys(ahead + out) if r[0]]

    def phrase(self, line: dict) -> str:
        return clip(" ".join(w for w in words(unpossessed(line.get("item") or "")) if w not in FUNCTION_WORDS and w not in set(item_words(line)) - set(kind_words(line))), self.store.query_limit())

    def missed(self, n: int, texts: list[str]) -> bool:
        line = self.ledger.line(n)
        curs = [self.cursor_of(t) for t in texts]
        return bool(curs) and all(c is not None for c in curs) and not any(self.keep(line, self.store.cards[i]) for c in curs for i in c["got"])

    def fit_query(self, values: list[str], head: str) -> str:
        limit = self.store.query_limit()
        tokens = list(dict.fromkeys(t for v in values for t in v.split()))
        tail = [head] if head and head not in {x for t in tokens for w in words(t) for x in variants(w)} else []
        while len(tokens) > 1 and len(" ".join(tokens + tail)) > limit:
            tokens.pop()
        return clip(" ".join(tokens + tail), limit)

    def heads(self, line: dict) -> list[str]:
        first, shelf = self.head(line), self.shelf(line)
        out = [first] if first else []
        for part in re.split(r",|&|\band\b", str(line.get("item") or "")) if shelf else []:
            h = self.head(dict(line, item=part)) if item_words({"item": part}) else ""
            if h and all(not variants(h) & variants(x) for x in out) and any(variants(h) & variants(w) for w in words(shelf)):
                out.append(h)
        return out or [""]

    def page(self, n: int, text: str, kind: str, extra: dict | None = None, role: str = "search", bound: bool = True) -> dict | None:
        k = self.key(text, extra, role)
        cur = self.cursors.get(k)
        if (cur and cur["ended"]) or (self.out_of_steps() if bound else self.walk_over()):
            return None
        result = self.store.search(text, cur["next"] if cur else None, role=role, extra=extra)
        if result["error"] and not result["cards"]:
            return None
        if cur is None:
            cur = self.cursors[k] = {"text": text, "kind": kind, "role": role, "extra": dict(extra or {}), "next": None, "page": 0, "rows": 0, "got": [], "held": 0, "sorted": True, "last": None, "ids": None, "count": None,
                                     "ended": False, "stop": None}
        if k not in self.asked.setdefault(n, []):
            self.asked[n].append(k)
        arg = (self.store.roles.get(role) or {}).get("price_arg")
        if role == "search":
            self.engine.note(text, result["cards"], result["count"], cur["page"])
        elif arg in cur["extra"] and isinstance(cur["extra"][arg], (int, float)):
            self.engine.note_cap(self.key("", {a: v for a, v in cur["extra"].items() if a != arg}, role)[1], cur["extra"][arg], result["cards"], result["count"], cur["page"])
        self.file(n, cur, result["cards"])
        self.advance(n, cur, result)
        return cur

    def file(self, n: int, cur: dict, cards: list[dict]) -> None:
        qwords = set(words(cur["text"]))
        for m, line in enumerate(self.ledger.lines, 1):
            covered = {k for k, v in value_words(line, self.ledger.wants(m)) if words(self.engine.searchable(v)) and set(words(self.engine.searchable(v))) <= qwords}
            hits, cover = self.hits.setdefault(m, {}), self.cover.setdefault(m, {})
            for rank, card in enumerate(cards):
                if m != n and not self.keep(line, self.store.cards.get(card["id"], card)):
                    continue
                if card["id"] not in line["seen"]:
                    line["seen"].append(card["id"])
                if covered:
                    hits[card["id"]] = min(hits.get(card["id"], 10**6), cur["rows"] + rank)
                    cover.setdefault(card["id"], set()).update(covered)

    def advance(self, n: int, cur: dict, result: dict) -> None:
        line = self.ledger.line(n)
        values = value_words(line, self.ledger.wants(n))
        cards = result["cards"]
        ids = [c["id"] for c in cards]
        prices = ([cur["last"]] if cur["last"] is not None else []) + [c["price"] for c in cards if isinstance(c.get("price"), (int, float))]
        cur["held"] += sum(1 for c in cards if self.keep(line, self.store.cards.get(c["id"], c)) and self.shown(n, c, values))
        again = ids == cur["ids"]
        cur.update(next=result["next"], page=cur["page"] + 1, rows=cur["rows"] + len(cards), ids=ids, sorted=cur["sorted"] and all(a <= b for a, b in zip(prices, prices[1:])), last=prices[-1] if prices else None,
                   count=result["count"] if cur["count"] is None else cur["count"])
        cur["got"] += [i for i in ids if i not in cur["got"]]
        count = cur["count"]
        reached = isinstance(count, (int, float)) and cur["rows"] >= count and not self.engine.facts.get("capped")
        why = "an empty page" if not cards else "the same page again" if again else (result.get("end") or "no next value") if cur["next"] is None else f"the count ({count}) reached" if reached else None
        if why and isinstance(count, (int, float)) and cur["rows"] < count:
            why += f"; the count ({count}) is above the {cur['rows']} rows read"
        cur["ended"] = why or False

    def read(self, n: int, text: str, by_value: bool, values: list, pages: int, cap: float | None = None, retry: bool = True) -> None:
        line = self.ledger.line(n)
        kind = self.kind(n, text) if by_value else "phrase"
        for _ in range(pages):
            seen = set(line["seen"])
            cur = self.page(n, text, kind)
            if cur is None:
                break
            if cur["page"] == 1 and not cur["rows"] and by_value:
                if retry:
                    self.retry_empty(n, text)
                break
            new = [i for i in cur["ids"] if i not in seen]
            shown_value = any(self.shown(n, self.store.cards.get(i, {}), values) for i in cur["ids"])
            if cap is not None and cur["sorted"]:
                why = "past the cap on a search whose prices rose" if not cur["ended"] and isinstance(cur["last"], (int, float)) and cur["last"] > cap else None
            else:
                why = (
                    "no new rows"
                    if not new
                    else (
                        "no next page"
                        if cur["next"] is None
                        else "value not shown" if (by_value and cur["page"] >= 3 and not shown_value and not self.engine.facts["hidden"]) else "tier dropped" if self.engine.tier_dropped(text, [self.store.cards[i] for i in cur["ids"]]) else None
                    )
                )
            if why:
                cur["stop"] = why
                trace_row(self.trace, "page_stop", line=n, query=text, page=cur["page"] - 1, why=why)
                break

    def kind(self, n: int, text: str) -> str:
        wants = self.ledger.wants(n)
        held = [k for k, v in value_words(self.ledger.line(n), wants) if words(v) and set(words(v)) <= set(words(text))]
        return "bound" if held and all(wants[k].get("op") in BOUNDS for k in held) else "value"

    def retry_empty(self, n: int, text: str) -> None:
        line = self.ledger.line(n)
        wants = self.ledger.wants(n)
        values = value_words(line, wants)
        bound = {w for k, v in values if wants[k].get("op") in BOUNDS for w in words(v)}
        heads = {x for h in self.heads(line) for x in variants(h)}
        if bound & set(words(text)):
            alt = " ".join(t for t in text.split() if not set(words(t)) <= bound)
        else:
            alt = next((s for _, v in values for s in [self.engine.searchable(v)] for h in self.heads(line) if self.fit_query([s], h) == text != s), "")
        if not alt or not set(words(alt)) - heads or self.sent(alt):
            return
        self.stop_at = max(self.stop_at, self.store.env.step + 1)
        self.read(n, alt, True, values, 1, retry=False)
        cur = self.cursors.get(self.key(alt))
        trace_row(self.trace, "retry_empty", line=n, query=text, alt=alt, rows=cur["rows"] if cur else None)

    def widen(self, n: int, line: dict) -> None:
        shelf = self.listed_shelf(line)
        rows = [self.store.cards[i] for i in line["seen"] if norm_text(self.store.cards.get(i, {}).get("category")) == norm_text(shelf)] if shelf else []
        item = {x for h in item_words(line) for x in variants(h)}
        hits, head = self.hits.get(n, {}), self.head(line)
        for _, value in value_words(line, self.ledger.wants(n)):
            vw = [w for w in words(value) if w not in FUNCTION_WORDS]
            if not vw or any(hits.get(c["id"], 10**6) < 10**6 for c in rows if any(f" {w} " in norm_text(c.get("text", "") + " " + c.get("title", "")) for w in vw)):
                continue
            near: dict[str, int] = {}
            for c in rows:
                text = norm_text(c.get("title", "") + " " + c.get("text", ""))
                if any(f" {w} " in text for w in vw if not w.replace(".", "").isdigit()) or f" {' '.join(vw)} " in text:
                    for w in set(text.split()):
                        if len(w) > 3 and w not in FUNCTION_WORDS and w not in vw and not w.replace(".", "").isdigit() and w not in item:
                            near[w] = near.get(w, 0) + 1
            for w, _ in sorted(near.items(), key=lambda kv: (-kv[1], kv[0]))[:2]:
                if not self.sent(f"{w} {head}".strip()) and self.page(n, f"{w} {head}".strip(), "widen"):
                    trace_row(self.trace, "widen", line=n, value=value, word=w)

    def short_queries(self, line: dict, values: list[tuple[int, str]], head: str, k: int) -> list[tuple[str, str]]:
        limit = self.store.query_limit()
        seen_text = [norm_text((self.store.cards.get(i) or {}).get("title", "") + " " + (self.store.cards.get(i) or {}).get("text", "")) for i in line.get("seen") or []]
        rarity = lambda w: sum(1 for t in seen_text if f" {w} " in t)
        out = []
        pool = [w for _, v in values for w in words(v) if w not in FUNCTION_WORDS]
        if len({j for j, _ in values}) >= 2 and pool:
            out.append((clip(" ".join(sorted(dict.fromkeys(pool), key=lambda w: (rarity(w), pool.index(w)))[:k]), limit), "value"))
        for _, value in values:
            vw = [w for w in words(value) if w not in FUNCTION_WORDS]
            if not vw:
                continue
            chosen = sorted(vw, key=lambda w: (rarity(w), vw.index(w)))[: max(1, k - 1)]
            chosen = [w for w in vw if w in chosen]
            out.append((clip(" ".join(chosen + ([head] if head and head not in chosen and len(chosen) < k else [])), limit), "value"))
        out.append((head, "phrase"))
        return [r for r in dict.fromkeys(out) if r[0]]

    def resume(self, n: int, within_share: bool = False) -> list[str] | None:
        if not within_share:
            self.stop_at = 10**9
        line = self.ledger.line(n)
        more = self.plan.rounds("Gather.resume")
        cursors = lambda: [(c["text"], c["extra"], c["next"], c["ended"]) for k in self.asked.get(n, []) for c in [self.cursors.get(k)] if c]
        rungs = (("store words", self.consult), ("search", self.next_search), ("cover listing", lambda m: self.next_listing(m, True)), ("band", self.band), ("value page", lambda m: self.next_page(m, True, True)),
                 ("deep page", lambda m: self.next_page(m, True)), ("listing", lambda m: self.next_listing(m, False)), ("phrase page", lambda m: self.next_page(m, False)))
        for name, rung in rungs:
            while more(name, cursors()):
                before = set(line["seen"])
                got = rung(n)
                if got is None:
                    break
                fresh = got if isinstance(got, list) else [i for i in self.ordered(n) if i not in before and not self.store.opened(i)]
                if fresh:
                    trace_row(self.trace, "dig_page", line=n, rung=name, new=len(fresh), first=fresh[:6])
                    return fresh
        trace_row(self.trace, "dig_spent", line=n, walk_over=self.walk_over())
        return None

    def next_search(self, n: int) -> bool | None:
        rung = next((r for r in self.ladder(n) if not self.sent(r[0])), None)
        if rung is None:
            return None
        self.read(n, rung[0], rung[1] != "phrase", value_words(self.ledger.line(n), self.ledger.wants(n)), 1)
        return True if self.sent(rung[0]) else None

    def next_listing(self, n: int, cover: bool) -> bool | None:
        line, extra = self.ledger.line(n), self.listing(n)
        unindexed = [(k, v) for k, v in value_words(line, self.ledger.wants(n)) if self.engine.indexed(self.engine.searchable(v)) is False]
        due = (any(self.shown(n, self.store.cards[i], unindexed) for i in line["seen"]) or self.over_cap(n)) if cover else self.listing_due(n)
        return True if extra and due and self.page(n, "", "listing", extra, "browse") else None

    def next_page(self, n: int, value: bool, fitting: bool = False) -> bool | None:
        line = self.ledger.line(n)
        live = [c for c in self.mine(n) if (c["kind"] in ("value", "bound")) == value and not c["ended"] and not (fitting and self.unserved(n, c))]
        if not live:
            return None
        cap = self.cover_cap(line)
        past = lambda c: cap is not None and c["sorted"] and isinstance(c["last"], (int, float)) and c["last"] > cap
        cur = max(live, key=lambda c: (not past(c), c["kind"] not in ("bound", "widen"), self.cheapest(line) and c["sorted"], c["held"] / max(1, c["rows"]), -c["page"]))
        page = cur["page"]
        self.read(n, cur["text"], value, value_words(line, self.ledger.wants(n)), 1, cap)
        return True if cur["page"] > page else None

    def readable(self, cur: dict) -> int:
        more = cur["rows"] + max(1, cur["rows"] // max(1, cur["page"])) * max(1, self.plan.walk_left() // max(1, len(self.ledger.lines)) // 2)
        cap = self.served(cur)
        return more if cap is None else min(more, cap)

    def served(self, cur: dict) -> int | None:
        return self.store.rows_reach(cur["role"], max(1, cur["rows"] // max(1, cur["page"])))

    def unserved(self, n: int, cur: dict) -> bool:
        cap = self.served(cur)
        return cap is not None and isinstance(cur["count"], (int, float)) and cur["count"] > max(len(cur["got"]), cap) and not self.read_whole(n, cur)

    def read_whole(self, n: int, cur: dict) -> bool:
        if not isinstance(cur["count"], (int, float)):
            return bool(cur["ended"])
        return len(self.base_rows(n, cur)) >= cur["count"]

    def base_rows(self, n: int, cur: dict) -> list[str]:
        forms = lambda c: [variants(w) for w in words(c["text"])]
        adds = lambda c: c is not cur and c["extra"] == cur["extra"] and all(any(f & g for g in forms(c)) for f in forms(cur))
        return list(dict.fromkeys(cur["got"] + [i for c in self.mine(n) if adds(c) for i in c["got"]])) if self.engine.facts.get("or") is False else list(cur["got"])

    def consult(self, n: int) -> bool | None:
        line, wants = self.ledger.line(n), self.ledger.wants(n)
        mine = self.mine(n)
        seen = [self.store.cards[i] for i in line["seen"] if i in self.store.cards]
        opened = [self.store.cards[i] for i in line["verdicts"] if i in self.store.cards]
        mark = (len(opened), tuple(want_tag(w) for w in wants), len(mine))
        self.suggested[n] = [t for t in self.suggested.get(n, []) if not self.sent(t)]
        if self.suggested[n] or not opened or self.consulted.get(n) == mark:
            return None
        self.consulted[n] = mark
        limit = self.store.query_limit()
        label = lambda i, w: (line["verdicts"].get(i, {}).get(want_tag(w)) or ("UNKNOWN",))[0]
        reqs = "\n".join(f"{k}. {want_text(w)} (STORE USE: {store_use(opened, w, line)})" for k, w in enumerate(wants, 1)) or "(none)"
        dropped = [w for w in line["wants"] if w.get("dropped")]
        searches = "\n".join(f'"{c["text"]}": {c["count"] if c["count"] is not None else len(c["got"])} rows found, {c["page"]} page(s) read, {"no more pages" if c["ended"] else "more pages"}' for c in mine) or "(none)"
        shown = lambda c: f'{c["id"]} | {cut_text(c.get("title") or "", 120)} | {c.get("category") or ""} | {c.get("price")}'
        cards = "\n".join(shown(c) + " | " + ", ".join(f"{k} {label(c['id'], w)}" for k, w in enumerate(wants, 1)) for c in opened[-8:]) or "(none)"
        rows = "\n".join(shown(self.store.cards[i]) for i in [i for i in self.ordered(n) if not self.store.opened(i)][:8]) or "(none)"
        task = f"LINE: {line['item']} (objective {line.get('objective')}, cap {line.get('cap_each')})\nREQUIREMENTS:\n{reqs}\nDROPPED:\n" + ("\n".join(want_text(w) for w in dropped) or "(none)")
        answer = self.store.brain.ask("SEARCH", SEARCH, task, f"SEARCHES:\n{searches}\nOPENED:\n{cards}\nROWS:\n{rows}\nLIMIT: {limit} characters", SEARCH_SCHEMA) or {}
        known = content(" ".join([self.ledger.request, line.get("item") or ""] + [" ".join([w.get("key") or "", " ".join(w.get("values") or []), w.get("quote") or ""]) for w in wants]
                                 + [" ".join([c.get("title") or "", c.get("category") or "", c.get("brand") or "", c.get("seller") or ""] + [f"{k} {v}" for k, v in (c.get("facts") or {}).items()]) for c in seen + opened]))
        gone = content(" ".join(" ".join(w.get("values") or []) for w in dropped)) - content(" ".join([line.get("item") or ""] + [" ".join(w.get("values") or []) for w in wants]))
        said = {frozenset(content(c["text"])) for c in mine}
        kept, refused = [], []
        for s in (x for x in answer.get("searches") or [] if isinstance(x, dict)):
            text = clip(" ".join(str(s.get("text") or "").split()), limit)
            ws = content(text)
            why = "empty" if not ws else "a word no request, requirement, record or row gives" if not ws <= known else "already sent" if frozenset(ws) in said or self.sent(text) else "a dropped requirement's word" if ws & gone else None
            (refused.append((text, why)) if why else kept.append(text))
            said.add(frozenset(ws))
        self.suggested[n] = kept[:2]
        trace_row(self.trace, "store_words", line=n, terms=answer.get("terms"), searches=kept[:2], refused=refused, answered=bool(answer))
        return True if self.suggested[n] else None

    def mine(self, n: int) -> list[dict]:
        return [self.cursors[k] for k in self.asked.get(n, []) if k in self.cursors and self.cursors[k]["role"] == "search"]

    def listing(self, n: int) -> dict | None:
        spec = self.store.roles.get("browse") or {}
        line, arg = self.ledger.line(n), spec.get("price_arg")
        shelf, cap = self.shelf(line) or (self.listed_shelf(line) if self.over_cap(n) else None), self.line_cap(line)
        if not spec.get("tool") or not spec.get("category_arg") or not shelf:
            return None
        if not arg or not isinstance(cap, (int, float)):
            return {spec["category_arg"]: shelf}
        wider = [c["extra"] for c in self.cursors.values() if c["role"] == "browse" and c["kind"] == "listing" and c["extra"].get(spec["category_arg"]) == shelf and isinstance(c["extra"].get(arg), (int, float)) and c["extra"][arg] >= cap]
        return min(wider, key=lambda e: e[arg]) if wider else {spec["category_arg"]: shelf, arg: cap}

    def listing_due(self, n: int) -> bool:
        line = self.ledger.line(n)
        if any(c["kind"] in ("value", "bound") and c["page"] and not c["rows"] for c in self.mine(n)):
            return True
        got = list(dict.fromkeys(i for c in self.mine(n) for i in c["got"]))
        return bool(got) and 2 * sum(1 for i in got if self.keep(line, self.store.cards[i])) < len(got)

    def over_cap(self, n: int) -> bool:
        line, cap = self.ledger.line(n), self.line_cap(self.ledger.line(n))
        got = [self.store.cards[i] for c in self.mine(n) for i in c["got"]]
        priced = [c["price"] for c in got if isinstance(c.get("price"), (int, float))]
        return isinstance(cap, (int, float)) and bool(priced) and all(p > cap for p in priced)

    def any_line(self, line: dict) -> bool:
        return line.get("objective") == "any" and not self.cheapest(line)

    def cheapest(self, line: dict) -> bool:
        return line.get("objective") == "cheapest" or self.ledger.total_objective == "cheapest"

    def bands(self, n: int) -> dict | None:
        line = self.ledger.line(n)
        spec = self.store.roles.get("browse") or {}
        extra, arg = self.listing(n), spec.get("price_arg")
        if not self.cheapest(line) or not extra or not arg or self.engine.facts.get("price_cap") is False:
            return None
        whole = (tool_props(self.store.tools.get(spec.get("tool"), {})).get(arg) or {}).get("type") == "integer"
        base = {a: v for a, v in extra.items() if a != arg}
        pick = self.store.cards.get(line.get("pick") or "", {}).get("price")
        pick = pick if isinstance(pick, (int, float)) else None
        target = pick if pick is not None else self.line_cap(line)
        if not isinstance(target, (int, float)) or target < 0:
            return None
        target = (math.ceil(target) if pick is not None else math.floor(target)) if whole else target
        found = {c["extra"][arg]: c for c in self.cursors.values() if c["role"] == "browse" and arg in c["extra"] and {a: v for a, v in c["extra"].items() if a != arg} == base}
        return {"base": base, "arg": arg, "target": target, "whole": whole, "found": found, "pick": pick}

    def band_rows(self, n: int, cur: dict, pick: float | None) -> list[str]:
        line = self.ledger.line(n)
        rows = [i for i in cur["got"] if i != line.get("pick") and self.keep(line, self.store.cards[i]) and isinstance(self.store.cards[i].get("price"), (int, float)) and (pick is None or self.store.cards[i]["price"] <= pick)]
        return sorted(rows, key=lambda i: (self.store.cards[i]["price"], cur["got"].index(i)))

    def band(self, n: int, bound: bool = False) -> list[str] | None:
        state = self.bands(n)
        if state is None or self.band_covered(n, state):
            return None
        line, found, pick = self.ledger.line(n), state["found"], state["pick"]
        done = lambda c: c["ended"] and all(self.store.opened(i) for i in self.band_rows(n, c, pick))
        top = state["target"]
        more = self.plan.rounds("Gather.band")
        while True:
            if not more(top, sorted(found)):
                return None
            cur = found.get(top) or self.page(n, "", "band", {**state["base"], state["arg"]: top}, "browse", bound)
            if cur is None:
                return None
            found[top] = cur
            readable = self.readable(cur)
            if cur["ended"] or cur["page"] > 1 or not isinstance(cur["count"], (int, float)) or cur["count"] <= readable:
                break
            prices = [self.store.cards[i]["price"] for i in line["seen"] if self.keep(line, self.store.cards[i]) and isinstance(self.store.cards[i].get("price"), (int, float))]
            floor = max([min(prices, default=0)] + [cap for cap, c in found.items() if cap < top and done(c)])
            lower = math.floor((floor + top) / 2) if state["whole"] else round((floor + top) / 2, 2)
            if lower <= floor or (lower not in found and len(found) >= 4):
                trace_row(self.trace, "band_partial", line=n, top=top, count=cur["count"], readable=readable, pick=pick)
                return None
            top = lower
        while not cur["ended"]:
            if not more(cur["next"], len(cur["got"])) or self.page(n, "", "band", cur["extra"], "browse", bound) is None:
                return None
        rows = [i for i in self.band_rows(n, cur, pick) if not self.store.opened(i)]
        if not rows and top < state["target"]:
            trace_row(self.trace, "band_partial", line=n, top=top, target=state["target"], pick=pick)
        return rows or None

    def band_covered(self, n: int, state: dict) -> bool:
        return any(cap >= state["target"] and c["ended"] and all(self.store.opened(i) for i in self.band_rows(n, c, state["pick"])) for cap, c in state["found"].items())

    def cover_rows(self, n: int) -> list[str]:
        line = self.ledger.line(n)
        pick = self.store.cards.get(line.get("pick") or "", {}).get("price")
        if not self.cheapest(line) or not isinstance(pick, (int, float)):
            return []
        rows = [i for i in line["seen"] if i != line["pick"] and self.keep(line, self.store.cards[i]) and price_of(self.store.cards[i], 0) <= pick]
        return sorted(rows, key=lambda i: (price_of(self.store.cards[i], math.inf), line["seen"].index(i)))

    def covered(self, n: int) -> bool:
        return self.cover_ground(n)[0]

    def cover_ground(self, n: int) -> tuple[bool, str]:
        line = self.ledger.line(n)
        if not self.cheapest(line) and line.get("objective") != "best":
            return True, "the line is neither cheapest nor best"
        unopened = [i for i in self.cover_rows(n) if not self.store.opened(i)]
        if unopened:
            return False, f"{len(unopened)} kept rows not shown priced above the pick are unopened: {unopened[:6]}"
        state = self.bands(n)
        if state is not None:
            done = self.band_covered(n, state)
            return done, f"band of {state['base']} up to {state['target']}: " + ("read to its end, every kept row at or under the pick opened" if done else "not yet read to its end or not every kept row at or under the pick opened")
        own = [c for c in self.mine(n) if c["kind"] == "value"] or [c for c in self.mine(n) if c["kind"] == "phrase"]
        pick = self.store.cards.get(line.get("pick") or "", {}).get("price")
        stated = (self.store.roles.get("search") or {}).get("order") == "price"
        past = lambda c: stated and c["sorted"] and isinstance(pick, (int, float)) and isinstance(c["last"], (int, float)) and c["last"] >= pick
        open_ = [c["text"] for c in own if not (self.read_whole(n, c) or past(c))]
        ground = "no value or phrase search sent" if not own else f"searches not read whole: {open_}" if open_ else "every value search read whole" + (" or passed the pick's price in the stated price order" if stated else "")
        return bool(own) and not open_, ground

    def cover_cap(self, line: dict) -> float | None:
        pick = self.store.cards.get(line.get("pick") or "", {}).get("price") if self.cheapest(line) else None
        caps = [c for c in (self.line_cap(line), pick) if isinstance(c, (int, float))]
        return min(caps) if caps and self.engine.facts.get("price_sorted") else None

    def line_cap(self, line: dict) -> float | None:
        n = next((k for k, l in enumerate(self.ledger.lines, 1) if l is line), None)
        room = self.room(n) if n else None
        share = room if isinstance(room, (int, float)) else self.ledger.total_cap
        caps = [c for c in (line.get("cap_each"), share / max(1, line["qty"]) if isinstance(share, (int, float)) else None) if isinstance(c, (int, float))]
        return min(caps) if caps else None

    def room(self, n: int) -> float | None:
        cap, lines = self.ledger.total_cap, self.ledger.lines
        memo = ("room", n, cap, self.store.env.step, tuple((len(l["verdicts"]), sum(map(len, l["verdicts"].values())), len(l["wants"])) for l in lines))
        if memo not in self.memo:
            least = {m: self.cheapest_pass(m) * lines[m - 1]["qty"] for m in range(1, len(lines) + 1)} if isinstance(cap, (int, float)) else {}
            self.memo[memo] = cap - sum(v for m, v in least.items() if m != n) if least and sum(least.values()) <= cap else None
        return self.memo[memo]

    def cheapest_pass(self, m: int) -> float:
        line, wants = self.ledger.line(m), self.ledger.wants(m)
        own = line.get("cap_each")
        buyable = lambda c: isinstance(c.get("price"), (int, float)) and not (self.ledger.in_stock and c.get("stock") is False) and not (isinstance(own, (int, float)) and c["price"] > own)
        return min((c["price"] for i, got in line["verdicts"].items() for c in [self.store.cards.get(i) or {}] if buyable(c) and all(w.get("op") != "ref" and got.get(want_tag(w), ("",))[0] == "PASS" for w in wants)), default=0)

    def walk_over(self) -> bool:
        return self.plan.walk_over()

    def out_of_steps(self) -> bool:
        return self.walk_over() or self.store.env.step >= self.stop_at

    def title_counts(self, line: dict) -> dict[str, int]:
        candidates = kind_words(line)
        seen = [self.store.cards.get(i) or {} for i in line.get("seen") or []]
        counts = {w: 0 for w in candidates}
        for field in ("title", "category"):
            for card in seen:
                text = norm_text(card.get(field, ""))
                for w in candidates:
                    counts[w] += any(f" {v} " in text for v in variants(w))
            if any(counts.values()):
                break
        return counts

    def title_head(self, line: dict) -> str:
        candidates, counts = kind_words(line), self.title_counts(line)
        if not candidates:
            return ""
        best = max(candidates, key=lambda w: (counts[w], candidates.index(w)))
        return best if counts[best] else candidates[-1]

    def head(self, line: dict) -> str:
        candidates = kind_words(line)
        if not candidates:
            return ""
        seen = [self.store.cards.get(i) or {} for i in line.get("seen") or []]
        counts, shelf = self.title_counts(line), self.shelf(line)
        nouns = [w for w in kind_words(dict(line, category=None)) if shelf and any(variants(w) & variants(x) for x in words(shelf))]
        if nouns:
            rows = [(norm_text(c.get("category", "")) == norm_text(shelf), norm_text(c.get("title", "") + " " + c.get("category", ""))) for c in seen]
            carry = lambda w, on: sum(1 for here, t in rows if here == on and any(f" {v} " in t for v in variants(w)))
            word = min(nouns, key=lambda w: (carry(w, False) > carry(w, True), -counts.get(w, 0), -nouns.index(w)))
        else:
            word = self.title_head(line)
        spelled: dict[str, int] = {}
        for card in seen:
            for form in sorted(variants(word)):
                spelled[form] = spelled.get(form, 0) + norm_text(card.get("category", "") + " " + card.get("title", "")).count(f" {form} ")
        return max(spelled, key=lambda f: (spelled[f], f == word, f)) if spelled and max(spelled.values()) else word

    def enumerated(self) -> list[str]:
        for role in ("browse", "search"):
            spec = self.store.roles.get(role) or {}
            enum = (tool_props(self.store.tools.get(spec.get("tool"), {})).get(spec.get("category_arg") or "") or {}).get("enum")
            if isinstance(enum, list) and enum:
                return [str(e) for e in enum]
        return []

    def shelf(self, line: dict) -> str | None:
        order, names, within = self.shelf_terms(line)
        item, head = names(line.get("item") or ""), self.title_head(line)
        allowed = item + [w for want in line.get("wants") or [] if not want.get("dropped") and want.get("op") in ("is", "one_of") and want.get("source") != "admit" for v in want.get("values") or [] for w in names(v)]
        fits = [c for c in self.enumerated() or order if names(c) and head and within(names(c), allowed) and within([head], names(c))]
        rank = {norm_text(c): k for k, c in enumerate(order)}
        bound = self.named_shelf(line) or min(fits, key=lambda c: (-sum(1 for w in item if within([w], names(c))), rank.get(norm_text(c), len(order)), c), default=None)
        return next((c for c in order if norm_text(c) == norm_text(bound)), bound) if bound else None

    def shelf_terms(self, line: dict) -> tuple:
        counts: dict[str, int] = {}
        for i in line.get("seen") or []:
            cat = (self.store.cards.get(i) or {}).get("category") or ""
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        names = lambda text: [w for w in words(unpossessed(text)) if w not in FUNCTION_WORDS]
        within = lambda part, whole: all(any(variants(w) & variants(x) for x in whole) for w in part)
        return sorted(counts, key=lambda c: (-counts[c], c)), names, within

    def named_shelf(self, line: dict) -> str | None:
        order, names, within = self.shelf_terms(line)
        pool = self.enumerated() or order
        named = names(line.get("category") or "")
        core = [w for w in named if any(within([w], names(c)) for c in pool)]
        same = lambda c, ws: ws and within(names(c), ws) and within(ws, names(c))
        return next((c for c in pool if same(c, named)), None) or next((c for c in pool if same(c, core)), None)

    def file_category(self, n: int, line: dict) -> bool:
        _, names, within = self.shelf_terms(line)
        phrase = " ".join(names(line.get("category") or ""))
        if not phrase or within(phrase.split(), names(line.get("item") or "")) or self.named_shelf(line):
            return False
        if file_want(self.ledger, n, {"key": "type", "op": "is", "values": [phrase], "quote": str(line.get("category")), "source": "category"}, lambda w: w.get("source") == "category"):
            return False
        trace_row(self.trace, "category_want", line=n, phrase=phrase)
        return True

    def listed_shelf(self, line: dict) -> str | None:
        bound = self.shelf(line)
        if bound:
            return bound
        counts: dict[str, int] = {}
        for i in line["seen"]:
            card = self.store.cards.get(i) or {}
            cat = card.get("category") or ""
            text = norm_text(card.get("title", "") + " " + cat)
            if cat and any(f" {v} " in text for w in kind_words(line) for v in variants(w)):
                counts[cat] = counts.get(cat, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def keep(self, line: dict, card: dict) -> bool:
        return self.set_aside(line, card) is None

    def set_aside(self, line: dict, card: dict) -> str | None:
        n = next((k for k, l in enumerate(self.ledger.lines, 1) if l is line), None)
        cap = self.line_cap(line)
        if isinstance(cap, (int, float)) and isinstance(card.get("price"), (int, float)) and card["price"] > cap:
            return f"price {card['price']} over the line's cap {cap}" + ("" if cap == line.get("cap_each") else ", the total cap per unit or the room per unit it leaves after the other lines' cheapest passing cards (a cheaper pass there widens it)")
        if self.ledger.in_stock and card.get("stock") is False:
            return "shown sold out"
        if card["id"] in self.admitted.get(n, {}):
            return None
        text = norm_text(card.get("title", "") + " " + card.get("category", ""))
        bound, shelf = self.shelf(line), self.listed_shelf(line)
        if not any(f" {v} " in text for w in kind_words(line) for v in variants(w)) and not (shelf is not None and norm_text(card.get("category")) == norm_text(shelf)):
            return f"{KIND_RULE} ({', '.join(kind_words(line))}) in its title or shelf"
        if bound and card.get("category") and norm_text(card["category"]) != norm_text(bound):
            return f"on the shelf {card['category']!r}, while the line is bound to {bound!r}"
        return None

    def kind_doubt(self, n: int) -> list[str]:
        line = self.ledger.line(n)
        values = value_words(line, self.ledger.wants(n))
        return [i for i in line["seen"] if str(self.set_aside(line, self.store.cards[i]) or "").startswith(KIND_RULE) and self.shown(n, self.store.cards[i], values)]

    def admit(self, n: int, ident: str, why: str) -> list[dict]:
        line, card = self.ledger.line(n), self.store.cards[ident]
        was = self.set_aside(line, card)
        self.admitted.setdefault(n, {})[ident] = why
        _, names, within = self.shelf_terms(line)
        category = str(line.get("category") or "")
        phrases = [str(line.get("item") or "")] + ([category] if names(category) and not within(names(category), names(line.get("item") or "")) else [])
        made = [{"key": "kind", "op": "is", "values": [p], "quote": p, "source": "admit", "shelf": self.shelf(line)} for p in dict.fromkeys(phrases) if names(p)]
        filed = [w for w in made if file_want(self.ledger, n, w, lambda x, v=norm_text(w["quote"]): x.get("source") == "admit" and v in map(norm_text, x.get("values") or [])) is None]
        trace_row(self.trace, "admit", line=n, id=ident, why=why, was=was, still=self.set_aside(line, card), filed=filed)
        return filed

    def excluded(self, n: int) -> list[dict]:
        line, wants = self.ledger.line(n), self.ledger.wants(n)
        cards, facts, ev = self.store.cards, self.engine.facts, self.engine.evidence
        out: list[dict] = []
        add = lambda field, value, reason, evidence=None: out.append({field: value, "reason": reason, "evidence": evidence})
        order, doubt = set(self.ordered(n)), self.kind_doubt(n)
        for i in sorted(line["seen"], key=lambda i: i not in doubt):
            why = self.set_aside(line, cards[i]) or (None if i in order else "held back behind the rows that print a value the engine saw unindexed, until they are all opened, none is the pick and the shelf listing has no more")
            why = f"{why}; yet its own words show a value of the line, so the kind words may not be the store's" if i in doubt else why
            if why:
                add("id", i, why, {k: cards[i].get(k) for k in ("title", "category", "price", "stock")})
        for k, v in value_words(line, wants):
            s = self.engine.searchable(v)
            if self.engine.indexed(s) is False:
                add("query", s, "left out of the line's searches: its own search came back empty; a search that brings rows holding it brings it back")
            elif s != v:
                add("query", v, f"searched as {s!r}: E4 saw numbers unindexed (a later pair of searches can overturn it)", ev.get("numbers"))
            if wants[k].get("op") in BOUNDS:
                add("query", " ".join(str(x) for x in wants[k].get("values") or []), "a number bound's number never enters a search (it passes many numbers); a search for it reaches rows that print it exactly", wants[k].get("key"))
        ladder = self.ladder(n)
        if value_words(line, wants) and all(kind != "phrase" for _, kind in ladder) and self.phrase(line) and not self.sent(self.phrase(line)):
            add("query", self.phrase(line), "not sent: value searches reached rows of the line (B1); it joins the ladder once they show none")
        for text, cur in ((t, self.cursor_of(t)) for t, _ in ladder):
            if cur and cur["text"] != text:
                add("query", text, f"not sent: {cur['text']!r} differs only in singular and plural and brought {cur['rows']} rows (the index is taken to read both alike)")
        for c in (self.cursors[k] for k in self.asked.get(n, []) if k in self.cursors):
            if c["ended"] or c["stop"]:
                why = f"ended: {c['ended']}" if c["ended"] else f"stopped for the looks: {c['stop']} (the dig may read on)"
                add("query", c["text"] or json.dumps(c["extra"], sort_keys=True), why, {"pages": c["page"], "rows": c["rows"], "count": c["count"]})
        for c in self.mine(n):
            if self.unserved(n, c):
                add("query", c["text"], f"its count ({c['count']}) is above the {self.served(c)} rows the store serves of one search: the dig asks the SEARCH seat for searches in the store's words before its deeper pages (Gather.consult)", {})
        done, ground = self.cover_ground(n)
        add("query", "cover", ("complete: " if done else "incomplete: ") + ground)
        compare = self.store.roles.get("compare") or {}
        if compare.get("facts") is False:
            add("query", compare.get("tool"), "the compare tool opens nothing: it showed no attribute table for a product whose record shows one", compare.get("facts_seen"))
        if facts.get("price_cap") is False:
            add("query", (self.store.roles.get("browse") or {}).get("price_arg"), "no price bands: a capped listing showed a row over its cap (E7); two caps whose counts differ bring them back", ev.get("price_cap"))
        if isinstance(facts.get("first_words"), int):
            add("query", None, f"every search is cut to its {facts['first_words']} most separating words (E5; two searches sharing them with other rows overturn it)", ev.get("first_words"))
        for tool, (first, why) in self.store.first_page.items():
            if first is False:
                add("query", tool, "its default page is not its first: first pages ask for the lowest page", why)
        return out

    def shown(self, n: int, card: dict, values: list) -> set:
        return self.reading(n, card)[0] & {k for k, _ in values}

    def reading(self, n: int, card: dict) -> tuple[set, set]:
        line, wants = self.ledger.line(n), self.ledger.wants(n)
        memo = (n, card.get("id"), card.get("title"), card.get("text"), tuple(want_tag(w) for w in wants))
        if memo in self.memo:
            return self.memo[memo]
        toks = self.row_words(card)
        raws = {r for _, r, _ in toks}
        shown, against, texts = set(), set(), value_words(line, wants)
        for k, want in enumerate(wants):
            op = want.get("op")
            if op in BOUNDS:
                values = [[t for t, _, _ in surface(v)] for v in map(str, want.get("values") or [])]
                unit = next((u for v in values for u in [[t for t in v if not t.replace(".", "").isdigit()]] if u), [])
                limits = [float(t) for v in values for t in v if t.replace(".", "").isdigit()]
                got = [x for _, xs in frames(toks, ["0"] + unit) for x in xs] if unit and limits else []
                meets = lambda x: x >= limits[0] if op == "at_least" else x <= limits[-1] if op == "at_most" else limits[0] <= x <= limits[-1]
                hits = {meets(x) for x in got}
            else:
                hits = set()
                for vt in ([t for t, _, _ in surface(v)] for j, v in texts if j == k and words(v)):
                    own = [float(t) for t in vt if t.replace(".", "").isdigit()]
                    got = {xs == own for _, xs in frames(toks, vt)} | ({True} if len(vt) > 1 and "".join(vt) in raws else set())
                    hits |= got if any(t.isalpha() for t in vt) else got & {True}
            if True in hits:
                shown.add(k)
            elif False in hits:
                against.add(k)
        self.memo[memo] = (shown, against)
        return shown, against

    def forms(self, n: int, value: str) -> list[str]:
        line = self.ledger.line(n)
        vt = [t for t, _, _ in surface(value)]
        own = [float(t) for t in vt if t.replace(".", "").isdigit()]
        memo = (n, value, len(line["seen"]))
        if not vt or memo in self.memo:
            return self.memo.get(memo, [])
        rows = [self.row_words(self.store.cards[i]) for i in line["seen"] if self.keep(line, self.store.cards[i])]
        counts: dict[str, int] = {}
        for toks in rows:
            said = {" ".join(dict.fromkeys(r for _, r, _ in toks[at : at + len(vt)])) for at, xs in frames(toks, vt) if xs == own}
            said |= {r for _, r, _ in toks if len(vt) > 1 and r == "".join(vt)}
            for form in said:
                counts[form] = counts.get(form, 0) + 1
        mine = counts.pop(" ".join(words(value)), 0)
        found = [f for f in sorted(counts, key=lambda f: (-counts[f], f)) if counts[f] >= 2 and counts[f] > mine][:2]
        if not found and not mine:
            heads = {x for h in self.heads(line) for x in variants(h)}
            said = lambda w: sum(1 for toks in rows if any(variants(w) & variants(t) for t, _, _ in toks))
            text = value.lower()
            for joined in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", text):
                parts = {w: said(w) for w in joined.split("-") if w.isalpha() and w not in FUNCTION_WORDS and not variants(w) & heads}
                best = max(parts, key=lambda w: (parts[w], -joined.split("-").index(w)), default=None)
                text = text.replace(joined, best) if best and parts[best] >= 2 else text
            found = [" ".join(words(text))] if text != value.lower() else []
        self.memo[memo] = found
        return found

    def row_words(self, card: dict) -> list[tuple[str, str, int]]:
        memo = (card.get("id"), card.get("title"), card.get("text"))
        if memo not in self.memo:
            self.memo[memo] = surface(f"{card.get('title', '')} {card.get('text', '')}")
        return self.memo[memo]

    def ordered(self, n: int) -> list[str]:
        line = self.ledger.line(n)
        values = value_words(line, self.ledger.wants(n))
        kept = [i for i in line["seen"] if self.keep(line, self.store.cards[i])]
        unindexed = [(k, v) for k, v in values if self.engine.indexed(self.engine.searchable(v)) is False]
        showing = [i for i in kept if self.shown(n, self.store.cards[i], unindexed)]
        hold = bool(showing) and (line.get("pick") in showing or any(not self.store.opened(i) for i in showing) or self.listing_live(n))
        rest = [] if hold else [i for i in kept if i not in showing]
        state = self.bands(n) if self.cheapest(line) else None
        cheap = {i for c in (state or {}).get("found", {}).values() if c["ended"] for i in self.band_rows(n, c, state["pick"])} if state else set()
        cheap |= {i for i in self.cover_rows(n) if isinstance(self.store.cards[i].get("price"), (int, float))}
        rank = lambda i: self.score(n, self.store.cards[i], values, line, i in cheap)
        return sorted(showing, key=rank) + sorted(rest, key=rank)

    def listing_live(self, n: int) -> bool:
        extra = self.listing(n)
        cur = self.cursors.get(self.key("", extra, "browse")) if extra else None
        return bool(extra) and not (cur and cur["ended"])

    def score(self, n: int, card: dict, values: list, line: dict, cheap: bool) -> tuple:
        ident = card["id"]
        best, position = self.hits.get(n, {}).get(ident, 10**6), line["seen"].index(ident)
        member = self.cover.get(n, {}).get(ident, set()) if self.engine.facts.get("or") is False else set()
        shown, against = self.reading(n, card)
        rank = (-len(shown | member), len(against), -len(shown))
        price = price_of(card, math.inf)
        if cheap:
            return (0, *rank, price, best, position)
        if self.cheapest(line):
            return (1, *rank, price, best, position) if self.engine.facts.get("or") is False else (1, *rank, best, price, position)
        return (1, *rank, best, position, price)

    def first_opens(self, n: int, ids: list[str], k: int) -> list[str]:
        line, cover = self.ledger.line(n), self.cover.get(n, {})
        values = value_words(line, self.ledger.wants(n))
        hidden = lambda i: bool(self.engine.facts.get("hidden")) and bool(cover.get(i, set()) - self.shown(n, self.store.cards[i], values))
        family = lambda i: " ".join(w for w in words(self.store.cards[i].get("title", "")) if not any(ch.isdigit() for ch in w))
        ranked = sorted(ids, key=lambda i: not hidden(i))
        out, families = [], set()
        for i in ranked:
            if family(i) not in families:
                out.append(i)
                families.add(family(i))
        return (out + [i for i in ranked if i not in out])[:k]

def key_words(key: str) -> str:
    return norm_text(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(key)))


def key_form(words_: list[str]) -> str:
    out = []
    for w in words_:
        w = one_form(w)
        out.append(w[:-3] + "or" if len(w) >= 5 and w.endswith("our") else w[:-2] + "er" if len(w) >= 5 and w.endswith("re") and w[-3] not in "aeiouy" else w)
    return " ".join(out)


def own_keys(card: dict, want: dict, line: dict) -> list[str]:
    target = key_form(key_words(want.get("key", "")).split())
    items = {v for w in item_words(line) for v in variants(w)}
    found = []
    for key in card.get("facts") or {}:
        kw = key_words(key).split()
        if target and target in (key_form(kw), key_form([w for w in kw if w not in items])):
            found.append(key)
    return found


def spelled(text: Any) -> str:
    return key_form(words(split_joins(unpossessed(text))))


def symbols(text: Any) -> str:
    return "".join(re.findall(r"[^\w\s.\-]", unpossessed(text)))


def same_value(a: Any, b: Any) -> bool:
    return spelled(a) == spelled(b) and symbols(a) == symbols(b)


def unit_number(text: Any) -> tuple[float, str] | None:
    number = r"(?:(?<![\w.])[-+])?\d[\d,]*(?:\.\d+)?"
    found, rest = re.findall(number, str(text)), re.sub(number, " ", str(text))
    return (float(found[0].replace(",", "")), f"{spelled(rest)} {symbols(rest)}".strip()) if len(found) == 1 else None


def unit_proven(want: dict, quote: Any) -> bool:
    said = {x for w in words(split_joins(quote)) for x in variants(w)}
    return all(variants(w) & said for v in want.get("values") or [] for w in words(split_joins(v)) if not re.fullmatch(r"[\d.]+", w))


def holds_value(quote: Any, values: list) -> bool:
    said = {x for w in words(split_joins(quote)) for x in variants(w)}
    content = lambda v: [w for w in words(split_joins(v)) if w not in FUNCTION_WORDS]
    return any(content(v) and all(variants(w) & said for w in content(v)) and all(ch in str(quote) for ch in symbols(v)) for v in map(str, values))


def speaks_of(note: str, want: dict, line: dict) -> bool:
    items = {x for w in item_words(line) for x in variants(w)}
    said = set(words(note))
    for value in map(str, want.get("values") or []):
        content = [w for w in key_words(value).split() if w not in FUNCTION_WORDS and not variants(w) & items]
        if [w for w in content if not w.replace(".", "").isdigit()]:
            if key_named(note, " ".join(content)):
                return True
        elif key_named(note, want.get("key") or "") and all(w in said for w in content):
            return True
    return False


def quoted_keys(card: dict, quote: Any) -> list[str]:
    facts, said = card.get("facts") or {}, norm_text(quote)
    return [k for k, v in facts.items() if norm_text(k) in said or (said in norm_text(f"{k} {v}") and set(said.split()) & set(norm_text(v).split()))] if said.strip() else []


def names_attribute(key: str, want: dict, line: dict, test: Any = all) -> bool:
    items = {key_form([w]) for w in item_words(line)}
    target = [w for w in key_form(key_words(want.get("key") or "").split()).split() if w not in items and w not in FUNCTION_WORDS]
    return not target or test(w in key_form(key_words(key).split()).split() for w in target)


def own_cited(card: dict, want: dict, line: dict, quote: Any) -> bool:
    return want.get("op") == "fits" or any(names_attribute(k, want, line) for k in quoted_keys(card, quote))


def fact_items(value: str) -> list[str]:
    return [p for p in re.split(r"[,;/]", str(value)) if p.strip()]


def code_verdict(card: dict, want: dict, line: dict) -> tuple[str, str, str] | None:
    op, values = want.get("op"), [str(v) for v in want.get("values") or []]
    if want.get("source") in ("category", "admit"):
        shelf, named = str(card.get("category") or ""), values if want["source"] == "category" else [str(want.get("shelf") or "")]
        return ("PASS", "shelf", shelf) if shelf and any(same_value(shelf, v) for v in named) else None
    keys = own_keys(card, want, line)
    said = [fact_items(card["facts"][k]) for k in keys]
    if not values or not keys or any(len(s) != 1 for s in said):
        return None
    quote = "; ".join(f"{k}: {card['facts'][k]}" for k in keys)
    if op in ("is", "one_of", "not"):
        return ("FAIL" if op == "not" else "PASS", "own_key", quote) if all(any(same_value(s[0], v) for v in values) for s in said) else None
    bound, got = [unit_number(v) for v in values], [unit_number(s[0]) for s in said]
    if op not in BOUNDS or None in bound + got or len({u for _, u in bound + got}) != 1 or (op == "between") != (len(bound) == 2):
        return None
    meets = lambda x: x >= bound[0][0] if op == "at_least" else x <= bound[-1][0] if op == "at_most" else bound[0][0] <= x <= bound[1][0]
    met = {meets(x) for x, _ in got}
    return (("PASS" if met.pop() else "FAIL"), "own_key", quote) if len(met) == 1 else None


def store_use(cards: list[dict], want: dict, line: dict) -> str:
    with_key, writing, examples = 0, 0, []
    for card in cards:
        keys = own_keys(card, want, line)
        if keys:
            with_key += 1
            value = str(card["facts"][keys[0]])
            writing += any(same_value(i, v) for i in fact_items(value) for v in want.get("values") or [])
            if value not in examples and len(examples) < 6:
                examples.append(value[:60])
    return f"{with_key} of {len(cards)} opened records have the key; {writing} write the value; values seen: {examples}" + (" (few records)" if len(cards) < 4 else "")


def card_head(card: dict) -> list[str]:
    who = "".join(f" | {name}: {card[key]}" for key, name in (("seller", "seller"), ("brand", "brand")) if card.get(key))
    facts = "; ".join(f"{k}: {v}" for k, v in (card.get("facts") or {}).items())
    return [f"ID {card['id']} | title: {card.get('title')} | shelf: {card.get('category')} | price: {card.get('price')} | in stock: {card.get('stock')}{who}",
            f"  facts: {facts or ('(none read) the record as written: ' + card['raw_text'] if card.get('raw_text') else '(none)')}"]


def card_texts(card: dict) -> list[tuple[str, str]]:
    texts = card.get("texts") or ({"row": card["text"]} if card.get("text") else {})
    rank = lambda label: ("page", "record", "row").index(label.split(".")[0]) if label.split(".")[0] in ("page", "record", "row") else 3
    return [(k, texts[k]) for k in sorted(texts, key=rank) if not (card.get("raw_text") and not card.get("facts") and rank(k) < 2)]


def card_view(card: dict) -> str:
    return "\n".join(card_head(card) + [f"  {label}: {cut_text(text, 800)}" for label, text in card_texts(card)])


def record_view(card: dict) -> str:
    head, fields = card_head(card), card_texts(card)
    room, given = max(1500, 6000 - sum(len(h) for h in head)), {}
    for k, (label, text) in enumerate(sorted(fields, key=lambda f: len(f[1]))):
        given[label] = cut_text(text, max(300, room // (len(fields) - k)))
        room -= len(given[label])
    return "\n".join(head + [f"  {label}: {given[label]}" for label, _ in fields])


class Judge:

    def __init__(self, brain: Brain, store: Store, ledger: Ledger, trace: list) -> None:
        self.brain, self.store, self.ledger, self.trace = brain, store, ledger, trace
        self.filed: dict[tuple, int] = {}
        self.disputes: dict[tuple, str] = {}
        self.renamed: set[tuple] = set()

    def file(self, n: int, ident: Any, k: Any, verdict: tuple, open_pairs: set) -> None:
        wants = self.ledger.wants(n)
        if not (isinstance(ident, str) and isinstance(k, int) and (ident, k) in open_pairs and 1 <= k <= len(wants)):
            trace_row(self.trace, "judge_drop", line=n, id=ident, want=k, verdict=verdict, reason="not a pair that was sent")
            return
        verdicts, tag = self.ledger.line(n)["verdicts"], want_tag(wants[k - 1])
        kept, at = verdicts.get(ident, {}).get(tag), version(self.store.cards[ident])
        due = (n, ident, tag)
        if kept and kept[0] != "UNKNOWN" and self.filed.get(due) == at and due not in self.disputes:
            trace_row(self.trace, "judge_drop", line=n, id=ident, want=k, verdict=verdict, kept=kept, reason="a verdict on this version stands")
            return
        if kept and kept[0] != verdict[0]:
            trace_row(self.trace, "verdict_changed", line=n, id=ident, want=k, was=kept, now=verdict, version=at, dispute=self.disputes.get(due))
        verdicts.setdefault(ident, {})[tag] = verdict
        self.filed[due] = at
        self.disputes.pop(due, None)

    def stale(self, n: int) -> list[str]:
        tags, at = {want_tag(w) for w in self.ledger.wants(n) if w.get("op") != "ref"}, {i: version(self.store.cards[i]) for i in self.ledger.line(n)["verdicts"]}
        return [i for i, got in self.ledger.line(n)["verdicts"].items() if any(self.filed.get((n, i, tag), at[i]) != at[i] for tag in tags & set(got))]

    def dispute(self, n: int, ident: str, note: str) -> None:
        line, card = self.ledger.line(n), self.store.cards[ident]
        named = [w for w in self.ledger.wants(n) if w.get("op") != "ref" and speaks_of(str(note), w, line) and not code_verdict(card, w, line)]
        for want in named:
            self.disputes[(n, ident, want_tag(want))] = str(note)
        trace_row(self.trace, "dispute", line=n, id=ident, note=note, reopened=[want_tag(w) for w in named])
        if named:
            self.judge(n, [ident])

    def judge(self, n: int, ids: list[str]) -> None:
        line, wants = self.ledger.line(n), self.ledger.wants(n)
        verdicts = line["verdicts"]
        shelf = [self.store.cards[i] for i in verdicts if self.store.cards.get(i, {}).get("facts")]
        pending: dict[str, list[int]] = {}
        reused: dict[str, list[int]] = {}
        skipped: dict[str, list[int]] = {}
        for i in dict.fromkeys(ids):
            got, card = verdicts.setdefault(i, {}), self.store.cards[i]
            at = version(card)
            for k, want in enumerate(wants, 1):
                due = (n, i, want_tag(want))
                if want.get("op") == "ref":
                    skipped.setdefault(i, []).append(k)
                elif want_tag(want) in got and self.filed.get(due) == at and due not in self.disputes:
                    reused.setdefault(i, []).append(k)
                elif cv := code_verdict(card, want, line):
                    self.file(n, i, k, cv, {(i, k)})
                else:
                    pending.setdefault(i, []).append(k)
        if pending:
            open_pairs = {(i, k) for i, ks in pending.items() for k in ks}
            source = {"disclosed": " [disclosed by the shopper]", "notice": " [from a store notice]", "chain": " [value read from another listing]"}
            said = lambda w: want_text(w) + source.get(w.get("source"), "")
            reqs = "\n".join(f"{k}. {said(wants[k - 1])}  (STORE USE: {store_use(shelf, wants[k - 1], line)})" for k in sorted({k for ks in pending.values() for k in ks}))
            shown = {i: record_view(self.store.cards[i]) for i in pending}
            raised = lambda i, ks: "".join(f"\n  point raised on requirement {k}: {self.disputes[(n, i, want_tag(wants[k - 1]))]}" for k in ks if (n, i, want_tag(wants[k - 1])) in self.disputes)
            now = "RECORDS:\n" + "\n".join(shown[i] + f"\n  requirements to judge: {', '.join(map(str, ks))}" + raised(i, ks) for i, ks in pending.items())
            answer = self.brain.ask("RECORD", RECORD, f"ITEM: {line['item']}\nREQUIREMENTS:\n{reqs}", now, RECORD_SCHEMA) or {}
            again: dict[tuple, str] = {}
            for v in answer.get("verdicts") or []:
                v = v if isinstance(v, dict) else {}
                i, k, where, quote = str(v.get("id")), v.get("want") if isinstance(v.get("want"), int) else None, str(v.get("where") or "none"), str(v.get("quote") or "")
                label = v.get("label") if v.get("label") in ("PASS", "FAIL") else "UNKNOWN"
                if (i, k) in open_pairs and label != "UNKNOWN":
                    want, card = wants[k - 1], self.store.cards[i]
                    held = label == "FAIL" and want.get("op") in ("is", "one_of") and where in ("own_key", "sibling") and holds_value(quote, want.get("values") or [])
                    why = "the quote is not in the record" if not stands(quote, shown[i]) else "the FAIL's own words hold the required value" if held else None
                    if why:
                        trace_row(self.trace, "verdict_unproven", line=n, id=i, want=k, label=label, where=where, quote=quote, reason=why)
                        label, where = "UNKNOWN", "held" if held and stands(quote, shown[i]) else where
                    elif where == "own_key" and not own_cited(card, want, line, quote):
                        due = (n, i, want_tag(want))
                        if (due, version(card)) not in self.renamed:
                            self.renamed.add((due, version(card)))
                            cited, attr = quoted_keys(card, quote), want.get("key")
                            again[due] = f"the quote cites {cited or 'no fact key'} as the own key of '{attr}', but no cited key is named '{attr}': own_key is only a key named as this attribute; a key holding this same attribute under another name is sibling; a key holding another fact (a related but different quantity or property) proves nothing about this requirement, so judge it from the other evidence, else UNKNOWN"
                        where = "sibling"
                self.file(n, i, k, (label, where, quote, "inferred" if label == "PASS" and v.get("support") == "inferred" else "direct"), open_pairs)
            cut = [i for i in pending if "more characters left out)" in shown[i]]
            if cut or not answer:
                trace_row(self.trace, "judge_short", line=n, cut=cut, answered=bool(answer), reason="a record view was cut (the rest stays unread), or RECORD gave no answer (the pairs keep their verdicts and stay due)")
        trace_row(self.trace, "judge", line=n, versions={i: version(self.store.cards[i]) for i in ids}, asked=pending, reused=reused, skipped=skipped, verdicts={i: verdicts.get(i) for i in ids})
        if pending and again:
            self.disputes.update(again)
            trace_row(self.trace, "own_key_misnamed", line=n, pairs=[list(due[1:]) for due in again], reason="an own-key verdict cites no key named as the attribute: filed as sibling and asked once more with the point")
            self.judge(n, sorted({due[1] for due in again}))

    def label(self, n: int, ident: str, want: dict) -> str:
        if want.get("op") == "ref":
            return "UNKNOWN"
        seen = self.ledger.line(n)["verdicts"].get(ident, {}).get(want_tag(want), ("UNKNOWN",))[0]
        doubt = want.get("conflict") or (isinstance(want.get("when"), dict) and want.get("cond") != "PASS")
        return "UNKNOWN" if seen == "FAIL" and doubt else seen

    def status(self, n: int, ident: str) -> str:
        wants, got = self.ledger.wants(n), self.ledger.line(n)["verdicts"].get(ident, {})
        labels = [self.label(n, ident, w) for w in wants]
        if "FAIL" in labels:
            return "FAIL"
        if all(l == "PASS" for l in labels):
            return "PASS"
        unproven = [w for w, l in zip(wants, labels) if l != "PASS" and got.get(want_tag(w), ("", ""))[1] != "held"]
        others = [j for j in self.ledger.line(n)["verdicts"] if j != ident]
        return "UNKNOWN" if any(self.label(n, j, w) in ("PASS", "FAIL") for w in unproven for j in others) else "WEAK"

    def proven(self, n: int, ident: str) -> int:
        got = self.ledger.line(n)["verdicts"].get(ident, {})
        loose = [w for w in self.unit_gap(n, ident) if got[want_tag(w)][1] != "own_key"] + [w for w in self.ledger.wants(n) if got.get(want_tag(w), ())[3:4] == ("inferred",)]
        return sum(1 for want in self.ledger.wants(n) if self.label(n, ident, want) == "PASS" and want not in loose)

    def loose(self, n: int, ident: str) -> int:
        line, card, got = self.ledger.line(n), self.store.cards[ident], self.ledger.line(n)["verdicts"].get(ident, {})
        return sum(1 for w in self.ledger.wants(n) if (v := got.get(want_tag(w))) and v[:2] == ("PASS", "sibling") and not any(names_attribute(k, w, line, any) for k in quoted_keys(card, v[2])))

    def unit_gap(self, n: int, ident: str) -> list[dict]:
        got = self.ledger.line(n)["verdicts"].get(ident, {})
        return [w for w in self.ledger.wants(n) if w.get("op") in BOUNDS and (v := got.get(want_tag(w))) and v[0] == "PASS" and not unit_proven(w, v[2])]

    def evidence_keys(self, n: int) -> set:
        line = self.ledger.line(n)
        stated = [w for w in self.ledger.wants(n) if w.get("source") == "stated" and w.get("op") not in ("ref", "fits")]
        tags, out = {want_tag(w) for w in stated}, set()
        for i, got in line["verdicts"].items():
            card = self.store.cards.get(i) or {}
            for want in stated:
                out.update(own_keys(card, want, line))
            for tag, verdict in got.items():
                if tag in tags and verdict[0] in ("PASS", "FAIL") and verdict[1] in ("own_key", "sibling"):
                    out.update(quoted_keys(card, verdict[2]))
        return out


def version(card: dict) -> int:
    return hash(json.dumps([card.get(k) for k in ("title", "category", "seller", "brand", "text", "facts", "texts", "raw_text", "small")], sort_keys=True, default=str))


def want_tag(want: dict) -> str:
    return f"{want.get('key')}|{want.get('op')}|{'/'.join(map(str, want.get('values') or []))}"

EVIDENCE_RANK = {"own_key": 0, "sibling": 1, "options": 1, "shelf": 1, "title": 2, "small_print": 3, "description": 3, "none": 4}
TIERS = {"PASS": 0, "WEAK": 1}


def number_of(text: Any) -> float | None:
    found = numbers_in(text)
    return found[0] if found else None


def price_of(card: dict | None, missing: float) -> float:
    price = (card or {}).get("price")
    return price if isinstance(price, (int, float)) else missing


def split_joins(text: Any) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", str(text))


def id_tokens(text: Any) -> set[str]:
    out = set()
    for run in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text).lower()):
        out.update(p for p in {run, *run.split("-")} if len(p) >= 3 and re.search(r"\d", p))
    return out


def same_id(a: str, b: str) -> bool:
    x, y = sorted((norm_text(a), norm_text(b)), key=len)
    if len(x.strip()) >= 3 and re.search(r"\d", x) and re.search(r"[a-z]", x) and x in y:
        return True
    runs = lambda t: re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(t).lower())
    whole = lambda t: {r for r in runs(t) if len(r) >= 3 and re.search(r"\d", r) and re.search(r"[a-z]", r)}
    parts = lambda t: {p for r in runs(t) for p in (r, *r.split("-"))}
    return bool(whole(a) & parts(b) or whole(b) & parts(a))


def clash(a: dict, b: dict) -> bool:
    ka, kb = (key_form(key_words(w.get("key") or "").split()) for w in (a, b))
    if not ka or ka != kb:
        return False
    va, vb = ([str(v) for v in w.get("values") or []] for w in (a, b))
    shared = any(same_value(x, y) for x in va for y in vb)
    ops = {a.get("op"), b.get("op")}
    if ops <= {"is", "one_of"}:
        return bool(va and vb) and not shared
    return "not" in ops and bool(ops & {"is", "one_of"}) and shared


def described_of(parse: dict) -> list[tuple[str, str]]:
    return [(str(d.get("key") or ""), str(d["value"])) for d in parse.get("described") or [] if isinstance(d, dict) and d.get("value")]


def mate(value: str, pairs: list) -> str | None:
    return next((p["to"] for p in pairs if same_value(p.get("from", ""), value)), None) or next((p["from"] for p in pairs if same_value(p.get("to", ""), value)), None)

class Picker:

    def __init__(self, brain: Brain, store: Store, ledger: Ledger, judge: Judge, gather: Gather, trace: list) -> None:
        self.brain, self.store, self.ledger, self.judge, self.gather, self.trace = brain, store, ledger, judge, gather, trace
        self.fitter = Fitter(self)
        self.chain_cursor: dict[str, Any] = {}
        self.hop_rows: dict[str, dict[str, int]] = {}
        self.hop_reads: dict[tuple, dict] = {}
        self.parses: dict[str, dict] = {}
        self.read_tried: set[str] = set()
        self.basket_version = 0

    def strength(self, n: int, ident: str) -> int:
        got = self.ledger.line(n)["verdicts"].get(ident, {})
        return max([EVIDENCE_RANK.get(v[1], 4) for v in got.values() if v[0] == "PASS"] or [4])

    def ranked(self, n: int) -> list[str]:
        line = self.ledger.line(n)
        status = {i: self.judge.status(n, i) for i in line["verdicts"] if self.gather.keep(line, self.store.cards.get(i, {}))}
        price = lambda i: price_of(self.store.cards[i], math.inf)
        if line["objective"] == "best" and line.get("best_key"):
            low = line.get("best_dir") == "min"

            def objective(i: str) -> tuple:
                keys = own_keys(self.store.cards[i], {"key": line["best_key"]}, line)
                value = number_of(self.store.cards[i]["facts"][keys[0]]) if keys else None
                return ((value if low else -value) if value is not None else float("inf"), price(i))

        elif self.gather.cheapest(line):
            objective = lambda i: (price(i), self.strength(n, i))
        else:
            objective = lambda i: (self.strength(n, i), price(i))
        admitted = self.gather.admitted.get(n, {})
        return sorted((i for i, s in status.items() if s in TIERS), key=lambda i: (TIERS[status[i]], i in admitted, -self.judge.proven(n, i), *objective(i), self.judge.loose(n, i), i != line.get("pick")))

    def top_tier(self, n: int, ids: list[str]) -> list[str]:
        tiers = {i: TIERS.get(self.judge.status(n, i), 3) for i in ids}
        return [i for i in ids if tiers[i] == min(tiers.values())] if ids else []

    def fit_total(self) -> None:
        lines = self.ledger.lines
        price = lambda i, missing: price_of(self.store.cards.get(i or ""), missing)
        cap = self.ledger.total_cap if isinstance(self.ledger.total_cap, (int, float)) else float("inf")
        counted = {m for m, l in enumerate(lines, 1) if self.gather.cheapest(l)}
        whole = self.ledger.total_objective == "cheapest" or (len(lines) > 1 and len(counted) == len(lines))
        if not whole and sum(price(l.get("pick"), 0) * l["qty"] for l in lines) <= cap:
            return
        units, bound = [], set()
        for n in range(1, len(lines) + 1):
            if n in bound:
                continue
            grp = self.fitter.group(n)
            members = [grp[0]] + grp[1] if grp else [n]
            bound.update(members)
            options = self.fitter.combos(*grp) if grp else [(i,) for i in self.top_tier(n, self.ranked(n))]
            units.append((members, options or [tuple(lines[m - 1].get("pick") for m in members)]))
        keep = 12
        while keep > 1 and math.prod(min(len(o), keep) for _, o in units) > 50000:
            keep -= 1
        units = [(m, o[:keep]) for m, o in units]
        best = None
        for combo in itertools.product(*(o for _, o in units)):
            picks = {m: i for (members, _), option in zip(units, combo) for m, i in zip(members, option)}
            ids = [i for i in picks.values() if i]
            if len(set(ids)) < len(ids):
                continue
            total = sum(price(i, 0) * lines[m - 1]["qty"] for m, i in picks.items())
            if total > cap:
                continue
            rank = sum(o.index(option) for (members, o), option in zip(units, combo) if any(m not in counted for m in members))
            key = (sum(price(i, math.inf) * lines[m - 1]["qty"] for m, i in picks.items() if m in counted), rank, total)
            if best is None or key < best[0]:
                best = (key, picks)
        if best is None:
            trace_row(self.trace, "basket", result="no combination within the total cap", cap=self.ledger.total_cap, options=[o[:4] for _, o in units])
            return
        for m, pick in best[1].items():
            if pick and pick != lines[m - 1].get("pick"):
                self.choose(m, pick)
        trace_row(self.trace, "basket", picks=best[1], key=best[0], combos=math.prod(len(o) for _, o in units))

    def choose(self, n: int, to: str | None = None, keep: bool = False) -> None:
        line = self.ledger.line(n)
        if to:
            line["pinned"], line["keep"] = to, keep
        self.refresh(n)
        self.condition(n)
        if self.fitter.group(n):
            self.fitter.choose(n)
        else:
            order = self.ranked(n)
            if line.get("pinned") in (order if line.get("keep") else self.top_tier(n, order)):
                order = [line["pinned"]] + [i for i in order if i != line["pinned"]]
            self.place(n, order)
        if to and line.get("pick") != to:
            trace_row(self.trace, "choose_refused", line=n, wanted=to, pick=line.get("pick"), status=self.judge.status(n, to) if to in line["verdicts"] else None, reason=f"the chosen record is not in the line's {'ranking' if keep else 'best tier'} now")

    def place(self, n: int, order: list) -> None:
        line = self.ledger.line(n)
        before = line.get("pick")
        line["pick"], line["standby"], line["candidates"] = (order[0] if order else None), (order[1] if len(order) > 1 else None), list(order[:5])
        if line["pick"] != before:
            self.basket_version += 1
            if before:
                trace_row(self.trace, "replace", line=n, was=before, now=line["pick"], why="out of the order" if before not in order else "outranked", candidates=order[:5], basket=self.basket_version)

    def refresh(self, n: int) -> None:
        stale = self.judge.stale(n)
        if stale:
            self.judge.judge(n, stale)
        for want in (w for w in self.ledger.wants(n) if w.get("trail") and not w.get("chain_at")):
            k = next((k for k, t in enumerate(want["trail"]) if version(self.store.cards[t["page"]]) != t["version"]), None)
            if k is None:
                continue
            hop = want["trail"][k]
            read = self.read_hop(hop["pointer"], hop["attribute"], hop["page"], want.get("relation"))
            same = all(read.get(x) == hop["read"].get(x) for x in ("next", "value", "pairs"))
            trace_row(self.trace, "chain_reread", line=n, page=hop["page"], hop=k + 1, same=same, read=read, reason="the hop page took new text since CHAIN_READ read it")
            if not read:
                continue
            want["trail"] = want["trail"][: k + 1 if same else k] + ([] if same else [dict(hop, read=read)])
            want["trail"][k]["version"] = version(self.store.cards[hop["page"]])
            if same:
                continue
            if read.get("next"):
                want.update(want["ref"], chain_at=read["next"])
            else:
                self.settle_chain(n, want)

    def fit_doubt(self, n: int) -> bool:
        return self.fitter.doubt(n)

    def chain_open(self, n: int) -> bool:
        return any(w.get("op") == "ref" and (not w.get("followed") or w.get("chain_at")) for w in self.ledger.wants(n))

    def open_and_judge(self, n: int, ids: list[str], page: bool = False) -> None:
        fresh = [i for i in ids if i not in self.store.records]
        compare = self.store.roles.get("compare") or {}
        if len(fresh) > 1 and compare.get("tool") and compare.get("ids_arg") and compare.get("facts") is not False:
            most, k = int(compare.get("most") or 1), 0
            while k < len(fresh) and not self.gather.walk_over():
                result = self.store.act(compare["tool"], {compare["ids_arg"]: fresh[k : k + most]})
                rows, _ = self.store.rows_list(result.get("observation"))
                if (result.get("error") or not rows) and most > 1:
                    most = max(1, most // 2)
                    compare["most"] = most
                    trace_row(self.trace, "compare_batch", most=most, error=result.get("error"))
                    continue
                k += most
                for row in rows:
                    card = self.store.card_of(row)
                    if card and card["id"] in self.store.cards and isinstance(row, (dict, str)):
                        self.store.absorb(self.store.cards[card["id"]], row, "facts")
        for i in fresh:
            if not self.store.cards.get(i, {}).get("facts") and not self.gather.walk_over():
                self.store.read_card(i, page)
        self.judge.judge(n, [i for i in ids if self.store.cards.get(i, {}).get("facts") or i in self.store.records])

    def condition(self, n: int) -> None:
        line = self.ledger.line(n)
        plain = [w for w in line["wants"] if not w.get("dropped") and not isinstance(w.get("when"), dict)]
        for want in (w for w in line["wants"] if isinstance(w.get("when"), dict) and w.get("dropped") in (None, False, "condition")):
            against = next((w for w in plain if clash(want, w)), None)
            if against and not want.get("conflict"):
                want["conflict"] = want_tag(against)
                self.ledger.unhandled.append({"kind": "conflict", "line": n, "conditional": want.get("quote"), "stated": against.get("quote"), "why": "a conditional requirement contradicts a stated one on the same attribute"})
                trace_row(self.trace, "condition_clash", line=n, want=want, stated=against, reason="the reading is in doubt: both stand, a FAIL on the conditional one is unproven, and the final check sees both sentences")
            cond = want["when"]
            m = cond.get("line") if isinstance(cond.get("line"), int) and 1 <= cond["line"] <= len(self.ledger.lines) else None
            pick = self.ledger.line(m).get("pick") if m else None
            verdict = code_verdict(self.store.cards[pick], {"key": cond.get("key"), "op": cond.get("op") or "is", "values": [str(v) for v in cond.get("values") or []]}, self.ledger.line(m)) if pick in self.store.cards else None
            want["cond"] = verdict[0] if verdict else "UNKNOWN"
            off = "condition" if verdict and verdict[0] == "FAIL" else False
            if off != (want.get("dropped") or False):
                want["dropped"] = off
                trace_row(self.trace, "condition", line=n, want=want, if_pick=pick, verdict=verdict)
                if not off:
                    self.judge.judge(n, list(line["verdicts"]))

    def follow_chain(self, n: int, want: dict) -> None:
        want["followed"] = True
        want.setdefault("ref", {k: want.get(k) for k in ("op", "values", "source")})
        trail = want.setdefault("trail", [])
        pointer = want.pop("chain_at", None) or str(want.get("quote") or " ".join(want["ref"]["values"] or []))
        while len(trail) < 7:
            parse = self.parse_pointer((want.get("context") or pointer) if not trail else pointer)
            if not parse:
                break
            if trail and described_of(parse) and self.identifies(parse, trail[-1]["page"]):
                hop = trail[-1]
                hop["read"] = dict(hop["read"], next=None, value=hop["read"].get("value") or self.own_value(hop["page"], hop["attribute"]))
                trace_row(self.trace, "chain_self", line=n, page=hop["page"], read=hop["read"], reason="the next pointer only describes the page it stands on: the chain ends on that page's value or pairing rule")
                break
            want["relation"] = "different" if not trail and want.get("context") and parse["relation"] == "different" else want.get("relation") or parse["relation"]
            want["distance"] = want.get("distance") or parse.get("distance")
            attribute = want.get("key") or parse.get("attribute")
            page, read = self.find_page(parse, pointer, attribute, [t["page"] for t in trail], want["relation"])
            if page and page not in self.ledger.chain_pages:
                self.ledger.chain_pages.append(page)
            if read is None:
                if page or self.gather.walk_over():
                    want["chain_at"] = pointer
                break
            trail.append({"pointer": pointer, "attribute": attribute, "page": page, "read": read, "version": version(self.store.cards[page])})
            if not read.get("next"):
                break
            pointer = read["next"]
        else:
            trace_row(self.trace, "chain_cap", line=n, pointer=pointer, hops=[t["page"] for t in trail], reason="seven hops read and the last points on: the chain stays unresolved")
        self.settle_chain(n, want)

    def settle_chain(self, n: int, want: dict) -> None:
        trail, line = want["trail"], self.ledger.line(n)
        hops = [t["page"] for t in trail]
        mine = [h for h in hops if self.gather.keep(line, self.store.cards[h])]
        line["seen"] += [h for h in mine if h not in line["seen"]]
        want["chain"] = hops
        if want.get("chain_at"):
            trace_row(self.trace, "chain_cut", line=n, want=want, hops=hops, left=self.store.env.left(), reason="the walk gate held the next hop back; the chain resumes from chain_at")
        else:
            end = trail[-1] if trail and not trail[-1]["read"].get("next") else None
            value = (self.own_value(end["page"], end["attribute"]) or end["read"].get("value")) if end else None
            pairs = (end["read"].get("pairs") or []) if end else []
            want.update(want["ref"])
            end_req = self.end_want(want.get("relation"), want.get("distance") or next((v for v in want["ref"]["values"] or [] if len(words(v)) <= 3), None), value, pairs)
            if end_req:
                want.update(end_req, source="chain")
            if end_req and (end_req["op"] != "within" or self.self_check(n, want, hops[-1])):
                trace_row(self.trace, "chain", line=n, want=want, hops=hops)
            else:
                want.update(want["ref"])
                trace_row(self.trace, "chain_open", line=n, want=want, hops=hops, value=value, pairs=pairs, reason="no end value, no pair for a partner, or a band in another unit the end page fails: the chain stays ref")
        if mine:
            self.judge.judge(n, mine)

    def self_check(self, n: int, want: dict, page: str) -> bool:
        line = self.ledger.line(n)
        known = page in line["verdicts"]
        self.judge.judge(n, [page])
        label = line["verdicts"].get(page, {}).get(want_tag(want), ("UNKNOWN",))[0]
        if not known and not self.gather.keep(line, self.store.cards[page]):
            line["verdicts"].pop(page, None)
        return label == "PASS"

    def parse_pointer(self, sentence: str) -> dict | None:
        if sentence in self.parses:
            return self.parses[sentence]

        def check(answer: dict) -> str | None:
            texts = [answer.get(k) for k in ("party", "kind", "distance")] + [d.get("value") for d in answer.get("described") or [] if isinstance(d, dict)]
            lost = [t for t in texts if isinstance(t, str) and norm_text(t).strip() and not stands(t, sentence)]
            return f"these are not written in the sentence: {lost[:6]}" if lost else None

        answer = self.brain.ask("CHAIN_PARSE", CHAIN_PARSE, "", "SENTENCE: " + sentence, CHAIN_PARSE_SCHEMA, check)
        trace_row(self.trace, "chain_parse", sentence=sentence, parse=answer)
        if answer and answer.get("relation"):
            self.parses[sentence] = answer
            return answer
        return None

    def find_page(self, parse: dict, pointer: str, attribute: Any, hops: list, relation: Any) -> tuple[str | None, dict | None]:
        party, kind = (unpossessed(parse.get(k) or "").strip() for k in ("party", "kind"))
        described = described_of(parse)
        party_w = norm_text(party)
        shows = lambda c: sum(1 for _, v in described if norm_text(split_joins(v)) in norm_text(split_joins(c.get("title", "") + " " + c.get("text", "") + " " + " ".join(f"{k} {x}" for k, x in (c.get("facts") or {}).items()))))
        seller = lambda c: norm_text(c.get("seller") or "")
        theirs = lambda c: not party_w.strip() or (seller(c).strip() and (party_w in seller(c) or seller(c) in party_w)) or (not seller(c).strip() and party_w in norm_text(c.get("title", "") + " " + c.get("text", "")))
        rows = self.hop_rows.setdefault(pointer, {})
        for i in (i for i, c in self.store.cards.items() if i not in hops and party_w.strip() and seller(c).strip() and theirs(c)):
            rows.setdefault(i, len(rows))
        tried: list = []
        fallback: tuple | None = None
        kinds = content(kind)

        def done(page: str | None, read: dict | None, how: str) -> tuple[str | None, dict | None]:
            trace_row(self.trace, "chain_find", party=party, product=kind, values=described, rows=len(rows), hop=page, how=how, weighed=tried[:12], unread=sum(1 for i in rows if i not in tried and i not in hops))
            return page, read

        rungs = self.ladder(party, kind, described)
        for at, text in enumerate([None] + rungs):
            if text is not None and self.chain_cursor.get(text, "first") is not None:
                if self.gather.walk_over():
                    return done(None, None, "held: the walk gate holds the next rung back")
                for i in self.chain_rows(text, theirs, shows, len(described), hops):
                    rows.setdefault(i, len(rows))
            if not described and len(listings := [i for i in sorted(rows, key=rows.get) if self.identifies(parse, i)]) > 1:
                trace_row(self.trace, "chain_find", party=party, product=kind, values=[], rows=len(rows), listings=listings[:12], reason="no described value, two or more listings of the kind: a page's gift decides")
                return self.read_listings(listings, pointer, attribute, relation)
            later = lambda i: at < len(rungs) and bool(kinds) and not content(self.store.cards[i].get("title", "") + " " + self.store.cards[i].get("category", "")) & kinds
            while left := [i for i in rows if i not in tried and i not in hops and not later(i)]:
                i = max(left, key=lambda i: self.likely(parse, i, rows[i], shows))
                card = self.store.cards[i]
                if not (card.get("facts") or self.store.opened(i)) and i not in self.read_tried:
                    if self.gather.walk_over():
                        return done(None, None, "held: the walk gate holds the next record read back")
                    self.read_tried.add(i)
                    self.store.read_card(i)
                    continue
                tried.append(i)
                if False in [self.holds_described(card, k, v) for k, v in described]:
                    continue
                named = self.identifies(parse, i)
                if named and all(own_keys(card, {"key": k}, {"item": ""}) for k, _ in described):
                    read = self.hop_read(i, pointer, attribute, relation)
                    return done(i, read, "settled: every described value holds on its own key")
                read = self.hop_read(i, pointer, attribute, relation if named else None)
                if read is None:
                    return done(None, None, "held: the walk gate holds the page or its reading back")
                if named or read.get("described") is True:
                    if self.gives(i, read, attribute, relation):
                        return done(i, read, "identified by code" if named else "CHAIN_READ reads it as the described listing")
                    rank = (bool(read.get("value") or read.get("pairs")), named)
                    fallback = fallback if fallback and fallback[0] >= rank else (rank, i, read)
        return done(fallback[1], fallback[2], "fallback: no candidate gives more") if fallback else done(None, None, "nothing found")

    def likely(self, parse: dict, ident: str, at: int, shows) -> tuple:
        card, described = self.store.cards[ident], described_of(parse)
        read = bool(card.get("facts")) or self.store.opened(ident)
        holds = [self.holds_described(card, k, v) for k, v in described] if read else []
        keyed = sum(1 for (k, _), h in zip(described, holds) if h is True and own_keys(card, {"key": k}, {"item": ""}))
        written = set(words(split_joins(record_view(card) if read else card.get("title", "") + " " + card.get("text", ""))))
        said = sum(1 for _, v in described for w in words(split_joins(v)) if w not in FUNCTION_WORDS and variants(w) & written)
        return (self.of_kind(parse, card), sum(h is True for h in holds) if read else shows(card), keyed, said, read, -at)

    def hop_read(self, ident: str, pointer: str, attribute: Any, relation: Any) -> dict | None:
        if (ident not in self.store.pages and (self.store.roles.get("page") or {}).get("tool") and self.gather.walk_over()) or self.gather.plan.walk_late():
            return None
        self.open_page(ident)
        return self.read_hop(pointer, attribute, ident, relation)

    def gives(self, ident: str, read: dict, attribute: Any, relation: Any) -> bool:
        if self.forward(read, attribute):
            return True
        value = self.own_value(ident, attribute) or read.get("value")
        return bool(value) and (relation != "partner" or mate(str(value), read.get("pairs") or []) is not None)

    def own_value(self, ident: str, attribute: Any) -> str | None:
        card = self.store.cards[ident]
        own = [str(card["facts"][k]) for k in own_keys(card, {"key": attribute or ""}, {"item": ""})]
        return own[0] if own and all(len(fact_items(v)) == 1 and same_value(v, own[0]) for v in own) else None

    def read_listings(self, listings: list[str], pointer: str, attribute: Any, relation: Any) -> tuple[str | None, dict | None]:
        first = None
        for i in listings[:5]:
            if self.gather.walk_over():
                return first or (i, None)
            self.open_page(i)
            read = self.read_hop(pointer, attribute, i, relation)
            gives = self.forward(read, attribute) or bool(read.get("pairs"))
            if not (gives or read.get("value")):
                continue
            if first is None and gives:
                return i, read
            if first is None:
                first = (i, read)
                continue
            if gives or norm_text(read["value"]) != norm_text(first[1]["value"]):
                trace_row(self.trace, "chain_ambiguous", pages=[first[0], i], reads=[first[1], read], reason="two listings of the party give the attribute different things and the pointer describes neither: the hop stays unresolved")
                return None, None
        if len(listings) > 5 and first is None:
            trace_row(self.trace, "chain_find_cut", listings=len(listings), read=listings[:5], reason="five listings read without one giving the attribute; the rest stay unread")
        return first if first else (None, None)

    def forward(self, read: dict, attribute: Any) -> bool:
        return isinstance(read.get("next"), str) and bool((set(words(attribute or "")) - FUNCTION_WORDS) & set(words(read["next"])))

    def of_kind(self, parse: dict, card: dict) -> bool:
        kinds = [w for w in words(unpossessed(parse.get("kind") or "")) if w not in FUNCTION_WORDS]
        shelf = content(card.get("category") or "")
        return all(variants(w) & set(words(card.get("title", "") + " " + card.get("category", ""))) for w in kinds) or (bool(shelf) and shelf <= content(parse.get("kind") or ""))

    def identifies(self, parse: dict, ident: str) -> bool:
        card, party = self.store.cards[ident], unpossessed(parse.get("party") or "").strip()
        named = not norm_text(party).strip() or (bool((card.get("seller") or "").strip()) and content(card.get("seller")) == content(party))
        return named and self.of_kind(parse, card) and all(self.holds_described(card, k, v) is True for k, v in described_of(parse))

    def read_hop(self, pointer: str, attribute: Any, page: str, relation: Any = None) -> dict:
        key = (pointer, str(attribute), page, version(self.store.cards[page]))
        if key in self.hop_reads:
            return self.hop_reads[key]
        view = record_view(self.store.cards[page])
        texts = lambda a: [(k, a.get(k)) for k in ("next", "value", "quote")] + [(f"pair {p.get('from')!r} -> {p.get('to')!r}", f"{p.get('from')}\n{p.get('to')}") for p in a.get("pairs") or [] if isinstance(p, dict)]
        lost = lambda a: [k for k, t in texts(a) if isinstance(t, str) and norm_text(t).strip() and not stands(t, view)]
        last: list = []
        asked: list = []

        def check(answer: dict) -> str | None:
            last[:] = [answer]
            if lost(answer):
                return f"these are not written on the PAGE: {lost(answer)}"
            if relation == "partner" and answer.get("value") and not answer.get("next") and not answer.get("pairs") and not asked:
                asked.append(True)
                return "the shopper's value is the partner of this page's value and pairs is empty: copy every pair of the rule the PAGE states for the ATTRIBUTE, or answer pairs [] again when it states none"
            return None

        answer = self.brain.ask("CHAIN_READ", CHAIN_READ, f"POINTER: {pointer}\nATTRIBUTE: {attribute}", "PAGE:\n" + view, CHAIN_READ_SCHEMA, check) or (last[0] if last else {})
        cut = lost(answer)
        read = {k: (None if k in cut else v) for k, v in answer.items() if k != "pairs"}
        if answer:
            read["pairs"] = [p for p in answer.get("pairs") or [] if isinstance(p, dict) and all(isinstance(p.get(x), str) and stands(p[x], view) for x in ("from", "to"))]
            self.hop_reads[key] = read
        trace_row(self.trace, "chain_read", id=page, read=read, cut=cut, asked_rule=bool(asked))
        return read

    def ladder(self, party: str, kind: str, described: list[tuple[str, str]]) -> list[str]:
        limit = self.store.query_limit()
        values = [v for _, v in described]
        bare = [" ".join(w for w in words(v) if not variants(w) & set(words(k))) or v for k, v in described]
        drops = [[b for j, b in enumerate(bare) if j != k] for k in sorted(range(len(bare)), key=lambda k: not re.search(r"\d", values[k]))] if len(bare) >= 2 else []
        if party:
            rungs = [[party] + values + [kind], [party] + bare + [kind]] + [[party] + d + [kind] for d in drops] + [[party, kind], [party], values + [kind], bare + [kind]]
        else:
            rungs = [values + [kind], bare + [kind]] + [d + [kind] for d in drops] + [[kind]]
        return [t for t in dict.fromkeys(clip(" ".join(p for p in parts if p), limit) for parts in rungs) if t]

    def chain_rows(self, text: str, theirs, shows, wanted: int, hops: list) -> dict[str, tuple]:
        found: dict[str, tuple] = {}
        more = self.gather.plan.rounds("Picker.chain_rows")
        while not self.gather.walk_over() and more(text, self.chain_cursor.get(text, "first")):
            token = self.chain_cursor.get(text, "first")
            if token is None:
                break
            result = self.store.search(text, None if token == "first" else token)
            self.chain_cursor[text] = result["next"]
            mine = [c for c in result["cards"] if c["id"] not in hops and theirs(c)]
            found.update({c["id"]: (shows(c), bool((c.get("seller") or "").strip())) for c in mine})
            if not result["next"] or not mine or any(m[0] == wanted and m[1] for m in found.values()):
                break
        return found

    def open_page(self, ident: str) -> None:
        spec = self.store.roles.get("page") or {}
        if ident in self.store.pages or not spec.get("tool"):
            if ident not in self.store.records:
                self.store.read_card(ident, True)
            return
        result = self.store.act(spec["tool"], {spec.get("id_arg") or "id": ident})
        if not result.get("error") and result.get("observation") is not None:
            self.store.absorb(self.store.cards[ident], result["observation"], "page")

    def holds_described(self, card: dict, key: str, value: str) -> bool | None:
        bare = " ".join(w for w in words(value) if not variants(w) & set(words(key))) or value
        facts = [str(card["facts"][k]) for k in own_keys(card, {"key": key}, {"item": ""})]
        if facts:
            nums, wanted = [unit_number(f) for f in facts], [x for x in (unit_number(value), unit_number(bare)) if x]
            if any(same_value(f, x) for f in facts for x in (value, bare)) or any(x in wanted for x in nums):
                return True
            return False if wanted and any(x and any(x[1] == w[1] for w in wanted) for x in nums) else None
        text, phrase = norm_text(split_joins(record_view(card))), norm_text(split_joins(value))
        hits = [m.start() for m in re.finditer(f"(?={re.escape(phrase)})", text)] if phrase.strip() else []
        if not hits:
            return None
        return any(not set(text[:at].split()[-2:]) & {"not", "no", "non", "without", "never", "nor"} for at in hits)

    def end_want(self, relation: Any, distance: Any, value: Any, pairs: list) -> dict | None:
        if not value:
            return None
        value = str(value)
        if relation in ("partner", "same"):
            paired = mate(value, pairs)
            if paired or relation == "partner":
                return {"op": "is", "values": [paired]} if paired else None
        if relation == "within":
            v, d = number_of(value), number_of(distance)
            unit_of = lambda t: re.sub(r"[\d.,\s]+", " ", str(t)).strip().lower()
            if v is None or d is None:
                return None
            if unit_of(distance) and unit_of(distance) != unit_of(value):
                return {"op": "within", "values": [value, str(distance)]}
            unit = unit_of(value)
            return {"op": "between", "values": [f"{round(v - d, 6):g} {unit}".strip(), f"{round(v + d, 6):g} {unit}".strip()]}
        op = {"same": "is", "different": "not", "at_least": "at_least", "at_most": "at_most"}.get(relation, "is")
        return {"op": op, "values": [value]}


class Fitter:

    def __init__(self, picker: Picker) -> None:
        self.p = picker
        self.pairs: dict[tuple[str, str], tuple[str, str]] = {}
        self.pair_at: dict[tuple[str, str], tuple[int, int]] = {}
        self.spans: dict[str, dict[str, list[str]]] = {}
        self.span_at: dict[str, int] = {}
        self.anchors: dict[int, list[str]] = {}
        self.within: dict[int, bool] = {}
        self.sent: set = set()
        self.seen: dict[int, tuple] = {}
        self.confirmed: dict[int, tuple] = {}
        self.last: tuple | None = None

    def group(self, n: int) -> tuple[int, list[int]] | None:
        fits = getattr(self.p.ledger, "fits", None) or []
        main = next((f["with"] for f in fits if f["line"] == n), n if any(f["with"] == n for f in fits) else None)
        return (main, sorted({f["line"] for f in fits if f["with"] == main})) if main else None

    def joined(self, main: str | None, part: str) -> str | None:
        a, b = self.spans.get(main or ""), self.spans.get(part)
        if not a or not b:
            return None
        return next((y for x in a["fits"] for y in b["own"] if same_id(x, y)), None) or next((x for x in b["fits"] for y in a["own"] if same_id(x, y)), None)

    def fit(self, main: str | None, part: str) -> str:
        label = self.pairs.get((main, part), ("UNKNOWN", ""))[0]
        return label if label == "FAIL" else "PASS" if label == "PASS" or self.joined(main, part) else "UNKNOWN"

    def marks(self, ident: str) -> set[str]:
        card, spans = self.p.store.cards.get(ident) or {}, self.spans.get(ident) or {}
        return id_tokens(" ".join(spans.get("own", []) + spans.get("fits", []) + [card.get("title", ""), card.get("text", "")]))

    def drive(self, main: int, deps: list[int]) -> tuple[dict[int, set], set]:
        g, opened = self.p.gather, self.p.store.opened
        shown = {k: [self.marks(i) for i in g.ordered(k)] for k in [main] + deps}
        rows = [m for k in shown for m in shown[k]]
        counts: dict[str, int] = {}
        for m in rows:
            for t in m:
                counts[t] = counts.get(t, 0) + 1
        noise = id_tokens(self.p.ledger.request) | {t for t, c in counts.items() if c * 2 > len(rows)}
        likely = {a: set().union(*[self.marks(i) for i in self.p.ranked(a)[:8] + [i for i in g.ordered(a) if not opened(i)][:8]]) for a in deps}
        toks = {a: (likely[a] - noise) or likely[a] for a in deps}
        lit = [a for a in deps if toks[a]]
        if not lit:
            return toks, set()
        scarce = min(lit, key=lambda a: (len(toks[a]), a))
        drive = toks[scarce] & set().union(*[m for k in shown if k != scarce for m in shown[k]])
        for a in lit:
            drive = drive & toks[a] if drive & toks[a] else drive
        return toks, drive

    def due(self, main: str, part: str) -> bool:
        return self.pair_at.get((main, part)) != (version(self.p.store.cards[main]), version(self.p.store.cards[part]))

    def judge(self, main: str, parts: list[str]) -> None:
        todo = [i for i in dict.fromkeys(parts) if self.due(main, i)]
        views = {i: record_view(self.p.store.cards[i]) for i in [main] + todo}
        at = {i: version(self.p.store.cards[i]) for i in views}
        for k in range(0, len(todo), 10):
            chunk = todo[k : k + 10]
            answer = self.p.brain.ask("FIT", FIT, "MAIN:\n" + views[main], "PARTS:\n" + "\n\n".join(views[i] for i in chunk), FIT_SCHEMA) or {}
            for r in answer.get("records") or []:
                i = str(r.get("id")) if isinstance(r, dict) else ""
                if i in views:
                    if self.span_at.get(i) != at[i]:
                        self.spans[i], self.span_at[i] = {"own": [], "fits": []}, at[i]
                    have = self.spans[i]
                    for side in ("own", "fits"):
                        have[side] += [x for x in r.get(side) or [] if isinstance(x, str) and stands(x, views[i]) and x not in have[side]]
            got = {str(v.get("id")): v for v in answer.get("verdicts") or [] if isinstance(v, dict)}
            for i in chunk:
                v = got.get(i, {})
                label, quote = v.get("label") if v.get("label") in ("PASS", "FAIL") else "UNKNOWN", str(v.get("quote") or "")
                if label != "UNKNOWN" and not stands(quote, views[i] + "\n" + views[main]):
                    label = "UNKNOWN"
                self.pairs[(main, i)], self.pair_at[(main, i)] = (label, quote), (at[main], at[i])
            trace_row(self.p.trace, "fit", main=main, spans={i: self.spans.get(i) for i in [main] + chunk}, pairs={i: (self.fit(main, i), self.pairs[(main, i)]) for i in chunk})

    def rank(self, main: int, deps: list[int]) -> tuple[list[str], dict[int, list[str]], list[str]]:
        mains, parts = self.p.ranked(main), {a: self.p.ranked(a) for a in deps}
        pick = self.p.ledger.line(main).get("pick")
        stale = [i for (m, i) in self.pairs if m == pick and self.due(m, i)]
        if stale:
            self.judge(pick, stale)
        toks, drive = self.drive(main, deps)
        score = lambda m: (sum(1 for a in deps if self.marks(m) & toks[a]), bool(self.marks(m) & drive))
        mains = sorted(mains, key=lambda m: (tuple(-x for x in score(m)), mains.index(m)))
        state = (len(self.p.store.records), len(self.p.store.pages))
        covered = lambda m: sum(1 for a in deps if any(self.fit(m, i) == "PASS" for i in parts[a]))
        anchored = any(covered(m) == len(deps) and self.cost(main, deps, m, parts) <= self.cap() for m in mains)
        for m in sorted(mains, key=lambda m: (-covered(m), mains.index(m))) if self.confirmed.get(main) != state and not anchored else []:
            near = [i for a in deps for i in [i for i in sorted(parts[a], key=lambda i: not self.marks(i) & self.marks(m)) if self.due(m, i)][: max(1, 10 // len(deps))]]
            if near:
                self.judge(m, near)
                break
        self.confirmed[main] = state
        anchors = [m for m in mains if covered(m) == len(deps)]
        return mains, parts, sorted(anchors, key=lambda m: self.cost(main, deps, m, parts) > self.cap())

    def cap(self) -> float:
        return self.p.ledger.total_cap if isinstance(self.p.ledger.total_cap, (int, float)) else float("inf")

    def cost(self, main: int, deps: list[int], m: str, parts: dict[int, list[str]], counted: list[int] | None = None) -> float:
        ledger, missing = self.p.ledger, 0 if counted is None else math.inf
        price, lines = lambda i: price_of(self.p.store.cards.get(i), missing), [main] + deps if counted is None else counted
        rest = sum(price(l["pick"]) * l["qty"] for k, l in enumerate(ledger.lines, 1) if k not in [main] + deps and l.get("pick")) if counted is None else 0
        own = price(m) * ledger.line(main)["qty"] if main in lines else 0
        return own + sum(min([price(i) for i in parts[a] if self.fit(m, i) == "PASS"] or [math.inf]) * ledger.line(a)["qty"] for a in deps if a in lines) + rest

    def choose(self, n: int) -> None:
        main, deps = self.group(n)
        mains, parts, anchors = self.rank(main, deps)
        within = lambda: any(self.cost(main, deps, m, parts) <= self.cap() for m in anchors)
        if not within() and all(self.p.ledger.line(k)["seen"] for k in [main] + deps) and self.reach(main, deps, mains):
            mains, parts, anchors = self.rank(main, deps)
        ledger = self.p.ledger
        fitting = lambda m, a: [i for i in parts[a] if self.fit(m, i) == "PASS"]
        counted = [k for k in [main] + deps if self.p.gather.cheapest(ledger.line(k))]
        anchors.sort(key=lambda m: (self.cost(main, deps, m, parts) > self.cap(), self.cost(main, deps, m, parts, counted) if counted else 0, -sum(len(fitting(m, a)) for a in deps), mains.index(m)))
        self.anchors[main], self.within[main] = anchors, within()
        best = anchors[0] if anchors else (max(mains, key=lambda m: (sum(1 for a in deps if fitting(m, a)), -mains.index(m))) if mains else None)
        if ledger.line(main).get("pinned") in (mains if ledger.line(main).get("keep") else anchors):
            best = ledger.line(main)["pinned"]
        self.p.place(main, ([best] if best else []) + [m for m in anchors if m != best])
        for a in deps:
            line = ledger.line(a)
            order = (fitting(best, a) or [i for i in parts[a] if self.fit(best, i) == "UNKNOWN"]) if best else parts[a]
            if line.get("pinned") in order or (line.get("keep") and line.get("pinned") in parts[a] and self.fit(best, line["pinned"]) != "FAIL"):
                order = [line["pinned"]] + [i for i in order if i != line["pinned"]]
            self.p.place(a, order)
        self.note(main)

    def note(self, main: int) -> None:
        ledger, deps = self.p.ledger, self.group(main)[1]
        best = ledger.line(main).get("pick")
        picks = {a: ledger.line(a).get("pick") for a in deps}
        chosen = [{"line": a, "pick": i, "main": best, "fit": (self.fit(best, i), self.joined(best, i) or self.pairs.get((best, i), ("", ""))[1]) if i else None} for a, i in picks.items()]
        for f in ledger.fits:
            f["pair"] = next((c for c in chosen if c["line"] == f["line"]), None)
        anchors, held = self.anchors.get(main) or [], best in (self.anchors.get(main) or []) and self.within.get(main)
        if (best, chosen) != self.last:
            self.last = (best, chosen)
            trace_row(self.p.trace, "fit_choose" if held else "fit_open", main_line=main, anchor=best, anchors=anchors[:4], chosen=chosen, reason=None if held else "no main has a fitting part on every dependent line within the total cap")

    def reach(self, main: int, deps: list[int], mains: list[str]) -> bool:
        p, g = self.p, self.p.gather
        state = (len(p.store.records), len(p.store.pages), sum(len(p.ledger.line(k)["seen"]) for k in [main] + deps))
        if self.seen.get(main) == state or g.walk_over():
            return False
        self.seen[main] = state
        opened, before = p.store.opened, len(p.store.records)
        toks, drive = self.drive(main, deps)
        rows = [i for i in g.ordered(main) if not opened(i) and self.marks(i) & drive][:3]
        if rows:
            p.open_and_judge(main, rows)
        elif drive:
            self.search(main, sorted(drive, key=lambda t: (-sum(1 for a in deps if t in toks[a]), t))[0], "main")
        fitting = lambda m, a: any(self.fit(m, i) == "PASS" for i in self.p.ranked(a))
        leading = sorted(mains, key=lambda m: (-sum(1 for a in deps if fitting(m, a)), mains.index(m)))[:3]
        for a in deps:
            for m in (m for m in leading if not fitting(m, a)):
                rows = [i for i in g.ordered(a) if not opened(i) and self.marks(i) & self.marks(m)][:3]
                if rows:
                    p.open_and_judge(a, rows)
                    break
                if self.search(a, self.code(m, a), "part"):
                    break
        return len(p.store.records) > before

    def code(self, main: str, n: int) -> str:
        spans = self.spans.get(main) or {"own": [], "fits": []}
        carried = lambda t: sum(1 for i in self.p.ledger.line(n)["seen"] if t in self.marks(i))
        codes = [c for c in dict.fromkeys(spans["fits"] + spans["own"]) if id_tokens(c)] or sorted(self.marks(main), key=lambda t: (-carried(t), t))
        return max(codes, key=lambda c: (max([carried(t) for t in id_tokens(c)] or [0]), -codes.index(c))) if codes else ""

    def search(self, n: int, code: str, direction: str) -> bool:
        p, g, line = self.p, self.p.gather, self.p.ledger.line(n)
        values = value_words(line, p.ledger.wants(n))
        text = " ".join(dict.fromkeys([code] + [v for _, v in values[:1]] + [g.head(line)])).strip()
        if not code or (n, norm_text(code)) in self.sent or g.walk_over():
            return False
        result = p.store.search(text)
        self.sent.add((n, norm_text(code)))
        line["seen"] += [c["id"] for c in result["cards"] if c["id"] not in line["seen"]]
        shown = [c["id"] for c in result["cards"] if norm_text(code) in norm_text(c.get("title", "") + " " + c.get("text", "")) and g.keep(line, p.store.cards[c["id"]])]
        trace_row(p.trace, "fit_search", line=n, text=text, direction=direction, rows=len(result["cards"]), shown=shown[:6])
        fresh = [i for i in shown if i not in p.store.records][:3]
        if fresh:
            p.open_and_judge(n, fresh)
        return True

    def combos(self, main: int, deps: list[int]) -> list[tuple]:
        anchors = self.p.top_tier(main, self.anchors.get(main) or [])
        out = []
        for m in anchors[:6]:
            per = [self.p.top_tier(a, [i for i in self.p.ranked(a) if self.fit(m, i) == "PASS"])[:3] for a in deps]
            out += [(m, *parts) for parts in itertools.product(*per)]
        return out

    def doubt(self, n: int) -> bool:
        grp = self.group(n)
        if not grp:
            return False
        main = grp[0]
        if n == main:
            return not self.within.get(main)
        pick = self.p.ledger.line(n).get("pick")
        return bool(pick) and self.fit(self.p.ledger.line(main).get("pick"), pick) != "PASS"

def same_key(a: str, b: str) -> bool:
    canon = lambda k: sorted(one_form(w) for w in key_words(k).split())
    return bool(key_words(a).strip()) and canon(a) == canon(b)


def key_named(reply: str, key: str) -> bool:
    said = set(words(reply))
    content = [w for w in key_words(key).split() if w not in FUNCTION_WORDS and not w.isdigit()]
    return bool(content) and all(variants(w) & said for w in content)


def file_want(ledger: Ledger, n: int, want: dict, alike=lambda w: False) -> dict | None:
    same = next((w for w in ledger.line(n)["wants"] if alike(w)), None)
    if same is None:
        ledger.line(n)["wants"].append(want)
    return same


def change_ledger(ledger: Ledger, change: dict, said: str, source: str, trace: list) -> set[int]:
    kind, mark = change.get("change"), hash(said)
    line = change.get("line")
    n = line if isinstance(line, int) and not isinstance(line, bool) and 1 <= line <= len(ledger.lines) else None
    quote, values = str(change.get("quote") or ""), [str(v) for v in change.get("values") or []]
    grounded = stands(quote, said)
    pointer = change.get("op") == "ref"
    if kind == "add" and n and change.get("key") is not None and (grounded if pointer else values and (grounded or all(stands(v, said) for v in values))):
        want = {"key": str(change["key"]), "op": change.get("op") or "is", "values": values, "quote": quote, "relation": change.get("relation"), "source": source, "added_in": mark}
        if pointer:
            want.update(followed=False, context=pointer_context(quote, said, want["key"]))
        alike = lambda w: "ref" in (w["op"], (w.get("ref") or {}).get("op")) if pointer else w["op"] == want["op"] and sorted(map(norm_text, w.get("values") or [])) == sorted(map(norm_text, values))
        same = file_want(ledger, n, want, lambda w: not w.get("dropped") and same_key(w.get("key") or "", want["key"]) and alike(w))
        trace_row(trace, "want_added", line=n, want=want, source=source, repeat=same is not None)
        return set() if same else {n}
    if kind == "drop":
        listed = [w for w in ledger.line(n)["wants"] if w.get("added_in") != mark and (not w.get("dropped") or w.get("dropped_in") == mark)] if n else []
        k = change.get("want")
        target = listed[k - 1] if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= len(listed) else next((w for w in listed if change.get("key") and w.get("op") != "ref" and same_key(w.get("key") or "", str(change["key"]))), None)
        found, spans = quote_span(quote, said)
        sentence = " ".join(said[a:b] for a, b in spans if found and a < found.end() and b > found.start())
        words_of = lambda parts: set().union(set(), *(content(t, False) for t in parts))
        shared = words_of([w.get("key") or "" for w in ledger.line(n)["wants"] if target is not None and w is not target and not w.get("dropped") and not same_key(w.get("key") or "", target.get("key") or "")]) if n else set()
        key = words_of([(target or {}).get("key") or ""])
        attr, vals = (key - shared) or key, words_of((target or {}).get("values") or [])
        named = target is not None and bool(content(quote, False) & (attr | vals) or content(sentence, False) & attr and not content(sentence, False) & vals)
        if grounded and named and not target.get("dropped"):
            own = content(target.get("quote") or "", False)
            twins = [w for w in ledger.line(n)["wants"] if w is not target and not w.get("dropped") and (same_key(w.get("key") or "", target.get("key") or "") and w.get("op") != "ref" or own and (
                     norm_text(w.get("quote") or "") == norm_text(target["quote"]) or w.get("source") == "category" and content(" ".join(w.get("values") or []), False) and content(" ".join(w.get("values") or []), False) <= own))]
            for w in [target] + twins:
                w.update(dropped=source, dropped_in=mark)
                trace_row(trace, "want_dropped", line=n, want=w, quote=quote, source=source, twin_of=None if w is target else target.get("key"))
            return {n}
        why = "no line and number of this reading" if target is None else "already given up" if target.get("dropped") else "its quote does not stand in the text" if not grounded else "its sentence does not name it"
        trace_row(trace, "drop_ambiguous", change=change, want=target, source=source, reason=why + ": the requirement stays")
        return set()
    if kind in ("cap", "cap_each") and isinstance(change.get("value"), (int, float)) and not isinstance(change.get("value"), bool) and cap_written(change["value"], said) and (kind == "cap" or n):
        if kind == "cap":
            ledger.total_cap = change["value"]
            trace_row(trace, "cap_changed", cap=change["value"], quote=quote, source=source)
            return set(range(1, len(ledger.lines) + 1))
        ledger.line(n)["cap_each"] = change["value"]
        trace_row(trace, "cap_changed", line=n, cap=change["value"], quote=quote, source=source)
        return {n}
    trace_row(trace, "change_refused", change=change, source=source, reason="its quote, values, number or line does not stand in the text it was read from")
    return set()


def cap_problem(answer: dict, said: str) -> str | None:
    wrong = [c.get("value") for c in answer.get("changes") or [] if isinstance(c, dict) and c.get("change") in ("cap", "cap_each") and isinstance(c.get("value"), (int, float)) and not cap_written(c["value"], said)]
    return f"the cap value {wrong[0]} is no number the text writes: give the amount exactly as the text writes it, in its own unit" if wrong else None


class Asker:

    def __init__(self, brain: Brain, store: Store, ledger: Ledger, judge: Judge, plan: StepPlan, gate: FinishGate, trace: list) -> None:
        self.brain, self.store, self.ledger, self.judge, self.plan, self.gate, self.trace = brain, store, ledger, judge, plan, gate, trace
        self.queue: dict[int, list[str]] = {}
        self.pointed: dict[int, list[str]] = {}
        self.form: dict = {"together": False, "count": None, "required": {}}
        self.questions = 0
        self.sent: dict[tuple, list[str]] = {}
        self.extra: set = set()
        self.talk: list[dict] = []
        self.width = 3
        self.deferred: set = set()
        self.planned = False

    def table(self, n: int) -> list[tuple[str, int, int, list[str]]]:
        line = self.ledger.line(n)
        cited = self.judge.evidence_keys(n)
        held: dict[str, dict[str, str]] = {}
        names: dict[str, str] = {}
        shown: dict[str, list[str]] = {}
        for i in line["verdicts"]:
            card = self.store.cards.get(i, {})
            stated = {k for w in line["wants"] for k in own_keys(card, w, line)}
            for key, value in (card.get("facts") or {}).items():
                kw = key_words(key)
                if not kw.strip() or kw.strip() in ("options", "option") or key in stated or any(same_key(key, c) for c in cited):
                    continue
                names.setdefault(kw, key)
                held.setdefault(kw, {}).setdefault(i, norm_text(value))
                text = " ".join(str(value).split())
                if len(shown.setdefault(kw, [])) < 3 and text and text.lower() not in [s.lower() for s in shown[kw]]:
                    shown[kw].append(cut_text(text, 40))
        order = sorted(held, key=lambda kw: (-len(held[kw]), -len(set(held[kw].values())), kw))
        twin = lambda kw, before: any(len(set(held[k]) & set(held[kw])) >= 2 and all(held[k][c] == held[kw][c] for c in set(held[k]) & set(held[kw])) for k in before)
        twins = [kw for k, kw in enumerate(order) if twin(kw, order[:k])]
        return [(names[kw], len(held[kw]), len(set(held[kw].values())), shown[kw]) for kw in [k for k in order if k not in twins] + twins]

    def plan_questions(self) -> None:
        tables = {n: self.table(n) for n in range(1, len(self.ledger.lines) + 1)}
        checks = [{"text": f"The shopper will check the {c['key']} themselves.", "line": c.get("line")} for c in self.ledger.checks if isinstance(c, dict) and c.get("key") and not c.get("value")]
        story = checks + self.ledger.story
        listing = []
        for n, rows in tables.items():
            stated = "; ".join(f"{w['key']} {w['op']} {' | '.join(map(str, w.get('values') or []))}".strip() for w in self.ledger.wants(n)) or "none"
            listing += [f"line {n}: {self.ledger.line(n)['item']}", f"  STATED: {stated}", "  CANDIDATES:"] + [f"  - {key} | {count} | {'; '.join(seen)}" for key, count, _, seen in rows[:20]]
        sentences = "\n".join(f"{k}. (line {s.get('line')}) {s['text']}" for k, s in enumerate(story, 1)) or "(none)"
        procedures = "\n".join(f"{k}. {p}" for k, p in enumerate(self.ledger.procedures, 1)) or "(none)"
        due = any(tables.values()) or story or self.ledger.procedures
        answer = self.brain.ask("STORY", STORY, "LINES:\n" + "\n".join(listing), f"SENTENCES:\n{sentences}\n\nPROCEDURES:\n{procedures}", STORY_SCHEMA) if due else None
        match = lambda n, name: next((r[0] for r in tables[n] if r[0] == name), None) or next((r[0] for r in tables[n] if name and same_key(r[0], name)), None)
        ranked: dict[int, list[str]] = {}
        sentence_keys: dict[int, list[str]] = {n: [] for n in tables}
        ungrounded = []
        for entry in (answer or {}).get("lines") or []:
            n = entry.get("line")
            if n not in tables:
                continue
            ranked[n] = [k for k in (match(n, str(x)) for x in entry.get("order") or []) if k]
            for pick in sorted(entry.get("pointed") or [], key=lambda p: p.get("sentence") or 0):
                k, name = pick.get("sentence"), str(pick.get("key") or "").strip()
                source = story[k - 1]["text"] if isinstance(k, int) and 1 <= k <= len(story) else ""
                key = match(n, name) or name
                if not (name and stands(pick.get("quote") or "", source)):
                    ungrounded.append(pick)
                elif key not in sentence_keys[n]:
                    sentence_keys[n].append(key)
        entries = (answer or {}).get("procedures") or []
        self.read_form(entries, tables, match, sentence_keys)
        self.pointed = {n: list(dict.fromkeys(self.form["required"].get(n, []) + sentence_keys[n])) for n in tables}
        self.gate.read_procedures(entries)
        for n, rows in tables.items():
            self.queue[n] = list(dict.fromkeys(self.pointed[n] + ranked.get(n, []) + [r[0] for r in rows]))
            if not rows and self.ledger.line(n)["seen"]:
                trace_row(self.trace, "ask_empty", reason="a line with kept rows has no candidate key yet: its queue fills once its records are judged (refill)", line=n, seen=len(self.ledger.line(n)["seen"]))
        self.planned = True
        trace_row(self.trace, "ask_plan", queue=self.queue, pointed=self.pointed, ranked=sorted(ranked), ungrounded=ungrounded, kept=self.gate.kept(), form=self.form, story=answer is not None)

    def read_form(self, entries: list, tables: dict, match, sentence_keys: dict[int, list[str]]) -> None:
        procs = self.ledger.procedures
        valid = [e for e in entries if isinstance(e, dict) and isinstance(e.get("procedure"), int) and 1 <= e["procedure"] <= len(procs)]
        counts = [e["questions"] for e in valid if isinstance(e.get("questions"), int) and not isinstance(e.get("questions"), bool) and e["questions"] >= 0]
        stored = lambda name: next((k for c in self.store.cards.values() for k in (c.get("facts") or {}) if same_key(k, name)), None)
        required: dict[int, list[str]] = {}
        refused, mapped = [], []
        for e in valid:
            for ask in (a for a in e.get("ask") or [] if isinstance(a, dict)):
                n, name = ask.get("line"), str(ask.get("key") or "").strip()
                grounded = n in tables and name and stands(ask.get("quote") or "", procs[e["procedure"] - 1])
                key = (match(n, name) or next((k for k in sentence_keys[n] if same_key(k, name)), None) or stored(name) or next(iter(sentence_keys[n]), None)) if grounded else None
                if key is None:
                    refused.append(dict(ask, procedure=e["procedure"]))
                    continue
                if not same_key(key, name):
                    mapped.append(dict(ask, procedure=e["procedure"], to=key))
                if key not in required.setdefault(n, []):
                    required[n].append(key)
        self.form = {"together": any(e.get("together") is True for e in valid), "count": min(counts) if counts else None, "required": required}
        if self.form["together"] or counts or required or refused:
            trace_row(self.trace, "ask_form", form=self.form, refused=refused, mapped=mapped, procedures=procs, reason="the request's procedures fix the form of the questions; the Asker follows it")

    def room(self) -> bool:
        return (self.form["count"] is None or self.questions < self.form["count"]) and not (self.form["together"] and self.questions)

    def refill(self) -> None:
        for n in self.queue:
            line = self.ledger.line(n)
            if not self.queue[n] and not line["told"] and (not line["asked"] or n in self.deferred):
                self.queue[n] = [k for k in dict.fromkeys(self.pointed.get(n, []) + [r[0] for r in self.table(n)]) if (n, k) not in self.sent]

    def order(self) -> list[int]:
        required = lambda n: any(k in self.queue[n] for k in self.form["required"].get(n, []))
        return sorted(self.queue, key=lambda n: (not required(n), bool(self.ledger.line(n)["told"]), len(self.ledger.line(n)["asked"]), n))

    def name(self, key: str, n: int) -> str:
        own = key_words(key).split()
        keys = {key_words(k).strip() for c in self.store.cards.values() for k in (c.get("facts") or {})}
        if len(own) > 1 and own[-1].isdigit() and " ".join(own[:-1]) in keys:
            own = own[:-1]
        return " ".join(own)

    def question(self, n: int, key: str) -> str:
        item = self.ledger.line(n)["item"]
        return f"What {self.name(key, n)} would you like for the {item}?"

    def bundle(self) -> list[tuple[int, str, str]]:
        k = max(sum(1 for q in self.queue.values() if q), (self.ledger.messages_left or 0) - self.gate.kept())
        k = k if self.form["count"] is None else min(k, self.form["count"] - self.questions)
        turn, rows = [(n, self.queue[n]) for n in self.order()], []
        for depth in range(max((len(q) for _, q in turn), default=0)):
            for n, q in turn:
                text = self.question(n, q[depth]) if depth < len(q) and len(rows) < k else None
                rows += [(n, q[depth], text)] if text else []
        return rows

    def message(self, n: int, first: str) -> list[tuple[int, str, str]]:
        q, spare = self.queue[n], (self.ledger.messages_left or 0) - self.gate.kept()
        lines = [m for m in self.order() if m != n and self.queue[m]]
        lines = [m for m in lines if not self.ledger.line(m)["told"]] or (lines if self.ledger.line(n)["told"] else [])
        own = max(1, spare - len(self.waiting()))
        if self.width > 1 and self.form["count"] is None and len(lines) + 1 > own:
            return [(n, q[0], first)] + [(m, self.queue[m][0], self.question(m, self.queue[m][0])) for m in lines[: -(-(len(lines) + 1) // own) - 1]]
        room = self.width - 1 if self.form["count"] is None else min(self.width - 1, self.form["count"] - self.questions - spare)
        more = q[1 : 1 + max(0, room)]
        return [(n, q[0], first)] + [(n, k, self.question(n, k)) for k in more]

    def text(self, rows: list[tuple[int, str, str]]) -> str:
        if len(rows) == 1:
            return rows[0][2]
        listed = []
        for n in dict.fromkeys(m for m, _, _ in rows):
            names = [self.name(key, m) for m, key, _ in rows if m == n]
            listed.append((names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}", self.ledger.line(n)["item"]))
        (head, item), rest = listed[0], listed[1:]
        return f"What {head} would you like for the {item}" + "".join(f",{' and' if k == len(rest) - 1 else ''} what {names} for the {it}" for k, (names, it) in enumerate(rest)) + "?"

    def told_keys(self, rows: list[tuple[int, str, str]], answer: dict) -> set[tuple[int, str]]:
        told = set()
        for need in answer.get("needs") or [] if answer.get("kind") == "told" else []:
            n, said = need["line"], str(need.get("key") or "")
            items = {key_form([w]) for w in item_words(self.ledger.line(n))}
            score = [(same_key(key, said), len(content(key, False) & content(said, False) - items), -k, key) for k, (m, key, _) in enumerate(rows) if m == n]
            best = max(score, default=None)
            if best and (best[0] or best[1]):
                told.add((n, best[3]))
        return told

    def say(self, text: str) -> str | None:
        spec = self.store.roles.get("message") or {}
        if not spec.get("tool") or not self.ledger.messages_left:
            return None
        result = self.store.act(spec["tool"], {spec.get("text_arg") or "text": text})
        self.ledger.messages_left -= 1
        if spec["tool"] in self.store.env.shut or (result.get("error") and result.get("observation") is None):
            self.ledger.messages_left = 0
            return None
        obs = result.get("observation")
        return raw_of(obs) if obs is not None else None

    def read(self, asked: list[tuple[int, str]], question: str, reply: str, seen: list | None = None) -> dict:
        lines = "\n".join(f"line {k}: {l['item']}" for k, l in enumerate(self.ledger.lines, 1))
        values = f"SEEN (values the store's records give for {asked[0][1]}): {seen}\n" if seen is not None else ""
        now = "".join(f"ASKED (line {n}): {key}\n" for n, key in asked) + f"MESSAGE: {question}\n{values}REPLY: {reply}\n\nREQUIREMENTS:\n{self.ledger.listing()}"
        tried: list[dict] = []
        check = lambda a: tried.append(a) or cap_problem(a, reply)
        answer = self.brain.ask("READ" if seen is None else "READ_ALT", READ, "LINES:\n" + lines, now, READ_SCHEMA, check) or (tried[-1] if tried else {"kind": "other", "needs": [], "changes": []})
        grounded = lambda d: all(stands(v, reply) for v in d.get("values") or [])
        loose = [d for d in answer.get("needs") or [] if not grounded(d)]
        if loose:
            trace_row(self.trace, "need_ungrounded", asked=asked, needs=loose, reason="a value the reply does not write proves nothing (T5)")
        answer["needs"] = [dict(d, relation=None if d.get("relation") in (None, "none") else d["relation"]) for d in answer.get("needs") or [] if grounded(d)]
        answer["changes"] = answer.get("changes") or []
        return answer

    def hear(self, asked: list[tuple[int, str]], question: str, reply: str) -> dict:
        answer = self.read(asked, question, reply)
        named = len(asked) == 1 and key_named(reply, asked[0][1])
        if named and not answer["needs"] and answer.get("kind") not in ("no_preference", "one_at_a_time"):
            n, key = asked[0]
            cards = [self.store.cards.get(i, {}) for i in self.ledger.line(n)["verdicts"]]
            seen = list(dict.fromkeys(str(v) for c in cards for k, v in (c.get("facts") or {}).items() if same_key(k, key)))[:8]
            first, answer = answer, self.read(asked, question, reply, seen)
            trace_row(self.trace, "reread", line=n, key=key, seen=seen, first=first, second=answer)
        return dict(answer, named=named)

    def converse(self, asked: list[tuple[int, str]], text: str) -> tuple[dict, set[int]] | None:
        reply = self.say(text)
        if reply is None:
            return None
        answer = self.hear(asked, text, reply)
        answer["needs"] = [dict(d, line=d["line"] if isinstance(d.get("line"), int) and 1 <= d["line"] <= len(self.ledger.lines) else asked[0][0] if asked else 1) for d in answer["needs"]]
        kind = answer.get("kind") if answer["needs"] or answer.get("kind") != "told" else "other"
        changed: set[int] = set()
        for need in answer["needs"] if kind == "told" else []:
            m = need["line"]
            got = change_ledger(self.ledger, dict(need, change="add"), reply, "disclosed", self.trace)
            if got:
                self.ledger.line(m)["told"].append(need["key"])
                self.queue[m] = [k for k in self.queue.get(m, []) if not same_key(k, need["key"])]
            changed |= got
        for change in answer["changes"]:
            if kind == "no_preference" and change.get("change") == "drop":
                trace_row(self.trace, "change_refused", change=change, source="reply", reason="a no-preference reply settles the asked keys alone and drops no other requirement")
            elif change.get("change") == "stock":
                self.gate.stock_heard(change.get("quote"))
            else:
                changed |= change_ledger(self.ledger, change, reply, "reply", self.trace)
        self.talk.append({"asked": asked, "lines": sorted({n for n, _ in asked}), "message": text, "reply": reply, "kind": kind, "changed": sorted(changed)})
        trace_row(self.trace, "asked", asked=asked, question=text, reply=reply, read=answer, changed=sorted(changed))
        return dict(answer, kind=kind), changed

    def ask(self, rows: list[tuple[int, str, str]], regather) -> bool:
        for n, key, text in rows:
            self.sent.setdefault((n, key), []).append(text)
        heard = self.converse([(n, key) for n, key, _ in rows], self.text(rows))
        if heard is None:
            return False
        self.questions += len(rows)
        if len(rows) > 1 and heard[0].get("kind") == "one_at_a_time":
            self.width, self.form["together"] = 1, False
            trace_row(self.trace, "ask_form", form=self.form, asked=[(n, key) for n, key, _ in rows], reason="the shopper refused several questions at once: no turn and no ask of them, single questions from here")
        else:
            self.settle(rows, heard[0])
        for m in sorted(heard[1]):
            regather(m)
        return True

    def waiting(self) -> list[int]:
        return [n for n, q in self.queue.items() if not q and not any(self.ledger.line(n)[k] for k in ("asked", "told", "verdicts"))]

    def ask_all(self, regather, final: bool = True) -> None:
        if not self.planned:
            self.plan_questions()
        more = self.plan.rounds("Asker.ask_all")
        held = lambda: self.gate.kept() + (0 if final else len(self.waiting()))
        while (self.ledger.messages_left or 0) > held() and self.plan.ask_fits() and self.room() and more(self.queue, self.questions, self.width, self.form):
            self.refill()
            n = next((k for k in self.order() if self.queue[k]), None)
            if n is None:
                break
            question = self.question(n, self.queue[n][0])
            if not self.ask(self.bundle() if self.form["together"] else self.message(n, question), regather):
                break
        left = {k: q[:3] for k, q in self.queue.items() if q}
        never = [k for k in self.queue if not self.ledger.line(k)["asked"]]
        if (self.ledger.messages_left or 0) > held():
            trace_row(self.trace, "ask_stop", messages_left=self.ledger.messages_left, kept=held(), unasked=left, never_asked=never, steps_left=self.store.env.left(),
                      form=self.form, questions=self.questions, reason="the step plan, the request's form of the questions or the loop guard holds the questions with keys left" if left else "no key is left to ask")
        elif [k for k in never if k in left]:
            trace_row(self.trace, "ask_lines", lines=[k for k in never if k in left], unasked=left, questions=self.questions,
                      reason="the question messages ran out before these lines with keys were asked: their stated requirements alone decide them")

    def settle(self, rows: list[tuple[int, str, str]], answer: dict) -> None:
        kind, alone = answer.get("kind"), len(rows) == 1
        told = set() if alone else self.told_keys(rows, answer)
        asked = [(n, key) for n, key, _ in rows]
        answered = set(asked[: min(asked.index(p) for p in told) + 1]) | told if told else set(asked[:1])
        for n, key, _ in rows:
            self.ledger.line(n)["asked"].append(key)
            if alone and answer.get("named") and kind == "other" and (n, key) not in self.extra:
                self.extra.add((n, key))
                trace_row(self.trace, "ask_again", line=n, key=key, reply=kind, reason="the reply names the key but tells nothing readable: one more ask")
                continue
            if not alone and (n, key) not in answered:
                continue
            self.queue[n] = [k for k in self.queue[n] if k != key]
            if kind not in ("one_at_a_time", "not_understood"):
                self.aside(n, key)
        kept = [(n, key) for n, key, _ in rows if not alone and key in self.queue[n]]
        if kept:
            trace_row(self.trace, "ask_kept", keys=kept, reply=kind, told=sorted(told), reason="names after the one the reply answered were masked by it: they lead the next message")

    def aside(self, n: int, key: str) -> None:
        values: dict[str, set] = {}
        for i in self.ledger.line(n)["verdicts"]:
            for k, v in (self.store.cards.get(i, {}).get("facts") or {}).items():
                values.setdefault(key_words(k), set()).update(norm_text(x) for x in fact_items(v) if norm_text(x).strip())
        mine = values.get(key_words(key)) or set()
        later = [k for k in self.queue[n] if values.get(key_words(k)) and values[key_words(k)] <= mine]
        if later:
            self.queue[n] = [k for k in self.queue[n] if k not in later] + later
            trace_row(self.trace, "ask_aside", line=n, key=key, later=later, reason="their values on the line's records are the answered key's values or some of them: asked last, never removed")

    def late(self, doubt: list[int]) -> set[int] | None:
        if not self.room():
            trace_row(self.trace, "late_skip", reason="the request's form of the questions allows no further question", form=self.form, questions=self.questions)
            return None
        self.refill()
        n = next((k for k in doubt + self.order() if self.queue.get(k)), None)
        if n is None:
            trace_row(self.trace, "late_skip", reason="no key left to ask", messages_left=self.ledger.messages_left)
            return None
        key = self.queue[n][0]
        question = self.question(n, key)
        changed: set[int] = set()
        self.deferred.clear()
        if not self.ask(self.message(n, question), lambda m: changed.add(m)):
            return None
        trace_row(self.trace, "late_ask", line=n, key=key, question=question, changed=sorted(changed))
        return changed

class Closer:

    def __init__(self, brain: Brain, store: Store, ledger: Ledger, picker: Picker, trace: list) -> None:
        self.brain, self.store, self.ledger, self.picker, self.trace = brain, store, ledger, picker, trace
        self.plan = picker.gather.plan
        self.applied: set = set()
        self.in_cart: dict[str, int] = {}
        self.viewed = False
        self.qty_key: str | None = None
        self.touched: dict[str, tuple[str, bool]] = {}
        self.heard = False

    def apply_notices(self) -> set[int]:
        fresh = [t for t in self.store.notices if t not in self.applied]
        self.applied.update(fresh)
        self.heard = bool(fresh)
        if not fresh:
            return set()
        said, tried = "\n".join(fresh), []
        answer = self.brain.ask("NOTICE", NOTICE, "LINES:\n" + self.ledger.listing(items=True), "NOTICES:\n" + said, NOTICE_SCHEMA, lambda a: tried.append(a) or cap_problem(a, said)) or (tried[-1] if tried else {})
        changed: set[int] = set()
        for c in (c for c in answer.get("changes") or [] if isinstance(c, dict)):
            if c.get("change") in ("stock", "price"):
                self.mark(c["change"], c.get("id"))
            elif c.get("change") in ("drop", "add", "cap", "cap_each"):
                changed |= change_ledger(self.ledger, c, said, "notice", self.trace)
        trace_row(self.trace, "notices", notices=fresh, changes=answer.get("changes"), changed=sorted(changed), touched=self.touched, answered=bool(answer))
        return changed

    def mark(self, kind: str, ident: Any = None) -> None:
        if ident in self.store.cards:
            self.touched[ident] = (kind, True)
        else:
            self.touched.update({i: (kind, False) for i in list(self.in_cart) or [l["pick"] for l in self.ledger.lines if l.get("pick")] if not self.touched.get(i, ("", False))[1]})

    def recheck(self) -> set[int]:
        due = self.plan.again() - 2
        ids = list(dict.fromkeys(list(self.touched) + list(self.in_cart)))
        named, self.touched = dict(self.touched), {}
        moved: set[int] = set()
        for k, ident in enumerate(ids):
            card = self.store.card(ident)
            before = (card.get("price"), card.get("stock"))
            read = self.plan.essential(1 + due) if k == 0 and self.ledger.refusals else self.plan.spare(1, due)
            if read:
                self.store.reread(ident)
            if (not read or ident in self.store.stale) and named.get(ident, ("", False))[0] == "stock" and (named[ident][1] or len(self.in_cart) == 1):
                card["stock"] = False
            elif not read:
                trace_row(self.trace, "recheck_held", id=ident, left=self.store.env.left(), reason="no step for this re-read before the refusal round's writes and the order")
            if (card.get("price"), card.get("stock")) != before:
                moved |= {n for n, l in enumerate(self.ledger.lines, 1) if l.get("pick") == ident}
        return moved

    def others(self, n: int) -> list[str]:
        fitter, line = self.picker.fitter, self.ledger.line(n)
        group = fitter.group(n)
        pool = (fitter.anchors.get(n) or []) if group and group[0] == n else self.picker.ranked(n)
        main = self.ledger.line(group[0]).get("pick") if group and group[0] != n else None
        fits = lambda i: self.picker.judge.status(n, i) in ("PASS", "WEAK") and (main is None or fitter.fit(main, i) == "PASS")
        return [i for i in pool if i != line.get("pick") and fits(i)][:3]

    def want(self, swap: dict | None = None) -> dict[str, int]:
        want: dict[str, int] = {}
        for n, line in enumerate(self.ledger.lines, 1):
            ident = (swap or {}).get(n) or line.get("pick")
            if ident:
                want[ident] = want.get(ident, 0) + line["qty"]
        return want

    def removal(self, ident: str) -> dict | None:
        spec = self.store.roles.get("cart_write") or {}
        qkey = spec.get("qty_arg") or "qty"
        if spec.get("op_arg") and spec.get("remove_value") is not None:
            return {spec["op_arg"]: spec["remove_value"], qkey: self.in_cart.get(ident)}
        q = tool_props(self.store.tools.get(spec.get("tool") or "", {})).get(qkey) or {}
        low, above = (q.get("minimum"), q.get("exclusiveMinimum")) if isinstance(q, dict) else (1, None)
        return {qkey: 0} if (low is None or low <= 0) and not (isinstance(above, (int, float)) and not isinstance(above, bool) and above >= 0) else None

    def owed(self, want: dict, whole: bool = False) -> int:
        spec = self.store.roles.get("cart_write") or {}
        if self.in_cart == want or not spec.get("tool"):
            return 0
        brought = [i for i, q in want.items() if self.in_cart.get(i) != q]
        writes = 1 if spec.get("mode") == "replace" and spec.get("list_arg") else len([i for i, q in self.in_cart.items() if want.get(i) != q and self.removal(i) is not None]) + len(brought)
        page = (self.store.roles.get("page") or {}).get("tool")
        return writes + (self.plan.view() + sum(1 for i in brought if page and i not in self.store.pages) if whole else 0)

    def write_cart(self) -> bool:
        if not (self.store.roles.get("cart_write") or {}).get("tool"):
            return False
        want = self.want()
        page = (self.store.roles.get("page") or {}).get("tool")
        for again in (False, True):
            wrote = self.put(want)
            for ident in [] if again else wrote:
                if page and ident not in self.store.pages and self.plan.essential(1):
                    self.picker.open_page(ident)
                    if all(self.store.shows.get(page, (False, False))):
                        continue
                if self.plan.spare(1, self.plan.view()):
                    self.store.reread(ident)
            if not self.view() or self.in_cart == want:
                break
        if self.in_cart != want:
            trace_row(self.trace, "handoff", reason="cart view differs from the picks", view=self.in_cart, picks=want, viewed=self.viewed)
        return True

    def put(self, want: dict) -> list[str]:
        spec = self.store.roles.get("cart_write") or {}
        if self.in_cart == want:
            return []
        if not self.plan.essential(self.owed(want)):
            trace_row(self.trace, "put_held", cart=self.in_cart, want=want, writes=self.owed(want), left=self.store.env.left(), reason="the writes this basket needs do not all fit before the order: the cart stays as it is")
            return []
        fixed = dict(spec.get("fixed") or {})
        self.viewed = False
        if spec.get("mode") == "replace" and spec.get("list_arg"):
            items = [{spec.get("item_id_key") or "id": i, spec.get("item_qty_key") or "qty": q} for i, q in want.items()]
            if self.store.act(spec["tool"], {**fixed, spec["list_arg"]: items}).get("error"):
                return []
            wrote = [i for i, q in want.items() if self.in_cart.get(i) != q]
            self.in_cart = dict(want)
            return wrote
        id_arg, qkey = spec.get("id_arg") or "id", spec.get("qty_arg") or "qty"
        stale = [i for i, n in self.in_cart.items() if want.get(i) != n]
        if [i for i in stale if self.removal(i) is None]:
            trace_row(self.trace, "handoff", reason="cart write cannot remove an item", items=[i for i in stale if self.removal(i) is None], spec=spec)
        for ident in (i for i in stale if self.removal(i) is not None):
            if not self.store.act(spec["tool"], {**fixed, id_arg: ident, **self.removal(ident)}).get("error"):
                self.in_cart.pop(ident)
        wrote = []
        add = {spec["op_arg"]: spec["add_value"]} if spec.get("op_arg") and spec.get("add_value") is not None else {}
        for ident, n in want.items():
            if self.in_cart.get(ident) != n and not self.store.act(spec["tool"], {**fixed, id_arg: ident, qkey: n, **add}).get("error"):
                self.in_cart[ident] = n
                wrote.append(ident)
        return wrote

    def view(self) -> bool:
        spec = self.store.roles.get("cart_view") or {}
        if not spec.get("tool") or not self.plan.essential(1):
            return False
        result = self.store.act(spec["tool"], {})
        shown = self.read_view(result.get("observation"))
        missing = sorted(i for i in self.in_cart if not self.viewed and i not in raw_of(result.get("observation")))
        if missing and not result.get("error"):
            trace_row(self.trace, "handoff", reason="cart view does not show an id written", missing=missing, view=raw_of(result.get("observation")))
        if shown is None:
            trace_row(self.trace, "handoff", reason="cart view unreadable: the cart is counted from the writes", counted=self.in_cart, view=raw_of(result.get("observation"))[:300])
            return False
        self.in_cart, self.viewed = shown, True
        return True

    def read_view(self, obs: Any) -> dict | None:
        known, rows, stack = set(self.store.cards), [], [obs]
        while stack:
            x = stack.pop()
            ident = next((v for v in x.values() if isinstance(v, str) and v in known), None) if isinstance(x, dict) else None
            if ident:
                rows.append((ident, x))
            elif isinstance(x, (dict, list)):
                stack.extend(x.values() if isinstance(x, dict) else x)
        count = lambda v: v if isinstance(v, int) and not isinstance(v, bool) else None
        price_key = ((self.store.fields.get("row") or {}) if isinstance(self.store.fields.get("row"), dict) else {}).get("price")
        for ident, row in rows:
            price, card = row.get(price_key) if isinstance(price_key, str) else None, self.store.cards[ident]
            if isinstance(price, (int, float)) and not isinstance(price, bool) and (not isinstance(card.get("price"), (int, float)) or float(card["price"]) != float(price)):
                trace_row(self.trace, "cart_price", id=ident, was=card.get("price"), now=price, reason="the cart view shows another price: the store's latest word stands")
                card["price"] = price
        written = [(ident, row) for ident, row in rows if ident in self.in_cart]
        hits = [m for m in ({k for k, v in row.items() if count(v) == self.in_cart[ident]} for ident, row in written) if m]
        keys = {k for k in set.intersection(*hits) if all(count(row.get(k)) is not None for _, row in written)} if hits else set()
        if self.qty_key is None and len(keys) == 1:
            self.qty_key = keys.pop()
        elif self.qty_key is None and written and not hits:
            trace_row(self.trace, "qty_unread", written=self.in_cart, rows=[row for _, row in written][:6], reason="no written row shows the count written in an integer field: the counts written stand")
        shown = {}
        for ident, row in rows:
            n = count(row.get(self.qty_key)) if self.qty_key else None
            shown[ident] = self.in_cart.get(ident, 1) if n is None else n
        shown = {i: n for i, n in shown.items() if n}
        if not rows:
            text = raw_of(obs)
            shown = {i: self.in_cart.get(i, 1) for i in known if i in text}
        return None if not shown and self.in_cart else shown

    def evidence(self, n: int, talk: list[dict]) -> str:
        line, pick = self.ledger.line(n), self.ledger.line(n).get("pick")
        rows = []
        for k, w in enumerate(self.ledger.wants(n), 1):
            src = w.get("source")
            said = next((t for t in talk if stands(w.get("quote") or "", t["reply"])), None) if src == "disclosed" else None
            where = (f"the shopper's reply {said['reply']!r} to {said['message']!r}" if said else f"the shopper's words {w.get('quote')!r}") if src == "disclosed" else (
                "pages " + " -> ".join(f"{t['page']} ({(t.get('read') or {}).get('quote') or (t.get('read') or {}).get('value')!r})" for t in w.get("trail") or []) if src == "chain" else f"{src or 'the request'}: {w.get('quote')!r}")
            got = line["verdicts"].get(pick, {}).get(want_tag(w)) if pick else None
            verdict = f"{self.picker.judge.label(n, pick, w)} ({got[1]}: {got[2]!r})" if got else "not judged"
            rows.append(f"  {k}. {want_text(w)} | from {where} | pick: {verdict}")
        return "\n".join(rows) or "  (none)"

    def judge_basket(self, talk: list[dict]) -> dict:
        parts, shown = [], {}
        for n, line in enumerate(self.ledger.lines, 1):
            pick = line.get("pick")
            shown[n] = self.others(n)
            fits = [f"\n  FIT (must fit line {f['with']}: {f['quote']!r}): {f['pair']}" for f in self.ledger.fits if f.get("line") == n and f.get("pair")]
            chains = [f"\n  CHAIN NOT FINISHED: {w['key']} from {w.get('quote') or w['values']!r}" for w in self.ledger.wants(n) if w.get("op") == "ref"]
            parts.append(
                f"LINE {n}: {line['qty']} x {line['item']} ({line['objective']}) cap {line.get('cap_each')}\n  EVIDENCE:\n{self.evidence(n, talk)}\n"
                f"  PICK: {record_view(self.store.cards[pick]) if pick else '(none)'}\n  OTHER CANDIDATES: " + (" || ".join(card_view(self.store.cards[i]) for i in shown[n]) or "(none)") + "".join(fits + chains)
            )
        dialogue = "\n".join(f"Q ({'line ' + ', '.join(map(str, t['lines'])) if t['lines'] else 'a procedure message'}): {t['message']}\nA: {t['reply']}" for t in talk) or "(none)"
        unhandled = "\n".join(f"- {json.dumps(u, ensure_ascii=False, default=str)}" for u in self.ledger.unhandled) or "(none)"
        task = f"REQUEST:\n{self.ledger.request}\n\nTOTAL CAP: {self.ledger.total_cap}\nSTORE NOTICES: {self.store.notices}\n\nSHOPPER DIALOGUE:\n{dialogue}\n\nUNHANDLED:\n{unhandled}"
        verdict = self.brain.ask("JUDGE", JUDGE, task, "\n\n".join(parts), JUDGE_SCHEMA)
        if verdict is None:
            trace_row(self.trace, "judge_missing", spent=round(self.brain.spent, 4))
            return {"order": True, "lines": [], "pointers": []}
        rows, dropped = {}, []
        for row in verdict.get("lines") if isinstance(verdict.get("lines"), list) else []:
            n = row.get("line") if isinstance(row, dict) else None
            if not (isinstance(n, int) and not isinstance(n, bool) and n in shown) or n in rows:
                dropped.append({"entry": row, "why": "no line of the order, or a second entry for one"})
                continue
            if row.get("use") is not None and row["use"] not in shown[n]:
                dropped.append({"entry": row, "why": "its use is none of the line's other candidates sent"})
                row = dict(row, use=None)
            rows[n] = row
        given = verdict.get("pointers") if isinstance(verdict.get("pointers"), list) else []
        pointers = [p for p in given if isinstance(p, dict) and isinstance(p.get("line"), int) and p["line"] in shown and str(p.get("text") or "").strip()]
        dropped += [{"entry": p, "why": "a pointer for no line of the order, or with no text"} for p in given if p not in pointers]
        if dropped:
            trace_row(self.trace, "judge_dropped", dropped=dropped, reason="JUDGE's answer is read only against what was sent")
        return dict(verdict, lines=list(rows.values()), pointers=pointers)

    def finish_call(self, role: str, args: dict) -> bool:
        spec = self.store.roles.get(role) or {}
        if not spec.get("tool"):
            return False
        result = self.store.act(spec["tool"], args)
        trace_row(self.trace, role, result=result, args=args)
        return bool(result.get("finished"))

    def audit(self, missing: list[int]) -> bool:
        line_of, shown, parts = self.ledger.line, {}, []
        for n in missing:
            line, wants = line_of(n), self.ledger.wants(n)
            certain = lambda i: any((code_verdict(self.store.cards[i], w, line) or ("",))[0] == "FAIL" for w in wants if w.get("op") != "ref")
            judged = [i for i in line["verdicts"] if self.picker.gather.keep(line, self.store.cards[i]) and not certain(i)][:6]
            rows = [i for i in self.picker.gather.ordered(n) if not self.store.opened(i)][:20]
            records = "\n".join(record_view(self.store.cards[i]) + "\n  OUR VERDICTS: " + "; ".join(f"{k} {self.picker.judge.label(n, i, w)}" for k, w in enumerate(wants, 1)) for i in judged)
            row = lambda c: f"{c['id']} {cut_text(c.get('title', ''), 80)} | shelf: {c.get('category')} | price: {c.get('price')} | in stock: {c.get('stock')}"
            reqs = "\n".join(f"  {k}. {want_text(w)}" for k, w in enumerate(wants, 1))
            parts.append(f"LINE {n}: {line['qty']} x {line['item']}\nREQUIREMENTS:\n{reqs}\nOPENED RECORDS:\n{records or '(none)'}\nROWS NOT OPENED:\n" + ("\n".join(row(self.store.cards[i]) for i in rows) or "(none)"))
            shown[n] = (judged, rows)
        answer = self.brain.ask("AUDIT", AUDIT, "REQUEST:\n" + self.ledger.request, "\n\n".join(parts), AUDIT_SCHEMA) or {}
        raised, opens = [], {}
        for entry in (e for e in answer.get("lines") or [] if isinstance(e, dict)):
            n = entry.get("line")
            if n not in shown:
                continue
            judged, rows = shown[n]
            wants = self.ledger.wants(n)
            points: dict[str, list[str]] = {}
            for proof in (f for f in entry.get("fits") or [] if isinstance(f, dict)):
                i, k, quote = str(proof.get("id")), proof.get("want"), str(proof.get("quote") or "")
                if i in judged and isinstance(k, int) and 1 <= k <= len(wants) and stands(quote, record_view(self.store.cards[i])):
                    points.setdefault(i, []).append(f"requirement {k} ({wants[k - 1]['key']} {wants[k - 1]['op']} {' | '.join(wants[k - 1].get('values') or [])}) is proven by the record's words {quote!r}")
            for i, said in points.items():
                self.picker.judge.dispute(n, i, "AUDIT: " + "; ".join(said))
                raised.append((n, i))
            opens[n] = [i for i in entry.get("open") or [] if i in rows][:4]
        for n, ids in opens.items():
            if ids and not self.plan.walk_over():
                self.picker.open_and_judge(n, ids)
        held = {n: [i for i in ids if not self.store.opened(i)] for n, ids in opens.items() if any(not self.store.opened(i) for i in ids)}
        trace_row(self.trace, "audit", lines=missing, answered=bool(answer), raised=raised, opens=opens, held=held, left=self.store.env.left(), reserve=self.plan.reserve())
        return bool(raised) or any(self.store.opened(i) for ids in opens.values() for i in ids)


class FinishGate:

    def __init__(self, ledger: Ledger, store: Store, closer: Closer, plan: StepPlan, trace: list) -> None:
        self.ledger, self.store, self.closer, self.plan, self.trace = ledger, store, closer, plan, trace
        self.needs: dict[int, str] = {}
        self.sent: set = set()
        self.pointers: dict[tuple, str] = {}
        self.held = False

    def read_procedures(self, entries: list) -> None:
        count = len(self.ledger.procedures)
        self.needs = {e["procedure"]: e["message"] for e in entries if isinstance(e, dict) and isinstance(e.get("procedure"), int) and 1 <= e["procedure"] <= count and e.get("message") in ("order", "decline")}
        if self.needs:
            trace_row(self.trace, "procedures", needs=self.needs, procedures=self.ledger.procedures)

    def due(self, when: str) -> list[int]:
        return [k for k, w in self.needs.items() if w == when and k not in self.sent]

    def kept(self) -> int:
        return max(len(self.due("order")), len(self.due("decline")))

    def free(self) -> int:
        return max(0, (self.ledger.messages_left or 0) - len(self.due("order")))

    def total(self) -> float:
        return sum(price_of(self.store.cards.get(l.get("pick") or ""), 0) * l["qty"] for l in self.ledger.lines)

    def over(self) -> bool:
        cap = self.ledger.total_cap
        return isinstance(cap, (int, float)) and not isinstance(cap, bool) and self.total() > cap

    def stock_heard(self, quote: Any) -> None:
        self.closer.mark("stock")
        trace_row(self.trace, "stock_heard", quote=quote, touched=self.closer.touched)

    def procedures(self, when: str, asker: Asker) -> set[int]:
        changed: set[int] = set()
        for k in self.due(when):
            if not self.plan.essential(self.plan.first() if when == "order" else 1) or not self.ledger.messages_left:
                trace_row(self.trace, "duty_missed", duty=f"procedure {k} message before the {when}", procedure=self.ledger.procedures[k - 1], left=self.store.env.left(), messages=self.ledger.messages_left)
                continue
            if when == "order":
                rows = [f"{l['qty']} x {self.store.cards[l['pick']].get('title')} ({l['pick']}, {self.store.cards[l['pick']].get('price')} each)" for l in self.ledger.lines if l.get("pick")]
                text = f"Ahead of ordering, as you asked ({self.ledger.procedures[k - 1]}), the basket: " + "; ".join(rows) + "."
            else:
                text = f"Ahead of declining, as you asked ({self.ledger.procedures[k - 1]}): " + self.why()
            self.sent.add(k)
            heard = asker.converse([], text)
            changed |= heard[1] if heard else set()
        return changed

    def why(self) -> str:
        said = lambda w: f"{w['key'] or 'the feature'} {w['op']} {'/'.join(map(str, w.get('values') or []))}"
        picker, out = self.closer.picker, []
        for n, line in enumerate(self.ledger.lines, 1):
            if line.get("pick"):
                continue
            wants, judged, cards = self.ledger.wants(n), list(line["verdicts"]), self.store.cards
            label = lambda i, w: picker.judge.label(n, i, w)
            rule = lambda i: picker.gather.set_aside(line, cards[i])
            unmet = [w for w in wants if not any(label(i, w) == "PASS" for i in judged)]
            whole = [i for i in judged if all(label(i, w) == "PASS" for w in wants)]
            least = min(line["seen"], key=lambda i: cards[i]["price"] if isinstance(cards[i].get("price"), (int, float)) else float("inf"), default=None)
            aside = None if least is None or any(picker.gather.keep(line, cards[i]) for i in line["seen"]) else picker.gather.set_aside(line, cards[least])
            text = ((f"all {len(line['seen'])} listings found are set aside, the cheapest for {aside}" if aside else "nothing of it could be opened") if not judged else "no listing found meets " + "; ".join(map(said, unmet)) if unmet else
                    "the listings that meet every requirement are set aside (" + "; ".join(dict.fromkeys(rule(i) or "none shown to fit the rest of the order" for i in whole)) + ")" if whole else "no listing meets " + " and ".join(map(said, wants)) + " together")
            if self.ledger.nearest and judged:
                price = lambda i: cards[i]["price"] if isinstance(cards[i].get("price"), (int, float)) else float("inf")
                near = min(judged, key=lambda i: (sum(label(i, w) == "FAIL" for w in wants) + (rule(i) is not None), sum(label(i, w) not in ("PASS", "FAIL") for w in wants), price(i)))
                lacks = [said(w) for w in wants if label(near, w) != "PASS"] + [r for r in [rule(near)] if r]
                text += f"; the nearest listing is {near}" + (f", short on {'; '.join(lacks)}" if lacks else "")
            out.append(f"for the {line['item']}, {text}")
        if self.over():
            least = sum(picker.gather.cheapest_pass(n) * l["qty"] for n, l in enumerate(self.ledger.lines, 1))
            out.append(f"no combination found of listings that meet every requirement stays within the total cap of {self.ledger.total_cap}" + (f": together they cost at least {least}" if least > self.ledger.total_cap else ""))
        return ("; ".join(out) or "no basket meets every requirement of the request") + "."

    def point(self, pointers: list) -> None:
        for p in pointers:
            self.pointers.setdefault((p["line"], str(p["text"]).strip()), "due")

    def duties(self, picker: Picker) -> set[int]:
        chain_pages = self.ledger.chain_pages
        owed = lambda: self.closer.owed(self.closer.want())
        for ident in [l["pick"] for l in self.ledger.lines if l.get("pick")] + [i for i in chain_pages if i in self.store.cards]:
            if ident not in self.store.pages and self.plan.essential(1 + owed()):
                picker.open_page(ident)
        unread = [i for i in [l["pick"] for l in self.ledger.lines if l.get("pick")] + chain_pages if i not in self.store.pages]
        if unread:
            trace_row(self.trace, "duty_missed", duty="pick and chain pages opened as pages", ids=unread, owed=owed(), left=self.store.env.left())
        judged: set[int] = set()
        for (n, name), state in list(self.pointers.items()):
            if state != "due":
                continue
            if not self.plan.spare(1, owed()):
                self.pointers[(n, name)] = "held"
                continue
            line = self.ledger.line(n)
            rows = [c["id"] for c in self.store.search(name)["cards"] if picker.gather.keep(line, self.store.cards.get(c["id"], c))]
            line["seen"] += [i for i in rows if i not in line["seen"]]
            fresh = [i for i in rows if not self.store.opened(i)][:2]
            if fresh and self.plan.spare(len(fresh), owed()):
                picker.open_and_judge(n, fresh)
                judged.add(n)
            self.pointers[(n, name)] = "read"
            trace_row(self.trace, "pointer", line=n, name=name, rows=rows[:6], opened=[i for i in fresh if self.store.opened(i)])
        held = [k for k, s in self.pointers.items() if s == "held"]
        if held:
            trace_row(self.trace, "duty_missed", duty="pointers searched and judged", pointers=held, left=self.store.env.left())
        return judged

    def order(self) -> bool:
        self.held = False
        if self.over():
            self.held = True
            trace_row(self.trace, "over_cap", total=self.total(), cap=self.ledger.total_cap, picks={n: l.get("pick") for n, l in enumerate(self.ledger.lines, 1)}, reason="order held: the basket is over the stated total cap")
            return False
        want = self.closer.want()
        whole = bool(want) and all(l.get("pick") for l in self.ledger.lines)
        if whole and any(self.closer.in_cart.get(i) != q for i, q in want.items()):
            self.closer.put(want)
        missing = [l.get("pick") for l in self.ledger.lines if not l.get("pick") or self.closer.in_cart.get(l["pick"]) != want.get(l["pick"])]
        if missing:
            self.held = True
            trace_row(self.trace, "handoff", reason="order held: the cart lacks a line's pick", missing=missing, cart=self.closer.in_cart)
            return False
        return self.closer.finish_call("order", dict((self.store.roles.get("order") or {}).get("args") or {}))

    def decline(self, asker: Asker) -> bool:
        self.procedures("decline", asker)
        spec = self.store.roles.get("decline") or {}
        args = dict(spec.get("args") or {})
        props = tool_props(self.store.tools.get(spec.get("tool") or "", {}))
        free = [k for k, v in props.items() if k not in args and isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")]
        rule = self.ledger.decline_rule if isinstance(self.ledger.decline_rule, str) and self.ledger.decline_rule.strip() else None
        if free:
            args[free[0]] = (f"As you asked for when nothing fits ({rule}): " if rule else "") + self.why()
        return self.closer.finish_call("decline", args)

class StepPlan:

    def __init__(self, env: Env, ledger: Ledger, store: Store) -> None:
        self.env, self.ledger, self.store = env, ledger, store
        self.opens, self.ask_at = 1, 0
        self.asked = False
        self.deferred = 0
        self.gate: FinishGate | None = None
        self.finishing = False
        self.until: int | None = None

    def set(self) -> dict:
        lines = max(1, len(self.ledger.lines))
        self.opens = max(1, min(6 if lines == 1 else 3, self.walk_left() // lines - 1))
        self.ask_at = self.env.step + max(self.env.max_steps // 4, sum(sum(self.phase_a(n, lines)) for n in range(1, lines + 1)))
        return {"reserve": self.reserve(), "walk": self.walk_left(), "opens": self.opens, "ask_at": self.ask_at, "first": self.first(), "again": self.again()}

    def writes(self) -> int:
        return 1 if (self.store.roles.get("cart_write") or {}).get("mode") == "replace" else max(1, len(self.ledger.lines))

    def view(self) -> int:
        return 1 if (self.store.roles.get("cart_view") or {}).get("tool") else 0

    def first(self) -> int:
        page = (self.store.roles.get("page") or {}).get("tool")
        own = bool(page) and page != (self.store.roles.get("facts") or {}).get("tool")
        unread = lambda l: l["pick"] not in self.store.pages if l.get("pick") else own
        pages = sum(1 for l in self.ledger.lines if unread(l)) + sum(1 for i in self.ledger.chain_pages if i not in self.store.pages) if page else 0
        return self.writes() + pages + self.view() + 1

    def again(self) -> int:
        one = 1 if (self.store.roles.get("cart_write") or {}).get("mode") == "replace" else 2
        owed = self.gate.closer.owed(self.gate.closer.want()) if self.ledger.refusals and self.gate else 0
        return 1 + max(one, owed) + self.view() + 1

    def short(self) -> bool:
        return self.finishing and (not all(l.get("pick") for l in self.ledger.lines) or bool(self.gate and self.gate.over()))

    def reserve(self) -> int:
        left = self.ledger.messages_left or 0
        messages = min(left, (self.gate.kept() if self.gate else 0) + self.deferred) if self.asked else left
        whole = (0 if self.ledger.refusals else self.first()) + self.again() + 1 + messages
        return min(whole, self.writes() + 2 + messages) if self.short() else whole

    def walk_left(self) -> int:
        return max(0, self.env.left() - self.reserve())

    def walk_over(self) -> bool:
        return self.env.left() <= self.reserve() or (self.until is not None and self.env.step >= self.until) or self.walk_late()

    def in_time(self, calls: int, seat: bool = False, walk: bool = False) -> bool:
        env = self.env
        return env.elapsed() + (SEAT_S if seat else 0.0) + calls * env.per_call() < TASK_S - (10 if walk else 5)

    def walk_late(self) -> bool:
        return self.env.elapsed() > WALK_S or not self.in_time(self.reserve() + 1, walk=True)

    def rounds(self, loop: str):
        seen: set = set()
        at, stopped = [self.env.step], []

        def more(*state: Any) -> bool:
            if self.env.step != at[0]:
                at[0] = self.env.step
                seen.clear()
            store, lines = self.store, self.ledger.lines
            held = [len(store.cards), len(store.records), len(store.pages), len(store.notices), self.ledger.messages_left]
            held += [[l.get("pick"), len(l["seen"]), len(l["verdicts"]), sum(1 for got in l["verdicts"].values() for v in got.values() if v[0] == "PASS"), [want_tag(w) for w in self.ledger.wants(n)]] for n, l in enumerate(lines, 1)]
            key = json.dumps([state, held], default=str)
            why = "the walk's time is up" if self.walk_late() else "the round would start from a state this loop already started from, with no store call since" if key in seen else None
            if why and not stopped:
                stopped.append(why)
                trace_row(self.store.trace, "loop_stop", loop=loop, reason=why, step=self.env.step, elapsed=round(self.env.elapsed(), 1), state=key)
            seen.add(key)
            return why is None

        return more

    def within(self, steps: int, work) -> Any:
        before = self.until
        end = self.env.step + max(0, steps)
        self.until = end if before is None else min(before, end)
        try:
            return work()
        finally:
            self.until = before

    def spare(self, calls: int, due: int = 0) -> bool:
        behind = due + 1 + (0 if self.ledger.refusals else self.again() + 1)
        return self.env.left() - calls >= behind and self.in_time(calls + behind)

    def essential(self, calls: int) -> bool:
        return self.env.left() - calls >= 1 and self.in_time(calls + 1)

    def ask_fits(self) -> bool:
        need = self.reserve()
        return self.env.elapsed() < WALK_S and self.env.left() >= need and self.in_time(need + 1, seat=True, walk=True)

    def judge_fits(self) -> bool:
        return self.in_time(self.first() + 1, seat=True)

    def late_fits(self) -> bool:
        need = self.reserve()
        return self.env.left() - 1 >= need and self.in_time(need + 2, seat=True, walk=True)

    def phase_a(self, n: int, lines_left: int) -> tuple[int, int]:
        part = self.walk_left() // max(1, lines_left)
        searches = 2 if part >= self.opens + 2 else 1
        widen = max(0, (self.ask_at - self.env.step) // max(1, lines_left) - searches - self.opens) if self.ask_at else 0
        return searches, max(1, min(6, self.opens + widen, part - searches))

    def share(self, looks: int, opens: int) -> tuple[int, int]:
        part = max(2, self.walk_left() // max(1, looks))
        k = min(opens, part // 2)
        return part - k, k


class Agent:

    def __init__(self, problem_data: dict) -> None:
        self.trace: list = []
        TRACE_T0[0] = time.monotonic()
        self.pace = Pace()
        self.env = Env(problem_data, self.pace, self.trace)
        self.brain = Brain(self.env, self.pace, self.trace)
        self.store = Store(self.env, self.brain, self.trace)
        request = self.env.view.get("query") or ""
        if self.env.view.get("opening") is not None:
            request += "\n\n" + json.dumps(self.env.view["opening"], ensure_ascii=False)
        self.ledger = Ledger(request)
        self.plan = StepPlan(self.env, self.ledger, self.store)
        self.gather = Gather(self.store, self.ledger, self.trace, plan=self.plan)
        self.judge = Judge(self.brain, self.store, self.ledger, self.trace)
        self.picker = Picker(self.brain, self.store, self.ledger, self.judge, self.gather, self.trace)
        self.closer = Closer(self.brain, self.store, self.ledger, self.picker, self.trace)
        self.gate = FinishGate(self.ledger, self.store, self.closer, self.plan, self.trace)
        self.plan.gate = self.gate
        self.asker = Asker(self.brain, self.store, self.ledger, self.judge, self.plan, self.gate, self.trace)
        self.vetoes: dict[int, dict] = {}
        self.judged_at: tuple | None = None
        self.acted: dict[int, int] = {}
        self.stalled: set = set()
        self.settled_actor: set = set()

    def run(self) -> list:
        with ThreadPoolExecutor(max_workers=2) as pool:
            tools = pool.submit(self.store.learn_tools)
            brief = pool.submit(read_request, self.brain, self.ledger.request, self.trace)
            tools.result()
            got = brief.result()
        roles = self.store.roles
        missing = lambda: [r for r, ok in (("search", roles.get("search")), ("order", roles.get("order")), ("open", roles.get("facts") or roles.get("page") or roles.get("compare"))) if not ok]
        if missing():
            trace_row(self.trace, "handoff", reason="tool map without " + " or ".join(missing()), roles=roles)
            self.store.learn_tools("MAP_ALT")
        if not got:
            trace_row(self.trace, "handoff", reason="no brief")
            request = self.ledger.request
            got = self.brain.ask("INTAKE_ALT", INTAKE, "", "REQUEST:\n" + request, {"type": "object"}, lambda a: brief_problem(a, request))
        if not got or {"search", "order"} & set(missing()):
            trace_row(self.trace, "handoff", reason="decline: " + ("no brief" if not got else "tool map without " + " or ".join(missing())), roles=roles)
            self.settle()
            return self.end()
        load_brief(self.ledger, got, self.trace)
        trace_row(self.trace, "step_plan", **self.plan.set())
        lines = len(self.ledger.lines)
        for n in range(1, lines + 1):
            self.look(n, 1, *self.plan.phase_a(n, lines - n + 1))
        self.questions(final=False)
        self.last_looks()
        if not self.plan.asked:
            self.questions()
        return self.finish()

    def questions(self, final: bool = True) -> None:
        if self.ledger.messages_left and self.env.elapsed() < WALK_S:
            self.asker.ask_all(self.regather, final)
        self.rejudge(self.closer.apply_notices())
        if not final and self.asker.waiting():
            trace_row(self.trace, "ask_wait", lines=self.asker.waiting(), messages_left=self.ledger.messages_left, reason="lines with no judged record keep a message each: they are asked after the final looks")
            return
        free = max(0, (self.ledger.messages_left or 0) - self.gate.kept())
        self.asker.deferred = set(sorted(n for n, q in self.asker.queue.items() if not q and not self.ledger.line(n)["told"])[:free])
        self.plan.deferred = len(self.asker.deferred)
        if self.asker.deferred:
            trace_row(self.trace, "ask_defer", lines=sorted(self.asker.deferred), messages_left=self.ledger.messages_left, reason="untold lines with no key while a message is free: each keeps a message for when judged records give it keys")
        self.plan.asked = True

    def last_looks(self) -> None:
        lines = len(self.ledger.lines)
        spare = lambda n: self.passing(n) and self.gather.any_line(self.ledger.line(n))
        order = sorted(range(1, lines + 1), key=lambda n: (self.passing(n), spare(n), n))
        for k, n in enumerate(order):
            line = self.ledger.line(n)
            if spare(n):
                if self.passes(n) < 2:
                    self.look(n, 1, 1, 1)
                continue
            group = [m for m in order[k:] if self.passing(m) == self.passing(n) and not spare(m)]
            pages = 3 if (n in self.gather.capped or not self.gather.any_line(line)) else 2
            self.look(n, pages, *self.plan.share(len(group), 3 if self.gather.any_line(line) else 6))

    def regather(self, m: int) -> None:
        askable = sum(1 for q in self.asker.queue.values() if q)
        others = sum(1 for k in range(1, len(self.ledger.lines) + 1) if k != m and not self.passing(k))
        self.look(m, 2, *self.plan.share(1 + min(self.ledger.messages_left or 0, askable) + others, 4), enough=True)

    def look(self, n: int, pages: int, searches: int, opens: int, enough: bool = False) -> None:
        line = self.ledger.line(n)
        opened = self.store.opened
        self.rejudge(self.closer.apply_notices())
        self.judge.judge(n, list(line["verdicts"]))
        self.picker.choose(n)
        if enough and self.gather.any_line(line) and self.passing(n):
            trace_row(self.trace, "look", line=n, settled="the opened cards pass the new requirement", pick=line.get("pick"), step=self.env.step)
            return
        if self.picker.chain_open(n):
            if self.plan.walk_over():
                trace_row(self.trace, "chain_held", line=n, reason="the walk gate holds the line's chain; JUDGE gets it as unfinished", left=self.env.left(), reserve=self.plan.reserve())
                return
            start = self.env.step
            self.plan.within(searches + opens, lambda: self.follow(n))
            if self.picker.chain_open(n):
                self.picker.choose(n)
                trace_row(self.trace, "chain_paused", line=n, reason="the chain spent the line's part; the line is gathered once it is done", step=self.env.step)
                return
            left = start + searches + opens - self.env.step
            opens = min(opens, max(0, left - 1))
            searches = left - opens
            if searches <= 0:
                self.picker.choose(n)
                return
        if self.plan.walk_over():
            trace_row(self.trace, "look", line=n, held="walk gate", pick=line.get("pick"), step=self.env.step, left=self.env.left(), reserve=self.plan.reserve())
            return
        order = self.gather.run(n, pages, resume=bool(line["verdicts"]) and not self.passing(n), steps=searches)
        short = opens - len([i for i in order if not opened(i)])
        if short > 0:
            order = self.gather.run(n, pages, steps=short)
        fresh = self.gather.first_opens(n, [i for i in order if not opened(i)], self.batch(opens))
        self.picker.open_and_judge(n, fresh)
        self.picker.choose(n)
        if not self.passing(n) or not self.gather.any_line(line):
            self.gather.consult(n)
        trace_row(self.trace, "look", line=n, searches=searches, opens=opens, opened=[i for i in fresh if opened(i)], pick=line.get("pick"), step=self.env.step, walk_left=self.plan.walk_left())

    def follow(self, n: int) -> None:
        for want in [w for w in self.ledger.wants(n) if w.get("op") == "ref" and (not w.get("followed") or w.get("chain_at"))]:
            self.picker.follow_chain(n, want)

    def rejudge(self, lines: set[int]) -> None:
        self.brain.adapting = True
        for n in sorted(lines):
            self.judge.judge(n, list(self.ledger.line(n)["verdicts"]))
            self.picker.choose(n)
        self.brain.adapting = False

    def finish(self) -> list:
        self.plan.finishing = True
        audited, late = False, False
        for attempt in range(6):
            if self.env.finished or self.env.over or self.env.left() <= 0:
                break
            self.rejudge(self.closer.apply_notices())
            for n in range(1, len(self.ledger.lines) + 1):
                self.picker.choose(n)
            self.dig()
            self.picker.fit_total()
            missing = [n for n, l in enumerate(self.ledger.lines, 1) if not l.get("pick")]
            if missing or self.gate.over():
                if missing and not audited:
                    audited = True
                    if self.closer.audit(missing):
                        continue
                if not missing:
                    trace_row(self.trace, "over_cap", total=self.gate.total(), cap=self.ledger.total_cap, rooms=self.rooms(), left=self.env.left(), reason="no combination of the judged cards is within the stated total cap and the dig has nothing more under the rooms: the basket is not ordered, the request's decline rule decides")
                if self.gate.decline(self.asker):
                    break
                self.refused()
                audited = False
                continue
            if self.final_check():
                continue
            if not late and self.gate.free() and self.plan.late_fits():
                late = True
                changed = self.asker.late([n for _, _, n in sorted(d for d in (self.doubt(k) for k in range(1, len(self.ledger.lines) + 1)) if d)])
                if changed:
                    self.rejudge(changed)
                    continue
            elif not late and self.gate.free():
                trace_row(self.trace, "late_skip", reason="the step plan holds the late question", messages=self.gate.free(), left=self.env.left(), reserve=self.plan.reserve())
            changed = self.gate.procedures("order", self.asker)
            if changed:
                self.rejudge(changed)
                continue
            written = {n: l.get("pick") for n, l in enumerate(self.ledger.lines, 1)}
            if self.closer.write_cart() and attempt < 5 and self.moved(written):
                continue
            version = self.picker.basket_version
            for n in self.gate.duties(self.picker):
                self.picker.choose(n)
            if self.picker.basket_version != version and attempt < 5:
                continue
            if self.gate.order() or (self.gate.held and self.keep_bought() and self.gate.order()):
                break
            if self.gate.held:
                break
            self.refused()
        self.settle()
        return self.end()

    def final_check(self) -> bool:
        key = (self.picker.basket_version, tuple(tuple(want_tag(w) for w in self.ledger.wants(n)) for n in range(1, len(self.ledger.lines) + 1)))
        if key == self.judged_at:
            return False
        if not self.plan.judge_fits():
            trace_row(self.trace, "judge_held", reason="the step plan holds the final check: the code-verified basket goes on", unchecked={n: l.get("pick") for n, l in enumerate(self.ledger.lines, 1)}, elapsed=round(self.env.elapsed(), 1))
            self.judged_at = key
            return False
        self.judged_at = key
        verdict = self.closer.judge_basket(self.asker.talk)
        self.gate.point(verdict.get("pointers") or [])
        rejected = any(isinstance(r, dict) and not r.get("ok") for r in verdict.get("lines") or [])
        return self.veto(verdict) or (rejected and not self.plan.walk_over())

    def moved(self, written: dict) -> bool:
        self.rejudge(self.closer.apply_notices())
        if self.closer.heard:
            self.closer.recheck()
        for n in range(1, len(self.ledger.lines) + 1):
            self.picker.choose(n)
        moves = {n: old for n, old in written.items() if old in self.closer.in_cart and self.ledger.line(n).get("pick") not in (None, old)}
        if not moves:
            return False
        forced = [n for n, old in moves.items() if old not in self.picker.ranked(n)]
        fits = lambda: self.plan.essential(self.closer.owed(self.closer.want()))
        if fits() if forced or self.ledger.refusals else self.plan.spare(self.closer.owed(self.closer.want(), whole=True)):
            trace_row(self.trace, "cart_moved", moves={n: (old, self.ledger.line(n)["pick"]) for n, old in moves.items()}, forced=forced, reason="a read or notice after the write moved the picks: the basket is written again")
            return True
        for n, old in moves.items():
            self.picker.choose(n, to=old)
        better = {n: (old, self.ledger.line(n)["pick"]) for n, old in moves.items() if self.ledger.line(n).get("pick") != old}
        if better and fits():
            for n, (_, now) in better.items():
                self.picker.choose(n, to=now)
            trace_row(self.trace, "cart_moved", moves=better, forced=forced, reason="the chooser refused the carted picks' pins (the new ones stand on better proof) and their writes fit before the order")
            return True
        kept = self.keep_bought() if better else True
        trace_row(self.trace, "cart_kept", moves={n: (old, self.ledger.line(n).get("pick")) for n, old in moves.items()}, whole=kept, left=self.env.left(), reason="the swap does not fit before the order: the cart's own basket stands")
        return False

    def keep_bought(self) -> bool:
        claims = {n: l["pick"] for n, l in enumerate(self.ledger.lines, 1) if l.get("pick") in self.closer.in_cart}
        for n in [k for k in range(1, len(self.ledger.lines) + 1) if k not in claims]:
            old = next((i for i in self.closer.in_cart if i not in claims.values() and i in self.picker.ranked(n)), None)
            if old:
                claims[n] = old
        for n, old in sorted(claims.items()):
            self.picker.choose(n, to=old, keep=True)
        want = self.closer.want()
        whole = bool(want) and all(l.get("pick") for l in self.ledger.lines) and all(self.closer.in_cart.get(i) == q for i, q in want.items())
        trace_row(self.trace, "cart_keep", whole=whole, cart=self.closer.in_cart, picks={n: l.get("pick") for n, l in enumerate(self.ledger.lines, 1)}, left=self.env.left())
        return whole

    def veto(self, verdict: dict) -> bool:
        moved = False
        for row in verdict.get("lines") or []:
            n = row["line"]
            if row.get("ok"):
                continue
            line, use, problem = self.ledger.line(n), row.get("use"), str(row.get("problem") or "")
            pick = line.get("pick")
            ours = self.judge.status(n, pick) if pick else None
            take = lambda: bool(use) and self.judge.status(n, use) == "PASS" and self.plan.essential(self.closer.owed(self.closer.want({n: use})))
            taken = ours != "PASS" and take()
            if not taken and pick:
                self.judge.dispute(n, pick, "JUDGE: " + problem)
                self.picker.choose(n)
                taken = line.get("pick") == pick and self.judge.status(n, pick) != "PASS" and take()
            if taken:
                self.picker.choose(n, to=use)
            self.vetoes[n] = {"pick": pick, "problem": problem, "use": use, "taken": bool(taken), "basket": self.picker.basket_version, "evidence": version(self.store.cards[pick]) if pick else None, "ours": ours}
            moved |= line.get("pick") != pick
            trace_row(self.trace, "veto", line=n, pick=pick, ours=ours, problem=problem, use=use, taken=bool(taken), now=line.get("pick"), basket=self.picker.basket_version)
        return moved

    def refused(self) -> None:
        changed = self.closer.apply_notices()
        self.ledger.refusals += 1
        moved = self.closer.recheck()
        for n, line in enumerate(self.ledger.lines, 1):
            if n in changed | moved:
                self.judge.judge(n, list(line["verdicts"]))
            if n in changed | moved or self.gather.cheapest(line):
                line.pop("pinned", None)
            if line.get("pick") in self.closer.in_cart and not self.gather.cheapest(line) and line["pick"] in self.picker.ranked(n):
                self.picker.choose(n, to=line["pick"])
        trace_row(self.trace, "refused", refusals=self.ledger.refusals, changed=sorted(changed), moved=sorted(moved), cart=self.closer.in_cart, left=self.env.left())

    def batch(self, steps: int) -> int:
        compare = self.store.roles.get("compare") or {}
        return steps * int(compare.get("most") or 1) if compare.get("tool") and compare.get("facts") is True else steps

    def unit(self) -> int:
        return self.batch(1) if self.batch(1) > 1 else 3

    def passing(self, n: int) -> bool:
        pick = self.ledger.line(n).get("pick")
        return bool(pick) and self.judge.status(n, pick) == "PASS"

    def passes(self, n: int) -> int:
        return sum(1 for i in self.picker.ranked(n) if self.judge.status(n, i) == "PASS")

    def doubt(self, n: int) -> tuple | None:
        line = self.ledger.line(n)
        pick = line.get("pick")
        vetoed = n in self.vetoes and self.vetoes[n]["pick"] == pick and not self.vetoes[n]["taken"]
        passes = self.passes(n)
        signs = (not pick, bool(pick) and (self.judge.status(n, pick) in ("FAIL", "UNKNOWN") or vetoed or bool(self.judge.unit_gap(n, pick))), self.picker.fit_doubt(n), passes < 2, not self.gather.covered(n), self.gate.over())
        cls = next((k for k, hit in enumerate(signs) if hit), None)
        return None if cls is None else (cls, passes, n)

    def ask_deferred(self) -> None:
        for n in sorted(self.asker.deferred):
            if (self.ledger.messages_left or 0) <= self.gate.kept() or not self.asker.room():
                return
            self.asker.refill()
            if not self.asker.queue.get(n):
                continue
            self.asker.deferred.discard(n)
            self.plan.deferred = len(self.asker.deferred)
            trace_row(self.trace, "ask_deferred", line=n, keys=self.asker.queue[n][:3], step=self.env.step)
            self.asker.ask(self.asker.message(n, self.asker.question(n, self.asker.queue[n][0])), self.regather)

    def dig(self) -> None:
        spent: set = set()
        unit, calls = self.unit(), 3
        more = self.plan.rounds("Agent.dig")
        rooms: dict = {}
        while not self.plan.walk_over():
            self.ask_deferred()
            rooms = self.rooms()
            due = sorted(d for n in range(1, len(self.ledger.lines) + 1) if n not in spent for d in [self.doubt(n)] if d)
            if not due or not more(due, sorted(spent), sorted(self.stalled), self.acted, sorted(self.settled_actor), [n for n, v in sorted(self.vetoes.items()) if v.get("acted")]):
                break
            if due[0][0] == 0 and due[-1][0] > 2:
                trace_row(self.trace, "dig_wait", waiting=[n for c, _, n in due if c > 2], short=[n for c, _, n in due if c == 0], step=self.env.step, reason="a line without a pick remains in the dig: only doubt classes 0-2 take a unit")
                due = [d for d in due if d[0] <= 2]
            for cls, _, n in due:
                if self.plan.walk_over():
                    break
                line = self.ledger.line(n)
                if self.picker.chain_open(n):
                    start = self.env.step
                    self.plan.within(calls, lambda: self.follow(n))
                    if self.env.step == start and self.picker.chain_open(n):
                        spent.add(n)
                    self.picker.choose(n)
                    trace_row(self.trace, "dig", line=n, doubt=cls, chain=True, open=self.picker.chain_open(n), pick=line.get("pick"))
                    continue
                veto = self.vetoes.get(n) or {}
                if veto and veto["pick"] == line.get("pick") and not veto["taken"] and not veto.get("acted"):
                    veto["acted"] = True
                    if self.act(n, f"the final check rejected the pick {veto['pick']} without naming another record it was shown: {veto['problem']}"):
                        continue
                if n in self.stalled:
                    self.stalled.discard(n)
                    if self.act(n, "the last round of opening brought the line no new passing record"):
                        continue
                before = self.passes(n)
                price = lambda i: price_of(self.store.cards[i], 0) * line["qty"]
                fresh = [i for i in self.gather.ordered(n) if not self.store.opened(i)]
                fresh = sorted(fresh, key=lambda i: (0, 0) if price(i) <= rooms[n] else (1, price(i)))[:unit] if n in rooms else fresh[:unit]
                if not fresh:
                    start = self.env.step
                    fresh = (self.plan.within(calls, lambda: self.gather.resume(n)) or [])[:unit]
                    if not fresh and self.env.step - start >= calls and not self.plan.walk_over():
                        trace_row(self.trace, "dig_bound", line=n, doubt=cls, calls=self.env.step - start, step=self.env.step, reason="the unit's store calls were spent before a rung brought a new row: no ladder end, the next unit goes on")
                        continue
                if not fresh:
                    if not self.act(n, "the code's ladder of searches is spent (Gather.resume gave nothing new)"):
                        spent.add(n)
                    continue
                self.picker.open_and_judge(n, fresh)
                self.picker.choose(n)
                if not any(self.store.opened(i) for i in fresh):
                    if not self.act(n, f"the rows opened ({', '.join(fresh)}) could not be read"):
                        spent.add(n)
                elif self.passes(n) <= before:
                    self.stalled.add(n)
                trace_row(self.trace, "dig", line=n, doubt=cls, opened=[i for i in fresh if self.store.opened(i)], pick=line.get("pick"), passing=self.passing(n), cover=self.gather.covered(n), room=rooms.get(n))
        trace_row(self.trace, "dig_end", doubts={n: self.doubt(n) for n in range(1, len(self.ledger.lines) + 1)}, spent=sorted(spent), rooms=rooms, walk_over=self.plan.walk_over(), left=self.env.left(), reserve=self.plan.reserve())

    def rooms(self) -> dict[int, float]:
        if not self.gate.over():
            return {}
        self.picker.fit_total()
        if not self.gate.over():
            return {}
        least = {n: self.gather.cheapest_pass(n) * line["qty"] for n, line in enumerate(self.ledger.lines, 1)}
        return {n: self.ledger.total_cap - sum(v for m, v in least.items() if m != n) for n in least}

    def tool_map(self) -> str:
        rows = {}
        for role in READ_ROLES:
            tool = (self.store.roles.get(role) or {}).get("tool")
            if tool in self.store.tools:
                fn = self.store.tools[tool].get("function", self.store.tools[tool])
                rows.setdefault(tool, [fn.get("description", ""), json.dumps(fn.get("parameters") or {}, sort_keys=True), []])[2].append(role)
        return "\n".join(f"- {tool} (roles: {', '.join(r[2])}): {r[0]} | schema: {r[1]}" for tool, r in rows.items())

    def act(self, n: int, why: str) -> bool:
        line = self.ledger.line(n)
        if n in self.settled_actor or self.plan.walk_over():
            return False
        if self.acted.get(n, 0) >= 3 or sum(self.acted.values()) >= 6:
            trace_row(self.trace, "actor_cap", line=n, why=why, calls=dict(self.acted), doubt=self.doubt(n), reason="the actor's calls for this line or task are spent: the line stays in doubt")
            return False
        self.acted[n] = self.acted.get(n, 0) + 1
        wants = self.ledger.wants(n)
        count = lambda w, label: sum(1 for i in line["verdicts"] if self.judge.label(n, i, w) == label)
        reqs = "\n".join(f"  {k}. {want_text(w)} | pass {count(w, 'PASS')}, fail {count(w, 'FAIL')}, unproven {len(line['verdicts']) - count(w, 'PASS') - count(w, 'FAIL')}" for k, w in enumerate(wants, 1))
        listing = self.gather.listing(n)
        cursors = self.gather.mine(n) + [c for c in [self.gather.cursor_of("", listing, "browse") if listing else None] if c]
        searches = "\n".join(f"- {c['role']} {c['text']!r} {json.dumps(c['extra'])} rows {c['rows']} pages {c['page']} {'ended' if c['ended'] else 'open'}" for c in cursors) or "(none)"
        best = (self.picker.ranked(n) + [i for i in line["verdicts"] if i not in self.picker.ranked(n)])[:5]
        cards = "\n".join(card_view(self.store.cards[i]) + "\n  VERDICTS: " + "; ".join(f"{k} {self.judge.label(n, i, w)}" for k, w in enumerate(wants, 1)) for i in best) or "(none)"
        aside = "\n".join(f"- {json.dumps(x, ensure_ascii=False, default=str)}" for x in self.gather.excluded(n)) or "(none)"
        recent = "\n".join(f"- step {r['step']} {r['tool']} {json.dumps(r['args'], ensure_ascii=False)} error {r['error']}: {cut_text(r['result'], 600)}" for r in self.env.recent[-4:]) or "(none)"
        relations = "\n".join(f"- {r.get('type')}: {r.get('quote')!r} ({r.get('use')})" for r in self.ledger.relations) + "".join(f"\n- not acted on: {json.dumps(u, ensure_ascii=False, default=str)}" for u in self.ledger.unhandled)
        left = f"walk steps {self.plan.walk_left()}, finish reserve {self.plan.reserve()}, money spent {round(self.brain.spent, 4)} of the walk money {WALK_MONEY}"
        now = (f"WHY: {why}\n\nLINE {n}: {line['qty']} x {line['item']} ({line['objective']}) cap {self.gather.line_cap(line)}\n{reqs}\n\nSEARCHES:\n{searches}\n\nCANDIDATES:\n{cards}\n\nSET ASIDE:\n{aside}\n\nRECENT:\n{recent}\n\n"
               f"RELATIONS:\n{relations.strip() or '(none)'}\n\nLEFT: {left}")
        answer = self.brain.ask("ACTOR", ACTOR, "TOOLS:\n" + self.tool_map(), now, ACTOR_SCHEMA) or {}
        actions = (answer.get("actions") if isinstance(answer.get("actions"), list) else [])[:2]
        trace_row(self.trace, "actor", line=n, why=why, answer=answer, answered=bool(answer))
        if answer.get("stop") or not actions:
            self.settled_actor.add(n)
            return False
        went = False
        for k, action in enumerate(actions):
            if self.plan.walk_over() or (k and self.doubt(n) is None):
                break
            call = self.actor_call(n, action)
            if isinstance(call, str):
                trace_row(self.trace, "actor_refused", line=n, action=action, reason=call)
                continue
            went = self.actor_run(n, *call, str(action.get("why") or "")) or went
        self.picker.choose(n)
        return went

    def actor_call(self, n: int, action: Any) -> tuple | str:
        if not isinstance(action, dict):
            return "not one call object"
        tool = str(action.get("tool") or "")
        roles = [r for r in READ_ROLES if (self.store.roles.get(r) or {}).get("tool") == tool]
        if not roles:
            return f"{tool!r} is no read tool of the map"
        args = self.store.fit_args(tool, {str(a.get("name")): json_value(a.get("value")) for a in action.get("args") or [] if isinstance(a, dict)})
        need = (self.store.tools[tool].get("function", self.store.tools[tool]).get("parameters") or {}).get("required") or []
        role = next((r for r in roles if all(args.get(k) == v for k, v in ((self.store.roles[r].get("fixed") or {}).items()))), roles[0])
        spec, line = self.store.roles[role], self.ledger.line(n)
        ids = [args.get(spec.get("id_arg") or "")] if role in ("facts", "page") else args.get(spec.get("ids_arg") or "") if role == "compare" else []
        if [k for k in need if k not in args]:
            return f"required arguments missing after fitting: {[k for k in need if k not in args]}"
        if role in ("facts", "page", "compare") and not (isinstance(ids, list) and ids and all(isinstance(i, str) and i in self.store.cards and (i in line["seen"] or i in line["verdicts"]) for i in ids)):
            return f"ids that are no rows of line {n}: {ids}"
        if self.store.seen_call(tool, args):
            return "a call already made with the same arguments"
        return role, tool, args

    def actor_run(self, n: int, role: str, tool: str, args: dict, why: str) -> bool:
        step = self.env.step
        spec = self.store.roles[role]
        line = self.ledger.line(n)
        named = [args[spec.get("id_arg") or "id"]] if role in ("facts", "page") else list(args[spec["ids_arg"]]) if role == "compare" else []
        filed = [w for ident in named if not self.gather.keep(line, self.store.cards[ident]) for w in self.gather.admit(n, ident, f"ACTOR named it: {why}")]
        if role in ("search", "browse"):
            before = set(line["seen"])
            extra = {k: v for k, v in args.items() if k not in (spec.get("query_arg"), spec.get("page_arg"))}
            self.gather.page(n, str(args.get(spec.get("query_arg") or "") or ""), "widen", extra or None, role, bound=False)
            fresh = [i for i in self.gather.ordered(n) if i not in before and not self.store.opened(i)][: self.unit()]
            if fresh and not self.plan.walk_over():
                self.picker.open_and_judge(n, fresh)
        elif role == "page":
            self.picker.open_page(named[0])
            self.judge.judge(n, named)
        elif role == "facts":
            self.store.read_card(named[0])
            self.judge.judge(n, named)
        else:
            self.picker.open_and_judge(n, named)
        if filed:
            self.judge.judge(n, list(line["verdicts"]))
        trace_row(self.trace, "actor_ran", line=n, role=role, tool=tool, args=args, steps=self.env.step - step, seen=len(line["seen"]), passes=self.passes(n), pick=line.get("pick"))
        return self.env.step > step

    def settle(self) -> None:
        if self.env.finished or self.env.over or self.env.left() <= 0:
            return
        self.rejudge(self.closer.apply_notices())
        if self.ledger.lines and all(l.get("pick") for l in self.ledger.lines):
            if self.closer.in_cart != self.closer.want() and not (self.closer.write_cart() and self.closer.in_cart == self.closer.want()):
                self.keep_bought()
            self.gate.order()
        if not self.env.finished and not self.gate.decline(self.asker):
            trace_row(self.trace, "handoff", reason="no finish call ended the episode", roles=self.store.roles)

    def end(self) -> list:
        trace_row(self.trace, "end", finished=self.env.finished, steps=self.env.step, spent=round(self.brain.spent, 4), kept=round(self.brain.kept, 4), state=self.ledger.state(self.env), actor=self.acted, vetoes=self.vetoes, talk=len(self.asker.talk))
        return self.trace

    def rescue(self, exc: Exception) -> list:
        self.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        for close in ([self.questions] if self.ledger.lines and not self.plan.asked else []) + ([self.finish] if self.ledger.lines else []) + [self.settle]:
            if self.env.finished or self.env.over or self.env.left() <= 0:
                break
            try:
                close()
            except Exception as again:
                self.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
            self.plan.asked = True
        return self.trace


def json_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip()[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def agent_main(problem_data: dict) -> list:
    agent = None
    try:
        agent = Agent(problem_data)
        return agent.run()
    except Exception as exc:
        if agent is None:
            return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
        try:
            return agent.rescue(exc)
        except Exception:
            return agent.trace
