from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections import Counter
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

HAIKU, SONNET, LUNA = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"
LADDER, LADDERS = (HAIKU, HAIKU, SONNET, LUNA), {"read": (SONNET, SONNET, HAIKU, LUNA), "ask": (SONNET, SONNET, HAIKU, LUNA)}
EFFORT = {"read": "low", "map": "none", "assistant": "none", "ask": "low", "hear": "none", "fit": "low", "link": "low", "ids": "none", "audit": "medium", "qty": "low"}
CALL_TIMEOUT, CALL_MIN, RUNG_TRIES, BACKOFF = 45.0, 6.0, 2, 1.0
DEADLINE, DEADLINE_MARGIN, SAFE_LEFT, P3_LEFT = 265.0, 10.0, 60.0, 90.0
SPEND_SONNET, SPEND_CAP = 0.05, 0.055
# the spend at which the episode closes, and the hard ceiling for the seats that must outlive SPEND_CAP
SPEND_CLOSE, SPEND_HARD, SPEND_EXEMPT = 0.045, 0.075, ("hear",)
PACE, PACE_MAX, PACE_EASE, REPLAY_MAX, WORLD_TIMEOUT = 0.15, 2.0, 0.95, 5, 30.0
RESERVE_ORDER, RESERVE_RECOVER, RESERVE_MESSAGE = 1, 3, 1
CALLS_PLAN = {"S": 11, "B": 15, "A": 22}
# questions: how many of a line's properties one message names, and how many planned per message the planner is asked for
WINDOW, PLAN_DEPTH = 5, 3
# how closely (shared words over all words) a shelf's name must fit the request's names for a line to rule out the other
# shelves of the same product
SHELF_CLOSE = 0.75
# a record's field naming whether a listing can be bought, and one naming its price
STOCK_KEY = re.compile(r"stock|avail|on.?hand|in.?store|ready", re.I)
PRICE_KEY = re.compile(r"price|cost", re.I)
# the name of a free-text argument that reads as a search's query
QUERY_NAME = re.compile(r"search|phrase|query|keyword|term|text|words?|find|look|^q$", re.I)
# a fixed-choice value of a search tool that reads as searching by words
WORD_MODE = re.compile(r"phrase|text|word|keyword|query|search|free|match", re.I)
# a request that warns that what was picked may change before the finish
RECHECK_RULE = re.compile(r"keep checking|can change|may change|might change|check (?:again|back)|re-?check", re.I)
# words of a 403 that says the run's inference key is used up rather than that a model is not allowed
KEY_SPENT = re.compile(r"key limit|limit exceeded|insufficient credit|requires more credit|out of credit", re.I)
# a stated property's name that is one generic word: it keeps only that very name out of the questions
GENERIC_HELD = ("type", "size", "style", "kind")
# the roles of the sentences that order no quantity, and the words of the request's own limits
NOT_COUNT_ROLES = ("cap", "budget", "goal", "procedure", "rule")
LIMIT_WORDS = re.compile(r"\b(?:messages?|moves?|steps?|turns?)\b", re.I)
# a need on stock rather than on the product (the store tells stock, and every pick is checked)
STOCK_NEED = re.compile(r"\b(?:availab\w*|in stock|out of stock|stock|on hand)\b", re.I)
# a field a store keeps for its own bookkeeping, not a property a shopper chooses by
NOT_A_PROPERTY = re.compile(r"\b(?:options?|variants?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|urls?|links?|images?|photos?)\b", re.I)
STEP_SECONDS, CALL_SECONDS, TIMING_SAMPLES = 1.6, 3.4, 3
BLOCKS_MAX, QUERIES_SHOWN, BLOCKED_SHOWN = 2, 8, 4
VALUES_PER_LINE, VALUE_WORDS, WIDE_ROWS, ORDER_ROWS, FREE_STEPS, QUERY_WORDS, PROBES, PROBE_ROWS = 3, 4, 6, 3, 3, 3, 5, 12
COVER_ROWS, COVER_PLAIN = 40, 8
FIRST_CARDS, QUESTIONS_UNSTATED, MENU_KEYS, MENU_EXAMPLES, TEXT_WORDS, KIND_WORD = {1: 6, 2: 4}, 3, 40, 3, 8, 3
JUDGE_CHARS, COMBINE_MAX, BUNDLE_DEPTH, DEFER_TOP = 1500, 40, 0.8, 3
AGENDA_TURNS, ESSENTIAL_QUERIES, REF_TURN_STEPS, PURSUE_READS, PICK_READS, REF_PAGES, CHASE_STALLS = 90, 2, 3, 4, 2, 4, 2
CHAIN_CARDS, STORE_WORDS = 5, 6
# copies of a lowest-price pick read as stand-ins, should it sell out at checkout
STAND_INS = 2
GOALS_CHEAPEST, VALUE_RELATIONS = ("cheapest", "lowest_total"), ("equal", "contains", "fits", "at least", "at most", "between")
HEARD_KINDS = ("disclose", "no_need", "not_understood", "one_at_a_time", "ack", "instruction", "budget", "price", "drop", "stock", "limit", "other")
CELL_CHARS, CARD_CHARS, TABLE_ROWS, LAST_CHARS = 160, 600, 20, 6000
RUNTIME_ERRORS = ("unknown_tool", "invalid_arguments", "budget_exceeded")
ROLES = ("search", "browse", "read", "compare", "cart_write", "cart_view", "message", "meter", "order", "decline", "finish", "other")
MEANINGS = ("query", "page", "shelf", "brand", "price_min", "price_max", "sort", "id", "ids", "quantity", "mode", "text", "other")
VALUE_MEANINGS = ("add", "remove", "set", "order", "decline", "words", "filter", "cheapest_first", "other")
SYSTEM_ROLES = ("cart_write", "message", "order", "decline", "finish")
EXPECTED = {"search": "rows", "browse": "rows", "read": "record", "compare": "rows", "message": "text", "order": "end", "decline": "end", "finish": "end"}
READ_TAGS = ("LINE", "NEED", "SENTENCE", "CAP", "GOAL", "BUDGET", "DUTY", "DECLINE", "REF", "DEPENDS")
MAP_TAGS = ("TOOL", "ARG", "VALUE")
ASK_TAGS, HEAR_TAGS, FIT_TAGS = ("ASK",), ("HEARD",), ("FIT",)
LINK_TAGS, IDS_TAGS, AUDIT_TAGS, QTY_TAGS, UNOPENED_SHOWN = ("TARGET",), ("IDS",), ("AUDIT",), ("QTY",), 15


READ_PROMPT = """You turn one shopper's request into a work order that a store assistant program follows.

Inputs
<request>: the shopper's words, one numbered sentence per row, exactly as written.
<tools>: the store's tool names with their one-line descriptions, so you can tell which words of the request name a tool.

Steps
1. Find each separate product the shopper wants to buy. Each one is a line, numbered from 1. Words that only describe a product (its use, its season, who it is for) stay inside that line.
2. For each line write its kind: the shortest name the shopper gives the product, one to three words with no need inside it; then any other names the shopper gives the same product, the name a store's shelf for it would carry, and the quantity to buy in this order with the words that set it. Numbers that tell the story (people in a household, things owned before, needs to come later) are not quantities: when the request says how many to order now, that number is the quantity; a pair or a set counts as 1.
3. For each line list every need it states: the attribute named, the relation the value must have (equal, contains, at_least, at_most, between, not, fits), the value spelled exactly as the shopper spelled it, and the shortest quote that states it. Price caps, stock or availability in any wording, quantity and budgets are not needs, and neither is a sentence that only suggests something. A need stays a need when the shopper adds that they will check it themselves or that the program need not worry about it. A product named by the name of a shelf or category ("spring/summer clothing", "sports & outdoor watches") needs that category: write it as a need on the category, not on another attribute. A need that allows several values ("leather or canvas") is one need whose value lists them all, never one need per value.
4. Give every sentence one role: product (names a product to buy), need (states a value for a product named elsewhere), cap, goal, budget, procedure (something the program must do or say at some moment), rule (what to skip, or what to do when nothing fits), reference (a value to take from another listing), compat (one product must fit another), clue (tells how a product will be used, where it goes, how it should look, feel or sound, a measurement taken or a comparison, and so hints at an attribute of it that no need states), invite (asks the program to ask the shopper), story (background that hints at no attribute of a product). When a sentence does several jobs, it is invite if it invites, else clue if it hints at an unstated attribute, else its main job. A clue whose attribute you cannot name is story. For a clue or an invitation write the line it speaks about (all when general) and the attribute it most likely points at (none when an invitation names none).
5. Write the price cap (on the entire order or on one line), the goal of each line or of the whole order, the message and step budgets the request states, every procedure the shopper asks for (what, when, by which means) and the rule for when nothing fits.
6. A need whose value is on another listing, named by its store, its kind or a fact rather than by an id, is a reference: write the line and attribute it fills, the store, kind and fact that point at that listing, and the relation. A line that has to fit another line depends on it.

Output
Think first if it helps. Then end with rows only, one item per row, each starting with its tag: a SENTENCE row for every sentence, the other rows only where they have something to say.
LINE <n> | kind: <words> | also: <other names, comma separated, or none> | shelf: <the name a store's shelf or category for it would carry> | qty: <number> | qty_quote: <the words that set how many to order now, or none>
NEED <n> | key: <attribute> | relation: <relation> | value: <value> | quote: <quote>
SENTENCE <number> | role: <role> | line: <n, all or none> | key: <for a clue or an invitation, the attribute it points at, else none>
CAP <total or n> | amount: <number> | unit: <unit words>
GOAL <all or n> | goal: <cheapest, lowest_total, highest, lowest or none> | key: <attribute for highest or lowest, else none>
BUDGET | messages: <number or none> | steps: <number or none>
DUTY | when: <before_finish, on_cart_change, before_swap or now> | by: <message or tool> | text: <quote>
DECLINE | text: <quote>
REF <n> | key: <attribute> | store: <name> | kind: <words> | fact: <the values that set that listing apart, without attribute names> | relation: <relation>
DEPENDS <n> | on: <n>

Example
LINE 1 | kind: tote bag | also: shopper | shelf: Tote Bags | qty: 2 | qty_quote: "two of the canvas totes"
NEED 1 | key: material | relation: equal | value: canvas | quote: "a canvas one, please"
NEED 1 | key: strap length | relation: at_least | value: 60 cm | quote: "straps of at least 60 cm"
SENTENCE 1 | role: product | line: 1 | key: none
SENTENCE 2 | role: story | line: none | key: none
SENTENCE 3 | role: clue | line: 1 | key: closure
CAP total | amount: 900 | unit: tokens
GOAL all | goal: lowest_total | key: none
BUDGET | messages: 2 | steps: 25
DUTY | when: before_finish | by: message | text: "say what it all costs before paying"
DECLINE | text: "if none of them works out, tell me plainly\""""

MAP_PROMPT = """You label the tools of an online store so that a program can call them.

Inputs
<tools>: each tool with its name, its description and its argument paths (type, allowed values, limits, whether required). A path such as entries[].ref is the field ref inside each element of the list entries.

Steps
1. Give each tool one role: search (finds listings from words), browse (lists a shelf or a filter without words), read (shows one listing in detail), compare (shows several listings at once), cart_write (changes the cart), cart_view (shows the cart), message (sends words to the shopper), meter (shows a budget or a count), order (buys the cart and ends the task), decline (ends the task without buying), finish (ends the task, and an argument chooses buying or not), other.
2. Give each argument path one meaning: query, page, shelf, brand, price_min, price_max, sort, id, ids, quantity, mode, text, other.
3. For an argument whose allowed values change what the call does, give each value one meaning: add, remove, set, order, decline, words (the call searches by words), filter (the call lists by shelf, brand or price without words), cheapest_first (rows come cheapest first), other.
4. Copy any number limit that a description states, such as how many ids one call takes.

Output
Rows only: one per tool, one per argument path, one per allowed value that changes the call.
TOOL <name> | role: <role> | limit: <stated limit or none>
ARG <name>.<path> | means: <meaning>
VALUE <name>.<path>=<value> | means: <meaning>

Example
TOOL lookup_things | role: search | limit: none
ARG lookup_things.words | means: query
ARG lookup_things.cursor | means: page
ARG lookup_things.order | means: sort
VALUE lookup_things.order=low_to_high | means: cheapest_first
TOOL basket_edit | role: cart_write | limit: none
ARG basket_edit.entries[].ref | means: id
ARG basket_edit.entries[].count | means: quantity
ARG basket_edit.action | means: mode
VALUE basket_edit.action=put | means: add
VALUE basket_edit.action=take | means: remove"""

ASSISTANT_PROMPT = """You are the hands of a store assistant program. Each turn you make exactly one call to one of the store's tools: the call that moves this turn's goal forward the most.

Inputs
<request>: the shopper's words.
<state>: what is known so far: the lines to buy with their needs, the budgets used, the searches made, the listings seen and opened, what results have confirmed, and calls the system did not send, with the reason.
<last>: the result of the previous call, if there was one.
<goal>: what this turn has to achieve.

Steps
1. Read the goal and the state. Never repeat a call the state lists as made.
2. Choose the tool and the arguments that reach the goal in the fewest steps: words taken from the request for a search, ids exactly as a result printed them for a read.
3. The system itself writes the cart, messages the shopper and makes the finishing call. Call those tools only when the goal asks for that action.

Output
One tool call and nothing else. Example: for the goal "line 1 (tote bag): the system's searches showed no listing for it; search once with other words the request gives for this product", a search call whose words are "shopper bag"."""

ASK_PROMPT = """You choose the questions a store assistant program puts to a shopper whose request leaves some needs unsaid. A need the shopper has not stated is disclosed only when a question names its attribute.

Inputs
<request>: the shopper's words.
<lines>: for each product to buy: the needs already stated, and the clue sentences of the request that speak about it (each suggests something without stating it, or invites questions), with the attribute the reader guessed.
<menus>: for each line, the attributes carried by the listings opened for it, as the store spells them, with how many listings carry each, up to three values, and how much the values differ (0 all the same, 1 all different).
<count>: how many questions may be sent.

Steps
1. For each clue sentence, decide which attribute of that line's menu it points at: think about what the sentence says about the product's use, look, feel, sound, place or size, not about the words it shares with an attribute. A clue points at exactly one attribute; an invitation points at the attribute it names or describes.
2. Each attribute a clue points at gets a question of its own, the clearest clues first.
3. With the rows left, cover each line's other deciding attributes: what buyers of that kind usually settle and what differs among the listings opened, two attributes of one line in a row, best first. Give each line one row before any line gets a second.
4. Never choose an attribute already stated, a label naming a variant or a bundle, or a field of free text, and never the same attribute twice for one line.

Output
Think first. Then end with rows only, no more than <count>, ordered as they should be sent.
ASK <line> | keys: <attribute>, <attribute> | why: <clue or usual> | quote: <the clue sentence, or none>

Example
ASK 1 | keys: closure | why: clue | quote: "my keys keep slipping out of open bags"
ASK 2 | keys: lining, colour | why: usual | quote: none"""

HEAR_PROMPT = """You read what a shopper or a store said to a store assistant program and write down what it changes.

Inputs
<sent>: the program's last message to the shopper, when the words below answer it.
<said>: the words to read.
<lines>: the products being bought, with their needs so far and the listing picked for each, if any.

Steps
1. Split <said> into the separate things it says.
2. Give each one a kind: disclose (the shopper states what they need for an attribute), no_need (any value will do, no preference), not_understood (the shopper asks to have it said again or more plainly), one_at_a_time (the shopper wants one question per message), ack (the shopper only acknowledges), instruction (the shopper asks the program to do something), budget (the money the shopper can spend changes), price (a listing's price changed), drop (the shopper no longer needs something, or lets go of a need stated before), stock (a listing is no longer available), limit (the number of messages or steps allowed changes), other.
3. For disclose write the line, the attribute, the relation (equal, contains, at_least, at_most, between, not, fits) and the value in the shopper's spelling. When the value is on another listing, named by its store, its kind or a fact, write those instead of a value.
4. For budget and limit write the new number as written; for instruction and drop write the words they rest on (for a let-go need, its line and attribute too); for stock and price write the listing's id as the value when the words or the picks in <lines> say which listing it is, and none when they do not.

Output
Rows only, one per thing said.
HEARD <kind> | line: <n or none> | key: <attribute or none> | relation: <relation or none> | value: <value or none> | store: <store or none> | of: <kind of listing or none> | fact: <fact or none> | amount: <number or none> | text: <the words it rests on>

Example
HEARD disclose | line: 1 | key: closure | relation: equal | value: zip | store: none | of: none | fact: none | amount: none | text: "a zip, so nothing falls out"
HEARD budget | line: none | key: none | relation: none | value: none | store: none | of: none | fact: none | amount: 450 | text: "I can now spend 450 in all\""""

LINK_PROMPT = """You follow pointers between listings for a store assistant program. A need can name another listing instead of a value, and a seller's note can send an attribute on to yet another listing.

Inputs
<need>: the attribute wanted, the relation the shopper asked for between that listing's value and the one to buy, and a description of where the value is held: a listing's store, its kind and a fact about it, or the listing itself.
<chain>: the listings already followed, in order.
<listings>: the listings found for the description, with everything the store showed about them.
<problem>: what was wrong with your last answer, if anything.

Steps
1. Find the listing the description points at: its store, its kind and the fact must all match. Near copies from the same store differ in the fact; a note saying that another version exists elsewhere is not a pointer. When none of the listings matches, write none.
2. Read that listing for the attribute: its own entry, its title, its description and its seller's note.
3. If a note sends the attribute to another listing, write that listing's store, kind and fact. Otherwise write the value the listing states.
4. Then write the condition a listing to buy must meet under the relation asked: for equal, the value itself; for not, the value, which the listing to buy must not have; for at_least or at_most, the value as the bound; for fits, the value this listing's own words say it goes with, or its own value when its words name none.

Output
One row only.
TARGET <listing id or none> | value: <value or none> | condition: <value a listing to buy must have, or none> | as: <equal, not, at_least, at_most or contains> | store: <next store or none> | of: <next kind or none> | fact: <next fact or none>

Examples
TARGET q-17 | value: none | condition: none | as: none | store: Hollowmere | of: lantern | fact: 300 lumen
TARGET q-9 | value: brass | condition: oak | as: equal | store: none | of: none | fact: none"""

QTY_PROMPT = """A store assistant program asks you how many of each product to buy in this order.

Inputs
<request>: what the shopper wrote, one numbered sentence per row.
<lines>: the products whose quantity may come from the wrong words of the request, each with the quantity read so far and the words it came from.

Steps
1. Look for the words that set how many of the product this order buys. A count of the people who will use it, or of what will be needed later, is no order quantity.
2. Give that number, and the words that set it, copied exactly as the shopper wrote them. When no words of the request set how many to buy, write qty: 1 and quote: none.

Output
Rows only, one per product.
QTY <line> | qty: <number> | quote: <the words>

Example
QTY 2 | qty: 4 | quote: "four of the blue ones, please\""""

IDS_PROMPT = """A store assistant program needs to know which parts go with which products.

Inputs
<listings>: product listings and the listings of the parts that have to fit them, with all that the store showed for each.

Steps
1. For every listing, collect the identifiers that link a part to the products it fits: model numbers, series names, part codes, and the products a part says it fits.
2. Spell them so that a part and a product that go together have at least one identifier written identically.

Output
Rows only, one per listing.
IDS <listing id> | ids: <identifiers, comma separated, or none>

Example
IDS q-40 | ids: rk-7, rk series
IDS q-52 | ids: rk-7"""

AUDIT_PROMPT = """You check a store assistant program's decision to buy nothing, before it is carried out.

Inputs
<request>: the shopper's words.
<lines>: each product with its needs.
<evidence>: for each product the program could not fill: its searches and whether they reached their end, the listings opened with the verdict on each need, and listings its searches showed that were never opened.

Steps
1. For each such product look for a listing that could still meet every need: an opened one whose failing verdict looks wrong, or an unopened one whose row suggests it fits.
2. Decide: decline_ok when some product has no such listing, so the order cannot be filled; order_instead when every product has a listing worth buying or opening.

Output
Think first if it helps. Then end with one row.
AUDIT <decline_ok or order_instead> | candidates: <opened listing ids that could fit, or none> | unopened: <listing ids worth opening, or none>

Example
AUDIT order_instead | candidates: none | unopened: q-17, q-31"""

FIT_PROMPT = """You check, need by need, whether listings meet a shopper's needs for one product.

Inputs
<product>: the product and its needs, numbered. Need 0 is that the listing is this product at all.
<listings>: every listing read for this product: its title and shelf; for each need, the entries the store wrote under that need's attribute (or "not listed"); its texts (descriptions, seller's notes); and its other entries.

Steps
1. Need 0: the listing's shelf decides when it names the product; else its title and its entries for the item's type or style do. They meet it when they hold the product's words in any order, together or apart, with other words between or around them (a "Rain Hooded Jacket Coat" is a rain coat, and so is a coat whose style entry is "rain"); they fail it when they name another product.
2. Every other need is decided by the entries shown beside it, the store's own entries for that attribute, and by nothing else. Each entry shown beside a need is on its attribute, whatever words the store adds to the name ("frame color" for color): the need is met when one of them holds the value asked and none of them holds only other values; when they disagree, it is not shown. An entry meets the need when it holds the value asked: the same words in any case, spelling or order, or with more words around them; an entry that lists several alternatives (such as "a / b" or "a, b") meets it when the value asked is among them; a number is compared as a number with its unit, and a range meets it only when it lies wholly inside what is asked. A different word of similar meaning does not meet it. For a need that a value must not have, an entry holding that value fails it. An entry that meets a need is overruled when the title states, for that same attribute, only other values (a title "HD/FHD Screen" against the entry "2560 x 1080"), or when a text does and neither the title nor another entry states the value asked (a text "Resolution: 1280x720" against that lone entry): the listing contradicts itself, and the need is not met. A text never overrules an entry that the title or another entry agrees with.
3. Mark each need met, not met (its entries hold another value), or not shown. When its entries are "not listed", the shelf, the title and the texts decide it the same way: the value asked stated in any of them meets it; another value stated there for that attribute fails it only when none of them states the value asked; only when they say nothing about it is it not shown. A figure that measures something else, or that does not say what it measures, decides nothing.
4. When a seller's note sends an attribute to another listing, that need is not shown here; quote the note.

Output
Think first if it helps. Then end with rows only, one per listing, with every need marked.
FIT <listing id> | marks: <need number> <met, not met or not shown>, ... | quote: <the listing's words for a need not met or not shown, or none>

Example
FIT q-17 | marks: 0 met, 1 met, 2 not met | quote: "strap length: 40 cm"
FIT q-23 | marks: 0 met, 1 met, 2 met | quote: none"""

FORMAT_AGAIN = """That answer holds no row in the format asked for. Give the same answer once more, written only as rows of that format."""
QUESTION_TEXT = """What {key} would you like for the {kind}?"""
LIST_TEXT = """For the {kind}, what {keys} or {last} would you like?"""
GOAL_FIND = """line {n} ({kind}): no search so far showed a listing for it; run one search with other words the request uses for this product"""
GOAL_CART = """put exactly these listings in the cart, with these quantities: {items}"""
GOAL_FINISH = """end the task now: {how}"""
HOW_ORDER = """buy what is in the cart"""
HOW_DECLINE = """decline, because no listing can be bought"""
ORDER_TEXT = """Ordering the cart as it stands."""
DECLINE_TEXT = """No listing in the store meets every need you gave, so I am not buying anything."""
ITEM_TEXT = """{qty} x {title} ({ident}) at {price} each"""
REPORT_TEXT = """Your cart now holds {items}. The total is {total}. Shall I place this order?"""


class StoreGone(Exception):
    pass


ROW = re.compile(r"^[\s>*`#-]*([A-Z][A-Z_]{2,})\b(.*)$")
FIELD_KEY = re.compile(r"[a-z_ ]{1,20}")


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def clean(text: Any) -> str:
    out = unicodedata.normalize("NFKC", str(text))
    out = "".join(c if c in "\n\t" or unicodedata.category(c)[0] != "C" else " " for c in out)
    return re.sub(r"[ \t]+", " ", out).strip()


def clip(text: Any, limit: int) -> str:
    """The text cut to the limit at a sentence's end, or else at a space, in its second half, with the count of what was cut."""
    whole = str(text)
    if len(whole) <= limit:
        return whole
    head, half = whole[:limit], limit // 2
    end = max(head.rfind(mark) for mark in (". ", "; ", "! ", "? ", "\n", "。"))
    if end > half:
        head = head[:end + 1]
    elif head.rfind(" ") > half:
        head = head[:head.rfind(" ")]
    return f"{head} [{len(whole) - len(head)} characters left out]"


def norm(value: Any) -> str:
    return re.sub(r"[\s_\-]+|(?<=[^\W\d_])\.(?=[^\W\d_])", " ", clean(value).lower())


def to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        digits = re.sub(r"[^\d\-]", "", value.strip())
        if digits.lstrip("-").isdigit():
            return int(digits)
    return None


def to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value.strip())
        try:
            return float(cleaned) if cleaned and cleaned != "-" else None
        except ValueError:
            return None
    return None


# a price written as text: a figure with at most a few marks of currency around it
PRICE_SHAPE = re.compile(r"\D{0,4}?\d[\d.,' ]*\D{0,4}")


def price_value(value: Any) -> float | None:
    """A price as a number: a number itself, or a text that is a figure (thousands commas dropped)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = clean(value)
    if sum(map(str.isalpha, text)) > 4 or PRICE_SHAPE.fullmatch(text) is None:
        return None
    try:
        return float(re.sub(r"[^\d.,]", "", text).replace(",", ""))
    except ValueError:
        return None


def amount(value: float | None) -> str:
    return "?" if value is None else str(int(value)) if float(value).is_integer() else str(value)


def numbers_in(text: str) -> set:
    out = set()
    for token in re.findall(r"\d[\d.,']*", text):
        token = token.rstrip(".,'")
        out |= {n for n in (to_number(re.sub(r"\D", "", token)), to_number(token.replace(",", "").replace("'", ""))) if n is not None}
    return out


def words_of(text: Any) -> list:
    return [w for w in re.split(r"[\W_]+", clean(text).lower()) if w]


def stem(word: str) -> str:
    """A word without a plural ending: "protectors" -> "protector", "cases" -> "case", "glasses" stays."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def query_words(text: Any) -> list:
    return [w for w in (x.strip(".,;:!?\"'()[]{}") for x in clean(text).lower().split()) if any(c.isalnum() for c in w)]


def alternatives(text: Any) -> list:
    out: list = [[]]
    for w in query_words(text):
        out.append([]) if w == "or" else out[-1].append(w)
    return [x for x in out if x]


def combine(options: list, cap: float | None, pair: Any) -> list | None:
    """The best choice of one option per line, by (unknowns, rank, cost) summed, within the cap on cost, where no two chosen
    options are known not to go together (`pair` says False; None counts as one more unknown): a depth-first search over
    the lines with the fewest options first, each step pruned by the least the lines after it could add."""
    if not options or not all(options):
        return None
    order = sorted(range(len(options)), key=lambda k: len(options[k]))
    floor = [tuple(sum(min(o[0][i] for o in options[k]) for k in order[d:]) for i in range(3)) for d in range(len(order) + 1)]
    found: dict = {}

    def hopeless(score: tuple, depth: int) -> bool:
        bound = tuple(a + b for a, b in zip(score, floor[depth]))
        return (cap is not None and bound[2] > cap) or ("score" in found and bound >= found["score"])

    def extend(depth: int, chosen: dict, score: tuple) -> None:
        if depth == len(order):
            found.update(score=score, pick=dict(chosen))
            return
        line = order[depth]
        for (unknown, rank, cost), ident in options[line]:
            fits = [pair(line, ident, other_line, other) for other_line, other in chosen.items()]
            if False in fits:
                continue
            step = (score[0] + unknown + fits.count(None), score[1] + rank, score[2] + cost)
            if not hopeless(step, depth + 1):
                chosen[line] = ident
                extend(depth + 1, chosen, step)
                del chosen[line]

    if not hopeless((0, 0, 0), 0):
        extend(0, {}, (0, 0, 0))
    return [found["pick"][k] for k in range(len(options))] if "pick" in found else None


def bare(key: str, kind: list) -> str:
    words = norm(key).split()
    while len(words) > 1 and any(len(w) >= KIND_WORD and (words[0].startswith(w) or w.startswith(words[0])) for w in kind):
        words = words[1:]
    return " ".join(words)


def labelled(node: list) -> dict:
    """A list of two-field entries as one mapping: the field whose values are distinct texts names the other's value."""
    label = next(k for k in node[0] if len({str(x[k]) for x in node}) == len(node) and all(isinstance(x[k], str) for x in node))
    return {x[label]: next(v for k, v in x.items() if k != label) for x in node}


def leaves(record: Any) -> list:
    """Every (key, value) at a record's ends, each value under the nearest key above it, in the record's order."""

    def walk(node: Any, key: str) -> Any:
        if pairs(node):
            node = labelled(node)
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, str(k))
        elif isinstance(node, list) and not all(is_scalar(v) for v in node):
            for v in node:
                yield from walk(v, key)
        elif node not in (None, "", []) and key:
            yield norm(key), cell(node, CARD_CHARS)

    return list(walk(record, ""))


def seen_id(head: str, ids: list) -> str:
    starts = [i for i in ids if i.startswith(head)] if head and head not in ids else []
    return starts[0] if len(starts) == 1 else head


def verdict_of(marks: str, count: int) -> tuple[str, int | None]:
    """A FIT row's marks as a verdict: fails at the first need not met, else unsure at the first need not marked met, else
    fits."""
    said: dict = {}
    for number, word in re.findall(r"(\d+)\W*([a-z][a-z ]*)", norm(marks)):
        said[int(number)] = "not met" if word.startswith("not met") else word.split()[0]
    marked = [said.get(i) for i in range(count)]
    if "not met" in marked:
        return "fails", marked.index("not met")
    unshown = next((i for i, mark in enumerate(marked) if mark != "met"), None)
    return ("unsure", unshown) if unshown is not None else ("fits", None)



def holds_value(entry: Any, value: Any) -> bool:
    """Whether an entry holds a value: all the words of one of its alternatives ("v neck" in "round neck, v neck")."""
    have = set(words_of(entry))
    return any(set(words) <= have for part in re.split(r",|/|\bor\b|\band\b", str(value or "")) if (words := words_of(part)))


def spared(marks: str, needs: list, entries: list | None = None) -> str:
    """The marks with "not shown" read for each need that a value must not have: not met when an entry of the listing holds
    that value (under whatever attribute), met otherwise."""
    banned = {i for i, x in enumerate(needs) if norm(x.get("relation")) in ("not", "not_equal", "none_of", "excludes")}
    # a store's bookkeeping (options, codes) and its long texts name values loosely ("red / xl"): only short entries
    # on properties count
    held = {i for i in banned if any(holds_value(v, needs[i].get("value")) for k, v in entries or []
                                     if not NOT_A_PROPERTY.search(k) and len(str(v).split()) < TEXT_WORDS)}

    def fix(hit: Any) -> str:
        i = int(hit.group(1))
        return hit.group(0) if i not in banned else f"{i} not met" if i in held else f"{i} met"

    return re.sub(r"(\d+)\W*not shown", fix, str(marks or ""))


def names_key(key: str, need: str) -> bool:
    a, b = sorted(([w for w in words_of(key) if len(w) > 1], [w for w in words_of(need) if len(w) > 1]), key=len)
    return bool(a) and all(any(w.startswith(x[:4]) or x.startswith(w[:4]) for w in b) for x in a)


def names_kind(name: Any, kind: str) -> bool:
    have, want = words_of(name), words_of(kind)
    return bool(have and want) and all(any(s.startswith(w) or w.startswith(s) for s in have) for w in want)


def written(text: str, value: str) -> bool:
    return holds(text, words_of(value)) or bool(numbers_in(value)) and numbers_in(value) <= numbers_in(text)


def holds(text: Any, want: list) -> bool:
    words = words_of(text)
    return bool(want) and any(words[i:i + len(want)] == want for i in range(len(words)))


def next_key(top: dict, current: Any, size: int, schema: dict) -> tuple[str | None, str | None]:
    """The field of a search reply that leads to the next page, and how paging counts: a token (a text without spaces), a
    page number one past the current, or an offset one page of rows on; the reply's one empty field when nothing tells."""
    empty = [k for k, v in top.items() if v is None]
    lone = empty[0] if len(empty) == 1 else None
    if "string" in kinds_of(schema):
        token = next((k for k, v in top.items() if isinstance(v, str) and v.strip() and not any(c.isspace() for c in v.strip())), lone)
        return token, "token"
    first = to_int(schema.get("minimum")) or 0
    start = first if current is None else to_int(current)
    numbers = {k: v for k, v in top.items() if isinstance(v, int)}
    hit = next((k for k, v in numbers.items() if v in (start + 1, start + size)), None)
    if hit is not None and (len(numbers) > 1 or not lone):
        return hit, "page" if numbers[hit] == start + 1 else "offset"
    return lone, "page" if first >= 1 else "offset"


def number_of(head: str) -> int | None:
    return int(head) if head.isdigit() else None


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def scalars(value: Any, out: set) -> set:
    """Every short text and every number a structure holds, added to `out`."""
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, bool) or node is None:
            continue
        if isinstance(node, (int, float)):
            out.add(str(node))
        elif isinstance(node, str):
            if 0 < len(node) <= 80:
                out.add(node)
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
    return out


def walk(value: Any, path: str = ""):
    yield path, value
    if isinstance(value, dict):
        for k, v in value.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for v in value:
            yield from walk(v, f"{path}[]")


def pairs(node: Any) -> bool:
    """Whether a list holds two or more entries of the same two scalar fields, one of them distinct texts (a label/value
    table such as [{"name": "color", "value": "red"}, ...])."""
    if not isinstance(node, list) or len(node) < 2 or not isinstance(node[0], dict):
        return False
    fields = node[0].keys()
    if len(fields) != 2 or any(not isinstance(x, dict) or x.keys() != fields for x in node):
        return False
    if not all(is_scalar(v) for x in node for v in x.values()):
        return False
    return any(all(isinstance(x[k], str) for x in node) and len({x[k] for x in node}) == len(node) for k in fields)


def row_list(payload: Any) -> tuple[str, list]:
    lists = [(p, v) for p, v in walk(payload) if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    return max(lists, key=lambda pv: len(pv[1]), default=("", []))


LISTING_PART = re.compile(r"([A-Za-z_][A-Za-z0-9_]{0,40}): (.*)", re.S)
WHOLE_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]{0,15})")
DECIMAL_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]{0,15})\.[0-9]{1,6}")
FACT_FIELDS = ("fact", "spec", "attribute", "feature", "detail")


def listing_text(text: Any) -> dict | None:
    """A listing the store wrote as one line of "field: value" parts joined by " | " ("id: p-1 | title: Desk lamp |
    price: 1999 | in_stock: True | facts: color=red; size=small"), as the record it stands for; None for any other
    text. A part that names no field belongs to the value before it (a title can hold " | " itself)."""
    if not isinstance(text, str) or " | " not in text:
        return None
    fields: dict = {}
    last = None
    for part in text.split(" | "):
        match = LISTING_PART.fullmatch(part)
        if match and match[1] not in fields:
            last = match[1]
            fields[last] = match[2]
        elif last is None:
            return None
        else:
            fields[last] += " | " + part
    if len(fields) < 3:
        return None
    record = {key: listing_value(key, value) for key, value in fields.items()}
    typed = any(isinstance(v, (bool, int, float)) or isinstance(v, str) and (v.strip().lower() in ("yes", "no") or price_value(v) is not None)
                for v in record.values())
    return record if typed else None


def listing_value(key: str, value: str) -> Any:
    """One field of a listing line as data: True, False and None, whole numbers, and "a=b; c=d" entries as a mapping
    (two or more pairs, or one pair in a field named for facts or specs); any other value stays text."""
    flag = value.strip().lower()
    if flag in ("true", "false", "none", "null"):
        return {"true": True, "false": False, "none": None, "null": None}[flag]
    if WHOLE_NUMBER.fullmatch(value):
        return int(value)
    if DECIMAL_NUMBER.fullmatch(value):
        return float(value)
    entries: dict = {}
    last = None
    for part in value.split("; "):
        name, equals, said = part.partition("=")
        if equals and 0 < len(name) <= 48 and name not in entries and not any(c in name for c in ",.:|"):
            last = name
            entries[name] = said
        elif last is None:
            return value
        else:
            entries[last] += "; " + part
    named = any(word in key.lower() for word in FACT_FIELDS)
    return entries if len(entries) >= 2 or (entries and named) else value


def as_records(obs: Any) -> Any:
    """A tool's reply with every listing line read as a record: a list made only of listing lines becomes a list of
    records, and a reply part that holds nothing but one listing line becomes that record. Notes, messages and every
    other text stay as they came."""
    if isinstance(obs, str):
        return listing_text(obs) or obs
    if isinstance(obs, list):
        records = [listing_text(x) for x in obs]
        lines = sum(1 for r in records if r)
        if lines and lines * 2 >= len(obs):
            return [r for r in records if r]
        return [as_records(x) for x in obs]
    if isinstance(obs, dict):
        if len(obs) == 1 and (record := listing_text(next(iter(obs.values())))):
            return record
        return {key: as_records(value) for key, value in obs.items()}
    return obs


def unwrap(obs: Any) -> tuple[Any, dict, bool]:
    """A tool's reply split into its payload and what it says in words (lists of short sentences, such as notes or
    warnings); True when the payload sat alone beside them."""
    if not isinstance(obs, dict):
        return obs, {}, False
    said, rest = {}, {}
    for k, v in obs.items():
        if isinstance(v, list) and v and all(isinstance(x, str) and len(x.split()) >= 2 and any(c.isalpha() for c in x) for x in v):
            said[k] = [clean(x) for x in v]
        else:
            rest[k] = v
    if not said:
        return obs, said, False
    if len(rest) == 1 and isinstance(only := next(iter(rest.values())), (dict, list)):
        return only, said, True
    return rest, said, False


def kind_of(payload: Any) -> str:
    if row_list(payload)[1]:
        return "rows"
    if isinstance(payload, str) or (isinstance(payload, dict) and len(payload) <= 2 and any(isinstance(v, str) and len(v.split()) >= 4 for v in payload.values())):
        return "text" if str(payload).strip() else "empty"
    return "record" if isinstance(payload, dict) and payload else "empty"


def cell(value: Any, limit: int) -> str:
    """One value as card text: a list of scalars joined, other structures as JSON, every text clipped to the limit."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list) and all(map(is_scalar, value)):
        return clip(", ".join(map(clean, value)), limit)
    return clip(dump(value) if isinstance(value, (dict, list)) else clean(value), limit)


def card(value: Any, skip: Any = None, limit: int = CARD_CHARS) -> list:
    """A reply as "path: value" lines, one per scalar or list of scalars, leaving out the part that is `skip`."""

    def walk(node: Any, path: str) -> Any:
        if node is skip:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list) and not all(is_scalar(v) for v in node):
            for i, v in enumerate(node):
                yield from walk(v, f"{path}[{i}]")
        else:
            yield f"{path or 'result'}: {cell(node, limit)}"

    return list(walk(value, ""))


def table(rows: list) -> list:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = [" | ".join(keys)] + [" | ".join(cell(r.get(k), CELL_CHARS) for k in keys) for r in rows[:TABLE_ROWS]]
    return out + ([f"[{len(rows) - TABLE_ROWS} more rows left out]"] if len(rows) > TABLE_ROWS else [])


def render(value: Any) -> str:
    payload, said, _ = unwrap(value)
    _, rows = row_list(payload)
    lines = [f"notice: {s}" for group in said.values() for s in group] + card(payload, rows) + (table(rows) if rows else [])
    return clip("\n".join(lines), LAST_CHARS)


def row_field(part: str) -> tuple[str, str]:
    """One "key: value" part of a row; a part without a field name is kept whole under the empty key."""
    key, colon, value = part.partition(":")
    key = key.strip().lower()
    if not (colon and FIELD_KEY.fullmatch(key)):
        key, value = "", part
    return key, value.strip().strip("`").strip().strip('"“”').strip()


def rows_of(text: str) -> list:
    """The tagged rows of a model's answer, as (tag, head, fields)."""
    out = []
    for raw in str(text or "").splitlines():
        match = ROW.match(raw.strip())
        if not match:
            continue
        head, *parts = match[2].split("|")
        if parts:
            out.append((match[1], head.strip(), dict(row_field(part) for part in parts)))
    return out


def text_of(message: dict | None) -> str:
    content = (message or {}).get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def fault_of(data: dict | None, error: dict | None, timed_out: bool, tools: bool) -> str | None:
    status = (error or {}).get("status")
    if data is None:
        return ("timeout" if timed_out else "transport") if status in (None, 200) else "5xx" if status >= 500 else str(status)
    choice = (data.get("choices") or [{}])[0] or {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    if choice.get("finish_reason") == "length" or choice.get("native_finish_reason") == "max_tokens":
        return "length"
    if tools and not message.get("tool_calls"):
        return "no call"
    return None if message.get("tool_calls") or text_of(message).strip() else "empty"


def resync(error: dict, sent: int) -> int:
    named = re.search(r"expect\w*\D{0,6}(\d+)", dump(error))
    return int(named.group(1)) if named and int(named.group(1)) > 0 else sent + 1


def kinds_of(schema: Any) -> list:
    kind = schema.get("type") if isinstance(schema, dict) else None
    return kind if isinstance(kind, list) else [kind] if kind else []


def loaded(value: Any, opener: str = "") -> Any:
    """A JSON text, parsed, when it opens with the bracket the schema wants (any text when no bracket is named); any other
    value, or text that does not parse, as it came."""
    if not isinstance(value, str) or not value.strip().startswith(opener):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def fit_choice(options: list, value: Any, path: str) -> tuple[Any, list]:
    """One of a closed list: the value itself, or the one option it matches when case and spacing are ignored."""
    if value in options:
        return value, []
    same = [o for o in options if norm(o) == norm(value)]
    if len(same) == 1:
        return same[0], []
    return value, [f"{path} must be one of {dump(options)}"]


def fit_array(schema: dict, value: Any, path: str, kinds: list) -> tuple[Any, list]:
    value = loaded(value, "[")
    each = schema.get("items") or {}
    fitted = [fit_value(each, item, f"{path}[{index}]") for index, item in enumerate(value if isinstance(value, list) else [value])]
    items, problems = [v for v, _ in fitted], [p for _, issues in fitted for p in issues]
    most, least = schema.get("maxItems"), schema.get("minItems")
    if isinstance(most, int) and len(items) > most:
        problems.append(f"{path} takes at most {most} items")
    if isinstance(least, int) and len(items) < least:
        problems.append(f"{path} needs at least {least} items")
    return items, problems


def fit_record(schema: dict, value: Any, path: str, kinds: list) -> tuple[Any, list]:
    value = loaded(value)
    if isinstance(value, dict):
        return fit_object(schema, value, path)
    return value, [f"{path} must be an object"]


def fit_number(schema: dict, value: Any, path: str, kinds: list) -> tuple[Any, list]:
    number = to_number(value)
    if number is None:
        # a schema that takes text as well keeps the text
        return value, [] if "string" in kinds and isinstance(value, str) else [f"{path} must be a number"]
    whole = "integer" in kinds
    if whole and number != int(number):
        return value, [f"{path} must be a whole number"]
    if whole or number.is_integer():
        number = int(number)
    low, high = schema.get("minimum"), schema.get("maximum")
    if isinstance(low, (int, float)) and number < low:
        return number, [f"{path} must be at least {low}"]
    if isinstance(high, (int, float)) and number > high:
        return number, [f"{path} must be at most {high}"]
    return number, []


def fit_flag(schema: dict, value: Any, path: str, kinds: list) -> tuple[Any, list]:
    if isinstance(value, bool):
        return value, []
    word = norm(value)
    if word in ("true", "false"):
        return word == "true", []
    return value, [f"{path} must be true or false"]


def clipped_words(text: str, limit: int) -> str:
    """The most leading words that fit in the limit, or the cut text when even the first word is longer."""
    words = text.split()
    for size in range(len(words), 0, -1):
        if len(kept := " ".join(words[:size])) <= limit:
            return kept
    return text[:limit]


def fit_text(schema: dict, value: Any, path: str, kinds: list) -> tuple[Any, list]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return value, [f"{path} must be a string"]
    limit, least = schema.get("maxLength"), schema.get("minLength")
    if isinstance(limit, int) and len(value) > limit:
        value = clipped_words(value, limit)
    if isinstance(least, int) and len(value) < least:
        return value, [f"{path} needs at least {least} characters"]
    return value, []


# the fitter for each JSON type, in the order a schema that allows several types is tried
SCHEMA_FITTERS = ((("array",), fit_array), (("object",), fit_record), (("integer", "number"), fit_number), (("boolean",), fit_flag),
                  (("string",), fit_text))


def fit_value(schema: Any, value: Any, path: str) -> tuple[Any, list]:
    """A value made to fit a tool argument's schema where it safely can (text to numbers, a lone item to a list, a too long
    text cut at a word), with the problems that remain, each naming the argument's path."""
    if not isinstance(schema, dict):
        return value, []
    if "const" in schema:
        fixed = schema["const"]
        return fixed, [] if norm(value) == norm(fixed) else [f"{path} must be {dump(fixed)}"]
    if "enum" in schema:
        return fit_choice(schema["enum"], value, path)
    kinds = kinds_of(schema)
    if not kinds or (value is None and "null" in kinds):
        return value, []
    fitter = next((f for names, f in SCHEMA_FITTERS if any(name in kinds for name in names)), None)
    return fitter(schema, value, path, kinds) if fitter else (value, [])


def fit_object(schema: dict, args: dict, path: str = "") -> tuple[dict, list]:
    """A tool's arguments fitted one by one; an argument the tool does not take is dropped (and named) when the schema is
    closed, and every required argument left out is named."""
    props, closed = schema.get("properties") or {}, schema.get("additionalProperties", True) is False
    out: dict = {}
    problems: list = []
    for key, value in args.items():
        where = f"{path}.{key}" if path else key
        if key in props:
            out[key], issues = fit_value(props[key], value, where)
            problems += issues
        elif closed:
            problems.append(f"{where} is not an argument of this tool (arguments: {', '.join(props) or 'none'})")
        else:
            out[key] = value
    lead = f"{path}." if path else ""
    for key in (k for k in schema.get("required") or [] if k not in out):
        options = (props.get(key) or {}).get("enum")
        problems.append(f"{lead}{key} is required" + (f", one of {dump(options)}" if options else ""))
    return out, problems


def schema_paths(schema: Any, prefix: str = "") -> list:
    """Every argument path a schema names, with its schema and whether it is required, nested objects and lists of objects
    included ("items[].id")."""
    props, needed = (schema or {}).get("properties") or {}, (schema or {}).get("required") or []
    out = []
    for name, sub in props.items():
        sub, path = sub if isinstance(sub, dict) else {}, f"{prefix}{name}"
        out.append((path, sub, name in needed))
        items = sub.get("items")
        if isinstance(items, dict) and "object" in kinds_of(items):
            out.extend(schema_paths(items, f"{path}[]."))
        elif "object" in kinds_of(sub):
            out.extend(schema_paths(sub, f"{path}."))
    return out


def describe(schema: dict, required: bool) -> str:
    """An argument's schema in a few words: its types, whether it is required, its options and its bounds."""
    notes = ["/".join(str(k) for k in kinds_of(schema)) or "any"]
    if required:
        notes.append("required")
    if "enum" in schema:
        notes.append("one of: " + ", ".join(str(o) for o in schema["enum"]))
    bounds = (("minimum", "min"), ("maximum", "max"), ("minItems", "min items"), ("maxItems", "max items"), ("maxLength", "max length"))
    notes.extend(f"{word} {schema[key]}" for key, word in bounds if key in schema)
    return ", ".join(notes)


def at_path(args: Any, path: str) -> list:
    nodes = [args]
    for part in path.split("."):
        name, many = part.removesuffix("[]"), part.endswith("[]")
        nodes = [n[name] for n in nodes if isinstance(n, dict) and name in n]
        if many:
            nodes = [x for n in nodes if isinstance(n, list) for x in n]
    return nodes


def place(args: dict, path: str, value: Any) -> None:
    head, dot, rest = path.partition(".")
    if head.endswith("[]"):
        for item in args.get(head[:-2]) or []:
            if isinstance(item, dict):
                place(item, rest, value)
    elif dot:
        place(args.setdefault(head, {}), rest, value)
    else:
        args[head] = value


def together(*calls: Any) -> list:
    box: dict = {}
    workers = [threading.Thread(target=lambda i=i, c=c: box.setdefault(i, c())) for i, c in enumerate(calls)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return [box.get(i) for i in range(len(calls))]


class Clock:
    def __init__(self) -> None:
        self.started = monotonic()

    def elapsed(self) -> float:
        return monotonic() - self.started

    def left(self) -> float:
        return DEADLINE - self.elapsed()


class Wire:

    def __init__(self) -> None:
        self.next_at, self.gap, self.lock = 0.0, PACE, threading.Lock()

    def post(self, path: str, body: dict, timeout: float) -> tuple[dict | None, dict | None, bool]:
        with self.lock:
            now = monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.gap
        sleep(at - now)
        started = monotonic()
        try:
            result = ProxyClient(timeout=max(1, int(timeout)), max_retries=1).post_verbose(path, json_data=body)
            data, error = result.data, result.error
        except Exception as exc:
            data, error = None, {"status": None, "body": repr(exc)}
        with self.lock:
            busy = (error or {}).get("status") in (429, 503)
            self.gap = min(PACE_MAX, self.gap * 2) if busy else max(PACE, self.gap * PACE_EASE)
            self.next_at = max(self.next_at, monotonic() + self.gap) if busy else self.next_at
        return data, error, monotonic() - started >= timeout


class Llm:

    def __init__(self, wire: Wire, clock: Clock, session: str, trace: list) -> None:
        self.wire, self.clock, self.session, self.trace = wire, clock, session, trace
        self.spend, self.calls, self.stopped, self.off = 0.0, 0, False, set()
        self.key_spent = False
        self.lock = threading.Lock()

    def complete(self, seat: str, messages: list, tools: list | None = None) -> dict | None:
        for model in LADDERS.get(seat, LADDER):
            if model in self.off or (model == SONNET and self.spend >= SPEND_SONNET):
                continue
            cached, tries = True, 0
            while tries < RUNG_TRIES:
                tries += 1
                timeout = min(CALL_TIMEOUT, self.clock.left() - DEADLINE_MARGIN)
                if self.stopped or self.spend >= (SPEND_HARD if seat in SPEND_EXEMPT else SPEND_CAP) or timeout < CALL_MIN:
                    return None
                body = self.body(seat, model, messages, tools, cached)
                started = monotonic()
                data, error, timed_out = self.wire.post("/inference/chat/completions", body, timeout)
                fault = fault_of(data, error, timed_out, bool(tools))
                self.note(seat, model, data, fault, monotonic() - started)
                if fault is None:
                    return data["choices"][0]["message"]
                if fault == "402" or (fault == "403" and KEY_SPENT.search(str((error or {}).get("body") or ""))):
                    # no credit left on the run's key: no model call goes out again, and no message (see Asker.put)
                    self.stopped = self.key_spent = True
                    self.trace.append({"kind": "key_spent", "fault": fault})
                    return None
                elif fault in ("401", "403", "404"):
                    allowed = str((error or {}).get("body") or "").partition("allowed_models")[2]
                    self.off |= {m for m in (HAIKU, SONNET, LUNA) if m not in allowed} if allowed else {model}
                elif fault in ("429", "5xx", "transport"):
                    sleep(min(4.0, BACKOFF * 2 ** tries * (0.5 + hash(dump(messages)) % 1000 / 1000)))
                    continue
                elif fault in ("400", "413", "422") and cached:
                    cached = False
                    continue
                break
        return None

    def body(self, seat: str, model: str, messages: list, tools: list | None, cached: bool) -> dict:
        effort = EFFORT[seat]
        body = {"model": model, "messages": messages, "reasoning": {"effort": effort}, "session_id": self.session, "usage": {"include": True}}
        if effort == "none":
            body["temperature"] = 0
        if cached:
            body["cache_control"] = {"type": "ephemeral"}
        if tools:
            body["tools"], body["tool_choice"] = tools, "required"
        return body

    def note(self, seat: str, model: str, data: dict | None, fault: str | None, seconds: float) -> None:
        usage = (data or {}).get("usage") or {}
        upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost") if usage.get("is_byok") else 0
        cost = (to_number(usage.get("cost")) or 0.0) + (to_number(upstream) or 0.0)
        with self.lock:
            self.spend, self.calls = self.spend + cost, self.calls + 1
            self.trace.append({"seat": seat, "model": model, "fault": fault, "cost": round(cost, 5), "seconds": round(seconds, 1)})


class Env:

    def __init__(self, wire: Wire, clock: Clock, binding: dict, finished: Any) -> None:
        self.wire, self.clock, self.binding = wire, clock, dict(binding)
        self.step, self.finished, self.seconds = 0, bool(finished), 0.0

    def send(self, name: str, args: dict) -> Any:
        if self.finished:
            raise StoreGone(f"{name} after the episode finished")
        envelope = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
        started, error, faults = monotonic(), None, 0
        while faults < REPLAY_MAX:
            data, error, _ = self.wire.post("/environment/call", envelope, min(WORLD_TIMEOUT, max(5.0, self.clock.left())))
            if data is not None:
                self.step, self.finished = int(data.get("step") or envelope["step"]), bool(data.get("finished"))
                self.seconds += monotonic() - started
                return as_records(data.get("observation"))
            status = (error or {}).get("status")
            busy = status in (429, 503) and self.clock.left() > DEADLINE_MARGIN * 2
            faults += not busy
            if status == 409:
                envelope["step"] = resync(error, envelope["step"])
            elif busy or status in (None, 500, 502, 504):
                sleep(min(4.0, BACKOFF * 2 ** min(faults, 2)))
            else:
                break
        raise StoreGone(f"{name} at step {envelope['step']}: {clip(dump(error), 200)}")


class Request:

    def __init__(self, text: str, opening: Any) -> None:
        self.text, self.opening = text, opening
        self.roles: dict = {}
        self.lines, self.caps, self.goals, self.refs, self.depends = {}, {}, {}, [], {}
        self.messages = self.steps = None
        self.duties, self.decline_rule, self.asked, self.version = [], "", set(), 1
        self.answers: dict = {}

    @property
    def sentences(self) -> list:
        return [x for x in re.split(r"(?<=[.!?…])\s+", clean(self.text)) if x.strip()]

    @property
    def numbered(self) -> str:
        return "\n".join(f"{i}. {x}" for i, x in enumerate(self.sentences, 1)) + (f"\n\nopening: {render(self.opening)}" if self.opening is not None else "")

    @property
    def full(self) -> str:
        return self.text if self.opening is None else f"{self.text}\n\nopening: {render(self.opening)}"

    @property
    def route(self) -> str:
        return "A" if self.depends else "B" if len(self.lines) > 1 else "S"

    def fill(self, rows: list) -> None:
        for tag, head, f in rows:
            n, qty = number_of(head), to_int(f.get("qty"))
            if tag == "LINE" and n and n >= 1:
                also = [a.strip() for a in f.get("also", "").split(",") if norm(a) not in ("", "none")]
                self.lines[n] = {"n": n, "kind": f.get("kind", ""), "also": also, "shelf": "" if norm(f.get("shelf")) == "none" else f.get("shelf", ""),
                                 "qty": qty if qty and qty >= 1 else 1, "qty_quote": "" if norm(f.get("qty_quote")) == "none" else f.get("qty_quote", "").strip('"'),
                                 "needs": [], "hints": []}
        self.lines = self.lines or {1: {"n": 1, "kind": "", "also": [], "shelf": "", "qty": 1, "qty_quote": "", "needs": [], "hints": []}}
        numbers = numbers_in(self.text)
        for tag, head, f in rows:
            n = number_of(head)
            line = self.lines.get(n)
            if tag == "NEED" and line and f.get("key") and not STOCK_NEED.search(f"{f.get('key')} {f.get('value')}") \
                    and not any(all(norm(x[k]) == norm(f.get(k, "")) for k in ("key", "relation", "value")) for x in line["needs"]):
                # a need the reading writes twice is kept once
                line["needs"].append({k: f.get(k, "") for k in ("key", "relation", "value", "quote")})
            elif tag == "SENTENCE" and n in range(1, len(self.sentences) + 1) and (self.roles.update({n: norm(f.get("role"))}) or norm(f.get("role")) in ("clue", "invite")):
                clue = {"key": "" if norm(f.get("key")) == "none" else f.get("key", ""), "quote": self.sentences[n - 1], "invites": norm(f.get("role")) == "invite"}
                about = self.lines.get(number_of(f.get("line", "")))
                for target in [about] if about else self.lines.values():
                    target["hints"].append(clue)
            elif tag == "CAP" and (line or head == "total") and to_number(f.get("amount")) in numbers:
                self.caps[n if line else "total"] = {"amount": to_number(f["amount"]), "unit": f.get("unit", "")}
            elif tag == "GOAL" and f.get("goal"):
                self.goals[n if line else "all"] = (f["goal"], f.get("key", "none"))
            elif tag == "BUDGET":
                self.messages, self.steps = to_int(f.get("messages")), to_int(f.get("steps"))
            elif tag == "DUTY" and f.get("text"):
                self.duties.append({"text": f["text"], "when": f.get("when", "before_finish"), "by": f.get("by", "message"), "done_at": None})
            elif tag == "DECLINE" and f.get("text"):
                self.decline_rule = f["text"]
            elif tag == "REF" and line:
                self.refs.append({"line": n, **{k: f.get(k, "") for k in ("key", "store", "kind", "fact", "relation")}})
            elif tag == "DEPENDS" and line and to_int(f.get("on")) in self.lines:
                self.depends[n] = to_int(f["on"])
        self.shelve()
        self.recount()

    def shelve(self) -> None:
        """A line's shelf as a need when the request names it word for word (two words or more) and no need is on it yet; a
        need read off the shelf's own name (its quote is that name) gives way to it."""
        said = " ".join(words_of(self.text))
        for line in self.lines.values():
            words = [w for w in words_of(line["shelf"]) if w != "and"]
            if len(words) < 2 or f" {' '.join(words)} " not in f" {said} ":
                continue
            named = set(words)
            line["needs"] = [x for x in line["needs"] if x.get("said") or {w for w in words_of(x.get("quote")) if w != "and"} != named]
            product = set(words_of(line["kind"])) | {w for name in line["also"] for w in words_of(name)}
            if named <= product:
                # the shelf is the product's own name: need 0 judges that already, and a category need would judge it
                # again by the stricter rule of an attribute
                continue
            if not any(names_key(x["key"], key) for x in line["needs"] for key in ("category", "shelf", "department")):
                line["needs"].append({"key": "category", "relation": "equal", "value": line["shelf"], "quote": ""})

    def hear(self, heard: dict, n: int | None) -> bool:
        line = self.lines.get(number_of(heard.get("line", "")) or n or (next(iter(self.lines)) if len(self.lines) == 1 else 0))
        key, value, store = (heard.get(k, "") for k in ("key", "value", "store"))
        if heard.get("kind") != "disclose" or not line or norm(key) in ("", "none"):
            return False
        if norm(store) not in ("", "none"):
            ref = {"line": line["n"], "key": key, "store": store, "kind": heard.get("of", ""), "fact": heard.get("fact", ""), "relation": heard.get("relation", "equal")}
            if any(all(norm(r.get(k)) == norm(ref[k]) for k in ref) for r in self.refs):
                return False
            self.refs.append(ref)
        elif norm(value) not in ("", "none"):
            need = {"key": key, "relation": heard.get("relation") or "equal", "value": value, "quote": heard.get("text", ""), "said": True}
            if any(all(norm(x[k]) == norm(need[k]) for k in ("key", "relation", "value")) for x in line["needs"]):
                return False
            line["needs"].append(need)
        else:
            return False
        self.version += 1
        return True

    def rebudget(self, figure: float) -> None:
        key = next(iter(self.caps)) if len(self.caps) == 1 else "total"
        self.caps[key] = {"amount": figure, "unit": (self.caps.get(key) or {}).get("unit") or self.unit()}
        self.version += 1

    def judged(self, n: int) -> list:
        line = self.lines[n]
        return [{"key": "", "relation": "is", "value": ", ".join([line["kind"], *line["also"]]) or "what the request describes", "quote": ""}] + line["needs"]

    def product_text(self, n: int) -> str:
        line, needs = self.lines[n], self.judged(n)
        return "\n".join([f"line {n}: {line['kind']} x{line['qty']}", f"need 0: the listing is {needs[0]['value']}"]
                         + [f"need {i}: {x['key']} {x['relation']} {x['value']}" + (f" (\"{x['quote']}\")" if x["quote"] else "") for i, x in enumerate(needs[1:], 1)])

    def drop(self, n: int | None, key: str) -> None:
        line = self.lines.get(n or (next(iter(self.lines)) if len(self.lines) == 1 else 0))
        if line and norm(key) not in ("", "none"):
            kind = self.kind_words(line["n"])
            line["needs"] = [x for x in line["needs"] if bare(x["key"], kind) != bare(key, kind)]
            self.refs = [r for r in self.refs if r["line"] != line["n"] or bare(r["key"], kind) != bare(key, kind)]
        elif line and len(self.lines) > 1:
            del self.lines[line["n"]]
        self.version += 1

    def kind_words(self, n: int) -> list:
        return words_of(" ".join([self.lines[n]["kind"], *self.lines[n]["also"]]))

    def sentence_of(self, quote: str) -> int | None:
        """The first sentence that holds a quote's words in their order, as whole words ("3" is not found in "30")."""
        words = words_of(quote)
        for i, sentence in enumerate(self.sentences, 1):
            have = words_of(sentence)
            if words and any(have[k:k + len(words)] == words for k in range(len(have) - len(words) + 1)):
                return i
        return None

    def not_a_count(self, quote: str, qty: int, roles: tuple) -> bool:
        """Whether a quote sets no order quantity: its sentence has one of these roles; or, outside a product or need
        sentence, it speaks of messages, moves or steps and its number is the request's own message or step limit ("two
        step stools" in a product sentence orders two)."""
        if len(words_of(quote)) < 2:
            # a one-word quote ("3") does not tell which sentence it came from: no refusal on a guess
            return False
        role = self.roles.get(self.sentence_of(quote))
        if role in roles:
            return True
        return role not in ("product", "need") and bool(LIMIT_WORDS.search(quote)) and qty in {self.messages, self.steps} - {None}

    def sourced(self, n: int) -> bool:
        """Whether a line's quantity is quoted from a sentence the reading gave the product or need role."""
        return self.roles.get(self.sentence_of(self.lines[n]["qty_quote"])) in ("product", "need")

    def recount(self) -> None:
        """A quantity the reading took from the request's own limits (a budget sentence, or its message or step limit) is
        one, and is checked again; a cap, goal, procedure or rule sentence may still order ("put two in the bag, then
        message me the total")."""
        for line in self.lines.values():
            if line["qty"] > 1 and line["qty_quote"] and self.not_a_count(line["qty_quote"], line["qty"], ("budget",)):
                line["qty"], line["qty_quote"] = 1, ""

    def doubtful_quantities(self) -> list:
        def sentence(quote: str) -> int | None:
            return next((i for i, x in enumerate(self.sentences, 1) if quote and norm(quote) in norm(x)), None)
        return [n for n, line in sorted(self.lines.items()) if (i := sentence(line["qty_quote"])) is None or self.roles.get(i) in ("story", "clue", "invite")]

    def requantify(self, rows: list) -> None:
        for _, head, f in rows:
            line, qty, quote = self.lines.get(number_of(head)), to_int(f.get("qty")), f.get("quote", "").strip('"')
            if line and qty and qty >= 1 and quote and norm(quote) != "none" and norm(quote) in norm(self.text) and not self.not_a_count(quote, qty, NOT_COUNT_ROLES):
                line["qty"], line["qty_quote"] = qty, quote
                self.version += 1

    def unresolved(self, n: int) -> bool:
        return any(r["line"] == n and r.get("value") is None for r in self.refs)

    def stated(self, n: int) -> set:
        return {norm(x["key"]) for x in self.lines[n]["needs"]} | {norm(r["key"]) for r in self.refs if r["line"] == n}

    def needs_of(self, n: int) -> str:
        return "; ".join(f"{x['key']} {x['relation']} {x['value']}" for x in self.lines[n]["needs"]) or "none stated"

    def needs_text(self) -> str:
        return "\n".join(f"line {n}: {line['kind']} | needs: {self.needs_of(n)}" for n, line in sorted(self.lines.items()))

    def cap(self, n: int) -> float | None:
        return (self.caps.get(n) or self.caps.get("total") or {}).get("amount")

    def goal(self, n: int) -> str:
        return (self.goals.get(n) or self.goals.get("all") or ("none",))[0]

    def queries(self, n: int) -> list:
        kind, values = self.kind_query(n), self.search_values(n)
        refs = [query_words(r["store"]) + query_words(r["fact"])[:QUERY_WORDS] for r in self.refs if r["line"] == n]
        each = [(v + kind, v) for v in values]
        plans = [([w for v in values for w in v] + kind, None)] * (len(values) > 1) + each[:1] + [(kind, None)] + each[1:] + [(r, None) for r in refs]
        return list({tuple(words): (words, alone) for words, alone in reversed(plans) if words}.values())[::-1]

    def dropped(self) -> set:
        return {w for c in self.caps.values() for w in [amount(c["amount"]), *words_of(c["unit"])]}

    def kind_query(self, n: int) -> list:
        return [w for w in query_words(self.lines[n]["kind"]) if w not in self.dropped()]

    def search_values(self, n: int) -> list:
        drop, kind = self.dropped(), set(self.kind_query(n))
        needs = sorted(self.lines[n]["needs"], key=lambda x: not x.get("said"))
        per_need = [[v for a in alternatives(x["value"]) if len(a) <= VALUE_WORDS and a != ["none"] and (v := [w for w in a if w not in drop]) and set(v) - kind]
                    for x in needs if norm(x["relation"]) in VALUE_RELATIONS]
        return [v for rank in range(max(map(len, per_need), default=0)) for alts in per_need if rank < len(alts) for v in alts[rank:rank + 1]][:VALUES_PER_LINE]

    def value_sets(self, n: int) -> list:
        return [set(words_of(" ".join(v))) for v in self.search_values(n)]

    def widen(self, n: int) -> list:
        line = self.lines[n]
        kind = words_of(line["kind"])
        plans = [words_of(a) for a in line["also"]] + [words_of(line["shelf"])] + [[w] for w in kind] * (len(kind) > 1)
        return [list(p) for p in dict.fromkeys(map(tuple, plans)) if p]

    def unit(self) -> str:
        return next((c["unit"] for _, c in sorted(self.caps.items(), key=lambda kc: kc[0] != "total") if c["unit"]), "")

    def money(self, value: float | None) -> str:
        caps, unit = {c["amount"] for c in self.caps.values()}, self.unit()
        for found in re.finditer(r"\d[\d.,']*", self.text):
            rest = self.text[found.end():]
            if to_number(found[0].rstrip(".,'").replace(",", "").replace("'", "")) not in caps:
                continue
            gap = rest[:len(rest) - len(rest.lstrip())][:1]
            if unit and rest.lstrip().lower().startswith(unit.lower()):
                return f"{amount(value)}{gap}{rest.lstrip()[:len(unit)]}"
            token = re.match(r"\s?([^\s\d.,;:!?()\"'\[\]]{1,8})", rest)
            if token:
                return f"{amount(value)}{gap}{token[1]}"
        return f"{amount(value)} {unit}".strip()

    def lines_text(self, catalog: Catalog) -> list:
        out = []
        for n, line in sorted(self.lines.items()):
            needs = self.needs_of(n)
            hints = "; ".join(f"{'invites' if x.get('invites') else 'suggests'}: \"{x['quote']}\"" + (f" (guess: {x['key']})" if x["key"] else "") for x in line["hints"]) or "none"
            goal, key = self.goals.get(n) or self.goals.get("all") or ("none", "none")
            goal += f" {key}" if key not in ("", "none") else ""
            refs = "; ".join(f"{r['key']} from {r['store']} {r['kind']} {r['fact']}".strip() for r in self.refs if r["line"] == n)
            shown, cheap = catalog.shown(n), catalog.cheapest(n, self.cap(n))
            out.append(f"line {n}: {line['kind'] or 'not read'} x{line['qty']} | goal: {goal} | needs: {needs} | clues: {hints}"
                       + (f" | references: {refs}" if refs else "") + (f" | fits line {self.depends[n]}" if n in self.depends else "")
                       + f" | seen {len(shown)}, opened {sum(i in catalog.records for i in shown)}, cheapest seen: {cheap or 'none'}"
                       + f" | coverage: {catalog.coverage(n)['grade']}")
        caps = "; ".join(f"{'whole order' if k == 'total' else f'line {k}'} at most {amount(c['amount'])} {c['unit']}".strip() for k, c in self.caps.items())
        duties = "; ".join(f"{d['text']} ({d['when']}, by {d['by']})" for d in self.duties)
        stated = f"messages {self.messages if self.messages is not None else 'not stated'}, steps {self.steps if self.steps is not None else 'not stated'}"
        return out + [f"caps: {caps or 'none stated'}", f"budgets the request states: {stated}", f"procedures: {duties or 'none stated'}",
                      f"if nothing fits: {self.decline_rule or 'not stated'}"]


class Catalog:

    def __init__(self) -> None:
        self.rows, self.records, self.shown_for, self.keys, self.seen = {}, {}, {}, {}, set()
        self.lists, self.facts, self.learned_from = {}, {}, 0
        self.row_keys, self.read_keys, self.resolved = set(), {}, {}
        self.epoch, self.read_at, self.changes, self.changed_at = 0, {}, 0, {}

    def learn(self, rows: list) -> None:
        if not rows or len(rows) < self.learned_from:
            return
        common = [k for k in rows[0] if all(k in r for r in rows)]

        def column(k: str) -> list:
            return [r[k] for r in rows]

        def distinct(k: str) -> int:
            return len({dump(v) for v in column(k)})

        strings = [k for k in common if all(isinstance(v, str) and v.strip() for v in column(k))]
        ident = [k for k in strings if distinct(k) == len(rows) and not any(any(c.isspace() for c in v) for v in column(k))]
        ident.sort(key=lambda k: not all(any(c.isdigit() for c in v) for v in column(k)))
        texts = [k for k in strings if k not in ident[:1] and sum(" " in v.strip() for v in column(k)) * 5 >= len(rows) * 4 and distinct(k) * 5 >= len(rows) * 4]
        numbers = [k for k in common if k not in ident[:1] and all(price_value(v) is not None and not isinstance(v, bool) for v in column(k))]
        flags = [k for k in common if all(isinstance(v, bool) for v in column(k))]
        title = min(texts, key=lambda k: sum(map(len, column(k))), default=None)
        repeats = [k for k in strings if k not in (*ident[:1], title) and distinct(k) < len(rows)]
        found = {"id": next(iter(ident), None), "title": title, "price": max(numbers, key=distinct, default=None), "stock": next(iter(flags), None),
                 "shelf": max(repeats, key=lambda k: Counter(column(k)).most_common(1)[0][1], default=None)}
        found = {role: key for role, key in found.items() if key}
        kept = {role: key for role, key in self.keys.items() if role in ("stock", "price") and role not in found}
        self.keys, self.learned_from = {**kept, **found} if len(rows) > self.learned_from else {**found, **self.keys}, len(rows)

    def add_rows(self, tool: str, rows: list, line: int | None, detail: bool) -> list:
        if not detail or not self.keys:
            self.learn(rows)
        ids = []
        for row in rows:
            ident = row.get(self.keys.get("id", ""))
            if not isinstance(ident, (str, int)) or isinstance(ident, bool):
                continue
            ident = str(ident)
            self.rows.setdefault(ident, {}).update(row)
            if detail:
                self.add_record(tool, ident, row)
            else:
                self.row_keys.update(row)
            if line:
                self.shown_for.setdefault(ident, set()).add(line)
            ids.append(ident)
        return ids

    def take_search(self, channel: dict, args: dict, payload: Any, line: int | None, step: int) -> None:
        rows, known = row_list(payload)[1], set(self.shown(line)) if line else set()
        ids = self.add_rows(channel["tool"], rows, line, False)
        top = {k: v for k, v in payload.items() if is_scalar(v) and not isinstance(v, bool)} if isinstance(payload, dict) else {}
        facts, page = self.facts.setdefault(channel["label"], {"order": [], "combine": [], "filter": []}), channel["page"]
        current = next(iter(at_path(args, page)), None) if page else None
        if page and "next" not in facts:
            key, paging = next_key(top, current, len(rows), channel["page_schema"])
            facts.update({"next": key, "paging": paging} if key else {})
        facts["count"] = facts.get("count") or next((k for k, v in top.items() if isinstance(v, int) and v >= len(rows) and k != facts.get("next")), None)
        facts["size"] = max(facts.get("size", 0), len(rows))
        count = top.get(facts["count"]) if facts["count"] else None
        if facts.get("next"):
            following = top.get(facts["next"])
        elif page and rows and len(rows) >= facts["size"] and "integer" in kinds_of(channel["page_schema"]):
            start = to_int(current) if current is not None else to_int(channel["page_schema"].get("minimum")) or 0
            following = start + (1 if (to_int(channel["page_schema"].get("minimum")) or 0) >= 1 else len(rows))
        else:
            following = None
        base = {k: v for k, v in args.items() if k != page}
        lst = self.lists.setdefault(dump([channel["tool"], base]), {"tool": channel["tool"], "channel": channel["label"], "way": channel["way"],
                                                                     "args": base, "line": line, "step": step, "count": count, "first": ids,
                                                                     "ids": [], "paged": [], "pages": 0, "fresh": 0, "started": current is None})
        fresh = [i for i in ids if i not in lst["ids"]]
        lst["ids"] += fresh
        lst["paged"].append(fresh)
        lst["pages"] += 1
        lst["fresh"] += len(set(ids) - known)
        lst["next"], lst["count"] = following, count if count is not None else lst["count"]
        lst["done"] = following in (None, "") or not rows or not fresh or (lst["count"] is not None and len(lst["ids"]) >= lst["count"])
        if lst["pages"] == 1 and lst["started"]:
            self.learn_channel(channel, lst, args)
        if len(rows) >= ORDER_ROWS:
            ids_order, prices = [str(r.get(self.keys.get("id", ""))) for r in rows], [self.price(i) for i in ids]
            facts["order"].append("id" if ids_order == sorted(ids_order) else "price" if None not in prices and prices == sorted(prices) else "relevance")
        limit = next(iter(at_path(args, channel["price_max"])), None) if channel["price_max"] else None
        if limit is not None and rows and all(self.price(i) is not None for i in ids):
            facts["filter"].append(all(self.price(i) <= limit for i in ids))

    def learn_channel(self, channel: dict, lst: dict, args: dict) -> None:
        facts, query = self.facts[channel["label"]], channel["query"]
        firsts = [x for x in self.lists.values() if x["channel"] == channel["label"] and x["count"] is not None and x["started"]]
        if any(x is not lst and x["first"] == lst["first"] and x["count"] == lst["count"] for x in firsts):
            lst["done"] = lst["duplicate"] = True
        words = set(words_of(next(iter(at_path(args, query)), ""))) if query else set()
        lst["words"] = sorted(words)
        top = max((x["count"] for x in firsts), default=0)
        if top and any(x is not lst and x["count"] == lst["count"] == top and x["first"] != lst["first"] and not set(x["words"]) <= words
                       and not words <= set(x["words"]) for x in firsts):
            facts["cap"] = top
        for other in firsts:
            theirs, same = set(other.get("words", [])), {k: v for k, v in other["args"].items() if k != query} == {k: v for k, v in lst["args"].items() if k != query}
            small, big = (other, lst) if theirs < words else (lst, other) if words < theirs else (None, None)
            if same and small and facts.get("cap") not in (small["count"], big["count"]) and big["count"] != small["count"]:
                facts["combine"].append("any" if big["count"] > small["count"] else "all")

    def fact(self, channel: str, knob: str) -> tuple[Any, str]:
        votes = (self.facts.get(channel) or {}).get(knob) or []
        if not votes:
            return None, "unknown"
        value, n = Counter(votes).most_common(1)[0]
        return value, "confirmed" if n == len(votes) and n >= (1 if knob == "combine" else 2) else "assumed"

    def to_page(self, n: int, values: list, kind: str, reach: dict, strict: bool = True) -> dict | None:
        def carries(lst: dict, ident: str) -> bool:
            if not values:
                return self.on_shelf(ident, kind)
            every = self.names_value(lst, values) and self.fact(lst["channel"], "combine")[0] == "all"
            return every or any(v <= self.row_words(ident) for v in values)

        alive = [x for x in self.lists.values() if x["line"] == n and not x["done"] and not x.get("duplicate") and x.get("next") not in (None, "")
                 and len(x["ids"]) < reach.get(x["channel"], 0) and (not strict or any(carries(x, i) for page in x["paged"][-2:] for i in page))]
        return max(alive, key=lambda x: (self.names_value(x, values), x["way"] == "filter", sum(carries(x, i) for i in x["ids"]) / x["pages"],
                                         -(x["count"] or 0)), default=None)

    def reach(self, channel: dict) -> float:
        facts, schema = self.facts.get(channel["label"]) or {}, channel["page_schema"]
        size, top = facts.get("size", 0), to_int(schema.get("maximum"))
        if not channel["page"]:
            return size
        if "string" in kinds_of(schema) or top is None:
            return float("inf")
        return top * size if facts.get("paging", "page" if (to_int(schema.get("minimum")) or 0) >= 1 else "offset") == "page" else top + size

    def summary(self) -> dict:
        return {c: {"next": f.get("next"), "paging": f.get("paging"), "size": f.get("size"), "cap": f.get("cap"),
                    **{k: self.fact(c, k) for k in ("combine", "order", "filter")}} for c, f in self.facts.items()}

    def shelf_of(self, kind: str) -> str | None:
        names = Counter(v for r in self.rows.values() if isinstance(v := r.get(self.keys.get("shelf", "")), str) and names_kind(v, kind))
        return next(iter(names.most_common(1)), (None,))[0]

    def on_kind(self, n: int, kind: str) -> int:
        return sum(self.on_shelf(i, kind) for i in self.shown(n))

    def coverage(self, n: int) -> dict:
        lists = sorted((x for x in self.lists.values() if x["line"] == n), key=lambda x: x["step"])
        whole = [x for x in lists if x["done"] and not x.get("duplicate") and not x.get("band") and x["count"] is not None and x["count"] != (self.facts.get(x["channel"]) or {}).get("cap")
                 and (x["way"] == "filter" or self.fact(x["channel"], "combine")[0] == "all")]
        recent = sum(x["fresh"] for x in lists[-2:])
        return {"lists": len(lists), "complete": len(whole), "rows": len(self.shown(n)), "new_in_last_two": recent,
                "grade": "C3" if whole else "C2" if len(lists) >= 2 and recent == 0 else "none"}

    def facts_text(self) -> list:
        out = []
        for channel in self.facts:
            notes = [f"{'every word must match' if v == 'all' else 'any one word matches'}" for v, s in [self.fact(channel, "combine")] if s == "confirmed"]
            notes += [f"rows come sorted by {v}" for v, s in [self.fact(channel, "order")] if s == "confirmed"]
            notes += [f"the price limit is {'kept' if v else 'not kept'}" for v, s in [self.fact(channel, "filter")] if s == "confirmed"]
            notes += [f"totals stop at {self.facts[channel]['cap']}"] if self.facts[channel].get("cap") else []
            out += [f"search {channel}: {'; '.join(notes)}"] if notes else []
        return out

    def add_record(self, tool: str, ident: str, payload: dict) -> None:
        if self.records.get(ident, {}).get(tool) != payload:
            self.changes += 1
            self.changed_at[ident] = self.changes
        self.read_at[ident] = self.epoch
        self.records.setdefault(ident, {})[tool] = payload
        self.read_keys.setdefault(tool, set()).update(payload)
        if "stock" not in self.keys:
            flag = next((k for k, v in payload.items() if isinstance(v, bool) and STOCK_KEY.search(k)), None)
            if flag:
                self.keys["stock"] = flag
        if "price" not in self.keys:
            cost = next((k for k, v in payload.items() if k.lower() == "price" and price_value(v) is not None), None) or \
                next((k for k, v in payload.items() if PRICE_KEY.search(k) and not re.search(r"ship|deliver|tax|fee", k, re.I) and price_value(v) is not None), None)
            if cost:
                self.keys["price"] = cost
        self.rows.setdefault(ident, {}).update({k: v for k, v in payload.items() if is_scalar(v) and k in self.keys.values()})

    def facts_of(self, ident: str) -> dict:
        return dict(reversed(self.attributes(ident)))

    def attributes(self, ident: str) -> list:
        return [pair for record in self.records.get(ident, {}).values() for pair in leaves(record)]

    def depth(self, tool: str) -> float:
        rows, recs = {norm(k) for k in self.row_keys}, [r[tool] for r in self.records.values() if tool in r]
        return sum(len({k for k, _ in leaves(r)} - rows) for r in recs) / len(recs) if recs else 0.0

    def menu(self, cards: list, stated: set, kind: list) -> list:
        table: dict = {}
        stated = {bare(k, kind) for k in stated}
        for ident in cards:
            for key, value in self.facts_of(ident).items():
                if key not in {norm(k) for k in self.row_keys} and bare(key, kind) not in stated and value not in self.rows \
                        and value != self.title(ident) and len(value.split()) < TEXT_WORDS:
                    names, values = table.setdefault(bare(key, kind), (set(), {}))
                    names.add(key)
                    values.setdefault(ident, value)
        out = [{"key": min(names, key=lambda k: (len(k), k)), "cards": len(values), "examples": list(dict.fromkeys(values.values()))[:MENU_EXAMPLES],
                "spread": round(1 - Counter(map(norm, values.values())).most_common(1)[0][1] / len(values), 2)} for names, values in table.values()]
        return sorted(out, key=lambda x: (-x["cards"], x["key"]))[:MENU_KEYS]

    def on_shelf(self, ident: str, kind: str) -> bool:
        return names_kind((self.rows.get(ident) or {}).get(self.keys.get("shelf", ""), ""), kind)

    def fit_card(self, ident: str, needs: list) -> str:
        pairs, row = self.attributes(ident), self.rows.get(ident, {})
        own = pairs + [pair for pair in leaves(row) if pair not in pairs]
        named = {i: [f"{k}: {v}" for k, v in own if names_key(k, x["key"])] for i, x in enumerate(needs) if i and norm(x["key"])}
        used = {entry for entries in named.values() for entry in entries}
        texts = list(dict.fromkeys(v for _, v in pairs if len(v.split()) >= TEXT_WORDS and v != self.title(ident)))
        header = {norm(self.keys.get(role, "")) for role in ("id", "title", "price", "stock", "shelf")}
        rest = "; ".join(dict.fromkeys(f"{k}: {v}" for k, v in pairs if k not in header and f"{k}: {v}" not in used and len(v.split()) < TEXT_WORDS and v not in self.rows))
        lines = [f"listing {ident} | {self.title(ident)} | shelf {row.get(self.keys.get('shelf', ''), 'not shown')}"]
        lines += [f"  need {i} ({x['key']} {x['relation']} {x['value']}): {'; '.join(dict.fromkeys(named[i])) or 'not listed'}" for i, x in enumerate(needs) if i in named]
        lines += [f"  text: {clip(t, JUDGE_CHARS)}" for t in texts]
        moved = [f"  {key} (its seller's note sends it to listing {source}, which states): {value}" for key, (value, source) in self.resolved.get(ident, {}).items()]
        return "\n".join(lines + moved + ([f"  other entries: {clip(rest, JUDGE_CHARS)}"] if rest else []))

    def card_text(self, ident: str) -> str:
        lines = [line for rec in [self.rows.get(ident, {}), *self.records.get(ident, {}).values()] for line in card(rec, None, JUDGE_CHARS)]
        moved = [f"{key} (its seller's note sends it to listing {source}, which states): {value}" for key, (value, source) in self.resolved.get(ident, {}).items()]
        return "\n".join([f"listing {ident}", *dict.fromkeys(lines), *moved])

    def of_store(self, store: str, kind: str, fact: str, skip: dict | set = frozenset()) -> list:
        name, facts = words_of(store), set(words_of(fact))
        found = [i for i, row in self.rows.items() if i not in skip and name and any(isinstance(v, str) and len(v.split()) < STORE_WORDS and words_of(v) == name for v in row.values())]
        text = {i: set(words_of(" ".join(scalars([self.rows[i], self.records.get(i, {})], set())))) for i in found}
        return sorted(found, key=lambda i: (not self.on_shelf(i, kind), -len(facts & text[i]), i))[:CHAIN_CARDS]

    def next_reader(self, ident: str, readers: list, second: bool) -> str | None:
        done = set(self.records.get(ident, {}))
        lacking = set().union(*self.read_keys.values()) - self.row_keys - set().union(*map(set, self.records.get(ident, {}).values()))
        fresh = [t for t in readers if t not in self.read_keys and t not in done]
        useful = sorted((t for t in readers if t not in done and self.read_keys.get(t, set()) & lacking), key=lambda t: -len(self.read_keys[t] & lacking))
        spare = [t for t in readers if t not in done] if second and len(done) < 2 else []
        return next(iter(fresh + useful + spare), None)

    def share(self, tool: str, readers: list) -> list | None:
        if tool not in self.read_keys:
            return None
        return sorted(self.read_keys[tool] - self.row_keys - set().union(*(self.read_keys.get(t, set()) for t in readers if t != tool)))

    def ranked(self, n: int, values: list, cap: float | None, kind: str) -> list:
        hits = Counter(i for x in self.lists.values() if x["line"] == n and self.names_value(x, values) for i in x["ids"])
        rows = [i for i in self.shown(n) if self.in_stock(i) is not False and (cap is None or (self.price(i) or 0) <= cap)]
        named = {i for i in rows if self.on_shelf(i, kind)}

        def key(ident: str) -> tuple:
            text, price = self.row_words(ident), self.price(ident)
            return ident not in named, -(hits[ident] + sum(v <= text for v in values)), price if price is not None else float("inf")

        return sorted(rows, key=key)

    def row_words(self, ident: str) -> set:
        """The words of every text a search row holds, at any depth and of any length."""
        texts: list = []
        stack = [self.rows[ident]]
        while stack:
            node = stack.pop()
            if isinstance(node, str):
                texts.append(node)
            elif isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return set(words_of(" ".join(texts)))

    def names_value(self, lst: dict, values: list) -> bool:
        return any(v <= set(lst.get("words", [])) for v in values)

    def price(self, ident: str) -> float | None:
        return price_value((self.rows.get(ident) or {}).get(self.keys.get("price", "")))

    def title(self, ident: str) -> str:
        return clean((self.rows.get(ident) or {}).get(self.keys.get("title", ""), ""))

    def in_stock(self, ident: str) -> bool | None:
        flag = (self.rows.get(ident) or {}).get(self.keys.get("stock", ""))
        return flag if isinstance(flag, bool) else None

    def shown(self, line: int) -> list:
        return [i for i, lines in self.shown_for.items() if line in lines]

    def cheapest(self, line: int, cap: float | None, read: bool = False, skip: frozenset | set = frozenset()) -> str | None:
        pool = [(p, i) for i in self.shown(line) if (p := self.price(i)) is not None and self.in_stock(i) is not False and (cap is None or p <= cap)
                and (i in self.records or not read) and i not in skip]
        return min(pool)[1] if pool else None


class Tools:

    def __init__(self, specs: list) -> None:
        self.specs = specs
        functions = [s.get("function") or s for s in specs if isinstance(s, dict)]
        self.schema = {f["name"]: f.get("parameters") or {} for f in functions if f.get("name")}
        self.description = {f["name"]: clean(f.get("description") or "") for f in functions if f.get("name")}
        self.paths = {name: {p: (sub, req) for p, sub, req in schema_paths(schema)} for name, schema in self.schema.items()}
        self.role, self.confirmed, self.means, self.values, self.stated, self.limits, self.shape = {}, set(), {}, {}, {}, {}, {}
        self.chosen: dict = {}            # a fallback mode value that gave way to the next after showing no rows
        self.refused: dict = {}           # channel label -> times the system's call did not fit the tool

    def brief(self) -> str:
        return "\n".join(f"{name}: {text}" for name, text in self.description.items())

    def mapping_text(self) -> str:
        return "\n\n".join("\n".join([f"tool: {name}", f"description: {self.description[name]}", "arguments:"]
                                     + ([f"  {p}: {describe(sub, req)}" for p, (sub, req) in paths.items()] or ["  none"]))
                           for name, paths in self.paths.items())

    def fill(self, rows: list) -> None:
        for tag, head, f in rows:
            name, _, path = head.partition(".")
            if tag == "TOOL" and head in self.schema and f.get("role") in ROLES:
                self.role[head], self.stated[head] = f["role"], f.get("limit", "none")
            elif tag == "ARG" and path in self.paths.get(name, {}) and f.get("means") in MEANINGS:
                means = f["means"]
                if means == "query" and not self.free_text(name, path):
                    # words go only into free text: a fixed-choice argument taken for the query is the call's mode
                    means = "mode" if self.choices(name, path) else "other"
                self.means[(name, path)] = means
            elif tag == "VALUE" and f.get("means") in VALUE_MEANINGS:
                path, _, value = path.partition("=")
                options = (self.paths.get(name, {}).get(path) or ({}, False))[0].get("enum") or []
                match = next((o for o in options if norm(o) == norm(value)), None)
                if match is not None:
                    self.values.setdefault((name, path), {})[f["means"]] = match
        for name in self.schema:
            # a search tool with no free-text query among its labels: the free-text argument whose name reads as one
            if self.role.get(name) in ("search", "browse") and not self.path(name, "query"):
                texts = [p for p in self.paths[name] if self.free_text(name, p) and self.means.get((name, p)) in (None, "text", "other")]
                pick = next((p for p in texts if QUERY_NAME.search(p.rpartition(".")[2])), None) or (texts[0] if len(texts) == 1 else None)
                if pick:
                    self.means[(name, pick)] = "query"

    def free_text(self, name: str, path: str) -> bool:
        spec = (self.paths.get(name, {}).get(path) or ({}, False))[0]
        return isinstance(spec, dict) and "string" in kinds_of(spec) and not spec.get("enum") and "const" not in spec

    def choices(self, name: str, path: str) -> list:
        spec = (self.paths.get(name, {}).get(path) or ({}, False))[0]
        if not isinstance(spec, dict):
            return []
        return list(spec.get("enum") or ([spec["const"]] if "const" in spec else []))

    def named(self, role: str) -> list:
        return sorted((n for n in self.schema if self.role.get(n) == role), key=lambda n: n not in self.confirmed)

    def path(self, name: str, *meanings: str) -> str | None:
        return next((p for p in self.paths.get(name, {}) if self.means.get((name, p)) in meanings), None)

    def id_values(self, name: str, args: dict) -> list:
        return sorted(scalars([at_path(args, p) for p in self.paths.get(name, {}) if self.means.get((name, p)) in ("id", "ids")], set()))

    def mode(self, name: str, *wanted: str) -> dict | None:
        out = {}
        for path, (_, required) in self.paths[name].items():
            if self.means.get((name, path)) != "mode":
                continue
            labelled = self.values.get((name, path), {})
            value = next((labelled[w] for w in wanted if w in labelled), None)
            if value is not None:
                out[path] = value
            elif required:
                return None
        return out

    def ids_cap(self, name: str) -> int:
        path = self.path(name, "ids", "id")
        schema = to_int((self.paths[name][path][0] if path else {}).get("maxItems")) or 1
        return self.limits.get(name) or min([schema, *(n for n in map(int, re.findall(r"\d+", self.stated.get(name, ""))) if n > 0)])

    def default_modes(self, name: str, taken: set) -> dict | None:
        """Every required argument of a search tool that offers a fixed set of values and is not one of the channel's own
        arguments: the value whose name reads as a text search, else the first. None when such an argument offers none."""
        out = {}
        for path, (spec, required) in self.paths.get(name, {}).items():
            if not required or path in taken:
                continue
            options = self.choices(name, path)
            if not options:
                continue
            labelled = self.values.get((name, path), {})
            if self.means.get((name, path)) == "sort":
                out[path] = labelled.get("cheapest_first", options[0])
                continue
            others = {v for k, v in labelled.items() if k != "words"}
            options = [o for o in options if o not in others] or options
            chosen = self.chosen.get((name, path))
            out[path] = chosen if chosen in options else next((o for o in options if isinstance(o, str) and WORD_MODE.search(o)), options[0])
        return out

    def channels(self) -> list:
        out = []
        for name in self.schema:
            paths = {m: self.path(name, m) for m in ("query", "page", "shelf", "price_max", "sort")}
            for way, needed in (("words", "query"), ("filter", "shelf")):
                mode = self.mode(name, way)
                fallback = None
                if way == "words" and self.role.get(name) in ("search", "browse") and paths["query"]:
                    taken = {p for m, p in paths.items() if p and m != "sort"} | set(mode or {})
                    fallback = self.default_modes(name, taken)
                    mode = None if mode is None and fallback is None else {**(fallback or {}), **(mode or {})}
                label = f"{name} ({way})"
                if self.role.get(name) in ("search", "browse") and paths[needed] and mode is not None and self.refused.get(label, 0) < 2:
                    out.append({"tool": name, "way": way, "mode": mode, "label": label, **paths, "fallback": sorted(fallback or {}),
                                "page_schema": self.paths[name][paths["page"]][0] if paths["page"] else {}})
        return out

    def channel_of(self, name: str, args: dict) -> dict | None:
        return next((c for c in self.channels() if c["tool"] == name and all(at_path(args, p) == [v] for p, v in c["mode"].items())
                     and (c["way"] == "words") == bool(c["query"] and at_path(args, c["query"]))), None)

    def option(self, name: str, path: str, text: str | None, kind: str) -> str | None:
        options = self.paths[name][path][0].get("enum")
        return next((o for o in options if names_kind(o, kind)), None) if options else text

    def cart_calls(self, name: str, target: dict, current: dict, adding: bool) -> list:
        id_path, qty_path = self.path(name, "id", "ids"), self.path(name, "quantity")
        put, remove = self.mode(name, "set", "add"), self.mode(name, "remove") or None
        if id_path is None or put is None:
            return []
        box = id_path.rpartition("[].")[0]

        def call(lines: list, mode: dict) -> dict | None:
            args: dict = {}
            if box:
                place(args, box, [{} for _ in lines])
                for item, (ident, qty) in zip(at_path(args, box + "[]"), lines):
                    place(item, id_path[len(box) + 3:], ident)
                    if qty_path and qty_path.startswith(box + "[]."):
                        place(item, qty_path[len(box) + 3:], qty)
            else:
                place(args, id_path, [lines[0][0]] if self.means[(name, id_path)] == "ids" else lines[0][0])
                if qty_path:
                    place(args, qty_path, lines[0][1])
            for path, value in mode.items():
                place(args, path, value)
            fitted, problems = fit_object(self.schema[name], args)
            return None if problems else fitted

        if box:
            calls = [call(list(target.items()), put)] if target else []
            calls += [call([(i, q)], remove) for i, q in current.items() if i not in target and adding and remove is not None]
            return [c for c in calls if c]
        gone = {i: q for i, q in current.items() if i not in target or (adding and target[i] < q)}
        calls = [call([(i, q)], remove) if remove is not None else call([(i, 0)], put) for i, q in gone.items()]
        for ident, qty in target.items():
            have = 0 if ident in gone else current.get(ident, 0)
            if have != qty:
                calls.append(call([(ident, qty - have if adding else qty)], put))
        return [c for c in calls if c]

    def finish_call(self, kind: str, text: str) -> tuple[str, dict] | None:
        for name in self.named(kind) + self.named("finish"):
            mode = self.mode(name, kind)
            if mode is None or (self.role[name] == "finish" and not mode):
                continue
            args: dict = {}
            for path, value in mode.items():
                place(args, path, value)
            for path in self.paths[name]:
                if self.means.get((name, path)) == "text":
                    place(args, path, text)
            fitted, problems = fit_object(self.schema[name], args)
            if not problems:
                return name, fitted
        return None

    def observe(self, name: str, kind: str, finished: bool) -> None:
        got, role = "end" if finished else kind, self.role.get(name)
        if role in EXPECTED and (EXPECTED[role] == got or (role == "read" and got == "rows")):
            self.confirmed.add(name)

    def roles_text(self) -> list:
        free = [n for n in self.schema if self.role.get(n) not in SYSTEM_ROLES]
        system = [f"{n} ({self.role[n]})" for n in self.schema if self.role.get(n) in SYSTEM_ROLES]
        facts = [f"{n} works as {self.role[n]}" for n in self.schema if n in self.confirmed]
        limits = [f"{n}: {text}" for n, text in self.stated.items() if norm(text) not in ("", "none")]
        return [f"tools you may call: {', '.join(free) or 'none'}", f"tools the system calls: {', '.join(system) or 'none'}",
                f"confirmed by results: {'; '.join(facts) or 'nothing yet'}"] + ([f"limits the descriptions state: {'; '.join(limits)}"] if limits else [])


class Cart:

    def __init__(self) -> None:
        self.lines, self.total, self.version, self.viewed, self.adding = [], None, 0, -1, False

    def counts(self) -> dict:
        return {i: q for i, q, _, _ in self.lines}

    def show(self, payload: Any, catalog: Catalog, wanted: dict) -> set | None:
        _, rows = row_list(payload)
        if not rows and not any(isinstance(v, list) for _, v in walk(payload)):
            return None
        lines, key, moved = [], catalog.keys.get("price", ""), set()
        for row in rows:
            values = [v for v in row.values() if is_scalar(v) and not isinstance(v, bool)]
            ident = next((str(v) for v in values if str(v) in catalog.rows or str(v) in wanted), None)
            if ident is None:
                continue
            if (shown := price_value(row.get(key))) is not None:
                moved |= {ident} if catalog.price(ident) not in (None, shown) else set()
                catalog.rows.setdefault(ident, {})[key] = row[key]
            price = catalog.price(ident)
            counts = [v for v in values if isinstance(v, int) and v >= 1 and v != price]
            lines.append((ident, wanted[ident] if wanted.get(ident) in counts else min(counts, default=1), price, catalog.title(ident)))
        computed = sum(q * p for _, q, p, _ in lines if p is not None)
        outside = [price_value(v) for p, v in walk(payload) if "[]" not in p and is_scalar(v) and not isinstance(v, bool)]
        self.lines, self.viewed = lines, self.version
        self.total = next((x for x in outside if x is not None and abs(x - computed) < 0.005), computed)
        return moved

    def assume(self, wanted: dict, catalog: Catalog) -> None:
        self.lines = [(i, q, catalog.price(i), catalog.title(i)) for i, q in wanted.items()]
        self.total = sum(q * p for _, q, p, _ in self.lines if p is not None)

    def report(self, money: Any) -> str:
        items = "; ".join(ITEM_TEXT.format(qty=q, title=t or i, ident=i, price=money(p)) for i, q, p, t in self.lines)
        return REPORT_TEXT.format(items=items, total=money(self.total))


class Budget:

    def __init__(self, max_steps: int, clock: Clock, llm: Llm, env: Env) -> None:
        self.max_steps, self.clock, self.llm, self.env = max_steps, clock, llm, env
        self.messages, self.sent, self.reserve, self.report = None, 0, RESERVE_ORDER + RESERVE_RECOVER, False

    def plan(self, request: Request, messaging: bool) -> None:
        self.messages, self.report = request.messages, messaging and any(d["by"] != "tool" for d in request.duties)
        self.reserve += RESERVE_MESSAGE if self.report else 0
        self.reserve += int(any(request.goal(n) in GOALS_CHEAPEST for n in request.lines))

    def allowance(self, planned_calls: int) -> int:
        step_seconds = self.env.seconds / self.env.step if self.env.step >= TIMING_SAMPLES else STEP_SECONDS
        spare = self.clock.left() - SAFE_LEFT - planned_calls * CALL_SECONDS
        return max(self.env.step, min(self.max_steps, self.env.step + int(spare / max(step_seconds, 0.05))))

    def room(self) -> int:
        return self.max_steps - self.env.step

    def messages_left(self) -> int | None:
        return None if self.messages is None else self.messages - self.sent

    def text(self) -> str:
        messages = f"{self.sent} sent of {self.messages}" if self.messages is not None else f"{self.sent} sent, budget not stated"
        time = f" | time left about {int(self.clock.left() // 30 * 30)} s" if self.clock.elapsed() > DEADLINE / 2 else ""
        return f"steps: {self.env.step} used of {self.max_steps}, {self.reserve} kept for the cart and the finish | messages: {messages}{time}"


class Seats:

    def __init__(self, llm: Llm) -> None:
        self.llm = llm

    def rows(self, seat: str, system: str, user: str, tags: tuple) -> list:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(2):
            answer = self.llm.complete(seat, messages)
            if answer is None:
                return []
            found = list({dump(r): r for r in rows_of(text_of(answer)) if r[0] in tags}.values())
            if found:
                return found
            messages = messages + [{"role": "assistant", "content": text_of(answer) or "-"}, {"role": "user", "content": FORMAT_AGAIN}]
        return []

    def read(self, request: str, tools: str) -> list:
        return self.rows("read", READ_PROMPT, f"<request>\n{request}\n</request>\n<tools>\n{tools}\n</tools>", READ_TAGS)

    def map(self, tools: str) -> list:
        return self.rows("map", MAP_PROMPT, f"<tools>\n{tools}\n</tools>", MAP_TAGS)

    def ask(self, request: str, lines: str, menus: str, count: int) -> list:
        user = f"<request>\n{request}\n</request>\n<lines>\n{lines}\n</lines>\n<menus>\n{menus}\n</menus>\n<count>\n{count}\n</count>"
        return self.rows("ask", ASK_PROMPT, user, ASK_TAGS)

    def link(self, need: str, chain: str, listings: str, problem: str) -> dict:
        user = f"<need>\n{need}\n</need>\n<chain>\n{chain or 'none'}\n</chain>\n<listings>\n{listings}\n</listings>\n<problem>\n{problem or 'none'}\n</problem>"
        return next(({"target": head, **f} for _, head, f in self.rows("link", LINK_PROMPT, user, LINK_TAGS)), {})

    def part_ids(self, listings: str) -> dict:
        return {head: {norm(x) for x in f.get("ids", "").split(",") if norm(x) not in ("", "none")}
                for _, head, f in self.rows("ids", IDS_PROMPT, f"<listings>\n{listings}\n</listings>", IDS_TAGS)}

    def audit(self, request: str, lines: str, evidence: str) -> dict:
        user = f"<request>\n{request}\n</request>\n<lines>\n{lines}\n</lines>\n<evidence>\n{evidence}\n</evidence>"
        return next(({"verdict": norm(head), **f} for _, head, f in self.rows("audit", AUDIT_PROMPT, user, AUDIT_TAGS)), {})

    def fit(self, product: str, listings: str) -> list:
        return self.rows("fit", FIT_PROMPT, f"<product>\n{product}\n</product>\n<listings>\n{listings}\n</listings>", FIT_TAGS)

    def qty(self, request: str, lines: str) -> list:
        return self.rows("qty", QTY_PROMPT, f"<request>\n{request}\n</request>\n<lines>\n{lines}\n</lines>", QTY_TAGS)

    def hear(self, sent: str, said: str, lines: str) -> list:
        user = f"<sent>\n{sent or 'nothing'}\n</sent>\n<said>\n{said}\n</said>\n<lines>\n{lines}\n</lines>"
        kinds = {norm(k): k for k in HEARD_KINDS}
        return [{"kind": kinds[norm(head)], **f} for _, head, f in self.rows("hear", HEAR_PROMPT, user, HEAR_TAGS) if norm(head) in kinds]

    def assistant(self, request: str, state: str, last: str, goal: str, tools: list) -> tuple[str, dict] | None:
        user = f"<request>\n{request}\n</request>\n<state>\n{state}\n</state>\n<last>\n{last or 'no call yet'}\n</last>\n<goal>\n{goal}\n</goal>"
        answer = self.llm.complete("assistant", [{"role": "system", "content": ASSISTANT_PROMPT}, {"role": "user", "content": user}], tools)
        function = ((answer or {}).get("tool_calls") or [{}])[0].get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            args = None
        return (function["name"], args) if function.get("name") and isinstance(args, dict) else None


class Listener:

    def prepare(self) -> None:
        read, mapped = together(lambda: self.seats.read(self.request.numbered, self.tools.brief()), lambda: self.seats.map(self.tools.mapping_text()))
        self.request.fill(read or [])
        if doubtful := self.request.doubtful_quantities() if read else []:
            lines = "\n".join(f"line {n}: {self.request.lines[n]['kind']} | qty {self.request.lines[n]['qty']} | from: \"{self.request.lines[n]['qty_quote'] or 'none'}\"" for n in doubtful)
            self.request.requantify(self.seats.qty(self.request.numbered, lines))
        self.tools.fill(mapped or [])
        self.record(map=[f"{tag} {head} " + " ".join(f"{k}={v}" for k, v in f.items()) for tag, head, f in (mapped or [])][:60])
        self.budget.plan(self.request, bool(self.tools.named("message")))
        self.record(route=self.request.route, lines=len(self.request.lines), read=bool(read), mapped=bool(mapped), reserve=self.budget.reserve)

    def hear(self, said: str, sent: str = "", n: int | None = None) -> list:
        self.noted.update(map(norm, said.splitlines()))
        picked = "\n".join(f"line {n} picked: {ident} {self.catalog.title(ident)}" for n, (ident, _) in self.picks().items())
        heard = self.seats.hear(sent, said, "\n".join(filter(None, [self.request.needs_text(), picked]))) if self.clock.left() > SAFE_LEFT / 2 else []
        for h in heard:
            figure = to_number(h.get("amount"))
            if self.request.hear(h, n):
                self.grown.add(number_of(h.get("line", "")) or n)
            elif h["kind"] == "budget" and figure in numbers_in(said):
                self.request.rebudget(figure)
                self.deep.clear()
            elif h["kind"] == "instruction" and norm(h.get("text")) not in {norm(d["text"]) for d in self.request.duties}:
                self.request.duties.append({"text": h.get("text", ""), "when": "before_finish", "by": "message", "done_at": None})
            elif h["kind"] == "drop":
                self.request.drop(number_of(h.get("line", "")) or n, h.get("key", ""))
                self.deep.clear()
            elif h["kind"] in ("stock", "price"):
                ident, line = clean(h.get("value")), number_of(h.get("line", ""))
                picks = self.picks()
                scope = {ident} if ident in self.catalog.rows else {picks[line][0]} if line in picks else set(self.cart.counts()) | set(self.target)
                if h["kind"] == "stock" and not any(note in said for note in self.vouched.get(ident, ())):
                    self.excluded.update(scope & {ident})
                self.event(h["kind"], scope)
        self.notices = [x for x in self.notices if x not in said]
        self.stale = self.stale or any(h["kind"] == "price" for h in heard)
        self.record(heard=[(h["kind"], h.get("line"), h.get("key"), h.get("value")) for h in heard])
        return heard

    def event(self, kind: str, scope: set) -> None:
        self.catalog.epoch += 1
        self.suspect |= scope
        self.deep -= {n for n in self.request.lines if scope & set(self.cards.get(n, []))}
        self.request.version += 1
        self.record(event=kind, scope=sorted(scope))

    def news(self) -> bool:
        if not self.notices:
            return False
        version = self.request.version
        self.hear("\n".join(self.notices))
        return self.request.version != version or self.stale


class Searcher:

    def untried(self, n: int) -> list:
        return [p for p in self.request.queries(n) if dump(p[0]) not in self.tried.get(n, set())]

    def next_query(self, n: int) -> None:
        words = next((c for c in self.tools.channels() if c["way"] == "words"), None)
        plan = next(iter(self.untried(n)), None)
        if plan is not None and words is None:
            # no search the system can send (none mapped, or set aside after refusals): the assistant seat searches,
            # while the line shows no row
            self.tried.setdefault(n, set()).add(dump(plan[0]))
            if not self.catalog.shown(n):
                self.free_find(n)
            return
        if plan is None or words is None:
            self.grown.discard(n)
            return
        self.tried.setdefault(n, set()).add(dump(plan[0]))
        self.try_plan(words, n, plan)

    def deepen(self, n: int) -> None:
        channels = self.tools.channels()
        words, filters = next((c for c in channels if c["way"] == "words"), None), [c for c in channels if c["way"] == "filter"]
        kind, state = self.request.lines[n]["kind"], self.deep_state.setdefault(n, {"wide": 0, "free": 0, "shelf": False, "rows": 0})
        if self.untried(n):
            return self.next_query(n)
        wide = [w for w in self.request.widen(n) if words and dump(w) not in self.tried.get(n, set())]
        if wide and self.catalog.on_kind(n, kind) < WIDE_ROWS:
            self.tried.setdefault(n, set()).add(dump(wide[0]))
            self.search(words, n, wide[0])
            return None
        if not self.catalog.shown(n) and state["free"] < FREE_STEPS:
            state["free"] += 1
            self.free_turn(GOAL_FIND.format(n=n, kind=kind or "see the request"), n)
            return None
        if not state["shelf"] and filters:
            state["shelf"] = True
            for channel in filters:
                shelf = self.tools.option(channel["tool"], channel["shelf"], self.catalog.shelf_of(kind) or self.request.lines[n]["shelf"] or None, kind)
                if shelf and self.search(channel, n, shelf=shelf) and self.request.goal(n) in GOALS_CHEAPEST and not any(m == n for m, _ in self.undercut):
                    self.probe(channel, n, shelf)
            return None
        values = self.request.value_sets(n)
        reach = {c["label"]: self.catalog.reach(c) for c in channels}
        lst = self.catalog.to_page(n, values, kind, reach) if state["rows"] < (COVER_ROWS if values else COVER_PLAIN) else None
        rows = len(lst["ids"]) if lst else 0
        if lst and self.follow(lst):
            state["rows"] += max(1, len(lst["ids"]) - rows)
            return None
        self.deep.add(n)
        return None

    def try_plan(self, channel: dict, n: int, plan: tuple) -> None:
        words, alone = plan
        lst = self.search(channel, n, words)
        if alone and lst is not None and not lst["ids"]:
            self.search(channel, n, alone)

    def search(self, channel: dict, n: int, words: list | None = None, shelf: str | None = None, limit: float | None = None) -> dict | None:
        tool, args = channel["tool"], {}
        for path, value in channel["mode"].items():
            place(args, path, value)
        if words:
            place(args, channel["query"], " ".join(words))
        if shelf:
            place(args, channel["shelf"], shelf)
        cheapest_first = self.tools.values.get((tool, channel["sort"]), {}).get("cheapest_first") if channel["sort"] else None
        if cheapest_first is not None and self.request.goal(n) in GOALS_CHEAPEST:
            place(args, channel["sort"], cheapest_first)
        cap, price = limit if limit is not None else self.request.cap(n), channel["price_max"]
        capped = cap is not None and price is not None and not fit_value(self.tools.paths[tool][price][0], int(cap) if float(cap).is_integer() else cap, price)[1]
        if capped:
            place(args, price, int(cap) if float(cap).is_integer() else cap)
        obs = self.page(tool, args, n)
        if obs is None and self.refusal:
            self.hand_search(channel, n)
            return None
        if capped and limit is None and obs is not None and not row_list(unwrap(obs)[0])[1] and self.catalog.fact(channel["label"], "filter")[1] != "confirmed":
            args.pop(price, None)
            obs = self.page(tool, args, n)
        lst = self.catalog.lists.get(self.list_key(channel, args))
        if lst is not None and limit is not None and (self.request.cap(n) is None or limit < self.request.cap(n)):
            lst["band"] = True
        if words and channel.get("fallback") and lst is not None and not lst["ids"] and channel["label"] not in self.mode_tried:
            # a mode value the map did not name showed no rows: the next value is tried once, and kept only if it shows rows
            self.mode_tried.add(channel["label"])
            before = {p: channel["mode"][p] for p in channel["fallback"]}
            for path, value in before.items():
                rest = [o for o in self.tools.choices(tool, path) if o != value]
                if rest:
                    self.tools.chosen[(tool, path)] = rest[0]
            again = next((c for c in self.tools.channels() if c["label"] == channel["label"]), None)
            if again and again["mode"] != channel["mode"]:
                retry = self.search(dict(again, price_max=None), n, words, shelf, limit)
                if retry is not None and retry["ids"]:
                    return retry
            for path, value in before.items():
                self.tools.chosen[(tool, path)] = value
        return lst

    def list_key(self, channel: dict, args: dict) -> str:
        return dump([channel["tool"], {k: v for k, v in fit_object(self.tools.schema[channel["tool"]], args)[0].items() if k != channel["page"]}])

    def probe(self, channel: dict, n: int, shelf: str) -> None:
        top = self.request.cap(n) or max((p for i in self.catalog.shown(n) if (p := self.catalog.price(i)) is not None), default=None)
        if channel["price_max"] is None or not top:
            return
        low, high, found = 0.0, float(top), None
        for _ in range(PROBES):
            lst = self.search(channel, n, shelf=shelf, limit=round(high)) if self.room() else None
            if not lst or lst.get("count") is None:
                break
            if 0 < lst["count"] <= PROBE_ROWS:
                found = lst
                break
            low, high = (low, (low + high) / 2) if lst["count"] else (high, high + (high - low))
        while found and not found["done"] and self.room() and self.follow(found):
            pass

    def follow(self, lst: dict) -> bool:
        channel = next(c for c in self.tools.channels() if c["label"] == lst["channel"])
        args = json.loads(dump(lst["args"]))
        place(args, channel["page"], lst["next"])
        if self.page(lst["tool"], args, lst["line"]) is None:
            lst["done"] = True
            return False
        return True

    def page(self, tool: str, args: dict, n: int) -> Any:
        self.refusal = ""
        fitted, problems = fit_object(self.tools.schema[tool], args)
        if problems:
            # the system's own call does not fit the tool: nothing is sent, and the search goes to the assistant seat
            self.refusal = f"{tool} {dump(fitted)}: " + "; ".join(problems[:3])
            self.blocked.append(self.refusal)
            self.record(blocked=tool, reason="; ".join(problems[:3]))
            return None
        if self.done_now(tool, fitted) or not self.room():
            return None
        return self.send(tool, fitted, n)

    def hand_search(self, channel: dict, n: int) -> None:
        """The system's search did not fit its tool: the channel is counted against (two refusals set it aside) and the
        assistant seat searches for the line at once, told why (up to FREE_STEPS turns a line)."""
        refused, self.refusal = self.refusal, ""
        self.tools.refused[channel["label"]] = self.tools.refused.get(channel["label"], 0) + 1
        self.free_find(n, refused)

    def free_find(self, n: int, refused: str = "") -> bool:
        """The line's search by the assistant seat (up to FREE_STEPS turns a line), told why the system's did not go."""
        state = self.deep_state.setdefault(n, {"wide": 0, "free": 0, "shelf": False, "rows": 0})
        if state["free"] >= FREE_STEPS or not self.room():
            return False
        state["free"] += 1
        kind = self.request.lines[n]["kind"]
        why = refused or (self.blocked[-1] if self.blocked else "")
        goal = GOAL_FIND.format(n=n, kind=kind or "see the request")
        if why:
            goal += f"; the system's own call did not fit the tool ({why}), so make the search with arguments the tool accepts"
        return self.free_turn(goal, n)

    def search_by_hand(self) -> bool:
        """No store call made yet and no line shows a row: before any finish, the assistant seat searches line by line (up to
        FREE_STEPS turns a line), told what the system's calls were refused for. True when a call went out."""
        if self.catalog.rows or self.free_steps() <= 0:
            return False
        sent = False
        for n in sorted(self.request.lines):
            kind = self.request.lines[n]["kind"]
            for _ in range(FREE_STEPS):
                if self.catalog.shown(n) or self.free_steps() <= 0 or self.must_close():
                    break
                if not self.free_find(n):
                    break
                sent = True
        return sent


class Reader:

    def unread(self, n: int, count: int) -> list:
        ranked = self.catalog.ranked(n, self.request.value_sets(n), self.request.cap(n), self.request.lines[n]["kind"])
        return self.screened(n, [i for i in ranked if i not in self.cards.get(n, [])])[:count]

    def screened(self, n: int, ranked: list) -> list:
        """The rows in reading order: a row whose words name another value of a property the line needs, and not the value
        wanted, goes after the others. A property's values are the ones the listings read so far show for it."""
        wants = [(x["key"], set(words_of(x["value"]))) for x in self.request.lines[n]["needs"]
                 if norm(x["relation"]) in ("equal", "contains") and norm(x["key"]) not in ("", "none") and words_of(x["value"])]
        if not wants or len(ranked) < 2:
            return ranked
        others = [(want, self.values_seen(key, want)) for key, want in wants]
        clashing = set()
        for ident in ranked:
            words = self.catalog.row_words(ident)
            if any(not want <= words and any(value <= words for value in seen) for want, seen in others):
                clashing.add(ident)
        if clashing:
            self.record(screened=n, last=sorted(clashing)[:6])
        return [i for i in ranked if i not in clashing] + [i for i in ranked if i in clashing]

    def values_seen(self, key: str, want: set) -> list:
        """The other values listings read so far give the property `key`: each a few words of three letters or more, none
        sharing a word with the value wanted."""
        out: list = []
        for ident in self.catalog.records:
            for name, value in self.catalog.facts_of(ident).items():
                if not isinstance(value, str) or not names_key(str(name), key):
                    continue
                for part in re.split(r"[,/;|]", value):
                    words = {w for w in words_of(part) if len(w) >= 3 and not w.isdigit()}
                    if words and len(words) <= 3 and not words & want and words not in out:
                        out.append(words)
        return out

    def cheaper(self, n: int, ident: str) -> list:
        values, kind, price = self.request.value_sets(n), self.request.lines[n]["kind"], self.catalog.price(ident)
        named = {i for x in self.catalog.lists.values() if x["line"] == n and self.catalog.names_value(x, values) for i in x["ids"]}
        near = (lambda i: self.catalog.on_shelf(i, kind)) if "shelf" in self.catalog.keys else named.__contains__
        unread = self.unread(n, len(self.catalog.shown(n)))
        rows = [i for i in unread if near(i) and price is not None and (self.catalog.price(i) or 0) < price]
        title = self.catalog.title(ident)
        copies = [i for i in unread if i not in rows and title and price is not None and self.catalog.price(i) == price and self.catalog.title(i) == title]
        read = sum(1 for i in self.cards.get(n, []) if i != ident and self.catalog.price(i) == price and self.catalog.title(i) == title)
        return sorted(rows, key=lambda i: (i not in named, self.catalog.price(i) or 0)) + copies[:max(0, STAND_INS - read)]

    def read_next(self, n: int) -> None:
        tool = self.table_tool()
        many = tool is not None and self.tools.means.get((tool, self.tools.path(tool, "ids", "id"))) == "ids"
        if batch := self.unread(n, self.tools.ids_cap(tool) if many else 1):
            self.read_into(n, batch)

    def read_into(self, n: int, ids: list, room: Any = None) -> None:
        self.cards.setdefault(n, []).extend(i for i in ids if i not in self.cards.get(n, []))
        if self.table_tool() is None:
            self.read_fresh(ids[0], n)
        tool = self.table_tool()
        todo = [i for i in ids if tool and tool not in self.catalog.records.get(i, {})]
        room = room or self.room
        if todo and self.tools.means.get((tool, self.tools.path(tool, "ids", "id"))) == "ids":
            self.bundle(tool, todo, n, room)
        for ident in todo if todo and self.tools.means.get((tool, self.tools.path(tool, "ids", "id"))) != "ids" else []:
            if not room() or self.read(tool, [ident], n) is None:
                return

    def table_tool(self) -> str | None:
        if any(t not in self.catalog.read_keys for t in self.readers()):
            return None
        readers = [t for t in self.readers() if self.catalog.depth(t) > 0]
        best = max(readers, key=self.catalog.depth, default=None)
        return next((t for t in self.tools.named("compare") if t in readers and self.catalog.depth(t) >= BUNDLE_DEPTH * self.catalog.depth(best)), best)

    def read_fresh(self, ident: str, n: int) -> None:
        while any(t not in self.catalog.read_keys for t in self.readers()) and self.room():
            tool = self.catalog.next_reader(ident, self.readers(), False)
            if tool is None or self.read(tool, [ident], n) is None:
                return

    def read_lines(self, ids: list) -> None:
        compare = next(iter(self.tools.named("compare")), None)
        if compare:
            self.bundle(compare, ids, None)
        for ident in ids if not compare else []:
            tool = self.catalog.next_reader(ident, self.readers(), False)
            if tool is None or not self.room() or self.read(tool, [ident], None) is None:
                return

    def readers(self) -> list:
        return [t for t in self.tools.named("read") + self.tools.named("compare") if self.tools.path(t, "ids", "id")
                and (t in self.catalog.read_keys or self.misses[t] < 2)]

    def room(self) -> bool:
        return not self.must_close() and self.clock.left() > P3_LEFT and (self.pass_end is None or self.env.step < self.pass_end)

    def limit_now(self) -> str:
        which = ("this pass's allowance" if self.pass_end is not None and self.env.step >= self.pass_end else "the closing reserve" if self.must_close()
                 else "time" if self.clock.left() <= P3_LEFT else "none")
        return f"{which}; {self.budget.room()} steps, {int(self.clock.left())} s, {self.budget.messages_left()} messages, ${self.llm.spend:.3f} spent"

    def bundle(self, tool: str, ids: list, n: int, room: Any = None) -> None:
        queue, room = [i for i in ids if tool not in self.catalog.records.get(i, {})], room or self.room
        while queue and room():
            chunk, queue = queue[:self.tools.ids_cap(tool)], queue[self.tools.ids_cap(tool):]
            obs = self.read(tool, chunk, n)
            if obs is not None and not row_list(unwrap(obs)[0])[1] and len(chunk) > 1:
                self.tools.limits[tool] = len(chunk) // 2
                queue = chunk + queue

    def read(self, tool: str, ids: list, n: int | None) -> Any:
        path = self.tools.path(tool, "ids", "id")
        if path is None:
            return None
        args: dict = {}
        place(args, path, ids if "array" in kinds_of(self.tools.paths[tool][path][0]) else ids[0])
        fitted, problems = fit_object(self.tools.schema[tool], args)
        self.misses[tool] += bool(problems)
        if problems or self.done_now(tool, fitted):
            return None
        obs = self.send(tool, fitted, n)
        if obs is not None:
            for ident in ids:
                self.read_by[(tool, str(ident))] = self.cart.version
        return obs

    def complete(self, ident: str, n: int | None, second: bool, room: Any) -> None:
        while room():
            tool = self.catalog.next_reader(ident, self.readers(), second)
            if tool is None or self.read(tool, [ident], n) is None:
                return


class Asker:

    def ask_next(self) -> None:
        """One message to the shopper. The plan holds windows of a line's properties; the shopper answers about the first
        of its points a message names, so what stands behind that point goes unheard. After an answer about a property,
        the properties behind it go out next, topped up from the line's next window; after a reply that names none (a
        point the shopper does not mind about, or none named), the window goes out once more reversed."""
        if self.pending is None:
            self.menus = {n: self.catalog.menu(self.cards.get(n, []), self.request.stated(n), self.request.kind_words(n)) for n in sorted(self.request.lines)}
            self.pending = self.plan_questions(self.menus)
            self.record(planned=list(self.pending))
        if not self.pending:
            unclear = next((k for k, v in self.request.answers.items() if v == "unclear" and k not in self.retried and k[0] in self.request.lines), None)
            if unclear:
                self.retried.add(unclear)
                self.put(unclear[0], (unclear[1],))
            return
        (n, keys), version = self.pending.pop(0), self.request.version
        keys = self.fitted(n, keys)
        kinds = self.put(n, keys)
        if kinds is None:
            self.pending = []
            return
        listed = len(keys) > 1 and not self.lists_proven
        self.list_blanks = self.list_blanks + 1 if listed and not kinds & {"disclose", "no_need"} else 0
        if listed and ("one_at_a_time" in kinds or ("not_understood" in kinds and "disclose" not in kinds) or self.list_blanks >= 2):
            # the shopper wants one thing at a time, did not follow a list, or met two lists with neither an answer nor a
            # "no preference": every window left goes out property by property, this one first, and none reversed
            self.window = 1
            self.pending = [(n, (k,)) for k in keys] + [(m, (k,)) for m, ks in self.pending for k in ks]
            self.record(lists="off")
            return
        # an answer about the first property of a window leaves the rest unheard, and the window reversed would stop at
        # the last point it names, which may be one the shopper does not mind, in front of the one it needs
        answered = next((i for i, k in enumerate(keys) if self.request.answers.get((n, k)) == "disclosed"), None)
        back = self.turned_window(n, keys) if answered is None else ()
        if self.request.version > version and self.pending and not self.replanned and self.questions() > 0:
            self.replanned, self.pending = True, self.plan_questions(self.menus)
        if answered is not None:
            behind = self.behind(n, keys[answered + 1:])
            if behind:
                self.pending.insert(self.turn_slot(n), (n, behind))
                self.record(behind=list(behind))
        elif back:
            self.pending.insert(self.turn_slot(n), (n, back))

    def behind(self, n: int, rest: tuple) -> tuple:
        """The properties a message named behind the one the shopper answered about, never heard, topped up to a window from
        the line's next planned window (which keeps what is left of it)."""
        keys = [k for k in rest if self.request.answers.get((n, k)) != "disclosed" and not self.said_already(n, k)]
        later = next((i for i, (m, _) in enumerate(self.pending) if m == n), None)
        if later is not None:
            more = [k for k in self.pending[later][1] if k not in keys]
            room = max(0, self.window - len(keys))
            keys, left = keys + more[:room], tuple(more[room:])
            if left:
                self.pending[later] = (n, left)
            else:
                del self.pending[later]
        return tuple(keys[:max(1, self.window)])

    def said_already(self, n: int, key: str) -> bool:
        """Whether the line already holds this property, from the request or an answer, under its name or a close one: a
        request that states a color family is not asked about color, which the shopper would answer first and waste the
        message on. A held name that is one generic word ("type", "size", "style", "kind") stands only for itself: a
        stated type keeps "display type" and "clock type" askable."""
        return any(norm(key) == held if held in GENERIC_HELD else names_key(key, held) for held in self.request.stated(n))

    def turned_window(self, n: int, keys: tuple) -> tuple:
        """The same window reversed, without what an answer settled, when it was asked forward and two properties are
        left; else nothing."""
        if len(keys) < 2 or self.window < 2 or (n, keys) in self.turned:
            return ()
        back = tuple(k for k in reversed(keys) if self.request.answers.get((n, k)) != "disclosed" and not self.said_already(n, k))
        if len(back) < 2:
            return ()
        self.turned.add((n, back))
        return back

    def turn_slot(self, n: int) -> int:
        """Where a reversed window goes in the plan: after the first windows of the lines not asked about yet, so each line
        gets a question before any line gets its second."""
        asked = {m for m, _ in self.request.asked}
        slot = 0
        for m, _ in self.pending:
            if m != n and m not in asked:
                slot += 1
            else:
                break
        return slot

    def fitted(self, n: int, keys: tuple) -> tuple:
        """The window cut to what one message can carry (the message argument's maxLength), so that no property is cut off
        in the middle of the text; the properties left out lead the line's next window."""
        limit = self.message_limit()
        size = len(keys)
        while limit and size > 1 and len(self.question_text(n, keys[:size])) > limit:
            size -= 1
        rest = keys[size:]
        if rest:
            later = next((i for i, (m, _) in enumerate(self.pending) if m == n), None)
            if later is None:
                self.pending.append((n, rest))
            else:
                self.pending[later] = (n, rest + tuple(k for k in self.pending[later][1] if k not in rest))
            self.record(window_cut=list(rest))
        return keys[:size]

    def message_limit(self) -> int | None:
        tool = next(iter(self.tools.named("message")), None)
        path = self.tools.path(tool, "text") if tool else None
        limit = ((((self.tools.schema.get(tool) or {}).get("properties") or {}).get(path) or {}) if path else {}).get("maxLength")
        return limit if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 else None

    def question_text(self, n: int, keys: tuple) -> str:
        kind = self.request.lines[n]["kind"] or "product"
        said = [self.spoken(k) for k in keys]
        return LIST_TEXT.format(kind=kind, keys=", ".join(said[:-1]), last=said[-1]) if len(keys) > 1 else QUESTION_TEXT.format(kind=kind, key=said[0])

    def put(self, n: int, keys: tuple) -> set | None:
        if self.llm.key_spent:
            # the run's key pays the shopper's reader too: a message now would fail the store's own reading
            return None
        tool = next(iter(self.tools.named("message")), None)
        path = self.tools.path(tool, "text") if tool else None
        text = self.question_text(n, keys)
        args, problems = fit_object(self.tools.schema[tool], {path: text}) if path else ({}, ["no message tool"])
        if problems:
            return None
        obs = self.send(tool, args, n)
        self.request.asked.update((n, k) for k in keys)
        heard = self.hear(render(obs), text, n)
        kinds = {h["kind"] for h in heard}
        for k in keys:
            named = any(h["kind"] == "disclose" and names_key(h.get("key", ""), k) for h in heard)
            self.request.answers[(n, k)] = "disclosed" if named else "none" if "no_need" in kinds and len(keys) == 1 else "unclear"
        if len(keys) > 1 and any(self.request.answers.get((n, k)) == "disclosed" for k in keys[1:]):
            # an answer about a property past the first: this shopper reads the whole message
            self.lists_proven = True
        return kinds

    def spoken(self, key: str) -> str:
        """A property as a question names it: a one- or two-letter token that no listing seen uses as a word is a leftover
        of the store's key (a one-letter prefix, a two-letter tag) and stays out of the words."""
        words = key.split()
        if len(words) < 2:
            return key
        known = self.store_vocabulary()
        # one letter is never a word of a property's name; two letters are when the listings use them as a word ("tv")
        kept = [w for w in words if len(w) > 2 or not w.isalpha() or (len(w) == 2 and w.lower() in known)]
        return " ".join(kept) or key

    def store_vocabulary(self) -> set:
        """Lower-case words of every listing record seen, read again only when records came in."""
        size = len(self.catalog.records)
        if self.vocabulary[0] != size:
            words: set = set()

            def gather(node: Any) -> None:
                # the listings' own words: titles, descriptions and attribute values, never the attribute names
                if isinstance(node, str):
                    words.update(re.findall(r"[a-z0-9]+", node.lower()))
                elif isinstance(node, dict):
                    for value in node.values():
                        gather(value)
                elif isinstance(node, list):
                    for value in node:
                        gather(value)

            for views in self.catalog.records.values():
                gather(views)
            self.vocabulary = (size, words)
        return self.vocabulary[1]

    def questions(self) -> int:
        kept = RESERVE_MESSAGE if self.budget.report else 0
        return (self.budget.messages if self.budget.messages is not None else QUESTIONS_UNSTATED + kept) - self.budget.sent - kept

    def plan_questions(self, menus: dict) -> list:
        """Each line's properties ranked (the clues' attributes the planner names first, then what it picks as usual, then
        the hints and the menu), cut into windows; each line's first window before any line's second."""
        count, lines = self.questions(), self.request.lines
        text = "\n".join(f"line {n} ({lines[n]['kind']}), {len(self.cards.get(n, []))} listings opened:\n"
                         + ("\n".join(f"- {m['key']} | on {m['cards']} | values: {'; '.join(m['examples'])} | differ: {m['spread']}" for m in menu)
                            or "- no attribute seen") for n, menu in menus.items())
        allowed = {n: {m["key"] for m in menu} | {norm(h["key"]) for h in lines[n]["hints"]} for n, menu in menus.items()}
        ranked: dict = {n: [] for n in menus}

        def take(n: int | None, keys: list) -> None:
            if n not in ranked:
                return
            for k in keys:
                if norm(k) not in ("", "none") and k in allowed[n] and (n, k) not in self.request.asked and not self.said_already(n, k) and k not in ranked[n]:
                    ranked[n].append(k)

        rows = self.seats.ask(self.request.full, "\n".join(self.request.lines_text(self.catalog)), text, max(1, count) * PLAN_DEPTH)
        for clue in (True, False):
            for _, head, f in rows:
                if (norm(f.get("why")) == "clue") == clue:
                    take(number_of(head), [norm(k) for k in (f.get("keys") or f.get("key") or "").split(",") if not NOT_A_PROPERTY.search(norm(k))])
        for n, menu in menus.items():
            # after the planner's picks, the line's hints and its menu: what most listings carry first, never a field that
            # holds the store's own bookkeeping (variants, codes, warranty, condition, dates) rather than a property
            usual = sorted((m for m in menu if not NOT_A_PROPERTY.search(m["key"])), key=lambda m: -int(m.get("cards") or 0))
            take(n, [norm(h["key"]) for h in lines[n]["hints"]] + [m["key"] for m in usual])
        width = max(1, self.window)
        cuts = {n: [tuple(keys[i:i + width]) for i in range(0, len(keys), width)] for n, keys in ranked.items()}
        plan = []
        for depth in range(max((len(c) for c in cuts.values()), default=0)):
            plan += [(n, cut[depth]) for n, cut in cuts.items() if depth < len(cut)]
        return plan[:max(0, count) * 2]


class Judge:

    def fit_lines(self, lines: list) -> None:
        jobs = {}
        for n in lines:
            product, needs = self.request.product_text(n), self.request.judged(n)
            cards = {c: self.catalog.fit_card(c, needs) for c in self.cards.get(n, []) if c in self.catalog.records}
            jobs[n] = {c: dump([product, text]) for c, text in cards.items() if self.verdicts.get((n, c), (None,) * 4)[3] != dump([product, text])}
        asked = [n for n in lines if jobs[n] and self.clock.left() > SAFE_LEFT]
        answers = together(*(lambda n=n: self.seats.fit(self.request.product_text(n), "\n\n".join(json.loads(sig)[1] for sig in jobs[n].values()))
                             for n in asked))
        for n, rows in zip(asked, answers):
            said = {ident: f for _, head, f in rows or [] if (ident := seen_id(head, list(jobs[n]))) in jobs[n] and f.get("marks")}
            count = len(self.request.judged(n))
            for ident, sig in jobs[n].items():
                f = said.get(ident, {})
                entries = self.catalog.attributes(ident) + leaves(self.catalog.rows.get(ident, {}))
                self.verdicts[(n, ident)] = (*verdict_of(spared(f["marks"], self.request.judged(n), entries), count), f.get("quote", ""), sig) if f else ("unjudged", None, "", None)

    def acquire(self, n: int) -> None:
        if (batch := self.unread(n, PURSUE_READS)) and self.room():
            self.read_into(n, batch)
            return
        reach = {c["label"]: self.catalog.reach(c) for c in self.tools.channels()}
        lst = self.catalog.to_page(n, self.request.value_sets(n), self.request.lines[n]["kind"], reach, strict=False)
        words = next((c for c in self.tools.channels() if c["way"] == "words"), None)
        moved = bool(lst and self.room() and self.follow(lst)) or any(words and self.room() and self.search(words, n, extra) for extra in self.request.widen(n))
        if moved and (batch := self.unread(n, PURSUE_READS)) and self.room():
            self.read_into(n, batch)

    def shelf_rules_out(self, ident: str, n: int) -> bool:
        """Whether a listing sits on a shelf of the line's product that the request's names for the line (its kind, other
        names, and the shelf name when the request says it) fit less closely than another shelf seen, when that other
        shelf fits them closely: "Desk and Wall Lamp Shades" for a line of desk and wall lamp shades rules out "Floor and
        Wall Lamp Shades" and "Floor Lamp Shades"."""
        key = self.catalog.keys.get("shelf", "")
        own = (self.catalog.rows.get(ident) or {}).get(key) if key else None
        line = self.request.lines.get(n)
        if not isinstance(own, str) or not line:
            return False
        fit = self.shelf_fit(n, key, line)
        if own not in fit or len(fit) < 2:
            return False
        best = max(fit.values())
        return best >= SHELF_CLOSE and fit[own] < best

    def shelf_fit(self, n: int, key: str, line: dict) -> dict:
        """How closely each shelf seen that holds the line's product words fits the request's names for the line (shared
        words over all words), kept until the catalog or the request changes."""
        at = (n, key, len(self.catalog.rows), self.request.version)
        if self.shelf_memo.get("at") == at:
            return self.shelf_memo["fit"]
        names = [line["kind"], *line["also"]]
        if words_of(line["shelf"]) and f" {' '.join(words_of(line['shelf']))} " in f" {' '.join(words_of(self.request.text))} ":
            names.append(line["shelf"])
        said = {stem(w) for name in names for w in words_of(name) if w != "and"}
        fit: dict = {}
        for row in self.catalog.rows.values():
            shelf = row.get(key)
            if said and isinstance(shelf, str) and shelf not in fit and names_kind(shelf, line["kind"]):
                have = {stem(w) for w in words_of(shelf) if w != "and"}
                fit[shelf] = len(have & said) / len(have | said)
        self.shelf_memo = {"at": at, "fit": fit}
        return fit

    def allowed(self, ident: str, n: int) -> bool:
        price, cap = self.catalog.price(ident), self.request.cap(n)
        if self.shelf_rules_out(ident, n):
            return False
        return ident not in self.excluded and self.catalog.in_stock(ident) is not False and not (cap is not None and price is not None and price > cap)

    def fits(self, ident: str, n: int) -> int | None:
        rank = {"fits": 0, "unsure": 1, "unjudged": 1}.get(self.verdicts.get((n, ident), ("",))[0])
        return None if rank is None or not self.allowed(ident, n) else max(rank, int(self.request.unresolved(n)))

    def score(self, ident: str, n: int) -> tuple:
        goal, key = self.request.goals.get(n) or self.request.goals.get("all") or ("none", "none")
        value = to_number(self.catalog.facts_of(ident).get(norm(key))) if goal in ("highest", "lowest") else None
        rank = 0 if goal not in ("highest", "lowest") else float("inf") if value is None else -value if goal == "highest" else value
        price = self.catalog.price(ident)
        return self.fits(ident, n), rank, (price if price is not None else float("inf")) * self.request.lines[n]["qty"]

    def pick(self, n: int) -> str | None:
        return min((c for c in self.cards.get(n, []) if self.fits(c, n) is not None), key=lambda c: (*self.score(c, n), c), default=None)

    def combination(self) -> dict:
        lines = sorted(self.request.lines)
        options = {n: sorted((self.score(c, n), c) for c in self.cards.get(n, []) if self.fits(c, n) is not None)[:COMBINE_MAX] for n in lines}
        filled = [n for n in lines if options[n]]
        cap = (self.request.caps.get("total") or {}).get("amount") if len(lines) > 1 else None
        best = combine([options[n] for n in filled], cap, lambda a, x, b, y: self.pair(filled[a], x, filled[b], y)) or [None] * len(filled)
        return {n: None for n in lines} | dict(zip(filled, best))

    def pair(self, n: int, first: str, m: int, second: str) -> bool | None:
        if first == second:
            return False
        if self.request.depends.get(n) != m and self.request.depends.get(m) != n:
            return True
        coded = self.code_pair(first, second)
        if coded is not None:
            return coded
        ours, theirs = self.compat.get(first), self.compat.get(second)
        return bool(ours & theirs) if ours and theirs else None


class Chains:

    def chains(self, lines: list | None = None, allowance: int | None = None) -> None:
        evidence = (self.catalog.changes, self.request.version, self.catalog.epoch)
        for ref in [r for r in self.request.refs if r.get("value") is None and r["line"] in self.request.lines and (lines is None or r["line"] in lines)
                    and r.get("stalls", 0) < CHASE_STALLS and ((r.get("state") or {}).get("paused") or r.get("seen") != evidence)]:
            state = ref.setdefault("state", {})
            state.pop("paused", None)
            before = (len(state.get("path", [])), len(state.get("ruled", {})), len(state.get("lists", [])), self.catalog.changes)
            ref["tries"] = ref.get("tries", 0) + 1
            self.pass_end = None if allowance is None else self.env.step + allowance
            found = self.chase({k: ref.get(k, "") for k in ("key", "relation", "store", "kind", "fact")}, state=state)
            self.pass_end = None
            ref["seen"] = (self.catalog.changes, self.request.version, self.catalog.epoch)
            ref["stalls"] = ref.get("stalls", 0) + (not found and before == (len(state.get("path", [])), len(state.get("ruled", {})), len(state.get("lists", [])), self.catalog.changes))
            ref["value"] = found[2] if found else None
            if found:
                condition, applied, value, source = found
                ref.update(condition=condition, applied=applied, source=source)
                self.request.lines[ref["line"]]["needs"].append({"key": ref["key"], "relation": applied, "value": condition,
                                                                 "quote": f"listing {source} states {value}; relation asked: {norm(ref.get('relation')) or 'equal'}"})
                self.request.version += 1
                self.grown.add(ref["line"])
        self.record(refs=self.request.refs)

    def deferrals(self, lines: list | None = None) -> None:
        for n in sorted(self.request.lines) if lines is None else lines:
            needs = self.request.judged(n)
            for ident in [c for c in self.cards.get(n, []) if self.verdicts.get((n, c), ("",))[0] == "unsure"][:DEFER_TOP]:
                _, i, quote = self.verdicts[(n, ident)][:3]
                tried = (ident, needs[i]["key"] if i and 0 < i < len(needs) else "", self.catalog.changed_at.get(ident, 0))
                if tried[1] and norm(quote) not in ("", "none") and tried[1] not in self.catalog.resolved.get(ident, {}) and tried not in self.deferred:
                    self.deferred.add(tried)
                    found = self.chase({"key": needs[i]["key"], "relation": "", "store": "", "kind": "", "fact": ""}, ident)
                    if found and found[3] != ident:
                        self.catalog.resolved.setdefault(ident, {})[needs[i]["key"]] = (found[2], found[3])
            self.fit_lines([n])

    def chase(self, desc: dict, start: str | None = None, state: dict | None = None) -> tuple | None:
        state = {} if state is None else state
        chain, ruled, why = state.setdefault("path", []), state.setdefault("ruled", {}), state.setdefault("why", {})
        for i in [i for i, v in ruled.items() if self.catalog.changed_at.get(i, 0) != v]:
            ruled.pop(i)
            why.pop(i, None)
        desc, problem = state.get("desc") or desc, ""
        while self.room():
            cards = [start] if start and not chain else self.find(desc, state)
            if not cards:
                break
            self.read_lines([c for c in cards if c not in self.catalog.records])
            where = f"listing {start} itself" if start and not chain else f"store {desc['store']}, kind {desc['kind']}, fact {desc['fact']}"
            need = f"attribute: {desc['key']}\nrelation asked: {norm(desc.get('relation')) or 'equal'}\nlisting: {where}"
            answer = self.seats.link(need, " -> ".join(chain), "\n\n".join(map(self.catalog.card_text, cards)), problem)
            target, value, store = seen_id(answer.get("target", ""), cards), answer.get("value", ""), answer.get("store", "")
            condition, applied = answer.get("condition", ""), norm(answer.get("as", ""))
            text = self.catalog.card_text(target) if target in cards else ""
            if target not in cards:
                ruled.update({c: self.catalog.changed_at.get(c, 0) for c in cards if c != start})
                why.update({c: "LINK found it is not the described listing" for c in cards if c != start})
                problem = ""
                continue
            if target in chain:
                state.update(stop="the chain came back to a listing met before", at=self.limit_now())
                return None
            if norm(value) in ("", "none") and norm(store) in ("", "none") and not problem:
                self.complete(target, None, False, self.room)
                problem = "every entry of the listing is shown now: answer with the value it gives, or with the listing its note passes the attribute on to"
                continue
            if target not in self.catalog.records:
                state.update(stop=f"listing {target} could not be read", at=self.limit_now())
                return None
            derived = norm(condition) not in ("", "none") and applied in ("equal", "contains")
            if (norm(value) not in ("", "none") and (not written(text, value) or (derived and not written(text, condition)))) \
                    or (norm(value) in ("", "none") and not holds(text, words_of(store))):
                if problem.startswith("your answer"):
                    state.update(stop=f"the value read is not written on listing {target}", at=self.limit_now())
                    return None
                problem = f"your answer ({dump(answer)}) does not match the listings: the value and the condition must be written on the target as it writes them, and the next store must be named in its text"
                continue
            chain.append(target)
            if norm(value) not in ("", "none"):
                self.complete(target, None, True, self.room)
                if target not in self.catalog.records:
                    state.update(stop=f"the source listing {target} could not be read", at=self.limit_now())
                    return None
                state.pop("stop", None)
                return (condition if derived else value), (applied if applied in VALUE_RELATIONS + ("not",) else norm(desc.get("relation")) or "equal"), value, target
            desc, problem = {"key": desc["key"], "relation": desc.get("relation"), "store": store, "kind": answer.get("of", ""), "fact": answer.get("fact", "")}, ""
            state["desc"] = desc
        paused = self.pass_end is not None and self.env.step >= self.pass_end
        state.update(stop="paused: this pass's steps are used" if paused else state.get("need") if self.room() else "no room", paused=paused, at=self.limit_now())
        return None

    def find(self, desc: dict, state: dict) -> list:
        ruled, lists, tried = state.setdefault("ruled", {}), state.setdefault("lists", []), state.setdefault("tried", [])
        channels = self.tools.channels()
        words, shelves = next((c for c in channels if c["way"] == "words"), None), [c for c in channels if c["way"] == "filter"]
        plans = [(words, "query", f"{desc['store']} {desc['fact']}"), (words, "query", desc["store"]), (words, "query", f"{desc['fact']} {desc['kind']}")]
        plans += [(c, "shelf", self.tools.option(c["tool"], c["shelf"], self.catalog.shelf_of(desc["kind"]) or desc["kind"], desc["kind"])) for c in shelves]
        while self.room():
            if found := self.catalog.of_store(desc["store"], desc["kind"], desc["fact"], ruled):
                state.pop("need", None)
                return found
            if lst := next((x for k in lists if (x := self.catalog.lists.get(k)) and not x["done"] and x.get("next") not in (None, "") and x["pages"] < REF_PAGES), None):
                lst["done"] = lst["done"] or not self.follow(lst)
                continue
            plan = next((p for p in plans if p[0] and p[2] and dump([p[0]["label"], p[1], p[2]]) not in tried), None)
            if plan is None:
                state["need"] = "every search path for the described listing is read"
                return []
            tried.append(dump([plan[0]["label"], plan[1], plan[2]]))
            channel, args = plan[0], {}
            for path, value in channel["mode"].items():
                place(args, path, value)
            place(args, channel[plan[1]], " ".join(query_words(plan[2])) if plan[1] == "query" else plan[2])
            self.page(channel["tool"], args, None)
            lists.append(self.list_key(channel, args))
        state["need"] = f"steps ({self.limit_now()})"
        return []

    def fit_sets(self) -> None:
        lines = sorted({n for pair in self.request.depends.items() for n in pair})
        cards = [c for n in lines for c in ([c for c in self.cards.get(n, []) if self.fits(c, n) is not None] or self.cards.get(n, []))[:DEFER_TOP * 2]
                 if c in self.catalog.records and c not in self.compat]
        if cards and self.clock.left() > SAFE_LEFT:
            self.compat.update({seen_id(h, cards): ids for h, ids in self.seats.part_ids("\n\n".join(map(self.catalog.card_text, cards))).items()})
            self.compat.update({c: self.compat.get(c, set()) for c in cards})


class Closer:

    def kept_for_finish(self) -> int:
        """Steps a cart call leaves for the finish: the order, and a report message only when the request asks for one."""
        built = self.tools.finish_call("order", ORDER_TEXT) and self.tools.finish_call("decline", DECLINE_TEXT)
        spare = 0 if built else 1      # a finish the code cannot build goes through the assistant seat
        return RESERVE_ORDER + (RESERVE_MESSAGE if self.budget.report else 0) + spare

    def closing_room(self) -> bool:
        """Room for a read at the close: steps beyond what the cart and the finish need (each line put in, or swapped when
        the cart holds lines already: a discard and an include), and time beyond the safe margin (one store call may take
        the world's whole timeout); not ended by the spend close."""
        return not self.env.finished and self.budget.room() > self.budget.reserve + self.cart_steps(len(self.request.lines)) + 2 \
            and self.clock.left() > SAFE_LEFT

    def cart_steps(self, lines: int) -> int:
        """The cart calls that putting this many lines in takes: one each, two each when the cart holds lines to swap."""
        return (2 if self.cart.lines else 1) * lines

    def must_close(self) -> bool:
        return self.env.finished or self.clock.left() < SAFE_LEFT or self.llm.stopped or self.free_steps() <= 0 or self.llm.spend >= SPEND_CLOSE

    def free_steps(self) -> int:
        needed, calls = self.budget.reserve + (1 + PICK_READS) * len(self.request.lines) + 1, max(0, CALLS_PLAN[self.request.route] - self.llm.calls)
        return min(self.budget.allowance(calls) - self.env.step, self.budget.room()) - needed

    def close(self, attempt: int = 0) -> list | None:
        if self.env.finished or self.budget.room() <= 0:
            return None
        self.news()
        kind = "order" if attempt >= 2 and self.cart.lines else self.decide()
        if kind == "order" and attempt < 2:
            self.phase = "P4"
            self.assemble()
            if self.news():
                self.assemble()
            self.phase = "P5"
            self.oblige()
            if self.recheck():
                # what the reads show is acted on whether or not a notice came with them, and a cart that changed is
                # reported again before the order
                self.news()
                self.assemble()
                self.oblige()
        self.phase = "P6"
        self.record(finishing=kind if self.cart.lines else "decline", verified=kind == "order" and self.whole, reported=self.reported == self.cart.version,
                    limit=self.limit_now())
        return self.finish(kind if self.cart.lines else "decline")

    def decide(self) -> str:
        failing = self.failing()
        if not failing or self.audited or self.clock.left() < SAFE_LEFT:
            return "decline" if failing else "order"
        self.audited, lines = True, sorted(self.request.lines)
        evidence = "\n\n".join(self.evidence(n) for n in failing)
        answer = self.seats.audit(self.request.full, self.request.needs_text(), evidence)
        unopened = {i.strip() for i in answer.get("unopened", "").split(",")}
        fresh = {n: [i for i in self.catalog.shown(n) if i in unopened and i not in self.cards.get(n, [])] for n in failing}
        self.record(audit=answer)
        if answer.get("verdict") != norm("order_instead"):
            return "decline"
        self.phase = "P3"
        for n, ids in fresh.items():
            self.read_into(n, ids, self.closing_room)
        self.fit_lines(lines)
        still = self.failing()
        return "decline" if still and (not any(fresh.values()) or self.unbacked()) else "order"

    def evidence(self, n: int) -> str:
        coverage, cap = self.catalog.coverage(n), self.request.cap(n)
        opened = [f"{c} {self.catalog.title(c)} at {amount(self.catalog.price(c))}: {v[0]}" + (f", need {v[1]}" if v[1] is not None else "") + (f" ({v[2]})" if norm(v[2]) not in ("", "none") else "")
                  for c in self.cards.get(n, []) if (v := self.verdicts.get((n, c)))]
        unopened = [f"{i} {self.catalog.title(i)} at {amount(self.catalog.price(i))}" for i in self.catalog.ranked(n, self.request.value_sets(n), cap, self.request.lines[n]["kind"])
                    if i not in self.cards.get(n, [])][:UNOPENED_SHOWN]
        return "\n".join([self.request.product_text(n), f"searches: {coverage['lists']}, read to their end: {coverage['complete']}, rows seen: {coverage['rows']}",
                          "opened:", *(opened or ["none"]), "never opened:", *(unopened or ["none"])])

    def failing(self) -> list:
        combo = self.combination()
        judged = {n: [(c, self.verdicts[(n, c)][0]) for c in self.cards.get(n, []) if (n, c) in self.verdicts] for n in self.request.lines}
        failed = [n for n, said in judged.items() if said and all(v == "fails" or c in self.excluded for c, v in said)]
        over = [n for n in self.request.lines if combo[n] is None and any(self.fits(c, n) is not None for c in self.cards.get(n, []))]
        empty = [n for n in sorted(self.request.lines) if not self.catalog.shown(n)]
        return list(dict.fromkeys([n for n in dict.fromkeys(failed + over + self.unbacked()) if self.catalog.coverage(n)["grade"] != "none"] + empty))

    def unbacked(self) -> list:
        """Lines whose shopper disclosed a need and none of whose judged listings is verified to fit."""
        out = []
        for n in self.request.lines:
            said = [(c, self.verdicts[(n, c)][0]) for c in self.cards.get(n, []) if (n, c) in self.verdicts]
            if not any(x.get("said") for x in self.request.lines[n]["needs"]):
                continue
            if said and not any(v == "fits" and c not in self.excluded for c, v in said):
                out.append(n)
        if out:
            self.record(unbacked=out)
        return out

    def unmet(self, n: int, ident: str) -> list:
        verdict = self.verdicts.get((n, ident), ("unjudged",))[0]
        return ([] if ident in self.catalog.records else ["not read"]) + ([] if verdict == "fits" else [f"FIT: {verdict}"]) \
            + (["a reference unresolved"] if self.request.unresolved(n) else []) + (["not read since an event"] if ident in self.suspect else [])

    def confirm(self) -> None:
        tool = next((t for t in self.readers() if self.catalog.keys.get("stock") in self.catalog.read_keys.get(t, set())), self.table_tool())
        for ident in sorted(self.suspect - self.excluded):
            if tool and self.catalog.read_at.get(ident, -1) < self.catalog.epoch and self.budget.room() > self.kept_for_finish():
                self.read(tool, [ident], None)
            if self.catalog.in_stock(ident) is False:
                self.excluded.add(ident)
        self.suspect = set()

    def likeliest(self, n: int, cap: float | None, skip: set) -> str | None:
        """The unread row to buy when no read listing is left: the one whose own words name the most of the line's needed
        values, then the cheapest."""
        values = self.request.value_sets(n)
        pool = [i for i in self.catalog.shown(n) if (p := self.catalog.price(i)) is not None and self.catalog.in_stock(i) is not False
                and (cap is None or p <= cap) and i not in skip]

        def named(ident: str) -> int:
            words = self.catalog.row_words(ident)
            return sum(1 for v in values if v and set(v) <= words)

        return min(pool, key=lambda i: (-named(i), self.catalog.price(i), i), default=None)

    def picks(self) -> dict:
        combo, out = self.combination(), {}
        for n in sorted(self.request.lines):
            cap, failed = self.request.cap(n), {c for c in self.cards.get(n, []) if self.verdicts.get((n, c), ("",))[0] == "fails"} | self.excluded
            failed |= {c for c in self.catalog.shown(n) if self.shelf_rules_out(c, n)}
            ident = combo[n] or self.catalog.cheapest(n, cap, True, failed) or self.likeliest(n, cap, failed)
            if ident:
                out[n] = (ident, "unread" if ident not in self.catalog.records else self.verdicts.get((n, ident), ("unjudged",))[0])
        return out

    def wanted(self) -> dict:
        target = {}
        for n, (ident, _) in self.picks().items():
            target[ident] = target.get(ident, 0) + self.quantity(n, ident)
        return target

    def quantity(self, n: int, ident: str) -> int:
        """The line's quantity; one when it is more than one, no product or need sentence states it, and the request's only
        cap, on its only line, would not pay for that many of the pick."""
        qty, cap, price = self.request.lines[n]["qty"], self.request.cap(n), self.catalog.price(ident)
        if qty > 1 and len(self.request.lines) == 1 and len(self.request.caps) == 1 and not self.request.sourced(n) \
                and cap is not None and price is not None and qty * price > cap:
            self.record(quantity_kept_to_one={n: qty, "cap": cap, "price": price})
            return 1
        return qty

    def assemble(self) -> None:
        if self.stale:
            self.stale = False
            self.view()
        for _ in range(3):
            self.confirm()
            self.fit_lines(sorted(self.request.lines))
            picks = self.picks()
            self.target = self.wanted()
            for n, (ident, _) in picks.items():
                self.cards.setdefault(n, []).extend([] if ident in self.cards.get(n, []) else [ident])
                self.complete(ident, n, True, lambda: self.budget.room() > self.budget.reserve + self.cart_steps(len(self.target)) + 2)
            self.fit_lines(sorted(self.request.lines))
            if self.wanted() == self.target:
                break
        picks, combo = self.picks(), self.combination()
        open_ = {n: self.unmet(n, ident) for n, (ident, _) in picks.items()}
        self.whole = len(picks) == len(self.request.lines) and all(combo[n] == ident and not open_[n] for n, (ident, _) in picks.items())
        self.record(buy={n: [ident, standing, sorted(self.catalog.records.get(ident, {})), open_[n]] for n, (ident, standing) in picks.items()},
                    complete=self.whole, questions_open=sorted(f"{n}:{k}" for (n, k), v in self.request.answers.items() if v == "unclear") + [f"{n}:{'/'.join(k)}" for n, k in self.pending or []])
        for _ in range(2):
            current = self.cart.counts()
            if current == self.target or not self.write(current):
                return
            self.view()
            self.cart.adding = self.cart.adding or any(q > self.target.get(i, 0) for i, q in self.cart.counts().items())

    def write(self, current: dict) -> bool:
        tool = next(iter(self.tools.named("cart_write")), None)
        calls, sent = self.tools.cart_calls(tool, self.target, current, self.cart.adding) if tool else [], []
        whole = self.budget.room() > self.kept_for_finish() + len(calls) - 1
        for args in calls if whole else []:
            if self.budget.room() > self.kept_for_finish():
                self.send(tool, args)
                sent.append(dump([tool, args]))
        if sent and not self.failed.intersection(sent) and self.cart.viewed != self.cart.version:
            self.cart.assume(self.target, self.catalog)
        if not calls and self.budget.room() > self.kept_for_finish():
            items = ", ".join(f"{ident} x{qty}" for ident, qty in self.target.items())
            return self.free_turn(GOAL_CART.format(items=items), allow=("cart_write",))
        return bool(sent)

    def view(self) -> bool:
        tool = next(iter(self.tools.named("cart_view")), None)
        if not tool or self.budget.room() <= self.kept_for_finish():
            return False
        args, problems = fit_object(self.tools.schema[tool], {})
        if not problems:
            self.send(tool, args)
        return not problems

    def recheck(self) -> bool:
        """Before the order, each cart listing is read with every tool that shows one listing and has not read it (a
        store may split a listing over a facts page and an offer page), and, when the request warns that things change
        while the errand runs, with each of them once more after the last cart change. True when a page was read."""
        if self.rechecked == self.cart.version or not self.cart.lines:
            return False
        self.rechecked = self.cart.version
        again = bool(RECHECK_RULE.search(self.request.text))
        tools = [t for t in self.tools.named("read") if self.tools.path(t, "ids", "id")]
        done = False
        for ident in sorted(self.cart.counts()):
            for tool in tools:
                seen = self.read_by.get((tool, ident))
                if seen is not None and not (again and seen < self.cart.version):
                    continue
                if (self.budget.room() > self.kept_for_finish() + 1 and self.clock.left() > SAFE_LEFT
                        and self.read(tool, [ident], None) is not None):
                    done = True
        return done

    def oblige(self) -> None:
        while self.report():
            self.phase = "P4"
            self.assemble()
            self.phase = "P5"

    def report(self) -> bool:
        tool = next(iter(self.tools.named("message")), None)
        path = self.tools.path(tool, "text") if tool else None
        if not (path and self.cart.lines and self.budget.messages_left() != 0 and self.budget.room() > RESERVE_ORDER) or self.reported == self.cart.version or self.llm.key_spent:
            return False
        if not self.whole and not any(d.get("by") != "tool" for d in self.request.duties):
            return False
        text = self.cart.report(self.request.money)
        args, problems = fit_object(self.tools.schema[tool], {path: text})
        if problems:
            return False
        obs, version, self.reported = self.send(tool, args), self.request.version, self.cart.version
        for duty in self.request.duties:
            duty["done_at"] = self.cart.version
        self.hear(render(obs), text)
        return self.request.version != version

    def finish(self, kind: str) -> list | None:
        call = self.tools.finish_call(kind, ORDER_TEXT if kind == "order" else DECLINE_TEXT)
        tool = next(iter(self.tools.named("message")), None)
        path = self.tools.path(tool, "text") if tool else None
        if kind == "decline" and self.request.decline_rule and path and self.budget.messages_left() != 0 and self.budget.room() > RESERVE_ORDER \
                and not (call and self.tools.path(call[0], "text")) and not self.done_now(tool, {path: DECLINE_TEXT}):
            self.send(tool, fit_object(self.tools.schema[tool], {path: DECLINE_TEXT})[0])
        if call and dump(list(call)) not in self.failed:
            self.send(*call)
        elif not self.free_turn(GOAL_FINISH.format(how=HOW_ORDER if kind == "order" else HOW_DECLINE), allow=("order", "decline", "finish")):
            return None
        if self.env.finished:
            return None
        refusal = self.last_obs.get("refusal") if isinstance(self.last_obs, dict) else None
        return [clean(x) for x in refusal] if isinstance(refusal, list) else [render(self.last_obs)]


# ---------- model codes, read by code
# a code as stores print one: letters, an optional dash or space, three to five digits, and x's standing for a digit each in
# a family ("ABC-1234", "XY1234", "ABC-12xx")
CODE = r"[A-Z]{2,5}[- ]?(?:\d{3,5}|\d{1,4}[xX]{1,3})"
CODE_RE = re.compile(r"(?<![\w-])(" + CODE + r")(?![\w-])")
OWN_CODE_RE = re.compile(r"\b(?i:model)(?:\s+(?i:ref(?:erence)?|no|number|code))?\.?\s*[:#]?\s*(" + CODE + r")(?![\w-])")
PART_OWN_RE = re.compile(r"[(\[]\s*(?i:part)\s+(" + CODE + r")\s*[)\]]")
TAKES_RE = re.compile(r"(?<![Nn]ot )(?<!n't )(?<![Nn]ever )\b(?i:compatible with|goes with|pairs with|works with|fits|for use with|made for|suits)\b\s*:?\s*([^.;]+)")
REFUSES_RE = re.compile(r"\b(?i:not compatible with|incompatible with|does not fit|doesn't fit|won't fit|will not fit|does not work with|doesn't work with|not for use with|not made for)\b\s*:?\s*([^.;]+)")
SPARES_RE = re.compile(r"\b(?i:except(?:\s+for)?|excluding|but not|other than|apart from)\b")
NOTE_RE = re.compile(r"\b(?i:part)\s+(" + CODE + r")\s+(?i:also)\s+(?i:fits|goes with|works with)\s*:?\s*([^.;]+)")


def code_key(code: str) -> str:
    """One spelling of a code: lower case, letters and digits only; a family keeps its x's ("abc12xx")."""
    return re.sub(r"[^0-9a-z]", "", code.lower())


def codes_in(text: str) -> list:
    return [code_key(c) for c in CODE_RE.findall(text or "")]


def covers(taken: str, code: str) -> bool:
    """Whether a code or family a listing names takes in a device's code: the same code, or a family whose letters and
    digits lead the code with one digit for each x."""
    if taken == code:
        return True
    stem = taken.rstrip("x")
    return stem != taken and len(taken) == len(code) and code.startswith(stem) and code[len(stem):].isdigit()


def code_marks(text: str) -> dict:
    """What a listing's own words say about model codes: the codes it is ('own'), its part code ('part'), what it takes in
    ('takes', codes and families) and leaves out ('spares'), and the notes it carries about other parts ('notes')."""
    own = [code_key(c) for c in OWN_CODE_RE.findall(text)]
    part = [code_key(c) for c in PART_OWN_RE.findall(text)]
    takes, spares, notes = [], [], {}
    for hit in NOTE_RE.finditer(text):
        notes.setdefault(code_key(hit.group(1)), []).extend(codes_in(hit.group(2)))
    for hit in REFUSES_RE.finditer(text):
        spares += codes_in(hit.group(1))
    for hit in TAKES_RE.finditer(text):
        # a "part P also fits" note speaks of another listing, and "Model: X" of this one: neither is what this one takes in
        before = text[max(0, hit.start() - 30):hit.start()]
        if re.search(r"(?i:part)\s+" + CODE + r"\s+(?i:also)\s*$", before):
            continue
        clause = hit.group(1)
        cut = SPARES_RE.search(clause)
        kept, left = (clause[:cut.start()], clause[cut.end():]) if cut else (clause, "")
        takes += codes_in(kept)
        spares += codes_in(left)
    return {"own": own, "part": part, "takes": takes, "spares": spares, "notes": notes}


class CodeFits:

    def code_index(self) -> dict:
        """Each read listing's code marks, with the notes other listings carry about it added to what it takes in; read
        again only when records came in."""
        size = len(self.catalog.records)
        if self.code_cache[0] == size:
            return self.code_cache[1]
        marks = {ident: code_marks(self.catalog.card_text(ident)) for ident in self.catalog.records}
        notes: dict = {}
        for mark in marks.values():
            for part, codes in mark["notes"].items():
                notes.setdefault(part, []).extend(codes)
        for mark in marks.values():
            for part in mark["part"]:
                mark["takes"] = mark["takes"] + notes.get(part, [])
        self.code_cache = (size, marks)
        return marks

    def takes_in(self, part: dict, codes: list) -> bool | None:
        """Whether a part's marks take in one of a device's codes: False when they leave it out or name only others, None
        when they name no code at all."""
        if not part["takes"]:
            return None
        if any(any(covers(s, c) for s in part["spares"]) for c in codes):
            return False
        return any(any(covers(t, c) for t in part["takes"]) for c in codes)

    def code_pair(self, first: str, second: str) -> bool | None:
        """The fit of two listings by their codes: one is a device that names its own code, the other a part that names
        what it takes in. None when the codes cannot tell (left to the model's reading)."""
        index = self.code_index()
        a, b = index.get(first), index.get(second)
        if not a or not b:
            return None
        for device, part in ((a, b), (b, a)):
            if device["own"] and not device["takes"] and part["takes"]:
                verdict = self.takes_in(part, device["own"])
                if verdict is not None:
                    if (first, second) not in self.code_said:
                        self.code_said.add((first, second))
                        self.record(code_pair=[first, second], fits=verdict)
                    return verdict
        return None


class ShoppingAssistant(Listener, Searcher, Reader, Asker, Judge, Chains, Closer, CodeFits):

    def __init__(self, problem_data: dict) -> None:
        environment = problem_data["environment"]
        view, binding = environment.get("policy_view") or {}, environment.get("binding") or {}
        self.clock, self.wire, self.trace = Clock(), Wire(), []
        self.llm = Llm(self.wire, self.clock, f"jb-{binding.get('session_id', '')}", self.trace)
        self.seats = Seats(self.llm)
        self.code_cache: tuple = (-1, {})
        self.code_said: set = set()
        self.env = Env(self.wire, self.clock, binding, view.get("finished"))
        self.request, self.tools = Request(view.get("query") or "", as_records(view.get("opening"))), Tools(view.get("tools") or [])
        self.catalog, self.cart = Catalog(), Cart()
        self.budget = Budget(int(view.get("max_steps") or 0), self.clock, self.llm, self.env)
        self.phase, self.last, self.blocked, self.made, self.failed, self.target = "P0", "", [], {}, set(), {}
        self.cards, self.used, self.grown, self.verdicts, self.excluded = {}, Counter(), set(), {}, set()
        self.last_obs, self.reported, self.compat = None, -1, {}
        self.notices, self.noted, self.stale, self.audited, self.misses = [], set(), False, False, Counter()
        self.pairs, self.suspect, self.pending, self.acted, self.stuck, self.pass_end, self.retried = False, set(), [], Counter(), set(), None, set()
        self.whole, self.seen_ledgers, self.spent, self.tried, self.deep, self.judged_at, self.menus, self.replanned = False, None, Counter(), {}, set(), {}, {}, False
        self.deep_state: dict = {}
        self.deferred: set = set()
        self.rechecked: int | None = None
        self.shelf_memo: dict = {}
        self.read_by: dict = {}
        self.refusal = ""                  # why the system's last call did not fit its tool (sent to the assistant seat)
        self.mode_tried: set = set()       # channels whose fallback mode value already gave way once
        self.pending: list | None = None
        self.window, self.turned, self.lists_proven, self.list_blanks = WINDOW, set(), False, 0
        self.vocabulary: tuple = (-1, set())
        self.undercut: set = set()
        self.vouched: dict = {}

    def run(self) -> None:
        if not self.env.finished:
            self.prepare()
            self.settle(True)

    def settle(self, exploring: bool) -> None:
        for attempt in range(RESERVE_RECOVER + 1):
            if exploring:
                self.explore()
                if attempt == 0 and self.search_by_hand():
                    self.explore()
            said = self.close(attempt)
            if said is None:
                return
            if attempt == 0:
                self.budget.reserve = RESERVE_ORDER + RESERVE_MESSAGE * self.budget.report
                self.hear("\n".join(said))

    def explore(self) -> None:
        for _ in range(AGENDA_TURNS):
            self.news()
            if (ledgers := (self.request.version, len(self.catalog.rows), self.catalog.changes)) != self.seen_ledgers:
                self.seen_ledgers, self.stuck = ledgers, set()
            job = None if self.must_close() or self.clock.left() <= P3_LEFT else self.next_job()
            if job is None:
                break
            (kind, n, act), before = job, self.progress()
            self.phase = kind
            act()
            moved = self.progress() != before
            self.stuck |= set() if moved else {(kind, n)}
            self.record(job=kind, line=n, moved=moved)
        self.phase = "P3"
        self.fit_lines(sorted(self.request.lines))
        self.fit_sets()

    def record(self, **fields: Any) -> None:
        self.trace.append({"phase": self.phase, "step": self.env.step, **fields})

    def free_turn(self, goal: str, line: int | None = None, allow: tuple = ()) -> bool:
        tries = 1 + BLOCKS_MAX if allow or self.llm.spend < SPEND_SONNET else 0
        for attempt in range(tries):
            choice = self.seats.assistant(self.request.full, self.state(goal), self.last, goal, self.tools.specs)
            if choice is None:
                return False
            fitted, reason = self.gate(*choice, allow, last=attempt == tries - 1)
            if reason is None:
                self.send(choice[0], fitted, line)
                return True
            self.blocked.append(f"{choice[0]} {dump(choice[1])}: {reason}")
            self.record(blocked=choice[0], reason=reason)
        return False

    def gate(self, name: str, args: dict, allow: tuple, last: bool = False) -> tuple[dict, str | None]:
        if name not in self.tools.schema:
            return args, "the store has no tool of this name"
        if self.tools.role.get(name) in SYSTEM_ROLES and self.tools.role.get(name) not in allow:
            return args, "the system makes this call itself at the right time"
        named = bool(allow) and any(self.tools.role.get(t) in allow for t in self.tools.schema)
        lenient = last and "finish" in allow and self.tools.role.get(name) in (None, "other")
        if named and self.tools.role.get(name) not in allow and not lenient:
            # the map named a tool for this: another tool is not it (on the last try any tool outside the system's roles may be)
            return args, (f"only a call of these kinds is wanted now: {', '.join(allow)}; the tools for it: "
                          + ", ".join(t for t in self.tools.schema if self.tools.role.get(t) in allow))
        fitted, problems = fit_object(self.tools.schema[name], args)
        if problems:
            return fitted, "the arguments do not fit the tool's schema: " + "; ".join(problems)
        unseen = [v for v in self.tools.id_values(name, fitted) if v not in self.catalog.seen]
        if unseen:
            return fitted, f"{unseen[0]} is not an id any result has shown; use an id exactly as a result printed it"
        if self.done_now(name, fitted):
            return fitted, f"this exact call was made at step {self.made[dump([name, fitted, self.catalog.epoch])]}; take another route"
        if not allow and self.budget.room() - 1 <= self.budget.reserve:
            return fitted, "no step is free outside the reserve kept for the cart and the finish"
        return fitted, None

    def progress(self) -> tuple:
        refs = [(len((r.get("state") or {}).get("path", [])), len((r.get("state") or {}).get("ruled", {})), r.get("value")) for r in self.request.refs]
        return len(self.catalog.rows), len(self.catalog.lists), self.catalog.changes, self.request.version, self.budget.sent, len(self.verdicts), dump(refs), \
            len(self.suspect), self.cart.version

    def next_job(self) -> tuple | None:
        lines = sorted(self.request.lines, key=lambda n: (self.spent[n], n))
        for tier in (self.job_stale, self.job_lead_reference, self.job_first_search, self.job_first_read, self.job_question, self.job_grown,
                     self.job_judge, self.job_commit, self.job_reference, self.job_no_fit, self.job_cheaper,
                     self.job_undercut, self.job_deepen, self.job_pick):
            for n in lines:
                job = tier(n)
                if job and (job[0], n) not in self.stuck:
                    return job
        return None

    def job_stale(self, n: int) -> tuple | None:
        return ("stale", n, self.confirm) if self.suspect and n == min(self.request.lines) else None

    def movable(self, n: int) -> bool:
        evidence = (self.catalog.changes, self.request.version, self.catalog.epoch)
        return any(r["line"] == n and r.get("value") is None and r.get("stalls", 0) < CHASE_STALLS and ((r.get("state") or {}).get("paused") or r.get("seen") != evidence)
                   for r in self.request.refs)

    def job_lead_reference(self, n: int) -> tuple | None:
        return ("lead reference", n, lambda: self.chains([n], REF_TURN_STEPS)) if not self.request.search_values(n) and self.movable(n) else None

    def job_first_search(self, n: int) -> tuple | None:
        return ("first search", n, lambda: self.next_query(n)) if len(self.tried.get(n, ())) < ESSENTIAL_QUERIES and self.untried(n) else None

    def job_first_read(self, n: int) -> tuple | None:
        read = sum(c in self.catalog.records for c in self.cards.get(n, []))
        return ("first read", n, lambda: self.read_next(n)) if read < FIRST_CARDS.get(len(self.request.lines), 3) and self.unread(n, 1) else None

    def job_question(self, n: int) -> tuple | None:
        open_ = self.pending is None or self.pending or any(v == "unclear" and k not in self.retried for k, v in self.request.answers.items())
        menus = all(m in self.deep or any(c in self.catalog.records for c in self.cards.get(m, [])) for m in self.request.lines)
        return ("question", n, self.ask_next) if n == min(self.request.lines) and menus and self.questions() > 0 and self.tools.named("message") and open_ else None

    def job_grown(self, n: int) -> tuple | None:
        return ("disclosed search", n, lambda: self.next_query(n)) if n in self.grown and self.untried(n) else None

    def mark(self, n: int) -> tuple:
        return sum(self.catalog.changed_at.get(c, 0) for c in self.cards.get(n, [])), self.request.version

    def job_judge(self, n: int) -> tuple | None:
        dirty = [m for m in sorted(self.request.lines) if self.judged_at.get(m) != self.mark(m) and any(c in self.catalog.records for c in self.cards.get(m, []))]
        return ("judge", n, lambda: self.judge(dirty)) if n in dirty else None

    def job_commit(self, n: int) -> tuple | None:
        picks = self.picks()
        ready = not self.cart.version and n == min(self.request.lines) and len(picks) == len(self.request.lines)
        return ("commit", n, self.assemble) if ready and all(standing == "fits" for _, standing in picks.values()) else None

    def job_reference(self, n: int) -> tuple | None:
        return ("reference", n, lambda: self.chains([n], REF_TURN_STEPS)) if self.movable(n) else None

    def job_no_fit(self, n: int) -> tuple | None:
        judged = any((n, c) in self.verdicts for c in self.cards.get(n, []))
        return ("no fit", n, lambda: self.acquire(n)) if judged and not any(self.fits(c, n) == 0 for c in self.cards.get(n, [])) else None

    def job_cheaper(self, n: int) -> tuple | None:
        pick = self.pick(n) if self.request.goal(n) in GOALS_CHEAPEST else None
        rows = self.cheaper(n, pick) if pick and self.fits(pick, n) == 0 else []
        return ("cheaper", n, lambda: self.read_into(n, rows[:PURSUE_READS])) if rows else None

    def job_undercut(self, n: int) -> tuple | None:
        if self.request.goal(n) not in GOALS_CHEAPEST:
            return None
        pick = self.pick(n)
        price = self.catalog.price(pick) if pick and self.fits(pick, n) == 0 else None
        channel = next((c for c in self.tools.channels() if c["way"] == "words"), None)
        words = self.first_words(n, channel) if channel and price is not None and (n, price) not in self.undercut else []
        if not words:
            return None
        return ("undercut", n, lambda: (self.undercut.add((n, price)), self.search(channel, n, words, limit=price)))

    def first_words(self, n: int, channel: dict) -> list:
        """The words of the line's first search by words that showed rows."""
        for lst in sorted((x for x in self.catalog.lists.values() if x["line"] == n and x["channel"] == channel["label"] and x["ids"]), key=lambda x: x["step"]):
            if channel["query"] and (words := query_words(next(iter(at_path(lst["args"], channel["query"])), ""))):
                return words
        return []

    def job_deepen(self, n: int) -> tuple | None:
        return ("deepen", n, lambda: self.deepen(n)) if n not in self.deep else None

    def job_pick(self, n: int) -> tuple | None:
        ident = self.picks().get(n, (None,))[0]
        return ("pick read", n, lambda: (self.cards.setdefault(n, []).extend([] if ident in self.cards.get(n, []) else [ident]), self.complete(ident, n, True, self.room))) \
            if ident and self.catalog.next_reader(ident, self.readers(), True) else None

    def judge(self, lines: list) -> None:
        self.fit_lines(lines)
        self.pass_end = self.env.step + REF_TURN_STEPS
        self.deferrals(lines)
        self.pass_end = None
        self.judged_at.update({m: self.mark(m) for m in lines})

    def done_now(self, name: str, args: dict) -> bool:
        return dump([name, args, self.catalog.epoch]) in self.made

    def send(self, name: str, args: dict, line: int | None = None) -> Any:
        obs = self.env.send(name, args)
        self.made[dump([name, args, self.catalog.epoch])], self.used[self.phase] = self.env.step, self.used[self.phase] + 1
        self.spent[line] += line is not None
        self.absorb(name, args, obs, line)
        return obs

    def one_read(self, name: str, args: dict, payload: dict) -> bool:
        """Whether a read reply is the record of the one listing the call asked for: it names that id, or it names no id at
        all (no id field, or an empty one) and holds no rows of other listings."""
        ids = self.tools.id_values(name, args)
        if len(ids) != 1:
            return False
        if ids[0] in scalars(payload, set()):
            return True
        key = self.catalog.keys.get("id", "")
        known = self.catalog.read_keys.get(name, set()) | set(self.catalog.keys.values())
        return (not key or payload.get(key) in (None, "")) and not row_list(payload)[1] and bool(set(payload) & known)

    def absorb(self, name: str, args: dict, obs: Any, line: int | None) -> None:
        refused = isinstance(obs, dict) and "refusal" in obs
        error = obs.get("error") if isinstance(obs, dict) and obs.get("error") in RUNTIME_ERRORS else None
        whole = obs.get("observation") if refused else obs
        payload, said, wrapped = unwrap(whole)
        shape = self.tools.shape.get(name)
        arrived = []
        for key, group in said.items():
            fresh = [x for x in group if norm(x) not in self.noted] if wrapped or (shape is not None and key not in shape) else []
            self.notices += fresh
            self.noted.update(map(norm, fresh))
            arrived += fresh
        if isinstance(whole, dict) and not wrapped and shape is None and error is None:
            self.tools.shape[name] = set(whole)
        role, rows = self.tools.role.get(name), row_list(payload)[1]
        scalars(obs, self.catalog.seen)
        if error is None:
            if role == "cart_write":
                self.cart.version += 1
            channel = self.tools.channel_of(name, args) if role in ("search", "browse") else None
            if role in ("cart_view", "cart_write"):
                if moved := self.cart.show(payload, self.catalog, self.target):
                    self.event("price", moved)
                if role == "cart_write" and self.cart.version == 1:
                    self.catalog.epoch += 1
                    self.suspect |= set(self.target) | set(self.cart.counts())
            elif channel:
                self.catalog.take_search(channel, args, payload, line, self.env.step)
            elif rows:
                self.catalog.add_rows(name, rows, line, role in ("read", "compare"))
            elif role == "read" and isinstance(payload, dict) and self.one_read(name, args, payload):
                self.catalog.add_record(name, self.tools.id_values(name, args)[0], payload)
                if arrived and self.catalog.in_stock(self.tools.id_values(name, args)[0]) is True:
                    self.vouched[self.tools.id_values(name, args)[0]] = set(arrived)
            elif role in ("read", "compare"):
                self.misses[name] += 1
            self.budget.sent += role == "message"
        else:
            self.failed.add(dump([name, args]))
            self.misses[name] += role in ("read", "compare")
            if error == "budget_exceeded" and role == "message":
                self.budget.messages = self.budget.sent
        self.tools.observe(name, "error" if error else kind_of(payload), self.env.finished)
        self.last, self.last_obs = render(obs), obs
        self.record(tool=name, args=args, line=line, result="error" if error else kind_of(payload), refused=refused, error=error,
                    finished=self.env.finished)

    def state(self, goal: str) -> str:
        lists = sorted(self.catalog.lists.values(), key=lambda x: x["step"])[-QUERIES_SHOWN:]
        queries = [f"search at step {x['step']}: {x['tool']} {dump(x['args'])} -> total {x['count'] if x['count'] is not None else 'not shown'}, "
                   f"{len(x['ids'])} rows over {x['pages']} pages{', more pages left' if not x['done'] else ''}" for x in lists]
        opened = ", ".join(f"{i} ({', '.join(tools)})" for i, tools in self.catalog.records.items())
        return "\n".join([f"phase: {self.phase}", f"goal: {goal}", self.budget.text(), *self.request.lines_text(self.catalog), *queries,
                          f"opened: {opened or 'none'}", *[f"not sent by the system: {b}" for b in self.blocked[-BLOCKED_SHOWN:]],
                          *self.tools.roles_text(), *self.catalog.facts_text()])


def agent_main(problem_data: dict) -> list:
    assistant = ShoppingAssistant(problem_data)
    try:
        assistant.run()
    except Exception as error:
        assistant.record(fault=str(error) if isinstance(error, StoreGone) else repr(error))
        try:
            assistant.settle(False)
        except Exception as again:
            assistant.record(fault=repr(again))
    return assistant.trace
